"""
Utility functions for RUL prediction.

Includes:
- Metrics calculation
- Loss functions
- Training helpers
- Visualization helpers
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate NASA scoring function (asymmetric penalty).

    Penalizes late predictions more heavily than early predictions.
    This is the official scoring function for the PHM08 challenge.

    Score = sum(exp(error/a1) - 1) for error < 0 (late predictions)
            sum(exp(error/a2) - 1) for error >= 0 (early predictions)

    where a1=13, a2=10 (standard values)
    """
    errors = y_pred - y_true
    a1, a2 = 13.0, 10.0

    score = 0.0
    for error in errors:
        if error < 0:  # Late prediction (predicted less than actual)
            score += np.exp(-error / a1) - 1
        else:  # Early prediction (predicted more than actual)
            score += np.exp(error / a2) - 1

    return score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all evaluation metrics."""
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'NASA_Score': nasa_score(y_true, y_pred),
    }


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss function for RUL prediction.

    Penalizes late predictions more heavily than early predictions,
    as late predictions are more dangerous in predictive maintenance.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for late predictions (0.5 means symmetric)
                   Higher alpha (>0.5) penalizes late predictions more
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate asymmetric loss.

        Args:
            y_pred: Predicted RUL values
            y_true: True RUL values

        Returns:
            Loss value
        """
        errors = y_pred - y_true

        # Different weights for positive and negative errors
        weights = torch.where(errors < 0,
                              torch.ones_like(errors) * self.alpha,
                              torch.ones_like(errors) * (1 - self.alpha))

        # Weighted MSE
        loss = torch.mean(weights * errors ** 2)

        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check for improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training."""

    def __init__(self, filepath: str, mode: str = 'min', verbose: bool = True):
        """
        Args:
            filepath: Path to save model
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.filepath = filepath
        self.mode = mode
        self.verbose = verbose
        self.best_score = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Save model if score improved.

        Args:
            score: Current metric value
            model: PyTorch model

        Returns:
            True if model was saved
        """
        if self.best_score is None:
            self.best_score = score
            self._save_model(model)
            return True

        # Check for improvement
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score

        if improved:
            if self.verbose:
                print(f"Score improved: {self.best_score:.4f} -> {score:.4f}")
            self.best_score = score
            self._save_model(model)
            return True

        return False

    def _save_model(self, model: nn.Module):
        """Save model state dict."""
        torch.save(model.state_dict(), self.filepath)
        if self.verbose:
            print(f"Model saved to {self.filepath}")


def get_device() -> torch.device:
    """Get available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Metric (RMSE)
    if 'train_rmse' in history:
        axes[1].plot(history['train_rmse'], label='Train RMSE')
    if 'val_rmse' in history:
        axes[1].plot(history['val_rmse'], label='Val RMSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Training History - RMSE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: Optional[np.ndarray] = None,
                     title: str = "RUL Predictions",
                     save_path: Optional[str] = None):
    """
    Plot predicted vs actual RUL.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        y_std: Standard deviation (for uncertainty)
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title(f'{title} - Scatter Plot')
    axes[0].legend()
    axes[0].grid(True)

    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{title} - Error Distribution')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_rul_trajectory(y_true: np.ndarray, y_pred: np.ndarray,
                        y_std: Optional[np.ndarray] = None,
                        num_samples: int = 10,
                        save_path: Optional[str] = None):
    """
    Plot RUL prediction trajectories for sample engines.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        y_std: Standard deviation (for uncertainty bands)
        num_samples: Number of sample trajectories to plot
        save_path: Path to save plot
    """
    indices = np.random.choice(len(y_true), size=min(num_samples, len(y_true)), replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Plot true and predicted RUL
        ax.plot([0, 1], [y_true[idx], 0], 'b-', linewidth=2, label='True')
        ax.plot([0, 1], [y_pred[idx], 0], 'r--', linewidth=2, label='Predicted')

        # Add uncertainty band if available
        if y_std is not None:
            upper = y_pred[idx] + 2 * y_std[idx]
            lower = y_pred[idx] - 2 * y_std[idx]
            ax.fill_between([0, 1], [upper, 0], [lower, 0], alpha=0.3, color='red')

        ax.set_xlabel('Time Progress')
        ax.set_ylabel('RUL')
        ax.set_title(f'Engine {idx}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


class CosineAnnealingWarmup:
    """Cosine annealing learning rate scheduler with warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int):
        """Update learning rate."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
