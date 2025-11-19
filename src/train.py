"""
Training Script for RUL Prediction Models

Features:
- Flexible model selection
- Multiple loss functions (MSE, MAE, Asymmetric)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard logging
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import models
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model
from models.tcn import create_tcn_model
from models.hybrid import create_hybrid_model

# Import utilities
from utils import (
    set_seed, get_device, count_parameters,
    AsymmetricLoss, EarlyStopping, ModelCheckpoint,
    CosineAnnealingWarmup, calculate_metrics,
    plot_training_history
)


class RULTrainer:
    """Trainer class for RUL prediction models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict,
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # Learning rate scheduler
        if config.get('use_scheduler', True):
            self.scheduler = CosineAnnealingWarmup(
                optimizer,
                warmup_epochs=config.get('warmup_epochs', 5),
                total_epochs=config['epochs'],
                min_lr=config.get('min_lr', 1e-6),
            )
        else:
            self.scheduler = None

        # Early stopping
        if config.get('early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=config.get('patience', 15),
                min_delta=config.get('min_delta', 1e-4),
                mode='min',
            )
        else:
            self.early_stopping = None

        # Model checkpoint
        checkpoint_dir = Path(config.get('checkpoint_dir', '../models'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{config['model_name']}_best.pth"

        self.checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            mode='min',
            verbose=True,
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': [],
        }

        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        predictions = []
        targets = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Handle different model outputs
                    output = self.model(X_batch)
                    if isinstance(output, tuple):
                        y_pred = output[0]
                    else:
                        y_pred = output

                    loss = self.criterion(y_pred, y_batch)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(X_batch)
                if isinstance(output, tuple):
                    y_pred = output[0]
                else:
                    y_pred = output

                loss = self.criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions.append(y_pred.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        metrics = calculate_metrics(targets, predictions)

        return avg_loss, metrics['RMSE']

    @torch.no_grad()
    def validate_epoch(self) -> tuple:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        predictions = []
        targets = []

        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            output = self.model(X_batch)
            if isinstance(output, tuple):
                y_pred = output[0]
            else:
                y_pred = output

            loss = self.criterion(y_pred, y_batch)

            # Track metrics
            total_loss += loss.item()
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        metrics = calculate_metrics(targets, predictions)

        return avg_loss, metrics['RMSE']

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("=" * 80 + "\n")

        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_rmse = self.train_epoch()

            # Validate
            val_loss, val_rmse = self.validate_epoch()

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                current_lr = self.scheduler.step(epoch)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint(val_loss, self.model)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("=" * 80 + "\n")

        return self.history


def create_model(model_type: str, num_features: int, config: dict) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model
        num_features: Number of input features
        config: Model configuration

    Returns:
        PyTorch model
    """
    if model_type == 'transformer':
        model = create_transformer_model(
            num_features=num_features,
            model_size=config.get('model_size', 'base'),
            uncertainty=config.get('uncertainty', False),
            dropout=config.get('dropout', 0.1),
        )
    elif model_type.startswith('lstm'):
        model = create_lstm_model(
            num_features=num_features,
            model_type=model_type,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            bidirectional=config.get('bidirectional', True),
        )
    elif model_type == 'tcn':
        model = create_tcn_model(
            num_features=num_features,
            model_size=config.get('model_size', 'base'),
            use_attention=config.get('use_attention', True),
            dropout=config.get('dropout', 0.2),
        )
    elif model_type == 'hybrid':
        model = create_hybrid_model(
            num_features=num_features,
            model_type=config.get('hybrid_type', 'base'),
            dropout=config.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Load preprocessed data
    print("Loading data...")
    data_path = Path(args.data_path)

    # Load data (assume it's already preprocessed and saved)
    # For now, we'll load from the data_loader
    from data_loader import CMAPSSDataLoader

    loader = CMAPSSDataLoader(
        data_dir=str(data_path),
        dataset_name=args.dataset,
    )

    data = loader.load_and_preprocess(
        sequence_length=args.sequence_length,
        max_rul=args.max_rul,
        val_split=args.val_split,
        stride=args.stride,
    )

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.FloatTensor(data['y_train']).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_val']).unsqueeze(1),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Features: {data['num_features']}")

    # Create model
    model_config = {
        'model_size': args.model_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'uncertainty': args.uncertainty,
    }

    model = create_model(args.model_type, data['num_features'], model_config)

    # Loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'asymmetric':
        criterion = AsymmetricLoss(alpha=args.alpha)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Training configuration
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_name': f"{args.model_type}_{args.dataset}",
        'checkpoint_dir': args.checkpoint_dir,
        'use_scheduler': args.use_scheduler,
        'warmup_epochs': args.warmup_epochs,
        'min_lr': args.min_lr,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'max_grad_norm': args.max_grad_norm,
        'use_amp': args.use_amp,
    }

    # Create trainer
    trainer = RULTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=train_config,
    )

    # Train
    history = trainer.train()

    # Save training history
    history_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")

    # Plot training history
    plot_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_training.png"
    plot_training_history(history, save_path=str(plot_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RUL Prediction Models')

    # Data
    parser.add_argument('--data_path', type=str, default='../data/CMAPSS',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='Dataset to use')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length for windows')
    parser.add_argument('--max_rul', type=int, default=125,
                        help='Maximum RUL for clipping')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')

    # Model
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'lstm_attention', 'lstm_simple',
                                 'gru_attention', 'tcn', 'hybrid'],
                        help='Model type')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for RNN models')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Use uncertainty quantification')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--loss', type=str, default='asymmetric',
                        choices=['mse', 'mae', 'asymmetric'],
                        help='Loss function')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha for asymmetric loss')

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='Use learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Other
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='../models',
                        help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Results directory')

    args = parser.parse_args()

    main(args)
