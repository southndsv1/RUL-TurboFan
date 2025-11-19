"""
Visualization Tools for RUL Prediction

Features:
- Attention weight visualization
- Feature importance analysis (SHAP)
- Prediction trajectory plots
- Error analysis
- Sensor correlation heatmaps
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import argparse

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class RULVisualizer:
    """Visualization tools for RUL prediction."""

    def __init__(self, model, device, feature_names: List[str]):
        """
        Args:
            model: Trained model
            device: Device for computation
            feature_names: List of feature names
        """
        self.model = model.to(device)
        self.device = device
        self.feature_names = feature_names
        self.model.eval()

    @torch.no_grad()
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights from model.

        Args:
            X: Input data of shape (num_samples, seq_len, num_features)

        Returns:
            Attention weights of shape (num_samples, seq_len)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Check if model has attention
        if hasattr(self.model, 'forward'):
            output = self.model(X_tensor, return_attention=True)
            if isinstance(output, tuple) and len(output) >= 2:
                attention = output[1] if output[1] is not None else output[2]
                if attention is not None:
                    return attention.cpu().numpy()

        print("Model does not support attention visualization")
        return None

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        num_samples: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Plot attention weight heatmaps.

        Args:
            attention_weights: Attention weights of shape (num_samples, seq_len)
            num_samples: Number of samples to plot
            save_path: Path to save plot
        """
        num_samples = min(num_samples, len(attention_weights))
        indices = np.random.choice(len(attention_weights), size=num_samples, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break

            ax = axes[i]
            attn = attention_weights[idx]

            # Plot heatmap
            im = ax.imshow(attn.reshape(1, -1), cmap='YlOrRd', aspect='auto')
            ax.set_xlabel('Time Step')
            ax.set_title(f'Sample {idx}')
            ax.set_yticks([])

            # Add colorbar
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to: {save_path}")

        plt.show()

    def plot_attention_over_time(
        self,
        X: np.ndarray,
        attention_weights: np.ndarray,
        sample_idx: int = 0,
        feature_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Plot attention weights alongside sensor readings over time.

        Args:
            X: Input data
            attention_weights: Attention weights
            sample_idx: Index of sample to plot
            feature_idx: Index of feature to plot
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot sensor readings
        time_steps = np.arange(X.shape[1])
        sensor_values = X[sample_idx, :, feature_idx]

        ax1.plot(time_steps, sensor_values, linewidth=2)
        ax1.set_ylabel(self.feature_names[feature_idx])
        ax1.set_title('Sensor Reading Over Time')
        ax1.grid(True)

        # Plot attention weights
        attn = attention_weights[sample_idx]
        ax2.bar(time_steps, attn, alpha=0.7, color='orange')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Attention Weights Over Time')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention over time plot saved to: {save_path}")

        plt.show()

    def plot_feature_attention(
        self,
        X: np.ndarray,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Aggregate attention weights by feature.

        Args:
            X: Input data
            attention_weights: Attention weights
            save_path: Path to save plot
        """
        # Compute weighted average of features by attention
        # attention_weights: (num_samples, seq_len)
        # X: (num_samples, seq_len, num_features)

        # Expand attention weights to match feature dimension
        attn_expanded = attention_weights[:, :, np.newaxis]  # (num_samples, seq_len, 1)

        # Weight features by attention
        weighted_features = X * attn_expanded  # (num_samples, seq_len, num_features)

        # Average over samples and time
        feature_importance = np.abs(weighted_features).mean(axis=(0, 1))  # (num_features,)

        # Normalize
        feature_importance = feature_importance / feature_importance.sum()

        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(self.feature_names))
        ax.barh(y_pos, feature_importance[sorted_indices], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in sorted_indices])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (Attention-Based)')
        ax.grid(True, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")

        plt.show()

        return feature_importance


def plot_rul_trajectories(
    predictions_df: pd.DataFrame,
    num_samples: int = 12,
    save_path: Optional[str] = None,
):
    """
    Plot RUL prediction trajectories for multiple engines.

    Args:
        predictions_df: DataFrame with columns: unit_id, true_RUL, predicted_RUL
        num_samples: Number of engines to plot
        save_path: Path to save plot
    """
    unique_units = predictions_df['unit_id'].unique()
    selected_units = np.random.choice(unique_units, size=min(num_samples, len(unique_units)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, unit_id in enumerate(selected_units):
        if i >= len(axes):
            break

        ax = axes[i]
        unit_data = predictions_df[predictions_df['unit_id'] == unit_id]

        true_rul = unit_data['true_RUL'].values[0]
        pred_rul = unit_data['predicted_RUL'].values[0]
        error = pred_rul - true_rul

        # Plot
        ax.barh(['True', 'Predicted'], [true_rul, pred_rul], color=['blue', 'orange'], alpha=0.7)
        ax.set_xlabel('RUL (cycles)')
        ax.set_title(f'Unit {unit_id} (Error: {error:.1f})')
        ax.grid(True, axis='x')

    # Remove empty subplots
    for i in range(len(selected_units), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RUL trajectories saved to: {save_path}")

    plt.show()


def plot_error_distribution(
    predictions_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot error distribution analysis.

    Args:
        predictions_df: DataFrame with predictions
        save_path: Path to save plot
    """
    errors = predictions_df['error'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=errors.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    ax.set_xlabel('Prediction Error (Predicted - True)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot
    ax = axes[0, 1]
    ax.boxplot([errors], vert=True, labels=['Errors'])
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error Box Plot')
    ax.grid(True, alpha=0.3)

    # 3. Q-Q plot
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')
    ax.grid(True, alpha=0.3)

    # 4. Error vs True RUL
    ax = axes[1, 1]
    ax.scatter(predictions_df['true_RUL'], errors, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Error vs True RUL')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")

    plt.show()


def plot_sensor_correlations(
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Plot correlation heatmap between sensors.

    Args:
        X: Input data of shape (num_samples, seq_len, num_features)
        feature_names: List of feature names
        save_path: Path to save plot
    """
    # Flatten time dimension and compute correlations
    # Reshape to (num_samples * seq_len, num_features)
    X_flat = X.reshape(-1, X.shape[2])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_flat.T)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax,
    )

    ax.set_title('Sensor Correlation Heatmap')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")

    plt.show()


def plot_model_comparison(
    results_dir: str,
    dataset: str,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of different models.

    Args:
        results_dir: Directory containing results
        dataset: Dataset name
        save_path: Path to save plot
    """
    results_dir = Path(results_dir)

    # Load metrics from different models
    model_types = ['transformer', 'lstm_attention', 'tcn', 'hybrid']
    results = []

    for model_type in model_types:
        metrics_file = results_dir / f"{model_type}_{dataset}_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'NASA_Score': metrics['NASA_Score'],
                })

    if not results:
        print("No results found for comparison")
        return

    df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ['RMSE', 'MAE', 'NASA_Score']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        ax = axes[i]
        ax.bar(df['Model'], df[metric], color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Add values on top of bars
        for j, v in enumerate(df[metric]):
            ax.text(j, v, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")

    plt.show()


def main(args):
    """Main visualization function."""
    from data_loader import CMAPSSDataLoader
    from train import create_model
    from utils import get_device

    # Load data
    loader = CMAPSSDataLoader(
        data_dir=args.data_path,
        dataset_name=args.dataset,
    )
    data = loader.load_and_preprocess(
        sequence_length=args.sequence_length,
        max_rul=args.max_rul,
        val_split=args.val_split,
    )

    # Load model
    device = get_device()
    model_config = {
        'model_size': args.model_size,
        'dropout': 0.1,
    }

    model_path = Path(args.checkpoint_dir) / f"{args.model_type}_{args.dataset}_best.pth"
    if model_path.exists():
        model = create_model(args.model_type, data['num_features'], model_config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Create visualizer
        visualizer = RULVisualizer(model, device, data['feature_names'])

        # Extract attention weights
        print("Extracting attention weights...")
        attention_weights = visualizer.get_attention_weights(data['X_test'][:100])

        if attention_weights is not None:
            # Plot attention heatmap
            save_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_attention_heatmap.png"
            visualizer.plot_attention_heatmap(attention_weights, save_path=str(save_path))

            # Plot feature importance
            save_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_feature_importance.png"
            visualizer.plot_feature_attention(data['X_test'][:100], attention_weights, save_path=str(save_path))

    # Load predictions
    predictions_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_predictions.csv"
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)

        # Plot trajectories
        save_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_trajectories.png"
        plot_rul_trajectories(predictions_df, save_path=str(save_path))

        # Plot error distribution
        save_path = Path(args.results_dir) / f"{args.model_type}_{args.dataset}_error_dist.png"
        plot_error_distribution(predictions_df, save_path=str(save_path))

    # Plot sensor correlations
    save_path = Path(args.results_dir) / f"{args.dataset}_sensor_correlations.png"
    plot_sensor_correlations(data['X_test'], data['feature_names'], save_path=str(save_path))

    # Plot model comparison
    save_path = Path(args.results_dir) / f"{args.dataset}_model_comparison.png"
    plot_model_comparison(args.results_dir, args.dataset, save_path=str(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize RUL Prediction Results')

    parser.add_argument('--data_path', type=str, default='../data/CMAPSS')
    parser.add_argument('--dataset', type=str, default='FD001')
    parser.add_argument('--model_type', type=str, default='transformer')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--max_rul', type=int, default=125)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--checkpoint_dir', type=str, default='../models')
    parser.add_argument('--results_dir', type=str, default='../results')

    args = parser.parse_args()
    main(args)
