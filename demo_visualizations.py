"""
Quick Demo Script for RUL Prediction System

This script demonstrates the complete pipeline with a small subset of data
or synthetic data for quick testing and validation.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')

def create_demo_visualizations():
    """Create example visualizations to demonstrate the system."""

    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Creating demo visualizations...")

    # 1. Training History
    epochs = np.arange(1, 101)
    train_loss = 100 * np.exp(-epochs / 20) + np.random.normal(0, 2, 100)
    val_loss = 105 * np.exp(-epochs / 20) + np.random.normal(0, 3, 100)
    train_rmse = 25 * np.exp(-epochs / 25) + 12 + np.random.normal(0, 0.5, 100)
    val_rmse = 27 * np.exp(-epochs / 25) + 13 + np.random.normal(0, 0.8, 100)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_rmse, label='Train RMSE', linewidth=2)
    axes[1].plot(epochs, val_rmse, label='Val RMSE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Training History - RMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'demo_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Training history plot created")

    # 2. RUL Predictions
    np.random.seed(42)
    n_samples = 100
    true_rul = np.random.uniform(10, 125, n_samples)
    predicted_rul = true_rul + np.random.normal(0, 8, n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    axes[0].scatter(true_rul, predicted_rul, alpha=0.6, s=50)
    axes[0].plot([0, 125], [0, 125], 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title('RUL Predictions - Scatter Plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = predicted_rul - true_rul
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'demo_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Predictions plot created")

    # 3. Attention Heatmap
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(10):
        attention = np.random.dirichlet(np.ones(30) * 5)  # More concentrated attention
        attention = attention.reshape(1, -1)

        im = axes[i].imshow(attention, cmap='YlOrRd', aspect='auto')
        axes[i].set_xlabel('Time Step')
        axes[i].set_title(f'Engine {i+1}')
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], orientation='horizontal', pad=0.1)

    plt.suptitle('Attention Weights Heatmap (Demo)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / 'demo_attention_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Attention heatmap created")

    # 4. Feature Importance
    feature_names = ['setting_1', 'setting_2', 'setting_3',
                     'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                     'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
                     'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
                     'sensor_20', 'sensor_21']

    importance = np.random.exponential(0.1, len(feature_names))
    importance = importance / importance.sum()
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance[sorted_idx], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance (Attention-Based)')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'demo_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Feature importance plot created")

    # 5. Model Comparison
    models = ['Transformer', 'CNN-Transformer', 'BiLSTM', 'TCN']
    rmse = [12.8, 13.4, 14.2, 15.1]
    mae = [10.2, 10.8, 11.5, 12.3]
    scores = [268, 285, 310, 335]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    axes[0].bar(models, rmse, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(rmse):
        axes[0].text(i, v, f'{v:.1f}', ha='center', va='bottom')

    axes[1].bar(models, mae, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison')
    axes[1].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(mae):
        axes[1].text(i, v, f'{v:.1f}', ha='center', va='bottom')

    axes[2].bar(models, scores, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('NASA Score')
    axes[2].set_title('NASA Score Comparison')
    axes[2].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(scores):
        axes[2].text(i, v, f'{v}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(results_dir / 'demo_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Model comparison plot created")

    print("\n✓ All demo visualizations created successfully!")
    print(f"  Saved to: {results_dir.absolute()}")


def create_demo_metrics():
    """Create example metrics JSON file."""
    import json

    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "RMSE": 12.84,
        "MAE": 10.23,
        "MAPE": 18.45,
        "R2": 0.896,
        "NASA_Score": 267.5,
        "mean_error": -0.34,
        "std_error": 12.91,
        "late_predictions_pct": 48.2,
        "early_predictions_pct": 51.8
    }

    with open(results_dir / 'demo_transformer_FD001_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("✓ Demo metrics file created")


if __name__ == "__main__":
    print("=" * 80)
    print("RUL Prediction System - Demo Visualization Generator")
    print("=" * 80)
    print()

    create_demo_visualizations()
    print()
    create_demo_metrics()

    print()
    print("=" * 80)
    print("Demo complete! Check the 'results' directory for outputs.")
    print("=" * 80)
