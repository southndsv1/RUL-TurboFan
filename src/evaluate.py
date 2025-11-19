"""
Evaluation Script for RUL Prediction Models

Features:
- Load trained models
- Evaluate on test data
- Calculate comprehensive metrics
- Generate predictions
- Compare with baselines
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict

from models.transformer import create_transformer_model
from models.lstm import create_lstm_model
from models.tcn import create_tcn_model
from models.hybrid import create_hybrid_model
from utils import calculate_metrics, get_device, plot_predictions
from train import create_model


class RULEvaluator:
    """Evaluator for RUL prediction models."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained PyTorch model
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data of shape (num_samples, seq_len, num_features)

        Returns:
            Predictions of shape (num_samples,)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        predictions = []

        for (X_batch,) in loader:
            X_batch = X_batch.to(self.device)

            # Forward pass
            output = self.model(X_batch)

            # Handle different output types
            if isinstance(output, tuple):
                y_pred = output[0]
            else:
                y_pred = output

            predictions.append(y_pred.cpu().numpy())

        predictions = np.concatenate(predictions)
        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X: Input data
            y_true: True RUL values

        Returns:
            Dictionary of metrics
        """
        print("\n" + "=" * 80)
        print("Evaluating Model")
        print("=" * 80)

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        # Print results
        print("\nTest Set Results:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  NASA Score: {metrics['NASA_Score']:.4f}")

        # Additional analysis
        errors = y_pred - y_true
        late_predictions = np.sum(errors < 0)
        early_predictions = np.sum(errors > 0)

        print("\nError Analysis:")
        print(f"  Mean Error: {errors.mean():.4f}")
        print(f"  Std Error: {errors.std():.4f}")
        print(f"  Late Predictions: {late_predictions} ({late_predictions / len(errors) * 100:.1f}%)")
        print(f"  Early Predictions: {early_predictions} ({early_predictions / len(errors) * 100:.1f}%)")

        # Percentiles
        print("\nError Percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th percentile: {np.percentile(errors, p):.4f}")

        print("=" * 80 + "\n")

        metrics['mean_error'] = float(errors.mean())
        metrics['std_error'] = float(errors.std())
        metrics['late_predictions_pct'] = float(late_predictions / len(errors) * 100)
        metrics['early_predictions_pct'] = float(early_predictions / len(errors) * 100)

        return metrics, y_pred


def load_model(model_path: str, model_type: str, num_features: int, config: dict) -> nn.Module:
    """
    Load trained model.

    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        num_features: Number of input features
        config: Model configuration

    Returns:
        Loaded model
    """
    # Create model
    model = create_model(model_type, num_features, config)

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    print(f"Model loaded from: {model_path}")

    return model


def compare_with_baselines(results: Dict, dataset: str) -> pd.DataFrame:
    """
    Compare results with published baselines.

    Args:
        results: Current model results
        dataset: Dataset name

    Returns:
        Comparison dataframe
    """
    # Literature baselines (approximate values)
    baselines = {
        'FD001': {
            'SVR': {'RMSE': 20.96, 'Score': 1380},
            'Random Forest': {'RMSE': 18.45, 'Score': 1070},
            'MLP': {'RMSE': 16.14, 'Score': 338},
            'CNN': {'RMSE': 18.45, 'Score': 1280},
            'LSTM': {'RMSE': 16.14, 'Score': 338},
            'BiLSTM': {'RMSE': 13.65, 'Score': 295},
            'GRU': {'RMSE': 15.04, 'Score': 306},
            'Transformer (Zhang et al.)': {'RMSE': 12.56, 'Score': 267},
            'CNN-LSTM': {'RMSE': 12.61, 'Score': 274},
        },
        'FD002': {
            'SVR': {'RMSE': 42.0, 'Score': 13600},
            'LSTM': {'RMSE': 21.85, 'Score': 4550},
            'BiLSTM': {'RMSE': 20.47, 'Score': 4300},
            'Transformer': {'RMSE': 19.82, 'Score': 3890},
        },
        'FD003': {
            'SVR': {'RMSE': 21.05, 'Score': 1600},
            'LSTM': {'RMSE': 16.18, 'Score': 852},
            'BiLSTM': {'RMSE': 14.74, 'Score': 730},
            'Transformer': {'RMSE': 13.89, 'Score': 685},
        },
        'FD004': {
            'SVR': {'RMSE': 45.35, 'Score': 19650},
            'LSTM': {'RMSE': 24.52, 'Score': 6780},
            'BiLSTM': {'RMSE': 23.31, 'Score': 5960},
            'Transformer': {'RMSE': 22.08, 'Score': 5340},
        },
    }

    if dataset not in baselines:
        print(f"No baseline data for {dataset}")
        return None

    # Create comparison dataframe
    data = []

    for method, metrics in baselines[dataset].items():
        data.append({
            'Method': method,
            'RMSE': metrics.get('RMSE', '-'),
            'NASA Score': metrics.get('Score', '-'),
        })

    # Add current model
    data.append({
        'Method': 'Our Model',
        'RMSE': results['RMSE'],
        'NASA Score': results['NASA_Score'],
    })

    df = pd.DataFrame(data)

    # Sort by RMSE
    df = df.sort_values('RMSE')

    print("\n" + "=" * 80)
    print(f"Comparison with Literature Baselines ({dataset})")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    return df


def main(args):
    """Main evaluation function."""
    # Get device
    device = get_device()

    # Load preprocessed data
    print("Loading data...")
    from data_loader import CMAPSSDataLoader

    loader = CMAPSSDataLoader(
        data_dir=args.data_path,
        dataset_name=args.dataset,
    )

    data = loader.load_and_preprocess(
        sequence_length=args.sequence_length,
        max_rul=args.max_rul,
        val_split=args.val_split,
    )

    # Model configuration
    model_config = {
        'model_size': args.model_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'uncertainty': args.uncertainty,
    }

    # Load model
    model_path = Path(args.checkpoint_dir) / f"{args.model_type}_{args.dataset}_best.pth"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    model = load_model(
        str(model_path),
        args.model_type,
        data['num_features'],
        model_config,
    )

    # Create evaluator
    evaluator = RULEvaluator(model, device)

    # Evaluate on test set
    metrics, y_pred = evaluator.evaluate(data['X_test'], data['y_test'])

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = results_dir / f"{args.model_type}_{args.dataset}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'unit_id': data['test_unit_ids'],
        'true_RUL': data['y_test'],
        'predicted_RUL': y_pred,
        'error': y_pred - data['y_test'],
    })
    predictions_path = results_dir / f"{args.model_type}_{args.dataset}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    # Plot predictions
    plot_path = results_dir / f"{args.model_type}_{args.dataset}_predictions.png"
    plot_predictions(
        data['y_test'],
        y_pred,
        title=f"{args.model_type} - {args.dataset}",
        save_path=str(plot_path),
    )

    # Compare with baselines
    if args.compare_baselines:
        comparison_df = compare_with_baselines(metrics, args.dataset)
        if comparison_df is not None:
            comparison_path = results_dir / f"{args.dataset}_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RUL Prediction Models')

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

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='../models',
                        help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Results directory')

    # Options
    parser.add_argument('--compare_baselines', action='store_true', default=True,
                        help='Compare with baseline methods')

    args = parser.parse_args()

    main(args)
