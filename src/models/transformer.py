"""
Transformer Model for RUL Prediction

Implements a Transformer-based architecture with:
- Positional encoding for temporal information
- Multi-head self-attention
- Feed-forward networks
- Layer normalization and dropout
- Optional multi-task learning (RUL + health state classification)
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds position information to the input embeddings using sine and cosine functions.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRUL(nn.Module):
    """
    Transformer model for RUL prediction.

    Architecture:
    1. Input projection: Map input features to d_model dimensions
    2. Positional encoding: Add temporal information
    3. Transformer encoder layers: Self-attention and feed-forward
    4. Aggregation: Pool over time dimension
    5. Output head: Predict RUL (and optionally health state)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        num_health_states: Optional[int] = None,
    ):
        """
        Args:
            num_features: Number of input features
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_health_states: Number of health states for classification (None to disable)
        """
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.num_health_states = num_health_states

        # Input projection: Map input features to d_model dimensions
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Aggregation: Multiple options
        self.aggregation = 'attention'  # 'mean', 'last', 'attention'

        if self.aggregation == 'attention':
            # Learnable attention pooling
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )

        # RUL prediction head
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Health state classification head (optional)
        if num_health_states is not None:
            self.health_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_health_states),
            )
        else:
            self.health_head = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_attention: If True, return attention weights

        Returns:
            rul_pred: RUL predictions of shape (batch_size, 1)
            health_pred: Health state predictions (if enabled) of shape (batch_size, num_health_states)
            attention_weights: Attention weights (if return_attention=True)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Aggregation over time
        if self.aggregation == 'mean':
            # Global average pooling
            x_pooled = x.mean(dim=1)  # (batch, d_model)
            attention_weights = None

        elif self.aggregation == 'last':
            # Use last time step
            x_pooled = x[:, -1, :]  # (batch, d_model)
            attention_weights = None

        elif self.aggregation == 'attention':
            # Attention pooling
            attn_scores = self.attention_pool(x)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
            x_pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)

            if return_attention:
                attention_weights = attn_weights.squeeze(-1)  # (batch, seq_len)
            else:
                attention_weights = None

        # RUL prediction
        rul_pred = self.rul_head(x_pooled)  # (batch, 1)

        # Health state classification (if enabled)
        if self.health_head is not None:
            health_pred = self.health_head(x_pooled)  # (batch, num_health_states)
        else:
            health_pred = None

        return rul_pred, health_pred, attention_weights

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Attention weights of shape (batch_size, seq_len)
        """
        with torch.no_grad():
            _, _, attention_weights = self.forward(x, return_attention=True)

        if attention_weights is not None:
            return attention_weights.cpu().numpy()
        else:
            return None


class TransformerRULWithUncertainty(nn.Module):
    """
    Transformer model with uncertainty quantification using MC Dropout.

    The model predicts both mean and variance of RUL.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        Args:
            num_features: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Variance prediction head (log variance for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            mean: Mean RUL predictions of shape (batch_size, 1)
            logvar: Log variance of shape (batch_size, 1)
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Attention pooling
        attn_scores = self.attention_pool(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)

        # Predict mean and log variance
        mean = self.mean_head(x_pooled)
        logvar = self.logvar_head(x_pooled)

        return mean, logvar

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using Monte Carlo Dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean: Mean predictions
            std: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                mean, _ = self.forward(x)
                predictions.append(mean.cpu().numpy())

        predictions = np.array(predictions)  # (n_samples, batch_size, 1)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return mean, std


def create_transformer_model(
    num_features: int,
    model_size: str = 'base',
    uncertainty: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Transformer models.

    Args:
        num_features: Number of input features
        model_size: 'small', 'base', 'large'
        uncertainty: Use uncertainty quantification
        **kwargs: Additional arguments

    Returns:
        Transformer model
    """
    # Model configurations
    configs = {
        'small': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
        },
        'base': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
        },
        'large': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
        },
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)

    if uncertainty:
        model = TransformerRULWithUncertainty(num_features=num_features, **config)
    else:
        model = TransformerRUL(num_features=num_features, **config)

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_len = 30
    num_features = 17

    # Create model
    model = create_transformer_model(num_features, model_size='base')

    # Random input
    x = torch.randn(batch_size, seq_len, num_features)

    # Forward pass
    rul_pred, health_pred, attn_weights = model(x, return_attention=True)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"RUL prediction shape: {rul_pred.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
