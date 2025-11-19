"""
Hybrid CNN+Transformer Model for RUL Prediction

Combines:
- CNN for local feature extraction from sensor readings
- Transformer for capturing long-range temporal dependencies
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class CNNFeatureExtractor(nn.Module):
    """
    CNN module for local feature extraction.

    Uses 1D convolutions to extract local patterns from sensor readings.
    """

    def __init__(
        self,
        num_features: int,
        num_filters: int = 64,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1,
    ):
        """
        Args:
            num_features: Number of input features
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes for multi-scale feature extraction
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=num_features,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        # Output dimension
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Extracted features of shape (batch, seq_len, output_dim)
        """
        # Transpose for Conv1d: (batch, num_features, seq_len)
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)  # (batch, num_filters, seq_len)
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes), seq_len)

        # Transpose back: (batch, seq_len, output_dim)
        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model for RUL prediction.

    Architecture:
    1. CNN: Extract local features from sensor readings
    2. Projection: Map CNN features to Transformer dimension
    3. Positional Encoding: Add temporal information
    4. Transformer: Capture long-range dependencies
    5. Attention Pooling: Aggregate temporal information
    6. FC Layers: RUL prediction
    """

    def __init__(
        self,
        num_features: int,
        cnn_filters: int = 64,
        cnn_kernels: list = [3, 5, 7],
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        Args:
            num_features: Number of input features
            cnn_filters: Number of CNN filters per kernel
            cnn_kernels: List of kernel sizes for CNN
            d_model: Transformer dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            num_features=num_features,
            num_filters=cnn_filters,
            kernel_sizes=cnn_kernels,
            dropout=dropout,
        )

        # 2. Project CNN features to Transformer dimension
        cnn_output_dim = cnn_filters * len(cnn_kernels)
        self.projection = nn.Linear(cnn_output_dim, d_model)

        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 5. Attention Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # 6. RUL Prediction Head
        self.rul_head = nn.Sequential(
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

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_attention: If True, return attention weights

        Returns:
            rul_pred: RUL predictions of shape (batch_size, 1)
            attention_weights: Attention weights (if return_attention=True)
        """
        # 1. CNN Feature Extraction
        x = self.cnn(x)  # (batch, seq_len, cnn_output_dim)

        # 2. Project to Transformer dimension
        x = self.projection(x)  # (batch, seq_len, d_model)

        # 3. Add Positional Encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # 4. Transformer Encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # 5. Attention Pooling
        attn_scores = self.attention_pool(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        x_pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)

        # 6. RUL Prediction
        rul_pred = self.rul_head(x_pooled)  # (batch, 1)

        if return_attention:
            return rul_pred, attn_weights.squeeze(-1)
        else:
            return rul_pred, None


class ResidualCNNBlock(nn.Module):
    """Residual CNN block for feature extraction."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DeepCNNTransformer(nn.Module):
    """
    Deep hybrid model with residual CNN blocks and Transformer.
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: int = 128,
        num_cnn_blocks: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        Args:
            num_features: Number of input features
            cnn_channels: Number of CNN channels
            num_cnn_blocks: Number of residual CNN blocks
            d_model: Transformer dimension
            nhead: Number of attention heads
            num_transformer_layers: Number of Transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1)

        # Residual CNN blocks
        self.cnn_blocks = nn.ModuleList([
            ResidualCNNBlock(cnn_channels, kernel_size=3, dropout=dropout)
            for _ in range(num_cnn_blocks)
        ])

        # Project to Transformer dimension
        self.cnn_to_transformer = nn.Linear(cnn_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # RUL head
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass."""
        # CNN processing
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.input_proj(x)

        for block in self.cnn_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # Project to Transformer
        x = self.cnn_to_transformer(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Attention pooling
        attn_scores = self.attention_pool(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)

        # RUL prediction
        rul_pred = self.rul_head(x_pooled)

        if return_attention:
            return rul_pred, attn_weights.squeeze(-1)
        else:
            return rul_pred, None


def create_hybrid_model(
    num_features: int,
    model_type: str = 'base',
    **kwargs
) -> nn.Module:
    """
    Factory function for hybrid models.

    Args:
        num_features: Number of input features
        model_type: 'base' or 'deep'
        **kwargs: Additional arguments

    Returns:
        Hybrid model
    """
    if model_type == 'base':
        model = HybridCNNTransformer(num_features=num_features, **kwargs)
    elif model_type == 'deep':
        model = DeepCNNTransformer(num_features=num_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_len = 30
    num_features = 17

    # Create model
    model = create_hybrid_model(num_features, model_type='base')

    # Random input
    x = torch.randn(batch_size, seq_len, num_features)

    # Forward pass
    rul_pred, attn_weights = model(x, return_attention=True)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"RUL prediction shape: {rul_pred.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
