"""
LSTM Model for RUL Prediction

Implements bidirectional LSTM with attention mechanism as a baseline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LSTMAttention(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for RUL prediction.

    Architecture:
    1. Bidirectional LSTM layers
    2. Attention mechanism to weight time steps
    3. Fully connected layers for regression
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Args:
            num_features: Number of input features
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
        )

        # RUL prediction head
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1),
        )

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
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)

        # Attention weights
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size * num_directions)

        # RUL prediction
        rul_pred = self.fc(context)  # (batch, 1)

        if return_attention:
            return rul_pred, attn_weights.squeeze(-1)  # (batch, seq_len)
        else:
            return rul_pred, None


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model without attention (simpler baseline).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Args:
            num_features: Number of input features
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # RUL prediction head
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            rul_pred: RUL predictions of shape (batch_size, 1)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)

        # Use last time step
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size * num_directions)

        # RUL prediction
        rul_pred = self.fc(last_output)  # (batch, 1)

        return rul_pred


class GRUAttention(nn.Module):
    """
    Bidirectional GRU with attention mechanism (alternative to LSTM).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Args:
            num_features: Number of input features
            hidden_size: Hidden size of GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Use bidirectional GRU
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layers
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention mechanism
        gru_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.Tanh(),
            nn.Linear(gru_output_size // 2, 1),
        )

        # RUL prediction head
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, 1),
        )

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
        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size * num_directions)

        # Attention weights
        attn_scores = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum of GRU outputs
        context = (gru_out * attn_weights).sum(dim=1)  # (batch, hidden_size * num_directions)

        # RUL prediction
        rul_pred = self.fc(context)  # (batch, 1)

        if return_attention:
            return rul_pred, attn_weights.squeeze(-1)
        else:
            return rul_pred, None


def create_lstm_model(
    num_features: int,
    model_type: str = 'lstm_attention',
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
) -> nn.Module:
    """
    Factory function to create LSTM/GRU models.

    Args:
        num_features: Number of input features
        model_type: 'lstm_attention', 'lstm_simple', 'gru_attention'
        hidden_size: Hidden size
        num_layers: Number of layers
        dropout: Dropout probability
        bidirectional: Use bidirectional RNN

    Returns:
        LSTM/GRU model
    """
    if model_type == 'lstm_attention':
        model = LSTMAttention(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    elif model_type == 'lstm_simple':
        model = SimpleLSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    elif model_type == 'gru_attention':
        model = GRUAttention(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_len = 30
    num_features = 17

    # Create model
    model = create_lstm_model(num_features, model_type='lstm_attention')

    # Random input
    x = torch.randn(batch_size, seq_len, num_features)

    # Forward pass
    rul_pred, attn_weights = model(x, return_attention=True)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"RUL prediction shape: {rul_pred.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
