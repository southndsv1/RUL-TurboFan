"""
Temporal Convolutional Network (TCN) for RUL Prediction

Implements TCN with:
- Dilated causal convolutions
- Residual connections
- Weight normalization
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List


class Chomp1d(nn.Module):
    """
    Removes padding from the end of a sequence to maintain causality.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Chomped tensor
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolutions and residual connection.

    Architecture:
    Conv1d -> Chomp1d -> ReLU -> Dropout ->
    Conv1d -> Chomp1d -> ReLU -> Dropout ->
    Residual connection
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Kernel size for convolution
            stride: Stride for convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super().__init__()

        # First convolution
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with multiple temporal blocks.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNRUL(nn.Module):
    """
    TCN model for RUL prediction.

    Architecture:
    1. Temporal Convolutional Network
    2. Global average pooling
    3. Fully connected layers for regression
    """

    def __init__(
        self,
        num_features: int,
        num_channels: List[int] = [64, 128, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_features: Number of input features
            num_channels: List of channel sizes for TCN layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_channels = num_channels

        # TCN
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # RUL prediction head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            rul_pred: RUL predictions of shape (batch_size, 1)
        """
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch, num_features, seq_len)

        # TCN encoding
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], seq_len)

        # Global average pooling over time
        pooled = tcn_out.mean(dim=2)  # (batch, num_channels[-1])

        # RUL prediction
        rul_pred = self.fc(pooled)  # (batch, 1)

        return rul_pred


class TCNRULWithAttention(nn.Module):
    """
    TCN model with attention mechanism for RUL prediction.
    """

    def __init__(
        self,
        num_features: int,
        num_channels: List[int] = [64, 128, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            num_features: Number of input features
            num_channels: List of channel sizes for TCN layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_channels = num_channels

        # TCN
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.Tanh(),
            nn.Linear(num_channels[-1] // 2, 1),
        )

        # RUL prediction head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_attention: If True, return attention weights

        Returns:
            rul_pred: RUL predictions of shape (batch_size, 1)
            attention_weights: Attention weights (if return_attention=True)
        """
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch, num_features, seq_len)

        # TCN encoding
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], seq_len)

        # Transpose for attention
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, num_channels[-1])

        # Attention weights
        attn_scores = self.attention(tcn_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = (tcn_out * attn_weights).sum(dim=1)  # (batch, num_channels[-1])

        # RUL prediction
        rul_pred = self.fc(context)  # (batch, 1)

        if return_attention:
            return rul_pred, attn_weights.squeeze(-1)
        else:
            return rul_pred, None


def create_tcn_model(
    num_features: int,
    model_size: str = 'base',
    use_attention: bool = True,
    kernel_size: int = 3,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create TCN models.

    Args:
        num_features: Number of input features
        model_size: 'small', 'base', 'large'
        use_attention: Use attention mechanism
        kernel_size: Kernel size for convolutions
        dropout: Dropout probability

    Returns:
        TCN model
    """
    # Model configurations
    configs = {
        'small': [32, 64, 64],
        'base': [64, 128, 128, 64],
        'large': [128, 256, 256, 128, 64],
    }

    num_channels = configs.get(model_size, configs['base'])

    if use_attention:
        model = TCNRULWithAttention(
            num_features=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    else:
        model = TCNRUL(
            num_features=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_len = 30
    num_features = 17

    # Create model
    model = create_tcn_model(num_features, model_size='base', use_attention=True)

    # Random input
    x = torch.randn(batch_size, seq_len, num_features)

    # Forward pass
    rul_pred, attn_weights = model(x, return_attention=True)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"RUL prediction shape: {rul_pred.shape}")
    print(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
