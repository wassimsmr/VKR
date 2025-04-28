import torch
import torch.nn as nn
import math
from Models.base import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""

    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding.

        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(BaseModel):
    """Transformer-based architecture for time series data."""

    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        """
        Initialize the Transformer model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super(Transformer, self).__init__(input_dim, output_dim)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        # Reshape input if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Transpose for transformer (seq_len, batch, feature)
        x = x.permute(1, 0, 2)

        # Embed input to model dimension
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Output layer
        x = self.output_layer(x)

        # Transpose back to (batch, seq_len, feature)
        x = x.permute(1, 0, 2)

        # Reshape output if necessary
        if x.shape[1] == 1:
            x = x.squeeze(1)

        return x

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward
        })
        return config


class PhysicsInformedTransformer(Transformer):
    """Physics-Informed Transformer for solving differential equations."""

    def compute_residual(self, x, ode_fn):
        """
        Compute the residual of the differential equation.

        Args:
            x (torch.Tensor): Input tensor
            ode_fn (callable): Function that defines the ODE

        Returns:
            torch.Tensor: Residual of the ODE
        """
        x.requires_grad_(True)
        y = self.forward(x)

        # Compute dy/dt
        dy_dt = torch.autograd.grad(
            y, x, torch.ones_like(y), create_graph=True
        )[0]

        # Compute the residual: dy/dt - f(t, y)
        residual = dy_dt - ode_fn(x, y)

        return residual