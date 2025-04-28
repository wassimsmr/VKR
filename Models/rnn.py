import torch
import torch.nn as nn
from Models.base import BaseModel


class RNN(BaseModel):
    """Recurrent Neural Network for time series data."""

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, bidirectional=False):
        """
        Initialize the RNN model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_dim (int): Dimension of hidden state
            num_layers (int): Number of recurrent layers
            bidirectional (bool): Whether to use bidirectional RNN
        """
        super(RNN, self).__init__(input_dim, output_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output layer
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, output_dim)

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

        # RNN forward pass
        output, _ = self.rnn(x)

        # Apply output layer
        output = self.fc(output)

        # Reshape output if necessary
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        })
        return config


class PhysicsInformedRNN(RNN):
    """Physics-Informed RNN for solving differential equations."""

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