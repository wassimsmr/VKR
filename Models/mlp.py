import torch
import torch.nn as nn
from Models.base import BaseModel


class MLP(BaseModel):
    """Standard Feedforward Neural Network (Multi-Layer Perceptron)."""

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation=nn.Tanh):
        """
        Initialize the MLP model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_dims (list): List of hidden layer dimensions
            activation (torch.nn.Module): Activation function
        """
        super(MLP, self).__init__(input_dim, output_dim, hidden_dims, activation)

        # Build layers
        layer_dims = [input_dim] + self.hidden_dims + [output_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation after the last layer
                layers.append(self.activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class PhysicsInformedMLP(MLP):
    """Physics-Informed MLP for solving differential equations."""

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