import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all neural network models."""

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation=nn.Tanh):
        """
        Initialize the base model.

        Args:
            input_dim (int): Dimension of input features (e.g., t for ODE)
            output_dim (int): Dimension of output features (e.g., y for ODE)
            hidden_dims (list): List of hidden layer dimensions
            activation (torch.nn.Module): Activation function
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 64]
        self.activation = activation

        # This will be implemented by subclasses
        self.layers = None

    def forward(self, x):
        """Forward pass through the network."""
        raise NotImplementedError("Subclasses must implement forward method")

    def compute_derivatives(self, x, order=1):
        """
        Compute derivatives of the network output with respect to input.

        Args:
            x (torch.Tensor): Input tensor
            order (int): Order of the derivative

        Returns:
            torch.Tensor: Derivatives of the output with respect to input
        """
        x = x.clone().detach().requires_grad_(True)
        y = self.forward(x)

        # First-order derivatives
        if order >= 1:
            dy_dx = []
            for i in range(self.output_dim):
                dy_dxi = torch.autograd.grad(
                    y[:, i], x, torch.ones_like(y[:, i]),
                    create_graph=True, retain_graph=True
                )[0]
                dy_dx.append(dy_dxi)

            first_derivatives = torch.stack(dy_dx, dim=1)

            if order == 1:
                return first_derivatives

            # Higher-order derivatives (recursive computation)
            derivatives = [first_derivatives]
            current_derivative = first_derivatives

            for n in range(2, order + 1):
                higher_order = []
                for i in range(self.output_dim):
                    for j in range(self.input_dim):
                        d_higher = torch.autograd.grad(
                            current_derivative[:, i, j], x, torch.ones_like(current_derivative[:, i, j]),
                            create_graph=True, retain_graph=True
                        )[0]
                        higher_order.append(d_higher)

                # Reshape to match the expected output dimensions
                current_derivative = torch.stack(higher_order, dim=1).reshape(-1, self.output_dim, self.input_dim)
                derivatives.append(current_derivative)

            return derivatives

        return y

    def get_config(self):
        """Get model configuration."""
        return {
            'type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation.__name__
        }