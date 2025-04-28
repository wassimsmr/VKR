import torch
from Models.mlp import MLP


class PINN(MLP):
    """Physics-Informed Neural Network."""

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation=torch.nn.Tanh):
        """
        Initialize the PINN model.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
            hidden_dims (list): List of hidden layer dimensions
            activation (torch.nn.Module): Activation function
        """
        super(PINN, self).__init__(input_dim, output_dim, hidden_dims, activation)

        # PINNs are essentially MLPs with specialized training
        # The architecture is the same, but the loss function will incorporate
        # physics constraints during training

    def compute_residual(self, x, ode_fn):
        """
        Compute the residual of the differential equation.

        Args:
            x (torch.Tensor): Input tensor
            ode_fn (callable): Function that defines the ODE in the form dy/dt = f(t, y)

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