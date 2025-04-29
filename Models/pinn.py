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


class PDE_PINN(PINN):
    """PINN specifically modified for PDE problems."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dims=None, activation=torch.nn.Tanh):
        super(PDE_PINN, self).__init__(input_dim, output_dim, hidden_dims, activation)

    def compute_pde_residual(self, x, t, pde):
        """
        Compute the residual of the PDE.

        Args:
            x (torch.Tensor): Spatial points
            t (torch.Tensor): Time points
            pde (callable): Function that defines the PDE

        Returns:
            torch.Tensor: Residual of the PDE
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        # Create input tensor
        input_tensor = torch.cat([x, t], dim=1)

        # Forward pass
        u = self.forward(input_tensor)

        # Compute derivatives
        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0]

        u_t = torch.autograd.grad(
            u, t, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        # Return PDE residual
        return pde(x, t, u, u_x, u_xx, u_t)