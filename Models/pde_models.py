import torch
import torch.nn as nn
from Models.base import BaseModel
from Models.mlp import MLP
from Models.rnn import RNN
from Models.lstm import LSTM



class PDENN(BaseModel):
    """Neural network for solving PDEs."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dims=None, activation=nn.Tanh):
        """
        Initialize the PDE neural network.

        Args:
            input_dim (int): Dimension of input features (typically 2 for x,y)
            output_dim (int): Dimension of output features (typically 1 for u)
            hidden_dims (list): List of hidden layer dimensions
            activation (torch.nn.Module): Activation function
        """
        super(PDENN, self).__init__(input_dim, output_dim, hidden_dims, activation)

        # Build layers
        layer_dims = [input_dim] + self.hidden_dims + [output_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation after the last layer
                layers.append(self.activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)

    def compute_pde_residual(self, x, y, pde):
        """
        Compute the residual of the PDE.

        Args:
            x (torch.Tensor): x-coordinate points
            y (torch.Tensor): y-coordinate points
            pde (callable): Function that defines the PDE

        Returns:
            torch.Tensor: Residual of the PDE
        """
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Create input tensor
        input_tensor = torch.cat([x, y], dim=1)

        # Forward pass
        u = self.forward(input_tensor)

        # Compute first derivatives
        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        u_y = torch.autograd.grad(
            u, y, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        # Compute second derivatives
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True
        )[0]

        # Compute mixed derivative
        u_xy = torch.autograd.grad(
            u_x, y, torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0]

        # Return PDE residual
        return pde(x, y, u, u_x, u_y, u_xx, u_yy, u_xy)


class PhysicsInformedPDEMLP(MLP):
    """Physics-Informed MLP for solving PDEs."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dims=None, activation=nn.Tanh):
        super(PhysicsInformedPDEMLP, self).__init__(input_dim, output_dim, hidden_dims, activation)

    def compute_pde_residual(self, x, y, pde):
        """Compute PDE residual."""
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Create input tensor
        input_tensor = torch.cat([x, y], dim=1)

        # Forward pass
        u = self.forward(input_tensor)

        # Compute derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        # Compute residual
        return pde(x, y, u, u_x, u_y, u_xx, u_yy)


class PhysicsInformedPDERNN(RNN):
    """Physics-Informed RNN for solving PDEs."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=64, num_layers=2):
        super(PhysicsInformedPDERNN, self).__init__(input_dim, output_dim, hidden_dim, num_layers)

    def forward(self, x):
        """Modified forward pass for 2D inputs."""
        # Reshape for RNN if needed
        if len(x.shape) == 2:  # [batch_size, 2]
            x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, 2]

        # RNN forward pass
        output, _ = self.rnn(x)

        # Apply output layer
        output = self.fc(output)

        # Reshape output if necessary
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    def compute_pde_residual(self, x, y, pde):
        """Compute PDE residual."""
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Create input tensor
        input_tensor = torch.cat([x, y], dim=1)

        # Forward pass
        u = self.forward(input_tensor)

        # Compute derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        # Compute residual
        return pde(x, y, u, u_x, u_y, u_xx, u_yy)


class PhysicsInformedPDELSTM(LSTM):
    """Physics-Informed LSTM for solving PDEs."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=64, num_layers=2):
        super(PhysicsInformedPDELSTM, self).__init__(input_dim, output_dim, hidden_dim, num_layers)

    def forward(self, x):
        """Modified forward pass for 2D inputs."""
        # Reshape for LSTM if needed
        if len(x.shape) == 2:  # [batch_size, 2]
            x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, 2]

        # LSTM forward pass
        output, _ = self.lstm(x)

        # Apply output layer
        output = self.fc(output)

        # Reshape output if necessary
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    def compute_pde_residual(self, x, y, pde):
        """Compute PDE residual."""
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Create input tensor
        input_tensor = torch.cat([x, y], dim=1)

        # Forward pass
        u = self.forward(input_tensor)

        # Compute derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        # Compute residual
        return pde(x, y, u, u_x, u_y, u_xx, u_yy)