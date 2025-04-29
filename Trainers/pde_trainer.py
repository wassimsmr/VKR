import torch
from Trainers.base import BaseTrainer
class PDETrainer(BaseTrainer):
    """Trainer for PDE problems."""

    def __init__(self, model, domain, pde, boundary_conditions, collocation_points=1000, boundary_points=200, **kwargs):
        """
        Initialize the PDE trainer.

        Args:
            model (torch.nn.Module): Neural network model
            domain (dict): Dictionary with keys 'x' and 'y', each a tuple (min, max)
            pde (callable): Function that defines the PDE
            boundary_conditions (dict): Dictionary of boundary conditions
            collocation_points (int): Number of collocation points
            boundary_points (int): Number of boundary points
            **kwargs: Additional parameters for the base trainer
        """
        super(PDETrainer, self).__init__(model, **kwargs)

        self.domain = domain
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.collocation_points = collocation_points
        self.boundary_points = boundary_points

        # Generate collocation points
        self.generate_collocation_points()

        # Generate boundary points
        self.generate_boundary_points()

    def generate_collocation_points(self):
        """Generate collocation points in the domain."""
        # Random points in the domain
        x_min, x_max = self.domain['x']
        y_min, y_max = self.domain['y']

        # Generate random points
        x = torch.rand(self.collocation_points, 1, device=self.device) * (x_max - x_min) + x_min
        y = torch.rand(self.collocation_points, 1, device=self.device) * (y_max - y_min) + y_min

        self.x_collocation = x
        self.y_collocation = y

    def generate_boundary_points(self):
        """Generate points on the boundary."""
        x_min, x_max = self.domain['x']
        y_min, y_max = self.domain['y']

        points_per_edge = self.boundary_points // 4

        # Bottom edge
        x_bottom = torch.linspace(x_min, x_max, points_per_edge, device=self.device).reshape(-1, 1)
        y_bottom = torch.ones_like(x_bottom) * y_min

        # Right edge
        y_right = torch.linspace(y_min, y_max, points_per_edge, device=self.device).reshape(-1, 1)
        x_right = torch.ones_like(y_right) * x_max

        # Top edge
        x_top = torch.linspace(x_max, x_min, points_per_edge, device=self.device).reshape(-1, 1)
        y_top = torch.ones_like(x_top) * y_max

        # Left edge
        y_left = torch.linspace(y_max, y_min, points_per_edge, device=self.device).reshape(-1, 1)
        x_left = torch.ones_like(y_left) * x_min

        # Combine all boundary points
        self.x_boundary = torch.cat([x_bottom, x_right, x_top, x_left], dim=0)
        self.y_boundary = torch.cat([y_bottom, y_right, y_top, y_left], dim=0)

    def compute_loss(self):
        """
        Compute the loss for PDE training.

        Returns:
            torch.Tensor: Total loss
        """
        # Ensure gradients are tracked
        x_collocation = self.x_collocation.clone().requires_grad_(True)
        y_collocation = self.y_collocation.clone().requires_grad_(True)

        # Compute PDE residual
        residual = self.model.compute_pde_residual(x_collocation, y_collocation, self.pde)
        residual_loss = torch.mean(residual ** 2)

        # Compute boundary loss
        x_boundary = self.x_boundary
        y_boundary = self.y_boundary

        # Create input tensor for boundary points
        boundary_input = torch.cat([x_boundary, y_boundary], dim=1)

        # Compute model output at boundary
        u_boundary = self.model(boundary_input)

        # Compute boundary condition
        bc_values = torch.zeros_like(u_boundary)

        for i in range(len(x_boundary)):
            bc_values[i] = self.pde.boundary_condition(x_boundary[i], y_boundary[i])

        # Compute boundary loss
        bc_loss = torch.mean((u_boundary - bc_values) ** 2)

        # Total loss
        total_loss = residual_loss + bc_loss

        return total_loss, (residual_loss.item(), bc_loss.item())

    def train(self, num_epochs, batch_size=None, verbose=True):
        """Train the model for PDE problems."""
        # Implement the training loop similar to UnsupervisedTrainer
        # but using the PDE-specific compute_loss method
        import time
        start_time = time.time()

        self.model.train()

        # Training loop
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss, (residual_loss, bc_loss) = self.compute_loss()
            loss.backward()
            self.optimizer.step()

            self.train_losses.append(loss.item())

            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
                      f"Residual Loss: {residual_loss:.4f}, BC Loss: {bc_loss:.4f}")

        self.training_time = time.time() - start_time

        return self.get_metrics()