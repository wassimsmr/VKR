import torch
import time
import numpy as np
from tqdm import tqdm
from Trainers.base import BaseTrainer


class UnsupervisedTrainer(BaseTrainer):
    """Trainer for unsupervised learning (PINN-style) without explicit data."""

    def __init__(self, model, domain, ode_fn, boundary_conditions, collocation_points=1000, **kwargs):
        """
        Initialize the unsupervised trainer.

        Args:
            model (torch.nn.Module): Neural network model (typically PINN)
            domain (tuple): Tuple (t_min, t_max) for the domain
            ode_fn (callable): Function that defines the ODE in the form dy/dt = f(t, y)
            boundary_conditions (dict): Dictionary of boundary conditions
            collocation_points (int): Number of collocation points
            **kwargs: Additional parameters for the base trainer
        """
        super(UnsupervisedTrainer, self).__init__(model, **kwargs)

        self.domain = domain
        self.ode_fn = ode_fn
        self.boundary_conditions = boundary_conditions
        self.collocation_points = collocation_points

        # Generate collocation points
        self.t_collocation = torch.linspace(domain[0], domain[1], collocation_points, dtype=torch.float32, device=self.device)
        self.t_collocation.requires_grad_(True)

    def compute_loss(self):
        """
        Compute the loss for PINN training.

        Returns:
            torch.Tensor: Total loss
        """
        # Residual loss at collocation points
        t_collocation = self.t_collocation.to(torch.float32)
        residual = self.model.compute_residual(t_collocation.unsqueeze(1), self.ode_fn)
        residual_loss = torch.mean(residual ** 2)

        # Boundary conditions loss
        bc_loss = torch.tensor(0.0, device=self.device)

        if 'initial_value' in self.boundary_conditions:
            # Ensure float32 data type
            t_initial = torch.tensor([[self.domain[0]]], dtype=torch.float32, device=self.device)
            y_initial_pred = self.model(t_initial)
            y_initial_true = torch.tensor([[self.boundary_conditions['initial_value']]], dtype=torch.float32,
                                          device=self.device)
            bc_loss += torch.mean((y_initial_pred - y_initial_true) ** 2)

        if 'initial_derivative' in self.boundary_conditions:
            # Ensure float32 data type
            t_initial = torch.tensor([[self.domain[0]]], dtype=torch.float32, device=self.device, requires_grad=True)
            y_initial = self.model(t_initial)
            dy_dt = torch.autograd.grad(
                y_initial, t_initial, torch.ones_like(y_initial), create_graph=True
            )[0]
            dy_dt_true = torch.tensor([[self.boundary_conditions['initial_derivative']]], dtype=torch.float32,
                                      device=self.device)
            bc_loss += torch.mean((dy_dt - dy_dt_true) ** 2)

        if 'final_value' in self.boundary_conditions:
            # Ensure float32 data type
            t_final = torch.tensor([[self.domain[1]]], dtype=torch.float32, device=self.device)
            y_final_pred = self.model(t_final)
            y_final_true = torch.tensor([[self.boundary_conditions['final_value']]], dtype=torch.float32,
                                        device=self.device)
            bc_loss += torch.mean((y_final_pred - y_final_true) ** 2)

        # Total loss
        total_loss = residual_loss + bc_loss

        return total_loss, (residual_loss.item(), bc_loss.item())

    def train(self, num_epochs, batch_size=None, verbose=True):
        """
        Train the model.

        Args:
            num_epochs (int): Number of training epochs
            batch_size (int, optional): Batch size (not used in unsupervised training)
            verbose (bool): Whether to display progress

        Returns:
            dict: Training metrics
        """
        import time
        start_time = time.time()  # Start timing

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

        # Calculate training time
        self.training_time = time.time() - start_time

        return self.get_metrics()

    def validate(self, exact_solution=None, t_points=None):
        """
        Validate the model against an exact solution if available.

        Args:
            exact_solution (callable, optional): Exact solution function
            t_points (torch.Tensor, optional): Points to evaluate the solution at

        Returns:
            float: Validation loss (L2 error if exact_solution is provided, otherwise 0)
        """
        if exact_solution is None or t_points is None:
            return 0.0

        self.model.eval()
        with torch.no_grad():
            t_points_tensor = torch.tensor(t_points, dtype=torch.float32, device=self.device).reshape(-1, 1)
            y_pred = self.model(t_points_tensor).cpu().numpy()
            y_exact = np.array([exact_solution(t) for t in t_points])

            l2_error = np.sqrt(np.mean((y_pred.reshape(-1) - y_exact) ** 2))

        self.model.train()
        return l2_error