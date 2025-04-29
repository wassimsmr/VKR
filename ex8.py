import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from Models import PhysicsInformedMLP, PhysicsInformedRNN, PhysicsInformedLSTM, PDE_PINN


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for experiment results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"pde_experiment_{timestamp}"
os.makedirs(results_dir, exist_ok=True)


# Define a 1D Heat Equation PDE
class HeatEquation:
    """
    1D Heat Equation: u_t = α * u_xx

    With boundary conditions:
    u(0, t) = 0
    u(1, t) = 0

    And initial condition:
    u(x, 0) = sin(πx)
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.name = f"Heat Equation: u_t = {alpha} * u_xx"

    def __call__(self, x, t, u, u_x=None, u_xx=None, u_t=None):
        """
        Compute the PDE residual.

        Args:
            x (torch.Tensor): Spatial points
            t (torch.Tensor): Time points
            u (torch.Tensor): Solution values
            u_x, u_xx, u_t: Derivatives (computed if not provided)

        Returns:
            torch.Tensor: Residual of the PDE
        """
        if u_t is None or u_xx is None:
            if u_t is None:
                u_t = torch.autograd.grad(
                    u, t, torch.ones_like(u), create_graph=True, retain_graph=True
                )[0]

            if u_xx is None:
                if u_x is None:
                    u_x = torch.autograd.grad(
                        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
                    )[0]

                u_xx = torch.autograd.grad(
                    u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
                )[0]

        # PDE residual: u_t - α * u_xx = 0
        return u_t - self.alpha * u_xx

    def exact_solution(self, x, t):
        """Exact solution: u(x, t) = exp(-α*π²*t) * sin(πx)"""
        return torch.exp(-self.alpha * (np.pi ** 2) * t) * torch.sin(np.pi * x)

    def initial_condition(self, x):
        """Initial condition: u(x, 0) = sin(πx)"""
        return torch.sin(np.pi * x)

    def boundary_condition(self, t):
        """Boundary conditions: u(0, t) = u(1, t) = 0"""
        return torch.zeros_like(t)


# Create a custom GRU model

# Define PDE and parameters
alpha = 0.01
pde = HeatEquation(alpha=alpha)

# Domain
x_domain = (0, 1)
t_domain = (0, 1)

# Define evaluation points
nx, nt = 100, 100  # 100 points discretization as specified
x_eval = torch.linspace(x_domain[0], x_domain[1], nx)
t_eval = torch.linspace(t_domain[0], t_domain[1], nt)

# Create meshgrid for evaluation
X, T = torch.meshgrid(x_eval, t_eval, indexing='ij')
X_flat = X.flatten().reshape(-1, 1)
T_flat = T.flatten().reshape(-1, 1)
inputs = torch.cat([X_flat, T_flat], dim=1)

# Compute exact solution
exact_solution = pde.exact_solution(X_flat, T_flat).reshape(nx, nt)

# Define neural network configurations
models_config = {
    'MLP': {
        'class': PhysicsInformedMLP,
        'params': {
            'input_dim': 2,  # (x, t)
            'output_dim': 1,  # u(x, t)
            'hidden_dims': [50, 50, 50],
            'activation': torch.nn.Tanh
        }
    },
    # Add this to your models_config dictionary in the experiment:
    'PINN': {
        'class': PDE_PINN,
        'params': {
            'input_dim': 2,  # (x, t)
            'output_dim': 1,  # u(x, t)
            'hidden_dims': [50, 50, 50],
            'activation': torch.nn.Tanh
        }
    },
    'RNN': {
        'class': PhysicsInformedRNN,
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dim': 50,
            'num_layers': 2
        }
    },
    'LSTM': {
        'class': PhysicsInformedLSTM,
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dim': 50,
            'num_layers': 2
        }
    },


}

# Set experiment parameters
num_epochs = 1500  # As in your example
learning_rate = 0.001

# Store results
results = {
    'pde': pde.name,
    'domain': {
        'x': x_domain,
        't': t_domain
    },
    'models': {},
    'predictions': {},
    'training_time': {},
    'losses': {}
}

# Store exact solution for two time slices (for plotting similar to your example)
t_indices = [0, -1]  # First and last time points
results['exact'] = {
    't1': exact_solution[:, t_indices[0]].numpy(),
    't2': exact_solution[:, t_indices[1]].numpy(),
    'x': x_eval.numpy()
}


# Custom loss function for the PDE
class PDELoss(torch.nn.Module):
    def __init__(self, pde, x_domain, t_domain):
        super(PDELoss, self).__init__()
        self.pde = pde
        self.x_domain = x_domain
        self.t_domain = t_domain

    def forward(self, model, x_collocation, t_collocation, x_boundary, t_boundary, x_initial, t_initial):
        """
        Compute loss for the PDE.

        Args:
            model: Neural network model
            x_collocation, t_collocation: Interior points
            x_boundary, t_boundary: Boundary points
            x_initial, t_initial: Initial condition points

        Returns:
            torch.Tensor: Loss value
        """
        # Ensure gradients are tracked
        x_collocation.requires_grad_(True)
        t_collocation.requires_grad_(True)

        # PDE residual loss
        inputs_collocation = torch.cat([x_collocation, t_collocation], dim=1)
        u_collocation = model(inputs_collocation)

        residual = model.compute_pde_residual(x_collocation, t_collocation, self.pde)
        pde_loss = torch.mean(residual ** 2)

        # Boundary condition loss
        inputs_boundary = torch.cat([x_boundary, t_boundary], dim=1)
        u_boundary = model(inputs_boundary)
        bc_target = self.pde.boundary_condition(t_boundary)
        bc_loss = torch.mean((u_boundary - bc_target) ** 2)

        # Initial condition loss
        inputs_initial = torch.cat([x_initial, t_initial], dim=1)
        u_initial = model(inputs_initial)
        ic_target = self.pde.initial_condition(x_initial)
        ic_loss = torch.mean((u_initial - ic_target) ** 2)

        # Total loss with weights
        total_loss = pde_loss + 10.0 * bc_loss + 10.0 * ic_loss

        return total_loss, (pde_loss.item(), bc_loss.item(), ic_loss.item())


# Train and evaluate each model
for model_name, config in models_config.items():
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}...")
    print(f"{'=' * 50}")

    # Create model
    model_class = config['class']
    model_params = config['params']
    model = model_class(**model_params)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create loss function
    loss_fn = PDELoss(pde, x_domain, t_domain)

    # Generate training points
    n_collocation = 10000
    n_boundary = 1000
    n_initial = 1000

    # Random interior points
    x_collocation = torch.rand(n_collocation, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_collocation = torch.rand(n_collocation, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]

    # Boundary points (x = 0 or x = 1)
    x_boundary = torch.cat([
        torch.zeros(n_boundary // 2, 1, device=device),
        torch.ones(n_boundary // 2, 1, device=device)
    ])
    t_boundary = torch.rand(n_boundary, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]

    # Initial points (t = 0)
    x_initial = torch.rand(n_initial, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_initial = torch.zeros(n_initial, 1, device=device)

    # Training loop
    start_time = time.time()
    losses = []

    # Method for computing residuals on model
    if not hasattr(model, 'compute_pde_residual'):
        # Add method dynamically for models that don't have it
        def compute_pde_residual(self, x, t, pde):
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


        import types

        model.compute_pde_residual = types.MethodType(compute_pde_residual, model)

    for epoch in range(num_epochs):
        # Forward pass and loss computation
        loss, (pde_loss, bc_loss, ic_loss) = loss_fn(
            model, x_collocation, t_collocation, x_boundary, t_boundary, x_initial, t_initial
        )

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}, "
                  f"PDE Loss: {pde_loss:.6f}, BC Loss: {bc_loss:.6f}, IC Loss: {ic_loss:.6f}")

    training_time = time.time() - start_time

    # Evaluate model
    model.eval()
    with torch.no_grad():
        inputs_eval = torch.cat([X_flat, T_flat], dim=1)
        predictions = model(inputs_eval).cpu().reshape(nx, nt)

    # Store results
    results['models'][model_name] = {
        'params': model_params
    }
    results['predictions'][model_name] = predictions.numpy()
    results['training_time'][model_name] = training_time
    results['losses'][model_name] = losses

    # Print summary
    print(f"\nModel: {model_name}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Loss: {losses[-1]:.6f}")

# Create visualization matching the style in your example
plt.figure(figsize=(12, 5))

# Get solutions at first and last time steps
t_indices = [0, -1]
time_labels = ['t1', 't2']
subplot_titles = ['x1', 'x2']

# Plot two subplots: one for t=0 and one for t=1
for i, (t_idx, time_label, title) in enumerate(zip(t_indices, time_labels, subplot_titles)):
    plt.subplot(1, 2, i + 1)

    # Plot exact solution
    plt.plot(results['exact']['x'], results['exact'][time_label],
             '-', label='Точное', color='blue', linewidth=1.5)

    # Plot predictions from each model
    colors = {'MLP': 'orange','PINN': 'red', 'RNN': 'green','LSTM': 'purple'}
    for model_name in models_config.keys():
        solution = results['predictions'][model_name][:, t_idx]
        plt.plot(x_eval.numpy(), solution, '-', label=model_name,
                 color=colors[model_name], linewidth=1.5)

    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid(True)
    if i == 0:  # Only add legend on the first subplot
        plt.legend()

    # Set y-axis limits if needed (to match your example)
    if i == 0:
        plt.ylim(0, 1.0)
    else:
        plt.ylim(0.993, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'pde_solutions.png'), dpi=300)

# Add figure title
plt.suptitle(f'Точные и предсказанные решения', fontsize=14)
plt.subplots_adjust(top=0.85)

# Save again with title
plt.savefig(os.path.join(results_dir, 'pde_solutions_with_title.png'), dpi=300)
plt.show()

# Create caption similar to your example
fig_caption = f"Рисунок: Графики решений для обозначенных параметров, {num_epochs} эпох обучения"
with open(os.path.join(results_dir, 'figure_caption.txt'), 'w', encoding='utf-8') as f:
    f.write(fig_caption)

print(f"\nExperiment completed! Results saved to: {results_dir}")