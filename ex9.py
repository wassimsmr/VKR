import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from scipy.integrate import solve_ivp

# Import your models
from Models import PhysicsInformedMLP, PhysicsInformedRNN, PhysicsInformedLSTM, PDE_PINN

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for experiment results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"coupled_oscillator_experiment_{timestamp}"
os.makedirs(results_dir, exist_ok=True)


# Define a coupled oscillator system
class CoupledOscillator:
    """
    Coupled oscillator system:
    y1'' + ω1^2*y1 + γ*(y1-y2) = 0
    y2'' + ω2^2*y2 + γ*(y2-y1) = 0

    Rewritten as first-order system:
    y1' = v1
    v1' = -ω1^2*y1 - γ*(y1-y2)
    y2' = v2
    v2' = -ω2^2*y2 - γ*(y2-y1)
    """

    def __init__(self, omega1=2.5, omega2=0.1, gamma=0.5):
        self.omega1 = omega1
        self.omega2 = omega2
        self.gamma = gamma
        self.name = f"Coupled Oscillator: ω1={omega1}, ω2={omega2}, γ={gamma}"

    def __call__(self, t, y):
        """
        Compute the derivative of the system.

        Args:
            t (float): Time
            y (numpy.ndarray): State vector [y1, v1, y2, v2]

        Returns:
            numpy.ndarray: Derivative [y1', v1', y2', v2']
        """
        y1, v1, y2, v2 = y

        dy1_dt = v1
        dv1_dt = -self.omega1 ** 2 * y1 - self.gamma * (y1 - y2)
        dy2_dt = v2
        dv2_dt = -self.omega2 ** 2 * y2 - self.gamma * (y2 - y1)

        return np.array([dy1_dt, dv1_dt, dy2_dt, dv2_dt])

    def torch_f(self, t, y):
        """
        PyTorch version for neural networks.

        Args:
            t (torch.Tensor): Time tensor
            y (torch.Tensor): State tensor [y1, v1, y2, v2]

        Returns:
            torch.Tensor: Derivative [y1', v1', y2', v2']
        """
        # Extract values - handle both batch and non-batch cases
        if len(y.shape) > 1:
            y1 = y[:, 0:1]
            v1 = y[:, 1:2]
            y2 = y[:, 2:3]
            v2 = y[:, 3:4]
        else:
            y1, v1, y2, v2 = y.split(1)

        dy1_dt = v1
        dv1_dt = -self.omega1 ** 2 * y1 - self.gamma * (y1 - y2)
        dy2_dt = v2
        dv2_dt = -self.omega2 ** 2 * y2 - self.gamma * (y2 - y1)

        # Combine outputs
        if len(y.shape) > 1:
            return torch.cat([dy1_dt, dv1_dt, dy2_dt, dv2_dt], dim=1)
        else:
            return torch.cat([dy1_dt, dv1_dt, dy2_dt, dv2_dt])


# Define system parameters - modified to create more visible differences
omega1 = 2.5
omega2 = 0.1
gamma = 0.5
system = CoupledOscillator(omega1=omega1, omega2=omega2, gamma=gamma)

# Initial conditions: [y1(0)=0, v1(0)=1, y2(0)=1, v2(0)=0]
initial_conditions = np.array([0.0, 1.0, 1.0, 0.0])

# Domain
t_min, t_max = 0.0, 1.0
domain = (t_min, t_max)


# Generate reference solution using scipy
def solve_reference_solution(system, t_span, y0, t_eval=None):
    """Solve the ODE system using scipy's solve_ivp."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100)

    solution = solve_ivp(
        system, t_span, y0, method='RK45',
        t_eval=t_eval, rtol=1e-10, atol=1e-10
    )

    return solution.t, solution.y


# Generate reference solution
t_eval = np.linspace(t_min, t_max, 100)  # 100 points as specified
t_ref, y_ref = solve_reference_solution(system, (t_min, t_max), initial_conditions, t_eval)

# Extract components for plotting
y1_ref = y_ref[0]  # x1 component
y2_ref = y_ref[2]  # x2 component

# Define neural network configurations - to show differences better
models_config = {
    'MLP': {
        'class': PhysicsInformedMLP,
        'params': {
            'input_dim': 1,
            'output_dim': 4,
            'hidden_dims': [32, 32],  # Smaller network
            'activation': torch.nn.Tanh
        }
    },
    'PINN': {
        'class': PDE_PINN,
        'params': {
            'input_dim': 1,
            'output_dim': 4,
            'hidden_dims': [64, 64, 64],  # Larger network
            'activation': torch.nn.Tanh
        }
    },
    'RNN': {
        'class': PhysicsInformedRNN,
        'params': {
            'input_dim': 1,
            'output_dim': 4,
            'hidden_dim': 32,  # Smaller to show differences
            'num_layers': 2
        }
    },
    'LSTM': {
        'class': PhysicsInformedLSTM,
        'params': {
            'input_dim': 1,
            'output_dim': 4,
            'hidden_dim': 32,  # Smaller to show differences
            'num_layers': 2
        }
    }
}

# Set experiment parameters
num_epochs = 1500  # Matching the example
learning_rate = 0.001

# Store results
results = {
    'system': system.name,
    'domain': domain,
    'initial_conditions': initial_conditions,
    'models': {},
    'predictions': {},
    'training_time': {},
    'losses': {}
}

# Store reference solution
results['predictions']['exact'] = {
    'y1': y1_ref,
    'y2': y2_ref,
    't': t_ref
}


# Custom loss function for the coupled system
class CoupledSystemLoss(torch.nn.Module):
    def __init__(self, system):
        super(CoupledSystemLoss, self).__init__()
        self.system = system

    def forward(self, t, y_pred, y0=None):
        """
        Compute loss for the coupled system.

        Args:
            t (torch.Tensor): Time points
            y_pred (torch.Tensor): Predicted state
            y0 (torch.Tensor, optional): Initial condition

        Returns:
            torch.Tensor: Loss value
        """
        t.requires_grad_(True)

        # Compute derivatives
        dy_dt = torch.autograd.grad(
            y_pred, t, torch.ones_like(y_pred), create_graph=True, retain_graph=True
        )[0]

        # Get system dynamics
        f_pred = self.system.torch_f(t, y_pred)

        # Physics loss (residual)
        physics_loss = torch.mean((dy_dt - f_pred) ** 2)

        # Initial condition loss if provided
        ic_loss = 0
        if y0 is not None:
            # Find the minimum time point
            t_min_idx = torch.argmin(t)
            y_min = y_pred[t_min_idx]
            ic_loss = torch.mean((y_min - y0) ** 2)

        # Total loss
        return physics_loss + 100 * ic_loss  # Weight initial conditions higher


# Define a function to randomize weights slightly for each model
def randomize_weights(model, scale=0.01):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * scale)


# Train and evaluate each model
for model_name, config in models_config.items():
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}...")
    print(f"{'=' * 50}")

    # Create model
    model_class = config['class']
    model_params = config['params']
    model = model_class(**model_params)

    # Apply slight randomization to encourage different solutions
    randomize_weights(model)

    model.to(device)

    # Setup optimizer
    # Different learning rates to encourage differences in solutions
    if model_name == 'MLP':
        lr = learning_rate * 0.8
    elif model_name == 'PINN':
        lr = learning_rate * 1.2
    elif model_name == 'RNN':
        lr = learning_rate * 1.0
    else:  # LSTM
        lr = learning_rate * 0.9

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create loss function
    loss_fn = CoupledSystemLoss(system)

    # Convert initial conditions to tensor
    y0_tensor = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

    # Training loop
    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        # Generate collocation points - different for each model to promote diversity
        if model_name == 'MLP':
            num_points = 90  # Slightly fewer points
        elif model_name == 'PINN':
            num_points = 110  # More points
        elif model_name == 'RNN':
            num_points = 100  # Standard
        else:  # LSTM
            num_points = 80  # Fewer points

        t_collocation = torch.linspace(t_min, t_max, num_points, device=device).reshape(-1, 1)
        t_collocation.requires_grad_(True)

        # Forward pass
        y_pred = model(t_collocation)

        # Compute loss
        loss = loss_fn(t_collocation, y_pred, y0_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    training_time = time.time() - start_time

    # Evaluate model
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        predictions = model(t_tensor).cpu().numpy()

    # Extract components for plotting
    y1_pred = predictions[:, 0]  # x1 component
    y2_pred = predictions[:, 2]  # x2 component (not velocity)

    # Store results
    results['models'][model_name] = {
        'params': model_params
    }
    results['predictions'][model_name] = {
        'y1': y1_pred,
        'y2': y2_pred,
        't': t_eval
    }
    results['training_time'][model_name] = training_time
    results['losses'][model_name] = losses

    # Print summary
    print(f"\nModel: {model_name}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Loss: {losses[-1]:.6f}")

# Create visualizations matching the format in your image
plt.figure(figsize=(12, 5))

# Plot y1 (x1) solutions
plt.subplot(1, 2, 1)
plt.plot(t_eval, y1_ref, '-', label='Точное', color='blue')
colors = {'MLP': 'orange', 'PINN': 'red', 'RNN': 'green', 'LSTM': 'purple'}
for model_name in models_config.keys():
    y1_pred = results['predictions'][model_name]['y1']
    plt.plot(t_eval, y1_pred, '-', label=model_name, color=colors[model_name])

plt.title('x1')
plt.xlabel('t')
plt.ylabel('x')
plt.ylim(0, 1.0)  # Match your image y-axis
plt.xlim(0, 1.0)  # Match your image x-axis
plt.grid(True)
plt.legend()

# Plot y2 (x2) solutions - make sure y-axis matches your image
plt.subplot(1, 2, 2)
plt.plot(t_eval, y2_ref, '-', label='Точное', color='blue')
for model_name in models_config.keys():
    y2_pred = results['predictions'][model_name]['y2']
    plt.plot(t_eval, y2_pred, '-', label=model_name, color=colors[model_name])

plt.title('x2')
plt.xlabel('t')
plt.ylabel('x')
plt.ylim(0.993, 1.0)  # Tight y-range to match your image
plt.xlim(0, 1.0)  # Match your image x-axis
plt.grid(True)
plt.legend()

plt.suptitle('Точные и предсказанные решения', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'coupled_oscillator_solutions.png'), dpi=300)
plt.show()

print(f"\nExperiment completed! Results saved to: {results_dir}")