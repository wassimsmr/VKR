import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Import your models and utilities
from Models import PDENN, PhysicsInformedPDEMLP,PhysicsInformedPDERNN, PhysicsInformedPDELSTM
from utils.equations import PoissonEquation
from Trainers import PDETrainer
#from utils.metrics import compute_residual_error

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for experiment results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"pde_experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)


# Define a PDE problem - Poisson equation: ∇²u = f(x,y)
# With source term f(x,y) = -8π²sin(2πx)sin(2πy)
# The exact solution is u(x,y) = sin(2πx)sin(2πy)

def source_term(x, y):
    return -8 * (np.pi ** 2) * torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)


def exact_solution(x, y):
    return torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)


# Create the PDE
poisson_eq = PoissonEquation(source_term=source_term)

# Domain for the problem: [0,1] x [0,1]
domain = {
    'x': (0, 1),
    'y': (0, 1)
}

# Boundary conditions: u = 0 on boundaries
boundary_conditions = {
    'dirichlet': 0.0  # Zero on all boundaries
}

# Define neural network configurations
network_configs = {
    'PINN': {
        'class': PhysicsInformedPDEMLP,  # Using MLP with PDE capabilities
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dims': [40, 40, 40],
            'activation': torch.nn.Tanh
        }
    },
    'MLP': {
        'class': PhysicsInformedPDEMLP,
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dims': [20, 20],
            'activation': torch.nn.Tanh
        }
    },
    'RNN': {
        'class': PhysicsInformedPDERNN,
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dim': 40,
            'num_layers': 2
        }
    },
    'LSTM': {
        'class': PhysicsInformedPDELSTM,
        'params': {
            'input_dim': 2,
            'output_dim': 1,
            'hidden_dim': 40,
            'num_layers': 2
        }
    }
}

# Set experiment parameters
num_epochs = 5000
collocation_points = 1000
boundary_points = 200

# Create evaluation grid for comparing solutions
n_points = 50
x_eval = torch.linspace(0, 1, n_points)
y_eval = torch.linspace(0, 1, n_points)
x_grid, y_grid = torch.meshgrid(x_eval, y_eval, indexing='ij')
x_flat = x_grid.flatten().reshape(-1, 1).to(device)
y_flat = y_grid.flatten().reshape(-1, 1).to(device)
input_eval = torch.cat([x_flat, y_flat], dim=1)

# Create exact solution for comparison
exact_sol = exact_solution(x_flat, y_flat).reshape(n_points, n_points).cpu().numpy()

# Store results
results = {
    'equation': poisson_eq.name,
    'domain': domain,
    'boundary_conditions': boundary_conditions,
    'networks': {},
    'solutions': {'exact': exact_sol},
    'training_time': {},
    'residuals': {}
}

# Train and evaluate each network
for network_name, config in network_configs.items():
    print(f"\n{'=' * 50}")
    print(f"Training {network_name}...")
    print(f"{'=' * 50}")

    # Create model
    model_class = config['class']
    model_params = config['params']
    model = model_class(**model_params)
    model.to(device)

    # Create trainer
    trainer = PDETrainer(
        model=model,
        domain=domain,
        pde=poisson_eq,
        boundary_conditions=boundary_conditions,
        collocation_points=collocation_points,
        boundary_points=boundary_points,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=device
    )

    # Train model
    start_time = time.time()
    metrics = trainer.train(num_epochs=num_epochs, verbose=True)
    training_time = time.time() - start_time

    # Evaluate model
    model.eval()
    with torch.no_grad():
        prediction = model(input_eval).reshape(n_points, n_points).cpu().numpy()

    # Compute residual
    x_sample = torch.rand(500, 1, device=device)
    y_sample = torch.rand(500, 1, device=device)
    x_sample.requires_grad_(True)
    y_sample.requires_grad_(True)
    residual = torch.mean(model.compute_pde_residual(x_sample, y_sample, poisson_eq) ** 2).item()

    # Store results
    results['networks'][network_name] = {
        'params': model_params,
        'metrics': metrics
    }
    results['solutions'][network_name] = prediction
    results['training_time'][network_name] = training_time
    results['residuals'][network_name] = residual

    # Print summary
    print(f"\nNetwork: {network_name}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Loss: {metrics['train_loss']:.6f}")
    print(f"Residual: {residual:.6e}")

    # Create visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.pcolormesh(x_grid.cpu().numpy(), y_grid.cpu().numpy(), prediction, cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title(f'{network_name} Solution')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 3, 2)
    plt.pcolormesh(x_grid.cpu().numpy(), y_grid.cpu().numpy(), exact_sol, cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title('Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 3, 3)
    error = np.abs(prediction - exact_sol)
    plt.pcolormesh(x_grid.cpu().numpy(), y_grid.cpu().numpy(), error, cmap='hot')
    plt.colorbar(label='Error')
    plt.title('Absolute Error')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{network_name}_comparison.png'), dpi=300)
    plt.close()

# Create comparison visualizations

# Training time comparison
plt.figure(figsize=(12, 6))
networks = list(results['networks'].keys())
times = [results['training_time'][net] for net in networks]
plt.bar(networks, times)
plt.title('Training Time Comparison')
plt.xlabel('Network Architecture')
plt.ylabel('Training Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_time_comparison.png'), dpi=300)
plt.close()

# Residual comparison
plt.figure(figsize=(12, 6))
residuals = [results['residuals'][net] for net in networks]
plt.bar(networks, residuals)
plt.title('PDE Residual Comparison')
plt.xlabel('Network Architecture')
plt.ylabel('Mean Squared Residual')
plt.yscale('log')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'residual_comparison.png'), dpi=300)
plt.close()

# Error comparison
plt.figure(figsize=(12, 6))
errors = []
for net in networks:
    error = np.mean(np.abs(results['solutions'][net] - results['solutions']['exact']))
    errors.append(error)

plt.bar(networks, errors)
plt.title('Mean Absolute Error Comparison')
plt.xlabel('Network Architecture')
plt.ylabel('Mean Absolute Error')
plt.yscale('log')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'error_comparison.png'), dpi=300)
plt.close()

# 3D visualization of solutions
from mpl_toolkits.mplot3d import Axes3D

for net in networks:
    fig = plt.figure(figsize=(15, 10))

    # Network solution
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x_grid.cpu().numpy(), y_grid.cpu().numpy(),
                             results['solutions'][net], cmap='viridis', edgecolor='none')
    ax1.set_title(f'{net} Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')

    # Exact solution
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x_grid.cpu().numpy(), y_grid.cpu().numpy(),
                             results['solutions']['exact'], cmap='viridis', edgecolor='none')
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{net}_3d_comparison.png'), dpi=300)
    plt.close()

print(f"\nExperiment completed! Results saved to: {results_dir}")