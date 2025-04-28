import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Import your models and utilities
from Models import PINN, PhysicsInformedMLP, PhysicsInformedRNN, PhysicsInformedLSTM, PhysicsInformedTransformer
from utils.equations import HarmonicOscillator
from Trainers import UnsupervisedTrainer
from Solvers import FDM
from utils.metrics import compute_all_errors

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for experiment results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Define a harmonic oscillator ODE: d²y/dt² + ω²y = 0
ho = HarmonicOscillator(
    omega=2.0,
    initial_position=1.0,
    initial_velocity=0.0
)
domain = (0, 5)  # Time domain from 0 to 5 seconds
boundary_conditions = {
    'initial_value': 1.0,  # y(0) = 1.0
    'initial_derivative': 0.0  # y'(0) = 0.0
}

# Define model configurations
models_config = {
    'MLP': {
        'class': PhysicsInformedMLP,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_dims': [64, 64, 64],
            'activation': torch.nn.Tanh
        }
    },
    'PINN': {
        'class': PINN,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_dims': [64, 64, 64],
            'activation': torch.nn.Tanh
        }
    },
    'RNN': {
        'class': PhysicsInformedRNN,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_dim': 64,
            'num_layers': 2
        }
    },
    'LSTM': {
        'class': PhysicsInformedLSTM,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_dim': 64,
            'num_layers': 2
        }
    },
    'Transformer': {
        'class': PhysicsInformedTransformer,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2
        }
    }
}

# Set experiment parameters
num_epochs = 3000
collocation_points = 2000
learning_rate = 0.001

# Create evaluation points for comparing solutions
t_eval = np.linspace(domain[0], domain[1], 1000)
exact_solution = ho.exact_solution(t_eval)

# Store results
results = {
    'models': {},
    'metrics': {},
    'training_time': {},
    'solutions': {'exact': exact_solution},
    'losses': {}
}

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

    # Create trainer
    trainer = UnsupervisedTrainer(
        model=model,
        domain=domain,
        ode_fn=ho,
        boundary_conditions=boundary_conditions,
        collocation_points=collocation_points,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        device=device
    )

    # Train model
    metrics = trainer.train(num_epochs=num_epochs, verbose=True)

    # Evaluate model
    t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    prediction = model(t_tensor).detach().cpu().numpy().reshape(-1)

    # Compute errors
    errors = compute_all_errors(prediction, exact_solution)

    # Store results
    results['models'][model_name] = model
    results['metrics'][model_name] = {**metrics, **errors}
    results['training_time'][model_name] = metrics['training_time']
    results['solutions'][model_name] = prediction
    results['losses'][model_name] = trainer.train_losses

    # Print summary
    print(f"\nModel: {model_name}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Final Loss: {metrics['train_loss']:.6f}")
    print(f"L2 Error: {errors['l2_error']:.6f}")
    print(f"L1 Error: {errors['l1_error']:.6f}")
    print(f"Max Error: {errors['max_error']:.6f}")

# Also solve using FDM for comparison
print("\nSolving with FDM...")
fdm_solver = FDM(
    equation=ho,
    domain=domain,
    boundary_conditions=boundary_conditions,
    num_points=1000
)
fdm_solver.solve(method='rk4')
fdm_solution = fdm_solver.evaluate(t_eval)
results['solutions']['FDM'] = fdm_solution

# Compute errors for FDM
fdm_errors = compute_all_errors(fdm_solution, exact_solution)
results['metrics']['FDM'] = fdm_errors

# Create visualizations

# 1. Solution Comparison
plt.figure(figsize=(12, 8))
plt.plot(t_eval, exact_solution, 'k-', linewidth=2, label='Exact')
for model_name, solution in results['solutions'].items():
    if model_name != 'exact':
        plt.plot(t_eval, solution, '--', linewidth=1.5, label=model_name)
plt.title('Harmonic Oscillator Solution Comparison', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('y(t)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig(os.path.join(results_dir, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Error Comparison
plt.figure(figsize=(12, 8))
for model_name, solution in results['solutions'].items():
    if model_name != 'exact':
        error = np.abs(solution - exact_solution)
        plt.semilogy(t_eval, error, linewidth=1.5, label=f"{model_name} Error")
plt.title('Absolute Error Comparison (Log Scale)', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('|y - y_exact|', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig(os.path.join(results_dir, 'error_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Training Loss History
plt.figure(figsize=(12, 8))
for model_name, losses in results['losses'].items():
    plt.semilogy(losses, linewidth=1.5, label=f"{model_name}")
plt.title('Training Loss History', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss (Log Scale)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig(os.path.join(results_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Training Time Comparison
model_names = list(results['training_time'].keys())
times = list(results['training_time'].values())
plt.figure(figsize=(12, 8))
plt.bar(model_names, times)
plt.title('Training Time Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.grid(True, axis='y')
plt.savefig(os.path.join(results_dir, 'training_time.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. L2 Error Comparison
model_names = list(results['metrics'].keys())
l2_errors = [results['metrics'][name].get('l2_error', 0) for name in model_names]
plt.figure(figsize=(12, 8))
plt.bar(model_names, l2_errors)
plt.title('L2 Error Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)
plt.grid(True, axis='y')
plt.savefig(os.path.join(results_dir, 'l2_error.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save results summary to text file
with open(os.path.join(results_dir, 'results_summary.txt'), 'w') as f:
    f.write("HARMONIC OSCILLATOR EXPERIMENT RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Equation: {ho.name}\n")
    f.write(f"Domain: {domain}\n")
    f.write(f"Boundary Conditions: {boundary_conditions}\n")
    f.write(f"Number of Epochs: {num_epochs}\n")
    f.write(f"Learning Rate: {learning_rate}\n\n")

    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 50 + "\n")
    f.write(
        f"{'Model':<15} {'Training Time (s)':<20} {'Final Loss':<15} {'L2 Error':<15} {'L1 Error':<15} {'Max Error':<15}\n")
    f.write("-" * 100 + "\n")

    for model_name in model_names:
        if model_name == 'FDM':
            training_time = 'N/A'
            final_loss = 'N/A'
        else:
            training_time = f"{results['training_time'].get(model_name, 0):.2f}"
            final_loss = f"{results['metrics'][model_name].get('train_loss', 0):.6e}"

        l2_error = f"{results['metrics'][model_name].get('l2_error', 0):.6e}"
        l1_error = f"{results['metrics'][model_name].get('l1_error', 0):.6e}"
        max_error = f"{results['metrics'][model_name].get('max_error', 0):.6e}"

        f.write(
            f"{model_name:<15} {training_time:<20} {final_loss:<15} {l2_error:<15} {l1_error:<15} {max_error:<15}\n")

print(f"\nExperiment completed! Results saved to: {results_dir}")