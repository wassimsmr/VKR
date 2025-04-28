import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Import your models and utilities
from Models import PINN, PhysicsInformedMLP, PhysicsInformedRNN, PhysicsInformedLSTM, \
    PhysicsInformedTransformer
from utils.equations import HarmonicOscillator, NonlinearODE
from Trainers import UnsupervisedTrainer
from Solvers import FDM, FEM
from Solvers.rk_methods import RungeKuttaSolver
from utils.metrics import compute_all_errors

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for experiment results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Define which type of experiment to run
experiment_type = "harmonic_oscillator"  # Options: "harmonic_oscillator", "nonlinear"

# Define the problem based on experiment type
if experiment_type == "harmonic_oscillator":
    # Harmonic oscillator: d²y/dt² + ω²y = 0
    equation = HarmonicOscillator(
        omega=2.0,
        initial_position=1.0,
        initial_velocity=0.0
    )
    domain = (0, 5)  # Time domain from 0 to 5 seconds
    boundary_conditions = {
        'initial_value': 1.0,  # y(0) = 1.0
        'initial_derivative': 0.0  # y'(0) = 0.0
    }
    equation_name = "Harmonic Oscillator: d²y/dt² + 4y = 0"

elif experiment_type == "nonlinear":
    # Nonlinear ODE: dy/dt = y²
    def nonlinear_ode(t, y):
        return y ** 2


    # Exact solution: y(t) = 1/(1-t)
    def exact_solution(t):
        return 1 / (1 - t)


    equation = NonlinearODE(
        f=nonlinear_ode,
        name="dy/dt = y²",
        exact_sol=exact_solution
    )
    domain = (0, 0.9)  # Avoid t=1 where solution blows up
    boundary_conditions = {'initial_value': 1.0}
    equation_name = "Nonlinear ODE: dy/dt = y²"
else:
    raise ValueError(f"Unknown experiment type: {experiment_type}")

# Define neural network configurations
neural_models = {
    'PINN': {
        'class': PINN,
        'params': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_dims': [64, 64, 64],
            'activation': torch.nn.Tanh
        }
    },
    'MLP': {
        'class': PhysicsInformedMLP,
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

# Define classical method configurations
classical_methods = {
    'FDM': {
        'class': FDM,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'num_points': 1000
        },
        'solve_params': {'method': 'rk4'}
    },
    'RK1': {
        'class': RungeKuttaSolver,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'method': 'rk1',
            'num_points': 1000,
            'adaptive': False
        },
        'solve_params': {}
    },
    'RK2': {
        'class': RungeKuttaSolver,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'method': 'rk2',
            'num_points': 1000,
            'adaptive': False
        },
        'solve_params': {}
    },
    'RK3': {
        'class': RungeKuttaSolver,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'method': 'rk3',
            'num_points': 1000,
            'adaptive': False
        },
        'solve_params': {}
    },
    'RK4': {
        'class': RungeKuttaSolver,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'method': 'rk4',
            'num_points': 1000,
            'adaptive': False
        },
        'solve_params': {}
    },
    'RKF45': {
        'class': RungeKuttaSolver,
        'params': {
            'equation': equation,
            'domain': domain,
            'boundary_conditions': boundary_conditions,
            'method': 'rkf45',
            'adaptive': True,
            'tolerance': 1e-6
        },
        'solve_params': {}
    }
}

# Set experiment parameters
num_epochs = 2000  # Reduced for faster experiment
collocation_points = 1000
learning_rate = 0.001

# Create evaluation points for comparing solutions
t_eval = np.linspace(domain[0], domain[1], 1000)
exact_solution = equation.exact_solution(t_eval)

# Store results
results = {
    'equation': equation_name,
    'domain': domain,
    'boundary_conditions': boundary_conditions,
    'neural_models': {},
    'classical_methods': {},
    'solutions': {'exact': exact_solution},
    'training_time': {},
    'solving_time': {},
    'error_metrics': {}
}

# Train and evaluate neural network models
print(f"\n{'=' * 80}")
print(f"NEURAL NETWORK MODELS")
print(f"{'=' * 80}")

for model_name, config in neural_models.items():
    print(f"\n{'-' * 50}")
    print(f"Training {model_name}...")

    # Create model
    model_class = config['class']
    model_params = config['params']
    model = model_class(**model_params)
    model.to(device)

    # Create trainer
    trainer = UnsupervisedTrainer(
        model=model,
        domain=domain,
        ode_fn=equation,
        boundary_conditions=boundary_conditions,
        collocation_points=collocation_points,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        device=device
    )

    # Train model
    start_time = time.time()
    metrics = trainer.train(num_epochs=num_epochs, verbose=True)
    training_time = time.time() - start_time

    # Evaluate model
    t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    prediction = model(t_tensor).detach().cpu().numpy().reshape(-1)

    # Compute errors
    errors = compute_all_errors(prediction, exact_solution)

    # Store results
    results['neural_models'][model_name] = {
        'params': model_params,
        'metrics': metrics
    }
    results['solutions'][model_name] = prediction
    results['training_time'][model_name] = training_time
    results['error_metrics'][model_name] = errors

    # Print summary
    print(f"\nModel: {model_name}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Loss: {metrics['train_loss']:.6f}")
    print(f"L2 Error: {errors['l2_error']:.6f}")
    print(f"L1 Error: {errors['l1_error']:.6f}")
    print(f"Max Error: {errors['max_error']:.6f}")

# Solve using classical methods
print(f"\n{'=' * 80}")
print(f"CLASSICAL METHODS")
print(f"{'=' * 80}")

for method_name, config in classical_methods.items():
    print(f"\n{'-' * 50}")
    print(f"Solving with {method_name}...")

    # Create solver
    solver_class = config['class']
    solver_params = config['params']
    solver = solver_class(**solver_params)

    # Solve equation
    start_time = time.time()
    solver.solve(**config.get('solve_params', {}))
    solving_time = time.time() - start_time

    # Evaluate solution
    solution = solver.evaluate(t_eval)

    # Compute errors
    errors = compute_all_errors(solution, exact_solution)

    # Store results
    results['classical_methods'][method_name] = {
        'params': solver_params,
    }
    results['solutions'][method_name] = solution
    results['solving_time'][method_name] = solving_time
    results['error_metrics'][method_name] = errors

    # Print summary
    print(f"\nMethod: {method_name}")
    print(f"Solving Time: {solving_time:.6f} seconds")
    print(f"L2 Error: {errors['l2_error']:.6f}")
    print(f"L1 Error: {errors['l1_error']:.6f}")
    print(f"Max Error: {errors['max_error']:.6f}")

# Create visualizations

# 1. Solution Comparison
plt.figure(figsize=(14, 10))

# Plot exact solution
plt.plot(t_eval, exact_solution, 'k-', linewidth=2.5, label='Exact')

# Plot neural network solutions
for model_name in neural_models.keys():
    if model_name in results['solutions']:
        plt.plot(t_eval, results['solutions'][model_name], '--', linewidth=1.5, label=f"{model_name}")

# Plot classical method solutions with different line style
for method_name in classical_methods.keys():
    if method_name in results['solutions']:
        plt.plot(t_eval, results['solutions'][method_name], ':', linewidth=1.5, label=f"{method_name}")

plt.title(f'Solution Comparison: {equation_name}', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('y(t)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12, loc='best')
plt.savefig(os.path.join(results_dir, 'solution_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Error Comparison (Log Scale)
plt.figure(figsize=(14, 10))

# Neural network errors
for model_name in neural_models.keys():
    if model_name in results['solutions']:
        error = np.abs(results['solutions'][model_name] - exact_solution)
        plt.semilogy(t_eval, error, '--', linewidth=1.5, label=f"{model_name} Error")

# Classical method errors with different line style
for method_name in classical_methods.keys():
    if method_name in results['solutions']:
        error = np.abs(results['solutions'][method_name] - exact_solution)
        plt.semilogy(t_eval, error, ':', linewidth=1.5, label=f"{method_name} Error")

plt.title('Absolute Error Comparison (Log Scale)', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('|y - y_exact| (Log Scale)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12, loc='best')
plt.savefig(os.path.join(results_dir, 'error_comparison_log.png'), dpi=300, bbox_inches='tight')

# 3. Computation Time Comparison
plt.figure(figsize=(14, 10))

# Prepare data for bar chart
all_methods = list(neural_models.keys()) + list(classical_methods.keys())
times = []

for method in all_methods:
    if method in results['training_time']:
        times.append(results['training_time'][method])
    elif method in results['solving_time']:
        times.append(results['solving_time'][method])
    else:
        times.append(0)

# Plot bars with different colors for neural vs classical
colors = ['blue'] * len(neural_models) + ['green'] * len(classical_methods)
plt.bar(all_methods, times, color=colors)

plt.title('Computation Time Comparison', fontsize=16)
plt.xlabel('Method', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.yscale('log')  # Log scale for better visualization
plt.grid(True, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'computation_time.png'), dpi=300, bbox_inches='tight')

# 4. L2 Error Comparison
plt.figure(figsize=(14, 10))

# Prepare data for bar chart
l2_errors = []
for method in all_methods:
    if method in results['error_metrics']:
        l2_errors.append(results['error_metrics'][method]['l2_error'])
    else:
        l2_errors.append(0)

# Plot bars with different colors for neural vs classical
plt.bar(all_methods, l2_errors, color=colors)

plt.title('L2 Error Comparison', fontsize=16)
plt.xlabel('Method', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)
plt.yscale('log')  # Log scale for better visualization
plt.grid(True, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'l2_error_comparison.png'), dpi=300, bbox_inches='tight')

# 5. Accuracy vs Computation Time (Scatter Plot)
plt.figure(figsize=(14, 10))

# Prepare data for scatter plot
x_values = []  # Computation times
y_values = []  # L2 errors
labels = []  # Method names
scatter_colors = []  # Point colors

for method in neural_models.keys():
    if method in results['training_time'] and method in results['error_metrics']:
        x_values.append(results['training_time'][method])
        y_values.append(results['error_metrics'][method]['l2_error'])
        labels.append(method)
        scatter_colors.append('blue')

for method in classical_methods.keys():
    if method in results['solving_time'] and method in results['error_metrics']:
        x_values.append(results['solving_time'][method])
        y_values.append(results['error_metrics'][method]['l2_error'])
        labels.append(method)
        scatter_colors.append('green')

# Create scatter plot with method labels
plt.figure(figsize=(14, 10))
scatter = plt.scatter(x_values, y_values, c=scatter_colors, s=100, alpha=0.7)

# Add method labels to each point
for i, label in enumerate(labels):
    plt.annotate(label, (x_values[i], y_values[i]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=12, fontweight='bold')

plt.title('Accuracy vs Computation Time', fontsize=16)
plt.xlabel('Computation Time (seconds)', fontsize=14)
plt.ylabel('L2 Error (Log Scale)', fontsize=14)
plt.xscale('log')  # Log scale for time
plt.yscale('log')  # Log scale for error
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'accuracy_vs_time.png'), dpi=300, bbox_inches='tight')

# Save results summary to text file
with open(os.path.join(results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"DIFFERENTIAL EQUATION SOLVER COMPARISON: {equation_name}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Equation: {equation_name}\n")
    f.write(f"Domain: {domain}\n")
    f.write(f"Boundary Conditions: {boundary_conditions}\n")
    f.write("\n")

    f.write("NEURAL NETWORK MODELS\n")
    f.write("-" * 80 + "\n")
    f.write(
        f"{'Model':<15} {'Training Time (s)':<20} {'Final Loss':<15} {'L2 Error':<15} {'L1 Error':<15} {'Max Error':<15}\n")
    f.write("-" * 100 + "\n")

    for model_name in neural_models.keys():
        if model_name in results['training_time'] and model_name in results['error_metrics']:
            training_time = f"{results['training_time'].get(model_name, 0):.2f}"
            final_loss = f"{results['neural_models'][model_name]['metrics'].get('train_loss', 0):.6e}"
            l2_error = f"{results['error_metrics'][model_name].get('l2_error', 0):.6e}"
            l1_error = f"{results['error_metrics'][model_name].get('l1_error', 0):.6e}"
            max_error = f"{results['error_metrics'][model_name].get('max_error', 0):.6e}"

            f.write(
                f"{model_name:<15} {training_time:<20} {final_loss:<15} {l2_error:<15} {l1_error:<15} {max_error:<15}\n")

    f.write("\n")
    f.write("CLASSICAL METHODS\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Method':<15} {'Solving Time (s)':<20} {'L2 Error':<15} {'L1 Error':<15} {'Max Error':<15}\n")
    f.write("-" * 90 + "\n")

    for method_name in classical_methods.keys():
        if method_name in results['solving_time'] and method_name in results['error_metrics']:
            solving_time = f"{results['solving_time'].get(method_name, 0):.6f}"
            l2_error = f"{results['error_metrics'][method_name].get('l2_error', 0):.6e}"
            l1_error = f"{results['error_metrics'][method_name].get('l1_error', 0):.6e}"
            max_error = f"{results['error_metrics'][method_name].get('max_error', 0):.6e}"

            f.write(f"{method_name:<15} {solving_time:<20} {l2_error:<15} {l1_error:<15} {max_error:<15}\n")

    f.write("\n")
    f.write("KEY FINDINGS\n")
    f.write("-" * 80 + "\n")

    # Find best method by error
    best_method = min(results['error_metrics'].items(), key=lambda x: x[1]['l2_error'])
    f.write(f"Best method by L2 error: {best_method[0]} with error {best_method[1]['l2_error']:.6e}\n")

    # Find fastest method
    fastest_neural = min((name for name in neural_models.keys() if name in results['training_time']),
                         key=lambda x: results['training_time'][x])
    fastest_classical = min((name for name in classical_methods.keys() if name in results['solving_time']),
                            key=lambda x: results['solving_time'][x])

    f.write(f"Fastest neural network: {fastest_neural} ({results['training_time'][fastest_neural]:.2f} seconds)\n")
    f.write(
        f"Fastest classical method: {fastest_classical} ({results['solving_time'][fastest_classical]:.6f} seconds)\n")

    # Overall trade-off winner
    f.write("\nNote: For the best trade-off between accuracy and computation time, ")
    f.write("refer to the 'accuracy_vs_time.png' scatter plot.\n")

print(f"\nExperiment completed! Results saved to: {results_dir}")
print(f"Check the 'results_summary.txt' file for detailed performance metrics.")

# Show plots if running in interactive environment
plt.show()