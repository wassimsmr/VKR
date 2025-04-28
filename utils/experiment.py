import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime


def run_unified_comparison(equation, domain, boundary_conditions, models_config,
                           numerical_methods=None, num_epochs=3000, device='cpu',
                           save_dir='experiments'):
    """
    Run a unified comparison of multiple models on the same equation.

    Args:
        equation: The differential equation object
        domain: Tuple (t_min, t_max) for the domain
        boundary_conditions: Dictionary of boundary conditions
        models_config: Dictionary of model configurations {name: (model_class, params)}
        numerical_methods: List of numerical methods to include {'fdm', 'fem'}
        num_epochs: Number of epochs for neural network training
        device: Device to run on ('cpu' or 'cuda')
        save_dir: Directory to save results and plots

    Returns:
        Dictionary of results
    """
    from Trainers import UnsupervisedTrainer
    from Solvers import FDM, FEM
    from utils.visualize import plot_comparison, plot_loss_history, plot_bar_chart, plot_error_comparison
    from utils.metrics import compute_all_errors

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Prepare evaluation points
    t_eval = np.linspace(domain[0], domain[1], 1000)

    # Get exact solution if available
    try:
        exact_solution = equation.exact_solution(t_eval)
        has_exact_solution = True
    except NotImplementedError:
        exact_solution = None
        has_exact_solution = False

    # Results collection
    results = {
        'equation': equation.name,
        'domain': domain,
        'boundary_conditions': boundary_conditions,
        'timestamp': timestamp,
        'models': {}
    }

    # Train and evaluate neural network models
    all_losses = {}
    all_solutions = {}
    training_times = {}
    error_metrics = {}

    if has_exact_solution:
        all_solutions['Exact'] = exact_solution

    # Train neural network models
    for model_name, (model_class, model_params) in models_config.items():
        print(f"\nTraining {model_name}...")

        # Create model
        model = model_class(**model_params)
        model.to(device)

        # Create trainer
        trainer = UnsupervisedTrainer(
            model=model,
            domain=domain,
            ode_fn=equation,
            boundary_conditions=boundary_conditions,
            collocation_points=2000,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=device
        )

        # Train model
        metrics = trainer.train(num_epochs=num_epochs, verbose=True)

        # Evaluate model
        t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = model(t_tensor).detach().cpu().numpy().reshape(-1)

        # Store results
        all_solutions[model_name] = prediction
        all_losses[model_name] = trainer.train_losses
        training_times[model_name] = metrics['training_time']

        # Compute errors if exact solution is available
        if has_exact_solution:
            error_metrics[model_name] = compute_all_errors(prediction, exact_solution)

        # Store model results
        results['models'][model_name] = {
            'type': 'neural_network',
            'class': model_class.__name__,
            'params': model_params,
            'metrics': metrics,
            'error_metrics': error_metrics.get(model_name, {})
        }

    # Run numerical methods
    if numerical_methods:
        if 'fdm' in numerical_methods:
            print("\nSolving with FDM...")
            fdm_solver = FDM(
                equation=equation,
                domain=domain,
                boundary_conditions=boundary_conditions,
                num_points=1000
            )
            fdm_solution = fdm_solver.solve(method='rk4')
            fdm_eval = fdm_solver.evaluate(t_eval)

            all_solutions['FDM'] = fdm_eval

            # Compute errors if exact solution is available
            if has_exact_solution:
                error_metrics['FDM'] = compute_all_errors(fdm_eval, exact_solution)

            # Store results
            results['models']['FDM'] = {
                'type': 'numerical',
                'class': 'FDM',
                'error_metrics': error_metrics.get('FDM', {})
            }

        if 'fem' in numerical_methods:
            try:
                print("\nSolving with FEM...")
                fem_solver = FEM(
                    equation=equation,
                    domain=domain,
                    boundary_conditions=boundary_conditions,
                    num_elements=100
                )
                fem_solution = fem_solver.solve()
                fem_eval = fem_solver.evaluate(t_eval)

                all_solutions['FEM'] = fem_eval

                # Compute errors if exact solution is available
                if has_exact_solution:
                    error_metrics['FEM'] = compute_all_errors(fem_eval, exact_solution)

                # Store results
                results['models']['FEM'] = {
                    'type': 'numerical',
                    'class': 'FEM',
                    'error_metrics': error_metrics.get('FEM', {})
                }
            except Exception as e:
                print(f"Error running FEM solver: {e}")

    # Generate plots
    # 1. Solution Comparison
    plt.figure(figsize=(12, 8))
    plot_comparison(
        t_eval,
        list(all_solutions.values()),
        list(all_solutions.keys()),
        title=f'Solution Comparison: {equation.name}',
        xlabel='t',
        ylabel='y(t)',
        save_path=os.path.join(experiment_dir, 'solution_comparison.png')
    )

    # 2. Training Loss History
    if all_losses:
        plot_loss_history(
            all_losses,
            title='Training Loss Comparison',
            xlabel='Epoch',
            ylabel='Loss',
            figsize=(12, 8),
            save_path=os.path.join(experiment_dir, 'loss_comparison.png')
        )

    # 3. Training Time Comparison
    if training_times:
        plot_bar_chart(
            list(training_times.keys()),
            list(training_times.values()),
            title='Training Time Comparison',
            xlabel='Model',
            ylabel='Time (s)',
            figsize=(12, 8),
            save_path=os.path.join(experiment_dir, 'training_time_comparison.png')
        )

    # 4. Error Comparison
    if error_metrics:
        # L2 Error
        l2_errors = {model: metrics.get('l2_error', 0) for model, metrics in error_metrics.items()}
        plot_bar_chart(
            list(l2_errors.keys()),
            list(l2_errors.values()),
            title='L2 Error Comparison',
            xlabel='Model',
            ylabel='L2 Error',
            figsize=(12, 8),
            save_path=os.path.join(experiment_dir, 'l2_error_comparison.png')
        )

        # Comprehensive error metrics
        plot_error_comparison(
            error_metrics,
            title='Error Metrics Comparison',
            figsize=(14, 10),
            save_path=os.path.join(experiment_dir, 'error_metrics_comparison.png')
        )

    # Save results to JSON
    results_file = os.path.join(experiment_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=lambda o: str(o))

    print(f"\nExperiment completed. Results saved to {experiment_dir}")

    return results


def generate_experiment_report(results, report_path=None):
    """
    Generate a readable report from experiment results.

    Args:
        results (dict): Results dictionary from run_unified_comparison
        report_path (str, optional): Path to save the report

    Returns:
        str: Report text
    """
    report = []

    # Header
    report.append("=" * 80)
    report.append(f"EXPERIMENT REPORT: {results['equation']}")
    report.append("=" * 80)
    report.append(f"Timestamp: {results['timestamp']}")
    report.append(f"Domain: {results['domain']}")
    report.append(f"Boundary Conditions: {results['boundary_conditions']}")
    report.append("")

    # Models section
    report.append("MODELS COMPARISON")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Type':<15} {'Training Time':<15} {'Final Loss':<15} {'L2 Error':<15}")
    report.append("-" * 80)

    for model_name, model_data in results['models'].items():
        model_type = model_data['type']
        training_time = model_data.get('metrics', {}).get('training_time', 'N/A')
        final_loss = model_data.get('metrics', {}).get('train_loss', 'N/A')
        l2_error = model_data.get('error_metrics', {}).get('l2_error', 'N/A')

        report.append(f"{model_name:<20} {model_type:<15} {training_time:<15.2f} {final_loss:<15.6f} {l2_error:<15.6f}")

    report.append("")

    # Detailed metrics section
    report.append("DETAILED METRICS")
    report.append("-" * 80)

    for model_name, model_data in results['models'].items():
        report.append(f"Model: {model_name}")

        if model_data['type'] == 'neural_network':
            report.append(f"  Class: {model_data['class']}")
            report.append(f"  Parameters: {model_data['params']}")

            # Training metrics
            metrics = model_data.get('metrics', {})
            report.append("  Training Metrics:")
            for metric_name, metric_value in metrics.items():
                report.append(f"    {metric_name}: {metric_value}")

        # Error metrics
        error_metrics = model_data.get('error_metrics', {})
        if error_metrics:
            report.append("  Error Metrics:")
            for metric_name, metric_value in error_metrics.items():
                report.append(f"    {metric_name}: {metric_value}")

        report.append("")

    report_text = "\n".join(report)

    if report_path:
        with open(report_path, 'w') as f:
            f.write(report_text)

    return report_text