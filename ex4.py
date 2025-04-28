import torch
import numpy as np
from utils.equations import LinearODE
from Models import PINN
from utils import compute_l2_error
from Trainers import UnsupervisedTrainer
from utils.visualize import plot_comparison

# Check if CUDA (GPU) is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple problem for comparison
ode = LinearODE(a=1.0, b=0.0, initial_value=1.0)
domain = (0, 2)
boundary_conditions = {'initial_value': 1.0}

# Create different model architectures
pinn_model = PINN(
    input_dim=1, output_dim=1,
    hidden_dims=[32, 32],
    activation=torch.nn.Tanh
)

deep_pinn_model = PINN(
    input_dim=1, output_dim=1,
    hidden_dims=[32, 32, 32, 32],
    activation=torch.nn.Tanh
)

wide_pinn_model = PINN(
    input_dim=1, output_dim=1,
    hidden_dims=[128],
    activation=torch.nn.Tanh
)

# Train each model
models_to_train = {
    'Standard PINN': pinn_model,
    'Deep PINN': deep_pinn_model,
    'Wide PINN': wide_pinn_model
}

results = {}
for name, model in models_to_train.items():
    trainer = UnsupervisedTrainer(
        model=model,
        domain=domain,
        ode_fn=ode,
        boundary_conditions=boundary_conditions,
        collocation_points=1000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=device
    )

    print(f"\nTraining {name}...")
    metrics = trainer.train(num_epochs=3000, verbose=True)

    # Evaluate
    t_eval = np.linspace(domain[0], domain[1], 1000)
    t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    y_pred = model(t_tensor).detach().cpu().numpy().reshape(-1)

    results[name] = {
        'solution': y_pred,
        'metrics': metrics,
        'model': model
    }

# Compare solutions
exact_solution = ode.exact_solution(t_eval)
solutions = [exact_solution] + [results[name]['solution'] for name in models_to_train.keys()]
labels = ['Exact'] + list(models_to_train.keys())

plot_comparison(
    t_eval,
    solutions,
    labels,
    title='Comparison of Different Neural Architectures',
    xlabel='t',
    ylabel='y(t)'
)

# Print metrics
print("\nPerformance Metrics:")
for name in models_to_train.keys():
    y_pred = results[name]['solution']
    error = compute_l2_error(y_pred, exact_solution)
    train_time = results[name]['metrics']['training_time']
    print(f"{name}: L2 Error = {error:.6f}, Training Time = {train_time:.2f}s")