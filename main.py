import torch
import numpy as np
from utils.equations import HarmonicOscillator
from Models import PINN
from Trainers import UnsupervisedTrainer
from utils.visualize import plot_comparison

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a linear ODE: dy/dt = y
# Define a harmonic oscillator ODE: d²y/dt² + ω²y = 0
ho = HarmonicOscillator(
    omega=2.0,
    initial_position=1.0,
    initial_velocity=0.0
)
domain = (0, 5)
boundary_conditions = {
    'initial_value': 1.0,
    'initial_derivative': 0.0
}

# Create and train a PINN model
pinn_model = PINN(
    input_dim=1,
    output_dim=1,
    hidden_dims=[64, 64, 64],
    activation=torch.nn.Tanh
)

pinn_trainer = UnsupervisedTrainer(
    model=pinn_model,
    domain=domain,
    ode_fn=ho,
    boundary_conditions=boundary_conditions,
    collocation_points=2000,
    optimizer=torch.optim.Adam(pinn_model.parameters(), lr=0.001),
    device=device
)

pinn_metrics = pinn_trainer.train(num_epochs=5000, verbose=True)

# Plot solution
t_eval = np.linspace(domain[0], domain[1], 1000)
exact_solution = ho.exact_solution(t_eval)
t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
pinn_solution = pinn_model(t_tensor).detach().cpu().numpy().reshape(-1)

plot_comparison(
    t_eval,
    [exact_solution, pinn_solution],
    ['Exact', 'PINN'],
    title='Harmonic Oscillator Solution: d²y/dt² + ω²y = 0',
    xlabel='t',
    ylabel='y(t)'
)