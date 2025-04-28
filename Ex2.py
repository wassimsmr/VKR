import torch
import numpy as np
from utils.equations import NonlinearODE
from Models import PINN
from Solvers import FDM
from Trainers import UnsupervisedTrainer
from utils.visualize import plot_comparison

# Check if CUDA (GPU) is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define nonlinear ODE function: dy/dt = y²
def nonlinear_ode(t, y):
    return y**2

# Define exact solution: y(t) = 1/(1-t)
def exact_solution(t):
    return 1 / (1 - t)

# Create nonlinear ODE
nl_ode = NonlinearODE(
    f=nonlinear_ode,
    name="dy/dt = y²",
    exact_sol=exact_solution
)

domain = (0, 0.9)  # Avoid t=1 where solution blows up
boundary_conditions = {'initial_value': 1.0}

# Create and train a PINN model with deeper network
pinn_model = PINN(
    input_dim=1,
    output_dim=1,
    hidden_dims=[64, 64, 64, 64],
    activation=torch.nn.Tanh
)

pinn_trainer = UnsupervisedTrainer(
    model=pinn_model,
    domain=domain,
    ode_fn=nl_ode,
    boundary_conditions=boundary_conditions,
    collocation_points=2000,
    optimizer=torch.optim.Adam(pinn_model.parameters(), lr=0.001),
    device=device
)

pinn_metrics = pinn_trainer.train(num_epochs=5000, verbose=True)

# Compare with FDM
fdm_solver = FDM(
    equation=nonlinear_ode,
    domain=domain,
    boundary_conditions=boundary_conditions,
    num_points=1000
)
fdm_solution = fdm_solver.solve(method='rk4')

# Plot and compare
t_eval = np.linspace(domain[0], domain[1], 1000)
exact_sol = exact_solution(t_eval)
t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
pinn_sol = pinn_model(t_tensor).detach().cpu().numpy().reshape(-1)
fdm_sol = fdm_solver.evaluate(t_eval)

plot_comparison(
    t_eval,
    [exact_sol, pinn_sol, fdm_sol],
    ['Exact', 'PINN', 'FDM'],
    title='Nonlinear ODE Solution: dy/dt = y²',
    xlabel='t',
    ylabel='y(t)'
)