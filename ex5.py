from Models import PINN, PhysicsInformedLSTM, PhysicsInformedTransformer
from utils.equations import LinearODE, HarmonicOscillator, NonlinearODE
from utils.experiment import run_unified_comparison, generate_experiment_report
import torch

# Define equation
ode = LinearODE(a=1.0, b=0.0, initial_value=1.0)
domain = (0, 2)
boundary_conditions = {'initial_value': 1.0}

# Define models to compare
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_config = {
    'PINN': (PINN, {
        'input_dim': 1,
        'output_dim': 1,
        'hidden_dims': [32, 32, 32],
        'activation': torch.nn.Tanh
    }),
    'LSTM': (PhysicsInformedLSTM, {  # Use physics-informed version
        'input_dim': 1,
        'output_dim': 1,
        'hidden_dim': 64,
        'num_layers': 2
    }),
    'Transformer': (PhysicsInformedTransformer, {  # Use physics-informed version
        'input_dim': 1,
        'output_dim': 1,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2
    })
}

# Run the unified comparison
results = run_unified_comparison(
    equation=ode,
    domain=domain,
    boundary_conditions=boundary_conditions,
    models_config=models_config,
    numerical_methods=['fdm', 'fem'],
    num_epochs=3000,
    device=device,
    save_dir='experiments'
)

# Generate and print the report
report = generate_experiment_report(results)
print(report)