# local_utils.py (replacement for colab_utils.py)
import os
import datetime
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def check_gpu_availability():
    """
    Check if GPU is available locally.

    Returns:
        str: Device to use ('cuda' or 'cpu')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_cached(0) / 1e9:.2f} GB")
    else:
        print("GPU is not available, using CPU instead.")

    return device


def save_experiment_results(experiment_name, config, metrics, figures_dir="figures", results_dir="results"):
    """
    Save experiment results locally.

    Args:
        experiment_name (str): Name of the experiment
        config (dict): Experiment configuration
        metrics (dict): Experiment metrics
        figures_dir (str): Directory to save figures
        results_dir (str): Directory to save results
    """
    # Create directories if they don't exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save configuration and metrics as JSON
    result = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config": config,
        "metrics": metrics
    }

    result_file = os.path.join(results_dir, f"{experiment_name}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {result_file}")

    return result_file


def save_model(model, model_name, models_dir="models"):
    """
    Save the model locally.

    Args:
        model (torch.nn.Module): Model to save
        model_name (str): Name of the model
        models_dir (str): Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_file = os.path.join(models_dir, f"{model_name}_{timestamp}.pth")
    torch.save(model.state_dict(), model_file)

    print(f"Model saved to {model_file}")

    return model_file


def load_model(model, model_file):
    """
    Load the model locally.

    Args:
        model (torch.nn.Module): Model to load weights into
        model_file (str): Path to the model file
    """
    model.load_state_dict(torch.load(model_file))

    return model