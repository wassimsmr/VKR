import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output


def plot_solution(t, y_true=None, y_pred=None, y_pred_label='Predicted', y_true_label='True',
                  title='Solution', xlabel='t', ylabel='y', figsize=(10, 6), save_path=None):
    """
    Plot the solution.

    Args:
        t (numpy.ndarray): Time points
        y_true (numpy.ndarray, optional): True solution
        y_pred (numpy.ndarray, optional): Predicted solution
        y_pred_label (str): Label for predicted solution
        y_true_label (str): Label for true solution
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    if y_true is not None:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        plt.plot(t, y_true, 'b-', label=y_true_label)

    if y_pred is not None:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        plt.plot(t, y_pred, 'r--', label=y_pred_label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if y_true is not None or y_pred is not None:
        plt.legend()

    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_loss_history(losses, val_losses=None, title='Training Loss', xlabel='Epoch',
                      ylabel='Loss', figsize=(10, 6), save_path=None):
    """
    Plot the loss history.

    Args:
        losses (list): Training losses
        val_losses (list, optional): Validation losses
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-', label='Training Loss')

    if val_losses:
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_comparison(t, solutions, labels, title='Solution Comparison',
                    xlabel='t', ylabel='y(t)', reference_idx=0,
                    figsize=(10, 6), save_path=None):
    """
    Plot a comparison of multiple solutions.

    Args:
        t (numpy.ndarray): Time points
        solutions (list): List of solutions to compare
        labels (list): Labels for each solution
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        reference_idx (int): Index of the reference solution (for line style)
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    for i, (solution, label) in enumerate(zip(solutions, labels)):
        if i == reference_idx:
            # Reference solution - solid line
            plt.plot(t, solution, 'k-', linewidth=2, label=label)
        else:
            # Other solutions - dashed lines
            plt.plot(t, solution, '--', linewidth=1.5, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_error(t, error, title='Error', xlabel='t', ylabel='Error',
               figsize=(10, 6), save_path=None):
    """
    Plot the error.

    Args:
        t (numpy.ndarray): Time points
        error (numpy.ndarray): Error values
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    if isinstance(error, torch.Tensor):
        error = error.detach().cpu().numpy()

    plt.plot(t, error)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def visualize_training(epoch, losses, t, y_true, y_pred, clear=True, freq=10):
    """
    Visualize training progress.

    Args:
        epoch (int): Current epoch
        losses (list): Training losses
        t (numpy.ndarray): Time points
        y_true (numpy.ndarray): True solution
        y_pred (numpy.ndarray): Predicted solution
        clear (bool): Whether to clear the output
        freq (int): Frequency of visualization
    """
    if epoch % freq != 0:
        return

    if clear:
        clear_output(wait=True)

    print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    plt.plot(t, y_true, 'b-', label='True')
    plt.plot(t, y_pred, 'r--', label='Predicted')
    plt.title('Solution')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_phase_portrait(solution, labels=None, title='Phase Portrait',
                        xlabel='x', ylabel='y', figsize=(10, 6), save_path=None):
    """
    Plot phase portrait for a system of ODEs.

    Args:
        solution (numpy.ndarray): Solution array with shape (n_points, n_variables)
        labels (list, optional): Labels for variables
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    if solution.shape[1] < 2:
        raise ValueError("Solution must have at least 2 variables to plot phase portrait")

    plt.figure(figsize=figsize)

    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()

    plt.plot(solution[:, 0], solution[:, 1])

    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_3d_phase_portrait(solution, labels=None, title='3D Phase Portrait',
                           figsize=(10, 8), save_path=None):
    """
    Plot 3D phase portrait for a system of ODEs.

    Args:
        solution (numpy.ndarray): Solution array with shape (n_points, n_variables)
        labels (list, optional): Labels for variables
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    if solution.shape[1] < 3:
        raise ValueError("Solution must have at least 3 variables to plot 3D phase portrait")

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()

    ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])

    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_loss_history(losses_dict, title='Training Loss', xlabel='Epoch', ylabel='Loss',
                      figsize=(10, 6), save_path=None):
    """
    Plot loss history for multiple models.

    Args:
        losses_dict (dict): Dictionary mapping model names to loss histories
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    for model_name, losses in losses_dict.items():
        plt.plot(losses, label=model_name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_bar_chart(labels, values, title='Bar Chart', xlabel='', ylabel='',
                   figsize=(10, 6), save_path=None):
    """
    Plot a bar chart.

    Args:
        labels (list): Labels for the x-axis
        values (list): Values for the y-axis
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)

    plt.bar(labels, values)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_error_comparison(error_metrics, title='Error Metrics Comparison',
                          figsize=(12, 8), save_path=None):
    """
    Plot comparison of different error metrics across models.

    Args:
        error_metrics (dict): Dictionary mapping model names to error metric dictionaries
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    # Extract common metrics
    common_metrics = set()
    for model, metrics in error_metrics.items():
        common_metrics.update(metrics.keys())
    common_metrics = sorted(list(common_metrics))

    # Create figure with subplots
    fig, axes = plt.subplots(len(common_metrics), 1, figsize=figsize)
    if len(common_metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(common_metrics):
        models = []
        values = []

        for model, metrics_dict in error_metrics.items():
            if metric in metrics_dict:
                models.append(model)
                values.append(metrics_dict[metric])

        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].grid(True, axis='y')
        axes[i].set_xticklabels(models, rotation=45)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_residuals(methods, residuals, title='Equation Residual Comparison',
                   xlabel='Method', ylabel='Mean Squared Residual',
                   figsize=(10, 6), save_path=None):
    """
    Plot a comparison of residuals for different methods.

    Args:
        methods (list): List of method names
        residuals (list): List of residual values
        title (str): Plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.bar(methods, residuals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)

    plt.show()