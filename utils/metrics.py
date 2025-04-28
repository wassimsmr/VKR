import numpy as np
import torch


def compute_l2_error(y_pred, y_true):
    """
    Compute the L2 error between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: L2 error
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute L2 error
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def compute_l1_error(y_pred, y_true):
    """
    Compute the L1 error between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: L1 error
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute L1 error
    return np.mean(np.abs(y_pred - y_true))


def compute_max_error(y_pred, y_true):
    """
    Compute the maximum error between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: Maximum error
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute max error
    return np.max(np.abs(y_pred - y_true))


def compute_relative_l2_error(y_pred, y_true, eps=1e-8):
    """
    Compute the relative L2 error between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values
        eps (float): Small value to avoid division by zero

    Returns:
        float: Relative L2 error
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute relative L2 error
    norm_diff = np.sqrt(np.mean((y_pred - y_true) ** 2))
    norm_true = np.sqrt(np.mean(y_true ** 2))

    return norm_diff / (norm_true + eps)


def compute_rmse(y_pred, y_true):
    """
    Compute the Root Mean Square Error (RMSE) between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: RMSE
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute RMSE
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def compute_mae(y_pred, y_true):
    """
    Compute the Mean Absolute Error (MAE) between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: MAE
    """
    # Same as L1 error
    return compute_l1_error(y_pred, y_true)


def compute_mape(y_pred, y_true, eps=1e-8):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values
        eps (float): Small value to avoid division by zero

    Returns:
        float: MAPE
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute MAPE
    return 100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))


def compute_r2_score(y_pred, y_true):
    """
    Compute the coefficient of determination (R^2) between predicted and true values.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        float: R^2 score
    """
    # Convert to numpy if tensor
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Reshape if needed
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute R^2 score
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    if ss_tot == 0:
        return 0.0  # Avoid division by zero

    return 1 - (ss_res / ss_tot)




def compute_all_errors(y_pred, y_true):
    """
    Compute all error metrics.

    Args:
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        y_true (numpy.ndarray or torch.Tensor): True values

    Returns:
        dict: Dictionary of error metrics
    """
    return {
        'l2_error': compute_l2_error(y_pred, y_true),
        'l1_error': compute_l1_error(y_pred, y_true),
        'max_error': compute_max_error(y_pred, y_true),
        'relative_l2_error': compute_relative_l2_error(y_pred, y_true),
        'rmse': compute_rmse(y_pred, y_true),
        'mae': compute_mae(y_pred, y_true),
        'mape': compute_mape(y_pred, y_true),
        'r2_score': compute_r2_score(y_pred, y_true)
    }