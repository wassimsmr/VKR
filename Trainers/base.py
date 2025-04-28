import torch
import time
import numpy as np
from tqdm import tqdm


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(self, model, optimizer=None, lr=0.001, loss_fn=None, device='cpu'):
        """
        Initialize the base trainer.

        Args:
            model (torch.nn.Module): Neural network model
            optimizer (torch.optim.Optimizer, optional): Optimizer
            lr (float): Learning rate if optimizer is not provided
            loss_fn (callable, optional): Loss function
            device (str): Device to run the training on ('cpu' or 'cuda')
        """
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.device = device

        # Move model to device
        self.model = self.model.to(self.device)

        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []
        self.training_time = 0

    def train(self, num_epochs, batch_size=None, verbose=True):
        """
        Train the model.

        Args:
            num_epochs (int): Number of training epochs
            batch_size (int, optional): Batch size
            verbose (bool): Whether to display progress

        Returns:
            dict: Training metrics
        """
        # This is an abstract method - each subclass will implement the actual training logic
        # The implementation should track training time and return metrics
        raise NotImplementedError("Subclasses must implement train method")

    def validate(self, *args, **kwargs):
        """Validate the model."""
        raise NotImplementedError("Subclasses must implement validate method")

    def get_metrics(self):
        """Get training metrics."""
        return {
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'training_time': self.training_time
        }

    def get_config(self):
        """Get trainer configuration."""
        return {
            'type': self.__class__.__name__,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'model': self.model.get_config()
        }