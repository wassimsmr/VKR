import torch
import time
import numpy as np
from tqdm import tqdm
from Trainers.base import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised learning with data."""

    def __init__(self, model, x_train, y_train, x_val=None, y_val=None, **kwargs):
        """
        Initialize the supervised trainer.

        Args:
            model (torch.nn.Module): Neural network model
            x_train (torch.Tensor): Training input data
            y_train (torch.Tensor): Training target data
            x_val (torch.Tensor, optional): Validation input data
            y_val (torch.Tensor, optional): Validation target data
            **kwargs: Additional parameters for the base trainer
        """
        super(SupervisedTrainer, self).__init__(model, **kwargs)

        # Convert numpy arrays to torch tensors if necessary
        self.x_train = self._to_tensor(x_train)
        self.y_train = self._to_tensor(y_train)

        if x_val is not None and y_val is not None:
            self.x_val = self._to_tensor(x_val)
            self.y_val = self._to_tensor(y_val)
        else:
            self.x_val = None
            self.y_val = None

    def _to_tensor(self, data):
        """Convert data to torch tensor if it's not already."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=self.device)

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
        self.model.train()
        start_time = time.time()

        # Training loop
        for epoch in range(num_epochs):
            if batch_size is None:
                # Train on all data at once
                self.optimizer.zero_grad()
                outputs = self.model(self.x_train)
                loss = self.loss_fn(outputs, self.y_train)
                loss.backward()
                self.optimizer.step()

                self.train_losses.append(loss.item())

                # Validation
                if self.x_val is not None and self.y_val is not None:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)

                if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                    log = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}"
                    if self.x_val is not None:
                        log += f", Val Loss: {val_loss:.4f}"
                    print(log)

            else:
                # Mini-batch training
                num_samples = len(self.x_train)
                num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

                epoch_loss = 0.0

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)

                    batch_x = self.x_train[start_idx:end_idx]
                    batch_y = self.y_train[start_idx:end_idx]

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.loss_fn(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item() * (end_idx - start_idx)

                epoch_loss /= num_samples
                self.train_losses.append(epoch_loss)

                # Validation
                if self.x_val is not None and self.y_val is not None:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)

                if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                    log = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}"
                    if self.x_val is not None:
                        log += f", Val Loss: {val_loss:.4f}"
                    print(log)

        self.training_time = time.time() - start_time

        return self.get_metrics()

    def validate(self):
        """
        Validate the model.

        Returns:
            float: Validation loss
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.x_val)
            val_loss = self.loss_fn(outputs, self.y_val).item()
        self.model.train()
        return val_loss