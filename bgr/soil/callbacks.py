import torch
from abc import ABC, abstractmethod

class Callback(ABC):
    @abstractmethod
    def __call__(self, model, metrics: dict):
        pass

class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0.0001, verbose=False, monitor="val_loss", mode="min"):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored metric to qualify as improvement.
            verbose (bool): Whether to print early stopping messages.
            monitor (str): Metric to monitor (e.g., "val_loss", "val_accuracy").
            mode (str): "min" to stop when the metric decreases, "max" to stop when it increases.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.monitor = monitor
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.should_stop = False

        # Initialize comparison function
        if mode == "min":
            self.compare = lambda current, best: current < best - self.min_delta
            self.best_metric = float("inf")
        elif mode == "max":
            self.compare = lambda current, best: current > best + self.min_delta
            self.best_metric = float("-inf")
        else:
            raise ValueError("mode should be either 'min' or 'max'")

    def __call__(self, model, metrics: dict):
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            raise ValueError(f"EarlyStopping requires '{self.monitor}' in metrics")
        
        if self.compare(metric_value, self.best_metric):
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.should_stop = True

class ModelCheckpoint(Callback):
    def __init__(self, save_path, monitor="val_loss", mode="min", verbose=True):
        """
        Args:
            save_path (str): Path to save the model checkpoint.
            monitor (str): Metric to monitor (e.g., "val_loss", "val_accuracy").
            mode (str): "min" to save when the metric decreases, "max" to save when it increases.
            verbose (bool): If True, prints messages when a new checkpoint is saved.
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_metric = None

        # Initialize comparison function
        if mode == "min":
            self.compare = lambda current, best: current < best
            self.best_metric = float("inf")
        elif mode == "max":
            self.compare = lambda current, best: current > best
            self.best_metric = float("-inf")
        else:
            raise ValueError("mode should be either 'min' or 'max'")

    def __call__(self, model, metrics: dict):
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            raise ValueError(f"ModelCheckpoint requires '{self.monitor}' in metrics")
        
        if self.compare(metric_value, self.best_metric):
            self.best_metric = metric_value
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Model checkpoint saved at '{self.save_path}' with {self.monitor}: {metric_value:.4f}")

