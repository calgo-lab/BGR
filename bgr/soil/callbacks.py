import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored metric to qualify as improvement.
            verbose (bool): Whether to print early stopping messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.should_stop = True


class ModelCheckpoint:
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

    def __call__(self, model, metric_value):
        """
        Checks if the model should be saved based on the monitored metric.

        Args:
            model (nn.Module): The PyTorch model to save.
            metric_value (float): The current value of the monitored metric.
        """
        if self.compare(metric_value, self.best_metric):
            self.best_metric = metric_value
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Model checkpoint saved at '{self.save_path}' with {self.monitor}: {metric_value:.4f}")

