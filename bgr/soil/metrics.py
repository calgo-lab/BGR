import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_accuracy(outputs, labels, k=5):
    """

    :param outputs:
    :param labels:
    :param k:
    :return:
    """

    # Get the top-k predictions
    _, top_k_predictions = outputs.topk(k, dim=1)
    # Check if the true label is in the top-k predictions
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    # Sum up the correct matches and return the mean accuracy
    return correct.any(dim=1).float().mean().item()


class TopKLoss(nn.Module):
    """

    """
    def __init__(self, k=5):
        super(TopKLoss, self).__init__()
        self.k = k

    def forward(self, outputs, targets):
        # Compute the top-k logits
        top_k_logits, top_k_indices = outputs.topk(self.k, dim=1)
        # Gather the target probabilities
        top_k_probabilities = F.log_softmax(top_k_logits, dim=1)
        # Create a mask for the correct class
        target_mask = top_k_indices.eq(targets.view(-1, 1))
        # Compute loss only for the top-k classes
        loss = -top_k_probabilities[target_mask].mean()
        return loss


def depth_marker_loss(predictions, targets, stop_token=100):
    """
    Computes the loss for the DepthMarkerPredictor.

    Args:
        predictions: Tensor of shape (pred_len, batch_size), predicted depth markers.
        targets: List of variable-length tensors (batch_size), true depth markers.
        stop_token: Value of the stop token.

    Returns:
        loss: Scalar loss value.
    """
    batch_size = len(targets)
    pred_len = predictions.size(0)

    loss = 0.0
    total_valid_steps = 0

    for batch_idx in range(batch_size):
        true_depths = targets[batch_idx]
        num_true_steps = len(true_depths)

        # Mask for valid steps
        valid_steps = min(num_true_steps, pred_len)
        mask = torch.arange(pred_len) < valid_steps
        # example: torch.arange(2) < 3 is [True, True]
        # while    torch.arange(3) < 2 is [True, True, False]

        # MSE Loss for valid steps
        true_values = torch.zeros(pred_len, device=predictions.device)
        true_values[:valid_steps] = true_depths
        loss += torch.sum((predictions[:, batch_idx] - true_values) ** 2 * mask.float())

        # Stop Token Penalty
        if valid_steps < pred_len:
            loss += (predictions[valid_steps, batch_idx] - stop_token) ** 2
            total_valid_steps += 1  # Count the stop token penalty as an extra step

        total_valid_steps += valid_steps

    # Average loss over all valid steps
    loss /= total_valid_steps
    return loss
