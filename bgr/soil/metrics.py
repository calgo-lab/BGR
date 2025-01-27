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


class DepthMarkerLoss(nn.Module):
    """MSE for padded tensors"""

    def __init__(self, lambda_mono=1.0, lambda_div=1.0):
        super(DepthMarkerLoss, self).__init__()
        self.lambda_mono = lambda_mono
        self.lambda_div = lambda_div

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, max_seq_len), predicted depth markers.
            targets: Tensor of shape (batch_size, max_seq_len), true depth markers.
        """
        # MSE for valid steps (up until the stop token)
        mse_loss = nn.functional.mse_loss(predictions, targets, reduction='mean')

        # Monotonicity term: the more increasing neighbors, the closer this term gets to 0
        mono_term = torch.relu(predictions[:, :-1] - predictions[:, 1:]).mean()

        # Diversity Loss (invert variance of predictions)
        div_term = -torch.var(predictions, dim=1).mean()

        total_loss = mse_loss + self.lambda_mono * mono_term + self.lambda_div * div_term

        return total_loss
