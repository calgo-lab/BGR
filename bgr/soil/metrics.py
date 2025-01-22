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

    def __init__(self, stop_token=100):
        super(DepthMarkerLoss, self).__init__()
        self.stop_token = stop_token

    def forward(self, predictions, targets):
        """
        Computes the loss for the DepthMarkerPredictor.

        Args:
            predictions: List of predicted depth markers. Contains batch_size lists of variable lengths.
            targets: List of true depth markers. Contains batch_size lists of variable lengths.
            stop_token: Value of the stop token.

        Returns:
            loss: Scalar loss value.
        """
        batch_size = len(targets)
        loss = 0.0

        for batch_idx in range(batch_size):
            true_depths = torch.Tensor(targets[batch_idx])
            num_true_steps = len(true_depths)

            pred_depths = torch.Tensor(predictions[batch_idx])
            pred_len = len(pred_depths) # can be higher or lower than it's supposed to be

            valid_steps = min(num_true_steps, pred_len)

            # MSE Loss for valid steps
            loss += torch.norm(pred_depths[:valid_steps] - true_depths[:valid_steps], p=2) ** 2

            # Stop Token Penalty
            #if valid_steps < pred_len:
            #    loss += (pred_depths[valid_steps] - stop_token) ** 2
            #    total_valid_steps += 1  # Count the stop token penalty as an extra step

        # Average loss over all valid steps
        loss /= batch_size
        return loss
