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


def depth_marker_loss(pred_seq, target_seq, mask):
    # Smooth L1 Loss or MSE for regression
    loss = F.smooth_l1_loss(pred_seq * mask, target_seq * mask, reduction="sum")
    loss /= mask.sum()  # Normalize by the number of valid elements
    return loss


def depth_tabular_loss(pred_depth, true_depth, pred_tabular, true_tabular):
    """

    :param pred_depth:
    :param true_depth:
    :param pred_tabular:
    :param true_tabular:
    :return:
    """
    depth_loss = F.smooth_l1_loss(pred_depth, true_depth)
    tabular_loss = F.mse_loss(pred_tabular, true_tabular)
    return depth_loss + tabular_loss