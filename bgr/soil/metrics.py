import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

class TopKHorizonAccuracy(nn.Module):
    def __init__(self, label_embeddings, k=5):
        """
        Args:
            label_embeddings (torch.Tensor): A fixed matrix of shape (num_labels, embedding_dim) containing the true label embeddings.
            k (int): The number of nearest neighbors to consider.
        """
        super(TopKHorizonAccuracy, self).__init__()
        self.label_embeddings = label_embeddings #nn.Parameter(label_embeddings, requires_grad=False)  # Store as a fixed tensor
        self.k = k

    def forward(self, predicted_embeddings, true_labels):
        """
        Args:
            predicted_embeddings (torch.Tensor): The model's predicted embeddings (batch_size, embedding_dim).
            true_labels (torch.Tensor): The true label indices corresponding to each row in `label_embeddings` (batch_size,).

        Returns:
            accuracy (float): The Top-K accuracy over the batch.
        """
        # Compute cosine similarity between predicted embeddings and all label embeddings
        similarity = torch.matmul(predicted_embeddings, self.label_embeddings.T)  # (batch_size, num_labels)

        # Get indices of top-k nearest embeddings for each prediction
        top_k_indices = torch.topk(similarity, self.k, dim=1).indices  # (batch_size, k)

        # Check if the true label index is in the top-k predictions
        correct = (top_k_indices == true_labels.unsqueeze(1)).any(dim=1)  # (batch_size,)

        # Compute the accuracy
        accuracy = correct.float().mean().item()
        return accuracy


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


class ShortestPathLoss(nn.Module):
    """Computes a top-k shortest path loss based on the number of edges needed to traverse in the graph to reach the least common ancestor of two terminal nodes."""
    def __init__(self, path_lengths_dict, k=5):
        """
        Args:
            path_lengths_dict (dict): A dictionary where keys are tuples of node index pairs and values are the lengths of the paths between them.
            k (int): The number of top predictions to consider.
        """
        super(ShortestPathLoss, self).__init__()
        self.path_lengths_dict = path_lengths_dict
        self.k = k

    def forward(self, predicted_logits, true_labels):
        """
        Args:
            predicted_logits (torch.Tensor): The predicted logits (batch_size, num_classes).
            true_labels (torch.Tensor): The true labels as integers (batch_size,).

        Returns:
            loss (float): The average shortest path length over the top-k predictions.
        """
        # Get the top-k predicted indices
        #top_k_indices = torch.topk(predicted_logits, self.k, dim=1).indices  # (batch_size, k)
        top_k_indices = torch.topk(predicted_logits, predicted_logits.size(1), dim=1).indices

        # Initialize total path length as a tensor with requires_grad=True
        total_path_length = torch.zeros(1, device=predicted_logits.device, requires_grad=True)

        weights = torch.tensor(
            [1.0 / (i + 1) for i in range(predicted_logits.size(1))],
            device=predicted_logits.device,
            dtype=torch.float
        )

        # Iterate through true labels and top-k predictions
        for true, top_k in zip(true_labels, top_k_indices):
            # Compute the shortest path length for each of the top-k predictions
            path_lengths = torch.stack([
                torch.tensor(
                    self.path_lengths_dict.get((true.item(), pred.item())) or 
                    self.path_lengths_dict.get((pred.item(), true.item())),
                    device=predicted_logits.device,
                    dtype=torch.float
                )
                for pred in top_k
            ])
            
            # Use the weighted mean of path lengths among the top-k predictions
            total_path_length = total_path_length + torch.sum(path_lengths * weights)

        # Compute the average path length over the batch
        loss = total_path_length / predicted_logits.size(0)
        return loss
    

def precision_recall_at_k(true_labels, topk_predictions, all_labels, average='macro'):
    """
    Computes Precision@K and Recall@K for multi-class classification using logits.

    Args:
        logits (torch.Tensor): The model's predicted logits (batch_size, num_classes).
        true_labels (torch.Tensor): The true labels (batch_size,).
        k (int): The number of top predictions to consider.

    Returns:
        precision_at_k (float): Precision@K over the batch.
        recall_at_k (float): Recall@K over the batch.
    """

    # Initialize predicted_labels with the top-1 prediction (i.e. first column)
    predicted_labels = topk_predictions[:, 0].clone()  # (batch_size,)
    
    # Check for each sample if the true label is among the top-K predictions
    relevant = (topk_predictions == true_labels.unsqueeze(1))  # (batch_size, k)
    hit = relevant.any(dim=1)  # (batch_size,) Boolean: True if true label is in top-K
    
    # For samples with a "hit", replace the predicted label with the true label
    predicted_labels[hit] = true_labels[hit]
    
    # Convert tensors to numpy arrays for sklearn functions
    y_pred = predicted_labels.cpu().numpy()
    y_true = true_labels.cpu().numpy()
    
    # Compute precision and recall using sklearn with the desired averaging
    precision_at_k = precision_score(y_true, y_pred, average=average, labels=all_labels, zero_division=0)
    recall_at_k = recall_score(y_true, y_pred, average=average, labels=all_labels, zero_division=0)
    
    return precision_at_k, recall_at_k


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


def depth_iou(preds: torch.Tensor, targets: torch.Tensor, stop_token=1.0):
    """
    Computes a 1D IoU metric for depth markers based on segment overlap.

    Args:
        preds (torch.Tensor): Predicted depth markers (batch_size, max_seq_len)
        targets (torch.Tensor): Ground truth depth markers (batch_size, max_seq_len)
        stop_token (float): Value indicating the stop token (default: 1.0)

    Returns:
        float: Average IoU score over the batch
    """
    batch_size = preds.shape[0]
    iou_scores = torch.zeros(batch_size, device=preds.device)  # Store per-sample IoUs

    for i in range(batch_size):
        true_depths = targets[i]
        pred_depths = preds[i]

        # Find stop token index and clip sequences
        stop_idx = (true_depths == stop_token).nonzero(as_tuple=True)[0]
        if len(stop_idx) == 0:
            continue  # No valid depths

        stop_idx = stop_idx[0].item()
        true_depths = true_depths[: stop_idx + 1]
        num_true_segments = len(true_depths)
        pred_depths = pred_depths[:num_true_segments]

        # Add ground level (0.0) as the first depth
        true_depths = torch.cat((torch.tensor([0.0], device=targets.device), true_depths))
        pred_depths = torch.cat((torch.tensor([0.0], device=preds.device), pred_depths))

        # Ensure we have at least one valid segment
        if num_true_segments < 1:
            continue

        # Stack depth pairs for efficient processing
        pred_pairs = torch.stack((pred_depths[:-1], pred_depths[1:]), dim=1)
        true_pairs = torch.stack((true_depths[:-1], true_depths[1:]), dim=1)

        # Compute disjoint mask - check if the intervals do not overlap
        # Two segments [a, b] and [c, d] don't overlap if b <= c or d <= a (for a <= b and c <= d)
        disjoint_mask = (pred_pairs[:, 1] <= true_pairs[:, 0]) | (true_pairs[:, 1] <= pred_pairs[:, 0])

        # Concatenate and sort
        all_pairs = torch.cat((pred_pairs, true_pairs), dim=1)  # Shape: (num_segments, 4)
        sorted_depths, _ = torch.sort(all_pairs, dim=1)  # Sort along dim=1

        # Compute intersection and union
        intersections = sorted_depths[:, 2] - sorted_depths[:, 1]  # Middle two
        unions = sorted_depths[:, 3] - sorted_depths[:, 0]  # Outermost

        # Set intersection to 0 where intervals are disjoint
        intersections[disjoint_mask] = 0.0

        # Compute IoU, handling zero division safely
        ious = torch.where(unions > 0, intersections / unions, torch.zeros_like(unions))

        # Store the mean IoU for this sample
        iou_scores[i] = ious.mean()

    #return iou_scores.mean().item()  # Average over the batch
    return iou_scores.mean() # Average over the batch and keep tensor attached to computation graph (requires_grad=True)


class DepthIoULoss(nn.Module):
    """1D Depth IoU Loss"""

    def __init__(self, stop_token=1.0):
        super(DepthIoULoss, self).__init__()
        self.stop_token = stop_token

    def forward(self, predictions, targets):
        """

        :param predictions:
        :param targets:
        :return:
        """
        return torch.tensor(1.0, device=predictions.device) - depth_iou(predictions, targets, self.stop_token)