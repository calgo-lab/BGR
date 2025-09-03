import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bgr.soil.utils import extract_segments


def soilnet_inference_mcd_depths(data_loader, model, device, num_inferences=100):
    """Run Monte Carlo Dropout inference on the depths"""

    model.to(device)
    model.eval()
    # Activate Monte Carlo Dropout in the depth predictor
    model.depth_marker_predictor.fc[3].__dict__['training'] = True
    
    all_pred_mean_depths, all_pred_std_depths = [], []
    for batch in data_loader:
        padded_images, image_mask, geotemp_features, _, _, _ = batch
        padded_images, image_mask, geotemp_features = (
            padded_images.to(device),
            image_mask.to(device),
            geotemp_features.to(device)
        )
        
        with torch.no_grad():
            
            all_pred_depths = []
            for _ in range(num_inferences):
                padded_pred_depths, _, _ = model(
                    padded_images,
                    image_mask,
                    geotemp_features[:, 1:], # 'index' column not used in model
                    #padded_true_depths,             # use predicted depths when not training 
                    #padded_segments_tabulars_labels # use predicted tabulars when not training
                )
            
                # Store predicted depths for all MCD inferences
                all_pred_depths.append(padded_pred_depths.cpu().numpy())

            # Store mean and std of MCD predictions
            all_pred_mean_depths.append(np.mean(all_pred_depths, axis=0))
            all_pred_std_depths.append(np.std(all_pred_depths, axis=0))
            
    return all_pred_mean_depths, all_pred_std_depths


def soilnet_inference(data_loader, model, device):
    """Run inference on Task 1 and 3"""
    
    model.to(device)
    model.eval()
    
    all_pred_depths, all_true_depths   = [], []
    all_pred_logits, all_true_hor_inds = [], []
    segment_geotemp_features = []
    for batch in data_loader:
        padded_images, image_mask, geotemp_features, padded_true_depths, padded_segments_tabulars_labels, padded_true_horizon_indices = batch
        padded_images, image_mask, geotemp_features, padded_true_depths, padded_segments_tabulars_labels, padded_true_horizon_indices = (
            padded_images.to(device),
            image_mask.to(device),
            geotemp_features.to(device),
            padded_true_depths.to(device),
            padded_segments_tabulars_labels.to(device),
            padded_true_horizon_indices.to(device)
        )
        
        with torch.no_grad():
            
            # Mask for valid indices
            mask = padded_true_horizon_indices != -1
            
            ### Predictions for all (sub)tasks
            padded_pred_depths, padded_pred_tabulars, padded_pred_logits = model(
                padded_images,
                image_mask,
                geotemp_features[:, 1:], # 'index' column not used in model
                #padded_true_depths,             # use predicted depths when not training 
                #padded_segments_tabulars_labels # use predicted tabulars when not training
            )
            
            # Store predicted horizon logits and true indices
            pred_logits = padded_pred_logits.view(-1, padded_pred_logits.size(-1))[mask.view(-1)]  # Apply mask
            all_pred_logits.append(pred_logits.cpu().numpy())
            
            true_horizon_indices = padded_true_horizon_indices.view(-1)[padded_true_horizon_indices.view(-1) != -1]
            all_true_hor_inds.append(true_horizon_indices.cpu().numpy())
            
            # Store predicted and true depths
            all_pred_depths.append(padded_pred_depths.cpu().numpy())
            all_true_depths.append(padded_true_depths.cpu().numpy())
            
            ### Store concatenated seg_geotemp features to train the residual predictor u(x) for the adaptive conformal depth regression
            # Crop image to segments
            segments = extract_segments(padded_images, image_mask, padded_pred_depths,
                                        model.segments_random_patches, model.patch_cnn_segment_size, 
                                        model.num_patches_per_segment, model.segment_random_patch_size, 
                                        model.stop_token, model.max_seq_len)
            if model.segments_random_patches:
                batch_size, num_segments, num_patches, C, H, W = segments.shape
            else:
                batch_size, num_segments, C, H, W = segments.shape
            # Encode each segment individually
            segment_features_list = []
            for i in range(num_segments):
                if model.segments_random_patches:
                    segment_patches = segments[:, i, :, :, :, :] # One additional dimension for the random patches
                    segment_features = model.segment_encoder(segment_patches)
                else:
                    segment = segments[:, i, :, :, :]
                    segment_features = model.segment_encoder(segment)                   
                segment_features_list.append(segment_features)
            segment_features = torch.stack(segment_features_list, dim=1)
            
            # Replicate geo_temp_features for each segment
            geo_temp_features = model.geo_temp_encoder(geotemp_features[:, 1:])
            geo_temp_features = geo_temp_features.unsqueeze(1).repeat(1, num_segments, 1)
            
            segment_geotemp_features.append( torch.cat([segment_features, geo_temp_features], dim=-1).cpu().numpy() )
            
    return segment_geotemp_features, all_pred_depths, all_true_depths, all_pred_logits, all_true_hor_inds


class ResidualPredictor(nn.Module):
    """Model to predict MSE error on the depths.
       Used for constructing adaptive confidence intervals on Task 1. 
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus() # ensure output is positive
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)
    

def trim_concat(true_depths, pred_depths, segment_geotemp_list):
    """# Trim true and predicted depths from trailing stop tokens and stack them"""

    trimmed_true_depths = []
    # Remove trailing 1.0's, keep one if present
    for sublist in true_depths.tolist():
        i = len(sublist) - 1
        while i > 0 and sublist[i] == 1.0 and sublist[i-1] == 1.0:
            i -= 1
        trimmed_true_depths.append(sublist[:i+1])
        
    # Remove stop tokens from predicted depths as well, inaccordance with the trimmed true depths
    trimmed_pred_depths = []
    for i in range(len(pred_depths)):
        trimmed_pred_depths.append(pred_depths[i][:len(trimmed_true_depths[i])].tolist())
        
    # Unroll the list of lists
    trimmed_true_depths_stacked = torch.tensor([item for sublist in trimmed_true_depths for item in sublist])
    trimmed_pred_depths_stacked = torch.tensor([item for sublist in trimmed_pred_depths for item in sublist])
    
    # Remove input paddings for the stop tokens
    trimmed_segment_geotemp_list = []
    for i in range(len(segment_geotemp_list)):
        trimmed_segment_geotemp_list.append(segment_geotemp_list[i][:len(trimmed_true_depths[i])].tolist())
        
    # Concatenate the (uneven) lists
    segment_geotemp_concat = torch.tensor(np.concatenate(trimmed_segment_geotemp_list, axis=0)).to(torch.float32)
    
    return trimmed_true_depths_stacked, trimmed_pred_depths_stacked, segment_geotemp_concat


def evaluate_coverage(prediction_sets, y_true, conf_levels):
    """Checks for different alpha levels whether the true label is covered by the prediction sets.

    Args:
        prediction_sets (dict): Dictionary of boolean prediction sets for different conf. levels
        y_true (array): True class indices
        conf_levels (list): Confidence levels

    Returns:
        list: Coverage rates for all conf. levels
    """
    
    empirical_coverages = []
    for cl in conf_levels:
        coverages = []
        # Check if the true label is in the predicted set
        for preds, y in zip(prediction_sets[cl], y_true):
            if preds[y]: coverages.append(1)
            else: coverages.append(0)

        empirical_coverages.append(np.mean(coverages))
    
    return empirical_coverages


def compute_calibration_curve_nonconformal(y_probs, y_true, n_bins=10):
    """Computes the calibration curve (confidences and accuraccies) for a nonconformal model, along with ECE and MAE."""
    
    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    mae, ece = 0.0, 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(mask):
            accs.append(np.mean(accuracies[mask]))
            confs.append(np.mean(confidences[mask]))

            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            bin_size = np.sum(mask) / len(y_true)
            
            mae += np.abs(bin_confidence - bin_accuracy) / n_bins
            ece += np.abs(bin_confidence - bin_accuracy) * bin_size
        else:
            accs.append(0.0)
            confs.append(0.0)

    return mae, ece, accs, confs


def split_pad(counts_until_stop, pred_depths_test_oracle_rand, max_seq_len):
    """Split and pad corrected predictions"""
    
    # Split corrected trimmed prediction into sublists according to counts_until_stop
    split_pred_depths_test_oracle_resid = []
    start = 0
    for count in counts_until_stop:
        split_pred_depths_test_oracle_resid.append(pred_depths_test_oracle_rand[start:start + count].tolist())
        start += count
        
    # Pad each sublist with 1.0 until max_seq_len
    padded_pred_depths_test_oracle_resid = []
    for sublist in split_pred_depths_test_oracle_resid:
        if len(sublist) < max_seq_len:
            padded = sublist + [1.0] * (max_seq_len - len(sublist))
        else:
            padded = sublist[:max_seq_len]
        padded_pred_depths_test_oracle_resid.append(padded)
        
    return padded_pred_depths_test_oracle_resid


def cumulate_set_sizes(predicted_lists):
    """Computes the cumulative histogram of the sizes in a list of predicted conformal lists.

    Args:
        predicted_lists (list): list of predicted lists of different sizes (may be strings or class indexes)
    """

    set_sizes = [len(pl) for pl in predicted_lists]
    counts, bin_edges = np.histogram(set_sizes, bins=range(0, max(set_sizes) + 1))
    cumulative_counts = np.cumsum(counts)
    normalized_cumulative_counts = cumulative_counts / cumulative_counts[-1]

    return normalized_cumulative_counts, bin_edges