import torch
import warnings

def check_index_duplicates(loader_tqdm):
    """Check whether there are duplicates in a data loader w.r.t. 'index' column."""
    seen = set()
    duplicates = []

    for _, batch in loader_tqdm:
        for idx in batch[:, 0]:  # 'index' is stored at the first position
            if idx in seen:
                duplicates.append(idx)
            else:
                seen.add(idx)

    print(f"Duplicate indices: {duplicates}")


def pad_tensor(true_depths, max_seq_len, stop_token, device='cpu'):
    """true_depths is a list with lists of variable lengths.
    we turn it into a tensor of padded tensors."""

    padded_targets = []
    for depths in true_depths:
        depth_len = len(depths)
        if depth_len > max_seq_len:
            depths = depths[:max_seq_len]
            depth_len = max_seq_len
        depths = depths + [stop_token] * (max_seq_len - depth_len)
        padded_targets.append(torch.tensor(depths, device=device))

    return torch.stack(padded_targets)

def unpad_image_using_mask(image_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Removes padding from an image tensor by cropping to the bounding box
        defined by the True values in a boolean mask.

        Args:
            image_padded (torch.Tensor): The padded image tensor (C, H_pad, W_pad).
            mask (torch.Tensor): The boolean mask tensor, assumed to be broadcastable
                            to (H_pad, W_pad). Typically (1, H_pad, W_pad) or (H_pad, W_pad).
                            True indicates valid pixels, forming a rectangle.

        Returns:
            torch.Tensor: The unpadded image tensor (C, H_orig, W_orig).
                        Returns a tensor with size (C, 0, 0) if the mask is all False.
        """
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask_2d = mask.squeeze(0) # Remove channel dim -> (H_pad, W_pad)
        elif mask.dim() == 2:
            mask_2d = mask # Already (H_pad, W_pad)
        else:
            raise ValueError(f"Mask must be broadcastable to (H_pad, W_pad), got shape {mask.shape}")

        # Check if there are any True values in the mask
        if not torch.any(mask_2d):
            warnings.warn("Attempting to unpad using an all-False mask. Returning empty tensor.", UserWarning)
            C = image_padded.shape[0]
            return torch.empty((C, 0, 0), device=image_padded.device, dtype=image_padded.dtype)

        # Find rows and columns that contain at least one True value
        rows_with_true = torch.any(mask_2d, dim=1)
        cols_with_true = torch.any(mask_2d, dim=0)

        # Find the first and last indices (bounding box)
        row_indices = rows_with_true.nonzero(as_tuple=True)[0]
        col_indices = cols_with_true.nonzero(as_tuple=True)[0]

        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]

        # Crop the image using the bounding box indices
        # Add 1 to max indices because Python slicing is exclusive at the end
        image_unpadded = image_padded[:, min_row:(max_row + 1), min_col:(max_col + 1)]

        return image_unpadded


def split_depth_markers(depth_markers, stop_token=1.0):
    """
    Splits a list of depth markers into sublists of [upper, lower] bounds, stopping at the first occurrence of the stop_token.

    Args:
        depth_markers (list or tensor): List of depth markers.
        stop_token (float): Value indicating the stop point.

    Returns:
        List of tuples: [(upper1, lower1), (upper2, lower2), ...].
    """
    # Convert to a list if it's a tensor
    if isinstance(depth_markers, torch.Tensor):
        depth_markers = depth_markers.tolist()

    # Add 0.0 as the initial upper bound
    depth_markers = [0.0] + depth_markers

    # Find the first occurrence of the stop_token and truncate
    if stop_token in depth_markers:
        stop_idx = depth_markers.index(stop_token)
        depth_markers = depth_markers[:stop_idx + 1]

    # Create pairs of (upper, lower) bounds
    return [(depth_markers[i], depth_markers[i + 1]) for i in range(len(depth_markers) - 1)]


def concat_img_geotemp_depth(img_geotemp_vector, depth_markers, stop_token=1.0):
    """
    Prepares inputs for each horizon by concatenating the image-geotemp vector with depth marker bounds.

    Args:
        img_geotemp_vector (Tensor): Tensor of shape (batch_size, feature_dim).
        depth_markers (Tensor): Tensor of shape (batch_size, total_horizons).

    Returns:
        List of Tensors: Horizon-specific inputs of shape (batch_size, total_horizons, feature_dim + 2).
    """
    batch_size = img_geotemp_vector.size(0)
    tab_inputs = []

    for i in range(batch_size):
        bounds = split_depth_markers(depth_markers[i], stop_token)
        for upper, lower in bounds:
            bounds_tensor = torch.tensor([upper, lower], device=img_geotemp_vector.device)
            tab_input = torch.cat([img_geotemp_vector[i], bounds_tensor])
            tab_inputs.append(tab_input)

    return torch.stack(tab_inputs, dim=0)  # Shape: (total_horizons, img_geotemp_dim + 2)
