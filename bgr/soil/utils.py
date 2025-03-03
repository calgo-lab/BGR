import torch

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
