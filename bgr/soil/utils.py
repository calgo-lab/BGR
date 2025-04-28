import torch
import torch.nn.functional as F
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

def reflect_pad_large(x: torch.Tensor, pad: tuple):
    """
    Reflect-pad tensor x by pad = (left, right, top, bottom),
    allowing pad sizes > (dim_size - 1) by applying padding iteratively
    in chunks compatible with F.pad(mode='reflect').

    Args:
        x (torch.Tensor): Input tensor, expected shape (..., H, W).
        pad (tuple): A tuple of 4 integers (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: The padded tensor.

    Raises:
        ValueError: If attempting to reflect pad a dimension with size <= 1.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    out = x
    # Assuming input tensor shape is (..., H, W)
    h_dim = -2  # Height dimension index
    w_dim = -1  # Width dimension index

    # --- Pad Height (Top, Bottom) ---
    # Pad Top
    rem = pad_top
    while rem > 0:
        current_h = out.shape[h_dim]
        if current_h <= 1:
            raise ValueError(f"Cannot apply reflection padding to dimension {h_dim} with size {current_h}")
        # Calculate how much to pad in this step: minimum of remaining padding
        # and the maximum allowed by F.pad (current_dim_size - 1)
        this = min(rem, current_h - 1)
        if this <= 0: # Should not happen if amount > 0 and current_h > 1, but safety check
             break
        # Create padding tuple (left, right, top, bottom) - only pad top
        pad_tuple = (0, 0, this, 0)
        out = F.pad(out, pad_tuple, mode='reflect')
        rem -= this

    # Pad Bottom
    rem = pad_bottom
    while rem > 0:
        current_h = out.shape[h_dim]
        if current_h <= 1:
            raise ValueError(f"Cannot apply reflection padding to dimension {h_dim} with size {current_h}")
        this = min(rem, current_h - 1)
        if this <= 0:
             break
        # Create padding tuple - only pad bottom
        pad_tuple = (0, 0, 0, this)
        out = F.pad(out, pad_tuple, mode='reflect')
        rem -= this

    # --- Pad Width (Left, Right) ---
    # Pad Left
    rem = pad_left
    while rem > 0:
        current_w = out.shape[w_dim]
        if current_w <= 1:
            raise ValueError(f"Cannot apply reflection padding to dimension {w_dim} with size {current_w}")
        this = min(rem, current_w - 1)
        if this <= 0:
             break
        # Create padding tuple - only pad left
        pad_tuple = (this, 0, 0, 0)
        out = F.pad(out, pad_tuple, mode='reflect')
        rem -= this

    # Pad Right
    rem = pad_right
    while rem > 0:
        current_w = out.shape[w_dim]
        if current_w <= 1:
            raise ValueError(f"Cannot apply reflection padding to dimension {w_dim} with size {current_w}")
        this = min(rem, current_w - 1)
        if this <= 0:
             break
        # Create padding tuple - only pad right
        pad_tuple = (0, this, 0, 0)
        out = F.pad(out, pad_tuple, mode='reflect')
        rem -= this

    return out

def tensor_random_crop_reflect(segment: torch.Tensor, ph: int):
    """
    Mimics transforms.RandomCrop(ph, pad_if_needed=True, padding_mode='reflect'),
    but supports padding sizes >> segment dims by using reflect_pad_large,
    performing operations entirely on the tensor's device.

    Args:
        segment (torch.Tensor): Input tensor, expected shape (C, H, W).
        ph (int): The target height and width of the random crop.

    Returns:
        torch.Tensor: A ph x ph random crop from the (potentially padded) segment.
    """
    if not isinstance(segment, torch.Tensor):
        raise TypeError("Input 'segment' must be a torch.Tensor")
    if segment.ndim < 3:
         raise ValueError("Input 'segment' must have at least 3 dimensions (..., H, W)")

    # Use last two dimensions for H, W assuming shape (..., H, W)
    H, W = segment.shape[-2], segment.shape[-1]
    device = segment.device

    # 1) Compute how much padding we *need*
    pad_h = max(ph - H, 0)
    pad_w = max(ph - W, 0)

    padded_segment = segment # Start with original segment

    if pad_h > 0 or pad_w > 0:
        # Calculate padding amounts for each side
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        # 2) Apply the potentially large padding reflectively using the corrected helper
        padding_tuple = (pad_left, pad_right, pad_top, pad_bottom)
        padded_segment = reflect_pad_large(segment, padding_tuple)

        # Update H and W to reflect the dimensions *after* padding
        H = padded_segment.shape[-2]
        W = padded_segment.shape[-1]

    # Ensure crop size is not larger than the (potentially padded) dimensions
    if H < ph or W < ph:
        # This should ideally not happen if padding logic is correct,
        # but adding a safeguard. Could also happen if initial H/W was 0.
        raise ValueError(f"Padded dimensions ({H}, {W}) are still smaller than crop size ({ph})")


    # 3) Uniformly sample a top-left corner (y, x) for the crop window
    # The range for randint is [low, high), so high should be H - ph + 1
    y = torch.randint(0, H - ph + 1, (), device=device)
    x = torch.randint(0, W - ph + 1, (), device=device)

    # 4) Extract the crop
    # Use slicing that works for tensors with leading dimensions (e.g., batch or channel)
    crop = padded_segment[..., y : y + ph, x : x + ph]

    return crop

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
