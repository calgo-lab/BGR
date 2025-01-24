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

# ToDo: any better way that avoids the masks? e.g. just consider the tensor up to stop_token
def pad_tensor(true_depths, max_seq_len, stop_token, device='cpu'):
    """true_depths is a list with lists of variable lengths.
    we turn it into a tensor of padded tensors."""
    padded_targets, masks = [], []

    for depths in true_depths:
        depth_len = len(depths)
        if depth_len > max_seq_len:
            depths = depths[:max_seq_len]
            depth_len = max_seq_len
        mask = [1] * depth_len + [0] * (max_seq_len - depth_len)
        depths = depths + [stop_token] * (max_seq_len - depth_len)
        padded_targets.append(torch.tensor(depths, device=device))
        masks.append(torch.tensor(mask, device=device))

    return torch.stack(padded_targets), torch.stack(masks)