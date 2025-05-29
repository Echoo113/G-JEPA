import torch
import numpy as np

# Global batch size setting
BATCH_SIZE = 4

def split_into_batches(patches: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Args:
        patches: Tensor, shape = (total_patch, T, F)
        batch_size: 需要的 batch size（例如 4）
    
    Returns:
        Tensor, shape = (B, N, T, F)，如果不足整除则截断多余 patch
    """
    total_patches, T, F = patches.shape
    assert total_patches >= batch_size, "总 patch 数不能小于 batch size"

    # 每个样本应该有的 patch 数 N
    N = total_patches // batch_size

    usable_patch_count = batch_size * N
    batched = patches[:usable_patch_count].view(batch_size, N, T, F)
    return batched

def prepare_batch_from_np(np_array: np.ndarray, batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    从 numpy array → Tensor，并按 batch_size 拼接
    
    Args:
        np_array: numpy array, shape = (total_patch, T, F)
        batch_size: 需要的 batch size，默认使用全局 BATCH_SIZE
    
    Returns:
        Tensor, shape = (B, N, T, F)
    """
    tensor = torch.tensor(np_array, dtype=torch.float)
    return split_into_batches(tensor, batch_size) 