import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

# ===================================================================
# ========= 你原有的代码 (保持不变) =========
# ===================================================================

class PatchDataset(Dataset):
    """
    用于加载【不带标签】的 .npz 文件 (包含 'x_patches', 'y_patches')。
    """
    def __init__(self, x_patches: np.ndarray, y_patches: np.ndarray):
        assert x_patches.shape == y_patches.shape, "x 和 y patch 的维度必须一致"
        self.x_patches = x_patches
        self.y_patches = y_patches

    def __len__(self):
        return len(self.x_patches)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_patches[idx], dtype=torch.float)
        y = torch.tensor(self.y_patches[idx], dtype=torch.float)
        return x, y

def create_patch_loader(npz_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    创建【不带标签】数据的 DataLoader。
    """
    npz = np.load(npz_path)
    x_patches = npz["x_patches"]
    y_patches = npz["y_patches"]
    dataset = PatchDataset(x_patches, y_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# ===================================================================
# ========= 新增与修改：用于加载带标签数据的 Helper 函数 =========
# ===================================================================



class LabeledPatchDataset(Dataset):
    """
    用于加载【带标签】的 .npz 文件。
    支持两种模式：
    1. 单文件模式：一个 .npz 文件包含 'x_patches', 'y_patches', 'labels'
    2. 双文件模式：一个 .npz 文件包含特征 ('x_patches', 'y_patches')，另一个包含标签 ('x_labels', 'y_labels')
    """
    def __init__(self, feature_data: dict, label_data: dict = None):
        """
        Args:
            feature_data: 包含 'x_patches' 和 'y_patches' 的字典
            label_data: 包含 'x_labels' 和 'y_labels' 的字典（可选）
        """
        self.x_patches = feature_data['x_patches']
        self.y_patches = feature_data['y_patches']
        
        # 处理标签数据
        if label_data is not None:
            # 双文件模式
            self.x_labels = label_data['x_labels']
            self.y_labels = label_data['y_labels']
        else:
            # 单文件模式
            self.x_labels = feature_data.get('x_labels')
            self.y_labels = feature_data.get('y_labels')
        
        # 验证数据一致性
        assert len(self.x_patches) == len(self.y_patches), "特征数据长度不匹配"
        if self.x_labels is not None:
            assert len(self.x_patches) == len(self.x_labels), "特征和标签长度不匹配"
            assert len(self.y_patches) == len(self.y_labels), "特征和标签长度不匹配"

    def __len__(self):
        return len(self.x_patches)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_patches[idx], dtype=torch.float)
        y = torch.tensor(self.y_patches[idx], dtype=torch.float)
        
        if self.x_labels is not None:
            x_label = torch.tensor(self.x_labels[idx], dtype=torch.long)
            y_label = torch.tensor(self.y_labels[idx], dtype=torch.long)
            return x, y, x_label, y_label
        return x, y

def create_labeled_loader(
    feature_npz_path: str,
    label_npz_path: str = None,
    batch_size: int = 64,
    shuffle: bool = True
) -> DataLoader:
    """
    创建【带标签】数据的 DataLoader。
    
    Args:
        feature_npz_path: 特征数据文件路径（包含 'x_patches', 'y_patches'）
        label_npz_path: 标签数据文件路径（包含 'x_labels', 'y_labels'），如果为None则从特征文件中读取标签
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader: 返回 (x, y, x_label, y_label) 或 (x, y) 的 DataLoader
    """
    feature_data = np.load(feature_npz_path)
    
    if label_npz_path is not None:
        label_data = np.load(label_npz_path)
    else:
        label_data = None
    
    dataset = LabeledPatchDataset(feature_data, label_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader