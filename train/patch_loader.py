# patch_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    """
    PatchDataset: 用于加载已保存的 Patch 数据 (x_patches, y_patches)
    
    每个样本形状：
    - x_patches[i]: (11, 16, 137) → 表示一个时间窗口内的 11 个 context patch，每个 patch 是 16x137 的时间段
    - y_patches[i]: (11, 16, 137) → 表示未来的 11 个 patch，目标输出

    举个例子：
    - x_patches.shape = (873, 11, 16, 137)
    - 意味着总共有 873 个训练样本
    - 每个训练样本中，有 11 个 patch（横向 patch 数），每个 patch 是 16 时间步 x 137 变量
    """
    def __init__(self, x_patches: np.ndarray, y_patches: np.ndarray):
        assert x_patches.shape == y_patches.shape, "x 和 y patch 的维度必须一致"
        self.x_patches = x_patches
        self.y_patches = y_patches

    def __len__(self):
        return len(self.x_patches)

    def __getitem__(self, idx):
        # 返回单个样本：一个输入 patch 组、一个目标 patch 组
        x = torch.tensor(self.x_patches[idx], dtype=torch.float)  # (11, 16, 137)
        y = torch.tensor(self.y_patches[idx], dtype=torch.float)  # (11, 16, 137)
        return x, y


def create_patch_loader(npz_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    封装创建 DataLoader 的函数，方便你快速加载 patch 数据进行训练

    参数说明：
    - npz_path: Patch 文件路径，比如 "data/SOLAR/patches/solar_train.npz"
    - batch_size: 每个 mini-batch 抽取的样本数（不是 patch 数！）
    - shuffle: 是否打乱样本顺序，训练集一般设 True，验证/测试设 False

    返回：
    - 一个 PyTorch 的 DataLoader，输出：(batch, 11, 16, 137)
    """
    npz = np.load(npz_path)
    x_patches = npz["x_patches"]
    y_patches = npz["y_patches"]

    dataset = PatchDataset(x_patches, y_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
