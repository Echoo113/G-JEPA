import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ========= PatchDataset =========
class PatchDataset(Dataset):
    """
    用于加载 patch 数据 (x_patches, y_patches, label)，每个 patch 是一个样本
    输入文件需包含以下字段：
        - 'x_patches': (N, T_x, F)
        - 'y_patches': (N, T_y, F)
        - 'label': (N,)
    """
    def __init__(self, npz_file):
        super().__init__()
        data = np.load(npz_file)

        self.x = data['x_patches']   # shape: (N, T_x, F)
        self.y = data['y_patches']   # shape: (N, T_y, F)
        self.label = data['label']   # shape: (N,)

        assert len(self.x) == len(self.y) == len(self.label), "Patch数据维度不一致"
        print(f"[Dataset INIT] Loaded {npz_file}")
        print(f"  ➤ x: {self.x.shape}, y: {self.y.shape}, label: {self.label.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)       # (T_x, F)
        y = torch.tensor(self.y[idx], dtype=torch.float32)       # (T_y, F)
        label = torch.tensor(self.label[idx], dtype=torch.float32)  # scalar
        return x, y, label


# ========= Loader 工具函数 =========
def get_loader(npz_file, batch_size, shuffle=True):
    dataset = PatchDataset(npz_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"[Loader READY] batch_size={batch_size}, total_batches={len(loader)}")
    return loader



