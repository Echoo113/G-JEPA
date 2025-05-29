import torch
import torch.nn as nn
import math
import numpy as np

def split_into_batches(patches: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Args:
        patches: Tensor, shape = (total_patch, T, F)
        batch_size: 需要的 batch size
    
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

def prepare_batch_from_np(np_array: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    从 numpy array → Tensor，并按 batch_size 拼接
    
    Args:
        np_array: numpy array, shape = (total_patch, T, F)
        batch_size: 需要的 batch size
    
    Returns:
        Tensor, shape = (B, N, T, F)
    """
    tensor = torch.tensor(np_array, dtype=torch.float)
    return split_into_batches(tensor, batch_size)

class PositionalEncoding(nn.Module):
    """
    动态生成的正弦-余弦位置编码（不使用固定 max_len）
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (B, N, D)
        Returns:
            Tensor, same shape, 加上位置编码
        """
        B, N, D = x.shape
        device = x.device

        position = torch.arange(N, dtype=torch.float, device=device).unsqueeze(1)  # (N, 1)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D)
        )

        pe = torch.zeros(N, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # → (1, N, D)
        return x + pe  # 自动 broadcast 到 (B, N, D)

class MyTimeSeriesEncoder(nn.Module):
    """
    仿 Huggingface TimeSeriesTransformer 的简化版 Encoder
    支持多变量、多 patch 输入
    """
    def __init__(
        self,
        patch_length: int = 30,
        num_vars: int = 137,
        latent_dim: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # 1) Patch embedding: 将 (T * F) 映射到 latent_dim
        self.patch_embedding = nn.Linear(patch_length * num_vars, latent_dim)

        # 2) 动态位置编码层（不使用 max_len）
        self.pos_encoding = PositionalEncoding(latent_dim)

        # 3) Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,  # 注意：我们中间会 transpose
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) 最后 LayerNorm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (B, N_ctx, T, F)
        Returns:
            Tensor, shape = (B, N_ctx, latent_dim)
        """
        B, N, T, F = x.shape

        # 1) 拉平成 (B, N, T*F)
        x = x.view(B, N, T * F)

        # 2) 线性映射
        x = self.patch_embedding(x)  # → (B, N, D)

        # 3) 加上位置编码（动态生成）
        x = self.pos_encoding(x)     # → (B, N, D)

        # 4) Transformer 编码：转置成 (N, B, D)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)        # → (B, N, D)

        # 5) LayerNorm
        x = self.norm(x)
        return x
