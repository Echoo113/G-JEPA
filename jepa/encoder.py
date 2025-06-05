import math
import torch
import torch.nn as nn


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
        return x + pe


class MyTimeSeriesEncoder(nn.Module):
    """
    改进版 TimeSeriesEncoder，保留 Patch 内部时序特征后再聚合到 latent
    
    针对 MSL 数据集优化：
    - 输入: (B, 9, 20, 55) - 9个patch，每个patch 20步，55个传感器
    - 输出: (B, 9, 128) - 9个patch的latent表示
    """
    def __init__(
        self,
        patch_length: int = 20,      # MSL: 每个patch 20步
        num_vars: int = 55,          # MSL: 55个传感器
        latent_dim: int = 128,       # 降低到128维，减少计算量
        time_layers: int = 2,        # 保持2层，视overfit情况调整
        patch_layers: int = 3,       # 减少到3层，减轻计算负担
        num_attention_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            patch_length: 每个 patch 的时间步数 T (MSL: 20)
            num_vars: 每个时间步的特征维度 F (MSL: 55)
            latent_dim: 最终输出的 patch-level latent 维度 D (128)
            time_layers: patch 内部时间 Transformer 层数 (2)
            patch_layers: patch 级别 Transformer 层数 (3)
            num_attention_heads: Attention 头数
            ffn_dim: feed-forward 层维度，默认 4 * latent_dim
            dropout: dropout 比例
        """
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # —— 第一阶段：patch 内部时序编码
        # 1) 1D Conv 将 (55 → hidden_dim)
        self.hidden_dim = latent_dim // 2  # 128 → 64
        self.conv1 = nn.Conv1d(
            in_channels=num_vars,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.act = nn.GELU()

        # 2) 时间位置编码 (20, hidden_dim)，可训练
        self.time_pos_emb = nn.Parameter(torch.randn(patch_length, self.hidden_dim))

        # 3) 小型 TransformerEncoder 处理时间维度
        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=min(self.hidden_dim // 16, 8),
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_time = nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=time_layers
        )

        # 4) 把 (20, hidden_dim) 平均池化或最大池化成 (hidden_dim)
        #    然后映射到 latent_dim
        self.fc_time2latent = nn.Linear(self.hidden_dim, latent_dim)

        # —— 第二阶段：patch 级别编码
        # 1) Patch‐level 位置编码
        self.patch_pos_encoding = PositionalEncoding(latent_dim)

        # 2) patch‐level TransformerEncoder
        encoder_layer_patch = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_patch = nn.TransformerEncoder(
            encoder_layer_patch,
            num_layers=patch_layers
        )

        # 3) 最后 LayerNorm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (B, 9, 20, 55) - MSL数据集
        Returns:
            Tensor, shape = (B, 9, 128) - 9个patch的latent表示
        """
        B, N, T, F = x.shape
        assert F == self.num_vars, f"输入特征维度 {F} 与 num_vars {self.num_vars} 不匹配"

        # —— Stage 1: patch 内部
        # 把每个 patch 展成 (B*N, 55, 20)，做 1D Conv
        x_ = x.view(B * N, T, F).transpose(1, 2)  # (B*N, 55, 20)
        h = self.act(self.conv1(x_))             # (B*N, 64, 20)

        # 转置到 (B*N, 20, 64)，加时间位置编码
        h = h.transpose(1, 2) + self.time_pos_emb.unsqueeze(0)  # (B*N, 20, 64)

        # 通过时间 Transformer，捕捉 patch 内部时序依赖
        h = self.transformer_time(h)  # (B*N, 20, 64)

        # 这里用平均池化得到 (B*N, 64)
        h_pooled = h.mean(dim=1)       # (B*N, 64)

        # 映射到 latent_dim
        patch_latent = self.fc_time2latent(h_pooled)  # (B*N, 128)
        patch_latent = patch_latent.view(B, N, self.latent_dim)  # (B, 9, 128)

        # —— Stage 2: patch 级别
        # 1) 加 Patch 位置编码
        patch_latent = self.patch_pos_encoding(patch_latent)  # (B, 9, 128)

        # 2) 通过 patch‐level Transformer
        y = self.transformer_patch(patch_latent)  # (B, 9, 128)

        # 3) 最后归一化
        y = self.norm(y)  # (B, 9, 128)
        return y
