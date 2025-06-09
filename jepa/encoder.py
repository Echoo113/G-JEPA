import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码模块，为Transformer添加时间顺序信息
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, N, D)
        Returns:
            加入位置编码后的张量，形状不变
        """
        B, N, D = x.shape
        device = x.device

        # 生成位置索引 (N, 1)
        position = torch.arange(N, dtype=torch.float, device=device).unsqueeze(1)
        # 计算位置频率项 (D//2,)
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))

        # 初始化位置编码矩阵 (N, D)
        pe = torch.zeros(N, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置为sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置为cos
        pe = pe.unsqueeze(0)  # (1, N, D)，方便与输入相加
        return x + pe

class MyTimeSeriesEncoder(nn.Module):
    """
    改进版时间序列Encoder：
    - 保留patch内部的时间结构和变量关系
    - 使用Transformer编码时序特征，不使用平均池化或flatten操作
    - 使用CLS Token方式聚合每个patch表示

    输入形状: (B, N, 16, 137) 表示每个batch有N个patch，每个patch 16个时间步，137个变量
    输出形状: (B, N, 1024) 表示每个patch编码为1024维的latent表示
    """
    def __init__(
        self,
        patch_length: int = 16,
        num_vars: int = 137,
        latent_dim: int = 1024,
        time_layers: int = 2,
        patch_layers: int = 3,
        num_attention_heads: int = 16,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # 1）将变量维度137线性映射到hidden_dim维（默认256）
        self.hidden_dim = latent_dim // 4
        self.var_proj = nn.Linear(num_vars, self.hidden_dim)  # (16, 137) → (16, 256)

        # 2）为每个patch加一个[CLS] token用于聚合
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))  # (1, 1, 256)

        # 3）时间位置编码，用于区分时间步顺序
        self.time_pos_encoder = PositionalEncoding(self.hidden_dim)

        # 4）patch内部使用时间Transformer编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=min(self.hidden_dim // 16, 16),
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.time_encoder = nn.TransformerEncoder(encoder_layer, num_layers=time_layers)

        # 5）将[CLS] token提取出的表示投影为最终latent表示（1024）
        self.cls_to_latent = nn.Linear(self.hidden_dim, latent_dim)

        # 6）patch级位置编码 + Transformer（如果需要堆叠多个patch）
        self.patch_pos_encoder = PositionalEncoding(latent_dim)
        patch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.patch_encoder = nn.TransformerEncoder(patch_encoder_layer, num_layers=patch_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, N, 16, 137)
        Returns:
            输出 latent 表示，形状为 (B, N, 1024)
        """
        B, N, T, F = x.shape
        assert F == self.num_vars

        # Step 1: 映射每个patch的变量维度，从137 → 256
        x = self.var_proj(x)  # (B, N, 16, 256)

        # Step 2: 插入[CLS] token → (B, N, 17, 256)
        cls = self.cls_token.expand(B * N, -1, -1)  # (B*N, 1, 256)
        patch = x.view(B * N, T, -1)                # (B*N, 16, 256)
        patch = torch.cat([cls, patch], dim=1)      # (B*N, 17, 256)

        # Step 3: 加位置编码
        patch = self.time_pos_encoder(patch)  # (B*N, 17, 256)

        # Step 4: Transformer编码patch内部结构
        patch_encoded = self.time_encoder(patch)  # (B*N, 17, 256)

        # Step 5: 提取[CLS] token表示作为patch表示
        patch_cls = patch_encoded[:, 0]  # (B*N, 256)

        # Step 6: 映射为latent维度
        patch_latent = self.cls_to_latent(patch_cls)  # (B*N, 1024)
        patch_latent = patch_latent.view(B, N, self.latent_dim)  # (B, N, 1024)

        # Step 7: patch级编码（用于多个patch堆叠时建模）
        patch_latent = self.patch_pos_encoder(patch_latent)
        patch_latent = self.patch_encoder(patch_latent)

        # Step 8: 最后归一化
        return self.norm(patch_latent)
