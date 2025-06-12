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

    输入形状: (B, N, T, F) 表示每个batch有N个patch，每个patch T个时间步，F个变量
    输出形状: (B, N, D) 表示每个patch编码为D维的latent表示
    """
    def __init__(
        self,
        patch_length: int,
        num_vars: int,
        latent_dim: int,
        time_layers: int = 2,
        patch_layers: int = 3,
        num_attention_heads: int = 16,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            patch_length: 每个patch的时间步数
            num_vars: 变量数量
            latent_dim: 输出的latent维度
            time_layers: 时间Transformer的层数
            patch_layers: patch级Transformer的层数
            num_attention_heads: 注意力头数
            ffn_dim: Feed-forward网络维度，默认4*latent_dim
            dropout: dropout比例
        """
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim

        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # 1）将变量维度映射到hidden_dim维
        self.hidden_dim = latent_dim // 4
        self.var_proj = nn.Linear(num_vars, self.hidden_dim)

        # 2）为每个patch加一个[CLS] token用于聚合
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

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

        # 5）将[CLS] token提取出的表示投影为最终latent表示
        self.cls_to_latent = nn.Linear(self.hidden_dim, latent_dim)

        # 6）patch级位置编码 + Transformer
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
            x: 输入张量，形状为 (B, N, T, F)
        Returns:
            输出 latent 表示，形状为 (B, N, D)
        """
        B, N, T, F = x.shape
        assert F == self.num_vars, f"Expected {self.num_vars} variables, but got {F}"

        # Step 1: 映射每个patch的变量维度
        x = self.var_proj(x)  # (B, N, T, hidden_dim)

        # Step 2: 插入[CLS] token
        cls = self.cls_token.expand(B * N, -1, -1)  # (B*N, 1, hidden_dim)
        patch = x.view(B * N, T, -1)                # (B*N, T, hidden_dim)
        patch = torch.cat([cls, patch], dim=1)      # (B*N, T+1, hidden_dim)

        # Step 3: 加位置编码
        patch = self.time_pos_encoder(patch)  # (B*N, T+1, hidden_dim)

        # Step 4: Transformer编码patch内部结构
        patch_encoded = self.time_encoder(patch)  # (B*N, T+1, hidden_dim)

        # Step 5: 提取[CLS] token表示作为patch表示
        patch_cls = patch_encoded[:, 0]  # (B*N, hidden_dim)

        # Step 6: 映射为latent维度
        patch_latent = self.cls_to_latent(patch_cls)  # (B*N, latent_dim)
        patch_latent = patch_latent.view(B, N, self.latent_dim)  # (B, N, latent_dim)

        # 删除patch级编码，直接返回第一阶段的结果
        return self.norm(patch_latent)
