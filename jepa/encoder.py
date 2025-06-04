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
    def __init__(
        self,
        patch_length: int = 16,
        num_vars: int = 137,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # —— 在这里先做一个小型 1D Conv，把 (T, F) → (T, hidden_dim)
        self.hidden_dim = latent_dim // 2  # 举例：hidden_dim = 128
        self.conv1 = nn.Conv1d(
            in_channels=num_vars,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.GELU()

        # 先对每个时间步 t 加位置编码 (T, hidden_dim)
        self.time_pos_emb = nn.Parameter(torch.randn(patch_length, self.hidden_dim))

        # 然后把 (T, hidden_dim) flatten → (T * hidden_dim) 再映射到 latent_dim
        self.fc_embed = nn.Linear(patch_length * self.hidden_dim, latent_dim)

        # —— 动态位置编码：对 patch 维度 (N) 做位置编码
        self.patch_pos_encoding = PositionalEncoding(latent_dim)

        # —— Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # —— 最后 LayerNorm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, F)
        """
        B, N, T, F = x.shape
        assert F == self.num_vars
        # 1) 先把 x -> (B*N, F, T)，做 1D Conv
        x_ = x.view(B * N, T, F).transpose(1, 2)   # (B*N, F, T)
        h = self.act(self.conv1(x_))              # (B*N, hidden_dim, T)
        # 2) 加时间位置编码 (T, hidden_dim)
        h = h.transpose(1, 2) + self.time_pos_emb.unsqueeze(0)  # (B*N, T, hidden_dim)
        # 3) Flatten 时间维度，再映射到 latent
        h = h.reshape(B * N, T * self.hidden_dim)      # (B*N, T*hidden_dim)
        patch_latent = self.fc_embed(h)           # (B*N, latent_dim)
        patch_latent = patch_latent.view(B, N, self.latent_dim)  # (B, N, latent_dim)

        # 4) 对 patch 维度做位置编码 (B, N, latent_dim)
        patch_latent = self.patch_pos_encoding(patch_latent)

        # 5) Transformer 编码：先转到 (N, B, D)
        y = patch_latent.transpose(0, 1)           # (N, B, D)
        y = self.transformer(y)                   # (N, B, D)
        y = y.transpose(0, 1)                      # (B, N, D)

        # 6) LayerNorm
        y = self.norm(y)                           # (B, N, D)
        return y
