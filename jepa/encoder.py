import torch
import torch.nn as nn
import math

# PositionalEncoding 类 (保持不变)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape; device = x.device
        position = torch.arange(N, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        pe = torch.zeros(N, D, device=device); pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term); pe = pe.unsqueeze(0)
        return x + pe

# ===================================================================
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓   请用下面的代码替换您文件中的旧版本   ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ===================================================================

class MyTimeSeriesEncoder(nn.Module):
    """
    【最终版】: 增加了可选的Instance Normalization层。
    """
    def __init__(
        self,
        patch_length: int,
        num_vars: int,
        latent_dim: int,
        time_layers: int = 2,
        patch_layers: int = 3,
        num_attention_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
        use_instance_norm: bool = False # <--- 增加了这个参数来接收开关
    ):
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars
        self.latent_dim = latent_dim
        self.use_instance_norm = use_instance_norm

        # 根据开关，条件性地创建Instance Normalization层
        if self.use_instance_norm:
            self.instance_norm = nn.InstanceNorm1d(num_vars, affine=True)

        if ffn_dim is None:
            ffn_dim = latent_dim * 4
        
        # --- 后续组件定义与您之前的版本相同 ---
        self.hidden_dim = latent_dim // 4
        self.var_proj = nn.Linear(num_vars, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.time_pos_encoder = PositionalEncoding(self.hidden_dim)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=max(1, self.hidden_dim // 16),
            dim_feedforward=self.hidden_dim * 4, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.time_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=time_layers)
        self.cls_to_latent = nn.Linear(self.hidden_dim, latent_dim)

        self.patch_pos_encoder = PositionalEncoding(latent_dim)
        patch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_attention_heads,
            dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.patch_encoder = nn.TransformerEncoder(patch_encoder_layer, num_layers=patch_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F = x.shape
        assert T == self.patch_length and F == self.num_vars, "输入尺寸与初始化参数不匹配"

        # 在最开始应用Instance Normalization
        if self.use_instance_norm:
            x_permuted = x.permute(0, 3, 1, 2)
            x_reshaped = x_permuted.reshape(B, F, N * T)
            x_norm = self.instance_norm(x_reshaped)
            x = x_norm.view(B, F, N, T).permute(0, 2, 3, 1)

        # 后续流程不变
        x = self.var_proj(x)
        patch = x.view(B * N, T, -1)
        cls_tokens = self.cls_token.expand(B * N, -1, -1)
        patch = torch.cat([cls_tokens, patch], dim=1)
        patch = self.time_pos_encoder(patch)
        patch_encoded = self.time_encoder(patch)
        patch_summary = self.cls_to_latent(patch_encoded[:, 0])
        patch_latents = patch_summary.view(B, N, self.latent_dim)
        patch_latents = self.patch_pos_encoder(patch_latents)
        contextual_latents = self.patch_encoder(patch_latents)
        final_latents = self.norm(contextual_latents)
        
        return final_latents