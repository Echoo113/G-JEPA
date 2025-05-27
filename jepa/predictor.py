# predictor.py
import torch
import torch.nn as nn
from einops import rearrange

class Predictor(nn.Module):
    def __init__(self, latent_dim=128, num_future=3, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.num_future = num_future

        # positional encoding + transformer
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # learnable future positional embeddings
        self.future_pos_embed = nn.Parameter(torch.randn(num_future, latent_dim))  # [N_future, D]

        # concatenate context summary with each future step's pos_embed
        self.out_proj = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, s_ctx):  # s_ctx: [B, N_ctx, D]
        B = s_ctx.size(0)
        h = self.transformer(self.pos_encoder(s_ctx))  # [B, N_ctx, D]
        summary = h.mean(dim=1)                        # [B, D]

        # each future step's positional encoding: copy to batch
        future_embed = self.future_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N_future, D]
        summary_expand = summary.unsqueeze(1).expand(-1, self.num_future, -1)  # [B, N_future, D]

        # concatenate [summary || pos_embed] → [B, N_future, 2D]
        concat = torch.cat([summary_expand, future_embed], dim=-1)  # [B, N_future, 2D]

        pred = self.out_proj(concat)  # [B, N_future, D]
        return pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
