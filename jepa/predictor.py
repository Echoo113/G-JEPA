import torch
import torch.nn as nn

class JEPPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_length: int,
        prediction_length: int,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = None
    ):
        """
        latent_dim:        Encoder 输出的 latent 向量维度 D
        context_length:    上下文 patch 数 N_ctx
        prediction_length: 目标 patch 数 N_tgt
        num_layers:        Transformer 层数
        num_heads:         Attention 头数
        ffn_dim:           Feed-forward 维度 (默认 4*D)
        """
        super().__init__()
        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # 使用标准 Transformer
        self.transformer = nn.Transformer(
            d_model=latent_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_dim,
            batch_first=True
        )

        # 可选的输出投影层
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        ctx_latents: torch.Tensor,
        tgt_latents: torch.Tensor = None
    ):
        """
        ctx_latents: [B, N_ctx, D]  已完成切 patch + Encoder 的上下文序列
        tgt_latents: [B, N_tgt, D]  （训练时）对应的目标序列，用于计算 loss
                     (推理时可留 None)
        """
        if tgt_latents is not None:
            # 训练模式：使用 teacher forcing
            output = self.transformer(ctx_latents, tgt_latents)
            output = self.projection(output)
            loss = nn.MSELoss()(output, tgt_latents)
            return output, loss
        else:
            # TODO: 实现推理模式（自回归预测）
            raise NotImplementedError("Inference mode not implemented yet.")
