import torch
import torch.nn as nn
from .encoder import PositionalEncoding

class JEPPredictor(nn.Module):
    """
    改进版 JEPPredictor，带 patch‐level positional encoding、teacher‐forcing shift & mask、自回归推理
    """
    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        prediction_length: int = None
    ):
        """
        Args:
            latent_dim:        与 Encoder 输出一致的 latent 维度 D
            num_layers:        Transformer Encoder/Decoder 层数
            num_heads:         注意力头数
            ffn_dim:           FeedForward 网络维度，默认 4 * latent_dim
            dropout:           Dropout 比例
            prediction_length: 需要预测的 patch 数 N_tgt
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.pred_len = prediction_length

        if ffn_dim is None:
            ffn_dim = latent_dim * 4

        # Patch‐level 位置编码
        self.pos_encoding = PositionalEncoding(latent_dim)

        # 标准 Transformer（Encoder+Decoder）
        self.transformer = nn.Transformer(
            d_model=latent_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影层
        self.projection = nn.Linear(latent_dim, latent_dim)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成 (sz, sz) 的上三角 mask，用于屏蔽未来位置
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask  # True 表示要屏蔽

    def forward(
        self,
        ctx_latents: torch.Tensor,
        tgt_latents: torch.Tensor = None
    ):
        """
        Args:
            ctx_latents: [B, N_ctx, D]
            tgt_latents: [B, N_tgt, D] 或 None
        Returns:
            如果 tgt_latents 不为 None：返回 (pred_latents, loss)
            如果 tgt_latents 为 None：返回 (pred_latents, None)
        """
        device = ctx_latents.device
        B, N_ctx, D = ctx_latents.shape

        # 给 ctx_latents 加位置编码
        ctx = self.pos_encoding(ctx_latents)  # (B, N_ctx, D)

        if tgt_latents is not None:
            # —— 训练模式（teacher‐forcing）
            B, N_tgt, D = tgt_latents.shape

            # 1) 构造 decoder 输入：在最前面加一个全零 <sos>，并去掉最后一个元素
            sos = torch.zeros(B, 1, D, device=device)
            tgt_input = torch.cat([sos, tgt_latents[:, :-1, :]], dim=1)  # (B, N_tgt, D)

            # 2) 给 tgt_input 加位置编码
            tgt = self.pos_encoding(tgt_input)  # (B, N_tgt, D)

            # 3) 构造 causal mask
            tgt_mask = self._generate_square_subsequent_mask(N_tgt, device)  # (N_tgt, N_tgt)

            # 4) 传入 Transformer
            output = self.transformer(
                src=ctx,         # (B, N_ctx, D)
                tgt=tgt,         # (B, N_tgt, D)
                tgt_mask=tgt_mask  # (N_tgt, N_tgt)
            )  # output: (B, N_tgt, D)

            # 5) 投影
            output = self.projection(output)  # (B, N_tgt, D)

            # 6) 计算 loss
            loss = nn.MSELoss()(output, tgt_latents)
            return output, loss

        else:
            # —— 推理模式（自回归）
            assert self.pred_len is not None, "prediction_length 未设置，无法进行自回归推理"
            N_tgt = self.pred_len

            # 1) 先准备空列表收集每一步生成的 latent
            preds = []

            # 2) 初始 decoder 输入为一个全零 <sos>
            prev = torch.zeros(B, 1, D, device=device)  # (B, 1, D)

            # 3) 逐步生成 N_tgt 步
            for t in range(N_tgt):
                # 每一步把 prev 做位置编码
                prev_pe = self.pos_encoding(prev)  # (B, t+1, D)

                # 不需要 mask 最后一行，因为我们只取最后一个位置的输出并拼接
                # 但为了 transformer 接口，需要提供一个 tgt_mask 大小为 (t+1, t+1)
                tgt_mask = self._generate_square_subsequent_mask(t + 1, device)

                # 4) 用历史 ctx 和当前 prev_pe 做一次前向
                out = self.transformer(
                    src=ctx,     # (B, N_ctx, D) 固定不变
                    tgt=prev_pe, # (B, t+1, D)
                    tgt_mask=tgt_mask  # (t+1, t+1)
                )  # (B, t+1, D)

                # 5) 取出 out 的最后一个时间步，做一次投影 -> 下一个 latent
                next_latent = self.projection(out[:, -1:, :])  # (B, 1, D)
                preds.append(next_latent)

                # 6) 把 next_latent 拼接到 prev，用于下一个循环
                prev = torch.cat([prev, next_latent], dim=1)  # (B, t+2, D)

            # 7) 把所有 step 拼成 (B, N_tgt, D)
            pred_latents = torch.cat(preds, dim=1)
            return pred_latents, None
