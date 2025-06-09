import torch
import torch.nn as nn
import math

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


class JEPPredictor(nn.Module):
    """
    JEPA Predictor：在 latent 空间做"历史 latent → 未来 latent"的 Transformer Seq2Seq，
    Decoder 输出仍然是 latent（不还原到原数据空间）。
    
    输入输出维度：
    - 输入: (B, N, D) - N个历史patch的latent，每个latent维度为D
    - 输出: (B, N, D) - N个未来patch的latent预测
    """
    def __init__(
        self,
        latent_dim: int,              # 与Encoder保持一致
        num_layers: int = 3,          # Transformer层数
        num_heads: int = 16,          # 注意力头数
        ffn_dim: int = None,
        dropout: float = 0.2,
        prediction_length: int = 9     # 预测的patch数量
    ):
        """
        Args:
            latent_dim:        Encoder 输出的 latent 维度 D
            num_layers:        Transformer Encoder/Decoder 层数
            num_heads:        注意力头数
            ffn_dim:         Feed‐forward 网络维度，默认 4 * latent_dim
            dropout:         dropout 比例
            prediction_length: 预测的 patch 数 N_tgt
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

        # Decoder 输出投影层（从 Transformer 输出直接映回 latent_dim）
        self.projection = nn.Linear(latent_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化所有权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成 (sz, sz) 的上三角 mask，用于屏蔽 Decoder 未来位置
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask  # True 表示该位置被屏蔽

    def forward(
        self,
        ctx_latents: torch.Tensor,
        tgt_latents: torch.Tensor = None
    ):
        """
        前向传播

        Args:
            ctx_latents: (B, N, D) - N个历史patch的latent
            tgt_latents: (B, N, D) - N个未来patch的latent，训练时需要
        Returns:
            如果 tgt_latents 不为 None（训练）：
                pred_latents:  (B, N, D)
                loss:          MSELoss(pred_latents, tgt_latents)
            如果 tgt_latents 为 None（推理）：
                pred_latents: (B, N, D)
                loss: None
        """
        device = ctx_latents.device
        B, N_ctx, D = ctx_latents.shape
        assert D == self.latent_dim, f"Expected latent dimension {self.latent_dim}, but got {D}"

        # 1) 给 ctx_latents 加位置编码
        ctx = self.pos_encoding(ctx_latents)  # (B, N, D)

        if tgt_latents is not None:
            # —— 训练模式：teacher forcing
            B, N_tgt, D = tgt_latents.shape
            assert D == self.latent_dim, f"Expected latent dimension {self.latent_dim}, but got {D}"
            assert N_tgt == self.pred_len, f"Expected prediction length {self.pred_len}, but got {N_tgt}"

            # 2) 构造 Decoder 输入：在最前面插入一个全零 <sos>，去掉最后一个位置
            sos = torch.zeros(B, 1, D, device=device)
            decoder_input = torch.cat([sos, tgt_latents[:, :-1, :]], dim=1)  # (B, N, D)

            # 3) 给 decoder_input 加位置编码
            dec = self.pos_encoding(decoder_input)  # (B, N, D)

            # 4) 构造 causal mask
            tgt_mask = self._generate_square_subsequent_mask(N_tgt, device)  # (N, N)

            # 5) 输入 Transformer
            output = self.transformer(
                src=ctx,
                tgt=dec,
                tgt_mask=tgt_mask
            )  # output: (B, N, D)

            # 6) 线性投影：保持在 latent_dim
            pred_latents = self.projection(output)  # (B, N, D)

            # 7) 损失计算
            loss = nn.MSELoss()(pred_latents, tgt_latents)
            return pred_latents, loss

        else:
            # —— 推理模式：自回归生成 N 个 latent
            assert self.pred_len is not None, "推理时必须提供 prediction_length"
            N_tgt = self.pred_len

            preds = []
            # 以一个全零 <sos> 开始
            prev = torch.zeros(B, 1, D, device=device)  # (B, 1, D)

            for t in range(N_tgt):
                # 1) 拼接上一步生成的 latent（或 <sos>）
                prev_pe = self.pos_encoding(prev)  # (B, t+1, D)

                # 2) 构造 causal mask：大小 (t+1, t+1)
                tgt_mask = self._generate_square_subsequent_mask(t + 1, device)

                # 3) 用 Transformer 进行一步推理：src=ctx, tgt=prev_pe
                out = self.transformer(
                    src=ctx,         # (B, N, D)
                    tgt=prev_pe,     # (B, t+1, D)
                    tgt_mask=tgt_mask
                )  # out: (B, t+1, D)

                # 4) 取出最后一个位置的隐藏向量，投影到下一个 latent
                next_latent = self.projection(out[:, -1:, :])  # (B, 1, D)
                preds.append(next_latent)

                # 5) 把 next_latent 拼回 prev，用于下一步生成
                prev = torch.cat([prev, next_latent], dim=1)  # (B, t+2, D)

            # 6) 拼成 (B, N, D)
            pred_latents = torch.cat(preds, dim=1)
            return pred_latents, None
