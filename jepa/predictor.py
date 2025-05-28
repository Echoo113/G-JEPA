import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction

class JEPPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_length: int,
        prediction_length: int,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = None,
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

        # 1) 准备 Config
        config = PatchTSTConfig(
            num_input_channels   = latent_dim,
            context_length       = context_length,
            prediction_length    = prediction_length,
            patch_length         = 1,
            patch_stride         = 1,
            d_model              = latent_dim,
            num_hidden_layers    = num_layers,
            num_attention_heads  = num_heads,
            ffn_dim              = ffn_dim,
            do_mask_input        = False,
        )

        # 2) 实例化预测模型
        self.model = PatchTSTForPrediction(config)

        # 3) 跳过默认的 patch 切分＋embedding 层
        #    直接把 (B, N_ctx, D) 当作 Transformer 输入
        #    注意：不同版本下层名称可能略有差异，请根据 print(self.model) 调整
        self.model.to_patch_embedding = nn.Identity()
        self.model.positional_encoding = nn.Identity()

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
        outputs = self.model(
            past_values   = ctx_latents,
            future_values = tgt_latents
        )
        preds = outputs.prediction_outputs  # [B, N_tgt, D]

        if tgt_latents is not None:
            return preds, outputs.loss      # (predictions, scalar loss)
        return preds                         # inference 模式，只返回预测值
