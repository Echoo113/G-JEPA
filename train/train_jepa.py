
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torch.nn as nn
import numpy as np
from jepa.encoder import MyTimeSeriesEncoder, prepare_batch_from_np
from jepa.predictor import JEPPredictor


# ========= 超参数设置 =========
BATCH_SIZE = 2
LATENT_DIM = 64
EPOCHS     = 30
PATCH_FILE = "data/SOLAR/patches/solar_patches.npz"  # 你之前存好的 npz 文件

# ========= 工具函数 =========
def prepare_batched_tensor(np_array: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    把 numpy 的 patch 数据变成 (B, N, T, F) 的 Tensor 输入 Encoder
    - 例如输入 shape = (172, 30, 137)，batch_size = 4
    - 输出 shape = (4, 43, 30, 137)
    """
    total, T, F = np_array.shape
    N = total // batch_size
    usable = batch_size * N
    return torch.tensor(np_array[:usable], dtype=torch.float).view(batch_size, N, T, F)

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """计算 MSE 和 MAE"""
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

# ========= Step 1: 加载 patch 数据 =========
print("\n[Step 1] Loading patch data...")
npz = np.load(PATCH_FILE)
ctx_long_np = npz['long_term_context']
fut_long_np = npz['long_term_future']
ctx_short_np = npz['short_term_context']
fut_short_np = npz['short_term_future']

# 准备 batched tensors
ctx_long_batch = prepare_batched_tensor(ctx_long_np, BATCH_SIZE)
fut_long_batch = prepare_batched_tensor(fut_long_np, BATCH_SIZE)
ctx_short_batch = prepare_batched_tensor(ctx_short_np, BATCH_SIZE)
fut_short_batch = prepare_batched_tensor(fut_short_np, BATCH_SIZE)

# ========= Step 2: 初始化模型 =========
encoder = MyTimeSeriesEncoder(
    patch_length=30,
    num_vars=137,
    latent_dim=LATENT_DIM,
    num_layers=2,
    num_attention_heads=2,
)

# 初始化两个 predictor
predictor_long = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=ctx_long_batch.shape[1],
    prediction_length=fut_long_batch.shape[1],
    num_layers=4,
    num_heads=4
)

predictor_short = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=ctx_short_batch.shape[1],
    prediction_length=fut_short_batch.shape[1],
    num_layers=4,
    num_heads=4
)

print(f"\n[Debug] Model configurations:")
print(f"- Encoder latent dim: {LATENT_DIM}")
print(f"- Long-term context/prediction: {ctx_long_batch.shape[1]}/{fut_long_batch.shape[1]}")
print(f"- Short-term context/prediction: {ctx_short_batch.shape[1]}/{fut_short_batch.shape[1]}")

# 分别创建优化器
optimizer_long = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor_long.parameters()),
    lr=1e-3
)

optimizer_short = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor_short.parameters()),
    lr=1e-3
)

# ========= Step 3: 训练长期预测 =========
print("\n[Step 3] Training long-term prediction...")

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    predictor_long.train()
    optimizer_long.zero_grad()

    ctx_L = encoder(ctx_long_batch)
    tgt_L = encoder(fut_long_batch)
    pred_L, loss_L = predictor_long(ctx_L, tgt_L)
    
    loss_L.backward()
    optimizer_long.step()
    
    # 计算长期预测的指标
    metrics_L = compute_metrics(pred_L, tgt_L)
    
    # 打印训练进度
    print(f"[Long-term Epoch {epoch:02d}] Loss: {loss_L.item():.4f}, MSE: {metrics_L['mse']:.4f}, MAE: {metrics_L['mae']:.4f}")

    # 在第一个epoch打印shape信息
    if epoch == 1:
        print(f"\n[Debug] Long-term shapes:")
        print(f"- Context: {ctx_L.shape}")
        print(f"- Target: {tgt_L.shape}")
        print(f"- Prediction: {pred_L.shape}")

print("\n[Long-term training completed]")

# ========= Step 4: 训练短期预测 =========
print("\n[Step 4] Training short-term prediction...")

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    predictor_short.train()
    optimizer_short.zero_grad()

    ctx_S = encoder(ctx_short_batch)
    tgt_S = encoder(fut_short_batch)
    pred_S, loss_S = predictor_short(ctx_S, tgt_S)
    
    loss_S.backward()
    optimizer_short.step()
    
    # 计算短期预测的指标
    metrics_S = compute_metrics(pred_S, tgt_S)

    # 打印训练进度
    print(f"[Short-term Epoch {epoch:02d}] Loss: {loss_S.item():.4f}, MSE: {metrics_S['mse']:.4f}, MAE: {metrics_S['mae']:.4f}")

    # 在第一个epoch打印shape信息
    if epoch == 1:
        print(f"\n[Debug] Short-term shapes:")
        print(f"- Context: {ctx_S.shape}")
        print(f"- Target: {tgt_S.shape}")
        print(f"- Prediction: {pred_S.shape}")

print("\n[Short-term training completed]")
print("\n[All training completed]")
