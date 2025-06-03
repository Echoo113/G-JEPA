import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= 设置 =========
BATCH_SIZE = 4
LATENT_DIM = 64
EPOCHS     = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA    = 1e-4

PATCH_FILE = "data/SOLAR/patches/solar_train.npz"
VAL_FILE   = "data/SOLAR/patches/solar_val.npz"
TEST_FILE  = "data/SOLAR/patches/solar_test.npz"
STATS_FILE = "data/SOLAR/solar_data_statistics.csv"

# 每个 patch 的时间步长和变量数
PATCH_LENGTH = 30
NUM_VARS     = 137
ONE_PATCH_SIZE = PATCH_LENGTH * NUM_VARS  # 4110

# ========= 工具函数 =========
def prepare_batched_tensor(np_array: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    将形状 (total_patches, T, F) 的 numpy 数组重组为 (B, N, T, F) 形式，
    其中 B=batch_size, N=total_patches//B.
    """
    total, T, F = np_array.shape
    N = total // batch_size
    usable = batch_size * N
    return torch.tensor(
        np_array[:usable], dtype=torch.float
    ).view(batch_size, N, T, F)

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """计算整体 MSE 和 MAE"""
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

def load_statistics(stats_path: str) -> tuple:
    """加载数据统计信息"""
    stats_df = pd.read_csv(stats_path, index_col=0)
    means = torch.tensor(stats_df["mean"].values, dtype=torch.float).view(1, 1, NUM_VARS)
    stds = torch.tensor(stats_df["std"].values, dtype=torch.float).view(1, 1, NUM_VARS)
    return means, stds

def unnormalize(x_flat: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    """
    还原预测值到原始尺度
    Args:
        x_flat: shape (B, 30*137) 的展平张量
        means: shape (1, 1, 137) 的均值
        stds: shape (1, 1, 137) 的标准差
    Returns:
        shape (B, 30*137) 的还原后的展平张量
    """
    B = x_flat.shape[0]
    x_reshaped = x_flat.view(B, PATCH_LENGTH, NUM_VARS)  # (B, T, F)
    x_restored = x_reshaped * stds + means
    return x_restored.view(B, -1)  # (B, 30*137)

def restore_and_evaluate(pred_flat: torch.Tensor, y_flat: torch.Tensor, stats_path: str) -> dict:
    """
    还原预测值和真实值到原始尺度，并计算指标
    Args:
        pred_flat: shape (B, 30*137) 的预测值
        y_flat: shape (B, 30*137) 的真实值
        stats_path: 统计信息文件路径
    Returns:
        包含原始尺度下 MSE 和 MAE 的字典
    """
    means, stds = load_statistics(stats_path)
    y_true_restored = unnormalize(y_flat, means, stds)
    y_pred_restored = unnormalize(pred_flat, means, stds)
    return compute_metrics(y_pred_restored, y_true_restored)

# ========= Step 1: 加载并冻结 encoder + predictor_short/predictor_long =========
print("\n[Step 1] Loading pretrained JEPA models...")

ckpt = torch.load('model/jepa_models.pt', map_location='cpu')
encoder = MyTimeSeriesEncoder(**ckpt['encoder_config'])
predictor_short = JEPPredictor(**ckpt['predictor_short_config'])
predictor_long  = JEPPredictor(**ckpt['predictor_long_config'])

encoder.load_state_dict(ckpt['encoder_state_dict'])
predictor_short.load_state_dict(ckpt['predictor_short_state_dict'])
predictor_long.load_state_dict( ckpt['predictor_long_state_dict'])

# 冻结参数
encoder.eval()
predictor_short.eval()
predictor_long.eval()
for p in encoder.parameters():        p.requires_grad = False
for p in predictor_short.parameters(): p.requires_grad = False
for p in predictor_long.parameters():  p.requires_grad = False

print("→ Encoder and both predictors are loaded and frozen.")

# ========= Step 2: 读取并准备训练/验证/测试数据 =========
print("\n[Step 2] Loading patch data...")

# 训练集
npz_train = np.load(PATCH_FILE)
ctx_long_train  = prepare_batched_tensor(npz_train['long_term_context'],  BATCH_SIZE)  # [B, N_ctx, 30,137]
fut_long_train  = prepare_batched_tensor(npz_train['long_term_future'],   BATCH_SIZE)  # [B, N_fut, 30,137]

# 验证集
npz_val = np.load(VAL_FILE)
ctx_long_val  = prepare_batched_tensor(npz_val['long_term_context'],  BATCH_SIZE)
fut_long_val  = prepare_batched_tensor(npz_val['long_term_future'],   BATCH_SIZE)

# 测试集
npz_test = np.load(TEST_FILE)
ctx_long_test  = prepare_batched_tensor(npz_test['long_term_context'],  BATCH_SIZE)
fut_long_test  = prepare_batched_tensor(npz_test['long_term_future'],   BATCH_SIZE)

print(f"ctx_long_train:  {tuple(ctx_long_train.shape)}, fut_long_train:  {tuple(fut_long_train.shape)}")
print(f"ctx_long_val:    {tuple(ctx_long_val.shape)},   fut_long_val:    {tuple(fut_long_val.shape)}")
print(f"ctx_long_test:   {tuple(ctx_long_test.shape)},  fut_long_test:   {tuple(fut_long_test.shape)}")

# ========= Step 3: 预先计算 latent-space 预测（仅一次） =========
print("\n[Step 3] Precomputing latent predictions for training/validation/test ...")

with torch.no_grad():
    # 3.a) 把 context 送进 encoder 得到 z_ctx（[B, N_ctx, latent_dim]）
    z_ctx_train  = encoder(ctx_long_train)  # 使用 long-term context
    z_ctx_val    = encoder(ctx_long_val)
    z_ctx_test   = encoder(ctx_long_test)

    # 3.b) 把 fut patches 也转为 latent
    z_fut_train  = encoder(fut_long_train)  # [B, N_fut, latent_dim]
    z_fut_val    = encoder(fut_long_val)
    z_fut_test   = encoder(fut_long_test)

    # 3.c) predictor_long 直接预测 "所有 future latent"：
    z_pred_train_full, _ = predictor_long(z_ctx_train, z_fut_train)
    z_pred_val_full,   _ = predictor_long(z_ctx_val,   z_fut_val)
    z_pred_test_full,  _ = predictor_long(z_ctx_test,  z_fut_test)

# z_pred_*_full 的形状均为 [B, N_fut, latent_dim]


# ========= Step 4: 定义 ForecastHead，用于把 latent（dim=64）映射回时序 patch（dim=4110） =========
class ForecastHead(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.head(x)


# 初始化 ForecastHead
forecast_head = ForecastHead(
    latent_dim=LATENT_DIM,
    output_dim=ONE_PATCH_SIZE  # 30*137=4110
)
optimizer = torch.optim.Adam(forecast_head.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
loss_fn = nn.MSELoss()


# ========= Step 5: 将 "[B, N_fut, latent_dim]" 展平成 "[B*N_fut, latent_dim]"，以及对应目标 "[B*N_fut, 4110]" =========
def flatten_for_head(z_pred_full, fut_patches):
    """
    输入：
      - z_pred_full: [B, N_fut, latent_dim]
      - fut_patches: [B, N_fut, T, F]
    返回：
      - z_flat:  [B*N_fut, latent_dim]
      - y_flat:  [B*N_fut, T*F]
    """
    B, N_fut, D = z_pred_full.shape
    # flatten latent
    z_flat = z_pred_full.view(B * N_fut, D)  # [B*N_fut, latent_dim]
    # flatten 真实值
    y_flat = fut_patches.view(B * N_fut, PATCH_LENGTH * NUM_VARS)  # [B*N_fut, 4110]
    return z_flat, y_flat

# 训练集、验证集的平坦化
z_train_flat,  y_train_flat  = flatten_for_head(z_pred_train_full, fut_long_train)
z_val_flat,    y_val_flat    = flatten_for_head(z_pred_val_full,   fut_long_val)
# 测试集的平坦化
z_test_flat,   y_test_flat   = flatten_for_head(z_pred_test_full,  fut_long_test)

print(f"\n→ z_train_flat: {tuple(z_train_flat.shape)},  y_train_flat: {tuple(y_train_flat.shape)}")
print(f"→ z_val_flat:   {tuple(z_val_flat.shape)},    y_val_flat:   {tuple(y_val_flat.shape)}")
print(f"→ z_test_flat:  {tuple(z_test_flat.shape)},   y_test_flat:   {tuple(y_test_flat.shape)}")

# ========= Step 6: 划分 ForecastHead 的训练/验证 split（如 75% 训练，25% 验证） =========
num_train = z_train_flat.size(0)
# 按 75%/25% 随机划分
idx = torch.randperm(num_train)
n_tr = int(num_train * 0.75)
train_idx = idx[:n_tr]
val_idx   = idx[n_tr:]

z_head_tr, y_head_tr = z_train_flat[train_idx], y_train_flat[train_idx]
z_head_va, y_head_va = z_train_flat[val_idx],   y_train_flat[val_idx]

print(f"\n[Step 6] ForecastHead train/val split: {z_head_tr.shape[0]} 训练样本, {z_head_va.shape[0]} 验证样本")


# ========= Step 7: 训练 ForecastHead =========
print("\n[Step 7] Training ForecastHead ...")

train_losses = []
val_losses   = []
best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    ## --- (1) 训练一步 ---
    forecast_head.train()
    optimizer.zero_grad()
    pred_tr = forecast_head(z_head_tr)    # shape [n_tr, 4110]
    loss_tr = loss_fn(pred_tr, y_head_tr)  # 对整个 "B*N_fut" 展平后的一批计算 MSE
    loss_tr.backward()
    optimizer.step()
    train_losses.append(loss_tr.item())

    ## --- (2) 验证一步 ---
    forecast_head.eval()
    with torch.no_grad():
        pred_va = forecast_head(z_head_va)
        loss_va = loss_fn(pred_va, y_head_va)
        val_losses.append(loss_va.item())
        scheduler.step(loss_va)

        # Early stopping
        if loss_va < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = loss_va.item()
            torch.save(forecast_head.state_dict(), 'model/best_forecast_head.pt')
            patience_counter = 0
        else:
            patience_counter += 1

    print(f"[Epoch {epoch:02d}] Train MSE={loss_tr.item():.6f} | Val MSE={loss_va.item():.6f}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"→ Early stopping at epoch {epoch}")
        break

# 加载最优权重
forecast_head.load_state_dict(torch.load('model/best_forecast_head.pt'))
print("[Step 7] Best ForecastHead loaded.")


# ========= Step 8: 测试时用 long-term predictor 预测 + ForecastHead 再映射回时序，计算指标 =========
print("\n[Step 8] Evaluating on test set (long-term predictor) ...")
forecast_head.eval()
with torch.no_grad():
    # Long-term predictor evaluation
    z_ctx_test_long = encoder(ctx_long_test)
    z_fut_test_long = encoder(fut_long_test)
    z_pred_test_long, _ = predictor_long(z_ctx_test_long, z_fut_test_long)
    
    # Flatten tensors for ForecastHead
    z_pred_test_long_flat = z_pred_test_long.view(-1, LATENT_DIM)  # [B*N, D]
    y_test_flat = fut_long_test.view(-1, PATCH_LENGTH * NUM_VARS)  # [B*N, 4110]
    
    # Pass through ForecastHead
    pred_test_flat_long = forecast_head(z_pred_test_long_flat)
    
    # Compute metrics in normalized space
    metrics_long = compute_metrics(pred_test_flat_long, y_test_flat)
    print("Long-term predictor → ForecastHead 预测结果 (归一化空间)：")
    print(f"  MSE = {metrics_long['mse']:.6f}, MAE = {metrics_long['mae']:.6f}")
    
    # Compute metrics in original space
    metrics_long_restored = restore_and_evaluate(pred_test_flat_long, y_test_flat, STATS_FILE)
    print("Long-term predictor → ForecastHead 预测结果 (原始空间)：")
    print(f"  MSE = {metrics_long_restored['mse']:.6f}, MAE = {metrics_long_restored['mae']:.6f}")

    # Short-term predictor evaluation
    z_ctx_test_short = encoder(ctx_long_test)
    z_fut_test_short = encoder(fut_long_test)
    z_pred_test_short, _ = predictor_short(z_ctx_test_short, z_fut_test_short)
    
    # Flatten tensors for ForecastHead
    z_pred_test_short_flat = z_pred_test_short.view(-1, LATENT_DIM)  # [B*N, D]
    
    # Pass through ForecastHead
    pred_test_flat_short = forecast_head(z_pred_test_short_flat)
    
    # Compute metrics in normalized space
    metrics_short = compute_metrics(pred_test_flat_short, y_test_flat)
    print("Short-term predictor → ForecastHead 预测结果 (归一化空间)：")
    print(f"  MSE = {metrics_short['mse']:.6f}, MAE = {metrics_short['mae']:.6f}")
    
    # Compute metrics in original space
    metrics_short_restored = restore_and_evaluate(pred_test_flat_short, y_test_flat, STATS_FILE)
    print("Short-term predictor → ForecastHead 预测结果 (原始空间)：")
    print(f"  MSE = {metrics_short_restored['mse']:.6f}, MAE = {metrics_short_restored['mae']:.6f}")


# ========= Step 9: 可视化 ForecastHead 的训练曲线 =========
print("\n[Step 9] Plotting ForecastHead training history ...")
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses,   label='Val MSE')
plt.title("ForecastHead Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

plt.show()

# ========= 全部完成 =========
print("\nAll done! 😊")
