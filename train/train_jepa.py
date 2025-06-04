import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_loader import create_patch_loader
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= 全局设置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE               = 32
LATENT_DIM               = 256
EPOCHS                   = 100
LEARNING_RATE            = 5e-4
WEIGHT_DECAY            = 1e-6
EARLY_STOPPING_PATIENCE  = 30
EARLY_STOPPING_DELTA     = 1e-6

PATCH_FILE_TRAIN         = "data/SOLAR/patches/solar_train.npz"
PATCH_FILE_VAL           = "data/SOLAR/patches/solar_val.npz"
PATCH_FILE_TEST          = "data/SOLAR/patches/solar_test.npz"

# 每个 patch 的时间步长和变量数（和你 DataLoader 里一致）
PATCH_LENGTH             = 16
NUM_VARS                 = 137

# ========= 工具函数 =========
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算 MSE 和 MAE 误差
    pred, target 形状均为 (B, N_patches, PATCH_LENGTH, NUM_VARS)
    """
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}


# ========= Step 1: 准备 DataLoader =========
print("[Step 1] Preparing DataLoaders...")

# 训练集：shuffle=True，每轮 epoch 都打乱样本顺序
train_loader = create_patch_loader(PATCH_FILE_TRAIN, BATCH_SIZE, shuffle=True)
# 验证集 / 测试集：shuffle=False（保持固定顺序）
val_loader   = create_patch_loader(PATCH_FILE_VAL,   BATCH_SIZE, shuffle=False)
test_loader  = create_patch_loader(PATCH_FILE_TEST,  BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")


# ========= Step 2: 初始化模型 =========
print("\n[Step 2] Initializing models...")

# Encoder：把每个 (B, N_ctx, T, F) → (B, N_ctx, D)
encoder = MyTimeSeriesEncoder(
    patch_length=PATCH_LENGTH,
    num_vars=NUM_VARS,
    latent_dim=LATENT_DIM,
    num_layers=6,                # 增加到6层
    num_attention_heads=8,       # 增加到8个头
    ffn_dim=LATENT_DIM*4,       # 设置FFN维度为latent_dim的4倍
    dropout=0.1                  # 添加dropout
).to(DEVICE)

# Predictor：把 (B, N_ctx, D) + (B, N_tgt, D) → 输出 (B, N_tgt, D) + loss
predictor = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=None,    # 会在 forward 时根据输入自动推断
    prediction_length=None, # 同上
    num_layers=4,
    num_heads=4
).to(DEVICE)

# 优化器：使用AdamW并添加weight decay
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(predictor.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters())}")


# ========= Step 3: 训练 + 验证 =========
print("\n[Step 3] Training and validating...")

best_val_mse = float('inf')
patience_counter = 0
best_state = None

# 用来记录每轮的指标
history = {
    'train_mse': [], 'train_mae': [],
    'val_mse':   [], 'val_mae':   []
}

for epoch in range(1, EPOCHS + 1):
    # ------ 训练阶段 ------
    encoder.train()
    predictor.train()
    running_train_mse = 0.0
    running_train_mae = 0.0
    train_samples = 0

    for x_batch, y_batch in train_loader:
        # x_batch, y_batch: (B, N_patches, PATCH_LENGTH, NUM_VARS)
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        # 1) 用 Encoder 编码上下文 patch（历史）
        ctx_latent = encoder(x_batch)  # (B, N_ctx, D)
        # 2) 用 Encoder 编码目标 patch（未来），作为监督信息
        tgt_latent = encoder(y_batch)  # (B, N_tgt, D)

        # 3) Predictor 用上下文 latent + 目标 latent 做 teacher-forcing 训练
        pred_latent, loss = predictor(ctx_latent, tgt_latent)

        # 4) 反向传播
        loss.backward()
        optimizer.step()

        # 5) 计算并累计训练指标
        with torch.no_grad():
            mse_val = nn.MSELoss(reduction='sum')(pred_latent, tgt_latent).item()
            mae_val = nn.L1Loss(reduction='sum')(pred_latent, tgt_latent).item()
            running_train_mse += mse_val
            running_train_mae += mae_val
            train_samples += pred_latent.numel()

    # 训练集平均误差
    avg_train_mse = running_train_mse / train_samples
    avg_train_mae = running_train_mae / train_samples
    history['train_mse'].append(avg_train_mse)
    history['train_mae'].append(avg_train_mae)

    # ------ 验证阶段 ------
    encoder.eval()
    predictor.eval()
    running_val_mse = 0.0
    running_val_mae = 0.0
    val_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            ctx_latent = encoder(x_batch)
            tgt_latent = encoder(y_batch)
            val_pred_latent, _ = predictor(ctx_latent, tgt_latent)

            mse_val = nn.MSELoss(reduction='sum')(val_pred_latent, tgt_latent).item()
            mae_val = nn.L1Loss(reduction='sum')(val_pred_latent, tgt_latent).item()
            running_val_mse += mse_val
            running_val_mae += mae_val
            val_samples += val_pred_latent.numel()

    avg_val_mse = running_val_mse / val_samples
    avg_val_mae = running_val_mae / val_samples
    history['val_mse'].append(avg_val_mse)
    history['val_mae'].append(avg_val_mae)

    # ------ Early Stopping 判断 ------
    if avg_val_mse < best_val_mse - EARLY_STOPPING_DELTA:
        best_val_mse = avg_val_mse
        patience_counter = 0
        best_state = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'epoch': epoch,
            'val_mse': avg_val_mse
        }
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # 更新学习率
    scheduler.step(avg_val_mse)

    # ------ 打印当前 epoch 信息 ------
    print(f"[Epoch {epoch:02d}] "
          f"Train MSE: {avg_train_mse:.6f}, MAE: {avg_train_mae:.6f} | "
          f" Val MSE: {avg_val_mse:.6f}, MAE: {avg_val_mae:.6f}")

    # First-epoch shape调试信息
    if epoch == 1:
        print(f"  [Debug Shapes] "
              f"x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")
        print(f"                ctx_latent: {ctx_latent.shape}, tgt_latent: {tgt_latent.shape}")
        print(f"                pred_latent: {val_pred_latent.shape}")

print("\n[Training completed]")


# ========= Step 4: 保存最佳模型 =========
if best_state is not None:
    os.makedirs("model", exist_ok=True)
    torch.save({
        'encoder_state_dict':   best_state['encoder'],
        'predictor_state_dict': best_state['predictor'],
        'latent_dim':           LATENT_DIM,
        'patch_length':         PATCH_LENGTH,
        'num_vars':             NUM_VARS
    }, "model/jepa_best.pt")
    print(f"Best model saved (epoch {best_state['epoch']}), val_mse = {best_state['val_mse']:.6f}")


# ========= Step 5: 测试评估（可选） =========
print("\n[Step 5] Testing on unseen data...")
encoder.load_state_dict(best_state['encoder'])
predictor.load_state_dict(best_state['predictor'])
encoder.eval()
predictor.eval()

running_test_mse = 0.0
running_test_mae = 0.0
test_samples = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        ctx_latent = encoder(x_batch)
        tgt_latent = encoder(y_batch)
        test_pred_latent, _ = predictor(ctx_latent, tgt_latent)

        mse_val = nn.MSELoss(reduction='sum')(test_pred_latent, tgt_latent).item()
        mae_val = nn.L1Loss(reduction='sum')(test_pred_latent, tgt_latent).item()
        running_test_mse += mse_val
        running_test_mae += mae_val
        test_samples += test_pred_latent.numel()

avg_test_mse = running_test_mse / test_samples
avg_test_mae = running_test_mae / test_samples

print(f"Test MSE: {avg_test_mse:.6f}, MAE: {avg_test_mae:.6f}")
