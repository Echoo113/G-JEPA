import sys
import os
import copy

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

# 训练超参数
BATCH_SIZE               = 64
LATENT_DIM               = 1024
EPOCHS                   = 100
LEARNING_RATE            = 5e-4
WEIGHT_DECAY            = 1e-6
EARLY_STOPPING_PATIENCE  = 20
EARLY_STOPPING_DELTA     = 1e-6
TRAIN_WEIGHT            = 0.4
VAL_WEIGHT              = 0.6

# EMA 相关参数
EMA_MOMENTUM            = 0.99  # EMA 动量参数
EMA_WARMUP_EPOCHS       = 10    # EMA 预热轮数
EMA_WARMUP_MOMENTUM     = 0.95  # 预热期的 EMA 动量

# 数据集配置
PATCH_LENGTH             = 16    # 每个patch 16步
NUM_VARS                 = 137   # 137个变量
PREDICTION_LENGTH        = 5     # 预测未来5个patch（从数据形状看是5而不是9）

# ========= 工具函数 =========
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算 MSE 和 MAE 误差
    pred, target 形状均为 (B, 5, 1024) - 预测5个patch
    """
    mse = nn.MSELoss(reduction='sum')(pred, target)
    mae = nn.L1Loss(reduction='sum')(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

@torch.no_grad()
def update_ema_encoder(encoder_online: nn.Module, encoder_target: nn.Module, momentum: float):
    """
    使用 EMA 更新 target encoder 的参数
    """
    for param_o, param_t in zip(encoder_online.parameters(), encoder_target.parameters()):
        param_t.data = momentum * param_t.data + (1 - momentum) * param_o.data

def get_ema_momentum(epoch: int) -> float:
    """
    根据当前 epoch 计算 EMA momentum
    - 预热期使用较小的 momentum
    - 预热期后使用正常 momentum
    """
    if epoch < EMA_WARMUP_EPOCHS:
        # 线性插值：从 EMA_WARMUP_MOMENTUM 到 EMA_MOMENTUM
        progress = epoch / EMA_WARMUP_EPOCHS
        return EMA_WARMUP_MOMENTUM + progress * (EMA_MOMENTUM - EMA_WARMUP_MOMENTUM)
    return EMA_MOMENTUM

# ========= Step 1: 准备 DataLoader =========
print("[Step 1] Preparing DataLoaders...")

train_loader = create_patch_loader("data/SOLAR/patches/solar_train.npz", BATCH_SIZE, shuffle=True)
val_loader   = create_patch_loader("data/SOLAR/patches/solar_val.npz",   BATCH_SIZE, shuffle=False)
test_loader  = create_patch_loader("data/SOLAR/patches/solar_test.npz",  BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# ========= Step 2: 初始化模型 =========
print("\n[Step 2] Initializing models...")

# 1) Online Encoder
encoder_online = MyTimeSeriesEncoder(
    patch_length=PATCH_LENGTH,
    num_vars=NUM_VARS,
    latent_dim=LATENT_DIM,
    time_layers=2,
    patch_layers=3,
    num_attention_heads=16,
    ffn_dim=LATENT_DIM*4,
    dropout=0.2
).to(DEVICE)

# 2) Target Encoder (EMA)
encoder_target = copy.deepcopy(encoder_online)
for param in encoder_target.parameters():
    param.requires_grad = False  # target encoder 不参与反向传播

# 3) Predictor
predictor = JEPPredictor(
    latent_dim=LATENT_DIM,
    num_layers=3,
    num_heads=16,
    ffn_dim=LATENT_DIM*4,
    dropout=0.2,
    prediction_length=PREDICTION_LENGTH
).to(DEVICE)

print(f"Initialized Online Encoder, Target Encoder (EMA), and Predictor (latent_dim={LATENT_DIM}).\n")

# 优化器：只优化 online encoder 和 predictor
optimizer = torch.optim.AdamW(
    list(encoder_online.parameters()) + list(predictor.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

print(f"Online Encoder parameters: {sum(p.numel() for p in encoder_online.parameters())}")
print(f"Target Encoder parameters: {sum(p.numel() for p in encoder_target.parameters())}")
print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters())}")

# ========= Step 3: 训练 + 验证 =========
print("\n[Step 3] Training and validating...")

best_val_mse = float('inf')
patience_counter = 0
best_state = None

history = {
    'train_mse': [], 'train_mae': [],
    'val_mse':   [], 'val_mae':   []
}

for epoch in range(1, EPOCHS + 1):
    # ------ 训练阶段 ------
    encoder_online.train()
    predictor.train()
    running_train_mse = 0.0
    running_train_mae = 0.0
    train_samples = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        # 使用 online encoder 编码输入
        ctx_latent = encoder_online(x_batch)
        # 使用 target encoder (EMA) 编码目标
        with torch.no_grad():
            tgt_latent = encoder_target(y_batch)
        # 预测目标表示
        pred_latent, loss = predictor(ctx_latent, tgt_latent)

        loss.backward()
        optimizer.step()

        # 更新 target encoder (EMA)
        momentum = get_ema_momentum(epoch)
        update_ema_encoder(encoder_online, encoder_target, momentum)

        # 计算指标
        metrics = compute_metrics(pred_latent, tgt_latent)
        running_train_mse += metrics['mse']
        running_train_mae += metrics['mae']
        train_samples += pred_latent.numel()

    # 训练集平均误差
    avg_train_mse = running_train_mse / train_samples
    avg_train_mae = running_train_mae / train_samples
    history['train_mse'].append(avg_train_mse)
    history['train_mae'].append(avg_train_mae)

    # ------ 验证阶段 ------
    encoder_online.eval()
    predictor.eval()
    running_val_mse = 0.0
    running_val_mae = 0.0
    val_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # 验证时也使用 online encoder 编码输入，target encoder 编码目标
            ctx_latent = encoder_online(x_batch)
            tgt_latent = encoder_target(y_batch)
            val_pred_latent, _ = predictor(ctx_latent, tgt_latent)

            metrics = compute_metrics(val_pred_latent, tgt_latent)
            running_val_mse += metrics['mse']
            running_val_mae += metrics['mae']
            val_samples += val_pred_latent.numel()

    avg_val_mse = running_val_mse / val_samples
    avg_val_mae = running_val_mae / val_samples
    history['val_mse'].append(avg_val_mse)
    history['val_mae'].append(avg_val_mae)

    # ------ Early Stopping 判断 ------
    combined_score = TRAIN_WEIGHT * avg_train_mse + VAL_WEIGHT * avg_val_mse
    
    if combined_score < best_val_mse - EARLY_STOPPING_DELTA:
        best_val_mse = combined_score
        patience_counter = 0
        best_state = {
            'encoder_online': encoder_online.state_dict(),
            'encoder_target': encoder_target.state_dict(),
            'predictor': predictor.state_dict(),
            'epoch': epoch,
            'train_mse': avg_train_mse,
            'val_mse': avg_val_mse,
            'combined_score': combined_score
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
          f"Val MSE: {avg_val_mse:.6f}, MAE: {avg_val_mae:.6f} | "
          f"Combined: {combined_score:.6f} | "
          f"EMA m: {get_ema_momentum(epoch):.3f}")

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
        'encoder_online_state_dict':  best_state['encoder_online'],
        'encoder_target_state_dict':  best_state['encoder_target'],
        'predictor_state_dict':       best_state['predictor'],
        'latent_dim':                 LATENT_DIM,
        'patch_length':               PATCH_LENGTH,
        'num_vars':                   NUM_VARS,
        'train_mse':                  best_state['train_mse'],
        'val_mse':                    best_state['val_mse'],
        'combined_score':             best_state['combined_score']
    }, "model/jepa_best.pt")
    print(f"Best model saved (epoch {best_state['epoch']})")
    print(f"  Train MSE: {best_state['train_mse']:.6f}")
    print(f"  Val MSE: {best_state['val_mse']:.6f}")
    print(f"  Combined Score: {best_state['combined_score']:.6f}")

# ========= Step 5: 测试评估 =========
print("\n[Step 5] Testing on unseen data...")
encoder_online.load_state_dict(best_state['encoder_online'])
encoder_target.load_state_dict(best_state['encoder_target'])
predictor.load_state_dict(best_state['predictor'])
encoder_online.eval()
predictor.eval()

running_test_mse = 0.0
running_test_mae = 0.0
test_samples = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        ctx_latent = encoder_online(x_batch)
        tgt_latent = encoder_target(y_batch)
        test_pred_latent, _ = predictor(ctx_latent, tgt_latent)

        metrics = compute_metrics(test_pred_latent, tgt_latent)
        running_test_mse += metrics['mse']
        running_test_mae += metrics['mae']
        test_samples += test_pred_latent.numel()

avg_test_mse = running_test_mse / test_samples
avg_test_mae = running_test_mae / test_samples

print(f"Test MSE: {avg_test_mse:.6f}, MAE: {avg_test_mae:.6f}")
