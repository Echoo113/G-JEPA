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

# 训练超参数
BATCH_SIZE               = 64
LATENT_DIM               = 128        # 降低到128维，减少计算量和过拟合
EPOCHS                   = 100
LEARNING_RATE            = 5e-4
WEIGHT_DECAY            = 1e-6
EARLY_STOPPING_PATIENCE  = 30
EARLY_STOPPING_DELTA     = 1e-6
# 训练集和验证集的权重（用于综合评估）
TRAIN_WEIGHT            = 0.4  # 训练集权重
VAL_WEIGHT              = 0.6  # 验证集权重

# MSL数据集路径
PATCH_FILE_TRAIN         = "data/MSL/patches/msl_train.npz"
PATCH_FILE_VAL           = "data/MSL/patches/msl_val.npz"
PATCH_FILE_TEST          = "data/MSL/patches/msl_final_test.npz"

# MSL数据集配置
PATCH_LENGTH             = 20         # MSL: 每个patch 20步
NUM_VARS                 = 55         # MSL: 55个传感器
PREDICTION_LENGTH        = 9          # MSL: 预测未来9个patch

# ========= 工具函数 =========
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算 MSE 和 MAE 误差
    pred, target 形状均为 (B, 9, 128) - MSL数据集
    """
    mse = nn.MSELoss(reduction='sum')(pred, target)
    mae = nn.L1Loss(reduction='sum')(pred, target)
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

# 1) Encoder
encoder = MyTimeSeriesEncoder(
    patch_length=PATCH_LENGTH,    # 20
    num_vars=NUM_VARS,           # 55
    latent_dim=LATENT_DIM,       # 128
    time_layers=2,               # patch 内部时间 Transformer 层数
    patch_layers=3,              # patch 级别 Transformer 层数（减少到3层）
    num_attention_heads=8,       # 注意力头数
    ffn_dim=LATENT_DIM*4,        # feed-forward 层维度
    dropout=0.2                  # 增加dropout到0.2，防止过拟合
).to(DEVICE)

# 2) Predictor
predictor = JEPPredictor(
    latent_dim=LATENT_DIM,       # 128
    num_layers=3,                # Transformer 层数（减少到3层）
    num_heads=4,                 # 注意力头数（128/4=32，保证维度能整除）
    ffn_dim=LATENT_DIM*4,        # feed-forward 层维度
    dropout=0.2,                 # 增加dropout到0.2，防止过拟合
    prediction_length=PREDICTION_LENGTH  # 9
).to(DEVICE)

print(f"Initialized Encoder and Predictor (latent_dim={LATENT_DIM}).\n")

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
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        ctx_latent = encoder(x_batch)
        tgt_latent = encoder(y_batch)
        pred_latent, loss = predictor(ctx_latent, tgt_latent)

        loss.backward()
        optimizer.step()

        # 使用 compute_metrics 计算指标
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

            # 使用 compute_metrics 计算指标
            metrics = compute_metrics(val_pred_latent, tgt_latent)
            running_val_mse += metrics['mse']
            running_val_mae += metrics['mae']
            val_samples += val_pred_latent.numel()

    avg_val_mse = running_val_mse / val_samples
    avg_val_mae = running_val_mae / val_samples
    history['val_mse'].append(avg_val_mse)
    history['val_mae'].append(avg_val_mae)

    # ------ Early Stopping 判断 ------
    # 计算综合指标：训练集和验证集的加权和
    combined_score = TRAIN_WEIGHT * avg_train_mse + VAL_WEIGHT * avg_val_mse
    
    if combined_score < best_val_mse - EARLY_STOPPING_DELTA:
        best_val_mse = combined_score
        patience_counter = 0
        best_state = {
            'encoder': encoder.state_dict(),
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
          f"Combined: {combined_score:.6f}")

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
        'num_vars':             NUM_VARS,
        'train_mse':            best_state['train_mse'],
        'val_mse':              best_state['val_mse'],
        'combined_score':       best_state['combined_score']
    }, "model/jepa_best.pt")
    print(f"Best model saved (epoch {best_state['epoch']})")
    print(f"  Train MSE: {best_state['train_mse']:.6f}")
    print(f"  Val MSE: {best_state['val_mse']:.6f}")
    print(f"  Combined Score: {best_state['combined_score']:.6f}")


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

        # 使用 compute_metrics 计算指标
        metrics = compute_metrics(test_pred_latent, tgt_latent)
        running_test_mse += metrics['mse']
        running_test_mae += metrics['mae']
        test_samples += test_pred_latent.numel()

avg_test_mse = running_test_mse / test_samples
avg_test_mae = running_test_mae / test_samples

print(f"Test MSE: {avg_test_mse:.6f}, MAE: {avg_test_mae:.6f}")
