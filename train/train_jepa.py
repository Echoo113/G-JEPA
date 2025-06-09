import sys
import os
import copy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from patch_loader import create_patch_loader
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= 全局设置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
BATCH_SIZE               = 64
LATENT_DIM               = 512
EPOCHS                   = 100
LEARNING_RATE            = 1e-4  # 保持较低的学习率
WEIGHT_DECAY             = 1e-6
EARLY_STOPPING_PATIENCE  = 15    # MODIFIED: 稍微增加耐心
EARLY_STOPPING_DELTA     = 1e-5

# 损失函数权重 (强化版)
RECONSTRUCTION_WEIGHT   = 1.0   # α: 重建损失权重
CONTRASTIVE_WEIGHT      = 25.0  # MODIFIED (β): 进一步提高对比损失权重，施加更大压力
TEMPERATURE             = 0.05  # MODIFIED: 降低温度系数，让对比任务更困难

# EMA 相关参数
EMA_MOMENTUM            = 0.99
EMA_WARMUP_EPOCHS       = 10
EMA_WARMUP_MOMENTUM     = 0.95

# 数据集配置
PATCH_LENGTH             = 20
NUM_VARS                 = 55
PREDICTION_LENGTH        = 9

# ========= 工具函数 (保持不变) =========
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = nn.MSELoss(reduction='mean')(pred, target)
    mae = nn.L1Loss(reduction='mean')(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

def compute_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    B, L, D = pred.shape
    pred_flat = pred.reshape(-1, D)
    target_flat = target.reshape(-1, D)
    pred_norm = F.normalize(pred_flat, dim=1)
    target_norm = F.normalize(target_flat, dim=1)
    similarity_matrix = torch.matmul(pred_norm, target_norm.t()) / temperature
    labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
    return F.cross_entropy(similarity_matrix, labels)

@torch.no_grad()
def update_ema_encoder(encoder_online: nn.Module, encoder_target: nn.Module, momentum: float):
    for param_o, param_t in zip(encoder_online.parameters(), encoder_target.parameters()):
        param_t.data = momentum * param_t.data + (1.0 - momentum) * param_o.data

def get_ema_momentum(epoch: int) -> float:
    if epoch < EMA_WARMUP_EPOCHS:
        progress = epoch / EMA_WARMUP_EPOCHS
        return EMA_WARMUP_MOMENTUM + progress * (EMA_MOMENTUM - EMA_WARMUP_MOMENTUM)
    return EMA_MOMENTUM

# ========= Step 1: 准备 DataLoader =========
print("[Step 1] Preparing DataLoaders...")
train_loader = create_patch_loader("data/MSL/patches/msl_train.npz", BATCH_SIZE, shuffle=True)
val_loader   = create_patch_loader("data/MSL/patches/msl_val.npz",   BATCH_SIZE, shuffle=False)
# ...

# ========= Step 2: 初始化模型 =========
print("\n[Step 2] Initializing models...")
encoder_online = MyTimeSeriesEncoder(
    patch_length=PATCH_LENGTH, num_vars=NUM_VARS, latent_dim=LATENT_DIM, time_layers=2,
    patch_layers=3, num_attention_heads=16, ffn_dim=LATENT_DIM*4, dropout=0.2
).to(DEVICE)
encoder_target = copy.deepcopy(encoder_online)
for param in encoder_target.parameters():
    param.requires_grad = False
predictor = JEPPredictor(
    latent_dim=LATENT_DIM, num_layers=3, num_heads=16, ffn_dim=LATENT_DIM*4,
    dropout=0.2, prediction_length=PREDICTION_LENGTH
).to(DEVICE)
print(f"Initialized models (latent_dim={LATENT_DIM}).\n")

optimizer = torch.optim.AdamW(
    list(encoder_online.parameters()) + list(predictor.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# MODIFIED: 使用余弦退火学习率调度器
# T_max 是学习率下降一个周期的总步数，通常设为总的训练轮数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

# ========= Step 3: 训练 + 验证 =========
print("\n[Step 3] Training and validating with enhanced parameters...")
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # ------ 训练阶段 ------
    encoder_online.train()
    predictor.train()
    total_train_recon, total_train_contra = 0.0, 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        
        ctx_latent = encoder_online(x_batch)
        with torch.no_grad():
            tgt_latent = encoder_target(y_batch)
        pred_latent, _ = predictor(ctx_latent, tgt_latent)
        
        loss_recon = nn.MSELoss(reduction='mean')(pred_latent, tgt_latent)
        loss_contra = compute_contrastive_loss(pred_latent, tgt_latent, TEMPERATURE)
        total_loss = (RECONSTRUCTION_WEIGHT * loss_recon) + (CONTRASTIVE_WEIGHT * loss_contra)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder_online.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()
        
        momentum = get_ema_momentum(epoch)
        update_ema_encoder(encoder_online, encoder_target, momentum)
        
        total_train_recon += loss_recon.item()
        total_train_contra += loss_contra.item()

    avg_train_recon = total_train_recon / len(train_loader)
    avg_train_contra = total_train_contra / len(train_loader)

    # ------ 验证阶段 ------
    encoder_online.eval()
    predictor.eval()
    total_val_recon, total_val_contra = 0.0, 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            ctx_latent = encoder_online(x_batch)
            tgt_latent = encoder_target(y_batch)
            val_pred_latent, _ = predictor(ctx_latent, tgt_latent)
            
            val_loss_recon = nn.MSELoss(reduction='mean')(val_pred_latent, tgt_latent)
            val_loss_contra = compute_contrastive_loss(val_pred_latent, tgt_latent, TEMPERATURE)
            
            total_val_recon += val_loss_recon.item()
            total_val_contra += val_loss_contra.item()
    
    avg_val_recon = total_val_recon / len(val_loader)
    avg_val_contra = total_val_contra / len(val_loader)
    
    # 早停和模型保存都基于验证集的总损失
    avg_val_total_loss = (RECONSTRUCTION_WEIGHT * avg_val_recon) + (CONTRASTIVE_WEIGHT * avg_val_contra)
    
    # MODIFIED: 更新学习率调度器（每个epoch后都调用）
    scheduler.step()

    # ------ Early Stopping 判断 ------
    if avg_val_total_loss < best_val_loss - EARLY_STOPPING_DELTA:
        best_val_loss = avg_val_total_loss
        patience_counter = 0
        best_state = {
            'encoder_online_state_dict': encoder_online.state_dict(),
            'encoder_target_state_dict': encoder_target.state_dict(),
            'predictor_state_dict': predictor.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss,
            'config': {
                'latent_dim': LATENT_DIM, 'patch_length': PATCH_LENGTH,
                'num_vars': NUM_VARS, 'prediction_length': PREDICTION_LENGTH
            }
        }
        print(f"✅ [Epoch {epoch:02d}] New best model found! Val Loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1

    # ------ 打印当前 epoch 信息 ------
    current_lr = scheduler.get_last_lr()[0]
    print(f"[Epoch {epoch:02d}] Train Recon: {avg_train_recon:.4f}, Contra: {avg_train_contra:.4f} | "
          f"Val Recon: {avg_val_recon:.4f}, Contra: {avg_val_contra:.4f} | "
          f"Val Total Loss: {avg_val_total_loss:.4f} | "
          f"LR: {current_lr:.6f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

# ... (后续的保存和测试评估逻辑保持不变)
# ========= Step 4: 保存最佳模型 =========
if best_state is not None:
    os.makedirs("model", exist_ok=True)
    save_path = "model/jepa_best_hybrid.pt"
    torch.save(best_state, save_path)
    print(f"\nBest model from epoch {best_state['epoch']} saved to {save_path} (Val Loss: {best_state['val_loss']:.4f})")
else:
    print("\nTraining finished without finding a better model.")

