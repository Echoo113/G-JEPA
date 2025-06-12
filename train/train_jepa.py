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

from patch_loader import get_loader  # 更新导入
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= NEW: 定义分类器模型 =========
class StrongClassifier(nn.Module):
    """简化版分类器，移除BatchNorm以提高稳定性"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加dropout
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x)



def apply_instance_norm(x, eps=1e-5):
    """
    对输入 x 的每个实例独立进行归一化
    输入:
        x: Tensor, shape (B, 1, T, F)
    输出:
        normalized x, shape 相同
    """
    # 保持Batch维度(B)和通道维度(1)独立，只在时间和特征维度上计算
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)

# ========= 全局设置 (MODIFIED) =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数 (保持不变)
BATCH_SIZE               = 256
LATENT_DIM               = 128
EPOCHS                   = 100  # 增加训练轮数
LEARNING_RATE            = 2e-5  # 降低学习率以减缓过拟合
WEIGHT_DECAY             = 1e-4  # 增加权重衰减以增强正则化
EARLY_STOPPING_PATIENCE  = 20    # 增加早停耐心值
EARLY_STOPPING_DELTA     = 1e-4  # 增加早停阈值

# --- NEW: 三个损失的权重 ---
W1 = 1.0  # L1: 自监督损失 (包含recon和contra)
W2 = 1.0  # L2: 来自pred_latent的分类损失
W3 = 1.0  # L3: 来自tgt_latent的分类损失

# 自监督损失内部权重 (保持不变)
RECONSTRUCTION_WEIGHT   = 0.9    # 重建损失权重
CONTRASTIVE_WEIGHT      = 0.1    # 对比损失权重
TEMPERATURE             = 0.1    # 增加温度参数

# EMA 相关参数 (保持不变)
EMA_MOMENTUM            = 0.95   # 降低EMA动量以加快收敛
EMA_WARMUP_EPOCHS       = 5      # 减少预热轮数
EMA_WARMUP_MOMENTUM     = 0.95

# 数据集配置 (保持不变)
X_PATCH_LENGTH          = 30  # 修改为实际的输入序列长度
NUM_VARS                = 1   # 修改为实际的变量数
PREDICTION_STEPS        = 10  # 修改为实际的预测序列长度
Y_PATCH_SIZE           = 10   # 修改为实际的预测序列长度

# ========= 工具函数 (保持不变) =========
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

# ========= Step 1: 准备 DataLoader (MODIFIED) =========
print("[Step 1] Preparing DataLoaders...")
train_loader = get_loader(
    npz_file="data/MSL/patches/train.npz",
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = get_loader(
    npz_file="data/MSL/patches/val.npz",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- NEW: 计算全局异常比例 ---
def compute_global_anomaly_ratio(loader):
    total_labels = []
    for _, _, labels in loader:
        total_labels.append(labels)
    all_labels = torch.cat(total_labels, dim=0)
    return all_labels.float().mean().item()

print("\nComputing global anomaly ratio...")
global_anomaly_ratio = compute_global_anomaly_ratio(train_loader)
fixed_pos_weight = torch.tensor([(1 - global_anomaly_ratio) / (global_anomaly_ratio + 1e-6)], device=DEVICE)
print(f"Global anomaly ratio: {global_anomaly_ratio:.4f}")
print(f"Fixed positive weight: {fixed_pos_weight.item():.4f}")

# 初始化BCE损失函数
bce_criterion = nn.BCEWithLogitsLoss(pos_weight=fixed_pos_weight)

# ========= Step 2: 初始化模型 (MODIFIED) =========
print("\n[Step 2] Initializing models...")
# 1. 编码历史patch的encoder (30步)
encoder_online = MyTimeSeriesEncoder(
    patch_length=X_PATCH_LENGTH,  # 30步
    num_vars=NUM_VARS,
    latent_dim=LATENT_DIM,
    time_layers=2,    # 增加时间层数
    patch_layers=2,   # 增加patch层数
    num_attention_heads=8,
    ffn_dim=LATENT_DIM*4,
    dropout=0.4       # 增加dropout以增强正则化
).to(DEVICE)

# 2. 编码未来patch的encoder (10步)
encoder_target = MyTimeSeriesEncoder(
    patch_length=Y_PATCH_SIZE,    # 10步，与predictor的patch_size保持一致
    num_vars=NUM_VARS,
    latent_dim=LATENT_DIM,
    time_layers=2,    # 增加时间层数
    patch_layers=2,   # 增加patch层数
    num_attention_heads=8,
    ffn_dim=LATENT_DIM*4,
    dropout=0.4       # 增加dropout以增强正则化
).to(DEVICE)

# 3. 初始化target encoder的参数（使用online encoder的参数）
for param_o, param_t in zip(encoder_online.parameters(), encoder_target.parameters()):
    param_t.data.copy_(param_o.data)
    param_t.requires_grad = False

# 4. 初始化predictor
predictor = JEPPredictor(
    latent_dim=LATENT_DIM,
    prediction_steps=PREDICTION_STEPS,
    patch_size=Y_PATCH_SIZE,
    num_layers=2,     # 增加层数
    num_heads=8,
    ffn_dim=LATENT_DIM*4,
    dropout=0.4       # 增加dropout以增强正则化
).to(DEVICE)

# 5. 初始化分类器
classifier1 = StrongClassifier(input_dim=LATENT_DIM).to(DEVICE)  # 使用增强版分类器
classifier2 = StrongClassifier(input_dim=LATENT_DIM).to(DEVICE)  # 使用增强版分类器

print(f"Initialized models:")
print(f"- encoder_online: patch_length={X_PATCH_LENGTH}")
print(f"- encoder_target: patch_length={Y_PATCH_SIZE}")
print(f"- predictor: prediction_steps={PREDICTION_STEPS}, patch_size={Y_PATCH_SIZE}")
print(f"- latent_dim={LATENT_DIM}\n")

# --- MODIFIED: 更新优化器，加入分类器参数 ---
optimizer = torch.optim.AdamW(
    list(encoder_online.parameters()) +
    list(predictor.parameters()) +
    list(classifier1.parameters()) +
    list(classifier2.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

# ========= Step 3: 训练 + 验证 (MODIFIED) =========
print("\n[Step 3] Training and validating with hybrid loss...")
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # ------ 训练阶段 ------
    encoder_online.train()
    encoder_target.train()  # 注意：虽然target encoder不参与反向传播，但需要设置为train模式以保持一致性
    predictor.train()
    classifier1.train()
    classifier2.train()
    total_train_L1, total_train_L2, total_train_L3 = 0.0, 0.0, 0.0

    for batch_idx, batch in enumerate(train_loader):
        x_batch, y_batch, labels_batch = batch
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)
       
        
        # 添加N_patch维度，使输入变为4D: (B, N_patch, T, F)
        x_batch = x_batch.unsqueeze(1)  # (B, T, F) → (B, 1, T, F)
        y_batch = y_batch.unsqueeze(1)  # (B, T, F) → (B, 1, T, F)
        
        # === 使用实例归一化 ===
        x_batch = apply_instance_norm(x_batch)
        y_batch = apply_instance_norm(y_batch)
        
        optimizer.zero_grad()
        
        # --- L1 损失 (自监督) ---
        ctx_latent = encoder_online(x_batch)  # 编码30步的历史patch
        with torch.no_grad():
            tgt_latent = encoder_target(y_batch)  # 编码10步的未来patch
        pred_latent, pred_loss = predictor(ctx_latent, tgt_latent)
      
        
        loss_recon = pred_loss  # 使用predictor返回的损失
        loss_contra = compute_contrastive_loss(pred_latent, tgt_latent, TEMPERATURE)
        loss_L1 = (RECONSTRUCTION_WEIGHT * loss_recon) + (CONTRASTIVE_WEIGHT * loss_contra)
        
        # --- L2, L3 损失 (监督) ---
        B, SEQ, D = pred_latent.shape
        
        # 将latent和label都拉平，以便每个补丁都能独立计算损失
        pred_latent_flat = pred_latent.reshape(B * SEQ, D)
        tgt_latent_flat = tgt_latent.reshape(B * SEQ, D)
        # 修复标签维度：将标签扩展到每个patch
        labels_batch_expanded = labels_batch.view(-1, 1).repeat(1, SEQ).view(-1, 1).float()
        
        # 使用固定的pos_weight，不再每个batch重新计算
        # L2: 预测latent的分类损失（更新encoder和predictor）
        logits_L2 = classifier1(pred_latent_flat)
        loss_L2 = bce_criterion(logits_L2, labels_batch_expanded)
        
        # L3: 真实latent的分类损失（更新encoder和classifier2）
        tgt_latent_for_L3 = encoder_online(y_batch)  # 使用encoder_online编码Y，让L3的梯度可以传播到encoder_online
        tgt_latent_for_L3_flat = tgt_latent_for_L3.reshape(B * SEQ, D)
        logits_L3 = classifier2(tgt_latent_for_L3_flat)
        loss_L3 = bce_criterion(logits_L3, labels_batch_expanded)
        
        # === 最终总损失 ===
        total_loss = (W1 * loss_L1) + (W2 * loss_L2) + (W3 * loss_L3)
        
        total_loss.backward()
        # Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(
            list(encoder_online.parameters()) +
            list(predictor.parameters()) +
            list(classifier1.parameters()) +
            list(classifier2.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        momentum = get_ema_momentum(epoch)
        update_ema_encoder(encoder_online, encoder_target, momentum)
        
        total_train_L1 += loss_L1.item()
        total_train_L2 += loss_L2.item()
        total_train_L3 += loss_L3.item()

    avg_train_L1 = total_train_L1 / len(train_loader)
    avg_train_L2 = total_train_L2 / len(train_loader)
    avg_train_L3 = total_train_L3 / len(train_loader)

    # ------ 验证阶段 ------
    encoder_online.eval()
    encoder_target.eval()
    predictor.eval()
    classifier1.eval()
    classifier2.eval()
    total_val_L1, total_val_L2, total_val_L3 = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch, labels_batch = batch
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            # 添加N_patch维度，使输入变为4D: (B, N_patch, T, F)
            x_batch = x_batch.unsqueeze(1)  # (B, T, F) → (B, 1, T, F)
            y_batch = y_batch.unsqueeze(1)  # (B, T, F) → (B, 1, T, F)
            
            # === 使用实例归一化 ===
            x_batch = apply_instance_norm(x_batch)
            y_batch = apply_instance_norm(y_batch)
            
            ctx_latent = encoder_online(x_batch)
            tgt_latent = encoder_target(y_batch)
            val_pred_latent, val_pred_loss = predictor(ctx_latent, tgt_latent)
            
            val_loss_L1 = (RECONSTRUCTION_WEIGHT * val_pred_loss) + \
                          (CONTRASTIVE_WEIGHT * compute_contrastive_loss(val_pred_latent, tgt_latent, TEMPERATURE))

            B, SEQ, D = val_pred_latent.shape
            pred_latent_flat = val_pred_latent.reshape(B * SEQ, D)
            tgt_latent_flat = tgt_latent.reshape(B * SEQ, D)
            labels_batch_expanded = labels_batch.view(-1, 1).repeat(1, SEQ).view(-1, 1).float()
            
            # 使用固定的pos_weight，不再每个batch重新计算
            logits_L2 = classifier1(pred_latent_flat)
            logits_L3 = classifier2(tgt_latent_flat)
            
            val_loss_L2 = bce_criterion(logits_L2, labels_batch_expanded)
            val_loss_L3 = bce_criterion(logits_L3, labels_batch_expanded)
            
            total_val_L1 += val_loss_L1.item()
            total_val_L2 += val_loss_L2.item()
            total_val_L3 += val_loss_L3.item()
    
    avg_val_L1 = total_val_L1 / len(val_loader)
    avg_val_L2 = total_val_L2 / len(val_loader)
    avg_val_L3 = total_val_L3 / len(val_loader)
    
    avg_val_total_loss = (W1 * avg_val_L1) + (W2 * avg_val_L2) + (W3 * avg_val_L3)
    
    scheduler.step()

    # ------ Early Stopping 判断 ------
    if avg_val_total_loss < best_val_loss - EARLY_STOPPING_DELTA:
        best_val_loss = avg_val_total_loss
        patience_counter = 0
        best_state = {
            'encoder_online_state_dict': encoder_online.state_dict(),
            'encoder_target_state_dict': encoder_target.state_dict(),
            'predictor_state_dict': predictor.state_dict(),
            'classifier1_state_dict': classifier1.state_dict(),
            'classifier2_state_dict': classifier2.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss,
            'config': {
                'latent_dim': LATENT_DIM,
                'patch_length': X_PATCH_LENGTH,
                'num_vars': NUM_VARS,
                'prediction_steps': PREDICTION_STEPS,
                'patch_size': Y_PATCH_SIZE
            }
        }
        print(f"✅ [Epoch {epoch:02d}] New best model found! Total Val Loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1

    # ------ 打印当前 epoch 信息 ------
    current_lr = scheduler.get_last_lr()[0]
    print(f"[Epoch {epoch:02d}] Train L1: {avg_train_L1:.4f}, L2: {avg_train_L2:.4f}, L3: {avg_train_L3:.4f} | "
          f"Val Total Loss: {avg_val_total_loss:.4f} | LR: {current_lr:.6f} | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

# ========= Step 4: 保存最佳模型 =========
if best_state is not None:
    os.makedirs("model", exist_ok=True)
    save_path = "model/jepa_best.pt"
    torch.save(best_state, save_path)
    print(f"\nBest model from epoch {best_state['epoch']} saved to {save_path} (Val Loss: {best_state['val_loss']:.4f})")
else:
    print("\nTraining finished without finding a better model.")