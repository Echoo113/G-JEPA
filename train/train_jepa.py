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

from patch_loader import create_labeled_loader  # 修正导入
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= NEW: 定义分类器模型 =========
class Classifier(nn.Module):
    """一个简单的MLP分类器，用于判断一个补丁的latent表示是否异常"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # 添加dropout防止过拟合
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ========= 全局设置 (MODIFIED) =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数 (保持不变)
BATCH_SIZE               = 64
LATENT_DIM               = 512
EPOCHS                   = 100
LEARNING_RATE            = 1e-4
WEIGHT_DECAY             = 1e-6
EARLY_STOPPING_PATIENCE  = 15
EARLY_STOPPING_DELTA     = 1e-5

# --- NEW: 三个损失的权重 ---
W1 = 1.0  # L1: 自监督损失 (包含recon和contra)
W2 = 5.0  # L2: 来自pred_latent的分类损失
W3 = 5.0  # L3: 来自tgt_latent的分类损失

# 自监督损失内部权重 (保持不变)
RECONSTRUCTION_WEIGHT   = 1.0
CONTRASTIVE_WEIGHT      = 5.0
TEMPERATURE             = 0.05

# EMA 相关参数 (保持不变)
EMA_MOMENTUM            = 0.99
EMA_WARMUP_EPOCHS       = 10
EMA_WARMUP_MOMENTUM     = 0.95

# 数据集配置 (保持不变)
PATCH_LENGTH             = 20
NUM_VARS                 = 55
PREDICTION_LENGTH        = 9 # 你的Predictor预测的是9个补丁

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
train_loader = create_labeled_loader(
    feature_npz_path="data/MSL/patches/msl_tune_train.npz",
    label_npz_path="data/MSL/patches/msl_tune_train_labels.npz",
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = create_labeled_loader(
    feature_npz_path="data/MSL/patches/msl_tune_val.npz",
    label_npz_path="data/MSL/patches/msl_tune_val_labels.npz",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ========= Step 2: 初始化模型 (MODIFIED) =========
print("\n[Step 2] Initializing models...")
# Encoder 和 Predictor 保持不变
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

# --- NEW: 初始化两个独立的分类器 ---
classifier1 = Classifier(input_dim=LATENT_DIM).to(DEVICE)
classifier2 = Classifier(input_dim=LATENT_DIM).to(DEVICE)

print(f"Initialized models and classifiers (latent_dim={LATENT_DIM}).\n")

# --- MODIFIED: 更新优化器，加入分类器参数 ---
optimizer = torch.optim.AdamW(
    list(encoder_online.parameters()) +
    list(predictor.parameters()) +
    list(classifier1.parameters()) +
    list(classifier2.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

# ========= Step 3: 训练 + 验证 (MODIFIED) =========
print("\n[Step 3] Training and validating with hybrid loss...")
best_val_loss = float('inf')
patience_counter = 0
best_state = None

# --- NEW: 定义分类损失函数 ---
bce_criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, EPOCHS + 1):
    # ------ 训练阶段 ------
    encoder_online.train()
    predictor.train()
    classifier1.train()
    classifier2.train()
    total_train_L1, total_train_L2, total_train_L3 = 0.0, 0.0, 0.0

    for batch in train_loader:
        if len(batch) == 4:  # 带标签的数据
            x_batch, y_batch, x_label, y_label = batch
            labels_batch = y_label  # 使用目标序列的标签
        else:  # 不带标签的数据
            x_batch, y_batch = batch
            labels_batch = torch.zeros(x_batch.size(0), 1, device=DEVICE)  # 默认标签为0
            
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        # --- L1 损失 (自监督) ---
        ctx_latent = encoder_online(x_batch)
        with torch.no_grad():
            tgt_latent = encoder_target(y_batch)
        pred_latent, _ = predictor(ctx_latent, tgt_latent)
        
        loss_recon = nn.MSELoss()(pred_latent, tgt_latent)
        loss_contra = compute_contrastive_loss(pred_latent, tgt_latent, TEMPERATURE)
        loss_L1 = (RECONSTRUCTION_WEIGHT * loss_recon) + (CONTRASTIVE_WEIGHT * loss_contra)
        
        # --- L2, L3 损失 (监督) ---
        B, SEQ, D = pred_latent.shape
        
        # 将latent和label都拉平，以便每个补丁都能独立计算损失
        pred_latent_flat = pred_latent.reshape(B * SEQ, D)
        tgt_latent_flat = tgt_latent.reshape(B * SEQ, D)
        labels_batch_flat = labels_batch.reshape(-1, 1).float()  # 修改这里，确保标签维度正确
        
        # 获取logits
        logits_L2 = classifier1(pred_latent_flat)
        logits_L3 = classifier2(tgt_latent_flat)
        
        # 计算分类损失
        loss_L2 = bce_criterion(logits_L2, labels_batch_flat)
        loss_L3 = bce_criterion(logits_L3, labels_batch_flat)
        
        # === 最终总损失 ===
        total_loss = (W1 * loss_L1) + (W2 * loss_L2) + (W3 * loss_L3)
        
        total_loss.backward()
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
    predictor.eval()
    classifier1.eval()
    classifier2.eval()
    total_val_L1, total_val_L2, total_val_L3 = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:  # 带标签的数据
                x_batch, y_batch, x_label, y_label = batch
                labels_batch = y_label  # 使用目标序列的标签
            else:  # 不带标签的数据
                x_batch, y_batch = batch
                labels_batch = torch.zeros(x_batch.size(0), 1, device=DEVICE)  # 默认标签为0
                
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            ctx_latent = encoder_online(x_batch)
            tgt_latent = encoder_target(y_batch)
            val_pred_latent, _ = predictor(ctx_latent, tgt_latent)
            
            val_loss_L1 = (RECONSTRUCTION_WEIGHT * nn.MSELoss()(val_pred_latent, tgt_latent)) + \
                          (CONTRASTIVE_WEIGHT * compute_contrastive_loss(val_pred_latent, tgt_latent, TEMPERATURE))

            B, SEQ, D = val_pred_latent.shape
            pred_latent_flat = val_pred_latent.reshape(B * SEQ, D)
            tgt_latent_flat = tgt_latent.reshape(B * SEQ, D)
            labels_batch_flat = labels_batch.reshape(-1, 1).float()  # 修改这里，确保标签维度正确
            
            logits_L2 = classifier1(pred_latent_flat)
            logits_L3 = classifier2(tgt_latent_flat)
            
            val_loss_L2 = bce_criterion(logits_L2, labels_batch_flat)
            val_loss_L3 = bce_criterion(logits_L3, labels_batch_flat)
            
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
                'patch_length': PATCH_LENGTH,
                'num_vars': NUM_VARS,
                'prediction_length': PREDICTION_LENGTH
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