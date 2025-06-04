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

BATCH_SIZE      = 32
LATENT_DIM      = 256
EPOCHS_FOCUS    = 40         # 增加训练轮数到40
LEARNING_RATE_FH = 5e-4      # 降低学习率
WEIGHT_DECAY    = 1e-6       # 添加权重衰减

# 下游任务用的同样数据加载器
PATCH_FILE_TRAIN = "data/SOLAR/patches/solar_train.npz"
PATCH_FILE_VAL   = "data/SOLAR/patches/solar_val.npz"
PATCH_FILE_TEST  = "data/SOLAR/patches/solar_test.npz"

# Patch 的长度和变量数，与主模型保持一致
PATCH_LENGTH = 16
NUM_VARS     = 137

def overfit_test(focus_head, encoder, train_loader, num_epochs=100):
    """在单个batch上测试FocusHead的过拟合能力"""
    print("\n[Overfit Test] Starting overfit test on a single batch...")
    
    # 获取第一个batch
    x_batch, y_batch = next(iter(train_loader))
    y_batch = y_batch.to(DEVICE)
    
    # 获取latent表示
    with torch.no_grad():
        tgt_latent = encoder(y_batch)
    
    # 打印latent的统计信息
    print(f"Latent stats - Mean: {tgt_latent.mean().item():.6f}, Std: {tgt_latent.std().item():.6f}")
    print(f"First patch latent (first 5 dims): {tgt_latent[0, 0, :5].cpu().numpy()}")
    
    # 准备优化器和损失函数
    optimizer = torch.optim.Adam(focus_head.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 记录原始数据的统计信息
    print(f"\nOriginal y_batch stats:")
    print(f"Mean: {y_batch.mean().item():.6f}")
    print(f"Std: {y_batch.std().item():.6f}")
    print(f"Min: {y_batch.min().item():.6f}")
    print(f"Max: {y_batch.max().item():.6f}")
    
    # 开始过拟合训练
    focus_head.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_values = focus_head(tgt_latent)
        loss = criterion(pred_values, y_batch)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            
            # 每10个epoch打印一次预测值和真实值的统计信息
            with torch.no_grad():
                print(f"Pred stats - Mean: {pred_values.mean().item():.6f}, Std: {pred_values.std().item():.6f}")
                print(f"True stats - Mean: {y_batch.mean().item():.6f}, Std: {y_batch.std().item():.6f}")
                
                # 打印第一个patch的第一个变量的时间序列
                if epoch == 0 or epoch == num_epochs - 1:
                    print("\nFirst patch, first variable comparison:")
                    print("Pred:", pred_values[0, 0, :, 0].cpu().numpy())
                    print("True:", y_batch[0, 0, :, 0].cpu().numpy())
    
    return focus_head

# ========= Step 0: 准备 DataLoader =========
print("[Step 0] Preparing DataLoaders (for downstream)...")
train_loader = create_patch_loader(PATCH_FILE_TRAIN, BATCH_SIZE, shuffle=True)
val_loader   = create_patch_loader(PATCH_FILE_VAL,   BATCH_SIZE, shuffle=False)
test_loader  = create_patch_loader(PATCH_FILE_TEST,  BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}\n")

# ========= Step 1: 加载已训练好的 JEPA 模型（Encoder + Predictor） =========
print("[Step 1] Loading pretrained JEPA model...")
checkpoint = torch.load("model/jepa_best.pt", map_location=DEVICE)

# 实例化 Encoder 和 Predictor，并载入权重
encoder = MyTimeSeriesEncoder(
    patch_length=PATCH_LENGTH,
    num_vars=NUM_VARS,
    latent_dim=LATENT_DIM,
    num_layers=2,
    num_attention_heads=2,
).to(DEVICE)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

predictor = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=None,
    prediction_length=None,
    num_layers=4,
    num_heads=4
).to(DEVICE)
predictor.load_state_dict(checkpoint["predictor_state_dict"])
predictor.eval()
for param in predictor.parameters():
    param.requires_grad = False

print(f"Loaded JEPA Encoder and Predictor (latent_dim={LATENT_DIM}).\n")

# ========= Step 2: 定义 Focus Head =========
# Focus Head 用于将 latent 表示"还原"到真实的数值 patch
class FocusHead(nn.Module):
    def __init__(self, latent_dim, patch_length, num_vars, hidden_dim=512):  # 增加hidden_dim到512
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.GELU()  # 使用GELU激活函数
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 增加一层
        self.fc3 = nn.Linear(hidden_dim, patch_length * num_vars)
        
    def forward(self, latent):
        B, N, D = latent.shape
        flat = latent.view(B * N, D)
        h = self.act(self.fc1(flat))
        h = self.act(self.fc2(h))  # 第二层
        out = self.fc3(h)
        return out.view(B, N, PATCH_LENGTH, NUM_VARS)

focus_head = FocusHead(LATENT_DIM, PATCH_LENGTH, NUM_VARS, hidden_dim=512).to(DEVICE)

# 先进行过拟合测试
focus_head = overfit_test(focus_head, encoder, train_loader)

# 如果过拟合测试成功，继续正常训练
optimizer_fh = torch.optim.AdamW(  # 使用AdamW优化器
    focus_head.parameters(), 
    lr=LEARNING_RATE_FH,
    weight_decay=WEIGHT_DECAY
)

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_fh, 
    mode="min", 
    factor=0.5, 
    patience=5,  # 增加patience
    verbose=True
)

criterion = nn.MSELoss()

print("[Step 2] Focus Head initialized with improved architecture.\n")

# ========= Step 3: 训练 Focus Head =========
print("[Step 3] Training Focus Head on true latent → real-value mapping...")
best_val_loss = float("inf")
patience = 0
max_patience = 10  # 增加早停的耐心值

for epoch in range(1, EPOCHS_FOCUS + 1):
    focus_head.train()
    running_loss = 0.0
    samples = 0

    for x_batch, y_batch in train_loader:
        y_batch = y_batch.to(DEVICE)

        with torch.no_grad():
            tgt_latent = encoder(y_batch)

        pred_values = focus_head(tgt_latent)
        loss = criterion(pred_values, y_batch)
        
        optimizer_fh.zero_grad()
        loss.backward()
        optimizer_fh.step()

        B, N_tgt, _, _ = y_batch.shape
        running_loss += loss.item() * (B * N_tgt)
        samples += (B * N_tgt)

    avg_train_loss = running_loss / samples

    # 验证集上验证
    focus_head.eval()
    running_val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_batch = y_batch.to(DEVICE)
            tgt_latent = encoder(y_batch)
            pred_values = focus_head(tgt_latent)
            loss = criterion(pred_values, y_batch)

            B, N_tgt, _, _ = y_batch.shape
            running_val_loss += loss.item() * (B * N_tgt)
            val_samples += (B * N_tgt)

    avg_val_loss = running_val_loss / val_samples
    
    # 更新学习率调度器
    scheduler.step(avg_val_loss)

    print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 改进的 Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience = 0
        torch.save(focus_head.state_dict(), "model/focus_head_best.pt")
    else:
        patience += 1
        if patience >= max_patience:
            print(f"Early stopping Focus Head at epoch {epoch}")
            break

print("\n[Focus Head training completed] Best Val Loss: {:.6f}\n".format(best_val_loss))

# ========= Step 4: 下游任务评估 （修正版） =========
print("[Step 4] Evaluating JEPA + Focus Head on downstream forecasting...")

# 载入最佳 Focus Head
focus_head.load_state_dict(torch.load("model/focus_head_best.pt", map_location=DEVICE))
focus_head.eval()

# 准备 MSE 和 MAE：使用 sum 模式先累计所有元素（patch 内每个值）的误差
mse_sum_fn = nn.MSELoss(reduction="sum")
mae_sum_fn = nn.L1Loss(reduction="sum")

total_mse = 0.0
total_mae = 0.0
total_elements = 0  # 用来统计"元素"总数： = B * N_tgt * PATCH_LENGTH * NUM_VARS

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)   # (B, N_ctx, PATCH_LENGTH, NUM_VARS)
        y_batch = y_batch.to(DEVICE)   # (B, N_tgt, PATCH_LENGTH, NUM_VARS)

        # 1) 编码上下文 patch
        ctx_latent = encoder(x_batch)  # (B, N_ctx, latent_dim)

        # 2) Teacher-forcing 下预测未来 latent
        tgt_latent = encoder(y_batch)  # (B, N_tgt, latent_dim)
        pred_latent, _ = predictor(ctx_latent, tgt_latent)  # (B, N_tgt, latent_dim)

        # 3) Focus Head 将 pred_latent → 预测值
        pred_values = focus_head(pred_latent)  # (B, N_tgt, PATCH_LENGTH, NUM_VARS)

        # 4) 计算 MSE 和 MAE：sum 模式先累计所有元素
        mse_batch = mse_sum_fn(pred_values, y_batch).item()
        mae_batch = mae_sum_fn(pred_values, y_batch).item()

        B, N_tgt, _, _ = y_batch.shape
        elements_in_batch = B * N_tgt * PATCH_LENGTH * NUM_VARS

        total_mse += mse_batch
        total_mae += mae_batch
        total_elements += elements_in_batch

# 最终的 avg MSE/MAE = 总误差 / 元素总数
avg_test_mse = total_mse / total_elements
avg_test_mae = total_mae / total_elements

print(f"\n[Test Set] Downstream Forecasting — per-element MSE: {avg_test_mse:.6f}, per-element MAE: {avg_test_mae:.6f}")
