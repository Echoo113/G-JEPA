import sys
import os
import copy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_loader import create_patch_loader
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========= 全局设置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE               = 32
LATENT_DIM               = 1024  # 更新为1024以匹配JEPA
PATCH_LENGTH             = 16
NUM_VARS                 = 137
PREDICTION_LENGTH        = 5     # 更新为5以匹配数据集

# 用于训练 ForecastHead 的超参数
HEAD_EPOCHS              = 50
HEAD_LR                  = 1e-5
WEIGHT_DECAY_HEAD        = 1e-6
EARLY_STOPPING_PATIENCE  = 15
EARLY_STOPPING_DELTA     = 1e-6

# 数据路径 & 模型路径
PATCH_FILE_TRAIN         = "data/SOLAR/patches/solar_train.npz"
PATCH_FILE_VAL           = "data/SOLAR/patches/solar_val.npz"
PATCH_FILE_TEST          = "data/SOLAR/patches/solar_test.npz"
JEPA_CHECKPOINT_PATH     = "model/jepa_best.pt"
FORECAST_HEAD_PATH       = "model/forecast_head_best.pt"

# ========= ForecastHead 定义（改进版：预测变量均值） =========
class ForecastHead(nn.Module):
    """
    将 latent 向量映射到变量均值空间。
    输入 shape = (batch_size, num_patches, LATENT_DIM)
    输出 shape = (batch_size, num_patches, NUM_VARS)
    """
    def __init__(self, latent_dim: int, num_vars: int):
        super().__init__()
        self.num_vars = num_vars

        # 改进的网络结构：添加 LayerNorm 和 Skip Connection
        self.norm1 = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, latent_dim * 2)
        self.norm2 = nn.LayerNorm(latent_dim * 2)
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_vars)  # 直接输出变量均值
        self.act = nn.GELU()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, LATENT_DIM)
        返回: (B, N, NUM_VARS)
        """
        b, n, ld = latent.shape
        x = latent.view(b * n, ld)  # (B*N, LATENT_DIM)
        
        # 添加 skip connection 和 layer norm
        identity = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.norm3(x)
        x = x + identity  # Skip connection
        x = self.fc3(x)
        
        return x.view(b, n, self.num_vars)

# ========= 工具函数 =========
def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算变量均值空间上的 MSE 和 MAE。
    pred, target 形状均为 (B, N, NUM_VARS)
    返回: {'mse': mse, 'mae': mae}
    """
    mse = nn.MSELoss(reduction="mean")(pred, target)
    mae = nn.L1Loss(reduction="mean")(pred, target)
    return {'mse': mse, 'mae': mae}

def compute_metrics_real(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算变量均值空间上的 MSE 和 MAE 误差。
    pred, target 形状均为 (B, N, NUM_VARS)
    """
    mse = nn.MSELoss(reduction="mean")(pred, target)
    mae = nn.L1Loss(reduction="mean")(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

def load_pretrained_jepa(checkpoint_path: str):
    """
    加载预训练的 JEPA 模型（包含 online encoder 和 predictor）。
    """
    # 1. 加载 online encoder
    encoder_online = MyTimeSeriesEncoder(
        patch_length=PATCH_LENGTH,
        num_vars=NUM_VARS,
        latent_dim=LATENT_DIM,
        time_layers=2,
        patch_layers=3,
        num_attention_heads=16,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.2
    ).to(DEVICE)

    # 2. 加载 EMA encoder
    encoder_ema = MyTimeSeriesEncoder(
        patch_length=PATCH_LENGTH,
        num_vars=NUM_VARS,
        latent_dim=LATENT_DIM,
        time_layers=2,
        patch_layers=3,
        num_attention_heads=16,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.2
    ).to(DEVICE)

    # 3. 加载 predictor
    predictor = JEPPredictor(
        latent_dim=LATENT_DIM,
        num_layers=3,
        num_heads=16,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.2,
        prediction_length=PREDICTION_LENGTH
    ).to(DEVICE)

    # 4. 加载权重
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder_online.load_state_dict(ckpt["encoder_online_state_dict"])
    encoder_ema.load_state_dict(ckpt["encoder_target_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    
    # 5. 设置为评估模式并冻结参数
    encoder_online.eval()
    encoder_ema.eval()
    predictor.eval()
    for p in encoder_online.parameters():
        p.requires_grad = False
    for p in encoder_ema.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    return encoder_online, encoder_ema, predictor

def train_and_save_forecast_head():
    """
    训练 ForecastHead，使用 predictor 生成的 latent 来预测变量均值。
    """
    # Step 1: 准备 DataLoader
    print("[Step 1] Preparing DataLoaders for head training...")
    train_loader = create_patch_loader(PATCH_FILE_TRAIN, BATCH_SIZE, shuffle=True)
    val_loader   = create_patch_loader(PATCH_FILE_VAL,   BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Step 2: 加载预训练的 JEPA 模型
    print("\n[Step 2] Loading pretrained JEPA models...")
    encoder_online, encoder_ema, predictor = load_pretrained_jepa(JEPA_CHECKPOINT_PATH)
    print("JEPA models loaded and frozen.")

    # Step 3: 初始化 ForecastHead
    print("\n[Step 3] Initializing ForecastHead...")
    head = ForecastHead(
        latent_dim=LATENT_DIM,
        num_vars=NUM_VARS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY_HEAD
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    best_val_mse = float('inf')
    patience_counter = 0
    best_state = None

    print("\n[Step 4] Training ForecastHead...")
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        running_train_mse = 0.0
        running_train_mae = 0.0
        train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # 1. 使用 online encoder 编码输入
            with torch.no_grad():
                ctx_latent = encoder_online(x_batch)
                tgt_latent = encoder_ema(y_batch)
                pred_latent, _ = predictor(ctx_latent, tgt_latent)

            # 2. 使用 ForecastHead 预测变量均值
            pred_mean = head(pred_latent)  # (B, N, NUM_VARS)
            target_mean = y_batch.mean(dim=2)  # (B, N, NUM_VARS)

            # 3. 计算损失并更新
            metrics = compute_loss(pred_mean, target_mean)
            loss = metrics['mse']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_mse += metrics['mse'].item()
            running_train_mae += metrics['mae'].item()
            train_batches += 1

        avg_train_mse = running_train_mse / train_batches
        avg_train_mae = running_train_mae / train_batches

        # 验证阶段
        head.eval()
        running_val_mse = 0.0
        running_val_mae = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # 1. 使用 online encoder 和 EMA encoder 生成 latent
                ctx_latent = encoder_online(x_batch)
                tgt_latent = encoder_ema(y_batch)
                pred_latent, _ = predictor(ctx_latent, tgt_latent)

                # 2. 使用 ForecastHead 预测变量均值
                pred_mean = head(pred_latent)
                target_mean = y_batch.mean(dim=2)

                metrics = compute_loss(pred_mean, target_mean)
                running_val_mse += metrics['mse'].item()
                running_val_mae += metrics['mae'].item()
                val_batches += 1

        avg_val_mse = running_val_mse / val_batches
        avg_val_mae = running_val_mae / val_batches

        # 基于验证集 MSE 进行早停判断
        if avg_val_mse < best_val_mse - EARLY_STOPPING_DELTA:
            best_val_mse = avg_val_mse
            patience_counter = 0
            best_state = {
                'head_state_dict': head.state_dict(),
                'epoch': epoch,
                'train_mse': avg_train_mse,
                'train_mae': avg_train_mae,
                'val_mse': avg_val_mse,
                'val_mae': avg_val_mae
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        scheduler.step(avg_val_mse)

        print(f"[Epoch {epoch:02d}] "
              f"Train MSE: {avg_train_mse:.6f} | "
              f"Train MAE: {avg_train_mae:.6f} | "
              f"Val MSE: {avg_val_mse:.6f} | "
              f"Val MAE: {avg_val_mae:.6f}")

    print("\n[Head Training Completed]")

    # Step 5: 保存最优 ForecastHead
    if best_state is not None:
        os.makedirs(os.path.dirname(FORECAST_HEAD_PATH), exist_ok=True)
        torch.save({
            'forecasthead_state_dict': best_state['head_state_dict'],
            'num_vars': NUM_VARS,
            'latent_dim': LATENT_DIM,
            'epoch': best_state['epoch'],
            'train_mse': best_state['train_mse'],
            'train_mae': best_state['train_mae'],
            'val_mse': best_state['val_mse'],
            'val_mae': best_state['val_mae']
        }, FORECAST_HEAD_PATH)
        print(f"Best ForecastHead saved at '{FORECAST_HEAD_PATH}' (epoch {best_state['epoch']})")
        print(f"  Train MSE: {best_state['train_mse']:.6f}")
        print(f"  Train MAE: {best_state['train_mae']:.6f}")
        print(f"  Val MSE: {best_state['val_mse']:.6f}")
        print(f"  Val MAE: {best_state['val_mae']:.6f}")

def forecast_with_head():
    """
    在测试集上运行推理：使用已训练好的 JEPA 和 ForecastHead。
    """
    print("[Step 1] Preparing DataLoader for test set…")
    test_loader = create_patch_loader(PATCH_FILE_TEST, BATCH_SIZE, shuffle=False)
    print(f"Test batches: {len(test_loader)}")

    print("\n[Step 2] Loading pretrained JEPA models…")
    encoder_online, encoder_ema, predictor = load_pretrained_jepa(JEPA_CHECKPOINT_PATH)
    print("JEPA models loaded.")

    print("\n[Step 3] Loading ForecastHead…")
    ckpt = torch.load(FORECAST_HEAD_PATH, map_location=DEVICE)
    forecast_head = ForecastHead(
        latent_dim = ckpt['latent_dim'],
        num_vars = ckpt['num_vars']
    ).to(DEVICE)
    forecast_head.load_state_dict(ckpt['forecasthead_state_dict'])
    forecast_head.eval()
    print("ForecastHead loaded.")

    print("\n[Step 4] Running inference on test data…")
    running_test_mse = 0.0
    running_test_mae = 0.0
    test_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # 1. 使用 online encoder 和 EMA encoder 生成 latent
            ctx_latent = encoder_online(x_batch)
            tgt_latent = encoder_ema(y_batch)
            pred_latent, _ = predictor(ctx_latent, tgt_latent)

            # 2. 使用 ForecastHead 预测变量均值
            pred_mean = forecast_head(pred_latent)
            target_mean = y_batch.mean(dim=2)

            # 3. 计算误差
            metrics = compute_metrics_real(pred_mean, target_mean)
            running_test_mse += metrics['mse']
            running_test_mae += metrics['mae']
            test_batches += 1

    avg_test_mse = running_test_mse / test_batches
    avg_test_mae = running_test_mae / test_batches

    print(f"\n[Test Results]")
    print(f"  MSE: {avg_test_mse:.6f}")
    print(f"  MAE: {avg_test_mae:.6f}")

if __name__ == "__main__":
    train_and_save_forecast_head()
    forecast_with_head()
