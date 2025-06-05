import sys
import os

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

BATCH_SIZE               = 64
LATENT_DIM               = 256
PATCH_LENGTH             = 16
NUM_VARS                 = 137

# 用于训练 ForecastHead 的超参数（极简网络、超低学习率）
HEAD_EPOCHS              = 50
HEAD_LR                  = 1e-5
WEIGHT_DECAY_HEAD        = 1e-6
EARLY_STOPPING_PATIENCE  = 15
EARLY_STOPPING_DELTA     = 1e-6

# 数据路径 & 模型路径
PATCH_FILE_TRAIN         = "data/SOLAR/patches/solar_train.npz"
PATCH_FILE_VAL           = "data/SOLAR/patches/solar_val.npz"
PATCH_FILE_TEST          = "data/SOLAR/patches/solar_test.npz"

# 之前训练好的 JEPA 模型检查点（包含 encoder 与 predictor 权重）
JEPA_CHECKPOINT_PATH     = "model/jepa_best.pt"
# 训练完成后保存 ForecastHead 的文件
FORECAST_HEAD_PATH       = "model/forecast_head_best.pt"


# ========= ForecastHead 定义（双隐层，确保有足够容量） =========
class ForecastHead(nn.Module):
    """
    将 latent 向量映射回原始 patch 空间 (PATCH_LENGTH × NUM_VARS)。
    输入 shape = (batch_size, num_patches, LATENT_DIM)
    输出 shape = (batch_size, num_patches, PATCH_LENGTH, NUM_VARS)
    """
    def __init__(self, latent_dim: int, patch_length: int, num_vars: int):
        super().__init__()
        self.patch_length = patch_length
        self.num_vars = num_vars

        # 双隐层网络
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, patch_length * num_vars)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, LATENT_DIM)
        返回: (B, N, PATCH_LENGTH, NUM_VARS)
        """
        b, n, ld = latent.shape
        x = latent.view(b * n, ld)                # (B*N, LATENT_DIM)
        out = self.net(x)                          # (B*N, PATCH_LENGTH*NUM_VARS)
        out = out.view(b, n, self.patch_length, self.num_vars)
        return out


# ========= 工具函数 =========
def compute_loss(recon: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算 patch 空间上的 MSE 和 MAE（mean）。
    recon, target 形状均为 (B, N, PATCH_LENGTH, NUM_VARS)
    返回: {'mse': mse, 'mae': mae}  # 返回 tensor，不调用 .item()
    """
    mse = nn.MSELoss(reduction="mean")(recon, target)
    mae = nn.L1Loss(reduction="mean")(recon, target)
    return {'mse': mse, 'mae': mae}


def compute_metrics_real(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    计算真实空间上的 MSE 和 MAE 误差（mean），返回字典。
    pred, target 形状均为 (B, N, PATCH_LENGTH, NUM_VARS)
    """
    mse = nn.MSELoss(reduction="mean")(pred, target)
    mae = nn.L1Loss(reduction="mean")(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}


def load_pretrained_encoder(checkpoint_path: str) -> MyTimeSeriesEncoder:
    """
    重建并加载已训练好的 Encoder（只需编码部分，不训练该部分）。
    """
    encoder = MyTimeSeriesEncoder(
        patch_length=PATCH_LENGTH,
        num_vars=NUM_VARS,
        latent_dim=LATENT_DIM,
        time_layers=2,
        patch_layers=4,
        num_attention_heads=8,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.1
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


def load_pretrained_jepa(checkpoint_path: str):
    """
    重建并加载已训练好的 Encoder 和 Predictor，用于推理阶段。
    """
    encoder = MyTimeSeriesEncoder(
        patch_length=PATCH_LENGTH,
        num_vars=NUM_VARS,
        latent_dim=LATENT_DIM,
        time_layers=2,
        patch_layers=4,
        num_attention_heads=8,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.1
    ).to(DEVICE)

    predictor = JEPPredictor(
        latent_dim=LATENT_DIM,
        num_layers=4,
        num_heads=4,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.1,
        prediction_length=None
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    encoder.eval()
    predictor.eval()
    return encoder, predictor


def train_and_save_forecast_head():
    """
    训练 ForecastHead，将 latent vector 还原为原始 patch 空间并保存到 FORECAST_HEAD_PATH。
    仅包含一个线性层，学习率极低，保证 loss 充分下降。
    """
    # Step 1: 准备 DataLoader
    print("[Step 1] Preparing DataLoaders for head training...")
    train_loader = create_patch_loader(PATCH_FILE_TRAIN, BATCH_SIZE, shuffle=True)
    val_loader   = create_patch_loader(PATCH_FILE_VAL,   BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Step 2: 加载并冻结 Encoder
    print("\n[Step 2] Loading pretrained encoder and freezing it...")
    encoder = load_pretrained_encoder(JEPA_CHECKPOINT_PATH)
    print("Encoder loaded and frozen. (只用于提取 latent 表示)")

    # Step 3: 初始化 ForecastHead
    print("\n[Step 3] Initializing ForecastHead...")
    head = ForecastHead(
        latent_dim=LATENT_DIM,
        patch_length=PATCH_LENGTH,
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

    best_total_loss = float('inf')
    patience_counter = 0
    best_state = None

    print("\n[Step 4] Training ForecastHead...")
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        running_train_mse = 0.0
        running_train_mae = 0.0
        train_batches = 0

        for _, y_batch in train_loader:
            # y_batch: (B, N, PATCH_LENGTH, NUM_VARS)
            y_batch = y_batch.to(DEVICE)

            # 用 y_batch 提取 latent（不计算 grad）
            with torch.no_grad():
                latent_y = encoder(y_batch)  # (B, N, LATENT_DIM)

            pred_patch = head(latent_y)      # (B, N, PATCH_LENGTH, NUM_VARS)

            metrics = compute_loss(pred_patch, y_batch)
            loss = metrics['mse']  # 使用 MSE 作为训练损失
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
            for _, y_batch in val_loader:
                y_batch = y_batch.to(DEVICE)
                latent_y = encoder(y_batch)       # (B, N, LATENT_DIM)
                pred_patch = head(latent_y)        # (B, N, PATCH_LENGTH, NUM_VARS)

                metrics = compute_loss(pred_patch, y_batch)
                running_val_mse += metrics['mse'].item()
                running_val_mae += metrics['mae'].item()
                val_batches += 1

        avg_val_mse = running_val_mse / val_batches
        avg_val_mae = running_val_mae / val_batches
        
        # 计算总损失（训练 + 验证）
        total_loss = avg_train_mse + avg_val_mse

        # 基于总损失进行早停判断
        if total_loss < best_total_loss - EARLY_STOPPING_DELTA:
            best_total_loss = total_loss
            patience_counter = 0
            best_state = {
                'head_state_dict': head.state_dict(),
                'epoch': epoch,
                'train_mse': avg_train_mse,
                'train_mae': avg_train_mae,
                'val_mse': avg_val_mse,
                'val_mae': avg_val_mae,
                'total_loss': total_loss
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        scheduler.step(total_loss)  # 使用总损失来调整学习率

        print(f"[Epoch {epoch:02d}] "
              f"Train MSE: {avg_train_mse:.6f} | "
              f"Train MAE: {avg_train_mae:.6f} | "
              f"Val MSE: {avg_val_mse:.6f} | "
              f"Val MAE: {avg_val_mae:.6f} | "
              f"Total Loss: {total_loss:.6f}")

    print("\n[Head Training Completed]")

    # Step 5: 保存最优 ForecastHead
    if best_state is not None:
        os.makedirs(os.path.dirname(FORECAST_HEAD_PATH), exist_ok=True)
        torch.save({
            'forecasthead_state_dict': best_state['head_state_dict'],
            'patch_length': PATCH_LENGTH,
            'num_vars': NUM_VARS,
            'latent_dim': LATENT_DIM,
            'epoch': best_state['epoch'],
            'train_mse': best_state['train_mse'],
            'train_mae': best_state['train_mae'],
            'val_mse': best_state['val_mse'],
            'val_mae': best_state['val_mae'],
            'total_loss': best_state['total_loss']
        }, FORECAST_HEAD_PATH)
        print(f"Best ForecastHead saved at '{FORECAST_HEAD_PATH}' (epoch {best_state['epoch']})")
        print(f"  Train MSE: {best_state['train_mse']:.6f}")
        print(f"  Train MAE: {best_state['train_mae']:.6f}")
        print(f"  Val MSE: {best_state['val_mse']:.6f}")
        print(f"  Val MAE: {best_state['val_mae']:.6f}")
        print(f"  Total Loss: {best_state['total_loss']:.6f}")


def forecast_with_head():
    """
    在测试集上运行推理：使用已训练好的 JEPA（Encoder+Predictor）和 ForecastHead
    将 Predictor 输出的 latent 恢复成真实 patch 并计算误差。
    """
    print("[Step 1] Preparing DataLoader for test set…")
    test_loader = create_patch_loader(PATCH_FILE_TEST, BATCH_SIZE, shuffle=False)
    print(f"Test batches: {len(test_loader)}")

    print("\n[Step 2] Loading pretrained JEPA (Encoder+Predictor)…")
    encoder, predictor = load_pretrained_jepa(JEPA_CHECKPOINT_PATH)
    print("JEPA models loaded.")

    print("\n[Step 3] Loading ForecastHead…")
    ckpt = torch.load(FORECAST_HEAD_PATH, map_location=DEVICE)
    forecast_head = ForecastHead(
        latent_dim = ckpt['latent_dim'],
        patch_length = ckpt['patch_length'],
        num_vars = ckpt['num_vars']
    ).to(DEVICE)
    forecast_head.load_state_dict(ckpt['forecasthead_state_dict'])
    forecast_head.eval()
    print("ForecastHead loaded.")

    print("\n[Step 4] Running inference on test data…")
    running_test_mse = 0.0
    running_test_mae = 0.0
    running_test_total = 0.0
    test_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)  # (B, N_context, PATCH_LENGTH, NUM_VARS)
            y_batch = y_batch.to(DEVICE)  # (B, N_target,  PATCH_LENGTH, NUM_VARS)

            # 1) 编码 x_batch, y_batch 到 latent
            ctx_latent = encoder(x_batch)   # (B, N_context, LATENT_DIM)
            tgt_latent = encoder(y_batch)   # (B, N_target,  LATENT_DIM)

            # 2) 使用 Predictor 得到 pred_latent
            pred_latent, _ = predictor(ctx_latent, tgt_latent)
            #    pred_latent: (B, N_target, LATENT_DIM)

            # 3) 用 ForecastHead 将 pred_latent 恢复到真实 patch 空间
            pred_real = forecast_head(pred_latent)
            #    pred_real: (B, N_target, PATCH_LENGTH, NUM_VARS)

            # 4) 计算与 y_batch 之间的 MSE、MAE
            metrics = compute_metrics_real(pred_real, y_batch)
            running_test_mse += metrics['mse']
            running_test_mae += metrics['mae']
            running_test_total += metrics['mse'] * y_batch.numel()  # 累加总损失
            test_batches += 1

    avg_test_mse = running_test_mse / test_batches
    avg_test_mae = running_test_mae / test_batches

    print(f"\n[Test Results]")
    print(f"  MSE: {avg_test_mse:.6f}")
    print(f"  MAE: {avg_test_mae:.6f}")



if __name__ == "__main__":
    # 先训练 ForecastHead，然后直接进行推理
    train_and_save_forecast_head()
    forecast_with_head()
