#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
forecast.py

下游任务：使用已训练好的 JEPA 模型（Encoder + Predictor）在 latent 空间进行未来序列预测，
并通过训练一个 FocusHead 将预测得到的 latent 恢复回真实 patch 值。最后在测试集上评估下游
预测性能（原始值空间的 MSE/MAE）。

新增要求：仅当训练集 Loss 和验证集 Loss 都足够小（即同时刷新最佳值）时，才保存 FocusHead 模型。
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 若项目根目录不在 sys.path，则将其添加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from patch_loader import create_patch_loader
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========== 全局设置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE               = 64
LATENT_DIM               = 256
EPOCHS_FOCUS             = 40
LEARNING_RATE_FH         = 5e-4
WEIGHT_DECAY             = 1e-6
EARLY_STOPPING_PATIENCE  = 10

PATCH_FILE_TRAIN         = "data/SOLAR/patches/solar_train.npz"
PATCH_FILE_VAL           = "data/SOLAR/patches/solar_val.npz"
PATCH_FILE_TEST          = "data/SOLAR/patches/solar_test.npz"

PATCH_LENGTH             = 16
NUM_VARS                 = 137

# ========== FocusHead 定义 ==========
class FocusHead(nn.Module):
    """
    将 JEPA 预测得到的 latent 映射回原始 patch 值。
    输入：pred_latent (B, N_tgt, latent_dim)
    输出：patch_values  (B, N_tgt, PATCH_LENGTH, NUM_VARS)
    """
    def __init__(self, latent_dim, patch_length, num_vars, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(hidden_dim, patch_length * num_vars)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, N_tgt, D = latent.shape
        flat = latent.view(B * N_tgt, D)

        h = self.fc1(flat)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout1(h)

        h = self.fc2(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.dropout2(h)

        out = self.fc3(h)
        out = out.view(B, N_tgt, PATCH_LENGTH, NUM_VARS)
        return out

def main():
    # -------- Step 1: 准备 DataLoader --------
    print("[Step 1] Preparing DataLoaders...")
    train_loader = create_patch_loader(PATCH_FILE_TRAIN, BATCH_SIZE, shuffle=True)
    val_loader   = create_patch_loader(PATCH_FILE_VAL,   BATCH_SIZE, shuffle=False)
    test_loader  = create_patch_loader(PATCH_FILE_TEST,  BATCH_SIZE, shuffle=False)
    print(f"  → Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}\n")

    # -------- Step 2: 加载预训练 JEPA 模型（Encoder + Predictor） --------
    print("[Step 2] Loading pretrained JEPA model...")
    checkpoint = torch.load("model/jepa_best.pt", map_location=DEVICE, weights_only=True)

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
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    predictor = JEPPredictor(
        latent_dim=LATENT_DIM,
        num_heads=4,
        num_layers=4,
        ffn_dim=LATENT_DIM * 4,
        dropout=0.1,
        prediction_length=None  # 训练时只用 teacher forcing
    ).to(DEVICE)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad = False

    print(f"  → Loaded encoder/predictor (latent_dim={LATENT_DIM}).\n")

    # -------- Step 3: 定义 FocusHead 并做 Overfit Test --------
    print("[Step 3] Defining FocusHead and running overfit test...")
    focus_head = FocusHead(LATENT_DIM, PATCH_LENGTH, NUM_VARS, hidden_dim=512).to(DEVICE)

    focus_head.train()
    print("  → Starting overfit test (single batch)...")
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    with torch.no_grad():
        print("    * Encoding x_batch → ctx_latent")
        ctx_latent = encoder(x_batch)    # (B, N_ctx, D)
        print("    * Encoding y_batch → tgt_latent")
        tgt_latent = encoder(y_batch)    # (B, N_tgt, D)
        print("    * Predictor forward → pred_latent")
        pred_latent, _ = predictor(ctx_latent, tgt_latent)  # (B, N_tgt, D)

    print(f"    - pred_latent mean={pred_latent.mean().item():.6f}, std={pred_latent.std().item():.6f}")
    print(f"    - y_batch mean={y_batch.mean().item():.6f}, std={y_batch.std().item():.6f}")

    overfit_optimizer = torch.optim.Adam(focus_head.parameters(), lr=1e-4)
    overfit_criterion = nn.MSELoss()
    for epoch in range(1, 31):
        overfit_optimizer.zero_grad()
        pred_values = focus_head(pred_latent)
        loss_of = overfit_criterion(pred_values, y_batch)
        loss_of.backward()
        overfit_optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                print(f"    [Overfit Epoch {epoch:02d}] Loss: {loss_of.item():.6f} | "
                      f"pred_values mean={pred_values.mean().item():.6f}, std={pred_values.std().item():.6f}")
    print("  → Overfit test completed.\n")

    # -------- Step 4: 训练 FocusHead（downstream） --------
    print("[Step 4] Training FocusHead on predicted latent → real patch mapping...")
    focus_head.train()
    optimizer_fh = torch.optim.AdamW(
        focus_head.parameters(),
        lr=LEARNING_RATE_FH,
        weight_decay=WEIGHT_DECAY
    )
    scheduler_fh = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_fh,
        mode="min",
        factor=0.5,
        patience=5
    )
    criterion_fh = nn.SmoothL1Loss()  # 或者 nn.MSELoss()

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS_FOCUS + 1):
        print(f"\n  >>> [Epoch {epoch:02d}] Starting training loop...")
        focus_head.train()
        running_train_loss = 0.0
        train_samples = 0

        # ——— 训练集循环 ———
        print("    * Training on train_loader...")
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader, 1):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            with torch.no_grad():
                ctx_latent = encoder(x_batch)
                tgt_latent = encoder(y_batch)
                pred_latent, _ = predictor(ctx_latent, tgt_latent)

            pred_values = focus_head(pred_latent)
            loss = criterion_fh(pred_values, y_batch)

            optimizer_fh.zero_grad()
            loss.backward()
            optimizer_fh.step()

            B, N_tgt, _, _ = y_batch.shape
            running_train_loss += loss.item() * (B * N_tgt)
            train_samples += (B * N_tgt)

            if batch_idx % 20 == 0:
                print(f"      [Batch {batch_idx}/{len(train_loader)}] partial train loss = {loss.item():.6f}")

        avg_train_loss = running_train_loss / train_samples
        print(f"    → Finished training epoch {epoch}, avg_train_loss = {avg_train_loss:.6f}")

        # ——— 验证集循环 ———
        print("    * Validating on val_loader...")
        focus_head.eval()
        running_val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader, 1):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                ctx_latent = encoder(x_batch)
                tgt_latent = encoder(y_batch)
                pred_latent, _ = predictor(ctx_latent, tgt_latent)
                pred_values = focus_head(pred_latent)
                loss = criterion_fh(pred_values, y_batch)

                B, N_tgt, _, _ = y_batch.shape
                running_val_loss += loss.item() * (B * N_tgt)
                val_samples += (B * N_tgt)

                if batch_idx % 10 == 0:
                    print(f"      [Val Batch {batch_idx}/{len(val_loader)}] partial val loss = {loss.item():.6f}")

        avg_val_loss = running_val_loss / val_samples
        print(f"    → Finished validation epoch {epoch}, avg_val_loss = {avg_val_loss:.6f}")
        print(f"    → Current learning rate: {scheduler_fh.optimizer.param_groups[0]['lr']:.6f}")

        scheduler_fh.step(avg_val_loss)

        # —— 模型保存逻辑 —— 
        if avg_train_loss < best_train_loss and avg_val_loss < best_val_loss:
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(focus_head.state_dict(), "model/focus_head_best.pt")
            print(f"    → Saved FocusHead (train_loss={best_train_loss:.6f}, val_loss={best_val_loss:.6f})")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"  >>> Early stopping triggered at epoch {epoch}")
                break

    print(f"\n[FocusHead training completed] Best Train Loss: {best_train_loss:.6f}, Best Val Loss: {best_val_loss:.6f}\n")

    # -------- Step 5: 下游任务评估（测试集） --------
    print("[Step 5] Evaluating on downstream forecasting (test set)...")
    focus_head.load_state_dict(torch.load("model/focus_head_best.pt", map_location=DEVICE))
    focus_head.eval()

    mse_sum_fn = nn.MSELoss(reduction="sum")
    mae_sum_fn = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_elements = 0  # B * N_tgt * PATCH_LENGTH * NUM_VARS

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader, 1):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            ctx_latent = encoder(x_batch)
            tgt_latent = encoder(y_batch)
            pred_latent, _ = predictor(ctx_latent, tgt_latent)
            pred_values = focus_head(pred_latent)

            mse_batch = mse_sum_fn(pred_values, y_batch).item()
            mae_batch = mae_sum_fn(pred_values, y_batch).item()

            B, N_tgt, _, _ = y_batch.shape
            elements = B * N_tgt * PATCH_LENGTH * NUM_VARS

            total_mse += mse_batch
            total_mae += mae_batch
            total_elements += elements

            if batch_idx % 10 == 0:
                print(f"    [Test Batch {batch_idx}/{len(test_loader)}] "
                      f"partialMSE={mse_batch/elements:.6f}, partialMAE={mae_batch/elements:.6f}")

    avg_test_mse = total_mse / total_elements
    avg_test_mae = total_mae / total_elements
    print(f"\n[Test Set] Downstream Forecasting — per-element MSE: {avg_test_mse:.6f}, MAE: {avg_test_mae:.6f}\n")


if __name__ == "__main__":
    main()
