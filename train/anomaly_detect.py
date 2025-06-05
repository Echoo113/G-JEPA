#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_jepa_anomaly.py

下游任务：使用预训练好的 JEPA Encoder + Predictor 对 MSL 数据集进行窗口级异常预测，
计算窗口级 F1 分数。核心思路：
  1) 冻结预训练好的 Encoder + Predictor，不进行微调；
  2) 对每个 sliding window 的 9 个 patch（shape=(9,20,55)）：
       - Encoder 提取 patch‐level latent
       - Predictor 在 latent 空间做 teacher‐forcing 预测，得到 pred_latent
       - 计算 patch‐level MSE 误差
       - 窗口级分数 = max(patch_errors)
  3) 在 tuning‐train (2648 窗口) 上搜索最优阈值，使得 F1 最大；
  4) 在 tuning‐val (294 窗口) 上验证该阈值；
  5) 在 final‐test (735 窗口) 上评估最终的 Precision/Recall/F1。

用到的预训练模型路径：model/jepa_best.pt  
用到的数据路径：data/MSL/patches/msl_tune_train.npz, msl_tune_val.npz, msl_final_test.npz  
原始 test label 文件：data/MSL/MSL_test_label.npy  
"""


import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import joblib

# —— 下面两个 import 假设你的项目结构里，这两个模块在相应路径下
#    MyTimeSeriesEncoder 定义在 jepa/encoder.py
#    JEPPredictor 定义在 jepa/predictor.py
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor


class LabelLoader:
    """
    LabelLoader：根据原始时间序列的二值标签（shape=(T,)），为每个 sliding window
    生成窗口级真值（任意一步出现异常即标为 1）。
    """
    def __init__(
        self,
        label_file: str,
        total_series_length: int = 73729,
        input_len: int = 100,
        output_len: int = 100,
        window_stride: int = 20,
        patch_len: int = 20,
        patch_stride: int = 10
    ):
        self.label = np.load(label_file)  # shape = (73729,)
        assert len(self.label) == total_series_length, "标签长度与原始时间序列不匹配"

        self.input_len = input_len
        self.output_len = output_len
        self.window_stride = window_stride
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.window_size = input_len + output_len

    def generate_window_labels(self, num_windows: int) -> np.ndarray:
        """
        对每个 sliding window 判断是否含有异常：
        只要窗口内任意一步为 True (1)，就标记该窗口为异常 (1)，否则为正常 (0)。
        """
        labels = []
        for i in range(num_windows):
            start_idx = i * self.window_stride
            end_idx = start_idx + self.window_size
            if end_idx > len(self.label):
                # 超出边界，默认标为正常 0
                labels.append(0)
            else:
                window_label = self.label[start_idx:end_idx]
                labels.append(int(np.any(window_label)))
        return np.array(labels, dtype=int)


def load_pretrained_models(
    encoder_cls,
    predictor_cls,
    model_path: str,
    device: torch.device,
    **encoder_kwargs
):
    """
    从 model_path 加载预训练权重，返回冻结后的 encoder 和 predictor。
    """
    # 载入 checkpoint
    ckpt = torch.load(model_path, map_location=device)
    encoder = encoder_cls(
        patch_length=encoder_kwargs["patch_length"],
        num_vars=encoder_kwargs["num_vars"],
        latent_dim=encoder_kwargs["latent_dim"],
        time_layers=encoder_kwargs["time_layers"],
        patch_layers=encoder_kwargs["patch_layers"],
        num_attention_heads=encoder_kwargs["num_attention_heads"],
        ffn_dim=encoder_kwargs["ffn_dim"],
        dropout=encoder_kwargs["dropout"]
    ).to(device)
    predictor = predictor_cls(
        latent_dim=encoder_kwargs["latent_dim"],
        num_heads=encoder_kwargs["predictor_num_heads"],
        num_layers=encoder_kwargs["predictor_num_layers"],
        ffn_dim=encoder_kwargs["latent_dim"] * 4,
        dropout=encoder_kwargs["predictor_dropout"],
        prediction_length=encoder_kwargs["prediction_length"]
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    encoder.eval()
    predictor.eval()

    # 冻结参数
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    return encoder, predictor


def compute_window_scores(
    x_patches_np: np.ndarray,
    y_patches_np: np.ndarray,
    encoder: MyTimeSeriesEncoder,
    predictor: JEPPredictor,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
    对一个 NPZ 中的所有窗口（N 窗口）计算窗口级分数：
      1) x_patches_np 形状 = (N, P, T_patch, C)  例如 (2648, 9, 20, 55)
      2) y_patches_np 同上
      3) 使用 encoder 一次性生成所有 patch 的 latent → ctx_latents, true_latents 形状 = (N, P, D)
      4) predictor 以 teacher forcing 方式生成 pred_latents 形状 = (N, P, D)
      5) patch_errors = mean((pred_latents - true_latents)^2, dim=2)  → shape=(N,P)
      6) window_scores = np.max(patch_errors, axis=1) → shape=(N,)
    返回：window_scores_np (np.ndarray, shape=(N,)).
    """
    num_windows, num_patches, patch_len, num_vars = x_patches_np.shape
    latent_dim = encoder.latent_dim

    window_scores = np.zeros((num_windows,), dtype=float)

    # 我们可以分批处理 Windows
    idxs = np.arange(num_windows)
    for start in range(0, num_windows, batch_size):
        end = min(start + batch_size, num_windows)
        batch_idxs = idxs[start:end]
        B = len(batch_idxs)

        # 从 numpy 转成 Tensor, 并搬到 device
        x_batch = torch.tensor(
            x_patches_np[batch_idxs], dtype=torch.float32, device=device
        )  # shape = (B, num_patches, patch_len, num_vars)
        y_batch = torch.tensor(
            y_patches_np[batch_idxs], dtype=torch.float32, device=device
        )

        # 1) Encoder: (B, P, T, C) → (B, P, D)
        with torch.no_grad():
            x_latents = encoder(x_batch)   # (B, num_patches, latent_dim)
            y_latents = encoder(y_batch)   # (B, num_patches, latent_dim)

            # 2) Predictor (teacher forcing): 输入 ctx=x_latents, tgt=y_latents，输出 pred_latents=(B,P,D)
            pred_latents, _ = predictor(x_latents, y_latents)

            # 3) 计算 patch‐level MSE
            #    patch_errors_tensor shape = (B, num_patches)
            patch_errors_tensor = torch.mean((pred_latents - y_latents).pow(2), dim=2)

        patch_errors_np = patch_errors_tensor.cpu().numpy()  # 转到 CPU，shape=(B, P)

        # 4) 窗口级分数 = max over P
        window_scores[start:end] = patch_errors_np.max(axis=1)

    return window_scores  # shape=(N,)


def search_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    num_candidates: int = 100
) -> (float, float):
    """
    在给定的 scores 和 labels 上，搜索能使 F1 最大的阈值。
    方法：在 score 的 [min,max] 区间均匀取 num_candidates 个 candidate thresholds，
          计算每个阈值下的 F1，返回 (best_threshold, best_f1)。
    """
    best_t = None
    best_f1 = 0.0

    min_s, max_s = float(scores.min()), float(scores.max())
    # 避免所有 score 完全相同导致 linspace 出错
    if min_s == max_s:
        # 在这种极端情况下，只能对比 score>min_s
        pred = (scores > min_s).astype(int)
        best_f1 = f1_score(labels, pred)
        return min_s, best_f1

    cands = np.linspace(min_s, max_s, num=num_candidates)
    for t in cands:
        pred = (scores > t).astype(int)
        f1 = f1_score(labels, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), best_f1


def evaluate_on_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
):
    """
    给定 scores (shape=(N,))、labels (shape=(N,)) 和 threshold，
    计算 Precision/Recall/F1，并返回一个 dict。
    """
    preds = (scores > threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_metrics(
    x_patches_np: np.ndarray,
    y_patches_np: np.ndarray,
    encoder: MyTimeSeriesEncoder,
    predictor: JEPPredictor,
    device: torch.device,
    batch_size: int = 64
) -> dict:
    """
    计算预测误差指标（MSE, MAE等）
    
    Args:
        x_patches_np: shape = (N, P, T_patch, C)
        y_patches_np: shape = (N, P, T_patch, C)
        encoder: 预训练的编码器
        predictor: 预训练的预测器
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        dict: 包含各项指标
    """
    num_windows, num_patches, patch_len, num_vars = x_patches_np.shape
    all_pred_errors = []
    all_true_values = []
    all_pred_values = []
    
    # 分批处理
    idxs = np.arange(num_windows)
    for start in range(0, num_windows, batch_size):
        end = min(start + batch_size, num_windows)
        batch_idxs = idxs[start:end]
        B = len(batch_idxs)
        
        # 转换为 tensor
        x_batch = torch.tensor(x_patches_np[batch_idxs], dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_patches_np[batch_idxs], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # 获取 latent representations
            x_latents = encoder(x_batch)
            y_latents = encoder(y_batch)
            
            # 预测
            pred_latents, _ = predictor(x_latents, y_latents)
            
            # 计算预测误差（在 latent 空间）
            patch_errors = torch.mean((pred_latents - y_latents).pow(2), dim=2)
            
            # 收集预测值和真实值
            all_pred_values.extend(y_batch.cpu().numpy().reshape(-1, num_vars))
            all_true_values.extend(y_batch.cpu().numpy().reshape(-1, num_vars))
            all_pred_errors.extend(patch_errors.cpu().numpy().reshape(-1))
    
    # 转换为 numpy 数组
    pred_values = np.array(all_pred_values)
    true_values = np.array(all_true_values)
    pred_errors = np.array(all_pred_errors)
    
    # 计算误差指标
    mse = mean_squared_error(true_values, pred_values)
    mae = mean_absolute_error(true_values, pred_values)
    
    return {
        "feature_error": mse,  # 论文中的 FE
        "mae": mae,
        "pred_errors": pred_errors  # 用于异常检测的误差分数
    }


def main():
    # -------------------- 全局设置 --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # JEPA Encoder + Predictor 超参数（需跟预训练时一致）
    encoder_kwargs = {
        "patch_length":       20,      # 每个 patch 时间步数
        "num_vars":           55,      # MSL 55 维
        "latent_dim":         128,
        "time_layers":        2,
        "patch_layers":       3,
        "num_attention_heads":8,
        "ffn_dim":            128 * 4,
        "dropout":            0.2,
        # JEPPredictor 专属
        "predictor_num_heads":   4,
        "predictor_num_layers":  3,
        "predictor_dropout":     0.2,
        "prediction_length":     9     # MSL 每个窗口 9 个 patch
    }
    pretrained_model_path = "model/jepa_best.pt"

    # Patch NPZ 路径
    patch_dir = "data/MSL/patches"
    tune_train_npz = os.path.join(patch_dir, "msl_tune_train.npz")   # 2648 窗口
    tune_val_npz   = os.path.join(patch_dir, "msl_tune_val.npz")     #  294 窗口
    final_test_npz = os.path.join(patch_dir, "msl_final_test.npz")   #  735 窗口

    # 原始 test label 文件
    label_file = "data/MSL/MSL_test_label.npy"
    total_series_length = 73729
    input_len = 100
    output_len = 100
    window_stride = 20
    patch_len = 20
    patch_stride = 10

    # 计算窗口数量（需要与 AnomalyPatchExtractor 一致）
    # msl_tune_train 中有 2648 个窗口；msl_tune_val 中有 294；msl_final_test 中有 735。
    # 这些数字也可以通过 np.load(...)[ "x_patches" ].shape[0] 自动获取。

    # -------------------- Step 1: 冻结并加载预训练模型 --------------------
    print("加载并冻结预训练 JEPA Encoder + Predictor...")
    encoder, predictor = load_pretrained_models(
        encoder_cls=MyTimeSeriesEncoder,
        predictor_cls=JEPPredictor,
        model_path=pretrained_model_path,
        device=device,
        **encoder_kwargs
    )
    print("预训练模型加载完成，参数已冻结。\n")

    # -------------------- Step 2: 读取 Tuning‐Train 数据，计算窗口得分 --------------------
    print("Step 2: 计算 Tuning‐Train (msl_tune_train.npz) 窗口级分数...")
    tune_train_data = np.load(tune_train_npz)
    x_tune_train = tune_train_data["x_patches"]  # shape = (2648, 9, 20, 55)
    y_tune_train = tune_train_data["y_patches"]

    num_windows_train = x_tune_train.shape[0]
    patches_per_window = x_tune_train.shape[1]  # =9

    # 计算 window_scores_train
    window_scores_train = compute_window_scores(
        x_patches_np=x_tune_train,
        y_patches_np=y_tune_train,
        encoder=encoder,
        predictor=predictor,
        device=device,
        batch_size=64
    )  # shape = (2648,)
    print(f"  已计算 {num_windows_train} 个窗口的得分。\n")

    # 同时生成 window_labels_train
    label_loader = LabelLoader(
        label_file=label_file,
        total_series_length=total_series_length,
        input_len=input_len,
        output_len=output_len,
        window_stride=window_stride,
        patch_len=patch_len,
        patch_stride=patch_stride
    )
    window_labels_train = label_loader.generate_window_labels(num_windows_train)  # shape=(2648,)

    # -------------------- Step 3: 读取 Tuning‐Val 数据，计算窗口得分 --------------------
    print("Step 3: 计算 Tuning‐Val (msl_tune_val.npz) 窗口级分数...")
    tune_val_data = np.load(tune_val_npz)
    x_tune_val = tune_val_data["x_patches"]  # shape=(294, 9, 20, 55)
    y_tune_val = tune_val_data["y_patches"]
    num_windows_val = x_tune_val.shape[0]

    window_scores_val = compute_window_scores(
        x_patches_np=x_tune_val,
        y_patches_np=y_tune_val,
        encoder=encoder,
        predictor=predictor,
        device=device,
        batch_size=64
    )  # shape=(294,)
    print(f"  已计算 {num_windows_val} 个窗口的得分。\n")

    window_labels_val = label_loader.generate_window_labels(num_windows_val)  # shape=(294,)

    # -------------------- Step 4: 在 Tuning‐Train 上搜索最优阈值 --------------------
    print("Step 4: 在 Tuning‐Train 上搜索最优阈值...")
    best_threshold, best_train_f1 = search_best_threshold(
        scores=window_scores_train,
        labels=window_labels_train,
        num_candidates=200
    )
    print(f"  最优阈值 (Train)：{best_threshold:.6f}, 对应 F1 = {best_train_f1:.4f}\n")

    # 在 Tuning‐Val 上验证该阈值
    val_metrics = evaluate_on_threshold(
        scores=window_scores_val,
        labels=window_labels_val,
        threshold=best_threshold
    )
    print("  在 Tuning‐Val 上评估：")
    print(f"    Precision = {val_metrics['precision']:.4f}, "
          f"Recall = {val_metrics['recall']:.4f}, "
          f"F1 = {val_metrics['f1']:.4f}\n")

    # -------------------- Step 5: 在 Final Test 上评估 --------------------
    print("Step 5: 在 Final‐Test (msl_final_test.npz) 上评估...")
    final_test_data = np.load(final_test_npz)
    x_final_test = final_test_data["x_patches"]  # shape=(735, 9, 20, 55)
    y_final_test = final_test_data["y_patches"]
    num_windows_test = x_final_test.shape[0]

    # 计算误差指标
    print("计算误差指标...")
    metrics = compute_metrics(
        x_patches_np=x_final_test,
        y_patches_np=y_final_test,
        encoder=encoder,
        predictor=predictor,
        device=device,
        batch_size=64
    )
    
    # 使用预测误差作为异常分数
    window_scores_test = metrics["pred_errors"].reshape(num_windows_test, -1).max(axis=1)
    
    window_labels_test = label_loader.generate_window_labels(num_windows_test)  # shape=(735,)

    test_metrics = evaluate_on_threshold(
        scores=window_scores_test,
        labels=window_labels_test,
        threshold=best_threshold
    )
    print("  在 Final‐Test 上评估结果：")
    print(f"    Precision = {test_metrics['precision']:.4f}, "
          f"Recall = {test_metrics['recall']:.4f}, "
          f"F1 = {test_metrics['f1']:.4f}")
    print(f"    Feature Error (MSE) = {metrics['feature_error']:.4f}")
    print(f"    MAE = {metrics['mae']:.4f}\n")

    print("======== 下游任务完成，报告如下 ========")
    print(f"Best Threshold (from Tuning‐Train): {best_threshold:.6f}")
    print(f"Tuning‐Train F1   = {best_train_f1:.4f}")
    print(f"Tuning‐Val   F1   = {val_metrics['f1']:.4f}")
    print(f"Final‐Test  Precision = {test_metrics['precision']:.4f}")
    print(f"Final‐Test  Recall    = {test_metrics['recall']:.4f}")
    print(f"Final‐Test  F1        = {test_metrics['f1']:.4f}")
    print(f"Final‐Test  Feature Error = {metrics['feature_error']:.4f}")
    print(f"Final‐Test  MAE          = {metrics['mae']:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
