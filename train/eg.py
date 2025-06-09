import sys
import os
import numpy as np
import torch
from itertools import product
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# --- 将项目根目录添加到Python路径中 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jepa.encoder import MyTimeSeriesEncoder

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JEPA_CKPT_PATH = "model/jepa_best.pt"

# --- 数据文件路径 ---
# 用于调优的数据集（将会被合并）
TRAIN_DATA_PATH = "data/MSL/patches/msl_tune_train.npz"
TRAIN_LABEL_PATH = "data/MSL/patches/msl_tune_train_labels.npz"
VAL_DATA_PATH = "data/MSL/patches/msl_tune_val.npz"
VAL_LABEL_PATH = "data/MSL/patches/msl_tune_val_labels.npz"
# 最终留出的、独立的测试集
TEST_DATA_PATH = "data/MSL/patches/msl_final_test.npz"
TEST_LABEL_PATH = "data/MSL/patches/msl_final_test_labels.npz"

BATCH_SIZE = 256
CV_FOLDS = 5 # 使用5折交叉验证以进行更稳健的评估

# ==================== 辅助函数 ====================
def extract_features(data_path, encoder):
    """一个辅助函数，用于加载数据并通过编码器提取特征。"""
    x_patches = np.load(data_path)['x_patches']
    dataset = TensorDataset(torch.from_numpy(x_patches).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    all_latents = []
    with torch.no_grad():
        for window_batch in loader:
            window_patches = window_batch[0].to(DEVICE)
            latent_sequence = encoder(window_patches)
            all_latents.append(latent_sequence.cpu())
            
    latents_np = torch.cat(all_latents).numpy()
    return latents_np.reshape(-1, latents_np.shape[-1])

# ==================== 主脚本 ====================
if __name__ == "__main__":
    ## 第1步: 加载预训练的JEPA编码器
    print("## 第1步: 加载预训练的JEPA编码器...")
    ckpt = torch.load(JEPA_CKPT_PATH, map_location=DEVICE, weights_only=True)
    encoder = MyTimeSeriesEncoder(
        patch_length=ckpt['patch_length'], num_vars=ckpt['num_vars'], latent_dim=ckpt['latent_dim']
    ).to(DEVICE)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    print("编码器加载成功。")

    # ----------------------------------------------------------------

    ## 第2步: 创建用于调优的合并数据集 (训练集 + 验证集)
    print("\n## 第2步: 创建用于调优的合并数据集 (训练集 + 验证集)...")
    train_latents = extract_features(TRAIN_DATA_PATH, encoder)
    train_labels = np.load(TRAIN_LABEL_PATH)['x_labels'].reshape(-1)
    
    val_latents = extract_features(VAL_DATA_PATH, encoder)
    val_labels = np.load(VAL_LABEL_PATH)['x_labels'].reshape(-1)
    
    # 合并数据集
    combined_latents = np.concatenate((train_latents, val_latents))
    combined_labels = np.concatenate((train_labels, val_labels))
    print(f"合并数据集创建完成。总Patch数: {len(combined_latents)}")

    # ----------------------------------------------------------------
    
    ## 第3步: 在合并数据集上进行超参数搜索
    print(f"\n## 第3步: 开始在合并数据集上进行 {CV_FOLDS}-折交叉验证网格搜索...")
    anomaly_ratio = np.mean(combined_labels)
    print(f"合并调优数据中的异常比例: {anomaly_ratio:.4f}")

    param_grid = {
        'n_neighbors': [50, 100, 150],
        'contamination': [anomaly_ratio, anomaly_ratio * 1.2],
        'p': [1, 2]
    }
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    results = {}
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for i, params in enumerate(param_combinations):
        fold_scores = []
        # print(f"  - 正在测试参数 {i+1}/{len(param_combinations)}: {params}")
        for train_index, test_index in skf.split(combined_latents, combined_labels):
            X_train, X_test = combined_latents[train_index], combined_latents[test_index]
            y_train, y_test = combined_labels[train_index], combined_labels[test_index]
            try:
                lof = LocalOutlierFactor(novelty=True, **params)
                lof.fit(X_train)
                preds_raw = lof.predict(X_test)
                preds = np.where(preds_raw == -1, 1, 0)
                score = f1_score(y_test, preds, pos_label=1, zero_division=0.0)
                fold_scores.append(score)
            except Exception:
                fold_scores.append(0.0)
        results[tuple(params.items())] = np.mean(fold_scores)
    
    best_params_tuple = max(results, key=results.get)
    best_params = dict(best_params_tuple)
    best_score = results[best_params_tuple]
    
    print("超参数搜索完成。")
    print("\n--- 最佳超参数组合 ---")
    print(best_params)
    print(f"\n最佳交叉验证F1分数: {best_score:.4f}")
    
    # ----------------------------------------------------------------

    ## 第4步: 使用最佳参数在所有调优数据上训练最终模型
    print("\n## 第4步: 使用最佳参数在所有调优数据上训练最终模型...")
    final_model = LocalOutlierFactor(novelty=True, **best_params)
    final_model.fit(combined_latents)
    print("最终模型训练完成。")

    # ----------------------------------------------------------------

    ## 第5步: 在独立的测试集上评估最终模型
    print("\n## 第5步: 在独立的测试集上评估最终模型...")
    test_latents = extract_features(TEST_DATA_PATH, encoder)
    test_labels = np.load(TEST_LABEL_PATH)['x_labels'].reshape(-1)
    
    test_preds_raw = final_model.predict(test_latents)
    test_preds = np.where(test_preds_raw == -1, 1, 0)

    print("\n" + "="*55)
    print("### 最终模型在【独立测试集】上的性能报告 ###")
    print("="*55)
    print(classification_report(test_labels, test_preds, target_names=['正常 (Class 0)', '异常 (Class 1)']))
    print("-" * 55)