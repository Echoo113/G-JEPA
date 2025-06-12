import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import IsolationForest # 核心模型

# --- 加入项目路径 ---
# 确保可以正确导入自定义模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 导入自定义模块 ---
from patch_loader import get_loader
from jepa.encoder import MyTimeSeriesEncoder

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/jepa_best.pt"  # 预训练的Encoder模型路径
TRAIN_FEATURE_PATH = "data/MSL/patches/train.npz"
TEST_FEATURE_PATH = "data/MSL/patches/test.npz"
BATCH_SIZE = 256  # 可以适当调大以加快特征提取速度
USE_INSTANCE_NORM = True  # 控制是否使用时间维度归一化

def apply_time_norm(x, eps=1e-5):
    """
    对单变量输入 (B, 1, T, 1) 沿时间维度进行归一化
    输入:
        x: Tensor, shape (B, 1, T, 1) —— 一个 batch 的 patch 序列
    输出:
        normalized x, shape 相同
    """
    mean = x.mean(dim=2, keepdim=True)  # 沿时间维度
    std = x.std(dim=2, keepdim=True)
    return (x - mean) / (std + eps)

@torch.no_grad()
def extract_embeddings(encoder, data_loader, description):
    """使用encoder提取指定数据集的潜在表示"""
    print(f"\n正在提取 {description} 的特征...")
    encoder.eval()
    
    all_embeddings = []
    all_labels = []

    for x_batch, _, labels_batch in data_loader:
        # unsqueeze(1) 添加一个维度以匹配模型输入 (B, N_patch, T, F)
        x_batch = x_batch.to(DEVICE).unsqueeze(1)
        
        # === 添加时间维度归一化 ===
        if USE_INSTANCE_NORM:
            x_batch = apply_time_norm(x_batch)
        
        latent = encoder(x_batch)
        B, SEQ, D = latent.shape
        latent_flat = latent.reshape(B * SEQ, D)
        labels_flat = labels_batch.reshape(-1, 1)

        all_embeddings.append(latent_flat.cpu().numpy())
        all_labels.append(labels_flat.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"{description} 特征提取完成，共 {len(all_embeddings)} 个样本。")
    return all_embeddings, all_labels

def main():
    # ========== 1. 加载预训练的Encoder ==========
    print("🚀 开始Encoder + Isolation Forest异常检测流程")
    print("\n[1/4] 正在加载预训练的Encoder模型...")
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"❌错误: 模型文件未找到于 '{MODEL_PATH}'。请确保路径正确。")
        return

    config = checkpoint.get("config")
    if config is None:
        print("❌错误: 检查点文件中缺少 'config' 字典。")
        return
        
    encoder_config = {
        'patch_length': config['patch_length'], 'num_vars': config['num_vars'], 'latent_dim': config['latent_dim'],
        'time_layers': 2, 'patch_layers': 2, 'num_attention_heads': 8, 'ffn_dim': config['latent_dim'] * 4, 'dropout': 0.3
    }

    encoder = MyTimeSeriesEncoder(**encoder_config).to(DEVICE)
    encoder.load_state_dict(checkpoint["encoder_online_state_dict"])
    print("✅ Encoder加载成功。")

    # ========== 2. 加载数据集并提取特征 ==========
    print("\n[2/4] 正在加载数据集并提取特征...")
    train_loader = get_loader(npz_file=TRAIN_FEATURE_PATH, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_loader(npz_file=TEST_FEATURE_PATH, batch_size=BATCH_SIZE, shuffle=False)

    train_embeddings, train_labels = extract_embeddings(encoder, train_loader, "训练集")
    test_embeddings, test_labels = extract_embeddings(encoder, test_loader, "测试集")
    
    # 筛选出用于训练的正常样本
    normal_train_mask = (train_labels == 0).flatten()
    normal_train_embeddings = train_embeddings[normal_train_mask]
    print(f"\n从训练集中筛选出 {len(normal_train_embeddings)} 个正常样本用于训练Isolation Forest。")
    
    # ========== 3. 训练Isolation Forest模型 ==========
    print("\n[3/4] 正在训练 Isolation Forest 模型...")
    # n_jobs=-1 使用所有可用的CPU核心以加速训练
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(normal_train_embeddings)
    print("✅ Isolation Forest 模型训练完成。")

    # ========== 4. 在测试集上评估模型 ==========
    print("\n[4/4] 正在使用 Isolation Forest 在测试集上进行预测与评估...")
    # 模型预测: +1 代表正常 (inlier), -1 代表异常 (outlier)
    test_preds_iso = iso_forest.predict(test_embeddings)
    
    # 将预测结果映射到 0/1 标签 (0: 正常, 1: 异常)
    # 这是评估指标所期望的格式
    test_preds_mapped = np.array([0 if p == 1 else 1 for p in test_preds_iso])
    
    # 计算评估指标
    f1 = f1_score(test_labels, test_preds_mapped)
    precision = precision_score(test_labels, test_preds_mapped)
    recall = recall_score(test_labels, test_preds_mapped)
    cm = confusion_matrix(test_labels, test_preds_mapped)
    
    print("\n--- Isolation Forest 在测试集上的最终评估结果 ---")
    print(f"  - F1 Score:  {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print("  - Confusion Matrix:")
    print(f"    {cm}")
    print("-------------------------------------------------")
    print("\n🎉 流程结束。")

if __name__ == "__main__":
    main()