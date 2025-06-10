import sys
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# --- 确保项目根目录在 sys.path 中，以便正确导入模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 从你的项目中导入必要的模块 ---
# 假设这些是你项目中定义的类和函数
from patch_loader import create_labeled_loader 
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor # Predictor 虽然不直接评估，但加载模型时需要它
from train_jepa import Classifier # 直接从你的训练脚本中导入分类器类定义

# ========= 全局设置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/jepa_best.pt"
BATCH_SIZE = 128 # 在评估时可以使用稍大的批量

# 测试集文件路径
TEST_FEATURE_PATH = "data/MSL/patches/msl_final_test.npz"
TEST_LABEL_PATH = "data/MSL/patches/msl_final_test_labels.npz"


@torch.no_grad() # 整个函数在无梯度模式下运行，以节省内存和计算资源
def evaluate_model(
    encoder: nn.Module, 
    classifier: nn.Module, 
    data_loader: DataLoader,
    model_name: str
) -> None:
    """
    一个通用的评估函数，用于计算并打印一套 Encoder+Classifier 的性能指标。
    """
    print("-" * 50)
    print(f"评估中: {model_name}")
    
    # --- 将模型设置为评估模式 ---
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    # --- 遍历测试集数据 ---
    for x_batch, _, labels_batch in data_loader: # 我们只需要 x 和 labels
        x_batch = x_batch.to(DEVICE)
        
        # 1. 通过编码器获取 Latent 表示
        # 注意: 这里的 ctx_latent 是整个补丁序列的latent
        ctx_latent = encoder(x_batch) # Shape: (B, 9, D)
        
        # 2. 拉平 Latent 和标签以匹配分类器输入
        B, SEQ, D = ctx_latent.shape
        latent_flat = ctx_latent.reshape(B * SEQ, D)
        labels_flat = labels_batch.reshape(B * SEQ, 1).cpu().numpy()
        
        # 3. 通过分类器获取 Logits
        logits = classifier(latent_flat) # Shape: (B*9, 1)
        
        # 4. 将 Logits 转换为二进制预测结果 (0 或 1)
        preds = torch.sigmoid(logits) > 0.5
        preds = preds.cpu().numpy()
        
        # 5. 收集所有预测和真实标签
        all_preds.append(preds)
        all_labels.append(labels_flat)
        
    # --- 拼接所有批次的结果 ---
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # --- 计算并打印性能指标 ---
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"性能指标 ({model_name}):")
    print(f"  - F1 Score:  {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print("混淆矩阵 (Confusion Matrix):")
    print(f"  [[TN, FP]\n   [FN, TP]]")
    print(f"{cm}")
    print("-" * 50)


def main():
    """主执行函数"""
    # --- 1. 加载保存的模型检查点 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到于 '{MODEL_PATH}'")
        return

    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = checkpoint['config']
    
    # --- 2. 根据保存的配置重新初始化所有模型 ---
    encoder_online = MyTimeSeriesEncoder(**config).to(DEVICE)
    encoder_target = MyTimeSeriesEncoder(**config).to(DEVICE) # Target Encoder 结构与 Online 一致
    classifier1 = Classifier(input_dim=config['latent_dim']).to(DEVICE)
    classifier2 = Classifier(input_dim=config['latent_dim']).to(DEVICE)
    
    # 加载已训练的权重
    encoder_online.load_state_dict(checkpoint['encoder_online_state_dict'])
    encoder_target.load_state_dict(checkpoint['encoder_target_state_dict'])
    classifier1.load_state_dict(checkpoint['classifier1_state_dict'])
    classifier2.load_state_dict(checkpoint['classifier2_state_dict'])
    
    print("模型权重加载成功。")

    # --- 3. 准备测试数据加载器 ---
    print("\n正在准备测试数据...")
    test_loader = create_labeled_loader(
        feature_npz_path=TEST_FEATURE_PATH,
        label_npz_path=TEST_LABEL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False # 评估时不需要打乱
    )
    
    # --- 4. 分别评估两个分支的性能 ---
    evaluate_model(encoder_online, classifier1, test_loader, "Online Encoder + Classifier 1")
    evaluate_model(encoder_target, classifier2, test_loader, "Target Encoder (EMA) + Classifier 2")


if __name__ == "__main__":
    print("Starting anomaly detection...")
    main()