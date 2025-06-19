import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# 假设您的自定义模块在项目的根目录下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保可以找到之前的模块
from jepa.encoder import MyTimeSeriesEncoder
from patch_loader import get_loader

# =============================================================================
#  组件定义: 严格使用与您训练脚本中相同的组件
# =============================================================================

class StrongClassifier(nn.Module):
    """用于下游任务的分类器"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def apply_instance_norm(x, eps=1e-5):
    """对输入 x 的每个实例独立进行归一化，与预训练时完全一致"""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)

# ========= 1. 全局和下游任务设置 =========
def setup_config():
    """集中管理所有配置"""
    config = {
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "PRETRAINED_MODEL_PATH": "model/jepa_final.pt",
        "DATA_PATH_PREFIX": "data/TSB-AD-U/patches/", # 请根据您的数据集路径修改
        "DOWNSTREAM_EPOCHS": 50,
        "DOWNSTREAM_LR": 1e-3,
        "DOWNSTREAM_BATCH_SIZE": 512,
        "LATENT_DIM": 128,
        "X_PATCH_LENGTH": 30,
        "NUM_VARS": 1,
    }
    return config

# ========= 2. 主执行函数 =========
def main():
    """主执行逻辑"""
    cfg = setup_config()
    print("🚀 [Step 1] Configuration loaded.")
    print(f"Using device: {cfg['DEVICE']}")

    # --- 加载数据集 ---
    print("\n📦 [Step 2] Loading datasets for downstream task...")
    try:
        # 使用验证集训练分类器
        train_classifier_loader = get_loader(
            npz_file=os.path.join(cfg["DATA_PATH_PREFIX"], "val.npz"),
            batch_size=cfg["DOWNSTREAM_BATCH_SIZE"],
            shuffle=True
        )
        # 使用测试集评估分类器
        test_classifier_loader = get_loader(
            npz_file=os.path.join(cfg["DATA_PATH_PREFIX"], "test.npz"),
            batch_size=cfg["DOWNSTREAM_BATCH_SIZE"],
            shuffle=False
        )
        print("✅ Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}. Please check your `DATA_PATH_PREFIX`.")
        return

    # --- 加载并冻结预训练Encoder ---
    print("\n🧊 [Step 3] Loading and freezing pre-trained encoder...")
    pretrained_encoder = MyTimeSeriesEncoder(
        patch_length=cfg["X_PATCH_LENGTH"],
        num_vars=cfg["NUM_VARS"],
        latent_dim=cfg["LATENT_DIM"],
        time_layers=2, patch_layers=2, num_attention_heads=8,
        ffn_dim=cfg["LATENT_DIM"] * 4, dropout=0.4
    ).to(cfg["DEVICE"])

    try:
        state_dict = torch.load(cfg["PRETRAINED_MODEL_PATH"], map_location=cfg["DEVICE"])
        # 加载 online encoder 的权重
        pretrained_encoder.load_state_dict(state_dict['encoder_online_state_dict'])
    except FileNotFoundError:
        print(f"❌ Error: Pre-trained model not found at '{cfg['PRETRAINED_MODEL_PATH']}'.")
        return

    for param in pretrained_encoder.parameters():
        param.requires_grad = False
    pretrained_encoder.eval()
    print("✅ Pre-trained encoder loaded and frozen.")

    # --- 初始化下游任务组件 ---
    print("\n✨ [Step 4] Initializing new classifier and optimizer...")
    downstream_classifier = StrongClassifier(input_dim=cfg["LATENT_DIM"]).to(cfg["DEVICE"])
    optimizer = torch.optim.AdamW(downstream_classifier.parameters(), lr=cfg["DOWNSTREAM_LR"])
    
    def compute_global_anomaly_ratio(loader):
        labels = [l for _, _, l in loader]
        return torch.cat(labels, dim=0).float().mean().item()

    anomaly_ratio = compute_global_anomaly_ratio(train_classifier_loader)
    pos_weight = torch.tensor([(1 - anomaly_ratio) / (anomaly_ratio + 1e-6)], device=cfg["DEVICE"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Classifier initialized. Anomaly ratio in training data: {anomaly_ratio:.4f}")

    # --- 训练分类器 ---
    print("\n💪 [Step 5] Starting classifier training...")
    for epoch in range(1, cfg["DOWNSTREAM_EPOCHS"] + 1):
        downstream_classifier.train()
        total_loss = 0
        for x_batch, _, labels_batch in train_classifier_loader:
            x_batch, labels_batch = x_batch.to(cfg["DEVICE"]), labels_batch.to(cfg["DEVICE"])

            # =================================================================
            #  严格仿照您的训练脚本进行数据处理和特征提取
            # =================================================================
            # 1. 添加N_patch维度，使输入变为4D: (B, 1, T, F)
            x_batch_4d = x_batch.unsqueeze(1)
            
            # 2. 使用实例归一化
            x_batch_normed = apply_instance_norm(x_batch_4d)
            
            optimizer.zero_grad()
            
            # 3. 使用冻结的encoder提取特征
            with torch.no_grad():
                features = pretrained_encoder(x_batch_normed) # Shape: [B, SEQ, D], 此处 SEQ=1

            # 4. 严格仿照L2/L3损失的维度处理方式
            B, SEQ, D = features.shape
            features_flat = features.reshape(B * SEQ, D)
            labels_expanded = labels_batch.view(-1, 1).repeat(1, SEQ).view(-1, 1).float()
            # =================================================================

            # 5. 通过分类器得到预测
            logits = downstream_classifier(features_flat)
            
            # 6. 计算损失并更新分类器
            loss = criterion(logits, labels_expanded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_classifier_loader)
        print(f"[Epoch {epoch:02d}/{cfg['DOWNSTREAM_EPOCHS']}] Training... Train Loss: {avg_train_loss:.4f}")
    
    print("\n🏁 Training finished.")

    # --- 在最后一个epoch后，进行最终评估 ---
    print("\n🧪 [Step 6] Starting final evaluation on the test set...")
    downstream_classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, _, labels_batch in test_classifier_loader:
            x_batch = x_batch.to(cfg["DEVICE"])

            # 同样地，严格仿照训练脚本进行处理
            x_batch_4d = x_batch.unsqueeze(1)
            x_batch_normed = apply_instance_norm(x_batch_4d)
            
            features = pretrained_encoder(x_batch_normed)
            
            B, SEQ, D = features.shape
            features_flat = features.reshape(B * SEQ, D)
            logits = downstream_classifier(features_flat)
            
            all_preds.append((logits > 0).cpu().int().numpy())
            all_labels.append(labels_batch.view(-1, 1).repeat(1, SEQ).view(-1, 1).numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    print("\n--- ✅ Final Test Results ---")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("--------------------------")

    print("\n🎉 Downstream task evaluation finished!")

if __name__ == "__main__":
    main()