import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 确保项目根目录在sys.path中 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 导入自定义模块 ---
from jepa.encoder import MyTimeSeriesEncoder

# ========= 配置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 文件路径配置 ---
PRETRAINED_MODEL_PATH = "model/jepa_best.pt"
TUNE_TRAIN_DATA_PATH = "data/MSL/patches/msl_tune_train.npz"
TUNE_TRAIN_LABELS_PATH = "data/MSL/patches/msl_tune_train_labels.npz"
TUNE_VAL_DATA_PATH = "data/MSL/patches/msl_tune_val.npz"
TUNE_VAL_LABELS_PATH = "data/MSL/patches/msl_tune_val_labels.npz"
FINAL_TEST_DATA_PATH = "data/MSL/patches/msl_final_test.npz"
FINAL_TEST_LABELS_PATH = "data/MSL/patches/msl_final_test_labels.npz"
CLASSIFIER_SAVE_PATH = "model/anomaly_classifier_best.pt"

# --- 分类器训练超参数 (优化后) ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-4  # MODIFIED: 降低初始学习率
EPOCHS = 10
HIDDEN_DIM_1 = 256 # MODIFIED: 增加分类头容量
HIDDEN_DIM_2 = 128

# ========= 辅助函数 =========
def find_best_threshold_and_metrics(probs, labels):
    """
    在验证集上寻找最佳F1分数对应的阈值
    """
    best_f1 = -1
    best_threshold = 0.5
    # 在0.01到0.99之间测试100个阈值
    for threshold in np.linspace(0.01, 0.99, 100):
        preds = (probs > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    # 使用最佳阈值计算所有指标
    final_preds = (probs > best_threshold).astype(int)
    accuracy = accuracy_score(labels, final_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, final_preds, average='binary', zero_division=0)
    
    return accuracy, precision, recall, f1, best_threshold

# ========= Step 1: 加载预训练的Encoder =========
print(f"[Step 1] Loading pre-trained JEPA model from: {PRETRAINED_MODEL_PATH}")
# ... (此部分代码与之前版本相同)
checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
config = checkpoint.get('config', {'latent_dim': 512, 'patch_length': 20, 'num_vars': 55})
encoder = MyTimeSeriesEncoder(
    patch_length=config['patch_length'], num_vars=config['num_vars'], latent_dim=config['latent_dim'],
    time_layers=2, patch_layers=3, num_attention_heads=16, ffn_dim=config['latent_dim']*4, dropout=0.2
).to(DEVICE)
encoder.load_state_dict(checkpoint['encoder_target_state_dict'])
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()
print("Pre-trained Target (EMA) Encoder has been loaded and frozen.")


# ========= Step 2: 定义分类器模型 (增强版) =========
class AnomalyClassifier(nn.Module):
    def __init__(self, frozen_encoder, h_dim1, h_dim2, latent_dim):
        super().__init__()
        self.encoder = frozen_encoder
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, h_dim1),
            nn.BatchNorm1d(h_dim1), # 增加BN层以稳定训练
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(h_dim2, 1) # 输出原始logit值
        )

    def forward(self, x_patch):
        with torch.no_grad():
            z = self.encoder(x_patch.unsqueeze(1)).squeeze(1)
        logits = self.classifier_head(z)
        return logits.squeeze(-1)

print("\n[Step 2] Enhanced Anomaly Classifier model defined.")
model = AnomalyClassifier(encoder, HIDDEN_DIM_1, HIDDEN_DIM_2, config['latent_dim']).to(DEVICE)


# ========= Step 3: 准备数据和处理类别不平衡 =========
print("\n[Step 3] Preparing data and handling class imbalance...")
# ... (此部分代码与之前版本相同)
def load_and_flatten(data_path, label_path):
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"找不到文件: {data_path} 或 {label_path}")
    features_data = np.load(data_path)['x_patches']
    labels_data = np.load(label_path)['x_labels']
    features_flat = features_data.reshape(-1, features_data.shape[2], features_data.shape[3])
    labels_flat = labels_data.flatten()
    return features_flat, labels_flat

train_features, train_labels = load_and_flatten(TUNE_TRAIN_DATA_PATH, TUNE_TRAIN_LABELS_PATH)
val_features, val_labels = load_and_flatten(TUNE_VAL_DATA_PATH, TUNE_VAL_LABELS_PATH)
test_features, test_labels = load_and_flatten(FINAL_TEST_DATA_PATH, FINAL_TEST_LABELS_PATH)

num_positives = np.sum(train_labels == 1)
num_negatives = np.sum(train_labels == 0)
pos_weight = torch.tensor([num_negatives / max(1, num_positives)], device=DEVICE)
print(f"Calculated pos_weight for BCE Loss: {pos_weight.item():.2f}")

train_dataset = TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).float())
val_dataset = TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).float())
test_dataset = TensorDataset(torch.from_numpy(test_features).float(), torch.from_numpy(test_labels).float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ========= Step 4: 训练分类器 (优化后) =========
print("\n[Step 4] Starting enhanced classifier training...")

optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# MODIFIED: 添加学习率调度器，监控验证集F1分数
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)

best_val_f1 = -1.0
best_threshold = 0.5

for epoch in range(1, EPOCHS + 1):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # --- 验证阶段 (寻找最佳阈值) ---
    model.eval()
    all_probs_val = []
    all_labels_val = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            logits = model(features)
            probs = torch.sigmoid(logits)
            all_probs_val.append(probs.cpu().numpy())
            all_labels_val.append(labels.numpy())

    all_probs_val = np.concatenate(all_probs_val)
    all_labels_val = np.concatenate(all_labels_val)

    # 找到当前epoch的最佳F1和阈值
    accuracy, precision, recall, f1, threshold = find_best_threshold_and_metrics(all_probs_val, all_labels_val)
    
    print(f"[Epoch {epoch:02d}] Val Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f} @ Threshold: {threshold:.2f}")

    # 更新学习率调度器
    scheduler.step(f1)
    
    # 保存F1分数最佳的模型和阈值
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_threshold = threshold
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_threshold': best_threshold,
            'best_f1_score': best_val_f1
        }, CLASSIFIER_SAVE_PATH)
        print(f"  ✅ New best model saved with F1 score: {best_val_f1:.4f} and Threshold: {best_threshold:.2f}")


# ========= Step 5: 在最终测试集上评估 =========
print("\n[Step 5] Evaluating on the final test set with the best threshold...")

if os.path.exists(CLASSIFIER_SAVE_PATH):
    classifier_checkpoint = torch.load(CLASSIFIER_SAVE_PATH)
    model.load_state_dict(classifier_checkpoint['model_state_dict'])
    final_best_threshold = classifier_checkpoint['best_threshold']
    print(f"Loaded best classifier. Using optimal threshold found during validation: {final_best_threshold:.2f}")

    model.eval()
    all_probs_test = []
    all_labels_test = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            logits = model(features)
            probs = torch.sigmoid(logits)
            all_probs_test.append(probs.cpu().numpy())
            all_labels_test.append(labels.numpy())

    all_probs_test = np.concatenate(all_probs_test)
    all_labels_test = np.concatenate(all_labels_test)
    
    # 使用找到的最佳阈值进行预测
    final_preds = (all_probs_test > final_best_threshold).astype(int)

    # 计算最终测试指标
    accuracy_test = accuracy_score(all_labels_test, final_preds)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(all_labels_test, final_preds, average='binary', zero_division=0)

    print("\n--- Final Test Results ---")
    print(f"  Accuracy:  {accuracy_test:.4f}")
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall:    {recall_test:.4f}")
    print(f"  F1 Score (Anomaly):  {f1_test:.4f}")
    print("--------------------------")

    # 可视化混淆矩阵
    cm = confusion_matrix(all_labels_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Final Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
else:
    print(f"错误：找不到已保存的最佳分类器模型 '{CLASSIFIER_SAVE_PATH}'。")
