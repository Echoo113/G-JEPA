import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 确保项目根目录在sys.path中，以便导入自定义模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 从你的项目中导入必要的模块 ---
# 假设这些模块可以被正确找到
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

# --- 分类器训练超参数 ---
BATCH_SIZE = 256  # 可以使用更大的批次，因为反向传播的参数很少
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_DIM = 128 # 分类头隐藏层维度

# ========= Step 1: 加载预训练的Encoder =========
print(f"[Step 1] Loading pre-trained JEPA model from: {PRETRAINED_MODEL_PATH}")
if not os.path.exists(PRETRAINED_MODEL_PATH):
    raise FileNotFoundError(f"错误：找不到预训练模型文件 '{PRETRAINED_MODEL_PATH}'。请先完成JEPA的预训练。")

checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)

# 优先从checkpoint中恢复模型配置
try:
    config = checkpoint['config']
    print("成功从checkpoint加载模型配置。")
except KeyError:
    print("警告: checkpoint中未找到'config'字典。将使用脚本中的默认值。")
    # 如果checkpoint中没有存配置，你需要在这里手动填写
    config = {
        'latent_dim': 512, 
        'patch_length': 20,
        'num_vars': 55
    }

# --- 初始化并冻结Encoder ---
encoder = MyTimeSeriesEncoder(
    patch_length=config['patch_length'],
    num_vars=config['num_vars'],
    latent_dim=config['latent_dim'],
    time_layers=2, patch_layers=3, num_attention_heads=16, 
    ffn_dim=config['latent_dim']*4, dropout=0.2
).to(DEVICE)

# 加载 Target (EMA) Encoder 的权重，这是下游任务的最佳实践
encoder.load_state_dict(checkpoint['encoder_target_state_dict'])

# 冻结Encoder的所有参数
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval() # 始终保持评估模式

print("Pre-trained Target (EMA) Encoder has been loaded and frozen.")

# ========= Step 2: 定义分类器模型 =========
class AnomalyClassifier(nn.Module):
    def __init__(self, frozen_encoder, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = frozen_encoder
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1) # 输出一个原始logit值
        )

    def forward(self, x_patch):
        # x_patch 形状: (B, PatchLen, NumVars)
        with torch.no_grad(): # 再次确保encoder不计算梯度
            # Encoder期望输入有序列维度，所以增加一个维度
            # (B, P, V) -> (B, 1, P, V)
            z = self.encoder(x_patch.unsqueeze(1)) 
            z = z.squeeze(1) # 移除序列维度 -> (B, D)
        
        logits = self.classifier_head(z)
        return logits.squeeze(-1)

print("\n[Step 2] Anomaly Classifier model defined.")
model = AnomalyClassifier(encoder, HIDDEN_DIM, config['latent_dim']).to(DEVICE)

# ========= Step 3: 准备数据和处理类别不平衡 =========
print("\n[Step 3] Preparing data and handling class imbalance...")

# --- 加载数据 ---
def load_and_flatten(data_path, label_path):
    # 确保文件存在
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"找不到数据文件或标签文件: {data_path}, {label_path}")
        
    features_data = np.load(data_path)['x_patches']
    labels_data = np.load(label_path)['x_labels']
    
    # 将序列和补丁维度展平
    # 特征: (N, Seq, PatchLen, Vars) -> (N*Seq, PatchLen, Vars)
    # 标签: (N, Seq) -> (N*Seq,)
    num_sequences, seq_len, patch_len, num_vars = features_data.shape
    features_flat = features_data.reshape(-1, patch_len, num_vars)
    labels_flat = labels_data.flatten()
    
    return features_flat, labels_flat

train_features, train_labels = load_and_flatten(TUNE_TRAIN_DATA_PATH, TUNE_TRAIN_LABELS_PATH)
val_features, val_labels = load_and_flatten(TUNE_VAL_DATA_PATH, TUNE_VAL_LABELS_PATH)
test_features, test_labels = load_and_flatten(FINAL_TEST_DATA_PATH, FINAL_TEST_LABELS_PATH)

print(f"Tune Train: {train_features.shape[0]} patches")
print(f"Tune Val: {val_features.shape[0]} patches")
print(f"Final Test: {test_features.shape[0]} patches")

# --- 计算损失权重以应对类别不平衡 ---
num_positives = np.sum(train_labels == 1)
num_negatives = np.sum(train_labels == 0)

if num_positives == 0:
    # 避免除以零错误
    pos_weight = torch.tensor([1.0], device=DEVICE)
    print("警告: 训练集中没有正样本(异常)，权重设置为1.0")
else:
    # pos_weight = (负样本数 / 正样本数)，给样本少的正类（异常）更高的权重
    pos_weight = torch.tensor([num_negatives / num_positives], device=DEVICE)

print(f"Class Imbalance Info: Negatives={num_negatives}, Positives={num_positives}")
print(f"Calculated pos_weight for BCE Loss: {pos_weight.item():.2f}")

# --- 创建DataLoader ---
train_dataset = TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).float())
val_dataset = TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).float())
test_dataset = TensorDataset(torch.from_numpy(test_features).float(), torch.from_numpy(test_labels).float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========= Step 4: 训练分类器 =========
print("\n[Step 4] Starting classifier training...")

# --- 优化器和带权重的损失函数 ---
# 只优化分类头的参数！
optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=LEARNING_RATE)
# 使用BCEWithLogitsLoss以获得更好的数值稳定性，并传入计算好的权重
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    # --- 验证阶段 ---
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算验证集指标
    accuracy = accuracy_score(all_labels, all_preds)
    # 对于不平衡数据，F1分数是比准确率更重要的指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    # 保存F1分数最佳的模型
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), CLASSIFIER_SAVE_PATH)
        print(f"  ✅ New best model saved with F1 score: {best_val_f1:.4f}")

# ========= Step 5: 在最终测试集上评估 =========
print("\n[Step 5] Evaluating on the final test set...")

# 加载性能最好的分类器
if os.path.exists(CLASSIFIER_SAVE_PATH):
    model.load_state_dict(torch.load(CLASSIFIER_SAVE_PATH))
    model.eval()

    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            preds = torch.sigmoid(logits) > 0.5
            all_preds_test.append(preds.cpu().numpy())
            all_labels_test.append(labels.cpu().numpy())

    all_preds_test = np.concatenate(all_preds_test)
    all_labels_test = np.concatenate(all_labels_test)

    # 计算最终测试指标
    accuracy_test = accuracy_score(all_labels_test, all_preds_test)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(all_labels_test, all_preds_test, average='binary', zero_division=0)

    print("\n--- Final Test Results ---")
    print(f"  Accuracy:  {accuracy_test:.4f}")
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall:    {recall_test:.4f}")
    print(f"  F1 Score:  {f1_test:.4f}")
    print("--------------------------")

    # 可视化混淆矩阵
    cm = confusion_matrix(all_labels_test, all_preds_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Final Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
else:
    print(f"错误：找不到已保存的最佳分类器模型 '{CLASSIFIER_SAVE_PATH}'。")
