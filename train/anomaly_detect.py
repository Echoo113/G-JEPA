import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    average_precision_score
)

# ---------------------------------------------------
# 1. 配置：数据文件路径
# ---------------------------------------------------
# Patch & 滑窗参数
TOTAL_SERIES_LENGTH = 73729
INPUT_LEN          = 100
OUTPUT_LEN         = 100
WINDOW_STRIDE      = 20
PATCH_LEN          = 20
PATCH_STRIDE       = 10

# 模型参数
LATENT_DIM         = 128  # 添加 latent 维度
NP                 = 9    # 每个窗口的 patch 数量

NUM_WINDOWS_TRAIN  = 2648
NUM_WINDOWS_VAL    = 294
NUM_WINDOWS_TEST   = 735

# 数据文件路径
TRAIN_NPZ_PATH     = "data/MSL/patches/msl_tune_train.npz"
VAL_NPZ_PATH       = "data/MSL/patches/msl_tune_val.npz"
TEST_NPZ_PATH      = "data/MSL/patches/msl_final_test.npz"
TRAIN_LABEL_PATH   = "data/MSL/patches/msl_tune_train_labels.npz"
VAL_LABEL_PATH     = "data/MSL/patches/msl_tune_val_labels.npz"
TEST_LABEL_PATH    = "data/MSL/patches/msl_final_test_labels.npz"
BASELINE_TRAIN_PATH = "data/MSL/patches/msl_train.npz"  # 用于计算baseline的数据

# JEPA 模型检查点
JEPA_CKPT_PATH     = "model/jepa_best.pt"

# ---------------------------------------------------
# 2. 配置：设备与模型
# ---------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你已经定义好了和训练时相同架构的 Encoder 和 Predictor：
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

encoder = MyTimeSeriesEncoder(
    patch_length=PATCH_LEN,
    num_vars=55,
    latent_dim=LATENT_DIM,
    time_layers=2,
    patch_layers=3,
    num_attention_heads=8,
    ffn_dim=LATENT_DIM * 4,
    dropout=0.2
).to(DEVICE)

predictor = JEPPredictor(
    latent_dim=LATENT_DIM,
    num_heads=4,
    num_layers=3,
    ffn_dim=LATENT_DIM * 4,
    dropout=0.2,
    prediction_length=NP  # 预测出来的 patch 数量 (与 x_patches, y_patches 中 Np 对齐)
).to(DEVICE)

# 2.1 加载 checkpoint 并冻结参数
ckpt = torch.load(JEPA_CKPT_PATH, map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder_state_dict'])
predictor.load_state_dict(ckpt['predictor_state_dict'])
encoder.eval()
predictor.eval()
for p in encoder.parameters():
    p.requires_grad = False
for p in predictor.parameters():
    p.requires_grad = False

# ---------------------------------------------------
# 3. 加载数据
# ---------------------------------------------------
# 3.1 加载 patch 数据
train_data = np.load(TRAIN_NPZ_PATH)
val_data   = np.load(VAL_NPZ_PATH)
test_data  = np.load(TEST_NPZ_PATH)

train_x = train_data["x_patches"]  # (2648, 9, 20, 55)
train_y = train_data["y_patches"]  # (2648, 9, 20, 55)
val_x   = val_data["x_patches"]    # (294, 9, 20, 55)
val_y   = val_data["y_patches"]    # (294, 9, 20, 55)
test_x  = test_data["x_patches"]   # (735, 9, 20, 55)
test_y  = test_data["y_patches"]   # (735, 9, 20, 55)

# 3.2 加载标签数据
train_labels_data = np.load(TRAIN_LABEL_PATH)
val_labels_data   = np.load(VAL_LABEL_PATH)
test_labels_data  = np.load(TEST_LABEL_PATH)

train_labels = train_labels_data["x_labels"]  # (2648, 9, 1)
val_labels   = val_labels_data["x_labels"]    # (294, 9, 1)
test_labels  = test_labels_data["x_labels"]   # (735, 9, 1)

# ---------------------------------------------------
# 4. 定义一个函数：提取单个窗口的误差
# ---------------------------------------------------
def calculate_baseline_latents():
    """使用msl_train.npz计算baseline latents"""
    print("计算baseline latents...")
    all_latents = []
    
    # 加载用于计算baseline的训练数据
    baseline_data = np.load(BASELINE_TRAIN_PATH)
    baseline_x = baseline_data["x_patches"]  # (N, Np, T, F)
    
    # 使用训练集计算baseline
    for i in range(len(baseline_x)):
        x_tensor = torch.from_numpy(baseline_x[i]).float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            latents = encoder(x_tensor)  # (1, Np, D)
            all_latents.append(latents.squeeze(0).cpu().numpy())
    
    all_latents = np.concatenate(all_latents, axis=0)  # (N_train * Np, D)
    baseline_latent = np.mean(all_latents, axis=0)  # (D,)
    print(f"Baseline latent shape: {baseline_latent.shape}")
    print(f"用于计算baseline的样本数: {len(baseline_x)}")
    print(f"每个样本的patch数: {baseline_x.shape[1]}")
    print(f"总patch数: {len(all_latents)}")
    return baseline_latent

def extract_window_errors(x_window: np.ndarray, y_window: np.ndarray, baseline_latent: np.ndarray):
    """
    Args:
      x_window: np.ndarray，shape = (Np, T, F)
      y_window: np.ndarray，shape = (Np, T, F)
      baseline_latent: np.ndarray，shape = (D,)
    Returns:
      mse_errors: np.ndarray，shape = (Np,)，表示每个 patch 的 MSE 误差
      baseline_errors: np.ndarray，shape = (Np,)，表示与 baseline 的差距
      true_latent: np.ndarray，shape = (Np, D)
      pred_latent: np.ndarray，shape = (Np, D)
    """
    encoder.to(DEVICE)
    predictor.to(DEVICE)

    x_tensor = torch.from_numpy(x_window).float().to(DEVICE).unsqueeze(0)  # (1, Np, T, F)
    y_tensor = torch.from_numpy(y_window).float().to(DEVICE).unsqueeze(0)  # (1, Np, T, F)

    with torch.no_grad():
        # 4.1 用同一个 Encoder 编码上下文 patch → ctx_latent: (1, Np, D)
        ctx_latent = encoder(x_tensor)   # (1, Np, D)
        # 4.2 用同一个 Encoder 编码真实未来 patch → tgt_latent: (1, Np, D)
        tgt_latent = encoder(y_tensor)   # (1, Np, D)

        # 4.3 Predictor 生成"预测未来 latent" → pred_latent: (1, Np, D)
        pred_latent, _ = predictor(ctx_latent, tgt_latent)  # (1, Np, D)

    # 4.4 把 torch.Tensor → CPU numpy
    pred_latent = pred_latent.squeeze(0).cpu().numpy()   # (Np, D)
    tgt_latent  = tgt_latent.squeeze(0).cpu().numpy()    # (Np, D)

    # 4.5 计算两种误差
    mse_errors = np.mean((pred_latent - tgt_latent)**2, axis=-1)  # (Np,)
    baseline_errors = np.mean((pred_latent - baseline_latent)**2, axis=-1)  # (Np,)

    return mse_errors, baseline_errors, tgt_latent, pred_latent

# ---------------------------------------------------
# 5. 在验证集上搜索最优阈值
# ---------------------------------------------------
# 5.1 计算baseline latent
baseline_latent = calculate_baseline_latents()

# 5.2 收集所有验证窗口的误差和标签
val_mse_errors = []
val_baseline_errors = []
val_true_labels = []

for i in range(NUM_WINDOWS_VAL):
    # 提取第 i 个窗口的误差
    mse_errors, baseline_errors, _, _ = extract_window_errors(val_x[i], val_y[i], baseline_latent)
    val_mse_errors.append(mse_errors)
    val_baseline_errors.append(baseline_errors)
    val_true_labels.append(val_labels[i].reshape(-1))

val_mse_errors = np.concatenate(val_mse_errors)
val_baseline_errors = np.concatenate(val_baseline_errors)
val_true_labels = np.concatenate(val_true_labels)

# 打印调试信息
print("\n[DEBUG] 验证集数据分布:")
print(f"验证集标签分布: {np.bincount(val_true_labels)}")
print(f"验证集标签比例: {np.mean(val_true_labels):.4f}")
print(f"MSE误差范围: [{val_mse_errors.min():.6f}, {val_mse_errors.max():.6f}]")
print(f"Baseline误差范围: [{val_baseline_errors.min():.6f}, {val_baseline_errors.max():.6f}]")

# 5.3 计算不同阈值下的指标
precision_vals, recall_vals, thresholds = precision_recall_curve(
    val_true_labels,
    val_mse_errors  # 使用MSE误差作为主要指标
)

# 5.4 计算每个阈值对应的 F1
f1_scores = 2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1] + 1e-8)

# 5.5 找到使 F1 最大的阈值
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_prec = precision_vals[best_idx]
best_rec = recall_vals[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n验证集上搜索得到的最优阈值 = {best_threshold:.6f}")
print(f"在该阈值下，Precision = {best_prec:.4f}, Recall = {best_rec:.4f}, F1 = {best_f1:.4f}")

# 计算验证集 AUC-ROC
val_auc = roc_auc_score(val_true_labels, val_mse_errors)
print(f"验证集 AUC-ROC = {val_auc:.4f}")

# ---------------------------------------------------
# 6. 在测试集上评估
# ---------------------------------------------------
# 6.1 收集所有测试窗口的预测结果
test_preds = []
test_true_labels = []
test_mse_errors = []
test_baseline_errors = []

for i in range(NUM_WINDOWS_TEST):
    # 提取第 i 个窗口的误差
    mse_errors, baseline_errors, _, _ = extract_window_errors(test_x[i], test_y[i], baseline_latent)
    test_mse_errors.append(mse_errors)
    test_baseline_errors.append(baseline_errors)
    test_true_labels.append(test_labels[i].reshape(-1))
    
    # 使用MSE误差和baseline误差的组合进行预测
    combined_errors = np.maximum(mse_errors, baseline_errors)  # 取两种误差的最大值
    window_preds = (combined_errors > best_threshold).astype(int)
    test_preds.append(window_preds)

test_preds = np.concatenate(test_preds)
test_true_labels = np.concatenate(test_true_labels)
test_mse_errors = np.concatenate(test_mse_errors)
test_baseline_errors = np.concatenate(test_baseline_errors)

# 6.2 计算测试集指标
test_prec = precision_score(test_true_labels, test_preds, pos_label=1)
test_rec = recall_score(test_true_labels, test_preds, pos_label=1)
test_f1 = f1_score(test_true_labels, test_preds, pos_label=1)
test_auc = roc_auc_score(test_true_labels, test_mse_errors)

print("\n=== 测试集最终结果（threshold = %.6f） ===" % best_threshold)
print(f"Test Precision = {test_prec:.4f}")
print(f"Test Recall    = {test_rec:.4f}")
print(f"Test F1-Score  = {test_f1:.4f}")
print(f"Test AUC-ROC   = {test_auc:.4f}")

# 打印两种误差的分布情况
print("\n[DEBUG] 测试集误差分布:")
print(f"MSE误差均值: {test_mse_errors.mean():.6f}")
print(f"Baseline误差均值: {test_baseline_errors.mean():.6f}")
print(f"组合误差均值: {np.maximum(test_mse_errors, test_baseline_errors).mean():.6f}")

# ---------------------------------------------------
# 7. 构建判别器数据集
# ---------------------------------------------------
class AnomalyDataset(Dataset):
    def __init__(self, x_patches, y_patches, labels, encoder, predictor, baseline_latent, device):
        """
        Args:
            x_patches: np.ndarray, shape = (N, Np, T, F)
            y_patches: np.ndarray, shape = (N, Np, T, F)
            labels: np.ndarray, shape = (N, Np, 1)
            encoder: JEPA encoder
            predictor: JEPA predictor
            baseline_latent: np.ndarray, shape = (D,)
            device: torch.device
        """
        self.device = device
        self.baseline_latent = baseline_latent
        
        # 1. 提取所有patch的特征
        features = []
        true_labels = []
        
        print("提取特征...")
        for i in range(len(x_patches)):
            # 获取当前窗口的数据
            x_window = x_patches[i]  # (Np, T, F)
            y_window = y_patches[i]  # (Np, T, F)
            window_labels = labels[i].reshape(-1)  # (Np,)
            
            # 转换为tensor
            x_tensor = torch.from_numpy(x_window).float().to(device).unsqueeze(0)  # (1, Np, T, F)
            y_tensor = torch.from_numpy(y_window).float().to(device).unsqueeze(0)  # (1, Np, T, F)
            
            with torch.no_grad():
                # 获取latent表示
                ctx_latent = encoder(x_tensor)  # (1, Np, D)
                tgt_latent = encoder(y_tensor)  # (1, Np, D)
                pred_latent, _ = predictor(ctx_latent, tgt_latent)  # (1, Np, D)
            
            # 转换为numpy
            pred_latent = pred_latent.squeeze(0).cpu().numpy()  # (Np, D)
            tgt_latent = tgt_latent.squeeze(0).cpu().numpy()    # (Np, D)
            
            # 对每个patch构造特征
            for j in range(len(pred_latent)):
                # 计算MSE误差
                mse_error = np.mean((pred_latent[j] - tgt_latent[j])**2)
                baseline_error = np.mean((pred_latent[j] - baseline_latent)**2)
                
                # 构造特征向量
                feature = np.concatenate([
                    pred_latent[j],                    # (D,)
                    pred_latent[j] - tgt_latent[j],    # (D,)
                    pred_latent[j] - baseline_latent,  # (D,)
                    np.array([mse_error]),            # (1,)
                    np.array([baseline_error])        # (1,)
                ])
                
                features.append(feature)
                true_labels.append(window_labels[j])
        
        self.features = np.array(features)  # (N_total, 3*D+2)
        self.labels = np.array(true_labels)  # (N_total,)
        
        print(f"数据集大小: {len(self.features)}")
        print(f"特征维度: {self.features.shape[1]}")
        print(f"标签分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

# ---------------------------------------------------
# 8. 定义判别器模型
# ---------------------------------------------------
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.mlp(x)

# ---------------------------------------------------
# 9. 训练判别器
# ---------------------------------------------------
def train_detector(train_dataset, val_dataset, input_dim, device, 
                  batch_size=64, num_epochs=50, lr=1e-3):
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = AnomalyDetector(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    best_val_f1 = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                probs = torch.sigmoid(outputs)
                
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # 计算验证集指标
        val_auc = roc_auc_score(val_labels, val_preds)
        val_ap = average_precision_score(val_labels, val_preds)
        
        # 使用最佳阈值计算F1
        precision, recall, thresholds = precision_recall_curve(val_labels, val_preds)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_f1 = np.max(f1_scores)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, F1: {best_f1:.4f}")
        
        # 保存最佳模型
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_model = model.state_dict().copy()
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model

# ---------------------------------------------------
# 10. 主函数
# ---------------------------------------------------
def main():
    # 1. 加载数据
    print("加载数据...")
    train_data = np.load(TRAIN_NPZ_PATH)
    val_data = np.load(VAL_NPZ_PATH)
    test_data = np.load(TEST_NPZ_PATH)
    
    train_labels_data = np.load(TRAIN_LABEL_PATH)
    val_labels_data = np.load(VAL_LABEL_PATH)
    test_labels_data = np.load(TEST_LABEL_PATH)
    
    # 提取数据
    train_x = train_data["x_patches"]  # (2648, 9, 20, 55)
    train_y = train_data["y_patches"]  # (2648, 9, 20, 55)
    val_x = val_data["x_patches"]      # (294, 9, 20, 55)
    val_y = val_data["y_patches"]      # (294, 9, 20, 55)
    test_x = test_data["x_patches"]    # (735, 9, 20, 55)
    test_y = test_data["y_patches"]    # (735, 9, 20, 55)
    
    train_labels = train_labels_data["x_labels"]  # (2648, 9, 1)
    val_labels = val_labels_data["x_labels"]      # (294, 9, 1)
    test_labels = test_labels_data["x_labels"]    # (735, 9, 1)
    
    print(f"训练集大小: {len(train_x)}")
    print(f"验证集大小: {len(val_x)}")
    print(f"测试集大小: {len(test_x)}")
    
    # 2. 计算baseline latent
    baseline_latent = calculate_baseline_latents()
    
    # 3. 构建数据集
    print("\n构建数据集...")
    train_dataset = AnomalyDataset(
        train_x, train_y, train_labels, 
        encoder, predictor, baseline_latent, DEVICE
    )
    val_dataset = AnomalyDataset(
        val_x, val_y, val_labels,
        encoder, predictor, baseline_latent, DEVICE
    )
    
    # 4. 训练判别器
    print("\n开始训练判别器...")
    input_dim = train_dataset.features.shape[1]
    detector = train_detector(
        train_dataset, val_dataset, input_dim, DEVICE,
        batch_size=64, num_epochs=50, lr=1e-3
    )
    
    # 5. 在测试集上评估
    print("\n在测试集上评估...")
    test_dataset = AnomalyDataset(
        test_x, test_y, test_labels,
        encoder, predictor, baseline_latent, DEVICE
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    detector.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = detector(features)
            probs = torch.sigmoid(outputs)
            
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    # 计算测试集指标
    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    
    # 使用最佳阈值计算F1
    precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    print("\n=== 测试集最终结果 ===")
    print(f"最佳阈值: {best_threshold:.6f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    print(f"Test F1: {best_f1:.4f}")
    
    # 保存模型
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model_state_dict': detector.state_dict(),
        'best_threshold': best_threshold
    }, 'model/anomaly_detector.pt')
    print("\n模型已保存到 model/anomaly_detector.pt")

if __name__ == "__main__":
    main()
