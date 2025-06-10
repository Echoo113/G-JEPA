import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# --- 加入项目路径 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 导入模块 ---
from patch_loader import create_labeled_loader
from jepa.encoder import MyTimeSeriesEncoder
from jepa.predictor import JEPPredictor

# ========== 分类器结构 ==========
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/jepa_best.pt"
TEST_FEATURE_PATH = "data/MSL/patches/msl_final_test.npz"
TEST_LABEL_PATH = "data/MSL/patches/msl_final_test_labels.npz"
BATCH_SIZE = 128


@torch.no_grad()
def evaluate_model(encoder, classifier, data_loader, model_name):
    print("-" * 50)
    print(f"正在评估: {model_name}")
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []

    for x_batch, y_batch, x_label, y_label in data_loader:
        x_batch = x_batch.to(DEVICE)
        labels_batch = y_label.float()  # 使用目标序列的标签

        latent = encoder(x_batch)  # shape: [B, SEQ, D]
        B, SEQ, D = latent.shape
        latent_flat = latent.reshape(B * SEQ, D)
        labels_flat = labels_batch.reshape(B * SEQ, 1).numpy()

        logits = classifier(latent_flat)  # [B*SEQ, 1]
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels_flat)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # === 评估指标 ===
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"[{model_name}] 评估指标:")
    print(f"  - F1 Score:  {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print("  - Confusion Matrix:")
    print(cm)
    print("-" * 50)


def load_pretrained_jepa(checkpoint_path: str):
    """
    加载预训练的 JEPA 模型组件：
    - encoder_online
    - encoder_ema (target encoder)
    - predictor
    - classifier1（用于pred_latent）
    - classifier2（用于tgt_latent）
    
    参数:
        checkpoint_path (str): 模型权重文件路径，例如 "model/jepa_best.pt"
    
    返回:
        Tuple[encoder_online, encoder_ema, predictor, classifier1, classifier2]
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {checkpoint_path}")

    print(f"📦 正在加载预训练模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']

    # === 构建 encoder 配置 ===
    encoder_config = {
        "patch_length": config["patch_length"],
        "num_vars": config["num_vars"],
        "latent_dim": config["latent_dim"],
        "time_layers": 2,
        "patch_layers": 3,
        "num_attention_heads": 16,
        "ffn_dim": config["latent_dim"] * 4,
        "dropout": 0.2
    }

    # === 初始化模型结构 ===
    encoder_online = MyTimeSeriesEncoder(**encoder_config).to(DEVICE)
    encoder_ema    = MyTimeSeriesEncoder(**encoder_config).to(DEVICE)
    predictor      = JEPPredictor(
        latent_dim=config["latent_dim"],
        num_layers=3,
        num_heads=16,
        ffn_dim=config["latent_dim"] * 4,
        dropout=0.2,
        prediction_length=config["prediction_length"]
    ).to(DEVICE)
    classifier1    = Classifier(input_dim=config["latent_dim"]).to(DEVICE)
    classifier2    = Classifier(input_dim=config["latent_dim"]).to(DEVICE)

    # === 加载状态字典 ===
    encoder_online.load_state_dict(checkpoint["encoder_online_state_dict"])
    encoder_ema.load_state_dict(checkpoint["encoder_ema_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    classifier1.load_state_dict(checkpoint["classifier1_state_dict"])
    classifier2.load_state_dict(checkpoint["classifier2_state_dict"])

    print("✅ 所有模型组件加载完毕！")
    
    return encoder_online, encoder_ema, predictor, classifier1, classifier2


def main():
    print("正在加载模型检查点...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    
    
    config = checkpoint["config"]

    # === 提取 Encoder 所需配置 ===
    encoder_config = {
        'patch_length': config['patch_length'],
        'num_vars': config['num_vars'],
        'latent_dim': config['latent_dim'],
        'time_layers': 2,
        'patch_layers': 3,
        'num_attention_heads': 16,
        'ffn_dim': config['latent_dim'] * 4,
        'dropout': 0.2
    }

    # === 初始化模型结构 ===
    encoder_online = MyTimeSeriesEncoder(**encoder_config).to(DEVICE)
    encoder_ema = MyTimeSeriesEncoder(**encoder_config).to(DEVICE)
    classifier1 = Classifier(input_dim=config["latent_dim"]).to(DEVICE)
    classifier2 = Classifier(input_dim=config["latent_dim"]).to(DEVICE)

    # === 加载权重 ===
    print("\n📦 正在加载模型权重...")
    try:
        encoder_online.load_state_dict(checkpoint["encoder_online_state_dict"])
        print("✅ encoder_online 加载成功")
    except KeyError as e:
        print(f"❌ encoder_online 加载失败: {e}")
        raise

    try:
        encoder_ema.load_state_dict(checkpoint["encoder_target_state_dict"])
        print("✅ encoder_ema 加载成功")
    except KeyError as e:
        print(f"❌ encoder_ema 加载失败: {e}")
        raise

    try:
        classifier1.load_state_dict(checkpoint["classifier1_state_dict"])
        print("✅ classifier1 加载成功")
    except KeyError as e:
        print(f"❌ classifier1 加载失败: {e}")
        raise

    try:
        classifier2.load_state_dict(checkpoint["classifier2_state_dict"])
        print("✅ classifier2 加载成功")
    except KeyError as e:
        print(f"❌ classifier2 加载失败: {e}")
        raise

    print("\n✅ 所有模型权重加载成功")

    # === 加载测试集 ===
    test_loader = create_labeled_loader(
        feature_npz_path=TEST_FEATURE_PATH,
        label_npz_path=TEST_LABEL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # === 执行评估 ===
    evaluate_model(encoder_online, classifier1, test_loader, "Online Encoder + Classifier1")
    evaluate_model(encoder_ema, classifier2, test_loader, "EMA Encoder + Classifier2")


if __name__ == "__main__":
    print("🚀 开始测试集推理与评估")
    main()
