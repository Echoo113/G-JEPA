import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 将项目根目录添加到Python路径，确保可以导入自定义模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 从你的项目中导入必要的模块 ---
from jepa.encoder import MyTimeSeriesEncoder
from patch_loader import AnomalyPatchExtractor # 假设你的数据加载器类叫这个名字

# ========= 配置参数 (必须与你的训练脚本保持一致!) =========
MODEL_PATH = "model/jepa_hybrid_best.pt"
TUNE_TRAIN_DATA_PATH = "data/MSL/patches/msl_tune_train.npz"
TUNE_TRAIN_LABELS_PATH = "data/MSL/patches/msl_tune_train_labels.npz"

# --- 模型和数据维度配置 ---
# 注意：这些参数必须和你训练时使用的完全一样！
# 根据你之前提供的数据维度，这里已经帮你填好了
PATCH_LENGTH = 20
NUM_VARS = 55
LATENT_DIM = 512 # 建议使用512，如果你训练时用了1024，请改回1024
PREDICTION_LENGTH = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    主执行函数
    """
    # ========= Step 1: 加载训练好的模型和Encoder权重 =========
    print(f"[Step 1] 正在从 {MODEL_PATH} 加载模型...")

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}。请先运行训练脚本。")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # 重新实例化一个与训练时结构相同的Encoder
    encoder_online = MyTimeSeriesEncoder(
        patch_length=PATCH_LENGTH,
        num_vars=NUM_VARS,
        latent_dim=checkpoint.get('latent_dim', LATENT_DIM), # 优先从文件中读取维度
        time_layers=2,
        patch_layers=3,
        num_attention_heads=16,
        ffn_dim=checkpoint.get('latent_dim', LATENT_DIM) * 4,
        dropout=0.2
    ).to(DEVICE)

    # 加载 online_encoder 的权重
    encoder_online.load_state_dict(checkpoint['encoder_online'])
    encoder_online.eval()  # 设置为评估模式
    print("Online Encoder 加载成功并已设置为评估模式。")

    # ========= Step 2: 加载微调数据集和标签 =========
    print("\n[Step 2] 正在加载微调数据集和标签...")

    try:
        x_patches_data, _ = AnomalyPatchExtractor.load_patch_split(TUNE_TRAIN_DATA_PATH)
        x_labels_data, _ = AnomalyPatchExtractor.load_patch_split(TUNE_TRAIN_LABELS_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保数据已生成。")
        return
        
    print(f"  - 微调数据 (x_patches) 维度: {x_patches_data.shape}")
    print(f"  - 微调标签 (x_labels) 维度: {x_labels_data.shape}")

    # ========= Step 3: 准备数据并提取特征 =========
    print("\n[Step 3] 正在准备数据并使用Encoder提取特征...")

    # 将数据从 (N, num_patches, patch_len, C) 展平为 (N * num_patches, patch_len, C)
    # 这样每个补丁都成为一个独立的样本
    num_samples, num_patches, _, _ = x_patches_data.shape
    x_patches_flat = x_patches_data.reshape(-1, PATCH_LENGTH, NUM_VARS)
    x_labels_flat = x_labels_data.reshape(-1)

    print(f"  - 已将数据展平，总补丁数: {x_patches_flat.shape[0]}")

    # 为了快速可视化，我们可以随机抽样一部分数据，例如2000个点
    # 如果数据量不大或者你想分析全部数据，可以注释掉这部分
    num_vis_samples = min(2000, x_patches_flat.shape[0])
    print(f"  - 将随机抽样 {num_vis_samples} 个点进行t-SNE可视化...")
    indices = np.random.choice(x_patches_flat.shape[0], num_vis_samples, replace=False)
    
    x_sample = torch.tensor(x_patches_flat[indices], dtype=torch.float32).to(DEVICE)
    labels_sample = x_labels_flat[indices]

    # 使用Encoder提取特征
    with torch.no_grad():
        latent_vectors = encoder_online(x_sample).cpu().numpy()

    print(f"  - 特征提取完成，特征向量维度: {latent_vectors.shape}")

    # ========= Step 4: 使用t-SNE进行降维和可视化 =========
    print("\n[Step 4] 正在使用t-SNE进行降维...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(latent_vectors)
    print("t-SNE降维完成。")

    print("正在绘制特征空间散点图...")
    plt.figure(figsize=(12, 10))
    
    # 分离正常点和异常点
    normal_indices = np.where(labels_sample == 0)
    anomaly_indices = np.where(labels_sample == 1)

    # 绘制正常点 (蓝色)
    plt.scatter(
        tsne_results[normal_indices, 0], 
        tsne_results[normal_indices, 1], 
        label='正常 (Normal)', 
        alpha=0.6, 
        c='steelblue',
        s=15 # 点的大小
    )
    # 绘制异常点 (红色)
    plt.scatter(
        tsne_results[anomaly_indices, 0], 
        tsne_results[anomaly_indices, 1], 
        label='异常 (Anomaly)', 
        alpha=0.9, 
        c='red',
        s=25 # 让异常点更突出
    )

    plt.title('Encoder输出特征的t-SNE二维可视化', fontsize=16)
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
