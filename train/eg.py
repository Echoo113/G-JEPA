import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

# --- 将项目根目录添加到Python路径，确保可以导入自定义模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 从你的项目中导入必要的模块 ---
# 假设这些模块可以被正确找到
from jepa.encoder import MyTimeSeriesEncoder

# ========= 配置参数 =========
# --- 路径配置 ---
PRETRAINED_MODEL_PATH = "model/jepa_best.pt"
TUNE_TRAIN_DATA_PATH = "data/MSL/patches/msl_tune_train.npz"
TUNE_TRAIN_LABELS_PATH = "data/MSL/patches/msl_tune_train_labels.npz"

# --- 可视化配置 ---
# 为了速度，我们可以只对一部分数据进行可视化。设置为 None 则使用全部数据。
SAMPLE_SIZE = 4000 
TSNE_PERPLEXITY = 40  # t-SNE的一个关键参数，通常在5-50之间
BATCH_SIZE = 256 # 用于特征提取的批次大小

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    主执行函数
    """
    # ========= Step 1: 加载训练好的模型和Encoder权重 =========
    print(f"[Step 1] 正在从 {PRETRAINED_MODEL_PATH} 加载预训练模型...")

    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"错误：找不到模型文件 '{PRETRAINED_MODEL_PATH}'。请先运行JEPA预训练脚本。")
        return

    # 加载checkpoint
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

    # 重新实例化一个与训练时结构相同的Encoder
    encoder = MyTimeSeriesEncoder(
        patch_length=config['patch_length'],
        num_vars=config['num_vars'],
        latent_dim=config['latent_dim'],
        time_layers=2, patch_layers=3, num_attention_heads=16,
        ffn_dim=config['latent_dim'] * 4, dropout=0.2
    ).to(DEVICE)

    # 加载 Online Encoder 的权重
    encoder.load_state_dict(checkpoint['encoder_target_state_dict'])
    encoder.eval()  # 设置为评估模式
    print("Online Encoder 加载成功并已设置为评估模式。")

    # ========= Step 2: 加载并重塑数据和标签 =========
    print("\n[Step 2] 正在加载并重塑微调数据集和标签...")

    try:
        features_data = np.load(TUNE_TRAIN_DATA_PATH)['x_patches']
        labels_data = np.load(TUNE_TRAIN_LABELS_PATH)['x_labels']
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保数据已生成。")
        return
        
    print(f"  - 原始数据维度: {features_data.shape}")
    print(f"  - 原始标签维度: {labels_data.shape}")

    # 将序列和补丁维度展平，为特征提取做准备
    # 特征: (N, Seq, PatchLen, Vars) -> (N*Seq, PatchLen, Vars)
    # 标签: (N, Seq) -> (N*Seq,)
    num_sequences, seq_len, patch_len, num_vars = features_data.shape
    features_flat = features_data.reshape(-1, patch_len, num_vars)
    labels_flat = labels_data.flatten()
    
    print(f"  - 重塑后数据维度 (所有补丁): {features_flat.shape}")
    print(f"  - 重塑后标签维度 (所有补丁): {labels_flat.shape}")

    # ========= Step 3: 使用Encoder批量提取特征 =========
    print("\n[Step 3] 正在批量提取特征向量...")

    dataset = TensorDataset(torch.from_numpy(features_flat).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(DEVICE)
            
            # Encoder期望输入有序列维度，所以为每个补丁增加一个虚拟的序列维度
            # (B, P, V) -> (B, 1, P, V)
            z = encoder(x_batch.unsqueeze(1))
            # 移除虚拟的序列维度，得到特征向量
            # (B, 1, D) -> (B, D)
            z = z.squeeze(1)
            
            all_embeddings.append(z.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"成功提取了 {all_embeddings.shape[0]} 个补丁的特征向量。")
    print(f"  - 特征向量维度: {all_embeddings.shape}")

    # ========= Step 4: 使用t-SNE降维并可视化 =========
    print(f"\n[Step 4] 正在使用t-SNE降维并进行可视化...")

    # 如果指定了样本大小，则进行随机采样
    num_total_points = all_embeddings.shape[0]
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < num_total_points:
        print(f"  - 将随机抽样 {SAMPLE_SIZE} / {num_total_points} 个点进行可视化...")
        indices = np.random.choice(num_total_points, SAMPLE_SIZE, replace=False)
        embeddings_to_plot = all_embeddings[indices]
        labels_to_plot = labels_flat[indices]
    else:
        embeddings_to_plot = all_embeddings
        labels_to_plot = labels_flat

    print("  - 正在运行 t-SNE... (这可能需要一些时间)")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_to_plot)
    print("  - t-SNE降维完成。")

    # 分离正常和异常点用于绘图
    normal_points = embeddings_2d[labels_to_plot == 0]
    anomaly_points = embeddings_2d[labels_to_plot == 1]
    
    print(f"  - 绘图点数: {len(embeddings_2d)} (正常: {len(normal_points)}, 异常: {len(anomaly_points)})")

    # 开始绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 12))
    
    # 绘制正常点 (蓝色)
    plt.scatter(
        normal_points[:, 0], normal_points[:, 1], 
        s=15, c='royalblue', alpha=0.6, label=f'Normal ({len(normal_points)})'
    )
    # 绘制异常点 (红色)
    plt.scatter(
        anomaly_points[:, 0], anomaly_points[:, 1], 
        s=40, c='red', marker='x', label=f'Anomaly ({len(anomaly_points)})'
    )

    plt.title('t-SNE Visualization of Encoder Feature Space', fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig('encoder_output_tsne.png')
   
if __name__ == "__main__":
    main()
