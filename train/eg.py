import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ==== Step 1: 加载训练数据 ====
data = np.load("data/MSL/patches/train.npz")
x_patches = data["x_patches"]  # (B, P, T, F)
labels = data["x_label"]       # (B,)

# ==== Step 2: Instance Normalization（窗口级）====
x_tensor = torch.tensor(x_patches, dtype=torch.float32)  # (B, P, T, F)
x_reshaped = x_tensor.permute(0, 3, 1, 2).reshape(x_tensor.shape[0], 1, -1)  # (B, 1, P*T)
mean = x_reshaped.mean(dim=2, keepdim=True)
std = x_reshaped.std(dim=2, keepdim=True) + 1e-6
x_norm = ((x_reshaped - mean) / std).view(x_tensor.shape[0], 1, x_tensor.shape[1], x_tensor.shape[2])
x_norm = x_norm.permute(0, 2, 3, 1)  # (B, P, T, F)

# ==== Step 3: 加载 Encoder ====
from jepa.encoder import MyTimeSeriesEncoder
import torch.nn as nn

# 构造一个临时的包含 Aggregator 的模型，只取出 [CLS] Token
class CLSExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MyTimeSeriesEncoder(patch_length=16, num_vars=1, latent_dim=128)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.aggregator = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        patch_latents = self.encoder(x)                 # (B, P, latent_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, latent_dim)
        full_sequence = torch.cat([cls_token, patch_latents], dim=1)  # (B, P+1, latent_dim)
        aggregated = self.aggregator(full_sequence)     # (B, P+1, latent_dim)
        return aggregated[:, 0]                         # 只取 CLS Token

# 加载 Encoder 权重
model = CLSExtractor()
state_dict = torch.load("model/best_encoder.pth")["online_encoder"]
model.encoder.load_state_dict(state_dict, strict=False)
model.eval()

# ==== Step 4: 提取 CLS Token ====
with torch.no_grad():
    cls_latent = model(x_norm)  # (B, latent_dim)
    cls_latent_np = cls_latent.numpy()

# ==== Step 5: t-SNE 降维并可视化 ====
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
latent_2d = tsne.fit_transform(cls_latent_np)

plt.figure(figsize=(8, 6))
plt.scatter(latent_2d[labels == 0, 0], latent_2d[labels == 0, 1], label='Normal', alpha=0.6, s=10)
plt.scatter(latent_2d[labels == 1, 0], latent_2d[labels == 1, 1], label='Anomaly', alpha=0.6, s=10, c='red')
plt.legend()
plt.grid(True)
plt.title("t-SNE of [CLS] Token Embedding (Train Set)")
plt.tight_layout()
plt.savefig("tsne_cls_train.png", dpi=300)
plt.show()
