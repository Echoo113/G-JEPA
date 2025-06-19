"""
主训练脚本：
- 调用您在encoder.py中定义的MyTimeSeriesEncoder。
- 使用[CLS] Token进行智能聚合，替代Mean Pooling。
- 支持窗口级Instance Normalization，并正确处理重建损失。
- 使用多任务学习（重建+分类）来训练一个Online Encoder。
- 同时通过EMA（指数移动平均）来维护一个影子EMA Encoder。
- 基于验证集F1-Score进行Early Stopping和模型保存。
- 将最佳的Online Encoder和EMA Encoder权重保存在同一个文件中。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import copy

# 导入您已经写好的Encoder
from jepa.encoder import MyTimeSeriesEncoder

# ===================================================================
# 步骤 0: 准备工作 - 定义所有模型架构
# ===================================================================

class EncoderWithHeads(nn.Module):
    """
    一个包含Encoder和双'头'的模块化组件。
    此版本使用[CLS] Token进行智能聚合。
    """
    def __init__(
        self,
        num_patches: int, 
        patch_length: int, 
        num_vars: int, 
        latent_dim: int, 
        agg_layers: int = 1, # 聚合Transformer的层数
        agg_heads: int = 8,  # 聚合Transformer的头数
        **encoder_kwargs     # 接收所有encoder需要的其他参数
    ):
        super().__init__()
        
        # 核心引擎: 调用您写好的Encoder
        self.encoder = MyTimeSeriesEncoder(
            patch_length=patch_length, num_vars=num_vars, latent_dim=latent_dim, **encoder_kwargs
        )
        
        # 聚合层: 使用一个[CLS] Token和Transformer层来智能聚合patch序列
        self.window_cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        agg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=agg_heads, dim_feedforward=latent_dim * 4,
            dropout=encoder_kwargs.get('dropout', 0.1), batch_first=True, activation='gelu'
        )
        self.aggregator = nn.TransformerEncoder(agg_encoder_layer, num_layers=agg_layers)

        # 还原部门: 重建头
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, num_patches * patch_length * num_vars)
        )
        
        # 质检部门: 异常分类头
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(latent_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T, F = x.shape
        
        # 1. Encoder加工，得到Patch的上下文表示
        patch_latents = self.encoder(x)
        
        # 2. 智能聚合，得到代表整个窗口的向量z
        cls_tokens = self.window_cls_token.expand(B, -1, -1)
        full_sequence = torch.cat([cls_tokens, patch_latents], dim=1)
        aggregated_sequence = self.aggregator(full_sequence)
        z = aggregated_sequence[:, 0] # 只取[CLS] Token的最终输出
        
        # 3. 将z送入双头
        x_recon_flat = self.reconstruction_head(z)
        x_recon = x_recon_flat.view(B, N, T, F)
        logit = self.anomaly_head(z)
        
        return x_recon, logit

class EMAModel(nn.Module):
    """顶层模型，管理Online网络和EMA网络。"""
    def __init__(self, model_args: dict, ema_momentum: float):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.online_network = EncoderWithHeads(**model_args)
        self.ema_network = copy.deepcopy(self.online_network)
        for param in self.ema_network.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_ema_network(self):
        """执行EMA更新"""
        for online_param, ema_param in zip(self.online_network.parameters(), self.ema_network.parameters()):
            ema_param.data.mul_(self.ema_momentum).add_(online_param.data, alpha=1 - self.ema_momentum)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 训练时，只有online_network执行前向传播
        return self.online_network(x)

# ==================================
#  主训练脚本
# ==================================
if __name__ == '__main__':
    # --- 超参数与设置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 120
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-4
    EMA_MOMENTUM = 0.999
    LATENT_DIM = 128
    LAMBDA_RECON = 1.0
    LAMBDA_ANOMALY = 5.0
    EARLY_STOP_PATIENCE = 15
    USE_INSTANCE_NORM = True 
    
    # --- 文件路径 ---
    DATA_DIR = "data/MSL/patches"
    SAVE_MODEL_PATH = "model/best_encoder.pth" 

    # --- 模型与数据参数 ---
    NUM_PATCHES, PATCH_LENGTH, NUM_VARS = 5, 16, 1

    # 步骤 1: 数据加载与准备
    print("--- 步骤 1: 数据加载与准备 ---")
    try:
        train_data_file = np.load(os.path.join(DATA_DIR, "train.npz"))
        X_train = torch.tensor(train_data_file['x_patches'], dtype=torch.float32)
        X_train_label = torch.tensor(train_data_file['x_label'], dtype=torch.float32)

        val_data_file = np.load(os.path.join(DATA_DIR, "val.npz"))
        X_val = torch.tensor(val_data_file['x_patches'], dtype=torch.float32)
        X_val_label = torch.tensor(val_data_file['x_label'], dtype=torch.float32)
    except FileNotFoundError:
        print(f"错误: 无法在 '{DATA_DIR}' 目录下找到数据文件。请确保路径正确。")
        exit()

    train_dataset = TensorDataset(X_train, X_train_label)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_val, X_val_label)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"数据加载完毕. 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}\n")

    # 步骤 2: 实例化模型, 损失函数, 优化器
    print("--- 步骤 2: 实例化模型, 损失函数, 优化器 ---")
    model_args = {
        'num_patches': NUM_PATCHES, 'patch_length': PATCH_LENGTH, 'num_vars': NUM_VARS, 
        'latent_dim': LATENT_DIM, 'time_layers': 2, 'patch_layers': 3, 
        'num_attention_heads': 8, 'dropout': 0.1,
        'agg_layers': 1, 'agg_heads': 8, 'use_instance_norm': USE_INSTANCE_NORM
    }
    model = EMAModel(model_args=model_args, ema_momentum=EMA_MOMENTUM).to(DEVICE)
    
    criterion_recon = nn.MSELoss()
    criterion_anomaly = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.online_network.parameters(), lr=LEARNING_RATE)
    print("模型及优化器准备就绪.\n")

    # --- 初始化训练状态变量 ---
    best_val_recon_loss = float('inf')
    patience_counter = 0

    # 步骤 3, 4, 5: 开始训练, 验证与学习的循环
    print("--- 步骤 3, 4, 5: 开始训练与验证循环 ---")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0
        
        for x_patches_raw, x_label in train_loader:
            x_patches_raw, x_label = x_patches_raw.to(DEVICE), x_label.to(DEVICE).unsqueeze(1)
            
            # 步骤 3a: 对输入进行预处理 (一次性归一化)
            # 这一步实现了您"计算一次，复用结果"的优化思路
            input_for_encoder = model.online_network.encoder.instance_norm(
                x_patches_raw.permute(0, 3, 1, 2).reshape(x_patches_raw.shape[0], NUM_VARS, -1)
            ).view(x_patches_raw.shape[0], NUM_VARS, NUM_PATCHES, PATCH_LENGTH).permute(0, 2, 3, 1) if USE_INSTANCE_NORM else x_patches_raw

            # 步骤 3b: 前向传播
            x_recon, logit = model(input_for_encoder)
            
            # 步骤 4: 计算损失 (重建损失的目标是归一化后的输入)
            loss_recon = criterion_recon(x_recon, input_for_encoder)
            loss_anomaly = criterion_anomaly(logit, x_label)
            total_loss = LAMBDA_RECON * loss_recon + LAMBDA_ANOMALY * loss_anomaly
            
            # 步骤 5: 学习与成长
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model._update_ema_network()

            train_loss_epoch += total_loss.item()
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] |loss anomaly: {loss_anomaly:.4f} | loss recon: {loss_recon:.4f} | train_loss: {avg_train_loss:.4f}")

        # --- 验证循环 (专注于重建损失) ---
        model.eval()
        total_recon_loss = 0.0  # 累计重建损失
        total_anomaly_loss = 0.0  # 累计异常损失

        with torch.no_grad():
            for x_patches_val, x_label_val in val_loader:
                x_patches_val, x_label_val = x_patches_val.to(DEVICE), x_label_val.to(DEVICE).unsqueeze(1)
                
                # 验证时也需要对输入进行同样的IN处理
                input_val = model.online_network.encoder.instance_norm(
                    x_patches_val.permute(0, 3, 1, 2).reshape(x_patches_val.shape[0], NUM_VARS, -1)
                ).view(x_patches_val.shape[0], NUM_VARS, NUM_PATCHES, PATCH_LENGTH).permute(0, 2, 3, 1) if USE_INSTANCE_NORM else x_patches_val
                
                # 获取重建输出和分类logits
                x_recon_val, logit_val = model.online_network(input_val)
                
                # 计算重建损失和异常损失
                recon_loss = criterion_recon(x_recon_val, input_val)
                anomaly_loss = criterion_anomaly(logit_val, x_label_val)
                
                total_recon_loss += recon_loss.item()
                total_anomaly_loss += anomaly_loss.item()
        
        # 计算平均损失
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_anomaly_loss = total_anomaly_loss / len(val_loader)
        
        # --- Debug: 打印验证集损失 ---
        print(f"  -> 验证集 | 重建损失: {avg_recon_loss:.6f} | 异常损失: {avg_anomaly_loss:.6f}")

        # 步骤 6: 保存最佳成果并检查Early Stopping
        if avg_recon_loss < best_val_recon_loss:
            best_val_recon_loss = avg_recon_loss
            patience_counter = 0
            state_to_save = {
                'online_encoder': model.online_network.encoder.state_dict(),
                'ema_encoder': model.ema_network.encoder.state_dict(),
                'epoch': epoch + 1,
                'recon_loss': avg_recon_loss
            }
            torch.save(state_to_save, SAVE_MODEL_PATH)
            print(f"  -> 发现更优模型！重建损失降低到 {avg_recon_loss:.6f}")
        else:
            patience_counter += 1
            print(f"  -> 未发现更优模型。Early stopping耐心计数: {patience_counter}/{EARLY_STOP_PATIENCE}")

        print("-" * 50)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n触发Early Stopping！验证集重建损失已连续 {EARLY_STOP_PATIENCE} 轮未改善。")
            break
        
    print("\n--- 训练完成 ---")
    print(f"最佳验证集重建损失为: {best_val_recon_loss:.6f}")
    print(f"最优的Encoder模型权重已保存在: {SAVE_MODEL_PATH}")
  