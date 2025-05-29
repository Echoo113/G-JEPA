import torch
import numpy as np
from encoder import MyTimeSeriesEncoder, prepare_batch_from_np

def main():
    # 设置batch size
    BATCH_SIZE = 8  # 在这里控制batch size
    
    # 1. 创建一些示例数据
    total_patches = 100  # 总patch数
    patch_length = 30    # 每个patch的时间长度
    num_vars = 137      # 特征维度
    
    # 创建随机numpy数组作为输入
    np_data = np.random.randn(total_patches, patch_length, num_vars)
    
    # 2. 使用prepare_batch_from_np转换为batched tensor，传入batch_size
    batched_data = prepare_batch_from_np(np_data, batch_size=BATCH_SIZE)  # shape: (B, N, T, F)
    print(f"Batched data shape: {batched_data.shape}")
    
    # 3. 初始化encoder
    encoder = MyTimeSeriesEncoder(
        patch_length=patch_length,
        num_vars=num_vars,
        latent_dim=64,
        num_layers=2,
        num_attention_heads=2
    )
    
    # 4. 前向传播
    encoded = encoder(batched_data)
    print(f"Encoded output shape: {encoded.shape}")  # 应该是 (B, N, latent_dim)
    
    # 5. 验证batch size是否正确
    assert encoded.shape[0] == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}, got {encoded.shape[0]}"
    
    print("\nExample completed successfully!")
    print(f"Input shape: {batched_data.shape}")
    print(f"Output shape: {encoded.shape}")

if __name__ == "__main__":
    main()
