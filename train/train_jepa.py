import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from jepa.encoder import ContextEncoder, TargetEncoder
from jepa.predictor import JEPPredictor

# 全局超参：碗数
NUM_BOWLS = 500

def get_device():
    """检测是否有 CUDA 可用，如果有就用 CUDA，否则用 CPU"""
    if torch.cuda.is_available():
        print("Using CUDA")
        return 'cuda'
    else:
        print("CUDA not available, using CPU")
        return 'cpu'

class ArrayPatchDataset(Dataset):
    def __init__(self, ctx_list, tgt_list):
        assert len(ctx_list) == len(tgt_list)
        self.ctx = ctx_list
        self.tgt = tgt_list

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        # 先拿出来
        ctx = self.ctx[idx]
        tgt = self.tgt[idx]

        # 如果是 numpy array，就用 from_numpy；如果已经是 Tensor，就直接转 float
        if isinstance(ctx, np.ndarray):
            ctx = torch.from_numpy(ctx).float()
        elif isinstance(ctx, torch.Tensor):
            ctx = ctx.float()
        else:
            raise TypeError(f"Unsupported type for ctx: {type(ctx)}")

        if isinstance(tgt, np.ndarray):
            tgt = torch.from_numpy(tgt).float()
        elif isinstance(tgt, torch.Tensor):
            tgt = tgt.float()
        else:
            raise TypeError(f"Unsupported type for tgt: {type(tgt)}")

        return {'context_patches': ctx, 'target_patches': tgt}

def train_on_split(
    split_name: str,
    ctx_arr,
    tgt_arr,
    batch_size=16,
    lr=1e-4,
    epochs=20,
    device=None  # 如果为 None，则自动检测
):
    """在单个 split（short 或 long）上跑训练"""
    if device is None:
        device = get_device()
        
    print(f"\n=== Training split: {split_name} ===")
    print(f"Context array type: {type(ctx_arr)}")
    print(f"Target array type: {type(tgt_arr)}")
    
    # 确保数据是 numpy 数组
    if isinstance(ctx_arr, list):
        ctx_arr = np.array(ctx_arr)
    if isinstance(tgt_arr, list):
        tgt_arr = np.array(tgt_arr)
    
    # 1) 计算每碗要装多少片 patch
    N_ctx, N_tgt = len(ctx_arr), len(tgt_arr)
    ctx_per_bowl = N_ctx // NUM_BOWLS
    tgt_per_bowl = N_tgt // NUM_BOWLS
    
    print(f"Original shapes - Context: {ctx_arr.shape}, Target: {tgt_arr.shape}")
    print(f"Patches per bowl - Context: {ctx_per_bowl}, Target: {tgt_per_bowl}")
    
    # 2) 丢掉余下的，保留整除部分
    ctx_used = ctx_arr[: ctx_per_bowl * NUM_BOWLS]
    tgt_used = tgt_arr[: tgt_per_bowl * NUM_BOWLS]
    
    # 3) 重塑成 [NUM_BOWLS, patches_per_bowl, patch_len]
    ctx_bowls = ctx_used.reshape(NUM_BOWLS, ctx_per_bowl, -1)
    tgt_bowls = tgt_used.reshape(NUM_BOWLS, tgt_per_bowl, -1)
    
    print(f"After reshaping - Context: {ctx_bowls.shape}, Target: {tgt_bowls.shape}")
    
    ds = ArrayPatchDataset(ctx_bowls, tgt_bowls)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # 从第一个样本推断维度
    sample_ctx = ds[0]['context_patches']
    N_ctx, feat_dim = sample_ctx.shape
    sample_tgt = ds[0]['target_patches']
    N_tgt, _ = sample_tgt.shape
    latent_dim = 16  # 或者你想用的其他值

    # new model instances per split
    enc_ctx   = ContextEncoder(input_dim=feat_dim, latent_dim=latent_dim).to(device)
    enc_tgt   = TargetEncoder(input_dim=feat_dim, latent_dim=latent_dim).to(device)
    predictor = JEPPredictor(
        latent_dim        = latent_dim,
        context_length    = ctx_per_bowl,  # 使用每碗的 patch 数作为上下文长度
        prediction_length = tgt_per_bowl,  # 使用每碗的 patch 数作为预测长度
    ).to(device)

    opt = torch.optim.Adam(
        list(enc_ctx.parameters()) +
        list(enc_tgt.parameters()) +
        list(predictor.parameters()),
        lr=lr
    )

    # 保存到各自子目录
    ckpt_dir = os.path.join('checkpoints', split_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        enc_ctx.train(); enc_tgt.train(); predictor.train()
        total_loss = 0.0

        for batch in loader:
            ctx_raw = batch['context_patches'].to(device)
            tgt_raw = batch['target_patches'].to(device)

            ctx_lat = enc_ctx(ctx_raw)
            tgt_lat = enc_tgt(tgt_raw)

            preds, loss = predictor(ctx_lat, tgt_lat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * ctx_raw.size(0)

        avg_loss = total_loss / len(ds)
        print(f"[{split_name}] Epoch {epoch:02d} — avg loss: {avg_loss:.4f}")

def train_all_splits(
    ctx_pkl, tgt_pkl,
    splits=None,
    **train_kwargs
):
    """一次性对所有 split 依次训练"""
    # 1) 加载两个 dict
    with open(ctx_pkl, 'rb') as f:
        ctx_dict = pickle.load(f)
    with open(tgt_pkl, 'rb') as f:
        tgt_dict = pickle.load(f)

    print("\nData structure info:")
    print("Context dict keys:", list(ctx_dict.keys()))
    print("Target dict keys:", list(tgt_dict.keys()))
    
    for split in ctx_dict:
        print(f"\n{split} split info:")
        print("Context:", ctx_dict[split].keys())
        print("Target:", tgt_dict[split].keys())

    if splits is None:
        splits = list(ctx_dict.keys())  # e.g. ['short','long']

    # 2) 对每个 split 调用 train_on_split
    for split in splits:
        ctx_arr = ctx_dict[split]['patches']  # 注意这里要取 'patches' 键
        tgt_arr = tgt_dict[split]['patches']  # 注意这里要取 'patches' 键
        train_on_split(split, ctx_arr, tgt_arr, **train_kwargs)

if __name__ == '__main__':
    # 路径按需修改
    ctx_pkl = 'data/SWAT/patches/all_patches.pkl'
    tgt_pkl = 'data/SWAT/pseudo_future_patches/all_pseudo_future_patches.pkl'

    train_all_splits(
        ctx_pkl, tgt_pkl,
        batch_size=16,
        lr=1e-4,
        epochs=20,
        device=None  # 自动检测设备
    ) 