import torch
import torch.nn as nn
import os, random
import argparse

from sktime.datasets import load_from_tsfile_to_dataframe
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.uea import collate_fn

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#spark dataset
class SparkDataset(Dataset):
    def __init__(self, dataframe, patch_size=30, window_size=5, stride=5):
        self.X = []
        self.Y = []
        for sensor_col in dataframe.columns:
            signal = dataframe[sensor_col].values
            for i in range(0, len(signal) - patch_size - window_size, stride):
                x_patch = signal[i : i + patch_size]
                y_patch = signal[i + patch_size : i + patch_size + window_size]
                self.X.append(x_patch)
                self.Y.append(y_patch)
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'electricity': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'illness': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom
}


data_path_dict = {
    'ETTh1': "../data/timeseries_lib/long_term_forecast/ETT-small",
    'ETTh2': "../data/timeseries_lib/long_term_forecast/ETT-small",
    'ETTm1': "../data/timeseries_lib/long_term_forecast/ETT-small",
    'ETTm2': "../data/timeseries_lib/long_term_forecast/ETT-small",
    'electricity': "../data/timeseries_lib/long_term_forecast/electricity",
    'exchange_rate': "../data/timeseries_lib/long_term_forecast/exchange_rate",
    'illness': "../data/timeseries_lib/long_term_forecast/illness",
    'traffic': "../data/timeseries_lib/long_term_forecast/traffic",
    'weather': "../data/timeseries_lib/long_term_forecast/weather"
}


dataset_config_dict = {
    'ETTh1': (96, 48, [96, 192, 336, 720], 1),
    'ETTh2': (96, 48, [96, 192, 336, 720], 1),
    'ETTm1': (96, 48, [96, 192, 336, 720], 1),
    'ETTm2': (96, 48, [96, 192, 336, 720], 1),
    'electricity': (96, 48, [96, 192, 336, 720], 12),
    'exchange_rate': (96, 48, [96, 192, 336, 720], 1),
    'illness': (36, 18, [24, 36, 48, 60], 1),
    'traffic': (96, 48, [96, 192, 336, 720], 12),
    'weather': (96, 48, [96, 192, 336, 720], 1)
}

# model_dict = {
#     'small_cls-v2': "model/small_cls/pretrain-2/checkpoints/epoch=1-step=24294.ckpt",
#     'small_cls-v8': "model/small_cls/pretrain-8/VQShape.ckpt",
#     'small_cls-v9': "model/small_cls/pretrain-9/VQShape.ckpt",
#     'small-v1': 'model/small/pretrain-1/VQShape.ckpt',
#     'base-v1': 'model/base/pretrain-1/VQShape.ckpt'
# }


def get_loader(dataset_name, flag, seq_len=96, label_len=48, pred_len=96, max_uts_per_batch=1024):
    dataset = data_dict[dataset_name](
        root_path=data_path_dict[dataset_name],
        data_path=f"{dataset_name}.csv",
        subsample_rate=dataset_config_dict[dataset_name][-1],
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features='M'
    )
    print(flag, len(dataset))

    batch_size = max(1, int(max_uts_per_batch/dataset.data_x.shape[-1]))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, loader


class Normalizer:
    def __init__(self):
        self.mean = None
        self.var = None

    def fit_transform(self, x: torch.Tensor):
        self.mean = x.mean(dim=-1, keepdims=True)
        self.var = x.var(dim=-1, keepdims=True)
        return (x - self.mean) / (self.var + 1e-5).sqrt()
    
    def transform(self, x: torch.Tensor):
        return (x - self.mean) / (self.var + 1e-5).sqrt()
    
    def inverse_transform(self, x: torch.Tensor):
        return x * (self.var + 1e-5).sqrt() + self.mean


def finetune(lit_model, model_name, dataset_name, seq_len=96, pred_len=96, label_len=0, batch_size=64, num_epoch=10):
    torch.set_float32_matmul_precision('medium')
    model = lit_model.model
    model.codebook.entropy_loss = 0
    device = lit_model.device
    max_uts_per_batch = 512 if model_name.startswith('small') else 256
    
    df = pd.read_csv("ae_labels.csv")
    df = df.select_dtypes(include=[np.number])

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    for col in df.columns:
        if df[col].dtype == object and df[col].str.contains("µA").any():
            df[col] = df[col].str.replace(" µA", "").astype(float)
    
    train_dataset = SparkDataset(df, patch_size=30, window_size=5, stride=5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    norm_len = lit_model.hparams.normalize_length
    x_len_norm = int(norm_len * seq_len / (seq_len + pred_len))
    y_len_norm = norm_len - x_len_norm
    num_x_patch = int(x_len_norm / lit_model.hparams.patch_size)

    updates_per_epoch = min(int(len(train_dataset) / batch_size), len(train_loader))
    update_frequency = max(int(len(train_loader) / updates_per_epoch), 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.001, 1, updates_per_epoch),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(num_epoch * updates_per_epoch - updates_per_epoch), eta_min=1e-6)
        ],
        milestones=[updates_per_epoch]
    )

    print(f"Update every {update_frequency} step(s).")
    step_count = 0

    for epoch in range(num_epoch):
        epoch_loss = 0.
        for x, y in tqdm(train_loader, desc=f"[finetune]-{model_name}-{dataset_name}-{seq_len}-{pred_len} Epoch {epoch+1}"):
            step_count += 1

            x = x.squeeze(1) if x.dim() == 3 else x
            y = y.squeeze(1) if y.dim() == 3 else y
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

            norm_len = lit_model.hparams.normalize_length
            x_interp = nn.functional.interpolate(x, size=norm_len, mode='linear')
            y_interp = nn.functional.interpolate(y, size=norm_len, mode='linear')

            x_flat = rearrange(x_interp, 'b c t -> (b c) t')
            y_flat = rearrange(y_interp, 'b c t -> (b c) t')

            # xy = torch.cat([x_flat, y_flat], dim=-1) 
            

            start_idx = 0
            while start_idx < x_flat.shape[0]:
                end_idx = min(start_idx + max_uts_per_batch, x_flat.shape[0])
                xy = x_flat[start_idx:end_idx] 

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # Note: device_type might need adjustment if strictly no CUDA
                    loss_dict = model(
                        xy.to(device), # model is on lit_model.device which is 'cpu' now
                        mode='forecast',
                        num_input_patch=num_x_patch,
                        finetune=True
                    )

                x_loss = loss_dict['ts_loss']
                z_loss = loss_dict['vq_loss']
                s_loss = loss_dict['shape_loss']
                dist_loss = loss_dict['dist_loss']

                loss = (
                    1.5 * x_loss.mean()
                    + lit_model.hparams.lambda_z * z_loss.mean()
                    + lit_model.hparams.lambda_s * s_loss.mean()
                    + lit_model.hparams.lambda_dist * dist_loss.mean()
                )
                loss /= update_frequency
                loss.backward()
                epoch_loss += loss.item()


                start_idx += max_uts_per_batch

            if step_count % update_frequency == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.3f}. LR: {scheduler.get_last_lr()[-1]}")

    return lit_model


def run_test(lit_model, model_name, dataset_name, seq_len=96, pred_len=96, label_len=0):
    torch.set_float32_matmul_precision('medium')
    model = lit_model.model
    device = lit_model.device # This will be 'cpu'
    max_uts_per_batch = 1024 if model_name.startswith('small') else 512

    df = pd.read_csv("GEM1h.csv")
    df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

    test_dataset = SparkDataset(df, patch_size=30, window_size=5, stride=5)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    norm_len = lit_model.hparams.normalize_length
    x_len_norm = int(norm_len * seq_len / (seq_len + pred_len))
    y_len_norm = norm_len - x_len_norm
    num_x_patch = int(x_len_norm / lit_model.hparams.patch_size)

    mae_sum = 0
    mse_sum = 0
    n_sample = 0

    for x, y in tqdm(test_loader, desc=f"[eval]-{model_name}-{dataset_name}-{seq_len}-{pred_len}"):
        x = x.squeeze(1) if x.dim() == 3 else x
        y = y.squeeze(1) if y.dim() == 3 else y
        x = x.unsqueeze(1) # [B, 1, T]
        y = y.unsqueeze(1)

        x_interp = nn.functional.interpolate(x, size=norm_len, mode='linear')
        y_interp = nn.functional.interpolate(y, size=norm_len, mode='linear')

        x_flat = rearrange(x_interp, 'b c t -> (b c) t')
        y_flat = rearrange(y_interp, 'b c t -> (b c) t')

        xy = x_flat # 这里我们默认是用 X 来预测 Y，和训练逻辑对齐

        with torch.no_grad():
           
            autocast_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16 if autocast_device_type == 'cuda' else torch.float32, enabled=autocast_device_type=='cuda'): # Enable autocast only for CUDA for now
                _, result_dict = model(xy.float().to(device), mode='forecast', num_input_patch=num_x_patch)

        y_pred = result_dict["recon"].cpu()[:, -y_len_norm:]
        y_true = y_flat[:, -y_len_norm:]


        mse = nn.functional.mse_loss(y_pred, y_true, reduction='mean')
        mae = nn.functional.l1_loss(y_pred, y_true, reduction='mean')

        mae_sum += mae.item() * x.shape[0]
        mse_sum += mse.item() * x.shape[0]
        n_sample += x.shape[0]

    mae = mae_sum / n_sample
    mse = mse_sum / n_sample
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return mae, mse, seq_len, pred_len, label_len



def tokenization(lit_model, model_name, dataset_name, seq_len=96, pred_len=96, label_len=0, batch_size=-1):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--label_len", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--finetune_epoch", type=int, default=5)
    args = parser.parse_args()

    # assert args.dataset in data_dict.keys() # This might fail if dataset is not in data_dict, depends on usage

    from vqshape.pretrain import LitVQShape # Assuming vqshape.pretrain is in PYTHONPATH
    ckpt_path = "checkpoints/uea_dim256_codebook512/VQShape.ckpt"
    lit_model = LitVQShape.load_from_checkpoint(ckpt_path, map_location='cpu') # Changed to 'cpu'
    # lit_model = LitVQShape.load_from_checkpoint(model_dict[f"{args.model_name}-v{args.version}"], 'cuda') # This line remains commented
    print(f"Model [{args.model_name}-v{args.version}] loaded on device: {lit_model.device}")


    def finetune(lit_model, model_name, dataset_name, seq_len=96, pred_len=96, label_len=0, batch_size=64, num_epoch=10):
        torch.set_float32_matmul_precision('medium')
        model = lit_model.model
        model.codebook.entropy_loss = 0
        device = lit_model.device # will be 'cpu'
        max_uts_per_batch = 512 if model_name.startswith('small') else 256
        
        df = pd.read_csv("ae_labels.csv")
        df = df.select_dtypes(include=[np.number])

        if "Time" in df.columns:
            df = df.drop(columns=["Time"])
        for col in df.columns:
            if df[col].dtype == object and df[col].str.contains("µA").any(): # type: ignore
                df[col] = df[col].str.replace(" µA", "").astype(float)
        
        train_dataset = SparkDataset(df, patch_size=30, window_size=5, stride=5)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        norm_len = lit_model.hparams.normalize_length
        x_len_norm = int(norm_len * seq_len / (seq_len + pred_len))
        # y_len_norm = norm_len - x_len_norm # y_len_norm is not used in finetune for loss calculation logic shown
        num_x_patch = int(x_len_norm / lit_model.hparams.patch_size)

        updates_per_epoch = min(int(len(train_dataset) / batch_size), len(train_loader))
        update_frequency = max(int(len(train_loader) / updates_per_epoch), 1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.001, 1, updates_per_epoch),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=int(num_epoch * updates_per_epoch - updates_per_epoch), eta_min=1e-6)
            ],
            milestones=[updates_per_epoch]
        )

        print(f"Update every {update_frequency} step(s).")
        step_count = 0
        
        autocast_enabled = device.type == 'cuda' # Enable autocast only for CUDA
        # For CPU, autocast can use bfloat16 if supported, or can be disabled.
        # Keeping it simple: enable for CUDA, disable for CPU in finetuning for broader compatibility.
        # If bfloat16 is desired for CPU, additional checks for torch.cpu.is_bf16_supported() would be needed.
        
        for epoch in range(num_epoch):
            epoch_loss = 0.
            for x, y in tqdm(train_loader, desc=f"[finetune]-{model_name}-{dataset_name}-{seq_len}-{pred_len} Epoch {epoch+1}"):
                step_count += 1

                x = x.squeeze(1) if x.dim() == 3 else x
                y = y.squeeze(1) if y.dim() == 3 else y
                x = x.unsqueeze(1)
                # y = y.unsqueeze(1) # y is not directly used in the model call in finetune, only x_flat (xy)

                norm_len_val = lit_model.hparams.normalize_length # ensure it's 'norm_len_val' to avoid conflict if 'norm_len' is a function
                x_interp = nn.functional.interpolate(x, size=norm_len_val, mode='linear')
                # y_interp = nn.functional.interpolate(y, size=norm_len_val, mode='linear') # Not used

                x_flat = rearrange(x_interp, 'b c t -> (b c) t')
                # y_flat = rearrange(y_interp, 'b c t -> (b c) t') # Not used
                
                start_idx = 0
                while start_idx < x_flat.shape[0]:
                    end_idx = min(start_idx + max_uts_per_batch, x_flat.shape[0])
                    xy_batch = x_flat[start_idx:end_idx] 

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=autocast_enabled):
                        loss_dict = model(
                            xy_batch.to(device),
                            mode='forecast',
                            num_input_patch=num_x_patch,
                            finetune=True
                        )

                    x_loss = loss_dict['ts_loss']
                    z_loss = loss_dict['vq_loss']
                    s_loss = loss_dict['shape_loss']
                    dist_loss = loss_dict['dist_loss']

                    loss = (
                        1.5 * x_loss.mean()
                        + lit_model.hparams.lambda_z * z_loss.mean()
                        + lit_model.hparams.lambda_s * s_loss.mean()
                        + lit_model.hparams.lambda_dist * dist_loss.mean()
                    )
                    loss /= update_frequency
                    loss.backward()
                    epoch_loss += loss.item()
                    start_idx += max_uts_per_batch

                if step_count % update_frequency == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.3f}. LR: {scheduler.get_last_lr()[-1]}")
        return lit_model

    def run_test(lit_model, model_name, dataset_name, seq_len=96, pred_len=96, label_len=0):
        torch.set_float32_matmul_precision('medium')
        model = lit_model.model
        device = lit_model.device # This will be 'cpu'
        max_uts_per_batch = 1024 if model_name.startswith('small') else 512

        df = pd.read_csv("GEM1h.csv")
        df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float) if isinstance(col.iloc[0], str) else col)


        test_dataset = SparkDataset(df, patch_size=30, window_size=5, stride=5)
        # Ensure batch_size for test_loader is at least 1, even if max_uts_per_batch is small
        test_batch_size = max(1, min(64, int(max_uts_per_batch / (df.shape[1] if df.shape[1] > 0 else 1)))) # df.shape[1] is num features if relevant
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


        norm_len_val = lit_model.hparams.normalize_length
        x_len_norm = int(norm_len_val * seq_len / (seq_len + pred_len))
        y_len_norm = norm_len_val - x_len_norm
        num_x_patch = int(x_len_norm / lit_model.hparams.patch_size)

        mae_sum = 0
        mse_sum = 0
        n_sample = 0
        
        autocast_enabled = device.type == 'cuda' # Enable autocast only for CUDA during inference too for consistency
        
        for x, y in tqdm(test_loader, desc=f"[eval]-{model_name}-{dataset_name}-{seq_len}-{pred_len}"):
            x = x.squeeze(1) if x.dim() == 3 else x
            y = y.squeeze(1) if y.dim() == 3 else y
            x = x.unsqueeze(1) 
            y = y.unsqueeze(1)

            x_interp = nn.functional.interpolate(x, size=norm_len_val, mode='linear')
            y_interp = nn.functional.interpolate(y, size=norm_len_val, mode='linear')

            x_flat = rearrange(x_interp, 'b c t -> (b c) t')
            y_flat = rearrange(y_interp, 'b c t -> (b c) t')

            xy_batch = x_flat 

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=autocast_enabled):
                    _, result_dict = model(xy_batch.float().to(device), mode='forecast', num_input_patch=num_x_patch)

            y_pred = result_dict["recon"].cpu()[:, -y_len_norm:]
            y_true = y_flat[:, -y_len_norm:]

            mse = nn.functional.mse_loss(y_pred, y_true, reduction='sum') # sum for accumulation
            mae = nn.functional.l1_loss(y_pred, y_true, reduction='sum')  # sum for accumulation

            mae_sum += mae.item() 
            mse_sum += mse.item()
            n_sample += x.shape[0] * y_true.shape[-1] # total number of points

        mae = mae_sum / n_sample 
        mse = mse_sum / n_sample
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return mae, mse, seq_len, pred_len, label_len


    if args.method == 'zeroshot':
        mae, mse, slen, plen, llen = run_test(
            lit_model, 
            args.model_name, 
            args.dataset, 
            args.seq_len, 
            args.pred_len, 
            args.label_len
        )
    elif args.method == 'finetune':
        lit_model_finetuned = finetune( # Ensure using the returned model
            lit_model, 
            args.model_name, 
            args.dataset, 
            args.seq_len, 
            args.pred_len, 
            args.label_len,
            batch_size=args.batch_size,
            num_epoch=args.finetune_epoch
        )
        mae, mse, slen, plen, llen = run_test(
            lit_model_finetuned, # Use the fine-tuned model
            args.model_name, 
            args.dataset, 
            args.seq_len, 
            args.pred_len, 
            args.label_len
        )
    else:
        raise NotImplementedError(f"Invalid method [{args.method}]")

    result_dict = {
        'model': [args.model_name],
        'version': [args.version],
        'method': [args.method],
        'seq_len': [args.seq_len],
        'label_len': [args.label_len],
        'pred_len': [args.pred_len],
        'mae': [mae],
        'mse': [mse]
    }

    save_dir = f"./benchmark/result/forecast/{args.model_name}-v{args.version}/{args.method}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_path = f"{save_dir}/{args.dataset}.csv"
    df_results = pd.DataFrame(result_dict) # Renamed to avoid conflict with df from csv read
    try:
        exist_results = pd.read_csv(file_path)
        df_results = pd.concat([exist_results, df_results], axis=0, ignore_index=True)
        print(f"Add results to {file_path}.")
    except FileNotFoundError:
        print(f"File {file_path} not found. Saving to new file.")
    df_results.to_csv(file_path, index=False)