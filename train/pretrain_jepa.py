import os
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from jepa.encoder import ContextEncoder, TargetEncoder
from jepa.predictor import Predictor
from jepa.loss import combined_loss



class PatchDataset(Dataset):
    def __init__(self, context_dir, future_dir, group_prefix):
        self.context_dir = context_dir
        self.future_dir = future_dir
        self.group_prefix = group_prefix
        self.context_patches = self._load_patches(self.context_dir, f"{group_prefix}_fused_patches.pkl")
        self.future_patches = self._load_patches(self.future_dir, f"{group_prefix}_pseudo_future.pkl")
        
    def _load_patches(self, dir_path, filename):
        with open(os.path.join(dir_path, filename), 'rb') as f:
            return pickle.load(f)
    
    def __len__(self):
        return min(
            len(self.context_patches["short"]["patches"]),
            len(self.context_patches["middle"]["patches"]),
            len(self.context_patches["long"]["patches"]),
            len(self.future_patches["short"]["patches"]),
            len(self.future_patches["middle"]["patches"]),
            len(self.future_patches["long"]["patches"]),
        )
    
    def __getitem__(self, idx):
        # get patches for each group    
        context_short = self.context_patches["short"]["patches"][idx]
        context_mid = self.context_patches["middle"]["patches"][idx]
        context_long = self.context_patches["long"]["patches"][idx]
        
        future_short = self.future_patches["short"]["patches"][idx]
        future_mid = self.future_patches["middle"]["patches"][idx]
        future_long = self.future_patches["long"]["patches"][idx]
        
        return {
            "context": {
                "short": context_short,
                "middle": context_mid,
                "long": context_long
            },
            "future": {
                "short": future_short,
                "middle": future_mid,
                "long": future_long
            }
        }
    
#helper function to print loss for each single group like u1, u2, u3, u4, u5, u6
def print_loss(loss, group_prefix):
    print(f"Group {group_prefix} Loss: {loss:.4f}")

def train(context_encoder, target_encoder, predictor, train_loader, optimizer, device, epoch):
    context_encoder.train()
    target_encoder.train()
    predictor.train()
    
    total_loss = 0
    total_short_loss = 0
    total_mid_loss = 0
    total_long_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                # get patches for each group    
        context_short = batch["context"]["short"].to(device)
        context_mid = batch["context"]["middle"].to(device)
        context_long = batch["context"]["long"].to(device)
        
        future_short = batch["future"]["short"].to(device)
        future_mid = batch["future"]["middle"].to(device)
        future_long = batch["future"]["long"].to(device)
        
        # ----------------------------
        #need to change this part further, diff term predictors should be used
        #----------------------------
        # short term
        s_ctx_short = context_encoder(context_short).unsqueeze(1)  # [B, 1, D]
        pred_short = predictor(s_ctx_short)  # [B, num_future, D]
        with torch.no_grad():
            target_short = target_encoder(future_short).unsqueeze(1)  # [B, 1, D]
            target_short = target_short.repeat(1, pred_short.size(1), 1)  # [B, num_future, D]
        
        # mid term
        s_ctx_mid = context_encoder(context_mid).unsqueeze(1)  # [B, 1, D]
        pred_mid = predictor(s_ctx_mid)  # [B, num_future, D]
        with torch.no_grad():
            target_mid = target_encoder(future_mid).unsqueeze(1)  # [B, 1, D]
            target_mid = target_mid.repeat(1, pred_mid.size(1), 1)  # [B, num_future, D]
        
        # long term
        s_ctx_long = context_encoder(context_long).unsqueeze(1)  # [B, 1, D]
        pred_long = predictor(s_ctx_long)  # [B, num_future, D]
        with torch.no_grad():
            target_long = target_encoder(future_long).unsqueeze(1)  # [B, 1, D]
            target_long = target_long.repeat(1, pred_long.size(1), 1)  # [B, num_future, D]
        
        # calculate loss for each term
        loss_short = combined_loss(pred_short, target_short)
        loss_mid = combined_loss(pred_mid, target_mid)
        loss_long = combined_loss(pred_long, target_long)
        
        # total loss is the weighted sum of the three terms
        loss = loss_short + loss_mid + loss_long
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_short_loss += loss_short.item()
        total_mid_loss += loss_mid.item()
        total_long_loss += loss_long.item()
    
    # calculate average loss
    avg_loss = total_loss / len(train_loader)
    avg_short_loss = total_short_loss / len(train_loader)
    avg_mid_loss = total_mid_loss / len(train_loader)
    avg_long_loss = total_long_loss / len(train_loader)
    
    return avg_loss, avg_short_loss, avg_mid_loss, avg_long_loss

def visualize_loss(losses, short_losses, mid_losses, long_losses):
    # plot loss curves
    plt.figure(figsize=(12, 8))
    # use different colors and line styles
    plt.plot(losses, 'k-', linewidth=2, label='Total Loss')  # black solid line
    plt.plot(short_losses, 'r--', linewidth=1.5, label='Short-term Loss')  # red dashed line
    plt.plot(mid_losses, 'b:', linewidth=1.5, label='Mid-term Loss')  # blue dotted line
    plt.plot(long_losses, 'g-.', linewidth=1.5, label='Long-term Loss')  # green dash-dot line
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Losses by Temporal Scale', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    # set y-axis range, leave some margin
    y_min = min(min(losses), min(short_losses), min(mid_losses), min(long_losses))
    y_max = max(max(losses), max(short_losses), max(mid_losses), max(long_losses))
    plt.ylim(y_min * 0.9, y_max * 1.1)
    # set x-axis ticks
    plt.xticks(range(0, len(losses), 2))
    plt.tight_layout()
    plt.savefig('jepa_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_all_groups_loss(
    all_losses, all_short_losses, all_mid_losses, all_long_losses, groups
):
    num_groups = len(groups)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns, last plot is the average
    axes = axes.flatten()
    epochs = len(all_losses[0])
    x = range(1, epochs + 1)

    # plot each group
    for i, group in enumerate(groups):
        ax = axes[i]
        ax.plot(x, all_losses[i], 'k-', label='Total')
        ax.plot(x, all_short_losses[i], 'r--', label='Short')
        ax.plot(x, all_mid_losses[i], 'b:', label='Mid')
        ax.plot(x, all_long_losses[i], 'g-.', label='Long')
        ax.set_title(f'Group {group}')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    # plot the average loss
    mean_loss = np.mean(all_losses, axis=0)
    mean_short = np.mean(all_short_losses, axis=0)
    mean_mid = np.mean(all_mid_losses, axis=0)
    mean_long = np.mean(all_long_losses, axis=0)
    ax = axes[-1]
    ax.plot(x, mean_loss, 'k-', label='Total')
    ax.plot(x, mean_short, 'r--', label='Short')
    ax.plot(x, mean_mid, 'b:', label='Mid')
    ax.plot(x, mean_long, 'g-.', label='Long')
    ax.set_title('Mean of All Groups')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('all_groups_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # set parameters
    context_dir = "data/patches"
    future_dir = "data/pseudo_future_patches"
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = ["u1", "u2", "u3", "u5", "u6", "u7"]

    os.makedirs("model", exist_ok=True)

    all_losses = []
    all_short_losses = []
    all_mid_losses = []
    all_long_losses = []

    for group in groups:
        print(f"\n=== Training for group {group} ===")
        # create encoder and predictor
        context_encoder = ContextEncoder(input_dim=30, latent_dim=128).to(device)
        target_encoder = TargetEncoder(input_dim=30, latent_dim=128).to(device)
        predictor = Predictor(latent_dim=128, num_future=3).to(device)

        # create data loader
        train_dataset = PatchDataset(context_dir, future_dir, group)
        if len(train_dataset) == 0:
            print(f"  [Warning] No data for group {group}, skipping.")
            all_losses.append([np.nan]*num_epochs)
            all_short_losses.append([np.nan]*num_epochs)
            all_mid_losses.append([np.nan]*num_epochs)
            all_long_losses.append([np.nan]*num_epochs)
            continue
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # create optimizer
        optimizer = optim.Adam(
            list(context_encoder.parameters()) +
            list(target_encoder.parameters()) +
            list(predictor.parameters()),
            lr=learning_rate
        )

        # training loop
        losses = []
        short_losses = []
        mid_losses = []
        long_losses = []

        for epoch in range(1, num_epochs + 1):
            avg_loss, avg_short_loss, avg_mid_loss, avg_long_loss = train(
                context_encoder, target_encoder, predictor,
                train_loader, optimizer, device, epoch
            )
            losses.append(avg_loss)
            short_losses.append(avg_short_loss)
            mid_losses.append(avg_mid_loss)
            long_losses.append(avg_long_loss)
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Short-term Loss: {avg_short_loss:.4f}")
            print(f"Mid-term Loss: {avg_mid_loss:.4f}")
            print(f"Long-term Loss: {avg_long_loss:.4f}")
            print("-" * 50)

        # save the predictor
        predictor_save_path = f"model/predictor_{group}.pth"
        torch.save(predictor.state_dict(), predictor_save_path)
        print(f"[✔] Predictor saved for {group} at {predictor_save_path}")

        all_losses.append(losses)
        all_short_losses.append(short_losses)
        all_mid_losses.append(mid_losses)
        all_long_losses.append(long_losses)

    # visualize all groups and average loss
    visualize_all_groups_loss(
        all_losses, all_short_losses, all_mid_losses, all_long_losses, groups
    )

if __name__ == "__main__":
    main()
