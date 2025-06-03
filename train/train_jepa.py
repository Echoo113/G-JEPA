import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torch.nn as nn
import numpy as np
from jepa.encoder import MyTimeSeriesEncoder, prepare_batch_from_np
from jepa.predictor import JEPPredictor
import matplotlib.pyplot as plt


# ========= settings =========
BATCH_SIZE = 2
LATENT_DIM = 64
EPOCHS     = 200
PATCH_FILE = "data/SOLAR/patches/solar_train.npz"
VAL_FILE   = "data/SOLAR/patches/solar_val.npz"

# ========= tools =========
def prepare_batched_tensor(np_array: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    Convert numpy patch data into (B, N, T, F) Tensor for Encoder input
    - For example, input shape = (172, 30, 137), batch_size = 4
    - Output shape = (4, 43, 30, 137)
    """
    total, T, F = np_array.shape
    N = total // batch_size
    usable = batch_size * N
    return torch.tensor(np_array[:usable], dtype=torch.float).view(batch_size, N, T, F)

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Calculate MSE and MAE"""
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return {'mse': mse.item(), 'mae': mae.item()}

# ========= Step 1: load patch data =========
print("\n[Step 1] Loading patch data...")
npz = np.load(PATCH_FILE)
ctx_long_np = npz['long_term_context']
fut_long_np = npz['long_term_future']
ctx_short_np = npz['short_term_context']
fut_short_np = npz['short_term_future']

# Load validation data
val_npz = np.load(VAL_FILE)
val_ctx = prepare_batched_tensor(val_npz['long_term_context'], BATCH_SIZE)
val_fut = prepare_batched_tensor(val_npz['long_term_future'], BATCH_SIZE)

# Prepare batched tensors
ctx_long_batch = prepare_batched_tensor(ctx_long_np, BATCH_SIZE)
fut_long_batch = prepare_batched_tensor(fut_long_np, BATCH_SIZE)
ctx_short_batch = prepare_batched_tensor(ctx_short_np, BATCH_SIZE)
fut_short_batch = prepare_batched_tensor(fut_short_np, BATCH_SIZE)

print(f"\n[Debug] Long-term shapes:")
print(f"- Context: {ctx_long_batch.shape}")
print(f"- Target: {fut_long_batch.shape}")
print(f"- Short-term shapes:")
print(f"- Context: {ctx_short_batch.shape}")
print(f"- Target: {fut_short_batch.shape}")

# ========= Step 2: initialize model =========
encoder = MyTimeSeriesEncoder(
    patch_length=30,
    num_vars=137,
    latent_dim=LATENT_DIM,
    num_layers=2,
    num_attention_heads=2,
)

# Initialize two predictors
predictor_long = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=ctx_long_batch.shape[1],
    prediction_length=fut_long_batch.shape[1],
    num_layers=4,
    num_heads=4
)

predictor_short = JEPPredictor(
    latent_dim=LATENT_DIM,
    context_length=ctx_short_batch.shape[1],
    prediction_length=fut_short_batch.shape[1],
    num_layers=4,
    num_heads=4
)

print(f"\n[Debug] Model configurations:")
print(f"- Encoder latent dim: {LATENT_DIM}")
print(f"- Long-term context/prediction: {ctx_long_batch.shape[1]}/{fut_long_batch.shape[1]}")
print(f"- Short-term context/prediction: {ctx_short_batch.shape[1]}/{fut_short_batch.shape[1]}")

# Initialize two optimizers
optimizer_long = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor_long.parameters()),
    lr=1e-3
)

optimizer_short = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor_short.parameters()),
    lr=1e-3
)

# ========= Step 3: train long-term prediction =========
print("\n[Step 3] Training long-term prediction...")

# Initialize metrics history
long_term_history = {
    'train_mse': [], 'train_mae': [],
    'val_mse': [], 'val_mae': []
}

best_val_mse = float('inf')
for epoch in range(1, EPOCHS + 1):
    # Training
    encoder.train()
    predictor_long.train()
    optimizer_long.zero_grad()

    

    ctx_L = encoder(ctx_long_batch)
    tgt_L = encoder(fut_long_batch)
    
   
    
    pred_L, loss_L = predictor_long(ctx_L, tgt_L)
    
    
    
    loss_L.backward()
    optimizer_long.step()
    
    # Calculate training metrics
    metrics_L = compute_metrics(pred_L, tgt_L)
    
    # Record metrics
    long_term_history['train_mse'].append(metrics_L['mse'])
    long_term_history['train_mae'].append(metrics_L['mae'])
    
    # Validation
    encoder.eval()
    predictor_long.eval()
    with torch.no_grad():
        
        
        val_ctx_L = encoder(val_ctx)
        val_tgt_L = encoder(val_fut)
       
        val_pred_L, _ = predictor_long(val_ctx_L, val_tgt_L)
        
       
        
        val_metrics_L = compute_metrics(val_pred_L, val_tgt_L)
        
        # Record validation metrics
        long_term_history['val_mse'].append(val_metrics_L['mse'])
        long_term_history['val_mae'].append(val_metrics_L['mae'])

    # Print training progress
    print(f"[Long-term Epoch {epoch:02d}] Loss: {loss_L.item():.4f}, MSE: {metrics_L['mse']:.4f}, MAE: {metrics_L['mae']:.4f}")
    print(f"                     Val   - MSE: {val_metrics_L['mse']:.4f}, MAE: {val_metrics_L['mae']:.4f}")

    # Print shape information in the first epoch
    if epoch == 1:
        print(f"\n[Debug] Long-term shapes:")
        print(f"- Context: {ctx_L.shape}")
        print(f"- Target: {tgt_L.shape}")
        print(f"- Prediction: {pred_L.shape}")

print("\n[Long-term training completed]")

# ========= Step 4: train short-term prediction =========
print("\n[Step 4] Training short-term prediction...")

# Initialize metrics history
short_term_history = {
    'train_mse': [], 'train_mae': [],
    'val_mse': [], 'val_mae': []
}

best_val_mse = float('inf')
for epoch in range(1, EPOCHS + 1):
    # Training
    encoder.train()
    predictor_short.train()
    optimizer_short.zero_grad()

    ctx_S = encoder(ctx_short_batch)
    tgt_S = encoder(fut_short_batch)
    pred_S, loss_S = predictor_short(ctx_S, tgt_S)
    
    loss_S.backward()
    optimizer_short.step()
    
    # Calculate training metrics
    metrics_S = compute_metrics(pred_S, tgt_S)
    
    # Record metrics
    short_term_history['train_mse'].append(metrics_S['mse'])
    short_term_history['train_mae'].append(metrics_S['mae'])

    # Validation
    encoder.eval()
    predictor_short.eval()
    with torch.no_grad():
        
        val_ctx_S = encoder(val_ctx)
        val_tgt_S = encoder(val_fut)
        
       
        
        val_pred_S, _ = predictor_short(val_ctx_S, val_tgt_S)
        
        
        val_metrics_S = compute_metrics(val_pred_S, val_tgt_S)
        
        # Record validation metrics
        short_term_history['val_mse'].append(val_metrics_S['mse'])
        short_term_history['val_mae'].append(val_metrics_S['mae'])

    # Print training progress
    print(f"[Short-term Epoch {epoch:02d}] Loss: {loss_S.item():.4f}, MSE: {metrics_S['mse']:.4f}, MAE: {metrics_S['mae']:.4f}")
    print(f"                     Val   - MSE: {val_metrics_S['mse']:.4f}, MAE: {val_metrics_S['mae']:.4f}")

    # Print shape information in the first epoch
    if epoch == 1:
        print(f"\n[Debug] Short-term shapes:")
        print(f"- Context: {ctx_S.shape}")
        print(f"- Target: {tgt_S.shape}")
        print(f"- Prediction: {pred_S.shape}")

print("\n[Short-term training completed]")
print("\n[All training completed]")

# ========= Step 5: Plot metrics: two subplots in one figure =========
print("\n[Step 5] Plotting metrics with two subplots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
epochs = range(1, EPOCHS + 1)

# Short-term subplot (left)
axes[0].plot(epochs, short_term_history['train_mse'], label='Train MSE', color='orange', linestyle='-')
axes[0].plot(epochs, short_term_history['val_mse'], label='Val MSE', color='orange', linestyle='--')
axes[0].plot(epochs, short_term_history['train_mae'], label='Train MAE', color='red', linestyle='-')
axes[0].plot(epochs, short_term_history['val_mae'], label='Val MAE', color='red', linestyle='--')
axes[0].set_title('Short-term Metrics')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True)

# Long-term subplot (right)
axes[1].plot(epochs, long_term_history['train_mse'], label='Train MSE', color='b', linestyle='-')
axes[1].plot(epochs, long_term_history['val_mse'], label='Val MSE', color='b', linestyle='--')
axes[1].plot(epochs, long_term_history['train_mae'], label='Train MAE', color='c', linestyle='-')
axes[1].plot(epochs, long_term_history['val_mae'], label='Val MAE', color='c', linestyle='--')
axes[1].set_title('Long-term Metrics')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True)

fig.suptitle('Short-term and Long-term Metrics Overview', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_overview.png', dpi=300, bbox_inches='tight')
plt.show()