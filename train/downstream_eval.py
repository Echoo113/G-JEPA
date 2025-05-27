import os
import pickle
import torch
from jepa.encoder import ContextEncoder
from jepa.predictor import Predictor
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG ====
context_dir = "data/downstream_patches"
model_dir = "model"
output_dir = "results/downstream_predictions"
os.makedirs(output_dir, exist_ok=True)

groups = ["u1", "u2", "u3", "u5", "u6", "u7"]
input_dim = 30
latent_dim = 128
num_future = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD SHARED CONTEXT ENCODER ====
context_encoder = ContextEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
context_encoder.eval()  # Freeze for inference

# ==== RUN PER GROUP ====
all_predictions = {g: {} for g in groups}
term_predictions = {"short": [], "middle": [], "long": []}  # 存储每个term的所有组预测

for group in groups:
    print(f"\n>>> Processing group {group}")

    # Load downstream patch
    patch_path = os.path.join(context_dir, f"{group}_downstream_patches.pkl")
    if not os.path.exists(patch_path):
        print(f"[!] Downstream patch for {group} not found. Skipping.")
        continue

    with open(patch_path, "rb") as f:
        patch_data = pickle.load(f)
    print("patch_data keys:", patch_data.keys())

    # Load trained predictor
    predictor = Predictor(latent_dim=latent_dim, num_future=num_future).to(device)
    predictor_path = os.path.join(model_dir, f"predictor_{group}.pth")
    if not os.path.exists(predictor_path):
        print(f"[!] Predictor for {group} not found. Skipping.")
        continue

    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor.eval()

    # For each type: short, middle, long
    for term in ["short", "middle", "long"]:
        if term not in patch_data or "patches" not in patch_data[term]:
            print(f"[!] No {term} patches for {group}.")
            continue
        patches = patch_data[term]["patches"].to(device)
        with torch.no_grad():
            s_ctx = context_encoder(patches).unsqueeze(1)   # [B, 1, D]
            predictions = predictor(s_ctx)                  # [B, num_future, D]
        print(f"  {term.capitalize()} predictions shape: {predictions.shape}")

        # Store predictions for this term
        term_predictions[term].append(predictions.cpu())
        all_predictions[group][term] = predictions.cpu()

# Save all predictions for each term as a dict
for term in ["short", "middle", "long"]:
    term_dict = {}
    for i, group in enumerate(groups):
        if i < len(term_predictions[term]):
            term_dict[group] = term_predictions[term][i]  # [B, num_future, D]
    
    save_path = os.path.join(output_dir, f"{term}_predictions_dict.pt")
    torch.save(term_dict, save_path)

    print(f"    Keys: {list(term_dict.keys())}")
    print(f"    Shapes: {[term_dict[g].shape for g in term_dict.keys()]}")

print("\nAll downstream evaluations finished.")

def visualize_downstream_predictions(all_predictions, groups, num_future):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    terms = ["short", "middle", "long"]
    colors = {"short": "r", "middle": "b", "long": "g"}
    x = np.arange(num_future)


    for i, group in enumerate(groups):
        ax = axes[i]
        group_values = []
        for term in terms:
            preds = all_predictions[group].get(term)
            if preds is not None and preds.shape[0] > 0:

                mean_curve = preds.mean(dim=(0,2)).numpy()  # [num_future]
                ax.plot(x, mean_curve, color=colors[term], label=term)
                group_values.append(mean_curve)
        
        if group_values:
            group_values = np.concatenate(group_values)
            y_min, y_max = group_values.min(), group_values.max()
            y_margin = (y_max - y_min) * 0.1  # add 10% margin
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
        ax.set_title(f"Group {group}")
        ax.set_xlabel("Future Step")
        ax.set_ylabel("Mean Predicted Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # last plot: all groups' mean
    ax = axes[-1]
    all_values = []
    for term in terms:
        all_term_preds = []
        for group in groups:
            preds = all_predictions[group].get(term)
            if preds is not None and preds.shape[0] > 0:
                all_term_preds.append(preds)
        if all_term_preds:
            all_term_preds = torch.cat(all_term_preds, dim=0)
            mean_curve = all_term_preds.mean(dim=(0,2)).numpy()
            ax.plot(x, mean_curve, color=colors[term], label=term)
            all_values.append(mean_curve)
    
    if all_values:
        all_values = np.concatenate(all_values)
        y_min, y_max = all_values.min(), all_values.max()
        y_margin = (y_max - y_min) * 0.1  # add 10% margin
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
    ax.set_title("All Groups Mean")
    ax.set_xlabel("Future Step")
    ax.set_ylabel("Mean Predicted Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("downstream_predictions_overview.png", dpi=300, bbox_inches='tight')
    plt.show()

visualize_downstream_predictions(all_predictions, groups, num_future)
