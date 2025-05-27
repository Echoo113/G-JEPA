import torch
import torch.nn.functional as F

#1. mse_loss: compute the mean square error between the predicted and target embeddings
#2. cosine_loss: compute the cosine similarity between the predicted and target embeddings
#3. combined_loss: compute the combined loss using both mse_loss and cosine_loss

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return F.mse_loss(pred, target)

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()

def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                 mse_weight: float = 0.5, cosine_weight: float = 0.5) -> torch.Tensor:
   
    mse = mse_loss(pred, target)
    cosine = cosine_loss(pred, target)
    return mse_weight * mse + cosine_weight * cosine
