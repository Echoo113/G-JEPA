import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """
    encode the context (history) patches to latent representation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [batch_size, patch_size]
        # output shape: [batch_size, latent_dim]
        return self.encoder(x)


class TargetEncoder(nn.Module):
    """
    encode the future patches to latent representation.
    can share the structure with ContextEncoder, or be independent.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
