import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

class TabularDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        n_continuous: int = 0,
        n_binary: int = 0,
        cat_sizes: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cat_sizes = cat_sizes if cat_sizes is not None else []
        total_cat_dim = sum(self.cat_sizes)
        
        # 1. Shared Decoder Trunk
        # Reconstructs from latent back to the last hidden layer size
        dec_layers = []
        prev_dim = latent_dim
        
        for h in hidden_dims:
            dec_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
            
        self.trunk = nn.Sequential(*dec_layers)
        
        # 2. Specific Heads
        # A. Continuous
        self.out_cont = nn.Linear(prev_dim, n_continuous) if n_continuous > 0 else None
        
        # B. Binary (Logits)
        self.out_bin = nn.Linear(prev_dim, n_binary) if n_binary > 0 else None
        
        # C. Categorical (Logits)
        # We output one big vector here, the Loss function handles splitting it
        self.out_cat = nn.Linear(prev_dim, total_cat_dim) if total_cat_dim > 0 else None

    def forward(self, z: Tensor):
        # Expand z back to hidden features
        h = self.trunk(z)
        
        # Predict specific types
        # Check for None to handle cases where we might only have continuous data, etc.
        recon_cont = self.out_cont(h) if self.out_cont is not None else torch.empty(z.size(0), 0, device=z.device)
        recon_bin = self.out_bin(h) if self.out_bin is not None else torch.empty(z.size(0), 0, device=z.device)
        recon_cat = self.out_cat(h) if self.out_cat is not None else torch.empty(z.size(0), 0, device=z.device)
        
        return recon_cont, recon_bin, recon_cat