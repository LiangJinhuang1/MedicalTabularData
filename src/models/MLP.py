import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class MLPEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int, 
        dropout: float = 0.1,
        batchnorm: bool = True,
        activation: str = "ReLU"
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            
            # Activation selection
            if activation == "GELU":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
            
        # Final projection to Latent Dimension (z)
        layers.append(nn.Linear(prev_dim, latent_dim)) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # Returns z [batch, latent_dim]