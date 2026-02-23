import torch
import torch.nn as nn
from torch import Tensor

from src.models.TabM.tabM import EnsembleView, LinearEnsemble, MLPBackboneMiniEnsemble, ScaleEnsemble

class TabMEncoder(nn.Module):
    """
    TabM-mini encoder using the same backbone/adapter as TabM.
    The only difference is the output dimension (latent_dim) and head aggregation.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        k_heads: int = 32,
        dropout: float = 0.1,
        batchnorm: bool = False,
        activation: str = 'ReLU',
    ):
        super().__init__()
        self.k = k_heads
        self.ensemble_view = EnsembleView(k=self.k)
        self.backbone = MLPBackboneMiniEnsemble(
            d_in=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batchnorm=batchnorm,
            activation=activation,
            k=self.k,
        )
        self.projection = LinearEnsemble(self.backbone.out_dim, latent_dim, k=self.k)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, D)
        x = self.ensemble_view(x)          # (B, K, D)
        x = self.backbone(x)               # (B, K, H)
        x = self.projection(x)             # (B, K, latent)
        return x.mean(dim=1)               # (B, latent)
