import torch
import torch.nn as nn
from torch import Tensor

from src.models.TabNet.TabNet import TabNetEncoder as CoreTabNetEncoder


class TabNetEncoder(nn.Module):
    """
    Wrapper around the official TabNet encoder that returns a single latent
    vector (sum of decision step outputs) plus the sparsity loss.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        dropout: float = 0.1,
        return_entropy: bool = True,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        group_attention_matrix=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.return_entropy = return_entropy
        self.encoder = CoreTabNetEncoder(
            input_dim=input_dim,
            output_dim=input_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix,
        )
        self.projection = nn.Linear(n_d, latent_dim)

    def forward(self, x: Tensor):
        steps_output, m_loss = self.encoder(x)
        latent = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        latent = self.projection(latent)
        if self.return_entropy:
            return latent, m_loss
        return latent

    def forward_masks(self, x: Tensor):
        return self.encoder.forward_masks(x)
