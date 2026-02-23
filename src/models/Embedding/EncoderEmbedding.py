import torch
import torch.nn as nn
from torch import Tensor


class EncoderEmbedding(nn.Module):
    def __init__(self, encoder, latent_dim, task: str = 'regression', head_dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.task = task
        self.outputs_probabilities = task == 'classification'

        # Head: regression (linear) or classification (linear + sigmoid)
        self.head = (
            nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1), nn.Sigmoid())
            if self.outputs_probabilities
            else nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1))
        )

    def forward(self, x):
        encoder_output = self.encoder(x) # Get Vector (32) or (z, entropy) tuple
        # Handle case where encoder returns (z, entropy) tuple (e.g., TabNetEncoder with return_entropy=True)
        if isinstance(encoder_output, tuple):
            z = encoder_output[0]  # Extract z, ignore entropy
        else:
            z = encoder_output
        y = self.head(z)    # Get Prediction (1)
        return y
