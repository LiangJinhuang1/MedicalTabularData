import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union

class MultiTaskEncoderEmbedding(nn.Module):
    """
    Multi-task encoder embedding that trains encoder, decoder, and head simultaneously.
    Same structure as EncoderEmbedding (encoder + head), but adds decoder for reconstruction loss.
    
    Structure:
    - encoder: MLPEncoder or TabMEncoder (returns z)
    - decoder: TabularDecoder (reconstructs x from z)
    - head: Simple Linear layer (regression head)
    
    Training:
    - Encoder, decoder, and head are trained simultaneously using:
    - Regression loss: MSE(y_hat, y)
    - Reconstruction loss: TabularLoss(x_recon, x)
    - Gradient flow: loss → (y_hat, x_recon) → (head, decoder) → z → encoder
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        task: str = 'regression',
        head_dropout: float = 0.1,
    ) -> None:
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.task = task
        self.outputs_probabilities = task == 'classification'
        self.head = (
            nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1), nn.Sigmoid())
            if self.outputs_probabilities
            else nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1))
        )

    def forward(self, x: Tensor):
        # 1. ENCODER STEP: x -> encoder -> z
        encoder_output = self.encoder(x)  # shape: (batch_size, latent_dim) or (z, entropy)
        
        # Handle encoder that may return (z, entropy) tuple
        if isinstance(encoder_output, tuple):
            z, entropy = encoder_output
            self._last_entropy = entropy
        else:
            z = encoder_output
            self._last_entropy = None
        
        # 2. REGRESSION: z -> head -> y_hat
        y_hat = self.head(z)  # shape: (batch_size, 1)
        
        # 3. DECODE: z -> decoder -> x_recon
        recon = self.decoder(z)  # Returns (recon_cont, recon_bin, recon_cat)
        recon_cont, recon_bin, recon_cat = recon
            
        # Return outputs for multi-task training:
        # 1. y_hat (prediction from head, for regression loss)
        # 2. z (Latent code, useful for debugging/visualization)
        # 3. recon_cont, recon_bin, recon_cat (reconstruction outputs, for reconstruction loss)
        return y_hat, z, recon_cont, recon_bin, recon_cat

