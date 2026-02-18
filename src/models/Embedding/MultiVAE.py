import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class MultitaskVAEEncoderEmbedding(nn.Module):
    """
    Multi-task VAE encoder embedding that trains encoder, VAE layers, decoder, and head simultaneously.
    
    Structure:
    - encoder: MLPEncoder or TabMEncoder (returns z_features)
    - VAE layers: mu and log_var layers (convert z_features to mu, log_var)
    - decoder: TabularDecoder (reconstructs x from z)
    - head: Simple Linear layer (regression head)
    
    Training:
    - Encoder, VAE layers (mu, log_var), decoder, and head are trained simultaneously using:
      - Regression loss: MSE(y_hat, y)
      - KL divergence loss: KL(q(z|x) || p(z))
      - Reconstruction loss: TabularLoss(x_recon, x)
    - Gradient flow: loss → (y_hat, x_recon) → (head, decoder) → z → (mu, log_var) → z_features → encoder
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
        self.latent_dim = latent_dim
        self.task = task
        self.outputs_probabilities = task == 'classification'
        
        # VAE layers: convert encoder output to mu and log_var
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)
        
        # Head: regression (linear) or classification (linear + sigmoid)
        self.head = (
            nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1), nn.Sigmoid())
            if self.outputs_probabilities
            else nn.Sequential(nn.BatchNorm1d(latent_dim), nn.Dropout(head_dropout), nn.Linear(latent_dim, 1))
        )
        
        # Initialize _last_entropy to None (will be set by forward() if encoder returns entropy)
        self._last_entropy = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
 
        # 1. ENCODER STEP: x -> encoder -> z_features
        encoder_output = self.encoder(x)  # shape: (batch_size, latent_dim) or (z_features, entropy)
        
        # Handle encoder that may return (z_features, entropy) tuple (e.g., TabNetEncoder with return_entropy=True)
        if isinstance(encoder_output, tuple):
            z_features, entropy = encoder_output
            # Store entropy for loss calculation (if needed)
            self._last_entropy = entropy
        else:
            z_features = encoder_output
            self._last_entropy = None
        
        # 2. VAE STEP: z_features -> (mu, log_var) -> z (reparameterized)
        mu = self.mu(z_features)
        log_var = self.log_var(z_features)
        
        # Reparameterization trick: z = mu + std * eps
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # Use mu directly in eval mode
        
        # 3. REGRESSION: z -> head -> y_hat
        y_hat = self.head(z)  # shape: (batch_size, 1)
        
        # 4. DECODE: z -> decoder -> x_recon
        recon = self.decoder(z)  # Returns (recon_cont, recon_bin, recon_cat)
        recon_cont, recon_bin, recon_cat = recon
        
        # Return outputs for multi-task training:
        # 1. y_hat (prediction from head, for regression loss)
        # 2. z (Latent code, useful for debugging/visualization)
        # 3. mu (for KL loss calculation)
        # 4. log_var (for KL loss calculation)
        # 5. recon_cont, recon_bin, recon_cat (reconstruction outputs, for reconstruction loss)
        return y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat
