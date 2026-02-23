import torch 
import torch.nn as nn
from torch import Tensor
from typing import Optional


class TabVAE(nn.Module):
    """
    Variational Autoencoder for tabular data.
    Used ONLY for pre-training weights (Unsupervised).
    Structure: x -> Encoder -> z_features -> (mu, log_var) -> z -> Decoder -> x_recon
    This class adds VAE layers (mu, log_var) on top of the encoder output.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
        # VAE layers: convert encoder output to mu and log_var
        # The encoder outputs features of size latent_dim, we add mu and log_var layers
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def encode(self, x: Tensor) -> tuple:
        """
        Encode input to mu and log_var.
        """
        # Get features from encoder
        encoder_output = self.encoder(x)  # shape: (batch_size, latent_dim) or (z, entropy)
        
        # Handle encoder that may return (z, entropy) tuple (e.g., TabNetEncoder with return_entropy=True)
        if isinstance(encoder_output, tuple):
            z_features, entropy = encoder_output
            # Store entropy for loss calculation (if needed)
            self._last_entropy = entropy
        else:
            z_features = encoder_output
            self._last_entropy = None
        
        # Convert to mu and log_var
        mu = self.mu(z_features)
        log_var = self.log_var(z_features)
        
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
        
    def forward(self, x: Tensor, training: bool = True):
        """
        Forward pass:
        1. Encode: x -> encoder -> z_features -> (mu, log_var) -> z (reparameterized)
        2. Decode: z -> decoder -> x_recon
        """
        # 1. Encode to mu and log_var
        mu, log_var = self.encode(x)
        
        # 2. Reparameterize to get z
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        
        # 3. Decode back to x
        # Returns (cont, bin, cat) tuple from decoder
        recon = self.decoder(z)
        
        # Return reconstruction and VAE components
        recon_cont, recon_bin, recon_cat = recon
        return recon_cont, recon_bin, recon_cat, mu, log_var, z



