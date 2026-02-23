import torch
import torch.nn as nn
from torch import Tensor


class VAEEncoderEmbedding(nn.Module):
    """
    VAE Encoder with regression head.
    Structure: encoder + VAE layers (mu, log_var) + head
    This class adds VAE layers (mu, log_var) on top of the encoder, then adds a regression head.
    """
    def __init__(self, encoder, latent_dim, task: str = 'regression', head_dropout: float = 0.1):

        super().__init__()
        self.encoder = encoder
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
    
    def forward(self, x):
        """
        Forward pass:
        1. Encode: x -> encoder -> z_features
        2. VAE: z_features -> (mu, log_var) -> z (reparameterized)
        3. Regression: z -> head -> y
        Returns: y (prediction)
        """
        # 1. Get features from encoder
        encoder_output = self.encoder(x)  # Get Vector (latent_dim) or (z_features, entropy) tuple
        # Handle case where encoder returns (z_features, entropy) tuple
        if isinstance(encoder_output, tuple):
            z_features = encoder_output[0]  # Extract z_features, ignore entropy
        else:
            z_features = encoder_output
        
        # 2. Convert to mu and log_var, then reparameterize to z
        mu = self.mu(z_features)
        log_var = self.log_var(z_features)
        
        # 3. Reparameterization trick: z = mu + std * eps
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # Use mu directly in eval mode
        
        # 4. Regression: z -> head -> y
        y = self.head(z)  # Get Prediction (1)
        return y
