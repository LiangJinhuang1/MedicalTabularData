import torch.nn as nn
from torch import Tensor


class TabAE(nn.Module):
    """
    Standard Autoencoder.
    Used ONLY for pre-training weights (Unsupervised).
    Structure: x -> Encoder -> z -> Decoder -> x_recon
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x: Tensor):
        # 1. Encode to z
        encoder_output = self.encoder(x)
        
        # Handle encoder that may return (z, entropy) tuple
        if isinstance(encoder_output, tuple):
            z, entropy = encoder_output
            self._last_entropy = entropy
        else:
            z = encoder_output
            self._last_entropy = None
        
        # 2. Decode back to x
        # Returns (cont, bin, cat) tuple from your decoder
        recon = self.decoder(z)
        
        return recon