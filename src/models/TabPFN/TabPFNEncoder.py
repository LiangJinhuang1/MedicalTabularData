import torch
import torch.nn as nn
import numpy as np
from typing import Optional

try:
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions.embedding.tabpfn_embedding import TabPFNEmbedding
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

class TabPFNEncoder(nn.Module):
    """
    Pure TabPFN Encoder.
    Outputs the raw Transformer embeddings (mean-pooled over ensembles).
    Designed to be used as a frozen backbone for a custom linear head.
    """
    
    def __init__(
        self,
        input_dim: int = None,  # Kept for interface compatibility (not used)
        latent_dim: int = None,  # Kept for interface compatibility (not used)
        device: str = 'cuda',
        base_path: Optional[str] = None
    ):
        super().__init__()
        if not DEPENDENCIES_INSTALLED:
            raise ImportError("Please install 'tabpfn' and 'tabpfn-extensions'.")

        self.device = torch.device(device)
        
        # Initialize TabPFN
        self.tabpfn = TabPFNRegressor(device=device, **({"base_path": base_path} if base_path else {}))
        self.embedding_extractor = TabPFNEmbedding(tabpfn_reg=self.tabpfn, n_fold=0)
        
        # Internal Context State (Crucial for TabPFN inference)
        self.X_context = None
        self.y_context = None
        self.is_fitted = False
        self._embedding_dim = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Sets the context data. TabPFN needs this to 'encode' new data points
        relative to the distribution of the training set.
        """
        if isinstance(X_train, torch.Tensor): X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor): y_train = y_train.detach().cpu().numpy()
        
        self.X_context = X_train.astype(np.float32)
        self.y_context = y_train.flatten().astype(np.float32)
        
        self.embedding_extractor.fit(self.X_context, self.y_context)
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw embeddings from the TabPFN Transformer.
        Output Shape: (batch_size, raw_embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted with context data via .fit() first.")

        # Convert input for TabPFN (Numpy)
        x_np = x.detach().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            # TabPFN generates embeddings by looking at the context (X_context) 
            # and the current query points (x_np)
            emb_np = self.embedding_extractor.get_embeddings(
                X_train=self.X_context,
                y_train=self.y_context,
                X=x_np,
                data_source="test"
            )
            
        embeddings = torch.from_numpy(emb_np).to(self.device).float()
        
        # Debug: Print actual embeddings shape from TabPFN
        print(f"[TabPFNEncoder] Input shape: {x.shape}, Raw embeddings shape: {embeddings.shape}, NumPy shape: {emb_np.shape}")
        
        # Handle different embedding shapes from TabPFN
        input_batch_size = x.shape[0]
        
        if embeddings.ndim == 3:
            # Shape: (n_ensemble, batch_size, embedding_dim)
            # Mean pool over ensemble dimension
            embeddings = embeddings.mean(dim=0)  # Result: (batch_size, embedding_dim)
        elif embeddings.ndim == 2:
            # Can be either (batch_size, embedding_dim) or (n_ensemble, embedding_dim)
            if embeddings.shape[0] == input_batch_size:
                # Shape is (batch_size, embedding_dim) - already correct
                pass
            else:
                # Shape is (n_ensemble, embedding_dim) for batch_size=1
                # Mean pool over ensemble dimension and ensure batch dimension
                embeddings = embeddings.mean(dim=0, keepdim=True)  # Result: (1, embedding_dim)
        elif embeddings.ndim == 1:
            # Single sample case -> (embedding_dim,)
            embeddings = embeddings.unsqueeze(0)  # Result: (1, embedding_dim)
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        # Safety check: ensure batch size matches
        if embeddings.shape[0] != input_batch_size:
            raise RuntimeError(
                f"Dimension mismatch! Input batch: {input_batch_size}, "
                f"Output batch: {embeddings.shape[0]}. "
                f"Original embeddings shape: {emb_np.shape}, "
                f"Final shape: {embeddings.shape}."
            )

        # Cache the dimension for the user to build their linear head
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[-1]

        return embeddings

    @property
    def embedding_dim(self):
        """Returns the size of the raw TabPFN embeddings (usually 192 or 512)."""
        if self._embedding_dim is None:
            print("Warning: embedding_dim is unknown until the first forward pass.")
        return self._embedding_dim

    def parameters(self, recurse: bool = True):
        # TabPFN is a frozen feature extractor in this context
        return iter([])

    def state_dict(self, *args, **kwargs):
        """Save context data so the encoder can be reloaded without re-fitting."""
        return {
            'X_context': self.X_context,
            'y_context': self.y_context,
            'is_fitted': self.is_fitted,
            'embedding_dim': self._embedding_dim
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        self.X_context = state_dict['X_context']
        self.y_context = state_dict['y_context']
        self.is_fitted = state_dict['is_fitted']
        self._embedding_dim = state_dict['embedding_dim']
        
        if self.is_fitted:
            self.embedding_extractor.fit(self.X_context, self.y_context)