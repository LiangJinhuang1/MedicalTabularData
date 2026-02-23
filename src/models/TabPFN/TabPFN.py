"""
TabPFN Model Wrapper
TabPFN is a pre-trained foundation model for tabular data.
This wrapper makes it compatible with the existing training framework.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not installed. Install with: pip install tabpfn")


class TabPFN(nn.Module):
    """
    Wrapper for TabPFN regression model to make it compatible with PyTorch training framework.
    
    Note: TabPFN is a pre-trained model that uses its own training mechanism (fit/predict).
    This wrapper adapts it to work with the existing training loop.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        device: str = 'cuda',
        only_inference: bool = False,
        base_path: Optional[str] = None
    ):
        """
        Initialize TabPFN model wrapper.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions (should be 1 for regression)
            device: Device to use ('cuda' or 'cpu')
            only_inference: If True, skip training and only do inference
            base_path: Base path for model weights (None = use default cache)
        """
        super().__init__()
        
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is not installed. Please install it with: pip install tabpfn"
            )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device_str = device
        self.only_inference = only_inference
        
        # Initialize TabPFN regressor
        if base_path is not None:
            self.tabpfn_model = TabPFNRegressor(
                device=device,
                base_path=base_path
            )
        else:
            self.tabpfn_model = TabPFNRegressor(device=device)
        
        # Store training data for fit() method
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
        # Move to device
        self.device = torch.device(device)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the TabPFN model on training data.
        This is called separately from the PyTorch training loop.
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training labels (numpy array)
        """
        # Ensure y_train is 1D for regression
        if y_train.ndim > 1:
            y_train = y_train.squeeze()
        
        self.tabpfn_model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.is_fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch compatibility.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TabPFN model must be fitted before forward pass. "
                "Call fit() method first or set only_inference=False during training."
            )
        
        # Convert to numpy for TabPFN
        x_np = x.detach().cpu().numpy()
        
        # Get predictions
        predictions = self.tabpfn_model.predict(x_np)
        
        # Ensure predictions are 2D (batch_size, output_dim)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Convert back to torch tensor
        predictions_tensor = torch.from_numpy(predictions).float().to(self.device)
        
        return predictions_tensor
    
    def train(self, mode: bool = True):
        """
        Override train() to handle TabPFN's special training mechanism.
        TabPFN doesn't use gradient-based training, so we just set the mode.
        """
        super().train(mode)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self
    
    def parameters(self):
        """
        Return empty iterator since TabPFN doesn't have trainable parameters
        in the traditional PyTorch sense.
        """
        # Return a dummy parameter to make optimizer work
        # In practice, TabPFN doesn't need an optimizer
        dummy_param = nn.Parameter(torch.zeros(1, device=self.device))
        yield dummy_param
    
    def state_dict(self):
        """Return state dict with fitted flag."""
        return {
            'is_fitted': self.is_fitted,
            'X_train_shape': self.X_train.shape if self.X_train is not None else None,
            'y_train_shape': self.y_train.shape if self.y_train is not None else None,
        }
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict. Note: TabPFN weights are handled internally."""
        self.is_fitted = state_dict.get('is_fitted', False)
        # Note: We can't restore X_train and y_train from state_dict
        # The model needs to be re-fitted if loading from checkpoint

