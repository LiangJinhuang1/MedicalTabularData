import torch
import torch.nn as nn


class LogRatioLoss(nn.Module):
    """
    Log Ratio Loss for regression tasks.
    
    This loss computes the squared difference of logarithms:
    loss = mean((log(y_pred + eps) - log(y_true + eps))^2)
    
    Args:
        eps: Small epsilon value to avoid log(0) (default: 1e-8)
    """
    def __init__(self, eps=1e-8):
        super(LogRatioLoss, self).__init__()
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        """
        Compute log ratio loss.
        
        Args:
            y_pred: Predicted values (tensor)
            y_true: True values (tensor)
        
        Returns:
            loss: Log ratio loss (scalar tensor)
        """
        # Ensure both tensors are positive by adding epsilon
        y_pred_safe = y_pred + self.eps
        y_true_safe = y_true + self.eps
        
        # Compute log ratio loss: (log(y_pred) - log(y_true))^2
        log_pred = torch.log(y_pred_safe)
        log_true = torch.log(y_true_safe)
        loss = torch.mean((log_pred - log_true) ** 2)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            # Fallback to MSE if log ratio fails
            loss = nn.MSELoss()(y_pred, y_true)
        
        return loss

