import torch.nn as nn
from src.training.Tabuarloss import TabularLoss
from src.training.log_ratio_loss import LogRatioLoss


def get_regression_loss_fn(loss_fn, use_log_ratio):
    """Select regression loss for tabular models."""
    if use_log_ratio:
        return LogRatioLoss()
    if isinstance(loss_fn, TabularLoss):
        return nn.MSELoss()
    return loss_fn
