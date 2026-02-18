import torch

def compute_kl_loss(mu, log_var, beta=1.0):
    """
    Compute KL divergence loss for VAE.
    This is a helper function that can be used independently of reconstruction loss.
    
    Args:
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution (will be converted to log_std internally)
        beta: Weight for KL loss (default: 1.0)
    
    Returns:
        kl_loss: KL divergence loss (scalar tensor)
    """
    # Convert log_var to log_std (log_std = 0.5 * log_var)
    log_std = 0.5 * log_var
    
    # Use numerically stable computation: kl = 0.5 * sum(exp(2*log_std) + mu^2 - 1 - 2*log_std)
    # clamp the log_std to avoid numerical instability
    log_std_clamped = torch.clamp(log_std, min=-5, max=5)
    kl = -0.5 * torch.mean(1 + 2*log_std_clamped - mu.pow(2) - (2*log_std_clamped).exp())
    
    # Check for NaN/Inf
    if torch.isnan(kl) or torch.isinf(kl):
        kl = torch.tensor(0.0, device=kl.device)
    
    kl_loss = beta * kl
    return kl_loss

