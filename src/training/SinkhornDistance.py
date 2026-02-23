import torch
import torch.nn as nn



class SinkhornDistance(nn.Module):
    """
    Computes the Sinkhorn Divergence between two batches of samples.
    Uses the Log-Sum-Exp trick for numerical stability.
    """
    def __init__(self, eps=0.1, max_iter=10):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
    
    def forward(self, x, y):
        """
        x: Latent codes from Encoder [Batch, Dim]
        y: Samples from Prior (Gaussian) [Batch, Dim]
        """
        batch_size = x.shape[0]
        
        # The Cost Matrix: Squared Euclidean Distance
        # C_ij = ||x_i - y_j||^2
        x_col = x.unsqueeze(1) # [B, 1, Dim]
        y_lin = y.unsqueeze(0) # [1, B, Dim]
        C = torch.sum((x_col - y_lin) ** 2, dim=2)

        # Optimal Transport via Sinkhorn-Knopp
        # We solve for potentials f and g in the log-domain
        
        # Initialize potentials to 0 (vectors of size Batch)
        f = torch.zeros_like(x[:, 0]) 
        g = torch.zeros_like(y[:, 0])
        
        # The Log-Domain Kernel
        # K_log = -C / epsilon
        K_log = -C / self.eps

        for _ in range(self.max_iter):
            # Update f (rows): f = -eps * LSE( (g - C) / eps )
            # We use LogSumExp to avoid underflow of small probabilities
            f_new = -self.eps * torch.logsumexp((g.unsqueeze(0) + K_log), dim=1)
            f = f_new
            
            # Update g (cols): g = -eps * LSE( (f - C) / eps )
            g_new = -self.eps * torch.logsumexp((f.unsqueeze(1) + K_log), dim=0)
            g = g_new

        # The Sinkhorn Distance
        # P = exp( (f + g - C) / eps )
        # Distance = mean( P * C ) 
        
        # Compute the Transport Plan in Log Domain
        P_log = (f.unsqueeze(1) + g.unsqueeze(0) - C) / self.eps
        
        # Compute distance in log space using LogSumExp for numerical stability
        # dist = mean(P * C) = mean(exp(P_log) * C)
        # We compute: log(P * C) = P_log + log(C + eps) to avoid log(0)
        # Then use LogSumExp: log(dist) = logsumexp(P_log + log(C + eps)) - log(batch_size^2)
        log_C = torch.log(C + 1e-8)  # Add small epsilon to avoid log(0)
        log_PC = P_log + log_C  # log(P * C) = log(P) + log(C)
        
        # Use LogSumExp to compute sum(P * C) in log space, then divide by batch_size^2
        log_sum_PC = torch.logsumexp(log_PC.flatten(), dim=0)
        log_dist = log_sum_PC - 2 * torch.log(torch.tensor(batch_size, dtype=torch.float32, device=x.device))
        
        # Convert back from log space
        dist = torch.exp(log_dist)
        
        # Final check for NaN/Inf
        if torch.isnan(dist) or torch.isinf(dist) or dist > 1e6:
            # Fallback: use simple L2 distance if Sinkhorn fails
            dist = torch.mean((x - y) ** 2)
        
        return dist