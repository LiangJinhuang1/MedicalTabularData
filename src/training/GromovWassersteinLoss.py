import torch
import torch.nn as nn
import torch.nn.functional as F

class GromovWassersteinLoss(nn.Module):
    def __init__(
        self,
        epsilon=1.0,
        max_iter=50,
        x_normalized=False,
        detach_transport: bool = True,
        reg_weight: float = 1,
        reg_eps: float = 1e-6,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.x_normalized = x_normalized
        self.detach_transport = detach_transport
        self.reg_weight = reg_weight
        self.reg_eps = reg_eps

    def forward(self, x, z, y=None):
        batch_size = x.size(0)
        
        # 1. Compute Cost Matrices (Squared Euclidean Distance: ||x_i - x_j||_2^2)
        # D* = ||x_i - x_j||_2^2 for input space
        # D_z = ||z_i - z_j||_2^2 for latent space
        # This enforces global isometry: if patients are distant in input space,
        # they should remain distant in latent space (prevents mode collapse)
        
        # Compute pairwise squared Euclidean distances
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # (batch_size, 1)
        C_x_sq = x_sq + x_sq.t() - 2 * torch.mm(x, x.t())  # (batch_size, batch_size)
        

        # Squared Euclidean distance (no sqrt)
        C_x = torch.clamp(C_x_sq, min=0.0)
        
        z_sq = torch.sum(z ** 2, dim=1, keepdim=True)  # (batch_size, 1)
        C_z_sq = z_sq + z_sq.t() - 2 * torch.mm(z, z.t())  # (batch_size, batch_size)
        
        # Squared Euclidean distance (no sqrt)
        C_z = torch.clamp(C_z_sq, min=0.0)
        
        # Normalize matrices to [0,1]
        scale_x = torch.mean(C_x).detach() + 1e-8
        C_x = C_x / scale_x
        
        scale_z = torch.mean(C_z).detach() + 1e-8
        C_z = C_z / scale_z

        # 2. Estimate Transport Plan T
        # Optionally detach to prevent backprop through the Sinkhorn loop.
        if self.detach_transport:
            with torch.no_grad():
                C_x_const = C_x.detach()
                C_z_const = C_z.detach()
                # Initialize T (Uniform)
                T = torch.ones(batch_size, batch_size, device=x.device) / batch_size
                # Pre-compute square terms for the cost matrix M
                C_x_sq = C_x_const ** 2
                C_z_sq = C_z_const ** 2
                # Marginal terms (constant across iterations)
                const_x = torch.sum(C_x_sq * (1 / batch_size), dim=1, keepdim=True)
                const_z = torch.sum(C_z_sq * (1 / batch_size), dim=1, keepdim=True).t()
                for _ in range(self.max_iter):
                    cross_term = torch.mm(torch.mm(C_x_const, T), C_z_const.t())
                    M = const_x + const_z - 2 * cross_term
                    exponent = -M / self.epsilon
                    max_result = torch.max(exponent, dim=1, keepdim=True)
                    if isinstance(max_result, tuple):
                        exponent_max = max_result[0]
                    else:
                        exponent_max = max_result.values
                    K = torch.exp(exponent - exponent_max)
                    for _ in range(5):
                        K = K / (K.sum(dim=1, keepdim=True) + 1e-8) * (1.0 / batch_size)
                        K = K / (K.sum(dim=0, keepdim=True) + 1e-8) * (1.0 / batch_size)
                    T = K
        else:
            C_x_const = C_x
            C_z_const = C_z
            # Initialize T (Uniform)
            T = torch.ones(batch_size, batch_size, device=x.device) / batch_size
            # Pre-compute square terms for the cost matrix M
            C_x_sq = C_x_const ** 2
            C_z_sq = C_z_const ** 2
            # Marginal terms (constant across iterations)
            const_x = torch.sum(C_x_sq * (1 / batch_size), dim=1, keepdim=True)
            const_z = torch.sum(C_z_sq * (1 / batch_size), dim=1, keepdim=True).t()
            for _ in range(self.max_iter):
                cross_term = torch.mm(torch.mm(C_x_const, T), C_z_const.t())
                M = const_x + const_z - 2 * cross_term
                exponent = -M / self.epsilon
                max_result = torch.max(exponent, dim=1, keepdim=True)
                if isinstance(max_result, tuple):
                    exponent_max = max_result[0]
                else:
                    exponent_max = max_result.values
                K = torch.exp(exponent - exponent_max)
                for _ in range(5):
                    K = K / (K.sum(dim=1, keepdim=True) + 1e-8) * (1.0 / batch_size)
                    K = K / (K.sum(dim=0, keepdim=True) + 1e-8) * (1.0 / batch_size)
                T = K

        # 3. Compute Final GW Loss 
        # The expansion of |D_x - D_z|^2 results in constant terms that depend
        # only on the distance matrices, not on the alignment T.
        
        # Term 1: Mean squared distance in Input space
        const_term_x = torch.mean(C_x ** 2) 
        
        # Term 2: Mean squared distance in Latent space
        const_term_z = torch.mean(C_z ** 2)
        
        # Cross Term: The structural alignment
        # Compute: sum((C_x @ T @ C_z^T) * T)
        # Note: T is computed in no_grad context but can be used in gradient computation
        cross_term_matrix = torch.mm(torch.mm(C_x, T), C_z.t()) * T
        cross_term = torch.sum(cross_term_matrix)
        
        # Ensure cross_term is a scalar
        if cross_term.dim() > 0:
            cross_term = cross_term.squeeze()
        
        gw_loss = const_term_x + const_term_z - 2 * cross_term
        
        # Ensure gw_loss is a scalar
        if gw_loss.dim() > 0:
            gw_loss = gw_loss.squeeze()
            if gw_loss.dim() > 0:
                gw_loss = gw_loss.mean()
        
        gw_loss = torch.clamp(gw_loss, min=0.0)

        # 4. Outcome-aware metric regression
        if self.reg_weight > 0.0 and y is not None:
            if y.dim() == 1:
                y = y.unsqueeze(1)
            y_diff = torch.abs(y - y.t())
            z_dist = torch.sqrt(torch.clamp(C_z_sq, min=1e-8))
            log_y = torch.log(y_diff + self.reg_eps)
            log_z = torch.log(z_dist + self.reg_eps)
            diff = (log_y - log_z) ** 2
            diag_mask = ~torch.eye(batch_size, device=diff.device, dtype=torch.bool)
            reg_loss = diff[diag_mask].mean()
            return gw_loss + self.reg_weight * reg_loss

        return gw_loss
