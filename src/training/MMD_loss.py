import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2, kernel_num = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num =kernel_num
        self.fix_sigma = None

    def gaussian_kernal(self, source, target, kernel_mul = 2, kernel_num =5, fix_sigma = None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim = 0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        #Bandwidth of Gaussian kernel
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # Add numerical stability: avoid division by zero and ensure positive bandwidth
            denominator = max(n_samples ** 2 - n_samples, 1)
            bandwidth = torch.sum(L2_distance.data) / denominator
            # Clamp bandwidth to avoid numerical issues
            bandwidth = torch.clamp(bandwidth, min=1e-6)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # Clamp bandwidths to avoid numerical instability
        bandwidth_list = [torch.clamp(b, min=1e-6) for b in bandwidth_list]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, z):
        batch_size = z.size()[0]
        
        # Ensure prior samples are on the same device as z
        prior_samples = torch.randn_like(z).detach()

        kernals = self.gaussian_kernal(z, prior_samples, self.kernel_mul, self.kernel_num, self.fix_sigma)

        XX = kernals[:batch_size, :batch_size]
        YY = kernals[batch_size:, batch_size:]
        XY = kernals[:batch_size, batch_size:]

        mmd = torch.mean(XX + YY - 2 * XY)
        
        # Clamp MMD to avoid numerical issues (should be non-negative)
        mmd = torch.clamp(mmd, min=0.0)
        
        return mmd