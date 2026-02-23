import torch
from src.utils.encoder_utils import encode_with_entropy
from src.utils.loss_utils import get_regression_loss_fn
from src.training.GromovWassersteinLoss import GromovWassersteinLoss


def train_gw(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    gw_weight=1.0,
    gw_epsilon=1.0,
    x_normalized: bool = False,
    detach_transport: bool = True,
    use_log_ratio: bool = False,
    reg_weight: float = 1.0,
    reg_eps: float = 1e-6,
):
    """
    Train GW model with regression loss and GW loss (Riemann metric learning).
    Total loss = regression_loss + gw_weight * gw_loss
    """
    model.train()
    train_total_loss_sum = torch.tensor(0.0, device=device)
    train_gw_loss_sum = torch.tensor(0.0, device=device)
    train_regression_loss_sum = torch.tensor(0.0, device=device)
    
    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'

    # Prepare GW loss (Riemann metric learning)
    gw_loss_fn = GromovWassersteinLoss(
        epsilon=gw_epsilon,
        x_normalized=x_normalized,
        detach_transport=detach_transport,
        reg_weight=reg_weight,
        reg_eps=reg_eps,
    )

    # Select regression loss function
    regression_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)

    for x, labels in train_loader:
        x = x.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        # Ensure labels have shape (batch_size, 1) to match model output
        if labels.dim() == 1:
            labels = labels.unsqueeze(1) 
        
        optimizer.zero_grad()

        # Get latent representation z from encoder
        z, _ = encode_with_entropy(model.encoder, x)
        
        # Get regression prediction y_hat from head
        y_hat = model.head(z)
        
        # Ensure y_hat has correct shape
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(1)
        elif y_hat.dim() == 3:  # TabM output: (batch_size, k, out_dim)
            y_hat = y_hat.mean(dim=1)  # Average over heads

        # Calculate regression loss
        regression_loss = regression_loss_fn(y_hat, labels)

        # Calculate GW loss between input space X and latent space Z
        gw_loss = gw_loss_fn(x, z, labels)

        # Total loss: regression + GW loss
        total_loss = regression_loss + gw_weight * gw_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses on GPU (defer CPU operations)
        train_total_loss_sum += total_loss.detach()
        train_gw_loss_sum += gw_loss.detach()
        train_regression_loss_sum += regression_loss.detach()
    
    # Calculate average losses 
    num_batches = len(train_loader)
    if num_batches == 0:
        return float('nan')
    avg_total_loss = (train_total_loss_sum / num_batches).item()
    avg_gw_loss = (train_gw_loss_sum / num_batches).item()
    avg_regression_loss = (train_regression_loss_sum / num_batches).item()
    
    # Print epoch summary
    print(
        f"GW Training - Avg GW Loss: {avg_gw_loss:.6f}, Avg Regression Loss: {avg_regression_loss:.6f}, "
        f"Avg Total Loss: {avg_total_loss:.6f}"
    )
    
    return avg_total_loss
