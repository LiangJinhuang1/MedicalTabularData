import torch
from src.utils.loss_utils import get_regression_loss_fn
from src.training.SinkhornDistance import SinkhornDistance
from src.training.kl_loss import compute_kl_loss


def train_multitask(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    dataset=None,
    tabular_loss_fn=None,
    recon_weight=1.0,
    lambda_ot=None,
    regularization_type=None,
    sinkhorn_eps=0.1,
    sinkhorn_max_iter=10,
    mmd_kernel_mul=2,
    mmd_kernel_num=5,
    use_log_ratio: bool = False,
    lambda_sparse: float = 0.0,
    return_components: bool = False,
):
    """
    Train multi-task model with regression and reconstruction losses.
    Optionally adds MMD/Sinkhorn regularization for WAE-style training.
    If return_components=True, returns (avg_total_loss, components dict).
    """
    model.train()
    train_total_loss_sum = torch.tensor(0.0, device=device)
    train_regression_loss_sum = torch.tensor(0.0, device=device)
    train_recon_loss_sum = torch.tensor(0.0, device=device)
    train_regularization_loss_sum = torch.tensor(0.0, device=device)
    train_sparsity_loss_sum = torch.tensor(0.0, device=device)
    valid_batches = 0
    
    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'

    # Prepare regularization (OT/MMD) 
    if lambda_ot is not None:
        if regularization_type not in {"sinkhorn", "mmd"}:
            raise ValueError(
                f"Invalid regularization_type: {regularization_type}. Must be 'sinkhorn' or 'mmd'."
            )
        if regularization_type == "sinkhorn":
            regularizer_fn = SinkhornDistance(eps=sinkhorn_eps, max_iter=sinkhorn_max_iter)
            mmd_fn = None
        else:
            from src.training.MMD_loss import MMD_loss

            regularizer_fn = None
            mmd_fn = MMD_loss(kernel_mul=mmd_kernel_mul, kernel_num=mmd_kernel_num)
    else:
        regularizer_fn = None
        mmd_fn = None
    
    regression_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)

    for x, labels in train_loader:
        x = x.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        # Ensure labels have shape (batch_size, 1) to match model output
        if labels.dim() == 1:
            labels = labels.unsqueeze(1) 
        
        optimizer.zero_grad()

        # MultiTaskEncoderEmbedding returns (y_hat, z, recon_cont, recon_bin, recon_cat)
        outputs = model(x)
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(f"Expected tuple/list outputs, got {type(outputs)}")
        if len(outputs) == 5:
            # Format with decoder: (y_hat, z, recon_cont, recon_bin, recon_cat)
            y_hat, z, recon_cont, recon_bin, recon_cat = outputs
        elif len(outputs) == 2:
            # Format without decoder: (y_hat, z)
            y_hat, z = outputs
            recon_cont, recon_bin, recon_cat = None, None, None
        else:
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
        
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(1)
        elif y_hat.dim() == 3: 
            y_hat = y_hat.mean(dim=1) 
        
        regression_loss = regression_loss_fn(y_hat, labels)

        # Reconstruction loss
        recon_loss = torch.tensor(0.0, device=device)
        if recon_cont is not None and tabular_loss_fn is not None and dataset is not None:
            # Standard tabular reconstruction loss 
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
            recon_loss, _ = tabular_loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat),
            )

        # Regularization loss (MMD/Sinkhorn)
        regularization_loss = torch.tensor(0.0, device=device)
        if lambda_ot is not None and z is not None and regularization_type is not None:
            if regularization_type == "sinkhorn" and regularizer_fn is not None:
                z_prior = torch.randn_like(z)
                regularization_loss = regularizer_fn(z, z_prior)
            elif regularization_type == "mmd" and mmd_fn is not None:
                regularization_loss = mmd_fn(z)

        # Total loss: regression + reconstruction + regularization
        total_loss = regression_loss + recon_weight * recon_loss
        if lambda_ot is not None:
            total_loss = total_loss + lambda_ot * regularization_loss
        
        # Add sparsity loss if encoder returned entropy (e.g., TabNetEncoder)
        sparsity_loss = torch.tensor(0.0, device=device)
        if hasattr(model, '_last_entropy') and model._last_entropy is not None:
            if lambda_sparse > 0:
                # Ensure entropy is a scalar tensor (not a vector)
                entropy = model._last_entropy
                if entropy.dim() > 0:
                    entropy = entropy.mean()
                sparsity_loss = lambda_sparse * entropy
                total_loss = total_loss + sparsity_loss
        
        # Skip batch if loss is NaN/Inf (WAE multi-task can spike)
        if not torch.isfinite(total_loss):
            continue
        
        total_loss.backward()
        
        # Gradient clipping to prevent explosion 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_total_loss_sum += total_loss.detach()
        train_regression_loss_sum += regression_loss.detach()
        train_recon_loss_sum += recon_loss.detach()
        train_regularization_loss_sum += regularization_loss.detach()
        train_sparsity_loss_sum += sparsity_loss.detach()
        valid_batches += 1
    
    if valid_batches == 0:
        return (float('nan'), None) if return_components else float('nan')
    avg_total = (train_total_loss_sum / valid_batches).item()
    if not return_components:
        return avg_total
    components = {
        "regression": (train_regression_loss_sum / valid_batches).item(),
        "reconstruction": (train_recon_loss_sum / valid_batches).item(),
        "regularization": (train_regularization_loss_sum / valid_batches).item(),
        "sparsity": (train_sparsity_loss_sum / valid_batches).item(),
    }
    return avg_total, components


def train_multi_vae(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    beta=1.0,
    dataset=None,
    tabular_loss_fn=None,
    recon_weight=1.0,
    use_log_ratio: bool = False,
    lambda_sparse: float = 0.0,
):
    """
    Train multi-task VAE model with regression, KL divergence, and reconstruction losses.
    """
    model.train()
    train_total_loss_sum = torch.tensor(0.0, device=device)
    
    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'
    
    regression_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)

    for x, labels in train_loader:
        x = x.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        # Ensure labels have shape (batch_size, 1) to match model output
        if labels.dim() == 1:
            labels = labels.unsqueeze(1) 
        
        optimizer.zero_grad()

        # MultitaskVAEEncoderEmbedding returns (y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat)
        outputs = model(x)
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(f"Expected tuple/list outputs, got {type(outputs)}")
        if len(outputs) == 7:
            # Format with decoder: (y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat)
            y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat = outputs
        elif len(outputs) == 4:
            # Format without decoder: (y_hat, z, mu, log_var)
            y_hat, z, mu, log_var = outputs
            recon_cont, recon_bin, recon_cat = None, None, None
        else:
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")

        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(1)
        elif y_hat.dim() == 3:
            y_hat = y_hat.mean(dim=1)

        # Compute KL divergence loss 
        kl_loss = compute_kl_loss(mu, log_var, beta=beta)
        regression_loss = regression_loss_fn(y_hat, labels)

        # Reconstruction loss 
        recon_loss = torch.tensor(0.0, device=device)
        if recon_cont is not None and tabular_loss_fn is not None and dataset is not None:
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
            recon_loss, _ = tabular_loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat),
            )

        # Total loss: regression + KL + reconstruction
        total_loss = regression_loss + kl_loss + recon_weight * recon_loss
        
        # Add sparsity loss if encoder returned entropy (e.g., TabNetEncoder)
        if hasattr(model, '_last_entropy') and model._last_entropy is not None:
            if lambda_sparse > 0:
                # Ensure entropy is a scalar tensor (not a vector)
                entropy = model._last_entropy
                if entropy.dim() > 0:
                    entropy = entropy.mean()
                sparsity_loss = lambda_sparse * entropy
                total_loss = total_loss + sparsity_loss
        
        total_loss.backward()
        
        # Gradient clipping to prevent explosion (especially important for KL divergence)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_total_loss_sum += total_loss.detach()
    
    if len(train_loader) == 0:
        return float('nan')
    return (train_total_loss_sum / len(train_loader)).item()
