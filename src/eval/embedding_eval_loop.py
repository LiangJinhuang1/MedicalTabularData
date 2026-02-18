import torch
from src.utils.encoder_utils import encode_with_entropy
from src.training.kl_loss import compute_kl_loss


def eval_tabae(model_tabae, val_loader, loss_fn, device, dataset):
    """
    Evaluate TabAE with TabularLoss (reconstruction loss).
    """
    model_tabae.eval()
    total_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            
            # Forward pass: TabAE returns (rec_cont, rec_bin, rec_cat)
            recon_cont, recon_bin, recon_cat = model_tabae(x)
            
            # Split input by variable type
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
            
            # Compute reconstruction loss
            loss, _ = loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat)
            )
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches


def eval_tabvae(model_tabvae, val_loader, loss_fn, device, dataset, beta=1.0):
    """
    Evaluate TabVAE with TabularLoss (reconstruction loss) and KL divergence loss.
    """
    model_tabvae.eval()
    total_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            
            # Forward pass: TabVAE returns (recon_cont, recon_bin, recon_cat, mu, log_var, z)
            recon_cont, recon_bin, recon_cat, mu, log_var, z = model_tabvae(x)
            
            # Split input by variable type
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
            
            # Compute reconstruction loss
            recon_loss, _ = loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat)
            )
            
            # Compute KL divergence loss using compute_kl_loss helper function
            kl_loss = compute_kl_loss(mu, log_var, beta=beta)
            
            # Total loss: reconstruction + KL
            loss = recon_loss + kl_loss
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches


def eval_tabwae(
    model_tabae,
    val_loader,
    loss_fn,
    device,
    dataset,
    lambda_ot=1.0,
    regularization_type='sinkhorn',
    sinkhorn_fn=None,
    mmd_fn=None,
):
    """
    Evaluate TabAE with TabularLoss (reconstruction loss) and MMD/Sinkhorn regularization (WAE-style).
    """
    model_tabae.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            # Encode once for both reconstruction and regularization
            z, _ = encode_with_entropy(model_tabae.encoder, x)
            recon_cont, recon_bin, recon_cat = model_tabae.decoder(z)

            # Standard WAE mode: reconstruction + OT regularization
            # Split input by variable type
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)

            # Compute reconstruction loss
            recon_loss, _ = loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat),
            )

            # Regularization loss (MMD or Sinkhorn)
            if regularization_type == 'mmd':
                ot_loss = mmd_fn(z)
            else:  # sinkhorn
                z_prior = torch.randn_like(z)  # Sample from prior (standard Gaussian)
                ot_loss = sinkhorn_fn(z, z_prior)

            loss = recon_loss + lambda_ot * ot_loss
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches
