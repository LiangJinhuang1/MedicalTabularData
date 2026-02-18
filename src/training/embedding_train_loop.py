import torch
import math
from src.utils.save_utils import save_checkpoint
from src.utils.encoder_utils import encode_with_entropy
from src.training.MMD_loss import MMD_loss
from src.training.SinkhornDistance import SinkhornDistance
from src.training.kl_loss import compute_kl_loss
from src.eval.embedding_eval_loop import eval_tabae, eval_tabvae, eval_tabwae


def train_tabae(model_tabae, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, loss_tracker, 
                experiment_dir=None, dataset=None, model_name='tabae', lambda_sparse=0.0001, logger=None):
    """
    Train TabAE with TabularLoss (unsupervised pre-training).
    """
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'
    
    for epoch in range(num_epochs):
        model_tabae.train()
        epoch_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_non_blocking)
            
            # Forward pass: TabAE returns (rec_cont, rec_bin, rec_cat)
            recon_cont, recon_bin, recon_cat = model_tabae(x)
            
            # Split input by variable type
            tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
            
            # Compute reconstruction loss
            recon_loss, _ = loss_fn(
                predictions=(recon_cont, recon_bin, recon_cat),
                targets=(tgt_cont, tgt_bin, tgt_cat)
            )
            
            # Add sparsity loss if encoder returned entropy (e.g., TabNetEncoder)
            total_loss = recon_loss
            entropy = getattr(model_tabae, '_last_entropy', None)
            if entropy is not None and lambda_sparse > 0:
                total_loss = total_loss + lambda_sparse * entropy
            
            # Check for NaN/Inf and skip if found
            if not torch.isfinite(total_loss):
                print(f'Warning: NaN/Inf loss detected at epoch {epoch}, skipping batch')
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.detach()
            valid_batches += 1
        
        if valid_batches == 0:
            avg_train_loss = float('nan')
        else:
            avg_train_loss = (epoch_loss / valid_batches).item()
        
        avg_val_loss = eval_tabae(model_tabae, val_loader, loss_fn, device, dataset)
        loss_tracker.update(epoch, model_name, avg_train_loss, avg_val_loss)

        if logger is not None:
            logger.info(
                f'{model_name} | Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            )
        
        if not (math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                if experiment_dir is not None:
                    is_best = True
                    save_checkpoint(
                        experiment_dir, epoch, model_tabae, optimizer,
                        avg_train_loss, avg_val_loss, model_name, is_best
                    )
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            msg = f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            print(msg)
            if logger is not None:
                logger.info(f'{model_name} | {msg}')
    
    msg = f'TabAE training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}'
    print(f'\n{msg}')
    if logger is not None:
        logger.info(f'{model_name} | {msg}')
    return best_val_loss


def train_tabvae(model_tabvae, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, loss_tracker, 
                 experiment_dir=None, dataset=None, model_name='tabvae', beta=1.0, lambda_sparse=0.0001, logger=None):
    """
    Train TabVAE with TabularLoss (reconstruction loss) and KL divergence loss (unsupervised pre-training).
    
    """
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'
    
    for epoch in range(num_epochs):
        model_tabvae.train()
        epoch_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_non_blocking)
            
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
            
            # Add sparsity loss if encoder returned entropy (e.g., TabNetEncoder)
            total_loss = recon_loss + kl_loss
            entropy = getattr(model_tabvae, '_last_entropy', None)
            if entropy is not None and lambda_sparse > 0:
                total_loss = total_loss + lambda_sparse * entropy
            
            # Check for NaN/Inf and skip if found
            if not torch.isfinite(total_loss):
                print(f'Warning: NaN/Inf loss detected at epoch {epoch}, skipping batch')
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent explosion (especially important for KL divergence)
            torch.nn.utils.clip_grad_norm_(model_tabvae.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.detach()
            valid_batches += 1
        
        if valid_batches == 0:
            avg_train_loss = float('nan')
        else:
            avg_train_loss = (epoch_loss / valid_batches).item()
        
        avg_val_loss = eval_tabvae(model_tabvae, val_loader, loss_fn, device, dataset, beta=beta)
        loss_tracker.update(epoch, model_name, avg_train_loss, avg_val_loss)

        if logger is not None:
            logger.info(
                f'{model_name} | Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            )
        
        if not (math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                if experiment_dir is not None:
                    is_best = True
                    save_checkpoint(
                        experiment_dir, epoch, model_tabvae, optimizer,
                        avg_train_loss, avg_val_loss, model_name, is_best
                    )
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            msg = f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            print(msg)
            if logger is not None:
                logger.info(f'{model_name} | {msg}')
    
    msg = f'TabVAE training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}'
    print(f'\n{msg}')
    if logger is not None:
        logger.info(f'{model_name} | {msg}')
    return best_val_loss


def train_tabwae(
    model_tabwae,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    loss_tracker,
    experiment_dir=None,
    dataset=None,
    model_name='tabwae',
    lambda_ot=1.0,
    regularization_type='sinkhorn',
    sinkhorn_eps=0.1,
    sinkhorn_max_iter=10,
    mmd_kernel_mul=2,
    mmd_kernel_num=5,
    lambda_sparse=0.0001,
    logger=None,
):
    """
    Train TabAE with TabularLoss (reconstruction loss) and MMD/Sinkhorn regularization (WAE-style).
    This is essentially TabAE with additional regularization in the training loop.
    """
   
    best_val_loss = float('inf')
    best_epoch = -1

    # Initialize regularization function based on type
    if regularization_type == 'sinkhorn':
        sinkhorn_fn = SinkhornDistance(eps=sinkhorn_eps, max_iter=sinkhorn_max_iter)
        mmd_fn = None
    elif regularization_type == 'mmd':
        sinkhorn_fn = None
        mmd_fn = MMD_loss(kernel_mul=mmd_kernel_mul, kernel_num=mmd_kernel_num)
    else:
        raise ValueError(f"Invalid regularization_type: {regularization_type}. Must be 'mmd' or 'sinkhorn'")

    pin_memory = train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'
    
    for epoch in range(num_epochs):
        model_tabwae.train()
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_loss_recon = torch.tensor(0.0, device=device)
        epoch_loss_reg = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_non_blocking)

            # Encode once for both reconstruction and regularization
            z, entropy = encode_with_entropy(model_tabwae.encoder, x)
            recon_cont, recon_bin, recon_cat = model_tabwae.decoder(z)

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

            # Add sparsity loss if encoder returned entropy
            loss = recon_loss + lambda_ot * ot_loss
            if entropy is not None and lambda_sparse > 0:
                sparsity_loss = lambda_sparse * entropy
                loss = loss + sparsity_loss
            
            # Check for NaN/Inf and skip if found
            if not torch.isfinite(loss):
                print(f'Warning: NaN/Inf loss detected at epoch {epoch}, skipping batch')
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_tabwae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses on GPU (defer CPU operations)
            epoch_loss += loss.detach()
            epoch_loss_recon += recon_loss.detach()
            epoch_loss_reg += ot_loss.detach()
            valid_batches += 1
        
        if valid_batches == 0:
            avg_train_loss = float('nan')
        else:
            avg_train_loss = (epoch_loss / valid_batches).item()
        
        avg_val_loss = eval_tabwae(
            model_tabwae,
            val_loader,
            loss_fn,
            device,
            dataset,
            lambda_ot,
            regularization_type,
            sinkhorn_fn,
            mmd_fn,
        )
        loss_tracker.update(epoch, model_name, avg_train_loss, avg_val_loss)

        if logger is not None:
            if valid_batches > 0:
                avg_recon = epoch_loss_recon / valid_batches
                avg_reg = epoch_loss_reg / valid_batches
            else:
                avg_recon = float('nan')
                avg_reg = float('nan')
            reg_name = regularization_type.upper()
            logger.info(
                f'{model_name} | Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} '
                f'(Recon: {avg_recon:.4f}, {reg_name}: {avg_reg:.4f}), Val Loss: {avg_val_loss:.4f}'
            )
        
        if not (math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                if experiment_dir is not None:
                    is_best = True
                    save_checkpoint(
                        experiment_dir, epoch, model_tabwae, optimizer,
                        avg_train_loss, avg_val_loss, model_name, is_best
                    )
        
        reg_name = regularization_type.upper()
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            if valid_batches > 0:
                avg_recon = epoch_loss_recon / valid_batches
                avg_reg = epoch_loss_reg / valid_batches
            else:
                avg_recon = float('nan')
                avg_reg = float('nan')
            msg = (f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} '
                   f'(Recon: {avg_recon:.4f}, {reg_name}: {avg_reg:.4f}), Val Loss: {avg_val_loss:.4f}')
            print(msg)
            if logger is not None:
                logger.info(f'{model_name} | {msg}')
    
    msg = f'TabWAE training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}'
    print(f'\n{msg}')
    if logger is not None:
        logger.info(f'{model_name} | {msg}')
    return best_val_loss
