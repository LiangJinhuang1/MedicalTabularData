import torch
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from src.training.SinkhornDistance import SinkhornDistance
from src.training.kl_loss import compute_kl_loss
from src.utils.loss_utils import get_regression_loss_fn


def eval_multitask(
    model,
    test_loader,
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
    deterministic_ot: bool = False,
    ot_seed: int = 0,
    return_components: bool = False,
):
    """
    Evaluate multi-task model with regression and reconstruction losses.
    Optionally adds MMD/Sinkhorn regularization for WAE-style evaluation.
    
    Args:
        model: MultiTaskEncoderEmbedding model
        test_loader: Test data loader
        loss_fn: Regression loss function (MSE)
        device: Device to run on
        dataset: Dataset object with get_variable_splits method (for reconstruction loss)
        tabular_loss_fn: TabularLoss function (for reconstruction loss)
        recon_weight: Weight for reconstruction loss (default: 1.0)
        lambda_ot: Weight for MMD/Sinkhorn regularization (if None, no regularization is applied)
        regularization_type: 'sinkhorn' or 'mmd' (required if lambda_ot is not None)
        sinkhorn_eps: Sinkhorn regularization parameter (default: 0.1)
        sinkhorn_max_iter: Sinkhorn iterations (default: 10)
        mmd_kernel_mul: MMD kernel multiplier (default: 2)
        mmd_kernel_num: MMD kernel number (default: 5)
        deterministic_ot: If True, use a fixed RNG for OT prior sampling (stable eval).
        ot_seed: Seed for deterministic OT sampling.
        return_components: If True, return component losses in an extra dict.
    """
    model.eval()
    total_loss = 0.0
    regression_loss_sum = 0.0
    recon_loss_sum = 0.0
    regularization_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    valid_batches = 0
    all_predictions = []
    all_labels = []

    # Prepare regularization (OT/MMD) if needed
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
    is_classification = isinstance(regression_loss_fn, torch.nn.BCELoss)
    ot_generator = None
    if deterministic_ot and lambda_ot is not None:
        ot_generator = torch.Generator()
        ot_generator.manual_seed(int(ot_seed))

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            # Ensure labels have shape (batch_size, 1) to match model output
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
            
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
            
            # Ensure y_hat has shape (batch_size, 1)
            if y_hat.dim() == 1:
                y_hat = y_hat.unsqueeze(1)
            elif y_hat.dim() == 3:  # TabM output: (batch_size, k, out_dim)
                y_hat = y_hat.mean(dim=1)  # Average over heads
            
            # Calculate regression loss
            regression_loss = regression_loss_fn(y_hat, labels)
            
            # Reconstruction loss
            recon_loss = torch.tensor(0.0, device=device)
            if recon_cont is not None and tabular_loss_fn is not None and dataset is not None:
                tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
                recon_loss, _ = tabular_loss_fn(
                    predictions=(recon_cont, recon_bin, recon_cat),
                    targets=(tgt_cont, tgt_bin, tgt_cat),
                )

            # Regularization loss (MMD/Sinkhorn) if needed
            regularization_loss = torch.tensor(0.0, device=device)
            if lambda_ot is not None and z is not None:
                if regularization_type == "sinkhorn" and regularizer_fn is not None:
                    if ot_generator is not None:
                        z_prior = torch.randn(z.shape, device="cpu", generator=ot_generator).to(z.device)
                    else:
                        z_prior = torch.randn_like(z)
                    regularization_loss = regularizer_fn(z, z_prior)
                elif regularization_type == "mmd" and mmd_fn is not None:
                    regularization_loss = mmd_fn(z)

            # Total loss: regression + reconstruction + regularization
            batch_loss = regression_loss + recon_weight * recon_loss
            if lambda_ot is not None:
                batch_loss = batch_loss + lambda_ot * regularization_loss
            
            sparsity_loss = torch.tensor(0.0, device=device)
            if hasattr(model, '_last_entropy') and model._last_entropy is not None:
                entropy = model._last_entropy
                if entropy.dim() > 0:
                    entropy = entropy.mean()
                sparsity_loss = entropy
            
            total_loss += batch_loss.item()
            regression_loss_sum += regression_loss.item()
            recon_loss_sum += recon_loss.item()
            regularization_loss_sum += regularization_loss.item()
            sparsity_loss_sum += sparsity_loss.item()
            all_predictions.append(y_hat)
            all_labels.append(labels)
            valid_batches += 1
    
    if valid_batches == 0:
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / valid_batches
    
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
        all_labels = torch.cat(all_labels, dim=0).cpu().detach()
        if is_classification:
            decision_threshold = getattr(loss_fn, 'decision_threshold', 0.5)
            probs = all_predictions
            preds = (probs > decision_threshold).float()
            labels_np = all_labels.numpy().ravel()
            preds_np = preds.numpy().ravel()
            probs_np = probs.numpy().ravel()
            f1 = f1_score(labels_np, preds_np)
            try:
                auc = roc_auc_score(labels_np, probs_np)
            except ValueError as e:
                print(f'Error calculating AUC: {e}')
                auc = float('nan')
            if return_components:
                components = {
                    "regression": regression_loss_sum / max(valid_batches, 1),
                    "reconstruction": recon_loss_sum / max(valid_batches, 1),
                    "regularization": regularization_loss_sum / max(valid_batches, 1),
                    "sparsity": sparsity_loss_sum / max(valid_batches, 1),
                }
                return avg_loss, f1, auc, components
            return avg_loss, f1, auc
        all_predictions = all_predictions.numpy()
        all_labels = all_labels.numpy()
        try:
            r2 = r2_score(all_labels, all_predictions)
        except ValueError as e:
            print(f'Error calculating R2 score: {e}')
            print(f'Labels shape: {all_labels.shape}')
            print(f'Predictions shape: {all_predictions.shape}')
            r2 = float('-inf')
    else:
        r2 = float('-inf')
    if return_components:
        components = {
            "regression": regression_loss_sum / max(valid_batches, 1),
            "reconstruction": recon_loss_sum / max(valid_batches, 1),
            "regularization": regularization_loss_sum / max(valid_batches, 1),
            "sparsity": sparsity_loss_sum / max(valid_batches, 1),
        }
        return avg_loss, r2, components
    return avg_loss, r2


def eval_multi_vae(
    model,
    test_loader,
    loss_fn,
    device,
    beta=1.0,
    dataset=None,
    tabular_loss_fn=None,
    recon_weight=1.0,
    use_log_ratio: bool = False,
):
    """
    Evaluate multi-task VAE model with regression, KL divergence, and reconstruction losses.
    
    Args:
        model: MultitaskVAEEncoderEmbedding model
        test_loader: Test data loader
        loss_fn: Regression loss function (MSE)
        device: Device to run on
        beta: Weight for KL loss (default: 1.0)
        dataset: Dataset object with get_variable_splits method (for reconstruction loss)
        tabular_loss_fn: TabularLoss function (for reconstruction loss)
        recon_weight: Weight for reconstruction loss (default: 1.0)
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    all_predictions = []
    all_labels = []
    
    regression_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)
    is_classification = isinstance(regression_loss_fn, torch.nn.BCELoss)

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device)

            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
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
            
            # Compute KL divergence loss using compute_kl_loss helper function
            kl_loss = compute_kl_loss(mu, log_var, beta=beta)
            
            # Calculate regression loss
            regression_loss = regression_loss_fn(y_hat, labels)
            
            # Reconstruction loss (VAE always uses standard tabular reconstruction)
            recon_loss = torch.tensor(0.0, device=device)
            if recon_cont is not None and tabular_loss_fn is not None and dataset is not None:
                tgt_cont, tgt_bin, tgt_cat = dataset.get_variable_splits(x)
                recon_loss, _ = tabular_loss_fn(
                    predictions=(recon_cont, recon_bin, recon_cat),
                    targets=(tgt_cont, tgt_bin, tgt_cat),
                )
            
            # Total loss: regression + KL + reconstruction
            batch_loss = regression_loss + kl_loss + recon_weight * recon_loss
            
            total_loss += batch_loss.item()
            all_predictions.append(y_hat)
            all_labels.append(labels)
            valid_batches += 1
    
    if valid_batches == 0:
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / valid_batches
    
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0).cpu().detach()
        all_labels = torch.cat(all_labels, dim=0).cpu().detach()
        if is_classification:
            decision_threshold = getattr(loss_fn, 'decision_threshold', 0.5)
            probs = all_predictions
            preds = (probs > decision_threshold).float()
            labels_np = all_labels.numpy().ravel()
            preds_np = preds.numpy().ravel()
            probs_np = probs.numpy().ravel()
            f1 = f1_score(labels_np, preds_np)
            try:
                auc = roc_auc_score(labels_np, probs_np)
            except ValueError as e:
                print(f'Error calculating AUC: {e}')
                auc = float('nan')
            return avg_loss, f1, auc
        all_predictions = all_predictions.numpy()
        all_labels = all_labels.numpy()
        try:
            r2 = r2_score(all_labels, all_predictions)
        except ValueError as e:
            print(f'Error calculating R2 score: {e}')
            print(f'Labels shape: {all_labels.shape}')
            print(f'Predictions shape: {all_predictions.shape}')
            r2 = float('-inf')
    else:
        r2 = float('-inf')
    return avg_loss, r2
