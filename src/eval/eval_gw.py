import torch
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from src.utils.encoder_utils import encode_with_entropy
from src.utils.loss_utils import get_regression_loss_fn
from src.training.GromovWassersteinLoss import GromovWassersteinLoss


def eval_gw(
    model,
    test_loader,
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
    Evaluate GW model with regression loss and GW loss (Riemann metric learning).
    Total loss = regression_loss + gw_weight * gw_loss
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    all_predictions = []
    all_labels = []

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
    is_classification = isinstance(regression_loss_fn, torch.nn.BCELoss)

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            # Ensure labels have shape (batch_size, 1) to match model output
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            # Get latent representation z from encoder (may return (z, entropy) tuple)
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
            batch_loss = regression_loss + gw_weight * gw_loss
            
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
