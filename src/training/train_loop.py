import torch
from src.utils.loss_utils import get_regression_loss_fn


def train_loop(model, loader, optimizer, loss_fn, device, model_type: str, use_log_ratio: bool = False, 
               lambda_sparse: float = 0.0):
    """
    Training loop for various models.
    
    Args:
        model: Model to train
        loader: Data loader
        optimizer: Optimizer
        loss_fn: Loss function (MSE or LogRatioLoss)
        device: Device to run on
        model_type: Type of model ('mlp', 'tabm', 'tabnet')
        use_log_ratio: Whether to use LogRatioLoss instead of MSE
        lambda_sparse: Weight for TabNet sparsity loss (entropy). If > 0, enables sparsity loss.
                      Default: 0.0 (disabled for backward compatibility)
    """
    
    
    model.train()
    # If the encoder is frozen, keep it in eval mode to avoid BN/Dropout drift.
    encoder = getattr(model, 'encoder', None)
    if encoder is not None:
        encoder_frozen = True
        for param in encoder.parameters():
            if param.requires_grad:
                encoder_frozen = False
                break
        if encoder_frozen:
            encoder.eval()
    total_loss = torch.tensor(0.0, device=device)
    
    pin_memory = loader.pin_memory if hasattr(loader, 'pin_memory') else False
    use_non_blocking = pin_memory and device.type == 'cuda'


    actual_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)

    for x, labels in loader:
        x = x.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)
        
        # Ensure labels have shape (batch_size, 1) to match model output
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        
        #forward pass
        outputs_raw = model(x)

        if model_type == 'mlp':
            loss = actual_loss_fn(outputs_raw, labels)
        elif model_type == 'tabm':
            # TabM output can be either:
            # - 3D: (batch_size, k, d_out) from direct TabM model
            # - 2D: (batch_size, d_out) from EncoderEmbedding (already averaged)
            if outputs_raw.dim() == 3:
                outputs = outputs_raw.mean(dim=1)
                loss = actual_loss_fn(outputs, labels)
            elif outputs_raw.dim() == 2:
                # EncoderEmbedding output: (batch_size, d_out) - already averaged
                loss = actual_loss_fn(outputs_raw, labels)
            else:
                raise ValueError(f'Unexpected TabM output dimension: {outputs_raw.dim()}, shape: {outputs_raw.shape}')
        elif model_type == 'tabnet':
            # TabNet can return either:
            # - Single output: (batch_size, output_dim) - backward compatible
            # - Tuple: (output, entropy) - when return_entropy=True
            if isinstance(outputs_raw, tuple):
                # Model returns (output, entropy)
                outputs_raw, entropy = outputs_raw
                pred_loss = actual_loss_fn(outputs_raw, labels)
                # Add sparsity loss (entropy) if lambda_sparse > 0
                if lambda_sparse > 0:
                    sparsity_loss = lambda_sparse * entropy
                    loss = pred_loss + sparsity_loss
                else:
                    loss = pred_loss
            else:
                # Backward compatible: only output, no entropy
                loss = actual_loss_fn(outputs_raw, labels)
        else:
            raise ValueError(f'Unknown model type {model_type}')


        #backward pass
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #update parameters
        optimizer.step()

        total_loss += loss.detach()
    
    return (total_loss / len(loader)).item()

    
