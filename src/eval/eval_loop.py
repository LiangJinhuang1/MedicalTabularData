import torch
from sklearn.metrics import f1_score, r2_score, roc_auc_score
import numpy as np
from src.utils.loss_utils import get_regression_loss_fn

def eval_loop(model, loader, loss_fn, device, model_type: str, use_log_ratio: bool = False):
    model.eval()
    total_loss = 0
    valid_batches = 0
    all_predictions = []
    all_labels = []

    # Select loss function based on use_log_ratio flag
    actual_loss_fn = get_regression_loss_fn(loss_fn, use_log_ratio)
    is_classification = isinstance(actual_loss_fn, torch.nn.BCELoss)

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1) 
            
            #forward pass
            outputs_raw = model(x)

            if model_type == 'mlp':
                outputs = outputs_raw
            elif model_type =='tabm':
                # TabM output can be either:
                # - 3D: (batch_size, k, d_out) from direct TabM model
                # - 2D: (batch_size, d_out) from EncoderEmbedding (already averaged)
                if outputs_raw.dim() == 3:
                    # Direct TabM output: average over heads
                    outputs = outputs_raw.mean(dim=1)
                elif outputs_raw.dim() == 2:
                    # EncoderEmbedding output: already averaged
                    outputs = outputs_raw
                else:
                    raise ValueError(f'Unexpected TabM output dimension: {outputs_raw.dim()}, shape: {outputs_raw.shape}')
            elif model_type == 'tabnet':
                # TabNet can return either:
                # - Single output: (batch_size, output_dim) - backward compatible
                # - Tuple: (output, entropy) - when return_entropy=True
                if isinstance(outputs_raw, tuple):
                    # Model returns (output, entropy), use only output for evaluation
                    outputs, _ = outputs_raw
                else:
                    # Backward compatible: only output
                    outputs = outputs_raw
            elif model_type == 'tabpfn':
                # TabPFN output is 2D: (batch_size, output_dim)
                outputs = outputs_raw
            else:
                raise ValueError(f'Unknown model type {model_type}')
            
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            loss = actual_loss_fn(outputs, labels)


            all_predictions.append(outputs)
            all_labels.append(labels)
            
            total_loss += loss.item()
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
