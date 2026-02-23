import torch
import torch.nn as nn
from typing import List, Tuple


class TabularLoss(nn.Module):
    def __init__(self, w_bin=None, w_cat=None, weight_denominator=25, cat_sizes: List[int] = None):
        """
        Args:
            w_bin: Weight for binary loss (if None, uses 1/weight_denominator)
            w_cat: Weight for categorical loss (if None, uses 1/weight_denominator)
            weight_denominator: Denominator for default weights (default: 25, so w_bin=w_cat=1/25)
            cat_sizes: List of number of categories for each categorical variable
                      (e.g., [2, 3, 4] means first cat var has 2 classes, second has 3, etc.)
        """
        super(TabularLoss, self).__init__()
        
        if w_bin is not None:
            self.w_bin = w_bin
        else:
            self.w_bin = 1.0 / weight_denominator
        
        if w_cat is not None:
            self.w_cat = w_cat
        else:
            self.w_cat = 1.0 / weight_denominator
        
        self.cat_sizes = cat_sizes if cat_sizes is not None else []
        
        self.loss_cont = nn.MSELoss()
        self.loss_bin  = nn.BCEWithLogitsLoss()  # Handles Sigmoid internally, expects logits
        self.loss_cat  = nn.CrossEntropyLoss()  # Handles Softmax internally, expects logits

    def forward(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        predictions: tuple (rec_cont, rec_bin, rec_cat_logits)
        targets: tuple (target_cont, target_bin, target_cat_onehot)
                 - target_cont: continuous values
                 - target_bin: binary values (0/1)
                 - target_cat_onehot: one-hot encoded categorical (will be converted to indices)
        """
        rec_cont, rec_bin, rec_cat_logits = predictions
        tgt_cont, tgt_bin, tgt_cat_onehot = targets
        
        if rec_cont.numel() > 0:
            ref_tensor = rec_cont
        elif rec_bin.numel() > 0:
            ref_tensor = rec_bin
        else:
            ref_tensor = rec_cat_logits
        
        def zero_loss_like(tensor):
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)
        
        # 1. Continuous Loss (MSE) 
        if rec_cont.numel() > 0:
            l_cont = self.loss_cont(rec_cont, tgt_cont)
        else:
            l_cont = zero_loss_like(ref_tensor)
        
        # 2. Binary Loss (BCEWithLogits)
        if rec_bin.numel() > 0:
            l_bin = self.loss_bin(rec_bin, tgt_bin)
        else:
            l_bin = zero_loss_like(ref_tensor)
        
        # 3. Categorical Loss (CrossEntropy)
        # If group_categorical=False, cat_sizes is empty, so skip categorical loss entirely
        if len(self.cat_sizes) > 0 and rec_cat_logits.numel() > 0:
            cat_losses = []
            start_idx = 0
            cat_start_idx = 0
            
            for cat_size in self.cat_sizes:
                # Extract logits for this categorical variable 
                cat_logits = rec_cat_logits[:, start_idx:start_idx + cat_size]
                # Extract one-hot for this categorical variable
                cat_onehot = tgt_cat_onehot[:, cat_start_idx:cat_start_idx + cat_size]
                # Convert one-hot to class indices
                cat_indices = torch.argmax(cat_onehot, dim=1).long()
                
                # CrossEntropyLoss expects logits and applies softmax internally
                cat_losses.append(self.loss_cat(cat_logits, cat_indices))
                start_idx += cat_size
                cat_start_idx += cat_size
            
            l_cat = torch.stack(cat_losses).mean() if cat_losses else zero_loss_like(ref_tensor)
        else:
            l_cat = zero_loss_like(ref_tensor)
        
        total_loss = l_cont + self.w_bin * l_bin + self.w_cat * l_cat
        return total_loss, (l_cont.item(), l_bin.item(), l_cat.item())