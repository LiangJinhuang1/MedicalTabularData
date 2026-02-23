import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def create_experiment_dir(
    output_root: Path = None,
    target_col: str = None,
    seed: int = None,
    subset_tag: str = None,
) -> Path:
    """
    Create experiment directory with timestamp, target column, and seed.
    """
    if output_root is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        output_root = project_root / "output"
    else:
        output_root = Path(output_root)
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create directory name with timestamp, target_col, and seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_parts = [timestamp]
    
    if target_col:
        safe_target_col = str(target_col).replace('/', '_').replace('\\', '_').replace(' ', '_')
        dir_parts.append(safe_target_col)
    
    if seed is not None:
        dir_parts.append(f"seed{seed}")

    if subset_tag:
        safe_subset_tag = str(subset_tag).replace('/', '_').replace('\\', '_').replace(' ', '_')
        dir_parts.append(safe_subset_tag)
    
    experiment_dir_name = "_".join(dir_parts)
    experiment_dir = output_root / experiment_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    
    return experiment_dir


def save_configs(experiment_dir: Path, config_dict: Dict[str, Any]):
    """
    Save configuration dictionary to YAML file.
    """
    configs_dir = experiment_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    config_file = configs_dir / "full_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    


def save_checkpoint(
    experiment_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float,
    model_name: str,
    is_best: bool = False
):
    """
    Save model checkpoint (only best models).
    """
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_name': model_name
    }
    
    # Save best model only
    if is_best:
        best_checkpoint_path = checkpoints_dir / f"{model_name}_best.pt"
        torch.save(checkpoint, best_checkpoint_path)


class LossTracker:
    
    def __init__(self, metric_name: str = "R2"):
        self.metric_name = metric_name
        self.train_losses: Dict[str, List[float]] = {}
        self.val_losses: Dict[str, List[float]] = {}
        self.val_r2_scores: Dict[str, List[float]] = {}
        self.val_auc_scores: Dict[str, List[float]] = {}
        self.best_val_loss: Dict[str, float] = {}
        self.best_val_r2: Dict[str, float] = {}
        self.best_val_auc: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}  # Best epoch based on loss
        self.best_r2_epoch: Dict[str, int] = {}  # Best epoch based on metric (R2/F1)
        self.best_auc_epoch: Dict[str, int] = {}  # Best epoch based on AUC
    
    def update(self, epoch: int, model_name: str, train_loss: float, val_loss: float, val_r2: float = None, val_auc: float = None):
        """
        Update losses for a specific model.
        
        Args:
            epoch: Current epoch number
            model_name: Name of the model (e.g., 'mlp', 'tabm', 'custom_model')
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_r2: Validation R2 score for this epoch (optional)
            val_auc: Validation AUC score for this epoch (optional)
        """
        # Initialize lists for new models
        if model_name not in self.train_losses:
            self.train_losses[model_name] = []
            self.val_losses[model_name] = []
            self.val_r2_scores[model_name] = []
            self.val_auc_scores[model_name] = []
            self.best_val_loss[model_name] = float('inf')
            self.best_val_r2[model_name] = float('-inf')
            self.best_val_auc[model_name] = float('-inf')
            self.best_epoch[model_name] = -1
            self.best_r2_epoch[model_name] = -1
            self.best_auc_epoch[model_name] = -1
        
        # Append losses (ensure float type)
        train_loss_val = float(train_loss)
        val_loss_val = float(val_loss)
            
        self.train_losses[model_name].append(train_loss_val)
        self.val_losses[model_name].append(val_loss_val)
        
        # Append R2 score if provided
        if val_r2 is not None:
            val_r2 = float(val_r2)
            self.val_r2_scores[model_name].append(val_r2)
        else:
            self.val_r2_scores[model_name].append(float('nan'))

        # Append AUC score if provided
        if val_auc is not None:
            val_auc = float(val_auc)
            self.val_auc_scores[model_name].append(val_auc)
        else:
            self.val_auc_scores[model_name].append(float('nan'))
        
        # Update best model
        best_val = self.best_val_loss[model_name]
        if not isinstance(best_val, (int, float)):
            current_best = float('inf')
            self.best_val_loss[model_name] = current_best
        else:
            current_best = float(best_val)
            
        if val_loss_val < current_best:
            self.best_val_loss[model_name] = val_loss_val
            self.best_epoch[model_name] = epoch
        
        # Update best R2
        if val_r2 is not None and not np.isnan(val_r2):
            current_best_r2 = self.best_val_r2[model_name]
            if not isinstance(current_best_r2, (int, float)) or np.isnan(current_best_r2):
                current_best_r2 = float('-inf')
            else:
                current_best_r2 = float(current_best_r2)
            
            if val_r2 > current_best_r2:
                self.best_val_r2[model_name] = val_r2
                self.best_r2_epoch[model_name] = epoch

        # Update best AUC
        if val_auc is not None and not np.isnan(val_auc):
            current_best_auc = self.best_val_auc[model_name]
            if not isinstance(current_best_auc, (int, float)) or np.isnan(current_best_auc):
                current_best_auc = float('-inf')
            else:
                current_best_auc = float(current_best_auc)
            
            if val_auc > current_best_auc:
                self.best_val_auc[model_name] = val_auc
                self.best_auc_epoch[model_name] = epoch
    
    def _group_by_variant_type(self, model_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Group models by variant type, then by base model (mlp/tabm/tabnet).
        
        Returns:
            Dictionary mapping variant_type -> {'mlp': [models], 'tabm': [models], 'tabnet': [models], 'other': [models]}
        """
        variant_groups = {}
        
        for model_name in model_names:
            # Determine variant type and base model
            variant_type = None
            base_model = None
            
            # Check for encoder models (they are their own category)
            # Handle wae_encoder with regularization_type suffix (e.g., 'wae_encoder_sinkhorn')
            if model_name == 'encoder':
                variant_type = 'encoder'
                base_model = 'encoder'
            elif model_name == 'vae_encoder':
                variant_type = 'vae_encoder'
                base_model = 'vae_encoder'
            elif model_name.startswith('wae_encoder'):
                variant_type = 'wae_encoder'
                base_model = 'wae_encoder'
            elif model_name.startswith('gw_encoder'):
                variant_type = 'gw_encoder'
                base_model = 'gw_encoder'
            # Check for simple models (mlp, tabm, tabnet)
            elif model_name == 'mlp':
                variant_type = 'simple'
                base_model = 'mlp'
            elif model_name == 'tabm':
                variant_type = 'simple'
                base_model = 'tabm'
            elif model_name == 'tabnet':
                variant_type = 'simple'
                base_model = 'tabnet'
            # Check for pretraining models without prefix (tabae, tabvae, tabwae)
            elif model_name == 'tabae':
                variant_type = 'pretraining_ae'
                base_model = 'other'
            elif model_name == 'tabvae':
                variant_type = 'pretraining_vae'
                base_model = 'other'
            elif model_name == 'tabwae':
                variant_type = 'pretraining_wae'
                base_model = 'other'
            # Check for MLP variants
            elif model_name.startswith('mlp_'):
                base_model = 'mlp'
                # Extract variant type from model name
                variant_part = model_name[4:]  # Remove "mlp_"
                # Check for pretraining models (TabAE, TabVAE, TabWAE)
                if variant_part == 'tabae':
                    variant_type = 'pretraining_ae'
                elif variant_part == 'tabvae':
                    variant_type = 'pretraining_vae'
                elif variant_part == 'tabwae':
                    variant_type = 'pretraining_wae'
                # Check for embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('embedding') and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'embedding'
                # Check for VAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('vae_embedding'):
                    variant_type = 'vae_embedding'
                # Check for WAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('wae_embedding'):
                    variant_type = 'wae_embedding'
                # Check for GW models (simple name: mlp_gw, no frozen/finetuned versions)
                elif variant_part == 'gw':
                    variant_type = 'gw'
                # Check for multi-task embedding (without vae/wae/gw)
                elif 'multi_task_embedding' in variant_part and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'multi_task_embedding'
                # Check for multi-VAE task embedding
                elif 'multi_vae_task_embedding' in variant_part:
                    variant_type = 'multi_vae_task_embedding'
                # Check for multi-WAE task embedding (e.g., 'multi_wae_task_embedding_sinkhorn')
                elif 'multi_wae_task_embedding' in variant_part:
                    variant_type = 'multi_wae_task_embedding'
                # Note: GW models don't have multi-task versions
                else:
                    variant_type = 'other'
            # Check for TABM variants
            elif model_name.startswith('tabm_'):
                base_model = 'tabm'
                # Extract variant type from model name
                variant_part = model_name[5:]  # Remove "tabm_"
                # Check for pretraining models (TabAE, TabVAE, TabWAE)
                if variant_part == 'tabae':
                    variant_type = 'pretraining_ae'
                elif variant_part == 'tabvae':
                    variant_type = 'pretraining_vae'
                elif variant_part == 'tabwae':
                    variant_type = 'pretraining_wae'
                # Check for embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('embedding') and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'embedding'
                # Check for VAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('vae_embedding'):
                    variant_type = 'vae_embedding'
                # Check for WAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('wae_embedding'):
                    variant_type = 'wae_embedding'
                # Check for GW models (simple name: tabm_gw, no frozen/finetuned versions)
                elif variant_part == 'gw':
                    variant_type = 'gw'
                # Check for multi-task embedding (without vae/wae/gw)
                elif 'multi_task_embedding' in variant_part and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'multi_task_embedding'
                # Check for multi-VAE task embedding
                elif 'multi_vae_task_embedding' in variant_part:
                    variant_type = 'multi_vae_task_embedding'
                # Check for multi-WAE task embedding (e.g., 'multi_wae_task_embedding_sinkhorn')
                elif 'multi_wae_task_embedding' in variant_part:
                    variant_type = 'multi_wae_task_embedding'
                # Note: GW models don't have multi-task versions
                else:
                    variant_type = 'other'
            # Check for TabNet variants
            elif model_name.startswith('tabnet_'):
                base_model = 'tabnet'
                # Extract variant type from model name
                variant_part = model_name[7:]  # Remove "tabnet_"
                # Check for pretraining models (TabAE, TabVAE, TabWAE)
                if variant_part == 'tabae':
                    variant_type = 'pretraining_ae'
                elif variant_part == 'tabvae':
                    variant_type = 'pretraining_vae'
                elif variant_part == 'tabwae':
                    variant_type = 'pretraining_wae'
                # Check for embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('embedding') and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'embedding'
                # Check for VAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('vae_embedding'):
                    variant_type = 'vae_embedding'
                # Check for WAE embedding variants (with frozen/finetuned suffix)
                elif variant_part.startswith('wae_embedding'):
                    variant_type = 'wae_embedding'
                # Check for GW models (simple name: tabnet_gw, no frozen/finetuned versions)
                elif variant_part == 'gw':
                    variant_type = 'gw'
                # Check for multi-task embedding (without vae/wae/gw)
                elif 'multi_task_embedding' in variant_part and 'vae' not in variant_part and 'wae' not in variant_part and 'gw' not in variant_part:
                    variant_type = 'multi_task_embedding'
                # Check for multi-VAE task embedding
                elif 'multi_vae_task_embedding' in variant_part:
                    variant_type = 'multi_vae_task_embedding'
                # Check for multi-WAE task embedding (e.g., 'multi_wae_task_embedding_sinkhorn')
                elif 'multi_wae_task_embedding' in variant_part:
                    variant_type = 'multi_wae_task_embedding'
                # Note: GW models don't have multi-task versions
                else:
                    variant_type = 'other'
            else:
                variant_type = 'other'
                base_model = 'other'
            
            # Initialize variant group if needed
            if variant_type not in variant_groups:
                variant_groups[variant_type] = {'mlp': [], 'tabm': [], 'tabnet': [], 'other': []}
            
            # Add to appropriate group
            if base_model in ['mlp', 'tabm', 'tabnet']:
                variant_groups[variant_type][base_model].append(model_name)
            else:
                variant_groups[variant_type]['other'].append(model_name)
        
        # Sort models within each group
        for variant_type in variant_groups:
            for base_model in variant_groups[variant_type]:
                variant_groups[variant_type][base_model] = sorted(variant_groups[variant_type][base_model])
        
        return variant_groups
    
    def save_plots(self, experiment_dir: Path, figsize: tuple = (10, 6)):
        """
        Save loss plots grouped by variant type. Each variant type gets its own plot file.
        MLP, TabM, and TabNet variants are shown side-by-side (3 columns).
        """
        if not self.train_losses:
            print("No loss data to plot.")
            return
        
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        model_names = sorted(self.train_losses.keys())
        variant_groups = self._group_by_variant_type(model_names)
        
        def format_display_name(model_name, variant_type):
            if variant_type == 'simple':
                return model_name.upper()
            if model_name.startswith('mlp_'):
                variant_part = model_name[4:]
                if variant_part.endswith('_frozen'):
                    base_variant = variant_part[:-7]
                    return base_variant.replace('_', ' ').title() + ' (Frozen)'
                if variant_part.endswith('_finetuned'):
                    base_variant = variant_part[:-10]
                    return base_variant.replace('_', ' ').title() + ' (Fine-tuned)'
                return variant_part.replace('_', ' ').title()
            if model_name.startswith('tabm_'):
                variant_part = model_name[5:]
                if variant_part.endswith('_frozen'):
                    base_variant = variant_part[:-7]
                    return base_variant.replace('_', ' ').title() + ' (Frozen)'
                if variant_part.endswith('_finetuned'):
                    base_variant = variant_part[:-10]
                    return base_variant.replace('_', ' ').title() + ' (Fine-tuned)'
                return variant_part.replace('_', ' ').title()
            if model_name.startswith('tabnet_'):
                variant_part = model_name[7:]
                if variant_part.endswith('_frozen'):
                    base_variant = variant_part[:-7]
                    return base_variant.replace('_', ' ').title() + ' (Frozen)'
                if variant_part.endswith('_finetuned'):
                    base_variant = variant_part[:-10]
                    return base_variant.replace('_', ' ').title() + ' (Fine-tuned)'
                return variant_part.replace('_', ' ').title()
            if model_name.endswith('_frozen'):
                base_name = model_name[:-7]
                return base_name.replace('_', ' ').title() + ' (Frozen)'
            if model_name.endswith('_finetuned'):
                base_name = model_name[:-10]
                return base_name.replace('_', ' ').title() + ' (Fine-tuned)'
            return model_name.replace('_', ' ').title()

        # Helper function to plot model variants on an axis
        def plot_model_variants(ax, model_variants, variant_type, base_model=None):
            """Plot all variants of a model type on the given axis."""
            # Remove duplicates
            model_variants = sorted(list(set(model_variants)))
            
            # Define colors for train and validation losses
            # Train colors (standard matplotlib colors)
            train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            # Validation colors (same color scale but shifted towards red/orange)
            # Each validation color is in the same position but shifted to red-orange spectrum
            # Blue -> Coral, Orange -> Red-Orange, Green -> Orange, Red -> Tomato, Purple -> Salmon, Brown -> Coral, Pink -> Coral, Gray -> Orange
            val_colors = ['#ff6b35', '#ff4500', '#ff8c00', '#ff6347', '#ff7f50', '#cd5c5c', '#ff6347', '#ff8c69']
            line_styles = ['-', '--', '-.', ':']
            
            # Plot each model variant (one train + one val line per variant)
            for idx, model_name in enumerate(model_variants):
                # Ensure model exists in our data
                if model_name not in self.train_losses or model_name not in self.val_losses:
                    continue
                
                train_losses_list = self.train_losses[model_name]
                val_losses_list = self.val_losses[model_name]
                
                # Skip if no data
                if not train_losses_list or not val_losses_list:
                    continue
                
                epochs = range(1, len(train_losses_list) + 1)
                
                # Assign unique colors for train and validation
                train_color = train_colors[idx % len(train_colors)]
                val_color = val_colors[idx % len(val_colors)]
                line_style = line_styles[(idx // len(train_colors)) % len(line_styles)]
                
                display_name = format_display_name(model_name, variant_type)
                
                # Plot train and validation losses with different colors
                ax.plot(epochs, train_losses_list, color=train_color, linestyle=line_style, 
                       linewidth=2, label=f'{display_name} Train', alpha=0.8)
                ax.plot(epochs, val_losses_list, color=val_color, linestyle=line_style, 
                       linewidth=2.5, label=f'{display_name} Val', alpha=1.0)
                
                # Mark best epoch for validation loss (use validation color)
                if self.best_epoch[model_name] >= 0 and self.best_epoch[model_name] < len(val_losses_list):
                    best_epoch_num = self.best_epoch[model_name] + 1
                    best_val_loss = val_losses_list[self.best_epoch[model_name]]
                    ax.plot(best_epoch_num, best_val_loss, 'o', color=val_color, 
                           markersize=8, markeredgecolor='black', markeredgewidth=1,
                           label=f'{display_name} Best (Epoch {best_epoch_num})')
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            
            # Set title based on variant type and base model
            variant_titles = {
                'simple': 'Simple Models',
                'pretraining_ae': 'Pre-training AE Models',
                'pretraining_vae': 'Pre-training VAE Models',
                'pretraining_wae': 'Pre-training WAE Models',
                'embedding': 'Embedding Models',
                'vae_embedding': 'VAE Embedding Models',
                'wae_embedding': 'WAE Embedding Models',
                'gw': 'GW Models',
                'multi_task_embedding': 'Multi-Task Embedding Models',
                'multi_vae_task_embedding': 'Multi-VAE Task Embedding Models',
                'multi_wae_task_embedding': 'Multi-WAE Task Embedding Models',
                'encoder': 'Encoder Models',
                'vae_encoder': 'VAE Encoder Models',
                'wae_encoder': 'WAE Encoder Models',
                'gw_encoder': 'GW Encoder Models',
                'other': 'Other Models'
            }
            title = variant_titles.get(variant_type, variant_type.replace('_', ' ').title())
            
            # Add base model identifier to title if provided
            if base_model:
                title = f'{base_model.upper()} - {title}'
            
            ax.set_title(f'{title} - Loss Curves', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

        def plot_auc_variants(ax, model_variants, variant_type, base_model=None):
            model_variants = sorted(list(set(model_variants)))
            auc_colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            line_styles = ['-', '--', '-.', ':']

            for idx, model_name in enumerate(model_variants):
                auc_list = self.val_auc_scores.get(model_name, [])
                if not auc_list:
                    continue
                if not any(not np.isnan(v) for v in auc_list):
                    continue

                epochs = range(1, len(auc_list) + 1)
                color = auc_colors[idx % len(auc_colors)]
                line_style = line_styles[(idx // len(auc_colors)) % len(line_styles)]
                display_name = format_display_name(model_name, variant_type)

                ax.plot(epochs, auc_list, color=color, linestyle=line_style,
                        linewidth=2, label=f'{display_name} AUC', alpha=0.9)

                best_epoch_idx = self.best_auc_epoch.get(model_name, -1)
                if best_epoch_idx >= 0 and best_epoch_idx < len(auc_list):
                    best_epoch_num = best_epoch_idx + 1
                    best_auc = auc_list[best_epoch_idx]
                    if not np.isnan(best_auc):
                        ax.plot(best_epoch_num, best_auc, 'o', color=color,
                                markersize=8, markeredgecolor='black', markeredgewidth=1,
                                label=f'{display_name} Best (Epoch {best_epoch_num})')

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('AUC', fontsize=12)

            variant_titles = {
                'simple': 'Simple Models',
                'pretraining_ae': 'Pre-training AE Models',
                'pretraining_vae': 'Pre-training VAE Models',
                'pretraining_wae': 'Pre-training WAE Models',
                'embedding': 'Embedding Models',
                'vae_embedding': 'VAE Embedding Models',
                'wae_embedding': 'WAE Embedding Models',
                'gw': 'GW Models',
                'multi_task_embedding': 'Multi-Task Embedding Models',
                'multi_vae_task_embedding': 'Multi-VAE Task Embedding Models',
                'multi_wae_task_embedding': 'Multi-WAE Task Embedding Models',
                'encoder': 'Encoder Models',
                'vae_encoder': 'VAE Encoder Models',
                'wae_encoder': 'WAE Encoder Models',
                'gw_encoder': 'GW Encoder Models',
                'other': 'Other Models'
            }
            title = variant_titles.get(variant_type, variant_type.replace('_', ' ').title())
            if base_model:
                title = f'{base_model.upper()} - {title}'
            ax.set_title(f'{title} - AUC Curves', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

        def has_valid_auc(model_variants):
            for model_name in model_variants:
                auc_list = self.val_auc_scores.get(model_name, [])
                if auc_list and any(not np.isnan(v) for v in auc_list):
                    return True
            return False
        
        # Create plots
        saved_plots = []
        
        # Define variant type order for plotting
        variant_order = ['simple', 'pretraining_ae', 'pretraining_vae', 'pretraining_wae',
                        'embedding', 'vae_embedding', 'wae_embedding', 'gw',
                        'multi_task_embedding', 'multi_vae_task_embedding', 'multi_wae_task_embedding',
                        'encoder', 'vae_encoder', 'wae_encoder', 'gw_encoder', 'other']
        
        # Plot each variant type
        for variant_type in variant_order:
            if variant_type not in variant_groups:
                continue
            
            mlp_models = variant_groups[variant_type]['mlp']
            tabm_models = variant_groups[variant_type]['tabm']
            tabnet_models = variant_groups[variant_type]['tabnet']
            other_models = variant_groups[variant_type]['other']
            
            # For MLP/TabM/TabNet variants: create side-by-side plot (3 columns)
            if mlp_models or tabm_models or tabnet_models:
                # Always create 3 columns for consistent layout
                num_cols = 3
                
                # Create figure with subplots (1 row, 3 columns)
                fig, axes = plt.subplots(1, num_cols, figsize=(figsize[0] * num_cols, figsize[1]))
                
                # Plot MLP variants
                if mlp_models:
                    plot_model_variants(axes[0], mlp_models, variant_type, base_model='mlp')
                else:
                    axes[0].axis('off')
                    axes[0].text(0.5, 0.5, 'No MLP models', ha='center', va='center', 
                               transform=axes[0].transAxes, fontsize=14)
                
                # Plot TabM variants
                if tabm_models:
                    plot_model_variants(axes[1], tabm_models, variant_type, base_model='tabm')
                else:
                    axes[1].axis('off')
                    axes[1].text(0.5, 0.5, 'No TabM models', ha='center', va='center', 
                               transform=axes[1].transAxes, fontsize=14)
                
                # Plot TabNet variants
                if tabnet_models:
                    plot_model_variants(axes[2], tabnet_models, variant_type, base_model='tabnet')
                else:
                    axes[2].axis('off')
                    axes[2].text(0.5, 0.5, 'No TabNet models', ha='center', va='center', 
                               transform=axes[2].transAxes, fontsize=14)
                
                plt.tight_layout()
                
                # Save plot with variant type name
                plot_filename = f"loss_curves_{variant_type}.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                saved_plots.append(plot_path)
                print(f"Saved plot for {variant_type} models: {plot_path}")
            
            # Plot other models (encoders, pretrained models without prefix, etc.) separately if they exist
            if other_models:
                fig, ax = plt.subplots(figsize=figsize)
                plot_model_variants(ax, other_models, variant_type)
                
                plt.tight_layout()
                
                # Save plot with variant type name
                # For pretraining models, use a more descriptive filename
                if variant_type in ['pretraining_ae', 'pretraining_vae', 'pretraining_wae']:
                    plot_filename = f"loss_curves_{variant_type}_pretrained.png"
                else:
                    plot_filename = f"loss_curves_{variant_type}_other.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                saved_plots.append(plot_path)
                print(f"Saved plot for {variant_type} other models: {plot_path}")

            # AUC plots (classification runs only)
            if has_valid_auc(mlp_models) or has_valid_auc(tabm_models) or has_valid_auc(tabnet_models):
                num_cols = 3
                fig, axes = plt.subplots(1, num_cols, figsize=(figsize[0] * num_cols, figsize[1]))

                if has_valid_auc(mlp_models):
                    plot_auc_variants(axes[0], mlp_models, variant_type, base_model='mlp')
                else:
                    axes[0].axis('off')
                    axes[0].text(0.5, 0.5, 'No MLP AUC data', ha='center', va='center',
                                 transform=axes[0].transAxes, fontsize=14)

                if has_valid_auc(tabm_models):
                    plot_auc_variants(axes[1], tabm_models, variant_type, base_model='tabm')
                else:
                    axes[1].axis('off')
                    axes[1].text(0.5, 0.5, 'No TabM AUC data', ha='center', va='center',
                                 transform=axes[1].transAxes, fontsize=14)

                if has_valid_auc(tabnet_models):
                    plot_auc_variants(axes[2], tabnet_models, variant_type, base_model='tabnet')
                else:
                    axes[2].axis('off')
                    axes[2].text(0.5, 0.5, 'No TabNet AUC data', ha='center', va='center',
                                 transform=axes[2].transAxes, fontsize=14)

                plt.tight_layout()

                plot_filename = f"auc_curves_{variant_type}.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                saved_plots.append(plot_path)
                print(f"Saved AUC plot for {variant_type} models: {plot_path}")

            if has_valid_auc(other_models):
                fig, ax = plt.subplots(figsize=figsize)
                plot_auc_variants(ax, other_models, variant_type)

                plt.tight_layout()

                if variant_type in ['pretraining_ae', 'pretraining_vae', 'pretraining_wae']:
                    plot_filename = f"auc_curves_{variant_type}_pretrained.png"
                else:
                    plot_filename = f"auc_curves_{variant_type}_other.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                saved_plots.append(plot_path)
                print(f"Saved AUC plot for {variant_type} other models: {plot_path}")
        
        print(f"\nTotal {len(saved_plots)} loss plot(s) have been saved.")
    
    def save_losses_to_file(self, experiment_dir: Path):
        """
        Save loss records to text file.
        
        Args:
            experiment_dir: Experiment directory path
        """
        if not self.train_losses:
            print("No loss data to save.")
            return
        
        losses_file = experiment_dir / "losses.txt"
        metric_label = self.metric_name
        model_names = sorted(self.train_losses.keys())
        
        with open(losses_file, 'w', encoding='utf-8') as f:
            header_parts = ["Epoch"]
            for model_name in model_names:
                header_parts.append(f"{model_name.upper()} Train")
                header_parts.append(f"{model_name.upper()} Val")
                header_parts.append(f"{model_name.upper()} {metric_label}")
                header_parts.append(f"{model_name.upper()} AUC")
            f.write(" | ".join(header_parts) + "\n")
            f.write("-" * (len(" | ".join(header_parts)) + 10) + "\n")
            
            # Find the maximum number of epochs across all models
            max_epochs = max(len(self.train_losses[model_name]) for model_name in model_names)
            for epoch in range(max_epochs):
                row_parts = [f"{epoch + 1:5d}"]
                for model_name in model_names:
                    if epoch < len(self.train_losses[model_name]):
                        train_loss = self.train_losses[model_name][epoch]
                        val_loss = self.val_losses[model_name][epoch]
                        val_r2 = self.val_r2_scores[model_name][epoch] if epoch < len(self.val_r2_scores[model_name]) else float('nan')
                        val_auc = self.val_auc_scores[model_name][epoch] if epoch < len(self.val_auc_scores[model_name]) else float('nan')
                    else:
                        train_loss = float('nan')
                        val_loss = float('nan')
                        val_r2 = float('nan')
                        val_auc = float('nan')
                    row_parts.append(f"{train_loss:13.4f}")
                    row_parts.append(f"{val_loss:11.4f}")
                    row_parts.append(f"{val_r2:11.4f}")
                    row_parts.append(f"{val_auc:11.4f}")
                f.write(" | ".join(row_parts) + "\n")
            
            # Summary section - Table format
            f.write("\n" + "=" * 100 + "\n")
            f.write("BEST MODEL RESULTS SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            # Table 1: Best models by validation loss
            f.write("Best Models by Validation Loss:\n")
            f.write("-" * 100 + "\n")
            # Header
            header_cols = ["Model", "Best Epoch", "Val Loss", f"Val {metric_label}", "Val AUC"]
            col_widths = [max(len(header_cols[0]), max(len(name) for name in model_names)) + 2,
                          len(header_cols[1]) + 2,
                          len(header_cols[2]) + 2,
                          len(header_cols[3]) + 2,
                          len(header_cols[4]) + 2]
            f.write(f"{header_cols[0]:<{col_widths[0]}} | {header_cols[1]:>{col_widths[1]}} | {header_cols[2]:>{col_widths[2]}} | {header_cols[3]:>{col_widths[3]}} | {header_cols[4]:>{col_widths[4]}}\n")
            f.write("-" * 100 + "\n")
            
            for model_name in model_names:
                best_epoch = int(self.best_epoch[model_name])
                best_loss = float(self.best_val_loss[model_name])
                best_r2 = self.best_val_r2.get(model_name, float('nan'))
                best_auc_at_loss = float('nan')
                if best_epoch >= 0 and best_epoch < len(self.val_auc_scores.get(model_name, [])):
                    best_auc_at_loss = self.val_auc_scores[model_name][best_epoch]
                
                if not np.isnan(best_r2) and best_r2 != float('-inf'):
                    f.write(f"{model_name:<{col_widths[0]}} | {best_epoch + 1:>{col_widths[1]}} | {best_loss:>{col_widths[2]}.4f} | {best_r2:>{col_widths[3]}.4f} | {best_auc_at_loss:>{col_widths[4]}.4f}\n")
                else:
                    auc_display = f"{best_auc_at_loss:.4f}" if not np.isnan(best_auc_at_loss) else "N/A"
                    f.write(f"{model_name:<{col_widths[0]}} | {best_epoch + 1:>{col_widths[1]}} | {best_loss:>{col_widths[2]}.4f} | {'N/A':>{col_widths[3]}} | {auc_display:>{col_widths[4]}}\n")
            
            # Table 2: Best models by metric score
            f.write("\n" + "-" * 100 + "\n")
            f.write(f"Best Models by {metric_label} Score:\n")
            f.write("-" * 100 + "\n")
            # Header
            header_cols_r2 = ["Model", "Best Epoch", "Val Loss", f"Val {metric_label}", "Val AUC"]
            f.write(f"{header_cols_r2[0]:<{col_widths[0]}} | {header_cols_r2[1]:>{col_widths[1]}} | {header_cols_r2[2]:>{col_widths[2]}} | {header_cols_r2[3]:>{col_widths[3]}} | {header_cols_r2[4]:>{col_widths[4]}}\n")
            f.write("-" * 100 + "\n")
            
            for model_name in model_names:
                best_r2_epoch = int(self.best_r2_epoch.get(model_name, -1))
                best_r2 = self.best_val_r2.get(model_name, float('nan'))
                best_auc_at_r2 = float('nan')
                if best_r2_epoch >= 0 and best_r2_epoch < len(self.val_auc_scores.get(model_name, [])):
                    best_auc_at_r2 = self.val_auc_scores[model_name][best_r2_epoch]
                
                if best_r2_epoch >= 0 and not np.isnan(best_r2) and best_r2 != float('-inf'):
                    best_r2_val = float(best_r2)
                    if best_r2_epoch < len(self.val_losses[model_name]):
                        best_r2_loss = float(self.val_losses[model_name][best_r2_epoch])
                    else:
                        best_r2_loss = float('nan')
                    
                    if not np.isnan(best_r2_loss):
                        f.write(f"{model_name:<{col_widths[0]}} | {best_r2_epoch + 1:>{col_widths[1]}} | {best_r2_loss:>{col_widths[2]}.4f} | {best_r2_val:>{col_widths[3]}.4f} | {best_auc_at_r2:>{col_widths[4]}.4f}\n")
                    else:
                        auc_display = f"{best_auc_at_r2:.4f}" if not np.isnan(best_auc_at_r2) else "N/A"
                        f.write(f"{model_name:<{col_widths[0]}} | {best_r2_epoch + 1:>{col_widths[1]}} | {'N/A':>{col_widths[2]}} | {best_r2_val:>{col_widths[3]}.4f} | {auc_display:>{col_widths[4]}}\n")
                else:
                    f.write(f"{model_name:<{col_widths[0]}} | {'N/A':>{col_widths[1]}} | {'N/A':>{col_widths[2]}} | {'N/A':>{col_widths[3]}} | {'N/A':>{col_widths[4]}}\n")
        
        print(f"\nThe loss records have been saved to: {losses_file}")
    
    def save_best_results_table(self, experiment_dir: Path):
        """
        Save best model results as a CSV table.
        
        Args:
            experiment_dir: Experiment directory path
        """
        if not self.train_losses:
            print("No loss data to save.")
            return
        
        model_names = sorted(self.train_losses.keys())
        
        import csv
        csv_file = experiment_dir / "best_results.csv"
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Header
            metric_label = self.metric_name
            writer.writerow([
                'Model',
                'Best Epoch (by Loss)',
                'Val Loss (Best)',
                f'Val {metric_label} (at Best Loss)',
                'Val AUC (at Best Loss)',
                f'Best Epoch (by {metric_label})',
                f'Val Loss (at Best {metric_label})',
                f'Val {metric_label} (Best)',
                f'Val AUC (at Best {metric_label})',
                'Best Epoch (by AUC)',
                'Val Loss (at Best AUC)',
                'Val AUC (Best)',
            ])
            
            for model_name in model_names:
                best_epoch = int(self.best_epoch[model_name])
                best_loss = float(self.best_val_loss[model_name])
                best_r2_at_loss = self.best_val_r2.get(model_name, float('nan'))
                best_r2_epoch = int(self.best_r2_epoch.get(model_name, -1))
                best_r2 = self.best_val_r2.get(model_name, float('nan'))
                best_auc_epoch = int(self.best_auc_epoch.get(model_name, -1))
                best_auc = self.best_val_auc.get(model_name, float('nan'))
                
                # Get loss at best R2 epoch
                if best_r2_epoch >= 0 and best_r2_epoch < len(self.val_losses[model_name]):
                    loss_at_best_r2 = float(self.val_losses[model_name][best_r2_epoch])
                else:
                    loss_at_best_r2 = float('nan')

                # AUC at best loss / best R2 epochs
                if best_epoch >= 0 and best_epoch < len(self.val_auc_scores.get(model_name, [])):
                    auc_at_best_loss = float(self.val_auc_scores[model_name][best_epoch])
                else:
                    auc_at_best_loss = float('nan')
                if best_r2_epoch >= 0 and best_r2_epoch < len(self.val_auc_scores.get(model_name, [])):
                    auc_at_best_r2 = float(self.val_auc_scores[model_name][best_r2_epoch])
                else:
                    auc_at_best_r2 = float('nan')

                # Loss at best AUC epoch
                if best_auc_epoch >= 0 and best_auc_epoch < len(self.val_losses.get(model_name, [])):
                    loss_at_best_auc = float(self.val_losses[model_name][best_auc_epoch])
                else:
                    loss_at_best_auc = float('nan')
                
                # Format values
                best_r2_at_loss_str = f"{best_r2_at_loss:.4f}" if not np.isnan(best_r2_at_loss) and best_r2_at_loss != float('-inf') else "N/A"
                best_r2_epoch_str = f"{best_r2_epoch + 1}" if best_r2_epoch >= 0 else "N/A"
                loss_at_best_r2_str = f"{loss_at_best_r2:.4f}" if not np.isnan(loss_at_best_r2) else "N/A"
                best_r2_str = f"{best_r2:.4f}" if not np.isnan(best_r2) and best_r2 != float('-inf') else "N/A"
                auc_at_best_loss_str = f"{auc_at_best_loss:.4f}" if not np.isnan(auc_at_best_loss) and auc_at_best_loss != float('-inf') else "N/A"
                auc_at_best_r2_str = f"{auc_at_best_r2:.4f}" if not np.isnan(auc_at_best_r2) and auc_at_best_r2 != float('-inf') else "N/A"
                best_auc_epoch_str = f"{best_auc_epoch + 1}" if best_auc_epoch >= 0 else "N/A"
                loss_at_best_auc_str = f"{loss_at_best_auc:.4f}" if not np.isnan(loss_at_best_auc) else "N/A"
                best_auc_str = f"{best_auc:.4f}" if not np.isnan(best_auc) and best_auc != float('-inf') else "N/A"
                
                writer.writerow([
                    model_name,
                    best_epoch + 1,
                    f"{best_loss:.4f}",
                    best_r2_at_loss_str,
                    auc_at_best_loss_str,
                    best_r2_epoch_str,
                    loss_at_best_r2_str,
                    best_r2_str,
                    auc_at_best_r2_str,
                    best_auc_epoch_str,
                    loss_at_best_auc_str,
                    best_auc_str,
                ])
        
        print(f"\nBest results table (CSV) has been saved to: {csv_file}")
