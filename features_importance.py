import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import shap
from torch.utils.data import Subset


from src.data.prepare_data import prepare_data_from_experiment
from src.data.dataloader import create_dataloader
from src.utils import model_factory
from src.utils.config import load_experiment_config


def _print_banner(title: str, width: int = 80, char: str = "="):
    line = char * width
    print(f"\n{line}")
    print(title)
    print(f"{line}\n")


def load_model_from_checkpoint(checkpoint_path, model_type, input_dim, config, device, full_dataset=None, 
                               train_features=None, train_labels=None, checkpoints_dir=None):
    """Load model from checkpoint."""
    return model_factory.load_model_from_checkpoint(
        checkpoint_path,
        model_type,
        input_dim,
        config,
        device,
        full_dataset=full_dataset,
        train_features=train_features,
        train_labels=train_labels,
        checkpoints_dir=checkpoints_dir,
        filter_mismatched=True,
    )


def compute_shap_importance(model, loader, device, model_type: str,
                            feature_names: list, n_samples: Optional[int] = None,
                            return_values: bool = False):
    model.eval()
    
    # 1. Collect Data
    all_data = []
    with torch.no_grad():
        for x, _ in loader:
            all_data.append(x.cpu().numpy())
    all_data = np.concatenate(all_data, axis=0)
    
    # 2. Optimize Background (Speed up KernelExplainer by 10x)
    # Instead of using 100 raw rows, we use 10 representative 'centroids'
    background_summary = shap.kmeans(all_data, 10) 
    
    # 3. Select explanation set
    n_explain = n_samples if n_samples else min(100, len(all_data))
    explain_data = all_data[:n_explain]

    # 4. Corrected Wrapper
    def model_wrapper(x_np):
        # Ensure model is in eval mode (important for VAE models)
        model.eval()
        # KernelExplainer sends numpy, PyTorch needs Tensors
        x_tensor = torch.from_numpy(x_np).float().to(device)
        with torch.no_grad():
            outputs = model(x_tensor)
            
            # Handle your specific Multi-task tuple returns
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Handle TabM base model (Average across the k_heads)
            # Note: TabM VAE/embedding models don't have 3D output, only TabM base model does
            if 'tabm' in model_type.lower() and not 'vae' in model_type.lower() and not 'embedding' in model_type.lower():
                if isinstance(outputs, torch.Tensor) and outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
            
            # Convert to numpy and ensure correct shape for SHAP
            # SHAP expects 1D array: (batch_size,)
            outputs_np = outputs.cpu().numpy()
            
            # Handle different output shapes
            if outputs_np.ndim == 0:
                # Scalar output (shouldn't happen for batch input, but handle it)
                outputs_np = outputs_np.reshape(1)
            elif outputs_np.ndim == 1:
                # Already 1D: (batch_size,)
                pass
            elif outputs_np.ndim == 2:
                # 2D: (batch_size, 1) or (batch_size, n_outputs)
                if outputs_np.shape[1] == 1:
                    # Use squeeze(axis=1) to only remove the second dimension, not batch dimension
                    outputs_np = outputs_np.squeeze(axis=1)  # (batch_size, 1) -> (batch_size,)
                else:
                    # Multiple outputs, take first one (regression output)
                    outputs_np = outputs_np[:, 0]
            else:
                # Higher dimensions, flatten or take first
                outputs_np = outputs_np.reshape(outputs_np.shape[0], -1)
                if outputs_np.shape[1] > 1:
                    outputs_np = outputs_np[:, 0]  # Take first output
                else:
                    outputs_np = outputs_np.squeeze(axis=1)
                
            return outputs_np

    # 5. Initialize Explainer
    explainer = shap.KernelExplainer(model_wrapper, background_summary)
    
    # 6. Compute (nsamples='auto' ensures mathematical convergence)
    shap_values = explainer.shap_values(explain_data, nsamples='auto')

    # Handle the SHAP list output (standard for KernelExplainer)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Convert to numpy array and compute mean importance across samples
    # shap_values shape: (n_samples, n_features)
    importance = np.abs(shap_values).mean(axis=0)
    
    if return_values:
        return importance, shap_values, explain_data
    return importance


def plot_shap_beeswarm(shap_values: np.ndarray,
                       features: np.ndarray,
                       feature_names: List[str],
                       save_path: Path,
                       max_display: int = 20):
    """Plot SHAP beeswarm (summary) plot."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_scores: np.ndarray, 
                            feature_names: List[str],
                            save_path: Path,
                            top_n: int = 20):
    """Plot SHAP feature importance."""
    fig, ax = plt.subplots(figsize=(8, 10))

    sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
    sorted_importance = importance_scores[sorted_indices]
    sorted_importance = np.array(sorted_importance).flatten().tolist()
    sorted_names = [feature_names[int(i)] for i in sorted_indices]

    bars = ax.barh(range(len(sorted_names)), sorted_importance, color='steelblue')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'SHAP - Top {top_n} Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def find_available_models(checkpoints_dir):
    """Find all available model checkpoints"""
    available_models = []
    checkpoint_files = list(checkpoints_dir.glob('*_best.pt'))
    
    # Pre-training models to exclude (TabAE, TabVAE, TabWAE)
    # These end with _tabae, _tabvae, or _tabwae
    # Note: We use endswith() to ensure frozen/finetuned models (which end with _frozen/_finetuned)
    # are NOT excluded, even though they may contain 'vae' or 'wae' in their names
    pretraining_suffixes = ['_tabae', '_tabvae', '_tabwae']
    
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace('_best', '')
        # Skip encoder checkpoints, pre-training models, and TabPFN models
        if model_name not in ['encoder', 'vae_encoder']:
            # Skip TabPFN models (not supported)
            if 'tabpfn' in model_name.lower():
                continue
            # Skip pre-training models (those ending with _tabae, _tabvae, or _tabwae)
            # This will NOT exclude frozen/finetuned models like:
            # - mlp_embedding_frozen (ends with _frozen, not _tabae)
            # - mlp_vae_embedding_frozen (ends with _frozen, not _tabvae)
            # - mlp_wae_embedding_frozen (ends with _frozen, not _tabwae)
            if not any(model_name.endswith(suffix) for suffix in pretraining_suffixes):
                available_models.append(model_name)
    
    return sorted(available_models)


def analyze_feature_importance(experiment_dir: Path,
                               model_name: Optional[str] = None,
                               top_n: int = 20,
                               n_shap_samples: Optional[int] = None):

    # Ensure experiment_dir is a Path object
    experiment_path = Path(experiment_dir)
    
    _, experiment_dir, _ = load_experiment_config(experiment_path)
    if not experiment_dir.is_dir():
        raise ValueError(f"Resolved experiment directory is not a directory: {experiment_dir}")
    
    _print_banner(
        f"Starting analysis for experiment: {experiment_dir.name}\n"
        f"Experiment directory: {experiment_dir}"
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare data using the reusable function
    print("Preparing data...")
    result = prepare_data_from_experiment(
        experiment_dir=experiment_path,
        max_samples=None,  # Use all samples for feature importance
        return_train=True  
    )
    test_features, test_labels, test_indices, full_dataset, config_dict, generator, train_features, train_labels, train_indices = result
    
    # Get configurations
    full_config = config_dict['full_config']
    train_args = config_dict['train_args']
    training_cfg = config_dict['training_cfg']
    
    # Get feature names from dataset
    if hasattr(full_dataset, 'feature_names'):
        feature_names = full_dataset.feature_names
        print(f'Retrieved {len(feature_names)} feature names from Dataset')
    else:
        print('Warning: Dataset does not have feature_names attribute, using generic names')
        input_dim = test_features.shape[1]
        feature_names = [f'Feature_{i}' for i in range(input_dim)]
    
    # Get input dimension
    input_dim = test_features.shape[1]
    
    # Create test dataset from indices for dataloader
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create data loader
    batch_size = int(training_cfg.get('batch_size', 64))
    test_loader = create_dataloader(test_dataset, batch_size, False, train_args)
    
    # Find available models
    checkpoints_dir = experiment_dir / 'checkpoints'
    available_models = find_available_models(checkpoints_dir)
    
    if not available_models:
        raise ValueError("No model checkpoints found in experiment directory")
    
    # If model name is specified, process only that model; otherwise process all models
    if model_name is not None:
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        models_to_process = [model_name]
    else:
        models_to_process = available_models
    
    print(f'\nFound {len(models_to_process)} model(s) to process: {models_to_process}')
    
    # Analyze feature importance for each model
    results_dir = experiment_dir / 'feature_importance'
    results_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_process:
        # Pre-training models should already be filtered out by find_available_models,
        # but add a safety check just in case
        if any(x in model_name for x in ['_tabae', '_tabvae', '_tabwae']):
            _print_banner(
                f"Skipping pre-training model: {model_name}\n"
                "Pre-training models (TabAE, TabVAE, TabWAE) are not used for regression tasks",
                width=60,
            )
            continue
        
        _print_banner(f"Processing model: {model_name}", width=60)
        
        checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
        if not checkpoint_path.exists():
            print(f'Warning: Checkpoint not found for {model_name}, skipping...')
            continue
        
        try:
            print(f'Loading model from {checkpoint_path}')
            model = load_model_from_checkpoint(
                checkpoint_path, model_name, input_dim, full_config, device, 
                full_dataset=full_dataset,
                train_features=train_features, 
                train_labels=train_labels,
                checkpoints_dir=checkpoints_dir
            )
            
            if n_shap_samples is None:
                print('Computing SHAP-based feature importance (using all test samples)...')
            else:
                print(f'Computing SHAP-based feature importance (n_samples: {n_shap_samples})...')
            try:
                shap_importance, shap_values, shap_data = compute_shap_importance(
                    model, test_loader, device, model_name, feature_names, n_shap_samples,
                    return_values=True
                )

                beeswarm_path = results_dir / f'{model_name}_shap_beeswarm.png'
                plot_shap_beeswarm(shap_values, shap_data, feature_names, beeswarm_path, max_display=top_n)
                print(f'SHAP beeswarm plot saved to: {beeswarm_path}')
            except Exception as e:
                print(f'Error computing SHAP importance/plot: {e}')
                import traceback
                traceback.print_exc()
                continue

            if shap_importance.max() > shap_importance.min():
                shap_importance = (
                    (shap_importance - shap_importance.min())
                    / (shap_importance.max() - shap_importance.min())
                )

            plot_path = results_dir / f'{model_name}_feature_importance.png'
            plot_feature_importance(shap_importance, feature_names, plot_path, top_n)
            
            print(f'\nFeature importance analysis completed for model {model_name}!')
            
        except Exception as e:
            print(f'Error processing {model_name}: {e}')
            import traceback
            traceback.print_exc()
            print(f'Skipping {model_name}...')
            continue
    
    _print_banner(
        f"Feature importance analysis completed for all models!\nResults saved to: {results_dir}",
        width=60,
    )


def analyze_multiple_experiments(experiment_dirs: List[Path],
                                 model_name: Optional[str] = None,
                                 top_n: int = 20,
                                 n_shap_samples: Optional[int] = None):

    _print_banner(f"Starting batch analysis for {len(experiment_dirs)} experiments")
    
    for idx, exp_dir in enumerate(experiment_dirs, 1):
        _print_banner(
            f"Experiment {idx}/{len(experiment_dirs)}: {exp_dir.name}",
            char="#",
        )
        
        try:
            analyze_feature_importance(
                experiment_dir=exp_dir,
                model_name=model_name,
                top_n=top_n,
                n_shap_samples=n_shap_samples
            )
            print(f'\n✓ Experiment {idx} analysis completed\n')
        except Exception as e:
            print(f'\n✗ Experiment {idx} analysis failed: {e}\n')
            import traceback
            traceback.print_exc()
            continue
    
    _print_banner(f"Batch analysis completed! Processed {len(experiment_dirs)} experiments")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Feature Importance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

        """
    )
    
    # Support single or multiple experiment directories
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment_dir', type=str, default=None,
                       help='Single experiment output directory path (must contain checkpoints and configs subdirectories)')
    group.add_argument('--experiment_dirs', type=str, nargs='+', default=None,
                       help='Multiple experiment output directory paths (space-separated)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (if not specified, process all available models)')
    parser.add_argument('--top_n', type=int, default=20,
                       help='Show top N important features')
    def parse_n_shap_samples(x):
        """Parse n_shap_samples argument, allowing None to use all samples"""
        if x is None or (isinstance(x, str) and x.lower() == 'none'):
            return None
        return int(x)
    
    parser.add_argument('--n_shap_samples', type=parse_n_shap_samples, default=None,
                       help='Number of samples to use for SHAP computation. If not specified, use all test samples.')
    
    args = parser.parse_args()
    
    # Handle single or multiple experiment directories
    if args.experiment_dir:
        # Single experiment
        analyze_feature_importance(
            experiment_dir=Path(args.experiment_dir),
            model_name=args.model,
            top_n=args.top_n,
            n_shap_samples=args.n_shap_samples
        )
    elif args.experiment_dirs:
        # Multiple experiments
        experiment_dirs = [Path(d) for d in args.experiment_dirs]
        analyze_multiple_experiments(
            experiment_dirs=experiment_dirs,
            model_name=args.model,
            top_n=args.top_n,
            n_shap_samples=args.n_shap_samples
        )


if __name__ == '__main__':
    main()
