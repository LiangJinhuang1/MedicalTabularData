"""
Weights & Biases Sweep Main Script
This script is called by wandb sweep to run hyperparameter optimization.
"""
import wandb
import os
from train import train
from src.utils.config import load_config, get_config_value, resolve_path
from src.utils.save_utils import create_experiment_dir
from src.utils.seed import set_seed
import argparse


def get_wandb_config_value(wandb_config, key: str, default):
    """
    Safely get value from wandb config, supporting both dict and attribute access.
    
    Args:
        wandb_config: wandb.config object
        key: Configuration key
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    try:
        if hasattr(wandb_config, key):
            return getattr(wandb_config, key)
        elif key in wandb_config:
            return wandb_config[key]
        else:
            return default
    except:
        return default


def apply_sweep_config_to_model_configs(wandb_config, base_models_config, pretraining_config):
    """
    Apply hyperparameters from wandb.config to model configurations.
    
    Args:
        wandb_config: wandb.config object containing sweep hyperparameters
        base_models_config: Base models configuration dict
        pretraining_config: Pre-training configuration dict
    
    Returns:
        Updated base_models_config and pretraining_config
    """
    # Update MLP base model config
    if 'base_models' not in base_models_config:
        base_models_config['base_models'] = {}
    if 'mlp' not in base_models_config['base_models']:
        base_models_config['base_models']['mlp'] = {}
    if 'encoder' not in base_models_config['base_models']['mlp']:
        base_models_config['base_models']['mlp']['encoder'] = {}
    
    mlp_encoder = base_models_config['base_models']['mlp']['encoder']
    mlp_hidden_dims = [
        get_wandb_config_value(wandb_config, 'mlp_hidden_dim_1', 512),
        get_wandb_config_value(wandb_config, 'mlp_hidden_dim_2', 256),
        get_wandb_config_value(wandb_config, 'mlp_hidden_dim_3', 128),
        get_wandb_config_value(wandb_config, 'mlp_hidden_dim_4', 64)
    ]
    # Remove None values (for backward compatibility with 3-layer configs)
    mlp_encoder['hidden_dims'] = [d for d in mlp_hidden_dims if d is not None]
    mlp_encoder['latent_dim'] = get_wandb_config_value(wandb_config, 'mlp_latent_dim', 64)
    mlp_encoder['dropout'] = get_wandb_config_value(wandb_config, 'mlp_dropout', 0.35)
    
    # Update TabM config
    if 'tabm' not in base_models_config['base_models']:
        base_models_config['base_models']['tabm'] = {}
    base_models_config['base_models']['tabm']['k_heads'] = get_wandb_config_value(wandb_config, 'tabm_k_heads', 12)
    base_models_config['base_models']['tabm']['dropout'] = get_wandb_config_value(wandb_config, 'tabm_dropout', 0.3)
    
    # Update TabNet config
    if 'tabnet' not in base_models_config['base_models']:
        base_models_config['base_models']['tabnet'] = {}
    base_models_config['base_models']['tabnet']['n_d'] = get_wandb_config_value(wandb_config, 'tabnet_n_d', 32)
    base_models_config['base_models']['tabnet']['n_a'] = get_wandb_config_value(wandb_config, 'tabnet_n_a', 32)
    base_models_config['base_models']['tabnet']['n_steps'] = get_wandb_config_value(wandb_config, 'tabnet_n_steps', 5)
    base_models_config['base_models']['tabnet']['gamma'] = get_wandb_config_value(wandb_config, 'tabnet_gamma', 1.5)
    base_models_config['base_models']['tabnet']['lambda_sparse'] = get_wandb_config_value(wandb_config, 'tabnet_lambda_sparse', 0.0001)
    
    # Update encoder config (for pre-training models)
    if 'pretraining' not in pretraining_config:
        pretraining_config['pretraining'] = {}
    if 'encoder' not in pretraining_config['pretraining']:
        pretraining_config['pretraining']['encoder'] = {}
    
    encoder = pretraining_config['pretraining']['encoder']
    encoder['latent_dim'] = get_wandb_config_value(wandb_config, 'encoder_latent_dim', 64)
    # Support 2, 3, or 4 hidden layers (4 layers for better performance: [512, 256, 128, 64])
    encoder_hidden_dims = [
        get_wandb_config_value(wandb_config, 'encoder_hidden_dim_1', 512),
        get_wandb_config_value(wandb_config, 'encoder_hidden_dim_2', 256),
        get_wandb_config_value(wandb_config, 'encoder_hidden_dim_3', 128),
        get_wandb_config_value(wandb_config, 'encoder_hidden_dim_4', 64)
    ]
    # Remove None values (for backward compatibility with 2-3 layer configs)
    encoder['hidden_dims'] = [d for d in encoder_hidden_dims if d is not None]
    encoder['dropout'] = get_wandb_config_value(wandb_config, 'encoder_dropout', 0.35)
    
    # Update VAE config
    if 'vae' not in pretraining_config['pretraining']:
        pretraining_config['pretraining']['vae'] = {}
    pretraining_config['pretraining']['vae']['beta'] = get_wandb_config_value(wandb_config, 'vae_beta', 1.0)
    
    # Update WAE config
    if 'wae' not in pretraining_config['pretraining']:
        pretraining_config['pretraining']['wae'] = {}
    pretraining_config['pretraining']['wae']['lambda_ot'] = get_wandb_config_value(wandb_config, 'wae_lambda_ot', 1.0)
    pretraining_config['pretraining']['wae']['sinkhorn_eps'] = get_wandb_config_value(wandb_config, 'wae_sinkhorn_eps', 0.1)
    
    return base_models_config, pretraining_config


def main():
    # Initialize wandb with increased timeout settings to handle network issues.
    #
    # IMPORTANT:
    # - Recent wandb versions use pydantic Settings and forbid unknown/extra fields.
    # - Do NOT pass private keys like _network_timeout/_request_timeout/_start_timeout
    #   via wandb.Settings(...); instead, rely on environment variables.
    #
    # Timeout values can be overridden via environment variables:
    # - WANDB_NETWORK_TIMEOUT
    # - WANDB_REQUEST_TIMEOUT
    # - WANDB_START_TIMEOUT
    network_timeout = int(os.getenv('WANDB_NETWORK_TIMEOUT', '60'))
    request_timeout = int(os.getenv('WANDB_REQUEST_TIMEOUT', '60'))
    start_timeout = int(os.getenv('WANDB_START_TIMEOUT', '60'))
    
    # Check if offline mode is requested via environment variable
    mode = os.getenv('WANDB_MODE', 'online')
    if mode == 'offline':
        print('⚠️  Running wandb in offline mode (set WANDB_MODE=offline)')
    
    # Get base_url from environment variable, default to https://api.wandb.ai
    # This fixes issues where wandb tries to connect to wrong endpoints (e.g., Kubernetes internal services)
    base_url = os.getenv('WANDB_BASE_URL', 'https://api.wandb.ai')
    if base_url != 'https://api.wandb.ai':
        print(f'⚠️  Using custom wandb base_url: {base_url}')

    # Ensure timeouts are available to wandb via env vars (works across wandb versions).
    os.environ.setdefault('WANDB_NETWORK_TIMEOUT', str(network_timeout))
    os.environ.setdefault('WANDB_REQUEST_TIMEOUT', str(request_timeout))
    os.environ.setdefault('WANDB_START_TIMEOUT', str(start_timeout))

    # Prefer configuring base_url via env var for maximum compatibility.
    if base_url and base_url != 'https://api.wandb.ai':
        os.environ['WANDB_BASE_URL'] = base_url

    # Build a minimal Settings object when possible; fall back to wandb.init without it.
    wandb_init_kwargs = {'mode': mode}
    try:
        if base_url and base_url != 'https://api.wandb.ai':
            # Some wandb versions support base_url in Settings, others don't.
            wandb_init_kwargs['settings'] = wandb.Settings(base_url=base_url)
        else:
            wandb_init_kwargs['settings'] = wandb.Settings()
    except Exception as e:
        print(f'⚠️  Could not construct wandb.Settings(...): {e}. Falling back to wandb.init without settings.')

    run = wandb.init(**wandb_init_kwargs)
    config = wandb.config
    
    # Parse command line arguments (for non-sweep parameters)
    parser = argparse.ArgumentParser(description='Wandb Sweep Training')
    parser.add_argument('--exp-path', type=str, default='configs/__base__/exp_path.yaml', 
                       help='Experiment paths YAML')
    parser.add_argument('--train-args', type=str, default='configs/__base__/train_argument.yaml', 
                       help='Training arguments YAML')
    parser.add_argument('--train-file', type=str, default=None, 
                       help='Override: training CSV path')
    parser.add_argument('--exclude-cols', type=str, nargs='+', action='append', default=None,
                       help='Override: Lists of columns to exclude')
    parser.add_argument('--base-models-config', type=str, default=None, 
                       help='Override: Base models config YAML path')
    parser.add_argument('--pretraining-config', type=str, default=None, 
                       help='Override: Pre-training config YAML path')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Parse known args only - ignore wandb sweep hyperparameters (they come from wandb.config)
    args, unknown = parser.parse_known_args()
    
    # Log any unknown arguments (these are sweep hyperparameters from wandb, which is expected)
    if unknown:
        print(f"Note: Ignoring wandb sweep hyperparameters passed as command line args (read from wandb.config instead): {len(unknown)} parameters")
    
    # Load config files
    paths_cfg = load_config(args.exp_path)
    train_args = load_config(args.train_args)
    
    # Get target_col
    target_col = get_config_value(paths_cfg, 'paths', 'target_col')
    
    if args.exclude_cols:
        exclude_cols_list = args.exclude_cols
    else:
        exclude_cols_list = train_args.get('exclude_cols', [])
    
    # Prepare config paths 
    train_file = args.train_file or get_config_value(paths_cfg, 'paths', 'train_file')
    train_file = resolve_path(train_file)
    
    # Get config paths with defaults
    base_models_config_path = args.base_models_config or get_config_value(
        paths_cfg, 'paths', 'base_models_config',
        default='configs/models/base_models.yaml'
    )
    pretraining_config_path = args.pretraining_config or get_config_value(
        paths_cfg, 'paths', 'pretraining_config',
        default='configs/models/pretraining.yaml'
    )
    data_config_path = get_config_value(
        paths_cfg, 'paths', 'data_config',
        default='configs/__base__/data_default.yaml'
    )
    
    # Load unified configs
    base_models_config = load_config(base_models_config_path)
    pretraining_config = load_config(pretraining_config_path)
    data_config = load_config(data_config_path)
    
    # Apply sweep hyperparameters to configs
    base_models_config, pretraining_config = apply_sweep_config_to_model_configs(
        config, base_models_config, pretraining_config
    )
    
    # Extract model configs from unified structure
    base_models = base_models_config.get('base_models', {})
    mlp_model_cfg = base_models.get('mlp', {}).get('encoder', {})
    tabm_model_cfg = base_models.get('tabm', {})
    tabnet_model_cfg = base_models.get('tabnet', {})
    
    # Extract pretraining configs
    pretraining = pretraining_config.get('pretraining', {})
    encoder_model_cfg = pretraining.get('encoder', {})
    tabm_encoder_model_cfg = pretraining.get('tabm_encoder', {})
    wae_config = pretraining.get('wae', {})
    
    # WAE encoder config: merge encoder config with WAE-specific overrides
    wae_encoder_model_cfg = encoder_model_cfg.copy()
    if wae_config.get('latent_dim') is not None:
        wae_encoder_model_cfg['latent_dim'] = wae_config['latent_dim']
    
    # Update training config with sweep hyperparameters
    training_cfg = train_args.get('training', {}).copy()
    
    # Get values from wandb config and validate ranges
    learning_rate = get_wandb_config_value(config, 'learning_rate', 0.001)
    weight_decay = get_wandb_config_value(config, 'weight_decay', 1e-5)
    gw_weight = get_wandb_config_value(config, 'gw_weight', 0.1)
    vae_beta = get_wandb_config_value(config, 'vae_beta', 1.0)
    wae_lambda_ot = get_wandb_config_value(config, 'wae_lambda_ot', 1.0)
    
    # Validate and warn if values seem out of expected range
    # (These are warnings, not errors, as wandb may sample edge cases)
    if learning_rate > 0.01:
        print(f"⚠️  WARNING: learning_rate={learning_rate} seems too large (expected < 0.01)")
    if weight_decay > 0.1:
        print(f"⚠️  WARNING: weight_decay={weight_decay} seems too large (expected < 0.1)")
    if gw_weight > 1.0:
        print(f"⚠️  WARNING: gw_weight={gw_weight} seems too large (expected < 1.0)")
    if gw_weight < 0.01:
        print(f"⚠️  WARNING: gw_weight={gw_weight} seems too small (expected >= 0.01)")
    if vae_beta > 1.0:
        print(f"⚠️  WARNING: vae_beta={vae_beta} seems too large (expected < 1.0)")
    if wae_lambda_ot > 1.0:
        print(f"⚠️  WARNING: wae_lambda_ot={wae_lambda_ot} seems too large (expected < 1.0)")
    
    training_cfg.update({
        'learning_rate': learning_rate,
        'batch_size': get_wandb_config_value(config, 'batch_size', 128),
        'weight_decay': weight_decay,
        'epochs': get_wandb_config_value(config, 'epochs', 100),
        'vae_beta': vae_beta,
        'lambda_ot': wae_lambda_ot,
        'sinkhorn_eps': get_wandb_config_value(config, 'wae_sinkhorn_eps', 0.1),
        'gw_weight': gw_weight,
        'use_log_ratio': get_wandb_config_value(config, 'use_log_ratio', False)
    })
    
    # Print key hyperparameters for debugging
    print(f'\n📊 Sweep Hyperparameters:')
    print(f'  learning_rate: {learning_rate}')
    print(f'  weight_decay: {weight_decay}')
    print(f'  gw_weight: {gw_weight}')
    print(f'  vae_beta: {vae_beta}')
    print(f'  wae_lambda_ot: {wae_lambda_ot}')
    
    # Use seed from command line argument if provided, otherwise from config
    seed = args.seed if args.seed is not None else train_args.get('seed', 42)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # For sweep, we don't need to save checkpoints or plots, so use a minimal experiment_dir
    # Still create it for compatibility, but we'll disable saving outputs
    experiment_dir = create_experiment_dir(output_root=None, target_col=target_col, seed=seed)
    print(f'\nThe experiment output directory: {experiment_dir}')
    print(f'Target column: {target_col}')
    print(f'Wandb Run: {run.name} ({run.id})')
    print(f'⚠️  Sweep mode: Checkpoints, plots, and configs will NOT be saved (see wandb for all info)')
    
    # Get config paths (for logging to wandb, not saving to disk)
    base_models_config_path = args.base_models_config or get_config_value(
        paths_cfg, 'paths', 'base_models_config', 
        default='configs/models/base_models.yaml'
    )
    pretraining_config_path = args.pretraining_config or get_config_value(
        paths_cfg, 'paths', 'pretraining_config', 
        default='configs/models/pretraining.yaml'
    )
    data_config_path = get_config_value(
        paths_cfg, 'paths', 'data_config', 
        default='configs/__base__/data_default.yaml'
    )
    
    # Ensure TabNet config contains all settings with defaults (matching train.py)
    if tabnet_model_cfg is None:
        tabnet_model_cfg = {}
    # Add default values if not present (updated for 50K data: n_d=32, n_a=32, n_steps=5)
    tabnet_model_cfg_complete = {
        'n_d': tabnet_model_cfg.get('n_d', 32),
        'n_a': tabnet_model_cfg.get('n_a', 32),
        'n_steps': tabnet_model_cfg.get('n_steps', 5),
        'gamma': tabnet_model_cfg.get('gamma', 1.5),
        **tabnet_model_cfg  # Preserve any additional settings
    }
    
    # Prepare config dict for wandb logging (not saving to disk)
    full_config = {
        'paths': {
            'train_file': str(train_file),
            'target_col': target_col,
            'base_models_config': str(base_models_config_path),
            'pretraining_config': str(pretraining_config_path),
        },
        'mlp_model': mlp_model_cfg,
        'tabm_model': tabm_model_cfg,
        'tabnet_model': tabnet_model_cfg_complete,
        'encoder_model': encoder_model_cfg,
        'tabm_encoder_model': tabm_encoder_model_cfg,
        'wae_encoder_model': wae_encoder_model_cfg,
        'pretraining': pretraining,
        'training': training_cfg,
        'seed': seed,
        'exclude_cols': exclude_cols_list,
        'wandb_run_id': run.id,
        'wandb_run_name': run.name,
        'config_paths': {
            'exp_path': str(args.exp_path),
            'train_args': str(args.train_args),
            'base_models_config': str(base_models_config_path),
            'pretraining_config': str(pretraining_config_path),
            'data_config': str(data_config_path),
        }
    }
    # Log config to wandb instead of saving to disk
    wandb.config.update(full_config)
    print(f'Config logged to wandb (not saved to disk)')
    
    # Run training
   
        experiment_dir = train(
            target_col=target_col,
            exclude_cols=exclude_cols_list,
            train_file=train_file,
            train_args=train_args,
            mlp_model_cfg=mlp_model_cfg,
            tabm_model_cfg=tabm_model_cfg,
            tabnet_model_cfg=tabnet_model_cfg,
            encoder_model_cfg=encoder_model_cfg,
            tabm_encoder_model_cfg=tabm_encoder_model_cfg,
            wae_encoder_model_cfg=wae_encoder_model_cfg,
            training_cfg=training_cfg,
            experiment_dir=experiment_dir,
            data_config=data_config,
            pretraining_config=pretraining,
            seed=seed,
            return_loss_tracker=True,  # Request loss tracker to be returned
            save_outputs=False  # Disable saving checkpoints and plots for sweep
        )
        
        # Get loss tracker from return value
        if isinstance(experiment_dir, tuple):
            experiment_dir, loss_tracker = experiment_dir
        else:
            # Fallback: try to load from saved file
            loss_tracker = None
            print("Warning: Could not get loss_tracker from train() return value")
        
        # Log best validation losses and R2 scores to wandb
        if loss_tracker is not None:
            # Find the best overall validation loss across all models
            best_overall_loss = float('inf')
            best_model_name_loss = None
            
            # Find the best overall R2 score across all models
            best_overall_r2 = float('-inf')
            best_model_name_r2 = None
            
            for model_name, val_loss in loss_tracker.best_val_loss.items():
                if isinstance(val_loss, (int, float)) and not (val_loss == float('inf') or val_loss == float('-inf')):
                    # Log individual model best losses
                    wandb.log({f'best_val_loss/{model_name}': val_loss})
                    
                    # Track overall best loss
                    if val_loss < best_overall_loss:
                        best_overall_loss = val_loss
                        best_model_name_loss = model_name
                    
                    # Also log best R2 if available
                    if model_name in loss_tracker.best_val_r2:
                        r2 = loss_tracker.best_val_r2[model_name]
                        if isinstance(r2, (int, float)) and not (r2 == float('inf') or r2 == float('-inf')):
                            wandb.log({f'best_val_r2/{model_name}': r2})
                            
                            # Track overall best R2
                            if r2 > best_overall_r2:
                                best_overall_r2 = r2
                                best_model_name_r2 = model_name
            
            # Log the best overall metrics (these are what the sweep can optimize)
            metrics_to_log = {}
            
            if best_model_name_loss:
                metrics_to_log['best_val_loss'] = best_overall_loss
                metrics_to_log['best_model_loss'] = best_model_name_loss
                print(f'\nBest validation loss: {best_overall_loss:.4f} (model: {best_model_name_loss})')
            else:
                print('\nWarning: Could not determine best validation loss')
            
            if best_model_name_r2:
                metrics_to_log['best_val_r2'] = best_overall_r2
                metrics_to_log['best_model_r2'] = best_model_name_r2
                print(f'Best validation R²: {best_overall_r2:.4f} (model: {best_model_name_r2})')
            else:
                print('Warning: Could not determine best validation R²')
            
            # Log all metrics at once
            if metrics_to_log:
                wandb.log(metrics_to_log)
        else:
            print('\nWarning: Loss tracker not available, cannot log metrics to wandb')
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 All results and metrics logged to wandb")
        print(f"🔗 Wandb run: {wandb.run.url if wandb.run else 'N/A'}")
        print(f"💾 No files saved to disk (sweep mode)")
        
    except Exception as e:
        print(f'Error during training: {e}')
        wandb.log({'error': str(e)})
        raise
    
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
