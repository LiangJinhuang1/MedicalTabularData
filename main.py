import copy
from train import train
from src.utils.config import load_config, get_config_value, resolve_path
from src.utils.save_utils import create_experiment_dir, save_configs
from src.utils.seed import set_seed
from src.utils.arg_parser import build_arg_parser


def _load_configs(exp_path, train_args_path):
    """Load experiment paths and training arguments configs."""
    paths_cfg = load_config(exp_path)
    train_args = load_config(train_args_path)
    return paths_cfg, train_args


def _resolve_paths(paths_cfg):
    """Resolve all config paths from exp_path config (single source of truth)."""
    train_file = resolve_path(get_config_value(paths_cfg, 'paths', 'train_file'))
    base_models_config_path = get_config_value(
        paths_cfg, 'paths', 'base_models_config',
        default='configs/models/base_models.yaml',
    )
    pretraining_config_path = get_config_value(
        paths_cfg, 'paths', 'pretraining_config',
        default='configs/models/pretraining.yaml',
    )
    data_config_path = get_config_value(
        paths_cfg, 'paths', 'data_config',
        default='configs/__base__/data_default.yaml',
    )
    return train_file, base_models_config_path, pretraining_config_path, data_config_path


def _extract_model_configs(base_models_config, pretraining_config):
    """Extract model configs from unified YAMLs."""
    base_models = base_models_config.get('base_models', {})
    mlp_model_cfg = base_models.get('mlp', {}).get('encoder', {})
    tabm_model_cfg = base_models.get('tabm', {})
    tabnet_model_cfg = base_models.get('tabnet', {})

    pretraining = pretraining_config.get('pretraining', {})
    encoder_model_cfg = pretraining.get('encoder', {})
    tabm_encoder_model_cfg = pretraining.get('tabm_encoder', {})
    wae_config = pretraining.get('wae', {})

    # WAE encoder config: merge encoder config with WAE-specific overrides
    wae_encoder_model_cfg = encoder_model_cfg.copy()
    if wae_config.get('latent_dim') is not None:
        wae_encoder_model_cfg['latent_dim'] = wae_config['latent_dim']

    return (
        mlp_model_cfg,
        tabm_model_cfg,
        tabnet_model_cfg,
        pretraining,
        encoder_model_cfg,
        tabm_encoder_model_cfg,
        wae_encoder_model_cfg,
    )


def _build_tabnet_config(tabnet_model_cfg):
    """Ensure TabNet config contains defaults (matching train.py)."""
    tabnet_model_cfg = tabnet_model_cfg or {}
    return {
        'n_d': tabnet_model_cfg.get('n_d', 8),
        'n_a': tabnet_model_cfg.get('n_a', 8),
        'n_steps': tabnet_model_cfg.get('n_steps', 3),
        'gamma': tabnet_model_cfg.get('gamma', 1.5),
        'n_independent': tabnet_model_cfg.get('n_independent', 2),
        'n_shared': tabnet_model_cfg.get('n_shared', 2),
        'epsilon': tabnet_model_cfg.get('epsilon', 1e-15),
        'virtual_batch_size': tabnet_model_cfg.get('virtual_batch_size', 128),
        'momentum': tabnet_model_cfg.get('momentum', 0.02),
        'mask_type': tabnet_model_cfg.get('mask_type', 'sparsemax'),
        'lambda_sparse': tabnet_model_cfg.get('lambda_sparse', 0.0001),
        **tabnet_model_cfg,  # Preserve any additional settings
    }


def _get_thresholds(task, training_cfg):
    """Resolve label thresholds for classification; return [None] for regression."""
    if task != 'classification':
        return [None]

    label_thresholds = training_cfg.get('label_thresholds', [])
    default_label_threshold = training_cfg.get('label_threshold')
    use_threshold_list = bool(training_cfg.get('use_threshold_list', False))

    if use_threshold_list and label_thresholds:
        return label_thresholds

    if default_label_threshold is None:
        if label_thresholds:
            default_label_threshold = label_thresholds[0]
        else:
            raise ValueError("classification task requires training.label_threshold or training.label_thresholds to be set")

    return [default_label_threshold]


def _get_subsets(task, keep_cols):
    """Select subset runs for the task."""
    if task == 'classification':
        # For classification: run once on full dataset only.
        return [('full', 'all')]

    subsets = [
        ('full', 'all'),
        ('with_intervention', 'with_intervention'),
        ('without_intervention', 'without_intervention'),
    ]
    if keep_cols:
        # Regression: add lab_only (feature subset from keep_cols)
        subsets.append(('lab_only', 'keep_columns'))
    return subsets


def _build_full_config(
    train_file,
    target_col,
    base_models_config_path,
    pretraining_config_path,
    data_config_path,
    local_train_args,
    data_config,
    mlp_model_cfg,
    tabm_model_cfg,
    tabnet_model_cfg_complete,
    encoder_model_cfg,
    tabm_encoder_model_cfg,
    wae_encoder_model_cfg,
    pretraining,
    local_training_cfg,
    seed,
    exclude_cols_list,
    args,
):
    """Compose the full config dict saved with each run."""
    return {
        'paths': {
            'train_file': str(train_file),
            'target_col': target_col,
            'base_models_config': str(base_models_config_path),
            'pretraining_config': str(pretraining_config_path),
        },
        # Store resolved configs for downstream analysis (single source of truth)
        'train_args': local_train_args,
        'data_config': data_config,
        'mlp_model': mlp_model_cfg,
        'tabm_model': tabm_model_cfg,
        'tabnet_model': tabnet_model_cfg_complete,
        # 'tabpfn_model': tabpfn_model_cfg,
        'encoder_model': encoder_model_cfg,
        'tabm_encoder_model': tabm_encoder_model_cfg,
        'wae_encoder_model': wae_encoder_model_cfg,
        'pretraining': pretraining,
        'training': local_training_cfg,
        'seed': seed,
        'exclude_cols': exclude_cols_list,
        'data_subset': local_train_args.get('data', {}),
        'config_paths': {
            'exp_path': str(args.exp_path),
            'train_args': str(args.train_args),
            'base_models_config': str(base_models_config_path),
            'pretraining_config': str(pretraining_config_path),
            'data_config': str(data_config_path),
        },
    }


def main():
    parser = build_arg_parser()
    
    args = parser.parse_args()
    
    # Load config files
    paths_cfg, train_args = _load_configs(args.exp_path, args.train_args)
    
    # Get target_col
    target_col = get_config_value(paths_cfg, 'paths', 'target_col')
    

    exclude_cols_list = train_args.get('exclude_cols', [])

    # Prepare config paths (single source of truth from exp_path)
    train_file, base_models_config_path, pretraining_config_path, data_config_path = _resolve_paths(paths_cfg)
    
    # Load unified configs
    base_models_config = load_config(base_models_config_path)
    pretraining_config = load_config(pretraining_config_path)
    data_config = load_config(data_config_path)
    
    # Extract model configs from unified structure
    (
        mlp_model_cfg,
        tabm_model_cfg,
        tabnet_model_cfg,
        pretraining,
        encoder_model_cfg,
        tabm_encoder_model_cfg,
        wae_encoder_model_cfg,
    ) = _extract_model_configs(base_models_config, pretraining_config)
    if getattr(args, 'lambda_ot', None) is not None:
        pretraining.setdefault('wae', {})['lambda_ot'] = args.lambda_ot
    # Use seed from config
    seed = train_args.get('seed', 42)
    
    # Ensure TabNet config contains all settings with defaults (matching train.py)
    tabnet_model_cfg_complete = _build_tabnet_config(tabnet_model_cfg)

    base_data_cfg = copy.deepcopy(train_args.get('data', {}))
    keep_cols = base_data_cfg.get('keep_cols') or base_data_cfg.get('keep_columns') or []
    base_training_cfg = train_args.get('training', {})
    task = base_training_cfg.get('task', 'regression')

    thresholds_to_run = _get_thresholds(task, base_training_cfg)
    subsets = _get_subsets(task, keep_cols)
    if args.subset is not None:
        requested = args.subset.strip()
        subsets = [(tag, val) for tag, val in subsets if tag == requested]
        if not subsets:
            raise SystemExit(
                f'--subset "{requested}" did not match any subset. '
                f'Valid for this task: {[s[0] for s in _get_subsets(task, keep_cols)]}'
            )

    for subset_tag, subset_value in subsets:
        for label_threshold in thresholds_to_run:
            local_train_args = copy.deepcopy(train_args)
            local_train_args.setdefault('data', {})
            local_train_args['data'] = {**base_data_cfg, 'subset': subset_value}

            local_training_cfg = copy.deepcopy(local_train_args.get('training', {}))
            if task == 'classification' and label_threshold is not None:
                local_training_cfg['label_threshold'] = label_threshold
            if getattr(args, 'gw_weight', None) is not None:
                local_training_cfg['gw_weight'] = args.gw_weight
            local_train_args['training'] = local_training_cfg

            # Set seed for reproducibility per run
            set_seed(seed)

            threshold_tag = None
            if task == 'classification' and label_threshold is not None:
                safe_threshold = str(label_threshold).replace('.', 'p')
                threshold_tag = f"{subset_tag}_thr{safe_threshold}"
            subset_run_tag = threshold_tag or subset_tag

            # Create experiment directory and save config for this training run
            experiment_dir = create_experiment_dir(
                output_root=None,
                target_col=target_col,
                seed=seed,
                subset_tag=subset_run_tag,
            )
            print(f'\nRunning subset: {subset_tag} (subset="{subset_value}")')
            if subset_value == 'keep_columns':
                print(f'Keep columns list size: {len(keep_cols)}')
            if label_threshold is not None:
                print(f'Label threshold: {label_threshold}')
            print(f'The experiment output directory: {experiment_dir}')
            print(f'Target column: {target_col}')

            full_config = _build_full_config(
                train_file,
                target_col,
                base_models_config_path,
                pretraining_config_path,
                data_config_path,
                local_train_args,
                data_config,
                mlp_model_cfg,
                tabm_model_cfg,
                tabnet_model_cfg_complete,
                encoder_model_cfg,
                tabm_encoder_model_cfg,
                wae_encoder_model_cfg,
                pretraining,
                local_training_cfg,
                seed,
                exclude_cols_list,
                args,
            )
            save_configs(experiment_dir, full_config)
            print(f'The config has been saved to: {experiment_dir / "configs"}')

            experiment_dir = train(
                target_col=target_col,
                exclude_cols=exclude_cols_list,
                train_file=train_file,
                train_args=local_train_args,
                mlp_model_cfg=mlp_model_cfg,
                tabm_model_cfg=tabm_model_cfg,
                tabnet_model_cfg=tabnet_model_cfg_complete,
                # tabpfn_model_cfg=tabpfn_model_cfg,
                encoder_model_cfg=encoder_model_cfg,
                tabm_encoder_model_cfg=tabm_encoder_model_cfg,
                wae_encoder_model_cfg=wae_encoder_model_cfg,
                training_cfg=local_training_cfg,
                experiment_dir=experiment_dir,
                data_config=data_config,
                pretraining_config=pretraining,
                seed=seed
            )
            print(f"\nResults saved to: {experiment_dir}")


if __name__ == '__main__':
    main()
