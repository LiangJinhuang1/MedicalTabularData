"""
Data preparation utilities for training scripts.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from typing import Tuple, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_data
from src.data.Dataset import Dataset
from src.utils.config import (
    load_config,
    resolve_path,
    get_config_value,
    get_variable_types,
    load_experiment_config,
)


def prepare_data(
    target_col: str,
    train_file: Union[str, Path],
    exclude_cols: Optional[list] = None,
    train_args: Optional[Dict] = None,
    data_config: Optional[Dict] = None,
    seed: Optional[int] = None,
    return_loaders: bool = False,
    return_features: bool = False,
    max_samples: Optional[int] = None,
    continuous_cols: Optional[list] = None,
    binary_cols: Optional[list] = None,
    categorical_cols: Optional[list] = None,
) -> Tuple:
    seed = 42 if seed is None else seed
    
    print(f'Setting random seed to: {seed} for reproducibility')
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Resolve train_file path
    train_file = resolve_path(train_file)
    
    print(f'Target column: {target_col}')
    print(f'Training data file: {train_file}')
    
    # Load data
    print(f'\nLoading data from {train_file}...')
    full_dataframe = load_data(train_file)
    print(f'Data shape: {full_dataframe.shape}')

    data_cfg_from_args = (train_args or {}).get('data', {})
    subset = data_cfg_from_args.get('subset', 'all')
    subset_col = data_cfg_from_args.get('subset_col', None)
    with_value = data_cfg_from_args.get('with_value', 1)
    without_value = data_cfg_from_args.get('without_value', 0)
    drop_subset_col = data_cfg_from_args.get('drop_subset_col', False)

    # Drop missing data before subset filtering
    missing_threshold = data_cfg_from_args.get('missing_threshold', 200)
    protected_cols = {target_col}
    if subset_col:
        protected_cols.add(subset_col)
    feature_cols = [col for col in full_dataframe.columns if col not in protected_cols]
    if feature_cols:
        null_counts = full_dataframe[feature_cols].isnull().sum()
        features_to_drop = null_counts[null_counts >= missing_threshold].index
        print(f'Dropping {len(features_to_drop)} features with more than {missing_threshold} missing values')
        if len(features_to_drop) > 0:
            full_dataframe = full_dataframe.drop(columns=features_to_drop)
    else:
        print('No features to drop')

    before_rows = len(full_dataframe)
    full_dataframe = full_dataframe.dropna()
    dropped_rows = before_rows - len(full_dataframe)
    print(f'Dropped {dropped_rows} rows with missing values before subsetting')
    print(f'Data shape after missing-value cleanup: {full_dataframe.shape}')
    if full_dataframe.empty:
        raise ValueError('Dropping missing values resulted in 0 rows. Check missing data handling.')

    # Optional subset filtering (e.g., with/without intervention)
    # Feature-only subset "keep_columns" is handled later and does not filter rows.
    if subset != 'all' and subset != 'keep_columns':
        if subset_col is None:
            raise ValueError("subset != 'all' requires train_args['data']['subset_col'] to be set")
        if subset_col not in full_dataframe.columns:
            raise ValueError(f"Subset column '{subset_col}' not found in the dataframe")

        if subset == 'with_intervention':
            full_dataframe = full_dataframe[full_dataframe[subset_col] == with_value]
            print(f"Using subset: with intervention ({len(full_dataframe)} rows)")
        elif subset == 'without_intervention':
            full_dataframe = full_dataframe[full_dataframe[subset_col] == without_value]
            print(f"Using subset: without intervention ({len(full_dataframe)} rows)")
        else:
            raise ValueError(f"Unknown subset type: {subset}")

        if full_dataframe.empty:
            raise ValueError(f"Subset '{subset}' resulted in 0 rows. Check subset_col/values.")

        if drop_subset_col and subset_col != target_col:
            full_dataframe = full_dataframe.drop(columns=[subset_col])

    # Handle excluded columns
    if exclude_cols:
        has_target = target_col in exclude_cols
        exclude_cols_filtered = [col for col in exclude_cols if col != target_col]
        if has_target:
            print(f'Warning: target_col "{target_col}" was in exclude_cols and will not be excluded')

        cols_to_exclude = [col for col in exclude_cols_filtered if col in full_dataframe.columns]
        if cols_to_exclude:
            full_dataframe = full_dataframe.drop(columns=cols_to_exclude)
            print(f'Excluded {len(cols_to_exclude)} columns: {cols_to_exclude}')
        elif exclude_cols_filtered:
            print(f'Warning: None of the specified exclude_cols {exclude_cols_filtered} were found in the dataframe')

        if target_col not in full_dataframe.columns:
            raise ValueError(f'Target column {target_col} was removed or not found in data')

    # Optional feature subset: keep only target + specified columns
    # Triggered when subset == 'keep_columns'
    if subset == 'keep_columns':
        keep_cols = data_cfg_from_args.get('keep_cols') or data_cfg_from_args.get('keep_columns') or []
        if not keep_cols:
            raise ValueError("subset 'keep_columns' requires train_args['data']['keep_cols'] (or keep_columns) to be set")
        keep_set = set(keep_cols)
        keep_set.add(target_col)
        missing_keep = [col for col in keep_cols if col not in full_dataframe.columns]
        if missing_keep:
            print(f'Warning: {len(missing_keep)} keep_cols not found in dataframe: {missing_keep}')
        cols_to_drop = [col for col in full_dataframe.columns if col not in keep_set]
        if cols_to_drop:
            full_dataframe = full_dataframe.drop(columns=cols_to_drop)
        print(f'Keeping {len(keep_set)} columns (target + keep_cols); dropped {len(cols_to_drop)} columns')

    # Task-specific target conversion (e.g., LVEF thresholding for classification)
    training_cfg = train_args.get('training', {}) if train_args else {}
    task = training_cfg.get('task', 'regression')
    if task == 'classification':
        label_threshold = training_cfg.get('label_threshold')
        if label_threshold is None:
            raise ValueError("classification task requires training.label_threshold to be set")
        positive_when = training_cfg.get('positive_when', 'above')
        if positive_when not in {'above', 'below'}:
            raise ValueError("training.positive_when must be 'above' or 'below'")

        if positive_when == 'above':
            full_dataframe[target_col] = (full_dataframe[target_col] > label_threshold).astype(int)
        else:
            full_dataframe[target_col] = (full_dataframe[target_col] < label_threshold).astype(int)

        print(f'Converted target to binary labels using threshold {label_threshold} (positive_when={positive_when})')

    # Optional log-transform of target for regression: log1p(x) = log(1+x).
    target_log_transform = bool(training_cfg.get('target_log_transform', False))
    if target_log_transform:
        if task == 'classification':
            raise ValueError('target_log_transform can only be used for regression tasks')
        target_values = full_dataframe[target_col].astype(float)
        if (target_values < 0).any():
            raise ValueError('target_log_transform (log1p) requires all target values to be >= 0')
        full_dataframe[target_col] = np.log1p(target_values)
        print(f'Applied log1p transform to target column "{target_col}"')

    # Normalize optional column lists
    continuous_cols = continuous_cols or []
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    
    # Get categorical grouping option and column_groups from config
    data_cfg = get_config_value(data_config, 'data', default={})
    group_categorical = data_cfg.get('group_categorical', True)
    column_groups = data_cfg.get('column_groups')
    
    # Optional log-transform of continuous feature columns (log1p)
    continuous_log_transform = bool(training_cfg.get('continuous_log_transform', False))
    if continuous_log_transform:
        cont_cols_for_log = [
            col for col in continuous_cols
            if col != target_col and col in full_dataframe.columns
        ]
        if cont_cols_for_log:
            cont_values = full_dataframe[cont_cols_for_log].astype(float)
            if (cont_values < 0).any().any():
                raise ValueError(
                    'continuous_log_transform (log1p) requires all continuous feature '
                    'values to be >= 0'
                )
            full_dataframe.loc[:, cont_cols_for_log] = np.log1p(cont_values)
            print(
                f'Applied log1p transform to {len(cont_cols_for_log)} continuous '
                f'feature columns'
            )
    
    # Get normalization option
    apply_normalization = training_cfg.get(
        'apply_normalization',
        (train_args or {}).get('apply_normalization', False),
    )

    # Create train/test split indices (used for normalization to avoid leakage)
    split_size = float((train_args or {}).get('split_size', training_cfg.get('split_size', 0.2)))
    total_size = len(full_dataframe)
    train_size = int(total_size * (1 - split_size))
    test_size = total_size - train_size
    if train_size <= 0 or test_size <= 0:
        raise ValueError(f'Invalid split_size={split_size}: train={train_size}, test={test_size}')

    perm = torch.randperm(total_size, generator=generator).tolist()
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]

    # Apply normalization using train split only (prevents leakage)
    norm_cols = [col for col in (continuous_cols or []) if col != target_col and col in full_dataframe.columns]
    normalize_here = bool(apply_normalization and norm_cols)
    if normalize_here:
        scaler = StandardScaler()
        train_cont = full_dataframe.iloc[train_indices][norm_cols]
        scaler.fit(train_cont.values)
        full_dataframe.loc[:, norm_cols] = scaler.transform(full_dataframe[norm_cols].values)

    # Create dataset (skip internal normalization if already applied)
    full_dataset = Dataset(
        full_dataframe,
        target_col,
        apply_normalization=False if normalize_here else apply_normalization,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        column_groups=column_groups,
        group_categorical=group_categorical,
        missing_threshold=missing_threshold,
    )
    
    print(f'Data Dimension: {len(full_dataset)}')
    
    # Create train/test split using precomputed indices (consistent with normalization)
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f'Train Data Dimension: {len(train_dataset)}')
    print(f'Val Data Dimension: {len(test_dataset)}')
    
    # Prepare return values
    result = [full_dataset, train_dataset, test_dataset, generator]
    
    # Create data loaders if requested
    if return_loaders:
        from src.data.dataloader import create_dataloader
        if train_args is None:
            raise ValueError("train_args must be provided when return_loaders=True")
        
        batch_size = int(training_cfg.get('batch_size', 64))
        train_loader = create_dataloader(train_dataset, batch_size, True, train_args)
        test_loader = create_dataloader(test_dataset, batch_size, False, train_args)
        result.extend([train_loader, test_loader])
        print(f'Created data loaders: batch_size={batch_size}')
    
    # Extract features if requested
    if return_features:
        test_indices = test_dataset.indices
        test_features = full_dataset.features[test_indices].numpy()
        test_labels = full_dataset.label[test_indices].numpy()
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(test_features):
            print(f'Limiting to {max_samples} samples')
            test_features = test_features[:max_samples]
            test_labels = test_labels[:max_samples]
            test_indices = test_indices[:max_samples]
        
        print(f'\nVal set feature shape: {test_features.shape}')
        print(f'Val set label range: [{test_labels.min():.4f}, {test_labels.max():.4f}]')
        result.extend([test_features, test_labels, test_indices])
    return tuple(result)


def prepare_data_from_experiment(
    experiment_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    return_train: bool = False,
) -> Tuple:
    """
    Reload data using the saved experiment configuration.

    Returns a tuple ordered for downstream analysis scripts:
    (test_features, test_labels, test_indices, full_dataset, config_dict, generator, ...)
    """
    full_config, experiment_dir, config_path = load_experiment_config(experiment_dir)
    train_args = full_config.get('train_args')
    if not train_args:
        raise ValueError('train_args not found in full_config')
    if not isinstance(train_args, dict):
        raise ValueError('train_args in full_config must be a dict')

    data_config = full_config.get('data_config')
    if not data_config:
        raise ValueError('data_config not found in full_config')
    if not isinstance(data_config, dict):
        raise ValueError('data_config in full_config must be a dict')

    paths_cfg = full_config.get('paths', {})
    train_file = paths_cfg.get('train_file')
    target_col = paths_cfg.get('target_col')
    if train_file is None:
        raise ValueError('train_file path not found in configuration file')
    if target_col is None:
        raise ValueError('target_col not found in configuration file')

    exclude_cols = full_config.get('exclude_cols', [])
    continuous_cols, binary_cols, categorical_cols = get_variable_types(data_config)

    prepared = prepare_data(
        target_col=target_col,
        train_file=train_file,
        exclude_cols=exclude_cols,
        train_args=train_args,
        data_config=data_config,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        return_loaders=False,
        return_features=True,
        max_samples=max_samples,
    )
    full_dataset, train_dataset, test_dataset, generator, test_features, test_labels, test_indices = prepared

    config_dict = {
        'full_config': full_config,
        'train_args': train_args,
        'training_cfg': train_args.get('training', {}),
        'data_config': data_config,
        'target_col': target_col,
        'paths_cfg': paths_cfg,
    }

    if return_train:
        train_indices = train_dataset.indices
        train_features = full_dataset.features[train_indices].numpy()
        train_labels = full_dataset.label[train_indices].numpy()
        return (
            test_features,
            test_labels,
            test_indices,
            full_dataset,
            config_dict,
            generator,
            train_features,
            train_labels,
            train_indices,
        )

    return (
        test_features,
        test_labels,
        test_indices,
        full_dataset,
        config_dict,
        generator,
    )


def get_model_configs(config_dict: Dict) -> Dict:
    full_config = config_dict['full_config']
    training_cfg = config_dict['training_cfg']
    
    mlp_model_cfg = full_config.get('mlp_model', {})
    tabm_model_cfg = full_config.get('tabm_model', {})
    tabnet_model_cfg = full_config.get('tabnet_model', {})
    tabpfn_model_cfg = full_config.get('tabpfn_model', {})
    encoder_model_cfg = full_config.get('encoder_model', {})
    wae_encoder_model_cfg = full_config.get('wae_encoder_model', {})
    encoder_latent_dim = encoder_model_cfg.get('latent_dim', 32)
    wae_latent_dim = wae_encoder_model_cfg.get('latent_dim', encoder_latent_dim)
    wae_hidden_dims = wae_encoder_model_cfg.get('hidden_dims', encoder_model_cfg.get('hidden_dims', [32, 16]))
    
    # Get WAE regularization type from pretraining config or training config
    pretraining = full_config.get('pretraining', {})
    wae_config = pretraining.get('wae', {})
    wae_regularization_type = wae_config.get('regularization_type', training_cfg.get('wae_regularization_type', 'sinkhorn'))
    
    return {
        'mlp_model_cfg': mlp_model_cfg,
        'tabm_model_cfg': tabm_model_cfg,
        'tabnet_model_cfg': tabnet_model_cfg,
        'tabpfn_model_cfg': tabpfn_model_cfg,
        'encoder_model_cfg': encoder_model_cfg,
        'wae_encoder_model_cfg': wae_encoder_model_cfg,
        'encoder_latent_dim': encoder_latent_dim,
        'wae_latent_dim': wae_latent_dim,
        'wae_hidden_dims': wae_hidden_dims,
        'wae_regularization_type': wae_regularization_type,
    }
