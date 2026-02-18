import torch
import numpy as np
import torch.nn as nn
import logging
import sys
import time
import math
from pprint import pformat
from torch.optim import Adam
from torch.nn import MSELoss
import copy
from src.data.prepare_data import prepare_data
from src.data.dataloader import create_dataloader
from src.utils.save_utils import (
    save_checkpoint, 
    LossTracker
)
from src.models.TabM.tabM import TabM
from src.models.TabM.TabMEncoder import TabMEncoder
from src.models.MLP import MLPEncoder
from src.models.TabNet.TabNet import TabNet
from src.models.TabNet.TabNetEncoder import TabNetEncoder
# from src.models.TabPFN.TabPFN import TabPFN
# from src.models.TabPFN.TabPFNEncoder import TabPFNEncoder
from src.training.train_loop import train_loop
from src.eval.eval_loop import eval_loop
from src.models.Embedding.EncoderEmbedding import EncoderEmbedding
from src.models.Embedding.TabAE import TabAE
from src.models.Embedding.TabVAE import TabVAE
from src.models.Embedding.VAEEncoderEmbedding import VAEEncoderEmbedding
from src.models.Embedding.Decoder import TabularDecoder
from src.training.embedding_train_loop import train_tabae, train_tabvae, train_tabwae
from src.models.Embedding.MultiAE import MultiTaskEncoderEmbedding
from src.models.Embedding.MultiVAE import MultitaskVAEEncoderEmbedding
from src.training.train_multi import train_multitask, train_multi_vae
from src.training.train_gw import train_gw
from src.eval.eval_multitask import eval_multitask, eval_multi_vae
from src.eval.eval_gw import eval_gw
from src.training.Tabuarloss import TabularLoss
from src.utils.config import load_config, get_config_value, get_variable_types




def setup_logger(experiment_dir=None):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if experiment_dir is not None:
        logs_dir = experiment_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _build_config_summary(
    target_col,
    train_file,
    seed,
    task,
    training_cfg,
    mlp_model_cfg,
    tabm_model_cfg,
    tabnet_model_cfg,
    encoder_model_cfg,
    wae_encoder_model_cfg,
    pretraining_config,
    train_args,
    exclude_cols,
):
    """Compact summary of configs for logging/debugging."""
    return {
        'target_col': target_col,
        'train_file': str(train_file) if train_file is not None else None,
        'seed': seed,
        'task': task,
        'training_cfg': training_cfg,
        'mlp_model_cfg': mlp_model_cfg,
        'tabm_model_cfg': tabm_model_cfg,
        'tabnet_model_cfg': tabnet_model_cfg,
        'encoder_model_cfg': encoder_model_cfg,
        'wae_encoder_model_cfg': wae_encoder_model_cfg,
        'pretraining_cfg': pretraining_config,
        'data_subset_cfg': (train_args or {}).get('data', {}),
        'exclude_cols': exclude_cols,
    }


def _prepare_tabnet_loaders(
    train_dataset,
    val_dataset,
    train_args,
    training_cfg,
    tabnet_model_cfg,
    train_loader,
    val_loader,
):
    """Build TabNet loaders; reuse base loaders if batch size matches."""
    batch_size = int(training_cfg.get('batch_size', 64))
    tabnet_batch_size = int(training_cfg.get('batch_size_tabnet', tabnet_model_cfg.get('batch_size_tabnet', batch_size)))
    if tabnet_batch_size != batch_size:
        train_loader_tabnet = create_dataloader(train_dataset, tabnet_batch_size, True, train_args)
        val_loader_tabnet = create_dataloader(val_dataset, tabnet_batch_size, False, train_args)
    else:
        train_loader_tabnet = train_loader
        val_loader_tabnet = val_loader
    return train_loader_tabnet, val_loader_tabnet


def create_frozen_finetuned_encoders(encoder_base):
    encoder_frozen = copy.deepcopy(encoder_base)
    encoder_finetuned = copy.deepcopy(encoder_base)
    for param in encoder_frozen.parameters():
        param.requires_grad = False
    for param in encoder_finetuned.parameters():
        param.requires_grad = True
    return encoder_frozen, encoder_finetuned


def load_vae_layers_from_tabvae(vae_embedding_model, tabvae_model):
    """
    Load VAE layers (mu, log_var) from a pretrained TabVAE model to VAEEncoderEmbedding.
    This is a helper function to avoid manual state_dict loading.
    
    Args:
        vae_embedding_model: VAEEncoderEmbedding model instance
        tabvae_model: Pretrained TabVAE model instance
    """
    vae_embedding_model.mu.load_state_dict(tabvae_model.mu.state_dict())
    vae_embedding_model.log_var.load_state_dict(tabvae_model.log_var.state_dict())


def load_checkpoint(model, checkpoints_dir, model_name, device):
    best_path = checkpoints_dir / f"{model_name}_best.pt"
    if best_path.exists():
        print(f'\nLoading best {model_name} from {best_path}')
        checkpoint = torch.load(best_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        
        
        model.load_state_dict(state_dict, strict=False)
        print(f'Best {model_name} loaded - Val Loss: {checkpoint["val_loss"]:.4f} at epoch {checkpoint["epoch"]+1}')
        return checkpoint
    else:
        print(f'\nWarning: Best {model_name} checkpoint not found at {best_path}, using current state')
        return None


def pretrain_and_load_checkpoint(pretrain_model, pretrain_optimizer, pretrain_fn, train_loader, val_loader, 
                                  loss_fn, device, num_epochs, loss_tracker, experiment_dir, 
                                  model_name, dataset=None, logger=None, **pretrain_kwargs):
    banner = '=' * 60
    print('\n' + banner)
    print(f'Pre-training {model_name} (Unsupervised)')
    print(banner)
    if logger is not None:
        logger.info(banner)
        logger.info(f'Pre-training {model_name} (Unsupervised)')
        logger.info(banner)
    
    # Pre-train
    pretrain_fn(pretrain_model, train_loader, val_loader, pretrain_optimizer, loss_fn, 
                device, num_epochs, loss_tracker, experiment_dir, dataset=dataset, 
                model_name=model_name, logger=logger, **pretrain_kwargs)
    
    # Load best checkpoint using the unified function
    checkpoints_dir = experiment_dir / "checkpoints"
    return load_checkpoint(pretrain_model, checkpoints_dir, model_name, device)


def train(target_col, exclude_cols=None, train_file=None, train_args=None, 
          mlp_model_cfg=None, tabm_model_cfg=None, tabnet_model_cfg=None, tabpfn_model_cfg=None,
          encoder_model_cfg=None, tabm_encoder_model_cfg=None, wae_encoder_model_cfg=None, training_cfg=None, experiment_dir=None, 
          data_config=None, pretraining_config=None, seed=None, return_loss_tracker=False, save_outputs=True):

    logger = setup_logger(experiment_dir)
    logger.info("Starting training run")
    run_start_time = time.time()

    # Get training_cfg from train_args if not provided
    if training_cfg is None:
        training_cfg = train_args.get('training', {}) if train_args else {}
    mlp_model_cfg = mlp_model_cfg or {}
    tabm_model_cfg = tabm_model_cfg or {}
    tabnet_model_cfg = tabnet_model_cfg or {}
    tabpfn_model_cfg = tabpfn_model_cfg or {}
    encoder_model_cfg = encoder_model_cfg or {}
    tabm_encoder_model_cfg = tabm_encoder_model_cfg or {}
    wae_encoder_model_cfg = wae_encoder_model_cfg or {}
    
    # Load variable types from data config
    if data_config is None:
        # Try to load from default path if not provided
        try:
            data_config = load_config('configs/__base__/data_default.yaml')
            print('Loaded data config from default path: configs/__base__/data_default.yaml')
        except Exception as e:
            print(f'Warning: Could not load data config: {e}')
            data_config = {}
    
    # Get variable type lists from config (for information only)
    continuous_cols, binary_cols, categorical_cols = get_variable_types(data_config)
    
    if continuous_cols or binary_cols or categorical_cols:
        logger.info(f'Loaded variable types from config: {len(continuous_cols)} continuous, {len(binary_cols)} binary, {len(categorical_cols)} categorical')
    else:
        logger.warning('No variable types found in config. Models will use default single-head decoder.')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    



    # Get seed from parameter or train_args (for backward compatibility)
    if seed is None:
        seed = train_args.get('seed', 42) if train_args else 42
    
    # Prepare data using the reusable function
    logger.info("Preparing data...")
    full_dataset, train_dataset, val_dataset, _, train_loader, val_loader = prepare_data(
        target_col=target_col,
        train_file=train_file,
        exclude_cols=exclude_cols,
        train_args=train_args,
        data_config=data_config,
        seed=seed, 
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        return_loaders=True,
        return_features=False
    )
    
    logger.info(f'Created datasets: {len(train_dataset)} train, {len(val_dataset)} val')

    # Base loaders use batch_size; TabNet can override with batch_size_tabnet.
    train_loader_tabnet, val_loader_tabnet = _prepare_tabnet_loaders(
        train_dataset,
        val_dataset,
        train_args,
        training_cfg,
        tabnet_model_cfg,
        train_loader,
        val_loader,
    )

    # Get input dimension from the dataset
    input_dim = full_dataset.features.shape[1]
    logger.info(f'Input dimension: {input_dim}')

    # Task config
    task = training_cfg.get('task', 'regression')
    use_log_ratio = bool(training_cfg.get('use_log_ratio', False))
    deterministic_ot_eval = bool(training_cfg.get('deterministic_ot_eval', True))
    ot_eval_seed = int(training_cfg.get('ot_eval_seed', seed if seed is not None else 42))
    head_dropout = float(training_cfg.get('head_dropout', 0.1))

    config_summary = _build_config_summary(
        target_col,
        train_file,
        seed,
        task,
        training_cfg,
        mlp_model_cfg,
        tabm_model_cfg,
        tabnet_model_cfg,
        encoder_model_cfg,
        wae_encoder_model_cfg,
        pretraining_config,
        train_args,
        exclude_cols,
    )
    logger.info("Config summary:\n%s", pformat(config_summary))
    logger.info("Deterministic OT eval: %s (seed=%s)", deterministic_ot_eval, ot_eval_seed)

    def _safe_last(values, default=float('nan')):
        if not values:
            return default
        return values[-1]

    def _is_nan(val):
        try:
            return math.isnan(float(val))
        except Exception:
            return True

    def _fmt(val):
        if _is_nan(val):
            return "nan"
        return f"{float(val):.4f}"

    def _log_model_metrics(epoch_idx, model_name, train_loss, val_loss, metric_val, auc_val=None):
        metric_label = 'F1' if task == 'classification' else 'R2'
        msg = (f'Epoch {epoch_idx + 1:4d} | {model_name}: '
               f'train={_fmt(train_loss)}, val={_fmt(val_loss)}, {metric_label}={_fmt(metric_val)}')
        if auc_val is not None and not _is_nan(auc_val):
            msg += f', AUC={_fmt(auc_val)}'
        logger.info(msg)

    def _log_component_metrics(model_name, train_comp, val_comp):
        if not train_comp or not val_comp:
            return
        msg = (
            f'{model_name} components: '
            f"train_reg={_fmt(train_comp.get('regression'))}, "
            f"train_recon={_fmt(train_comp.get('reconstruction'))}, "
            f"train_ot={_fmt(train_comp.get('regularization'))}, "
            f"val_reg={_fmt(val_comp.get('regression'))}, "
            f"val_recon={_fmt(val_comp.get('reconstruction'))}, "
            f"val_ot={_fmt(val_comp.get('regularization'))}"
        )
        logger.info(msg)

    def unpack_eval(result):
        if task == 'classification':
            if isinstance(result, tuple) and len(result) == 3:
                return result
            loss, f1 = result
            return loss, f1, float('nan')
        loss, r2 = result
        return loss, r2, float('nan')

    # Create MLP model using MLPEncoder + EncoderEmbedding
    mlp_latent_dim = mlp_model_cfg.get('latent_dim', 32)  # Default latent dim for MLP
    mlp_encoder_for_model = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=mlp_model_cfg.get('hidden_dims', [128, 64]),
        latent_dim=mlp_latent_dim,
        dropout=mlp_model_cfg.get('dropout', 0.3),
        batchnorm=mlp_model_cfg.get('batchnorm', True),
        activation=mlp_model_cfg.get('activation', 'ReLU')
    ).to(device)
    model_mlp = EncoderEmbedding(mlp_encoder_for_model, mlp_latent_dim, task=task, head_dropout=head_dropout).to(device)


    model_tabm = TabM(
        in_dim=input_dim,
        out_dim=1,
        hidden_dims=tabm_model_cfg.get('hidden_dims', [128, 128]),
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=tabm_model_cfg.get('dropout', 0.1),
        activation=tabm_model_cfg.get('activation', 'ReLU'),
        task=task,
    ).to(device)
    
    # Create TabNet model
    if tabnet_model_cfg is None:
        tabnet_model_cfg = {}
    # Get sparsity loss weight (default: 0.0001 if not specified)
    lambda_sparse = float(tabnet_model_cfg.get('lambda_sparse', 0.0001))
    group_categorical = get_config_value(data_config, 'data', 'group_categorical', default=True)
    tabnet_cat_idxs = tabnet_model_cfg.get('cat_idxs', []) or []
    tabnet_cat_dims = tabnet_model_cfg.get('cat_dims', []) or []
    tabnet_cat_emb_dim = tabnet_model_cfg.get('cat_emb_dim', 1)
    if not group_categorical:
        tabnet_cat_idxs = []
        tabnet_cat_dims = []
    elif not tabnet_cat_idxs and not tabnet_cat_dims:
        # Auto-detect categorical features only when they are integer-encoded (not one-hot)
        cat_idxs = getattr(full_dataset, 'categorical_indices', []) or []
        if cat_idxs:
            features_np = full_dataset.features.detach().cpu().numpy()
            cat_dims = []
            non_binary_found = False
            integer_only = True
            for idx in cat_idxs:
                col = features_np[:, idx]
                if not np.all(np.isfinite(col)):
                    integer_only = False
                    break
                if not np.allclose(col, np.round(col), atol=1e-4):
                    integer_only = False
                    break
                uniq = np.unique(col)
                if len(uniq) > 2:
                    non_binary_found = True
                cat_dims.append(len(uniq))
            if integer_only and non_binary_found:
                tabnet_cat_idxs = cat_idxs
                tabnet_cat_dims = cat_dims
                print(f'Auto-detected TabNet categorical features: {len(tabnet_cat_idxs)}')
    model_tabnet = TabNet(
        input_dim=input_dim,
        output_dim=1,
        n_d=tabnet_model_cfg.get('n_d', 8),
        n_a=tabnet_model_cfg.get('n_a', 8),
        n_steps=tabnet_model_cfg.get('n_steps', 3),
        gamma=tabnet_model_cfg.get('gamma', 1.5),
        n_independent=tabnet_model_cfg.get('n_independent', 2),
        n_shared=tabnet_model_cfg.get('n_shared', 2),
        epsilon=tabnet_model_cfg.get('epsilon', 1e-15),
        virtual_batch_size=tabnet_model_cfg.get('virtual_batch_size', 128),
        momentum=tabnet_model_cfg.get('momentum', 0.02),
        mask_type=tabnet_model_cfg.get('mask_type', 'sparsemax'),
        cat_idxs=tabnet_cat_idxs,
        cat_dims=tabnet_cat_dims,
        cat_emb_dim=tabnet_cat_emb_dim,
        return_entropy=True,  # Always enable entropy return for sparsity loss
        task=task,
    ).to(device)
    logger.info(f'TabNet sparsity loss enabled with lambda_sparse={lambda_sparse}')
    
    # # Create TabPFN model
    # if tabpfn_model_cfg is None:
    #     tabpfn_model_cfg = {}
    # # Determine device string for TabPFN
    # tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tabpfn_device = tabpfn_model_cfg.get('device', tabpfn_device)
    # model_tabpfn = TabPFN(
    #     input_dim=input_dim,
    #     output_dim=1,
    #     device=tabpfn_device,
    #     only_inference=tabpfn_model_cfg.get('only_inference', False),
    #     base_path=tabpfn_model_cfg.get('base_path', None)
    # ).to(device)
    
    # Initialize loss function and loss tracker
    if task == 'classification':
        loss_fn = nn.BCELoss()
        if use_log_ratio:
            print('Warning: use_log_ratio is ignored for classification tasks')
            use_log_ratio = False
    else:
        loss_fn = MSELoss()  # For regression models
    metric_name = 'F1' if task == 'classification' else 'R2'
    loss_tracker = LossTracker(metric_name=metric_name) 
    
    # # Fit TabPFN model on training data (TabPFN uses fit() method instead of gradient descent)
    # if not tabpfn_model_cfg.get('only_inference', False):
    #     print("Fitting TabPFN model on training data...")
    #     # Get training data as numpy arrays
    #     # Note: train_dataset is a Subset, so we need to get indices first
    #     train_indices = train_dataset.indices
    #     X_train_np = full_dataset.features[train_indices].numpy()
    #     y_train_np = full_dataset.label[train_indices].numpy()
    #     model_tabpfn.fit(X_train_np, y_train_np)
    #     print("TabPFN model fitted successfully.")
    #     
    #     # Evaluate TabPFN once (since it doesn't need training, performance is constant)
    #     # This is done before the training loop to avoid redundant evaluations
    #     print("\n" + "="*60)
    #     print("Evaluating TabPFN model (pre-trained, no training needed)...")
    #     print("="*60)
    #     train_loss_tabpfn, train_r2_tabpfn = eval_loop(model_tabpfn, train_loader, loss_fn, device, model_type='tabpfn', use_log_ratio=use_log_ratio)
    #     test_loss_tabpfn, test_r2_tabpfn = eval_loop(model_tabpfn, val_loader, loss_fn, device, model_type='tabpfn', use_log_ratio=use_log_ratio)
    #     # Record TabPFN results at epoch 0 (before training loop starts)
    #     # We'll mark it as evaluated once, and skip it in the training loop
    #     loss_tracker.update(0, 'tabpfn', train_loss_tabpfn, test_loss_tabpfn, test_r2_tabpfn)
    #     print(f"TabPFN Results - Train Loss: {train_loss_tabpfn:.4f}, Test Loss: {test_loss_tabpfn:.4f}, Test R2: {test_r2_tabpfn:.4f}")
    #     print("="*60 + "\n")

    encoder_latent_dim = encoder_model_cfg.get('latent_dim', 32)
    tabnet_latent_dim = tabnet_model_cfg.get('latent_dim', encoder_latent_dim)
    encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
    encoder_dropout = encoder_model_cfg.get('dropout', 0.1)
    encoder_batchnorm = encoder_model_cfg.get('batchnorm', True)
    encoder_activation = encoder_model_cfg.get('activation', 'ReLU')

    # Optional TabM encoder overrides (fallback to shared encoder settings)
    tabm_encoder_base_cfg = encoder_model_cfg.copy()
    tabm_encoder_base_cfg.update(tabm_encoder_model_cfg)
    tabm_encoder_latent_dim = tabm_encoder_base_cfg.get('latent_dim', encoder_latent_dim)
    tabm_encoder_hidden_dims = tabm_encoder_base_cfg.get('hidden_dims', encoder_hidden_dims)
    tabm_encoder_dropout = tabm_encoder_base_cfg.get('dropout', encoder_dropout)
    tabm_encoder_batchnorm = tabm_encoder_base_cfg.get('batchnorm', encoder_batchnorm)
    tabm_encoder_activation = tabm_encoder_base_cfg.get('activation', encoder_activation)
    
    # Create MLP Encoder for TabAE
    mlp_encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=encoder_hidden_dims,
        latent_dim=encoder_latent_dim,
        dropout=encoder_dropout,
        batchnorm=encoder_batchnorm,
        activation=encoder_activation
    ).to(device)
    
    # Create base Tabular Decoder template (reverse of encoder hidden dims)

    decoder_hidden_dims = list(reversed(encoder_hidden_dims))
    base_decoder = TabularDecoder(
        latent_dim=encoder_latent_dim,
        hidden_dims=decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=encoder_dropout
    ).to(device)
    tabm_decoder_hidden_dims = list(reversed(tabm_encoder_hidden_dims))
    tabm_base_decoder = TabularDecoder(
        latent_dim=tabm_encoder_latent_dim,
        hidden_dims=tabm_decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=tabm_encoder_dropout
    ).to(device)
    tabnet_base_decoder = TabularDecoder(
        latent_dim=tabnet_latent_dim,
        hidden_dims=decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=encoder_dropout
    ).to(device)
    
    # Create TabAE model for pre-training (MLP encoder) with independent decoder
    mlp_decoder = copy.deepcopy(base_decoder)
    mlp_tabae = TabAE(encoder=mlp_encoder, decoder=mlp_decoder).to(device)
    

    tabm_encoder = TabMEncoder(
        input_dim=input_dim,
        hidden_dims=tabm_encoder_hidden_dims,
        latent_dim=tabm_encoder_latent_dim,
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=tabm_encoder_dropout,
        batchnorm=tabm_encoder_batchnorm,
        activation=tabm_encoder_activation,
    ).to(device)
    
    # Create TabAE model for pre-training (TabM encoder) with independent decoder
    tabm_decoder = copy.deepcopy(tabm_base_decoder)
    tabm_tabae = TabAE(encoder=tabm_encoder, decoder=tabm_decoder).to(device)
    
    # Create TabNet Encoder for pre-training
    tabnet_encoder = TabNetEncoder(
        input_dim=input_dim,
        latent_dim=tabnet_latent_dim,
        n_d=tabnet_model_cfg.get('n_d', 8),
        n_a=tabnet_model_cfg.get('n_a', 8),
        n_steps=tabnet_model_cfg.get('n_steps', 3),
        gamma=tabnet_model_cfg.get('gamma', 1.5),
        dropout=encoder_dropout,
        n_independent=tabnet_model_cfg.get('n_independent', 2),
        n_shared=tabnet_model_cfg.get('n_shared', 2),
        epsilon=tabnet_model_cfg.get('epsilon', 1e-15),
        virtual_batch_size=tabnet_model_cfg.get('virtual_batch_size', 128),
        momentum=tabnet_model_cfg.get('momentum', 0.02),
        mask_type=tabnet_model_cfg.get('mask_type', 'sparsemax'),
        return_entropy=True  # Enable entropy return for sparsity loss
    ).to(device)
    
    # Create TabAE model for pre-training (TabNet encoder) with independent decoder
    tabnet_decoder = copy.deepcopy(tabnet_base_decoder)
    tabnet_tabae = TabAE(encoder=tabnet_encoder, decoder=tabnet_decoder).to(device)
    
    # # Create TabPFN Encoder
    # tabpfn_encoder = TabPFNEncoder(
    #     input_dim=input_dim,
    #     latent_dim=encoder_latent_dim,  
    #     device=tabpfn_device,
    #     base_path=tabpfn_model_cfg.get('base_path', None)
    # ).to(device)
    # 
    # # Fit TabPFN encoder with training data (needed for few-shot learning)
    # if not tabpfn_model_cfg.get('only_inference', False):
    #     print("Fitting TabPFN encoder with training data...")
    #     train_indices = train_dataset.indices
    #     X_train_np = full_dataset.features[train_indices].numpy()
    #     y_train_np = full_dataset.label[train_indices].numpy()
    #     tabpfn_encoder.fit(X_train_np, y_train_np)
    #     
    #     # Get embedding dimension by doing a forward pass with a dummy sample
    #     # This is needed to know the actual embedding dimension for EncoderEmbedding
    #     with torch.no_grad():
    #         dummy_input = torch.zeros(1, input_dim, device=device)
    #         _ = tabpfn_encoder(dummy_input)  # Forward pass to cache embedding_dim
    #         tabpfn_embedding_dim = tabpfn_encoder.embedding_dim  # Use property instead of shape
    #     print(f"TabPFN embedding dimension: {tabpfn_embedding_dim}")
    # else:
    #     # If only_inference, we'll determine embedding_dim later
    #     tabpfn_embedding_dim = None

    # Create TabVAE models with independent decoders
    mlp_vae_encoder = copy.deepcopy(mlp_encoder)
    mlp_vae_decoder = copy.deepcopy(base_decoder)
    mlp_tabvae = TabVAE(encoder=mlp_vae_encoder, decoder=mlp_vae_decoder, latent_dim=encoder_latent_dim).to(device)
    

    tabm_vae_encoder = copy.deepcopy(tabm_encoder)
    tabm_vae_decoder = copy.deepcopy(tabm_base_decoder)
    tabm_tabvae = TabVAE(encoder=tabm_vae_encoder, decoder=tabm_vae_decoder, latent_dim=tabm_encoder_latent_dim).to(device)
    
    # Create TabVAE model for pre-training (TabNet encoder)
    tabnet_vae_encoder = copy.deepcopy(tabnet_encoder)
    tabnet_vae_decoder = copy.deepcopy(tabnet_base_decoder)
    tabnet_tabvae = TabVAE(encoder=tabnet_vae_encoder, decoder=tabnet_vae_decoder, latent_dim=tabnet_latent_dim).to(device)

    # Get WAE-specific config (allow overrides beyond latent_dim)
    if wae_encoder_model_cfg is None:
        wae_encoder_model_cfg = {}
    wae_latent_dim = wae_encoder_model_cfg.get('latent_dim', encoder_latent_dim)
    wae_hidden_dims = wae_encoder_model_cfg.get('hidden_dims', encoder_hidden_dims)
    wae_dropout = wae_encoder_model_cfg.get('dropout', encoder_dropout)
    wae_batchnorm = wae_encoder_model_cfg.get('batchnorm', encoder_batchnorm)
    wae_activation = wae_encoder_model_cfg.get('activation', encoder_activation)
    wae_decoder_hidden_dims = list(reversed(wae_hidden_dims))
    
    # TabM WAE encoder config (merge TabM overrides + WAE overrides)
    tabm_wae_base_cfg = encoder_model_cfg.copy()
    tabm_wae_base_cfg.update(tabm_encoder_model_cfg)
    tabm_wae_base_cfg.update(wae_encoder_model_cfg)
    tabm_wae_latent_dim = tabm_wae_base_cfg.get('latent_dim', tabm_encoder_latent_dim)
    tabm_wae_hidden_dims = tabm_wae_base_cfg.get('hidden_dims', tabm_encoder_hidden_dims)
    tabm_wae_dropout = tabm_wae_base_cfg.get('dropout', tabm_encoder_dropout)
    tabm_wae_batchnorm = tabm_wae_base_cfg.get('batchnorm', tabm_encoder_batchnorm)
    tabm_wae_activation = tabm_wae_base_cfg.get('activation', tabm_encoder_activation)
    tabm_wae_decoder_hidden_dims = list(reversed(tabm_wae_hidden_dims))

    tabnet_wae_latent_dim = wae_encoder_model_cfg.get('latent_dim', tabnet_latent_dim)
    tabnet_wae_dropout = wae_encoder_model_cfg.get('dropout', encoder_dropout)

    # Create MLP Encoder for TabWAE (independent instance, WAE-specific cfg)
    mlp_wae_encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=wae_hidden_dims,
        latent_dim=wae_latent_dim,
        dropout=wae_dropout,
        batchnorm=wae_batchnorm,
        activation=wae_activation,
    ).to(device)
    mlp_wae_decoder = TabularDecoder(
        latent_dim=wae_latent_dim,
        hidden_dims=wae_decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=wae_dropout,
    ).to(device)
    mlp_tabwae = TabAE(encoder=mlp_wae_encoder, decoder=mlp_wae_decoder).to(device)

    # Create TabM Encoder for TabWAE (independent instance, WAE-specific cfg)
    tabm_wae_encoder = TabMEncoder(
        input_dim=input_dim,
        hidden_dims=tabm_wae_hidden_dims,
        latent_dim=tabm_wae_latent_dim,
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=tabm_wae_dropout,
        batchnorm=tabm_wae_batchnorm,
        activation=tabm_wae_activation,
    ).to(device)
    tabm_wae_decoder = TabularDecoder(
        latent_dim=tabm_wae_latent_dim,
        hidden_dims=tabm_wae_decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=tabm_wae_dropout,
    ).to(device)
    tabm_tabwae = TabAE(encoder=tabm_wae_encoder, decoder=tabm_wae_decoder).to(device)

    # Create TabNet Encoder for TabWAE (independent instance, WAE-specific latent/dropout)
    tabnet_wae_encoder = TabNetEncoder(
        input_dim=input_dim,
        latent_dim=tabnet_wae_latent_dim,
        n_d=tabnet_model_cfg.get('n_d', 8),
        n_a=tabnet_model_cfg.get('n_a', 8),
        n_steps=tabnet_model_cfg.get('n_steps', 3),
        gamma=tabnet_model_cfg.get('gamma', 1.5),
        dropout=tabnet_wae_dropout,
        n_independent=tabnet_model_cfg.get('n_independent', 2),
        n_shared=tabnet_model_cfg.get('n_shared', 2),
        epsilon=tabnet_model_cfg.get('epsilon', 1e-15),
        virtual_batch_size=tabnet_model_cfg.get('virtual_batch_size', 128),
        momentum=tabnet_model_cfg.get('momentum', 0.02),
        mask_type=tabnet_model_cfg.get('mask_type', 'sparsemax'),
        return_entropy=True,
    ).to(device)
    tabnet_wae_decoder = TabularDecoder(
        latent_dim=tabnet_wae_latent_dim,
        hidden_dims=wae_decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=wae_dropout,
    ).to(device)
    tabnet_tabwae = TabAE(encoder=tabnet_wae_encoder, decoder=tabnet_wae_decoder).to(device)
    # Create TabGW models (for Riemann metric learning with GW loss)
    # GW case: no pretraining, directly train encoder + regression head with GW loss
    mlp_gw_encoder = copy.deepcopy(mlp_encoder)
    tabm_gw_encoder = copy.deepcopy(tabm_encoder)
    
    # Create GW embedding models (encoder + regression head, no decoder)
    mlp_gw_model = EncoderEmbedding(mlp_gw_encoder, encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_gw_model = EncoderEmbedding(tabm_gw_encoder, tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_gw_encoder = copy.deepcopy(tabnet_encoder)
    tabnet_gw_model = EncoderEmbedding(tabnet_gw_encoder, tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # # TabPFN GW: use actual embedding dimension (determined dynamically)
    # tabpfn_gw_encoder = copy.deepcopy(tabpfn_encoder)
    # if tabpfn_embedding_dim is None:
    #     # Determine embedding dimension if not already done
    #     with torch.no_grad():
    #         dummy_input = torch.zeros(1, input_dim, device=device)
    #         _ = tabpfn_gw_encoder(dummy_input)  # Forward pass to cache embedding_dim
    #         tabpfn_embedding_dim = tabpfn_gw_encoder.embedding_dim  # Use property instead of shape
    # tabpfn_gw_model = EncoderEmbedding(tabpfn_gw_encoder, tabpfn_embedding_dim).to(device)
    
    # Store base decoder for multi-task models (each model gets its own copy to avoid weight sharing)
    
    # WAE: No regression stage needed - encoder is only trained for reconstruction (latent space)
    # We'll extract encoder directly from pretrained TabWAE for frozen/finetuned versions

    lr = float(training_cfg.get('learning_rate', 1e-3))
    # Allow model-specific learning rates from config
    lr_pretrain = float(training_cfg.get('lr_pretrain', lr))
    lr_pretrain_tabm = float(training_cfg.get('lr_pretrain_tabm', lr_pretrain))
    lr_tabnet = float(training_cfg.get('lr_tabnet', lr))
    lr_tabm = float(training_cfg.get('lr_tabm', lr))
    lr_gw = float(training_cfg.get('lr_gw', lr))
    # Allow separate learning rates for embedding and multi-task models
    lr_embedding = float(training_cfg.get('lr_embedding', lr))
    lr_embedding_tabm = float(training_cfg.get('lr_embedding_tabm', lr_tabm))
    lr_embedding_frozen_tabm = float(training_cfg.get('lr_embedding_frozen_tabm', lr_tabm))  
    lr_embedding_tabnet = float(training_cfg.get('lr_embedding_tabnet', lr_tabnet))
    # Fine-tuning: split encoder/head learning rates (smaller encoder LR by default)
    lr_embedding_encoder = float(training_cfg.get('lr_embedding_encoder', lr_embedding * 0.1))
    lr_embedding_encoder_tabm = float(training_cfg.get('lr_embedding_encoder_tabm', lr_embedding_tabm * 0.1))
    lr_embedding_encoder_tabnet = float(training_cfg.get('lr_embedding_encoder_tabnet', lr_embedding_tabnet * 0.1))
    lr_multitask = float(training_cfg.get('lr_multitask', lr))
    lr_multitask_tabm = float(training_cfg.get('lr_multitask_tabm', lr_tabm))
    lr_multitask_tabnet = float(training_cfg.get('lr_multitask_tabnet', lr_tabnet))
    lr_multitask_wae = float(training_cfg.get('lr_multitask_wae', lr_multitask * 0.5))  
    weight_decay = float(training_cfg.get('weight_decay', 1e-5))
    weight_decay_pretrain_tabm = float(training_cfg.get('weight_decay_pretrain_tabm', weight_decay))
    
    
    
    # Create TabularLoss for encoder models
    loss_weights_cfg = get_config_value(data_config, 'data', 'loss_weights', default={})
    weight_denominator = loss_weights_cfg.get('weight_denominator', 25)
    w_bin = loss_weights_cfg.get('w_bin', None)  
    w_cat = loss_weights_cfg.get('w_cat', None)  
    
    tabular_loss = TabularLoss(
        w_bin=w_bin, 
        w_cat=w_cat, 
        weight_denominator=weight_denominator,
        cat_sizes=full_dataset.cat_sizes
    )
    logger.info(f'Loss weights: w_bin={tabular_loss.w_bin:.6f}, w_cat={tabular_loss.w_cat:.6f} (denominator={weight_denominator})')

    # WAE configuration: try pretraining config first, then training_cfg, then defaults
    if pretraining_config:
        wae_config = pretraining_config.get('wae', {})
        lambda_ot = wae_config.get('lambda_ot', training_cfg.get('lambda_ot', 1.0))
        regularization_type = wae_config.get('regularization_type', training_cfg.get('wae_regularization_type', 'sinkhorn'))
        sinkhorn_eps = wae_config.get('sinkhorn_eps', training_cfg.get('sinkhorn_eps', 0.1))
        sinkhorn_max_iter = wae_config.get('sinkhorn_max_iter', training_cfg.get('sinkhorn_max_iter', 10))
        mmd_kernel_mul = wae_config.get('mmd_kernel_mul', training_cfg.get('mmd_kernel_mul', 2))
        mmd_kernel_num = wae_config.get('mmd_kernel_num', training_cfg.get('mmd_kernel_num', 5))
        
        # VAE beta configuration
        vae_config = pretraining_config.get('vae', {})
        vae_beta = vae_config.get('beta', training_cfg.get('vae_beta', 1.0))
    else:
        # Fallback to training_cfg (backward compatibility)
        lambda_ot = training_cfg.get('lambda_ot', training_cfg.get('lambda_mmd', 1.0))
        regularization_type = training_cfg.get('wae_regularization_type', 'sinkhorn')
        sinkhorn_eps = training_cfg.get('sinkhorn_eps', 0.1)
        sinkhorn_max_iter = training_cfg.get('sinkhorn_max_iter', 10)
        mmd_kernel_mul = training_cfg.get('mmd_kernel_mul', 2)
        mmd_kernel_num = training_cfg.get('mmd_kernel_num', 5)
        vae_beta = training_cfg.get('vae_beta', 1.0)

    # Get normalization flag for GW loss
    apply_normalization = training_cfg.get('apply_normalization', False)
    
    

    # Create optimizers
    optimizer_mlp = Adam(model_mlp.parameters(), lr=lr)
    optimizer_tabm = Adam(model_tabm.parameters(), lr=lr_tabm)
    optimizer_tabnet = Adam(model_tabnet.parameters(), lr=lr_tabnet)
    optimizer_mlp_tabae = Adam(mlp_tabae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    optimizer_tabm_tabae = Adam(tabm_tabae.parameters(), lr=lr_pretrain_tabm, weight_decay=weight_decay_pretrain_tabm)
    optimizer_tabnet_tabae = Adam(tabnet_tabae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    optimizer_mlp_tabvae = Adam(mlp_tabvae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    optimizer_tabm_tabvae = Adam(tabm_tabvae.parameters(), lr=lr_pretrain_tabm, weight_decay=weight_decay_pretrain_tabm)
    optimizer_tabnet_tabvae = Adam(tabnet_tabvae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    optimizer_mlp_tabwae = Adam(mlp_tabwae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    optimizer_tabm_tabwae = Adam(tabm_tabwae.parameters(), lr=lr_pretrain_tabm, weight_decay=weight_decay_pretrain_tabm)
    optimizer_tabnet_tabwae = Adam(tabnet_tabwae.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
    
    
    
    
    
    num_epochs = int(training_cfg.get('epochs', 10))
    
    # Pre-train all models (AE/VAE/WAE) - only pretrain, no regression stage
    # Encoders are only trained for reconstruction (latent space)
    # Note: GW models are NOT pretrained, they are trained directly with regression + GW loss
    
    # ========== TabAE Pre-training ==========
    # MLP TabAE
    pretrain_and_load_checkpoint(
        mlp_tabae, optimizer_mlp_tabae, train_tabae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'mlp_tabae', dataset=full_dataset, logger=logger
    )
    
    # TabM TabAE
    pretrain_and_load_checkpoint(
        tabm_tabae, optimizer_tabm_tabae, train_tabae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabm_tabae', dataset=full_dataset, logger=logger
    )
    
    # TabNet TabAE
    pretrain_and_load_checkpoint(
        tabnet_tabae, optimizer_tabnet_tabae, train_tabae, train_loader_tabnet, val_loader_tabnet,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabnet_tabae', dataset=full_dataset, lambda_sparse=lambda_sparse, logger=logger
    )
    
    # ========== TabVAE Pre-training ==========
    # MLP TabVAE
    pretrain_and_load_checkpoint(
        mlp_tabvae, optimizer_mlp_tabvae, train_tabvae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'mlp_tabvae', dataset=full_dataset, beta=vae_beta, logger=logger
    )
    
    # TabM TabVAE
    pretrain_and_load_checkpoint(
        tabm_tabvae, optimizer_tabm_tabvae, train_tabvae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabm_tabvae', dataset=full_dataset, beta=vae_beta, logger=logger
    )
    
    # TabNet TabVAE
    pretrain_and_load_checkpoint(
        tabnet_tabvae, optimizer_tabnet_tabvae, train_tabvae, train_loader_tabnet, val_loader_tabnet,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabnet_tabvae', dataset=full_dataset, beta=vae_beta, lambda_sparse=lambda_sparse, logger=logger
    )
    
    # ========== TabWAE Pre-training ==========
    # MLP TabWAE (with regularization)
    pretrain_and_load_checkpoint(
        mlp_tabwae, optimizer_mlp_tabwae, train_tabwae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'mlp_tabwae', dataset=full_dataset,
        lambda_ot=lambda_ot,
        regularization_type=regularization_type,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_max_iter=sinkhorn_max_iter,
        mmd_kernel_mul=mmd_kernel_mul,
        mmd_kernel_num=mmd_kernel_num,
        logger=logger,
    )
    
    # TabM TabWAE (with regularization)
    pretrain_and_load_checkpoint(
        tabm_tabwae, optimizer_tabm_tabwae, train_tabwae, train_loader, val_loader,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabm_tabwae', dataset=full_dataset,
        lambda_ot=lambda_ot,
        regularization_type=regularization_type,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_max_iter=sinkhorn_max_iter,
        mmd_kernel_mul=mmd_kernel_mul,
        mmd_kernel_num=mmd_kernel_num,
        logger=logger,
    )
    
    # TabNet TabWAE (with regularization)
    pretrain_and_load_checkpoint(
        tabnet_tabwae, optimizer_tabnet_tabwae, train_tabwae, train_loader_tabnet, val_loader_tabnet,
        tabular_loss, device, num_epochs, loss_tracker, experiment_dir,
        'tabnet_tabwae', dataset=full_dataset,
        lambda_ot=lambda_ot,
        regularization_type=regularization_type,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_max_iter=sinkhorn_max_iter,
        mmd_kernel_mul=mmd_kernel_mul,
        mmd_kernel_num=mmd_kernel_num,
        lambda_sparse=lambda_sparse,
        logger=logger,
    )
    
    # Note: Best checkpoints are already loaded by pretrain_and_load_checkpoint
    # Initialize GW models FROM pretrained TabAE encoder weights
    logger.info('=' * 60)
    logger.info('GW models initialized from TabAE encoders')
    logger.info('=' * 60)
    mlp_gw_model.encoder.load_state_dict(mlp_tabae.encoder.state_dict(), strict=False)
    tabm_gw_model.encoder.load_state_dict(tabm_tabae.encoder.state_dict(), strict=False)
    tabnet_gw_model.encoder.load_state_dict(tabnet_tabae.encoder.state_dict(), strict=False)
    # TabPFN GW: encoder is already fitted, no need to load from TabAE (TabPFN doesn't use TabAE)
    
    # Create GW optimizers
    optimizer_mlp_gw = Adam(mlp_gw_model.parameters(), lr=lr_gw, weight_decay=weight_decay)
    optimizer_tabm_gw = Adam(tabm_gw_model.parameters(), lr=lr_gw, weight_decay=weight_decay)
    optimizer_tabnet_gw = Adam(tabnet_gw_model.parameters(), lr=lr_gw, weight_decay=weight_decay)
    # optimizer_tabpfn_gw = Adam(tabpfn_gw_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Extract encoder bases directly from pretrained models (no regression stage)
    mlp_encoder_base = mlp_tabae.encoder
    tabm_encoder_base = tabm_tabae.encoder
    tabnet_encoder_base = tabnet_tabae.encoder
    # # TabPFN: use encoder directly (no TabAE pre-training)
    # tabpfn_encoder_base = tabpfn_encoder
    mlp_vae_encoder_base = mlp_tabvae.encoder
    tabm_vae_encoder_base = tabm_tabvae.encoder
    tabnet_vae_encoder_base = tabnet_tabvae.encoder
    mlp_wae_encoder_base = mlp_tabwae.encoder
    tabm_wae_encoder_base = tabm_tabwae.encoder
    tabnet_wae_encoder_base = tabnet_tabwae.encoder
    # Note: GW models don't need encoder_base extraction as they don't use frozen/finetuned versions

    # Create frozen and fine-tuned versions of encoders using factory function
    # Note: GW models are not included here as they don't use frozen/finetuned versions
    # Note: For TabPFN, we only create frozen version (no finetuned, no TabAE/TabWAE)
    encoder_bases = {
        'mlp': mlp_encoder_base,
        'tabm': tabm_encoder_base,
        'tabnet': tabnet_encoder_base,
        # 'tabpfn': tabpfn_encoder_base,  # Direct encoder, no TabAE
        'mlp_vae': mlp_vae_encoder_base,
        'tabm_vae': tabm_vae_encoder_base,
        'tabnet_vae': tabnet_vae_encoder_base,
        'mlp_wae': mlp_wae_encoder_base,
        'tabm_wae': tabm_wae_encoder_base,
        'tabnet_wae': tabnet_wae_encoder_base,
    }
    
    encoder_frozen = {}
    encoder_finetuned = {}
    for key, base in encoder_bases.items():
        frozen, finetuned = create_frozen_finetuned_encoders(base)
        encoder_frozen[key] = frozen
        encoder_finetuned[key] = finetuned
    
    # Create embedding models with frozen/finetuned encoders
    # For AE/WAE/GW: only copy encoder, head is new (will be trained)
    mlp_embedding_frozen = EncoderEmbedding(encoder_frozen['mlp'], encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_embedding_frozen = EncoderEmbedding(encoder_frozen['tabm'], tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_embedding_frozen = EncoderEmbedding(encoder_frozen['tabnet'], tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # # TabPFN frozen: use actual embedding dimension
    # tabpfn_embedding_frozen = EncoderEmbedding(encoder_frozen['tabpfn'], tabpfn_embedding_dim).to(device)
    mlp_wae_embedding_frozen = EncoderEmbedding(encoder_frozen['mlp_wae'], wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_wae_embedding_frozen = EncoderEmbedding(encoder_frozen['tabm_wae'], tabm_wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_wae_embedding_frozen = EncoderEmbedding(encoder_frozen['tabnet_wae'], tabnet_wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # GW models: no frozen/finetuned versions, train directly
    mlp_embedding_finetuned = EncoderEmbedding(encoder_finetuned['mlp'], encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_embedding_finetuned = EncoderEmbedding(encoder_finetuned['tabm'], tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_embedding_finetuned = EncoderEmbedding(encoder_finetuned['tabnet'], tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # TabPFN: no finetuned version (only frozen and gw)
    mlp_wae_embedding_finetuned = EncoderEmbedding(encoder_finetuned['mlp_wae'], wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_wae_embedding_finetuned = EncoderEmbedding(encoder_finetuned['tabm_wae'], tabm_wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_wae_embedding_finetuned = EncoderEmbedding(encoder_finetuned['tabnet_wae'], tabnet_wae_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # GW models: no frozen/finetuned versions, train directly
    
    # For VAE: copy encoder (from regression model) + VAE layers (from pretrained TabVAE)
    # Head is new (will be trained from scratch in frozen/finetuned versions)
    # Note: VAE layers (mu, log_var) were trained during pretraining, not during regression
    mlp_vae_embedding_frozen = VAEEncoderEmbedding(encoder_frozen['mlp_vae'], encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(mlp_vae_embedding_frozen, mlp_tabvae)
    
    tabm_vae_embedding_frozen = VAEEncoderEmbedding(encoder_frozen['tabm_vae'], tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(tabm_vae_embedding_frozen, tabm_tabvae)
    
    tabnet_vae_embedding_frozen = VAEEncoderEmbedding(encoder_frozen['tabnet_vae'], tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(tabnet_vae_embedding_frozen, tabnet_tabvae)
    
    mlp_vae_embedding_finetuned = VAEEncoderEmbedding(encoder_finetuned['mlp_vae'], encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(mlp_vae_embedding_finetuned, mlp_tabvae)
    
    tabm_vae_embedding_finetuned = VAEEncoderEmbedding(encoder_finetuned['tabm_vae'], tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(tabm_vae_embedding_finetuned, tabm_tabvae)
    
    tabnet_vae_embedding_finetuned = VAEEncoderEmbedding(encoder_finetuned['tabnet_vae'], tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    load_vae_layers_from_tabvae(tabnet_vae_embedding_finetuned, tabnet_tabvae)

    # Multi-task models: train encoder, decoder, and head simultaneously
    mlp_multi_task_embedding_model = MultiTaskEncoderEmbedding(mlp_encoder_base, copy.deepcopy(base_decoder), encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_multi_task_embedding_model = MultiTaskEncoderEmbedding(tabm_encoder_base, copy.deepcopy(tabm_base_decoder), tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_multi_task_embedding_model = MultiTaskEncoderEmbedding(tabnet_encoder_base, copy.deepcopy(tabnet_base_decoder), tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    # TabPFN: no multi-task version (only frozen and gw)
    mlp_multi_vae_task_embedding_model = MultitaskVAEEncoderEmbedding(mlp_vae_encoder_base, copy.deepcopy(base_decoder), encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabm_multi_vae_task_embedding_model = MultitaskVAEEncoderEmbedding(tabm_vae_encoder_base, copy.deepcopy(tabm_base_decoder), tabm_encoder_latent_dim, task=task, head_dropout=head_dropout).to(device)
    tabnet_multi_vae_task_embedding_model = MultitaskVAEEncoderEmbedding(tabnet_vae_encoder_base, copy.deepcopy(tabnet_base_decoder), tabnet_latent_dim, task=task, head_dropout=head_dropout).to(device)
    mlp_multi_wae_task_embedding_model = MultiTaskEncoderEmbedding(
        mlp_wae_encoder_base, copy.deepcopy(mlp_wae_decoder), wae_latent_dim, task=task, head_dropout=head_dropout
    ).to(device)
    tabm_multi_wae_task_embedding_model = MultiTaskEncoderEmbedding(
        tabm_wae_encoder_base, copy.deepcopy(tabm_wae_decoder), tabm_wae_latent_dim, task=task, head_dropout=head_dropout
    ).to(device)
    tabnet_multi_wae_task_embedding_model = MultiTaskEncoderEmbedding(
        tabnet_wae_encoder_base, copy.deepcopy(tabnet_wae_decoder), tabnet_wae_latent_dim, task=task, head_dropout=head_dropout
    ).to(device)
    # GW models: no multi-task versions, train directly with regression + GW loss

    # Create optimizers for frozen embedding models (only optimize head)
    optimizer_mlp_embedding_frozen = Adam(mlp_embedding_frozen.head.parameters(), lr=lr_embedding)
    optimizer_tabm_embedding_frozen = Adam(tabm_embedding_frozen.head.parameters(), lr=lr_embedding_frozen_tabm)
    optimizer_tabnet_embedding_frozen = Adam(tabnet_embedding_frozen.head.parameters(), lr=lr_embedding_tabnet)
    # optimizer_tabpfn_embedding_frozen = Adam(tabpfn_embedding_frozen.head.parameters(), lr=lr_embedding)
    optimizer_mlp_vae_embedding_frozen = Adam(mlp_vae_embedding_frozen.head.parameters(), lr=lr_embedding)
    optimizer_tabm_vae_embedding_frozen = Adam(tabm_vae_embedding_frozen.head.parameters(), lr=lr_embedding_frozen_tabm)
    optimizer_tabnet_vae_embedding_frozen = Adam(tabnet_vae_embedding_frozen.head.parameters(), lr=lr_embedding_tabnet)
    optimizer_mlp_wae_embedding_frozen = Adam(mlp_wae_embedding_frozen.head.parameters(), lr=lr_embedding)
    optimizer_tabm_wae_embedding_frozen = Adam(tabm_wae_embedding_frozen.head.parameters(), lr=lr_embedding_frozen_tabm)
    optimizer_tabnet_wae_embedding_frozen = Adam(tabnet_wae_embedding_frozen.head.parameters(), lr=lr_embedding_tabnet)
    
    # Create optimizers for fine-tuned embedding models (split encoder/head learning rates)
    def make_finetune_optimizer(model, lr_encoder, lr_head):
        head_params = list(model.head.parameters())
        if hasattr(model, 'mu'):
            head_params += list(model.mu.parameters())
        if hasattr(model, 'log_var'):
            head_params += list(model.log_var.parameters())
        return Adam(
            [
                {'params': model.encoder.parameters(), 'lr': lr_encoder},
                {'params': head_params, 'lr': lr_head},
            ]
        )

    optimizer_mlp_embedding_finetuned = make_finetune_optimizer(
        mlp_embedding_finetuned, lr_embedding_encoder, lr_embedding
    )
    optimizer_tabm_embedding_finetuned = make_finetune_optimizer(
        tabm_embedding_finetuned, lr_embedding_encoder_tabm, lr_embedding_tabm
    )
    optimizer_tabnet_embedding_finetuned = make_finetune_optimizer(
        tabnet_embedding_finetuned, lr_embedding_encoder_tabnet, lr_embedding_tabnet
    )
    # TabPFN: no finetuned version (only frozen and gw)
    optimizer_mlp_vae_embedding_finetuned = make_finetune_optimizer(
        mlp_vae_embedding_finetuned, lr_embedding_encoder, lr_embedding
    )
    optimizer_tabm_vae_embedding_finetuned = make_finetune_optimizer(
        tabm_vae_embedding_finetuned, lr_embedding_encoder_tabm, lr_embedding_tabm
    )
    optimizer_tabnet_vae_embedding_finetuned = make_finetune_optimizer(
        tabnet_vae_embedding_finetuned, lr_embedding_encoder_tabnet, lr_embedding_tabnet
    )
    optimizer_mlp_wae_embedding_finetuned = make_finetune_optimizer(
        mlp_wae_embedding_finetuned, lr_embedding_encoder, lr_embedding
    )
    optimizer_tabm_wae_embedding_finetuned = make_finetune_optimizer(
        tabm_wae_embedding_finetuned, lr_embedding_encoder_tabm, lr_embedding_tabm
    )
    optimizer_tabnet_wae_embedding_finetuned = make_finetune_optimizer(
        tabnet_wae_embedding_finetuned, lr_embedding_encoder_tabnet, lr_embedding_tabnet
    )
    
    # Create optimizers for multi-task models
    optimizer_mlp_multi_task_embedding = Adam(mlp_multi_task_embedding_model.parameters(), lr=lr_multitask)
    optimizer_tabm_multi_task_embedding = Adam(tabm_multi_task_embedding_model.parameters(), lr=lr_multitask_tabm)
    optimizer_tabnet_multi_task_embedding = Adam(tabnet_multi_task_embedding_model.parameters(), lr=lr_multitask_tabnet)
    # TabPFN: no multi-task version (only frozen and gw)
    optimizer_mlp_multi_vae_task_embedding = Adam(mlp_multi_vae_task_embedding_model.parameters(), lr=lr_multitask)
    optimizer_tabm_multi_vae_task_embedding = Adam(tabm_multi_vae_task_embedding_model.parameters(), lr=lr_multitask_tabm)
    optimizer_tabnet_multi_vae_task_embedding = Adam(tabnet_multi_vae_task_embedding_model.parameters(), lr=lr_multitask_tabnet)
    optimizer_mlp_multi_wae_task_embedding = Adam(mlp_multi_wae_task_embedding_model.parameters(), lr=lr_multitask_wae)
    optimizer_tabm_multi_wae_task_embedding = Adam(tabm_multi_wae_task_embedding_model.parameters(), lr=lr_multitask_wae)
    optimizer_tabnet_multi_wae_task_embedding = Adam(tabnet_multi_wae_task_embedding_model.parameters(), lr=lr_multitask_wae)
    
    # Get GW weight and warmup from config
    gw_weight = float(training_cfg.get('gw_weight', 1.0))
    gw_epsilon = float(training_cfg.get('gw_epsilon', 1.0))
    gw_warmup_epochs = int(training_cfg.get('gw_warmup_epochs', 0))
    gw_detach_transport = bool(training_cfg.get('gw_detach_transport', True))
    gw_ramp_epochs = 10  # epochs to linearly ramp gw_weight from 0 to gw_weight after warmup

    # Multi-task reconstruction weight (no warmup)
    multitask_recon_weight = float(training_cfg.get('multitask_recon_weight', training_cfg.get('multitask_recon_weight_finetune', 0.3)))

    
    # train
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Effective GW weight: 0 during warmup, then linear ramp to gw_weight
        if gw_warmup_epochs > 0 and epoch < gw_warmup_epochs:
            gw_weight_eff = 0.0
        elif gw_warmup_epochs > 0 and epoch >= gw_warmup_epochs:
            ramp_progress = min(1.0, (epoch - gw_warmup_epochs + 1) / gw_ramp_epochs)
            gw_weight_eff = gw_weight * ramp_progress
        else:
            gw_weight_eff = gw_weight
        
        # ========== Base Models ==========
        # MLP
        train_loss_mlp = train_loop(model_mlp, train_loader, optimizer_mlp, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp, test_r2_mlp, test_auc = unpack_eval(eval_loop(model_mlp, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp', train_loss_mlp, test_loss_mlp, test_r2_mlp, val_auc=test_auc)
        
        # TabM
        train_loss_tabm = train_loop(model_tabm, train_loader, optimizer_tabm, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm, test_r2_tabm, test_auc = unpack_eval(eval_loop(model_tabm, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm', train_loss_tabm, test_loss_tabm, test_r2_tabm, val_auc=test_auc)
        
        # TabNet
        train_loss_tabnet = train_loop(model_tabnet, train_loader_tabnet, optimizer_tabnet, loss_fn, device, 
                                        model_type='tabnet', use_log_ratio=use_log_ratio, 
                                        lambda_sparse=lambda_sparse)
        test_loss_tabnet, test_r2_tabnet, test_auc = unpack_eval(eval_loop(model_tabnet, val_loader_tabnet, loss_fn, device, 
                                                    model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet', train_loss_tabnet, test_loss_tabnet, test_r2_tabnet, val_auc=test_auc)
        
        # ========== Frozen Embedding Models ==========
        # MLP Embedding Frozen
        train_loss_mlp_embedding_frozen = train_loop(mlp_embedding_frozen, train_loader, optimizer_mlp_embedding_frozen, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_embedding_frozen, test_r2_mlp_embedding_frozen, test_auc = unpack_eval(eval_loop(mlp_embedding_frozen, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_embedding_frozen', train_loss_mlp_embedding_frozen, test_loss_mlp_embedding_frozen, test_r2_mlp_embedding_frozen, val_auc=test_auc)
        
        # TabM Embedding Frozen
        train_loss_tabm_embedding_frozen = train_loop(tabm_embedding_frozen, train_loader, optimizer_tabm_embedding_frozen, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_embedding_frozen, test_r2_tabm_embedding_frozen, test_auc = unpack_eval(eval_loop(tabm_embedding_frozen, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_embedding_frozen', train_loss_tabm_embedding_frozen, test_loss_tabm_embedding_frozen, test_r2_tabm_embedding_frozen, val_auc=test_auc)
        
        # TabNet Embedding Frozen
        train_loss_tabnet_embedding_frozen = train_loop(tabnet_embedding_frozen, train_loader_tabnet, optimizer_tabnet_embedding_frozen, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_embedding_frozen, test_r2_tabnet_embedding_frozen, test_auc = unpack_eval(eval_loop(tabnet_embedding_frozen, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_embedding_frozen', train_loss_tabnet_embedding_frozen, test_loss_tabnet_embedding_frozen, test_r2_tabnet_embedding_frozen, val_auc=test_auc)
        
        # # TabPFN Embedding Frozen
        # train_loss_tabpfn_embedding_frozen = train_loop(tabpfn_embedding_frozen, train_loader, optimizer_tabpfn_embedding_frozen, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        # test_loss_tabpfn_embedding_frozen, test_r2_tabpfn_embedding_frozen = eval_loop(tabpfn_embedding_frozen, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        # loss_tracker.update(epoch, 'tabpfn_embedding_frozen', train_loss_tabpfn_embedding_frozen, test_loss_tabpfn_embedding_frozen, test_r2_tabpfn_embedding_frozen)
        
        # MLP VAE Embedding Frozen
        train_loss_mlp_vae_embedding_frozen = train_loop(mlp_vae_embedding_frozen, train_loader, optimizer_mlp_vae_embedding_frozen, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_vae_embedding_frozen, test_r2_mlp_vae_embedding_frozen, test_auc = unpack_eval(eval_loop(mlp_vae_embedding_frozen, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_vae_embedding_frozen', train_loss_mlp_vae_embedding_frozen, test_loss_mlp_vae_embedding_frozen, test_r2_mlp_vae_embedding_frozen, val_auc=test_auc)
        
        # TabM VAE Embedding Frozen
        train_loss_tabm_vae_embedding_frozen = train_loop(tabm_vae_embedding_frozen, train_loader, optimizer_tabm_vae_embedding_frozen, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_vae_embedding_frozen, test_r2_tabm_vae_embedding_frozen, test_auc = unpack_eval(eval_loop(tabm_vae_embedding_frozen, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_vae_embedding_frozen', train_loss_tabm_vae_embedding_frozen, test_loss_tabm_vae_embedding_frozen, test_r2_tabm_vae_embedding_frozen, val_auc=test_auc)
        
        # TabNet VAE Embedding Frozen
        train_loss_tabnet_vae_embedding_frozen = train_loop(tabnet_vae_embedding_frozen, train_loader_tabnet, optimizer_tabnet_vae_embedding_frozen, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_vae_embedding_frozen, test_r2_tabnet_vae_embedding_frozen, test_auc = unpack_eval(eval_loop(tabnet_vae_embedding_frozen, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_vae_embedding_frozen', train_loss_tabnet_vae_embedding_frozen, test_loss_tabnet_vae_embedding_frozen, test_r2_tabnet_vae_embedding_frozen, val_auc=test_auc)
        
        # MLP WAE Embedding Frozen
        train_loss_mlp_wae_embedding_frozen = train_loop(mlp_wae_embedding_frozen, train_loader, optimizer_mlp_wae_embedding_frozen, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_wae_embedding_frozen, test_r2_mlp_wae_embedding_frozen, test_auc = unpack_eval(eval_loop(mlp_wae_embedding_frozen, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_wae_embedding_frozen', train_loss_mlp_wae_embedding_frozen, test_loss_mlp_wae_embedding_frozen, test_r2_mlp_wae_embedding_frozen, val_auc=test_auc)
        
        # TabM WAE Embedding Frozen
        train_loss_tabm_wae_embedding_frozen = train_loop(tabm_wae_embedding_frozen, train_loader, optimizer_tabm_wae_embedding_frozen, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_wae_embedding_frozen, test_r2_tabm_wae_embedding_frozen, test_auc = unpack_eval(eval_loop(tabm_wae_embedding_frozen, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_wae_embedding_frozen', train_loss_tabm_wae_embedding_frozen, test_loss_tabm_wae_embedding_frozen, test_r2_tabm_wae_embedding_frozen, val_auc=test_auc)
        
        # TabNet WAE Embedding Frozen
        train_loss_tabnet_wae_embedding_frozen = train_loop(tabnet_wae_embedding_frozen, train_loader_tabnet, optimizer_tabnet_wae_embedding_frozen, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_wae_embedding_frozen, test_r2_tabnet_wae_embedding_frozen, test_auc = unpack_eval(eval_loop(tabnet_wae_embedding_frozen, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_wae_embedding_frozen', train_loss_tabnet_wae_embedding_frozen, test_loss_tabnet_wae_embedding_frozen, test_r2_tabnet_wae_embedding_frozen, val_auc=test_auc)
        
        
        
        # ========== Fine-tuned Embedding Models ==========
        # MLP Embedding Fine-tuned
        train_loss_mlp_embedding_finetuned = train_loop(mlp_embedding_finetuned, train_loader, optimizer_mlp_embedding_finetuned, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_embedding_finetuned, test_r2_mlp_embedding_finetuned, test_auc = unpack_eval(eval_loop(mlp_embedding_finetuned, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_embedding_finetuned', train_loss_mlp_embedding_finetuned, test_loss_mlp_embedding_finetuned, test_r2_mlp_embedding_finetuned, val_auc=test_auc)
        
        # TabM Embedding Fine-tuned
        train_loss_tabm_embedding_finetuned = train_loop(tabm_embedding_finetuned, train_loader, optimizer_tabm_embedding_finetuned, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_embedding_finetuned, test_r2_tabm_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabm_embedding_finetuned, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_embedding_finetuned', train_loss_tabm_embedding_finetuned, test_loss_tabm_embedding_finetuned, test_r2_tabm_embedding_finetuned, val_auc=test_auc)
        
        # TabNet Embedding Fine-tuned
        train_loss_tabnet_embedding_finetuned = train_loop(tabnet_embedding_finetuned, train_loader_tabnet, optimizer_tabnet_embedding_finetuned, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_embedding_finetuned, test_r2_tabnet_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabnet_embedding_finetuned, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_embedding_finetuned', train_loss_tabnet_embedding_finetuned, test_loss_tabnet_embedding_finetuned, test_r2_tabnet_embedding_finetuned, val_auc=test_auc)
        
        
        
        # MLP VAE Embedding Fine-tuned
        train_loss_mlp_vae_embedding_finetuned = train_loop(mlp_vae_embedding_finetuned, train_loader, optimizer_mlp_vae_embedding_finetuned, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_vae_embedding_finetuned, test_r2_mlp_vae_embedding_finetuned, test_auc = unpack_eval(eval_loop(mlp_vae_embedding_finetuned, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_vae_embedding_finetuned', train_loss_mlp_vae_embedding_finetuned, test_loss_mlp_vae_embedding_finetuned, test_r2_mlp_vae_embedding_finetuned, val_auc=test_auc)
        
        # TabM VAE Embedding Fine-tuned
        train_loss_tabm_vae_embedding_finetuned = train_loop(tabm_vae_embedding_finetuned, train_loader, optimizer_tabm_vae_embedding_finetuned, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_vae_embedding_finetuned, test_r2_tabm_vae_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabm_vae_embedding_finetuned, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_vae_embedding_finetuned', train_loss_tabm_vae_embedding_finetuned, test_loss_tabm_vae_embedding_finetuned, test_r2_tabm_vae_embedding_finetuned, val_auc=test_auc)
        
        # TabNet VAE Embedding Fine-tuned
        train_loss_tabnet_vae_embedding_finetuned = train_loop(tabnet_vae_embedding_finetuned, train_loader_tabnet, optimizer_tabnet_vae_embedding_finetuned, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_vae_embedding_finetuned, test_r2_tabnet_vae_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabnet_vae_embedding_finetuned, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_vae_embedding_finetuned', train_loss_tabnet_vae_embedding_finetuned, test_loss_tabnet_vae_embedding_finetuned, test_r2_tabnet_vae_embedding_finetuned, val_auc=test_auc)
        
        # MLP WAE Embedding Fine-tuned
        train_loss_mlp_wae_embedding_finetuned = train_loop(mlp_wae_embedding_finetuned, train_loader, optimizer_mlp_wae_embedding_finetuned, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio)
        test_loss_mlp_wae_embedding_finetuned, test_r2_mlp_wae_embedding_finetuned, test_auc = unpack_eval(eval_loop(mlp_wae_embedding_finetuned, val_loader, loss_fn, device, model_type='mlp', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_wae_embedding_finetuned', train_loss_mlp_wae_embedding_finetuned, test_loss_mlp_wae_embedding_finetuned, test_r2_mlp_wae_embedding_finetuned, val_auc=test_auc)
        
        # TabM WAE Embedding Fine-tuned
        train_loss_tabm_wae_embedding_finetuned = train_loop(tabm_wae_embedding_finetuned, train_loader, optimizer_tabm_wae_embedding_finetuned, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio)
        test_loss_tabm_wae_embedding_finetuned, test_r2_tabm_wae_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabm_wae_embedding_finetuned, val_loader, loss_fn, device, model_type='tabm', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_wae_embedding_finetuned', train_loss_tabm_wae_embedding_finetuned, test_loss_tabm_wae_embedding_finetuned, test_r2_tabm_wae_embedding_finetuned, val_auc=test_auc)
        
        # TabNet WAE Embedding Fine-tuned
        train_loss_tabnet_wae_embedding_finetuned = train_loop(tabnet_wae_embedding_finetuned, train_loader_tabnet, optimizer_tabnet_wae_embedding_finetuned, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio)
        test_loss_tabnet_wae_embedding_finetuned, test_r2_tabnet_wae_embedding_finetuned, test_auc = unpack_eval(eval_loop(tabnet_wae_embedding_finetuned, val_loader_tabnet, loss_fn, device, model_type='tabnet', use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_wae_embedding_finetuned', train_loss_tabnet_wae_embedding_finetuned, test_loss_tabnet_wae_embedding_finetuned, test_r2_tabnet_wae_embedding_finetuned, val_auc=test_auc)
        
        
        
        # ========== Multi-task Models ==========
        # MLP Multi-task
        train_loss_mlp_multi = train_multitask(
            mlp_multi_task_embedding_model, train_loader, optimizer_mlp_multi_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio
        )
        test_loss_mlp_multi, test_r2_mlp_multi, test_auc = unpack_eval(eval_multitask(
            mlp_multi_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio
        ))
        loss_tracker.update(epoch, 'mlp_multi_task_embedding', train_loss_mlp_multi, test_loss_mlp_multi, test_r2_mlp_multi, val_auc=test_auc)
        
        # TabM Multi-task
        train_loss_tabm_multi = train_multitask(
            tabm_multi_task_embedding_model, train_loader, optimizer_tabm_multi_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio
        )
        test_loss_tabm_multi, test_r2_tabm_multi, test_auc = unpack_eval(eval_multitask(
            tabm_multi_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio
        ))
        loss_tracker.update(epoch, 'tabm_multi_task_embedding', train_loss_tabm_multi, test_loss_tabm_multi, test_r2_tabm_multi, val_auc=test_auc)
        
        # TabNet Multi-task
        train_loss_tabnet_multi = train_multitask(
            tabnet_multi_task_embedding_model, train_loader_tabnet, optimizer_tabnet_multi_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, lambda_sparse=lambda_sparse
        )
        test_loss_tabnet_multi, test_r2_tabnet_multi, test_auc = unpack_eval(eval_multitask(
            tabnet_multi_task_embedding_model, val_loader_tabnet, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio
        ))
        loss_tracker.update(epoch, 'tabnet_multi_task_embedding', train_loss_tabnet_multi, test_loss_tabnet_multi, test_r2_tabnet_multi, val_auc=test_auc)
        
        
        
        # MLP Multi-task VAE
        train_loss_mlp_multi_vae = train_multi_vae(
            mlp_multi_vae_task_embedding_model, train_loader, optimizer_mlp_multi_vae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta
        )
        test_loss_mlp_multi_vae, test_r2_mlp_multi_vae, test_auc = unpack_eval(eval_multi_vae(
            mlp_multi_vae_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta
        ))
        loss_tracker.update(epoch, 'mlp_multi_vae_task_embedding', train_loss_mlp_multi_vae, test_loss_mlp_multi_vae, test_r2_mlp_multi_vae, val_auc=test_auc)
        
        # TabM Multi-task VAE
        train_loss_tabm_multi_vae = train_multi_vae(
            tabm_multi_vae_task_embedding_model, train_loader, optimizer_tabm_multi_vae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta
        )
        test_loss_tabm_multi_vae, test_r2_tabm_multi_vae, test_auc = unpack_eval(eval_multi_vae(
            tabm_multi_vae_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta
        ))
        loss_tracker.update(epoch, 'tabm_multi_vae_task_embedding', train_loss_tabm_multi_vae, test_loss_tabm_multi_vae, test_r2_tabm_multi_vae, val_auc=test_auc)
        
        # TabNet Multi-task VAE
        train_loss_tabnet_multi_vae = train_multi_vae(
            tabnet_multi_vae_task_embedding_model, train_loader_tabnet, optimizer_tabnet_multi_vae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta, lambda_sparse=lambda_sparse
        )
        test_loss_tabnet_multi_vae, test_r2_tabnet_multi_vae, test_auc = unpack_eval(eval_multi_vae(
            tabnet_multi_vae_task_embedding_model, val_loader_tabnet, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio, beta=vae_beta
        ))
        loss_tracker.update(epoch, 'tabnet_multi_vae_task_embedding', train_loss_tabnet_multi_vae, test_loss_tabnet_multi_vae, test_r2_tabnet_multi_vae, val_auc=test_auc)
        
        # MLP Multi-task WAE
        train_loss_mlp_multi_wae, train_comp_mlp_multi_wae = train_multitask(
            mlp_multi_wae_task_embedding_model, train_loader, optimizer_mlp_multi_wae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            return_components=True
        )
        eval_out = eval_multitask(
            mlp_multi_wae_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            deterministic_ot=deterministic_ot_eval, ot_seed=ot_eval_seed,
            return_components=True
        )
        if task == 'classification':
            test_loss_mlp_multi_wae, test_r2_mlp_multi_wae, test_auc, val_comp_mlp_multi_wae = eval_out
        else:
            test_loss_mlp_multi_wae, test_r2_mlp_multi_wae, val_comp_mlp_multi_wae = eval_out
            test_auc = float('nan')
        loss_tracker.update(epoch, 'mlp_multi_wae_task_embedding', train_loss_mlp_multi_wae, test_loss_mlp_multi_wae, test_r2_mlp_multi_wae, val_auc=test_auc)
        _log_component_metrics('mlp_multi_wae_task_embedding', train_comp_mlp_multi_wae, val_comp_mlp_multi_wae)
        
        # TabM Multi-task WAE
        train_loss_tabm_multi_wae, train_comp_tabm_multi_wae = train_multitask(
            tabm_multi_wae_task_embedding_model, train_loader, optimizer_tabm_multi_wae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            return_components=True
        )
        eval_out = eval_multitask(
            tabm_multi_wae_task_embedding_model, val_loader, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            deterministic_ot=deterministic_ot_eval, ot_seed=ot_eval_seed,
            return_components=True
        )
        if task == 'classification':
            test_loss_tabm_multi_wae, test_r2_tabm_multi_wae, test_auc, val_comp_tabm_multi_wae = eval_out
        else:
            test_loss_tabm_multi_wae, test_r2_tabm_multi_wae, val_comp_tabm_multi_wae = eval_out
            test_auc = float('nan')
        loss_tracker.update(epoch, 'tabm_multi_wae_task_embedding', train_loss_tabm_multi_wae, test_loss_tabm_multi_wae, test_r2_tabm_multi_wae, val_auc=test_auc)
        _log_component_metrics('tabm_multi_wae_task_embedding', train_comp_tabm_multi_wae, val_comp_tabm_multi_wae)
        
        # TabNet Multi-task WAE
        train_loss_tabnet_multi_wae, train_comp_tabnet_multi_wae = train_multitask(
            tabnet_multi_wae_task_embedding_model, train_loader_tabnet, optimizer_tabnet_multi_wae_task_embedding, 
            loss_fn, device, dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            lambda_sparse=lambda_sparse,
            return_components=True
        )
        eval_out = eval_multitask(
            tabnet_multi_wae_task_embedding_model, val_loader_tabnet, loss_fn, device, 
            dataset=full_dataset, tabular_loss_fn=tabular_loss, 
            recon_weight=multitask_recon_weight, use_log_ratio=use_log_ratio,
            lambda_ot=lambda_ot, regularization_type=regularization_type,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_max_iter=sinkhorn_max_iter,
            mmd_kernel_mul=mmd_kernel_mul, mmd_kernel_num=mmd_kernel_num,
            deterministic_ot=deterministic_ot_eval, ot_seed=ot_eval_seed,
            return_components=True
        )
        if task == 'classification':
            test_loss_tabnet_multi_wae, test_r2_tabnet_multi_wae, test_auc, val_comp_tabnet_multi_wae = eval_out
        else:
            test_loss_tabnet_multi_wae, test_r2_tabnet_multi_wae, val_comp_tabnet_multi_wae = eval_out
            test_auc = float('nan')
        loss_tracker.update(epoch, 'tabnet_multi_wae_task_embedding', train_loss_tabnet_multi_wae, test_loss_tabnet_multi_wae, test_r2_tabnet_multi_wae, val_auc=test_auc)
        _log_component_metrics('tabnet_multi_wae_task_embedding', train_comp_tabnet_multi_wae, val_comp_tabnet_multi_wae)
        
        
        
        # ========== GW Models ==========
        # MLP GW
        train_loss_mlp_gw = train_gw(mlp_gw_model, train_loader, optimizer_mlp_gw, loss_fn, device, 
                                    gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                    detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio)
        test_loss_mlp_gw, test_r2_mlp_gw, test_auc = unpack_eval(eval_gw(mlp_gw_model, val_loader, loss_fn, device,
                                                    gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                                    detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'mlp_gw', train_loss_mlp_gw, test_loss_mlp_gw, test_r2_mlp_gw, val_auc=test_auc)
        
        # TabM GW
        train_loss_tabm_gw = train_gw(tabm_gw_model, train_loader, optimizer_tabm_gw, loss_fn, device,
                                        gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                        detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio)
        test_loss_tabm_gw, test_r2_tabm_gw, test_auc = unpack_eval(eval_gw(tabm_gw_model, val_loader, loss_fn, device,
                                                        gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                                        detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabm_gw', train_loss_tabm_gw, test_loss_tabm_gw, test_r2_tabm_gw, val_auc=test_auc)
        
        # TabNet GW
        train_loss_tabnet_gw = train_gw(tabnet_gw_model, train_loader_tabnet, optimizer_tabnet_gw, loss_fn, device,
                                        gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                        detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio)
        test_loss_tabnet_gw, test_r2_tabnet_gw, test_auc = unpack_eval(eval_gw(tabnet_gw_model, val_loader_tabnet, loss_fn, device,
                                                    gw_weight=gw_weight_eff, gw_epsilon=gw_epsilon, x_normalized=apply_normalization,
                                                    detach_transport=gw_detach_transport, use_log_ratio=use_log_ratio))
        loss_tracker.update(epoch, 'tabnet_gw', train_loss_tabnet_gw, test_loss_tabnet_gw, test_r2_tabnet_gw, val_auc=test_auc)
        
        # # TabPFN GW
        # train_loss_tabpfn_gw = train_gw(tabpfn_gw_model, train_loader, optimizer_tabpfn_gw, loss_fn, device,
        #                                gw_weight=gw_weight, x_normalized=apply_normalization, use_log_ratio=use_log_ratio)
        # test_loss_tabpfn_gw, test_r2_tabpfn_gw = eval_gw(tabpfn_gw_model, val_loader, loss_fn, device,
        #                                                 gw_weight=gw_weight, x_normalized=apply_normalization, use_log_ratio=use_log_ratio)
        # loss_tracker.update(epoch, 'tabpfn_gw', train_loss_tabpfn_gw, test_loss_tabpfn_gw, test_r2_tabpfn_gw)
        
        # ========== Print Progress ==========
        summary = (
            f'{epoch + 1:5d}  | MLP train/val={train_loss_mlp:13.4f}/{test_loss_mlp:11.4f} | '
            f'TABM train/val={train_loss_tabm:13.4f}/{test_loss_tabm:11.4f} | '
            f'TABNET train/val={train_loss_tabnet:13.4f}/{test_loss_tabnet:11.4f} | '
            f'Frozen AE train/val={train_loss_mlp_embedding_frozen:13.4f}/{test_loss_mlp_embedding_frozen:11.4f} | '
            f'Fine-tuned AE train/val={train_loss_mlp_embedding_finetuned:13.4f}/{test_loss_mlp_embedding_finetuned:11.4f}'
        )
        print(summary)
        logger.info(summary)
        
        # ========== Save Checkpoints (only best models) ==========
        checkpoint_configs = [
            # Base models
            (model_mlp, optimizer_mlp, train_loss_mlp, test_loss_mlp, 'mlp'),
            (model_tabm, optimizer_tabm, train_loss_tabm, test_loss_tabm, 'tabm'),
            (model_tabnet, optimizer_tabnet, train_loss_tabnet, test_loss_tabnet, 'tabnet'),
            # Frozen embedding models
            (mlp_embedding_frozen, optimizer_mlp_embedding_frozen, train_loss_mlp_embedding_frozen, test_loss_mlp_embedding_frozen, 'mlp_embedding_frozen'),
            (tabm_embedding_frozen, optimizer_tabm_embedding_frozen, train_loss_tabm_embedding_frozen, test_loss_tabm_embedding_frozen, 'tabm_embedding_frozen'),
            (tabnet_embedding_frozen, optimizer_tabnet_embedding_frozen, train_loss_tabnet_embedding_frozen, test_loss_tabnet_embedding_frozen, 'tabnet_embedding_frozen'),
            # (tabpfn_embedding_frozen, optimizer_tabpfn_embedding_frozen, train_loss_tabpfn_embedding_frozen, test_loss_tabpfn_embedding_frozen, 'tabpfn_embedding_frozen'),
            # Fine-tuned embedding models
            (mlp_embedding_finetuned, optimizer_mlp_embedding_finetuned, train_loss_mlp_embedding_finetuned, test_loss_mlp_embedding_finetuned, 'mlp_embedding_finetuned'),
            (tabm_embedding_finetuned, optimizer_tabm_embedding_finetuned, train_loss_tabm_embedding_finetuned, test_loss_tabm_embedding_finetuned, 'tabm_embedding_finetuned'),
            (tabnet_embedding_finetuned, optimizer_tabnet_embedding_finetuned, train_loss_tabnet_embedding_finetuned, test_loss_tabnet_embedding_finetuned, 'tabnet_embedding_finetuned'),
            # VAE embedding models (frozen)
            (mlp_vae_embedding_frozen, optimizer_mlp_vae_embedding_frozen, train_loss_mlp_vae_embedding_frozen, test_loss_mlp_vae_embedding_frozen, 'mlp_vae_embedding_frozen'),
            (tabm_vae_embedding_frozen, optimizer_tabm_vae_embedding_frozen, train_loss_tabm_vae_embedding_frozen, test_loss_tabm_vae_embedding_frozen, 'tabm_vae_embedding_frozen'),
            (tabnet_vae_embedding_frozen, optimizer_tabnet_vae_embedding_frozen, train_loss_tabnet_vae_embedding_frozen, test_loss_tabnet_vae_embedding_frozen, 'tabnet_vae_embedding_frozen'),
            # VAE embedding models (finetuned)
            (mlp_vae_embedding_finetuned, optimizer_mlp_vae_embedding_finetuned, train_loss_mlp_vae_embedding_finetuned, test_loss_mlp_vae_embedding_finetuned, 'mlp_vae_embedding_finetuned'),
            (tabm_vae_embedding_finetuned, optimizer_tabm_vae_embedding_finetuned, train_loss_tabm_vae_embedding_finetuned, test_loss_tabm_vae_embedding_finetuned, 'tabm_vae_embedding_finetuned'),
            (tabnet_vae_embedding_finetuned, optimizer_tabnet_vae_embedding_finetuned, train_loss_tabnet_vae_embedding_finetuned, test_loss_tabnet_vae_embedding_finetuned, 'tabnet_vae_embedding_finetuned'),
            # WAE embedding models (frozen)
            (mlp_wae_embedding_frozen, optimizer_mlp_wae_embedding_frozen, train_loss_mlp_wae_embedding_frozen, test_loss_mlp_wae_embedding_frozen, 'mlp_wae_embedding_frozen'),
            (tabm_wae_embedding_frozen, optimizer_tabm_wae_embedding_frozen, train_loss_tabm_wae_embedding_frozen, test_loss_tabm_wae_embedding_frozen, 'tabm_wae_embedding_frozen'),
            (tabnet_wae_embedding_frozen, optimizer_tabnet_wae_embedding_frozen, train_loss_tabnet_wae_embedding_frozen, test_loss_tabnet_wae_embedding_frozen, 'tabnet_wae_embedding_frozen'),
            # WAE embedding models (finetuned)
            (mlp_wae_embedding_finetuned, optimizer_mlp_wae_embedding_finetuned, train_loss_mlp_wae_embedding_finetuned, test_loss_mlp_wae_embedding_finetuned, 'mlp_wae_embedding_finetuned'),
            (tabm_wae_embedding_finetuned, optimizer_tabm_wae_embedding_finetuned, train_loss_tabm_wae_embedding_finetuned, test_loss_tabm_wae_embedding_finetuned, 'tabm_wae_embedding_finetuned'),
            (tabnet_wae_embedding_finetuned, optimizer_tabnet_wae_embedding_finetuned, train_loss_tabnet_wae_embedding_finetuned, test_loss_tabnet_wae_embedding_finetuned, 'tabnet_wae_embedding_finetuned'),
            # Multi-task models
            (mlp_multi_task_embedding_model, optimizer_mlp_multi_task_embedding, train_loss_mlp_multi, test_loss_mlp_multi, 'mlp_multi_task_embedding'),
            (tabm_multi_task_embedding_model, optimizer_tabm_multi_task_embedding, train_loss_tabm_multi, test_loss_tabm_multi, 'tabm_multi_task_embedding'),
            (tabnet_multi_task_embedding_model, optimizer_tabnet_multi_task_embedding, train_loss_tabnet_multi, test_loss_tabnet_multi, 'tabnet_multi_task_embedding'),
            # Multi-task VAE models
            (mlp_multi_vae_task_embedding_model, optimizer_mlp_multi_vae_task_embedding, train_loss_mlp_multi_vae, test_loss_mlp_multi_vae, 'mlp_multi_vae_task_embedding'),
            (tabm_multi_vae_task_embedding_model, optimizer_tabm_multi_vae_task_embedding, train_loss_tabm_multi_vae, test_loss_tabm_multi_vae, 'tabm_multi_vae_task_embedding'),
            (tabnet_multi_vae_task_embedding_model, optimizer_tabnet_multi_vae_task_embedding, train_loss_tabnet_multi_vae, test_loss_tabnet_multi_vae, 'tabnet_multi_vae_task_embedding'),
            # Multi-task WAE models
            (mlp_multi_wae_task_embedding_model, optimizer_mlp_multi_wae_task_embedding, train_loss_mlp_multi_wae, test_loss_mlp_multi_wae, 'mlp_multi_wae_task_embedding'),
            (tabm_multi_wae_task_embedding_model, optimizer_tabm_multi_wae_task_embedding, train_loss_tabm_multi_wae, test_loss_tabm_multi_wae, 'tabm_multi_wae_task_embedding'),
            (tabnet_multi_wae_task_embedding_model, optimizer_tabnet_multi_wae_task_embedding, train_loss_tabnet_multi_wae, test_loss_tabnet_multi_wae, 'tabnet_multi_wae_task_embedding'),
            # GW models
            (mlp_gw_model, optimizer_mlp_gw, train_loss_mlp_gw, test_loss_mlp_gw, 'mlp_gw'),
            (tabm_gw_model, optimizer_tabm_gw, train_loss_tabm_gw, test_loss_tabm_gw, 'tabm_gw'),
            (tabnet_gw_model, optimizer_tabnet_gw, train_loss_tabnet_gw, test_loss_tabnet_gw, 'tabnet_gw'),
            # (tabpfn_gw_model, optimizer_tabpfn_gw, train_loss_tabpfn_gw, test_loss_tabpfn_gw, 'tabpfn_gw'),
        ]

        # Detailed per-model metrics for this epoch
        if checkpoint_configs:
            lines = []
            for _, _, _, _, model_name in checkpoint_configs:
                train_loss = _safe_last(loss_tracker.train_losses.get(model_name, []))
                val_loss = _safe_last(loss_tracker.val_losses.get(model_name, []))
                metric_val = _safe_last(loss_tracker.val_r2_scores.get(model_name, []))
                auc_val = _safe_last(loss_tracker.val_auc_scores.get(model_name, []))
                metric_label = 'F1' if task == 'classification' else 'R2'
                line = f'{model_name}: train={_fmt(train_loss)}, val={_fmt(val_loss)}, {metric_label}={_fmt(metric_val)}'
                if not _is_nan(auc_val):
                    line += f', AUC={_fmt(auc_val)}'
                lines.append(line)
            logger.info("Epoch %d detailed metrics:\n%s", epoch + 1, "\n".join(lines))
        
        for model, optimizer, train_loss, test_loss, model_name in checkpoint_configs:
            is_best = (test_loss == float(loss_tracker.best_val_loss.get(model_name, float('inf'))))
            if is_best and save_outputs:
                save_checkpoint(experiment_dir, epoch, model, optimizer, train_loss, test_loss, model_name, is_best)

        epoch_time = time.time() - epoch_start_time
        logger.info("Epoch %d completed in %.2fs", epoch + 1, epoch_time)
    logger.info('Training complete!')

    # Best metrics summary
    if loss_tracker.best_val_loss:
        summary_lines = []
        metric_label = 'F1' if task == 'classification' else 'R2'
        for model_name in sorted(loss_tracker.best_val_loss.keys()):
            best_loss = loss_tracker.best_val_loss.get(model_name, float('nan'))
            best_epoch = loss_tracker.best_epoch.get(model_name, -1)
            best_metric = loss_tracker.best_val_r2.get(model_name, float('nan'))
            best_metric_epoch = loss_tracker.best_r2_epoch.get(model_name, -1)
            best_auc = loss_tracker.best_val_auc.get(model_name, float('nan'))
            best_auc_epoch = loss_tracker.best_auc_epoch.get(model_name, -1)
            line = (f'{model_name}: best_val_loss={_fmt(best_loss)} (epoch {best_epoch + 1}), '
                    f'best_{metric_label}={_fmt(best_metric)} (epoch {best_metric_epoch + 1})')
            if not _is_nan(best_auc):
                line += f', best_AUC={_fmt(best_auc)} (epoch {best_auc_epoch + 1})'
            summary_lines.append(line)
        logger.info("Best validation summary:\n%s", "\n".join(summary_lines))
    
    # save the plots and the losses
    if save_outputs:
        loss_tracker.save_plots(experiment_dir)
        loss_tracker.save_losses_to_file(experiment_dir)
        loss_tracker.save_best_results_table(experiment_dir)
    
    if save_outputs:
        logger.info(f'All results have been saved to: {experiment_dir}')
        logger.info(f'  - config file: {experiment_dir / "configs"}')
        logger.info(f'  - checkpoint: {experiment_dir / "checkpoints"}')
        logger.info(f'  - loss plots: {experiment_dir / "plots"}')
        logger.info(f'  - loss records: {experiment_dir / "losses.txt"}')
        logger.info(f'  - best results CSV: {experiment_dir / "best_results.csv"}')
    else:
        logger.info('Training complete! (Outputs disabled - see wandb for all information)')

    total_time = time.time() - run_start_time
    logger.info("Total training time: %.2fs", total_time)
    
    if return_loss_tracker:
        return experiment_dir, loss_tracker
    else:
        return experiment_dir
    
