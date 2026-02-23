import copy
from pathlib import Path
from typing import Dict, Optional

import torch

from src.models.MLP import MLPEncoder
from src.models.TabM.tabM import TabM
from src.models.TabM.TabMEncoder import TabMEncoder
from src.models.TabNet.TabNet import TabNet
from src.models.TabNet.TabNetEncoder import TabNetEncoder
from src.models.Embedding.EncoderEmbedding import EncoderEmbedding
from src.models.Embedding.VAEEncoderEmbedding import VAEEncoderEmbedding
from src.models.Embedding.TabAE import TabAE
from src.models.Embedding.TabVAE import TabVAE
from src.models.Embedding.Decoder import TabularDecoder
from src.models.Embedding.MultiAE import MultiTaskEncoderEmbedding
from src.models.Embedding.MultiVAE import MultitaskVAEEncoderEmbedding


def _get_cfg(config: Optional[Dict], key: str, fallback_key: Optional[str] = None) -> Dict:
    if config is None:
        return {}
    cfg = config.get(key)
    if cfg is None and fallback_key:
        cfg = config.get(fallback_key)
    return cfg or {}


def create_models(input_dim, config, device, full_dataset=None, tabpfn_embedding_dim=None):
    """
    Create models dynamically based on model name patterns.
    """
    mlp_model_cfg = _get_cfg(config, 'mlp_model', 'mlp_model_cfg')
    tabm_model_cfg = _get_cfg(config, 'tabm_model', 'tabm_model_cfg')
    tabnet_model_cfg = _get_cfg(config, 'tabnet_model', 'tabnet_model_cfg')
    if tabnet_model_cfg is None:
        tabnet_model_cfg = {}
    encoder_model_cfg = _get_cfg(config, 'encoder_model', 'encoder_model_cfg')
    wae_encoder_model_cfg = _get_cfg(config, 'wae_encoder_model', 'wae_encoder_model_cfg')
    training_cfg = _get_cfg(config, 'training')
    task = training_cfg.get('task', 'regression')

    models = {}

    # Get encoder configuration
    encoder_latent_dim = encoder_model_cfg.get('latent_dim', 32)
    encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', encoder_model_cfg.get('hidden_dim', [32, 16]))
    encoder_dropout = encoder_model_cfg.get('dropout', 0.1)
    encoder_batchnorm = encoder_model_cfg.get('batchnorm', True)
    encoder_activation = encoder_model_cfg.get('activation', 'ReLU')

    # Head dropout: read from training config if available, otherwise fall back to encoder dropout
    head_dropout = training_cfg.get('head_dropout', encoder_dropout)

    # WAE encoder configuration (allow overrides beyond latent_dim)
    wae_encoder_cfg = wae_encoder_model_cfg or {}
    wae_latent_dim = wae_encoder_cfg.get('latent_dim', encoder_latent_dim)
    wae_hidden_dims = wae_encoder_cfg.get('hidden_dims', wae_encoder_cfg.get('hidden_dim', encoder_hidden_dims))
    wae_dropout = wae_encoder_cfg.get('dropout', encoder_dropout)
    wae_batchnorm = wae_encoder_cfg.get('batchnorm', encoder_batchnorm)
    wae_activation = wae_encoder_cfg.get('activation', encoder_activation)

    # Create decoder for TabAE/TabVAE/Multi-task models (if dataset info available)
    if full_dataset is not None:
        decoder_hidden_dims = list(reversed(encoder_hidden_dims))
        base_decoder = TabularDecoder(
            latent_dim=encoder_latent_dim,
            hidden_dims=decoder_hidden_dims,
            n_continuous=full_dataset.n_continuous,
            n_binary=full_dataset.n_binary,
            cat_sizes=full_dataset.cat_sizes,
            dropout=encoder_dropout,
        ).to(device)
        wae_decoder_hidden_dims = list(reversed(wae_hidden_dims))
        wae_decoder = TabularDecoder(
            latent_dim=wae_latent_dim,
            hidden_dims=wae_decoder_hidden_dims,
            n_continuous=full_dataset.n_continuous,
            n_binary=full_dataset.n_binary,
            cat_sizes=full_dataset.cat_sizes,
            dropout=wae_dropout,
        ).to(device)
    else:
        decoder_hidden_dims = list(reversed(encoder_hidden_dims))
        base_decoder = TabularDecoder(
            latent_dim=encoder_latent_dim,
            hidden_dims=decoder_hidden_dims,
            n_continuous=input_dim,
            n_binary=0,
            cat_sizes=[],
            dropout=encoder_dropout,
        ).to(device)
        wae_decoder_hidden_dims = list(reversed(wae_hidden_dims))
        wae_decoder = TabularDecoder(
            latent_dim=wae_latent_dim,
            hidden_dims=wae_decoder_hidden_dims,
            n_continuous=input_dim,
            n_binary=0,
            cat_sizes=[],
            dropout=wae_dropout,
        ).to(device)

    # ========== Base Models ==========
    mlp_latent_dim = mlp_model_cfg.get('latent_dim', 32)
    mlp_encoder_for_model = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=mlp_model_cfg.get('hidden_dims', mlp_model_cfg.get('hidden_size', [128, 64])),
        latent_dim=mlp_latent_dim,
        dropout=mlp_model_cfg.get('dropout', 0.3),
        batchnorm=mlp_model_cfg.get('batchnorm', True),
        activation=mlp_model_cfg.get('activation', 'ReLU'),
    ).to(device)
    models['mlp'] = EncoderEmbedding(
        mlp_encoder_for_model,
        mlp_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    models['tabm'] = TabM(
        in_dim=input_dim,
        out_dim=1,
        hidden_dims=tabm_model_cfg.get('hidden_dims', tabm_model_cfg.get('hidden_size', [128, 128])),
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=tabm_model_cfg.get('dropout', 0.1),
    ).to(device)

    try:
        models['tabnet'] = TabNet(
            input_dim=input_dim,
            output_dim=1,
            n_d=tabnet_model_cfg.get('n_d', 8),
            n_a=tabnet_model_cfg.get('n_a', 8),
            n_steps=tabnet_model_cfg.get('n_steps', 3),
            gamma=tabnet_model_cfg.get('gamma', 1.5),
        ).to(device)
    except Exception as e:
        print(f'Warning: Could not create TabNet model: {e}')

    # ========== Create Encoders ==========
    mlp_encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=encoder_hidden_dims,
        latent_dim=encoder_latent_dim,
        dropout=encoder_dropout,
        batchnorm=encoder_batchnorm,
        activation=encoder_activation,
    ).to(device)

    tabm_encoder = TabMEncoder(
        input_dim=input_dim,
        hidden_dims=encoder_hidden_dims,
        latent_dim=encoder_latent_dim,
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=encoder_dropout,
    ).to(device)

    tabnet_encoder = None
    try:
        tabnet_encoder = TabNetEncoder(
            input_dim=input_dim,
            latent_dim=encoder_latent_dim,
            n_d=tabnet_model_cfg.get('n_d', 8),
            n_a=tabnet_model_cfg.get('n_a', 8),
            n_steps=tabnet_model_cfg.get('n_steps', 3),
            gamma=tabnet_model_cfg.get('gamma', 1.5),
            dropout=encoder_dropout,
            return_entropy=True,
            n_independent=tabnet_model_cfg.get('n_independent', 2),
            n_shared=tabnet_model_cfg.get('n_shared', 2),
            epsilon=tabnet_model_cfg.get('epsilon', 1e-15),
            virtual_batch_size=tabnet_model_cfg.get('virtual_batch_size', 128),
            momentum=tabnet_model_cfg.get('momentum', 0.02),
            mask_type=tabnet_model_cfg.get('mask_type', 'sparsemax'),
            group_attention_matrix=tabnet_model_cfg.get('group_attention_matrix', None),
        ).to(device)
    except Exception as e:
        print(f'Warning: Could not create TabNet encoder: {e}')
        tabnet_encoder = None

    tabpfn_encoder = None

    # ========== Embedding Models ==========
    models['mlp_embedding'] = EncoderEmbedding(
        copy.deepcopy(mlp_encoder),
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)
    models['tabm_embedding'] = EncoderEmbedding(
        copy.deepcopy(tabm_encoder),
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)
    if tabnet_encoder is not None:
        models['tabnet_embedding'] = EncoderEmbedding(
            copy.deepcopy(tabnet_encoder),
            encoder_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== VAE Embedding Models ==========
    mlp_vae_encoder = copy.deepcopy(mlp_encoder)
    models['mlp_vae_embedding'] = VAEEncoderEmbedding(
        mlp_vae_encoder,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabm_vae_encoder = copy.deepcopy(tabm_encoder)
    models['tabm_vae_embedding'] = VAEEncoderEmbedding(
        tabm_vae_encoder,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    if tabnet_encoder is not None:
        tabnet_vae_encoder = copy.deepcopy(tabnet_encoder)
        models['tabnet_vae_embedding'] = VAEEncoderEmbedding(
            tabnet_vae_encoder,
            encoder_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== WAE Embedding Models ==========
    mlp_wae_encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=wae_hidden_dims,
        latent_dim=wae_latent_dim,
        dropout=wae_dropout,
        batchnorm=wae_batchnorm,
        activation=wae_activation,
    ).to(device)
    models['mlp_wae_embedding'] = EncoderEmbedding(
        mlp_wae_encoder,
        wae_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabm_wae_encoder = TabMEncoder(
        input_dim=input_dim,
        hidden_dims=wae_hidden_dims,
        latent_dim=wae_latent_dim,
        k_heads=tabm_model_cfg.get('k_heads', 8),
        dropout=wae_dropout,
        batchnorm=wae_batchnorm,
        activation=wae_activation,
    ).to(device)
    models['tabm_wae_embedding'] = EncoderEmbedding(
        tabm_wae_encoder,
        wae_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabnet_wae_encoder = None
    if tabnet_encoder is not None:
        try:
            tabnet_wae_encoder = TabNetEncoder(
                input_dim=input_dim,
                latent_dim=wae_latent_dim,
                n_d=tabnet_model_cfg.get('n_d', 8),
                n_a=tabnet_model_cfg.get('n_a', 8),
                n_steps=tabnet_model_cfg.get('n_steps', 3),
                gamma=tabnet_model_cfg.get('gamma', 1.5),
                dropout=wae_dropout,
                return_entropy=True,
                n_independent=tabnet_model_cfg.get('n_independent', 2),
                n_shared=tabnet_model_cfg.get('n_shared', 2),
                epsilon=tabnet_model_cfg.get('epsilon', 1e-15),
                virtual_batch_size=tabnet_model_cfg.get('virtual_batch_size', 128),
                momentum=tabnet_model_cfg.get('momentum', 0.02),
                mask_type=tabnet_model_cfg.get('mask_type', 'sparsemax'),
                group_attention_matrix=tabnet_model_cfg.get('group_attention_matrix', None),
            ).to(device)
        except Exception as e:
            print(f'Warning: Could not create TabNet WAE encoder: {e}')
            tabnet_wae_encoder = None
        if tabnet_wae_encoder is not None:
            models['tabnet_wae_embedding'] = EncoderEmbedding(
                tabnet_wae_encoder,
                wae_latent_dim,
                task=task,
                head_dropout=head_dropout,
            ).to(device)

    # ========== Multi-task Models ==========
    mlp_decoder_multi = copy.deepcopy(base_decoder)
    models['mlp_multi_task_embedding'] = MultiTaskEncoderEmbedding(
        copy.deepcopy(mlp_encoder),
        mlp_decoder_multi,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabm_decoder_multi = copy.deepcopy(base_decoder)
    models['tabm_multi_task_embedding'] = MultiTaskEncoderEmbedding(
        copy.deepcopy(tabm_encoder),
        tabm_decoder_multi,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    if tabnet_encoder is not None:
        tabnet_decoder_multi = copy.deepcopy(base_decoder)
        models['tabnet_multi_task_embedding'] = MultiTaskEncoderEmbedding(
            copy.deepcopy(tabnet_encoder),
            tabnet_decoder_multi,
            encoder_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== Multi-task VAE Models ==========
    mlp_decoder_multi_vae = copy.deepcopy(base_decoder)
    models['mlp_multi_vae_task_embedding'] = MultitaskVAEEncoderEmbedding(
        copy.deepcopy(mlp_vae_encoder),
        mlp_decoder_multi_vae,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabm_decoder_multi_vae = copy.deepcopy(base_decoder)
    models['tabm_multi_vae_task_embedding'] = MultitaskVAEEncoderEmbedding(
        copy.deepcopy(tabm_vae_encoder),
        tabm_decoder_multi_vae,
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    if tabnet_encoder is not None:
        tabnet_decoder_multi_vae = copy.deepcopy(base_decoder)
        models['tabnet_multi_vae_task_embedding'] = MultitaskVAEEncoderEmbedding(
            copy.deepcopy(tabnet_vae_encoder),
            tabnet_decoder_multi_vae,
            encoder_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== Multi-task WAE Models ==========
    mlp_decoder_multi_wae = copy.deepcopy(wae_decoder)
    models['mlp_multi_wae_task_embedding'] = MultiTaskEncoderEmbedding(
        copy.deepcopy(mlp_wae_encoder),
        mlp_decoder_multi_wae,
        wae_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    tabm_decoder_multi_wae = copy.deepcopy(wae_decoder)
    models['tabm_multi_wae_task_embedding'] = MultiTaskEncoderEmbedding(
        copy.deepcopy(tabm_wae_encoder),
        tabm_decoder_multi_wae,
        wae_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)

    if tabnet_wae_encoder is not None:
        tabnet_decoder_multi_wae = copy.deepcopy(wae_decoder)
        models['tabnet_multi_wae_task_embedding'] = MultiTaskEncoderEmbedding(
            copy.deepcopy(tabnet_wae_encoder),
            tabnet_decoder_multi_wae,
            wae_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== GW Models ==========
    models['mlp_gw'] = EncoderEmbedding(
        copy.deepcopy(mlp_encoder),
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)
    models['tabm_gw'] = EncoderEmbedding(
        copy.deepcopy(tabm_encoder),
        encoder_latent_dim,
        task=task,
        head_dropout=head_dropout,
    ).to(device)
    if tabnet_encoder is not None:
        models['tabnet_gw'] = EncoderEmbedding(
            copy.deepcopy(tabnet_encoder),
            encoder_latent_dim,
            task=task,
            head_dropout=head_dropout,
        ).to(device)

    # ========== Pre-training Models (TabAE, TabVAE, TabWAE) ==========
    mlp_decoder_tabae = copy.deepcopy(base_decoder)
    models['mlp_tabae'] = TabAE(encoder=copy.deepcopy(mlp_encoder), decoder=mlp_decoder_tabae).to(device)

    tabm_decoder_tabae = copy.deepcopy(base_decoder)
    models['tabm_tabae'] = TabAE(encoder=copy.deepcopy(tabm_encoder), decoder=tabm_decoder_tabae).to(device)

    if tabnet_encoder is not None:
        tabnet_decoder_tabae = copy.deepcopy(base_decoder)
        models['tabnet_tabae'] = TabAE(encoder=copy.deepcopy(tabnet_encoder), decoder=tabnet_decoder_tabae).to(device)

    mlp_decoder_tabvae = copy.deepcopy(base_decoder)
    mlp_vae_encoder_tabvae = copy.deepcopy(mlp_encoder)
    models['mlp_tabvae'] = TabVAE(encoder=mlp_vae_encoder_tabvae, decoder=mlp_decoder_tabvae, latent_dim=encoder_latent_dim).to(device)

    tabm_decoder_tabvae = copy.deepcopy(base_decoder)
    tabm_vae_encoder_tabvae = copy.deepcopy(tabm_encoder)
    models['tabm_tabvae'] = TabVAE(encoder=tabm_vae_encoder_tabvae, decoder=tabm_decoder_tabvae, latent_dim=encoder_latent_dim).to(device)

    if tabnet_encoder is not None:
        tabnet_decoder_tabvae = copy.deepcopy(base_decoder)
        tabnet_vae_encoder_tabvae = copy.deepcopy(tabnet_encoder)
        models['tabnet_tabvae'] = TabVAE(encoder=tabnet_vae_encoder_tabvae, decoder=tabnet_decoder_tabvae, latent_dim=encoder_latent_dim).to(device)

    mlp_decoder_tabwae = copy.deepcopy(wae_decoder)
    models['mlp_tabwae'] = TabAE(encoder=copy.deepcopy(mlp_wae_encoder), decoder=mlp_decoder_tabwae).to(device)

    tabm_decoder_tabwae = copy.deepcopy(wae_decoder)
    models['tabm_tabwae'] = TabAE(encoder=copy.deepcopy(tabm_wae_encoder), decoder=tabm_decoder_tabwae).to(device)

    if tabnet_wae_encoder is not None:
        tabnet_decoder_tabwae = copy.deepcopy(wae_decoder)
        models['tabnet_tabwae'] = TabAE(encoder=copy.deepcopy(tabnet_wae_encoder), decoder=tabnet_decoder_tabwae).to(device)

    return models


def load_model_from_checkpoint(
    checkpoint_path,
    model_type,
    input_dim,
    config,
    device,
    full_dataset=None,
    train_features=None,
    train_labels=None,
    checkpoints_dir: Optional[Path] = None,
    filter_mismatched: bool = True,
):
    """
    Load model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    base_model_type = model_type
    if model_type.endswith('_frozen'):
        base_model_type = model_type[:-7]
    elif model_type.endswith('_finetuned'):
        base_model_type = model_type[:-10]

    tabpfn_embedding_dim = None

    models = create_models(input_dim, config, device, full_dataset=full_dataset,
                           tabpfn_embedding_dim=tabpfn_embedding_dim)

    if base_model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type} (base: {base_model_type}). "
            f"Available: {list(models.keys())}"
        )

    model = models[base_model_type]

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if filter_mismatched:
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Skipping {key} due to size mismatch "
                          f"(checkpoint: {value.shape}, model: {model_state_dict[key].shape})")

        missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")

        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    # Handle VAE models: load mu and log_var layers from pretrained TabVAE if available
    if 'vae' in model_type.lower() and checkpoints_dir is not None:
        if model_type.startswith('mlp_'):
            tabvae_name = 'mlp_tabvae'
        elif model_type.startswith('tabm_'):
            tabvae_name = 'tabm_tabvae'
        elif model_type.startswith('tabnet_'):
            tabvae_name = 'tabnet_tabvae'
        else:
            tabvae_name = None

        if tabvae_name:
            tabvae_path = Path(checkpoints_dir) / f'{tabvae_name}_best.pt'
            if tabvae_path.exists() and hasattr(model, 'mu') and hasattr(model, 'log_var'):
                try:
                    print(f'  Loading VAE layers (mu, log_var) from {tabvae_name}...')
                    tabvae_checkpoint = torch.load(tabvae_path, map_location=device)
                    tabvae_models = create_models(input_dim, config, device, full_dataset=full_dataset)
                    if tabvae_name in tabvae_models:
                        tabvae_model = tabvae_models[tabvae_name]
                        tabvae_model.load_state_dict(tabvae_checkpoint.get('model_state_dict', tabvae_checkpoint), strict=False)
                        if hasattr(tabvae_model, 'mu') and hasattr(tabvae_model, 'log_var'):
                            model.mu.load_state_dict(tabvae_model.mu.state_dict())
                            model.log_var.load_state_dict(tabvae_model.log_var.state_dict())
                            print(f'  VAE layers loaded successfully from {tabvae_name}')
                except Exception as e:
                    print(f'  Warning: Could not load VAE layers from {tabvae_name}: {e}')

    model.eval()
    return model
