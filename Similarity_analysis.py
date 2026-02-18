import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import random

warnings.filterwarnings('ignore')

# Imports from your src
from src.data.prepare_data import prepare_data_from_experiment, get_model_configs
from src.utils.config import load_config, get_config_value
from src.models.MLP import MLPEncoder
from src.models.TabM.tabM import TabM
from src.models.TabM.TabMEncoder import TabMEncoder
from src.models.TabNet.TabNetEncoder import TabNetEncoder
# from src.models.TabPFN.TabPFNEncoder import TabPFNEncoder
from src.models.Embedding.EncoderEmbedding import EncoderEmbedding
from src.models.Embedding.VAEEncoderEmbedding import VAEEncoderEmbedding
from src.models.Embedding.TabAE import TabAE
from src.models.Embedding.TabVAE import TabVAE
from src.models.Embedding.Decoder import TabularDecoder
from src.models.Embedding.MultiAE import MulltiTaskEnecoderEmbedding
from src.models.Embedding.MultiVAE import MultitaskVAEEncoderEmbedding
from scipy.stats import pearsonr

def compute_cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X).float()
    else:
        X_t = X.float()
        
    # Normalize to unit vectors (Shape-based comparison)
    X_norm = F.normalize(X_t, p=2, dim=1)
    
    # Similarity = 1.0 (Identical Profile) to -1.0 (Opposite Profile)
    similarity = torch.mm(X_norm, X_norm.t())
    return similarity.numpy()

def plot_similarity_heatmap(matrix: np.ndarray, title: str, save_path: Path, max_samples: int = 50, original_indices: np.ndarray = None, is_kl: bool = False):
    plt.figure(figsize=(10, 8))
    # Slice for readability
    n_samples = min(max_samples, matrix.shape[0])
    viz_data = matrix[:n_samples, :n_samples]
    
    # Use original indices if provided, otherwise use position indices
    if original_indices is not None:
        tick_labels = original_indices[:n_samples]
    else:
        tick_labels = np.arange(n_samples)
    
    
    if is_kl:
        cmap = 'viridis_r'  
        cbar_label = 'KL Divergence'
        threshold = np.percentile(viz_data[viz_data > 0], 25)  
        annot_matrix = np.where((viz_data < threshold) & (viz_data > 0), 
                               np.round(viz_data, 2).astype(str), '')
    else:
        cmap = 'viridis'
        cbar_label = 'Cosine Similarity'
        # For cosine: show values > 0.6 (more similar)
        annot_matrix = np.where(viz_data > 0.6, 
                               np.round(viz_data, 2).astype(str), '')
    
    # Create heatmap with original indices as labels
    if n_samples > 30:
        step = max(1, n_samples // 20) 
        x_tick_labels = [str(tick_labels[i]) if i % step == 0 else '' for i in range(n_samples)]
        y_tick_labels = [str(tick_labels[i]) if i % step == 0 else '' for i in range(n_samples)]
    else:
        x_tick_labels = [str(idx) for idx in tick_labels]
        y_tick_labels = [str(idx) for idx in tick_labels]
    
    sns.heatmap(viz_data, cmap=cmap, square=True, 
                xticklabels=x_tick_labels, yticklabels=y_tick_labels,
                annot=annot_matrix, fmt='', cbar_kws={'label': cbar_label})
    plt.title(f'{title} Heatmap')
    plt.xlabel('Patient Index', fontweight='bold')
    plt.ylabel('Patient Index', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def filter_state_dict(state_dict: dict, model: torch.nn.Module, strict: bool = False) -> dict:
    """
    Filter state_dict to only include keys that exist in the model and have matching shapes.
    
    Args:
        state_dict: The checkpoint state_dict to filter
        model: The model to match against
        strict: If True, raise error on size mismatches. If False, skip mismatched keys.
    
    Returns:
        Filtered state_dict with only matching keys
    """
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            elif strict:
                raise RuntimeError(
                    f"Size mismatch for {key}: checkpoint has shape {value.shape}, "
                    f"model expects shape {model_state_dict[key].shape}"
                )
            else:
                print(f"Warning: Skipping {key} due to size mismatch "
                      f"(checkpoint: {value.shape}, model: {model_state_dict[key].shape})")
        # Skip keys that don't exist in model (e.g., output heads from multi-task models)
    
    missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    
    return filtered_state_dict

def extract_latent_representation(model, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Extract latent representation from model.
    
    Note: For TabPFN, the encoder has no trainable weights - it's a frozen feature extractor.
    The encoder.forward() uses torch.no_grad() internally, which is correct since there are
    no weights to backpropagate through. The embeddings are still valid for similarity analysis.
    """
    model.eval()
    with torch.no_grad():
        features_tensor = features.to(device) if isinstance(features, torch.Tensor) else torch.from_numpy(features).float().to(device)
        
        # Handle different model types
        if isinstance(model, TabAE):
            # TabAE forward returns (recon_cont, recon_bin, recon_cat), need z from encoder
            latent = model.encoder(features_tensor)
        elif isinstance(model, TabVAE):
            # TabVAE returns (recon_cont, recon_bin, recon_cat, mu, log_var, z)
            _, _, _, mu, _, _ = model(features_tensor, training=False)
            latent = mu  # Use mu as latent representation
        elif isinstance(model, EncoderEmbedding):
            # EncoderEmbedding's encoder returns z directly
            # For TabPFN: encoder.forward() uses no_grad internally (no trainable weights),
            # but embeddings are still valid for similarity computation
            latent = model.encoder(features_tensor)
        elif isinstance(model, VAEEncoderEmbedding):
            # VAEEncoderEmbedding: encoder returns z_features, need to compute mu manually
            z_features = model.encoder(features_tensor)
            latent = model.mu(z_features)  # Use mu as latent representation  
        elif isinstance(model, MulltiTaskEnecoderEmbedding):
            # MultiTaskEncoderEmbedding returns (y_hat, z, recon_cont, recon_bin, recon_cat)
            _, latent, _, _, _ = model(features_tensor)
        elif isinstance(model, MultitaskVAEEncoderEmbedding):
            # MultitaskVAEEncoderEmbedding returns (y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat)
            # For similarity, we use mu as latent representation
            _, _, mu, _, _, _, _ = model(features_tensor)
            latent = mu
        else:
            # Fallback: try to get encoder and call it
            if hasattr(model, 'encoder'):
                latent = model.encoder(features_tensor)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
        
        return latent.cpu().numpy()

def extract_vae_latent_distribution(model, features: torch.Tensor, device: torch.device) -> tuple:
    """
    Extract mu and log_var from VAE models for KL divergence calculation.
    """
    model.eval()
    with torch.no_grad():
        features_tensor = features.to(device) if isinstance(features, torch.Tensor) else torch.from_numpy(features).float().to(device)
        
        if isinstance(model, TabVAE):
            # TabVAE returns (y_linear, z, mu, log_var)
            y_linear, z, mu, log_var = model(features_tensor, training=False)
        elif isinstance(model, VAEEncoderEmbedding):
            # VAEEncoderEmbedding: use its encoder + VAE layers to get (mu, log_var)
            z_features = model.encoder(features_tensor)
            mu = model.mu(z_features)
            log_var = model.log_var(z_features)
        elif isinstance(model, MultitaskVAEEncoderEmbedding):
            # MultitaskVAEEncoderEmbedding returns (y_hat, z, mu, log_var, recon_cont, recon_bin, recon_cat)
            _, _, mu, log_var, _, _, _ = model(features_tensor)
        else:
            raise ValueError(f"Unknown VAE model type: {type(model)}")
        
        return mu.cpu().numpy(), log_var.cpu().numpy()

def compute_kl_divergence_matrix_from_vae(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Compute symmetric pairwise KL divergence for VAE latent distributions.
    For two diagonal Gaussians N(mu_i, sigma_i^2) and N(mu_j, sigma_j^2),
    we compute 0.5 * (KL(i || j) + KL(j || i)) to obtain a symmetric metric.
    """
    if isinstance(mu, np.ndarray):
        mu_t = torch.from_numpy(mu).float()
        log_var_t = torch.from_numpy(log_var).float()
    else:
        mu_t = mu.float()
        log_var_t = log_var.float()
    
    n_samples = mu_t.shape[0]
    kl_matrix = np.zeros((n_samples, n_samples))
    
    # Convert log_var to variance
    var_t = torch.exp(log_var_t)
    
    # Loop over samples; for analysis this O(N^2) loop is acceptable
    for i in range(n_samples):
        mu_i = mu_t[i]          # (latent_dim,)
        var_i = var_t[i]        # (latent_dim,)
        
        # Differences and ratios for KL(i || j)
        # KL(N(mu_i, var_i) || N(mu_j, var_j)) for all j
        # For diagonal covariance:
        # KL = 0.5 * [ sum(var_i / var_j) + sum((mu_j - mu_i)^2 / var_j) - k + sum(log(var_j / var_i)) ]
        var_ratio = var_i.unsqueeze(0) / var_t              # (n_samples, latent_dim)
        mu_diff = mu_t - mu_i.unsqueeze(0)                  # (n_samples, latent_dim)
        mu_diff_sq = mu_diff ** 2                           # (n_samples, latent_dim)
        log_var_ratio = log_var_t - log_var_t[i].unsqueeze(0)  # (n_samples, latent_dim)
        
        kl_forward = 0.5 * (
            var_ratio.sum(dim=1)
            + (mu_diff_sq / var_t).sum(dim=1)
            - mu_t.shape[1]
            + log_var_ratio.sum(dim=1)
        )
        
        # KL(j || i): swap roles of i and j
        # KL(N(mu_j, var_j) || N(mu_i, var_i))
        var_ratio_back = var_t / var_i.unsqueeze(0)
        mu_diff_sq_back = mu_diff_sq                          # (x - y)^2 == (y - x)^2
        log_var_ratio_back = -log_var_ratio                   # log(var_i / var_j) = -log(var_j / var_i)
        
        kl_backward = 0.5 * (
            var_ratio_back.sum(dim=1)
            + (mu_diff_sq_back / var_i.unsqueeze(0)).sum(dim=1)
            - mu_t.shape[1]
            + log_var_ratio_back.sum(dim=1)
        )
        
        # Symmetric KL
        kl_sym = 0.5 * (kl_forward + kl_backward)
        kl_matrix[i] = kl_sym.cpu().numpy()
    
    return kl_matrix

def plot_correlation_scatter(input_cosine: np.ndarray, latent_metric: np.ndarray, 
                            model_name: str, save_path: Path, is_kl: bool = False,
                            target_labels: np.ndarray = None, thresholds: list = [0.35, 0.4, 0.5]):

    if target_labels is None:
        # Fallback to single plot if no target labels
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        correlation, p_value = pearsonr(input_cosine, latent_metric)
        ax.scatter(input_cosine, latent_metric, alpha=0.5, s=20, edgecolors='none')
        ax.set_xlabel('Input Space Cosine Similarity', fontsize=11, fontweight='bold')
        if is_kl:
            ax.set_ylabel('Latent Space KL Divergence', fontsize=11, fontweight='bold')
            title_suffix = 'KL Divergence'
        else:
            ax.set_ylabel('Latent Space Cosine Similarity', fontsize=11, fontweight='bold')
            min_val = min(input_cosine.min(), latent_metric.min())
            max_val = max(input_cosine.max(), latent_metric.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                    label='Perfect Correlation', alpha=0.7)
            title_suffix = 'Cosine Similarity'
        ax.set_title(f'{model_name} - Input vs Latent {title_suffix}\nCorrelation: {correlation:.3f} (p={p_value:.2e})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate correlation
    correlation, p_value = pearsonr(input_cosine, latent_metric)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    n = len(target_labels)
    triu_indices = np.triu_indices(n, k=1)
    
    # Get LVEF values for each sample in the pair
    lvef_i = target_labels[triu_indices[0]]  # LVEF values for first sample in each pair
    lvef_j = target_labels[triu_indices[1]]  # LVEF values for second sample in each pair
    
    # Loop through each threshold to create 3 plots
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        
        # Create 4 cases based on threshold comparison
        # Case 1: y_i < threshold and y_j < threshold (both below)
        case1_mask = (lvef_i < threshold) & (lvef_j < threshold)
        
        # Case 2: y_i > threshold and y_j > threshold (both above)
        case2_mask = (lvef_i > threshold) & (lvef_j > threshold)
        
        # Case 3: y_i < threshold and y_j > threshold (i below, j above)
        case3_mask = (lvef_i < threshold) & (lvef_j > threshold)
        
        # Case 4: y_i > threshold and y_j < threshold (i above, j below)
        case4_mask = (lvef_i > threshold) & (lvef_j < threshold)
        
        
        
        # Plot Case 2: both above (blue)
        if case2_mask.sum() > 0:
            ax.scatter(input_cosine[case2_mask], latent_metric[case2_mask], 
                      alpha=0.6, s=20, edgecolors='none', color='#1f77b4', 
                      label=f'Both > {threshold}')
        
        # Plot Case 3: i below, j above (orange)
        if case3_mask.sum() > 0:
            ax.scatter(input_cosine[case3_mask], latent_metric[case3_mask], 
                      alpha=0.6, s=20, edgecolors='none', color='#ff7f0e', 
                      label=f'i < {threshold}, j > {threshold}')
        
        # Plot Case 4: i above, j below (green)
        if case4_mask.sum() > 0:
            ax.scatter(input_cosine[case4_mask], latent_metric[case4_mask], 
                      alpha=0.6, s=20, edgecolors='none', color='#2ca02c', 
                      label=f'i > {threshold}, j < {threshold}')

        # Plot Case 1: both below (red)
        if case1_mask.sum() > 0:
            ax.scatter(input_cosine[case1_mask], latent_metric[case1_mask], 
                      alpha=0.6, s=20, edgecolors='none', color='#d62728', 
                      label=f'Both < {threshold}')
        
        # Add identity line (y=x) for reference (only for cosine similarity)
        if not is_kl:
            min_val = min(input_cosine.min(), latent_metric.min())
            max_val = max(input_cosine.max(), latent_metric.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
        
        # Set title and labels
        ax.set_title(f'Threshold: {threshold:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Input Space Cosine Similarity', fontsize=10, fontweight='bold')
        
        # Only put y-label on the first plot
        if i == 0:
            if is_kl:
                ax.set_ylabel('Latent Space KL Divergence', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('Latent Space Cosine Similarity', fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Move legend inside to save space
        ax.legend(title='Status', loc='upper left', fontsize=8)
    
    # Add overall title with correlation info
    fig.suptitle(f'{model_name} - Input vs Latent Space\nCorrelation: {correlation:.3f} (p={p_value:.2e})', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_tabae_model(input_dim, latent_dim, hidden_dims, encoder_cfg, full_dataset, device, 
                      encoder_type='mlp', model_configs=None):
    """Helper function to create TabAE model."""
    encoder_dropout = encoder_cfg.get('dropout', 0.1)
    
    if encoder_type == 'mlp':
        encoder_batchnorm = encoder_cfg.get('batchnorm', True)
        encoder_activation = encoder_cfg.get('activation', 'ReLU')
        encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=encoder_dropout,
            batchnorm=encoder_batchnorm,
            activation=encoder_activation
        ).to(device)
    elif encoder_type == 'tabnet':
        tabnet_model_cfg = model_configs.get('tabnet_model_cfg', {}) if model_configs else {}
        encoder = TabNetEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            n_d=tabnet_model_cfg.get('n_d', 8),
            n_a=tabnet_model_cfg.get('n_a', 8),
            n_steps=tabnet_model_cfg.get('n_steps', 3),
            gamma=tabnet_model_cfg.get('gamma', 1.5),
            dropout=encoder_dropout
        ).to(device)
    # elif encoder_type == 'tabpfn':
    #     tabpfn_model_cfg = model_configs.get('tabpfn_model_cfg', {}) if model_configs else {}
    #     tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     tabpfn_device = tabpfn_model_cfg.get('device', tabpfn_device)
    #     encoder = TabPFNEncoder(
    #         input_dim=input_dim,
    #         latent_dim=latent_dim,
    #         device=tabpfn_device,
    #         base_path=tabpfn_model_cfg.get('base_path', None)
    #     ).to(device)
    else:  # tabm
        tabm_model_cfg = model_configs.get('tabm_model_cfg', {}) if model_configs else {}
        encoder = TabMEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            k_heads=tabm_model_cfg.get('k_heads', 8),
            adapter_dim=tabm_model_cfg.get('adapter_dim', None),
            dropout=encoder_dropout
        ).to(device)
    
    # # For TabPFN, decoder hidden_dims should use a default since TabPFN doesn't use hidden_dims
    # if encoder_type == 'tabpfn':
    #     decoder_hidden_dims = [64, 32]  # Default decoder structure for TabPFN
    # else:
    decoder_hidden_dims = list(reversed(hidden_dims))
    decoder = TabularDecoder(
        latent_dim=latent_dim,
        hidden_dims=decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=encoder_dropout
    ).to(device)
    
    return TabAE(encoder=encoder, decoder=decoder).to(device)


def create_tabvae_model(input_dim, latent_dim, hidden_dims, encoder_cfg, full_dataset, device,
                       encoder_type='mlp', model_configs=None):
    """Helper function to create TabVAE model."""
    encoder_dropout = encoder_cfg.get('dropout', 0.1)
    
    if encoder_type == 'mlp':
        encoder_batchnorm = encoder_cfg.get('batchnorm', True)
        encoder_activation = encoder_cfg.get('activation', 'ReLU')
        encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=encoder_dropout,
            batchnorm=encoder_batchnorm,
            activation=encoder_activation
        ).to(device)
    elif encoder_type == 'tabnet':
        tabnet_model_cfg = model_configs.get('tabnet_model_cfg', {}) if model_configs else {}
        encoder = TabNetEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            n_d=tabnet_model_cfg.get('n_d', 8),
            n_a=tabnet_model_cfg.get('n_a', 8),
            n_steps=tabnet_model_cfg.get('n_steps', 3),
            gamma=tabnet_model_cfg.get('gamma', 1.5),
            dropout=encoder_dropout
        ).to(device)
    # elif encoder_type == 'tabpfn':
    #     tabpfn_model_cfg = model_configs.get('tabpfn_model_cfg', {}) if model_configs else {}
    #     tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     tabpfn_device = tabpfn_model_cfg.get('device', tabpfn_device)
    #     encoder = TabPFNEncoder(
    #         input_dim=input_dim,
    #         latent_dim=latent_dim,
    #         device=tabpfn_device,
    #         base_path=tabpfn_model_cfg.get('base_path', None)
    #     ).to(device)
    else:  # tabm
        tabm_model_cfg = model_configs.get('tabm_model_cfg', {}) if model_configs else {}
        encoder = TabMEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            k_heads=tabm_model_cfg.get('k_heads', 8),
            adapter_dim=tabm_model_cfg.get('adapter_dim', None),
            dropout=encoder_dropout
        ).to(device)
    
    # # For TabPFN, decoder hidden_dims should use a default since TabPFN doesn't use hidden_dims
    # if encoder_type == 'tabpfn':
    #     decoder_hidden_dims = [64, 32]  # Default decoder structure for TabPFN
    # else:
    decoder_hidden_dims = list(reversed(hidden_dims))
    decoder = TabularDecoder(
        latent_dim=latent_dim,
        hidden_dims=decoder_hidden_dims,
        n_continuous=full_dataset.n_continuous,
        n_binary=full_dataset.n_binary,
        cat_sizes=full_dataset.cat_sizes,
        dropout=encoder_dropout
    ).to(device)
    
    return TabVAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim).to(device)


def load_model_from_checkpoint(model, checkpoint_path, device):
    """Helper function to load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    filtered_state_dict = filter_state_dict(checkpoint['model_state_dict'], model, strict=False)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model


def compute_and_plot_similarity(model, test_features_tensor, device, input_cosine_values, 
                                model_name, results_dir, target_labels, test_indices, 
                                max_samples_for_heatmap, n_samples, is_vae=False):
    """Helper function to compute similarity and create plots."""
    if is_vae:
        mu, log_var = extract_vae_latent_distribution(model, test_features_tensor, device)
        latent_kl = compute_kl_divergence_matrix_from_vae(mu, log_var)
        latent_kl_values = latent_kl[np.triu_indices(n_samples, k=1)]
        mu_cosine_sim = compute_cosine_similarity_matrix(mu)
        mu_cosine_values = mu_cosine_sim[np.triu_indices(n_samples, k=1)]
        
        # Plot KL divergence
        correlation_path = results_dir / f'{model_name.replace(" ", "_").lower()}_correlation_kl.png'
        plot_correlation_scatter(input_cosine_values, latent_kl_values, 
                               f'{model_name} (KL Divergence)', correlation_path, is_kl=True,
                               target_labels=target_labels)
        print(f'Saved correlation plot (KL) to {correlation_path}')
        
        # Plot cosine similarity
        correlation_path = results_dir / f'{model_name.replace(" ", "_").lower()}_correlation_cosine.png'
        plot_correlation_scatter(input_cosine_values, mu_cosine_values, 
                               f'{model_name} (Cosine Similarity)', correlation_path, is_kl=False,
                               target_labels=target_labels)
        print(f'Saved correlation plot (Cosine) to {correlation_path}')
        
        # Heatmap
        heatmap_path = results_dir / f'{model_name.replace(" ", "_").lower()}_latent_heatmap.png'
        plot_similarity_heatmap(mu_cosine_sim, f'{model_name} Latent Space (mu)', 
                               heatmap_path, max_samples=max_samples_for_heatmap, 
                               original_indices=test_indices, is_kl=False)
        print(f'Saved latent space heatmap to {heatmap_path}')
    else:
        latent_features = extract_latent_representation(model, test_features_tensor, device)
        latent_cosine_sim = compute_cosine_similarity_matrix(latent_features)
        latent_cosine_values = latent_cosine_sim[np.triu_indices(n_samples, k=1)]
        
        # Plot correlation
        correlation_path = results_dir / f'{model_name.replace(" ", "_").lower()}_correlation.png'
        plot_correlation_scatter(input_cosine_values, latent_cosine_values, 
                               model_name, correlation_path, is_kl=False,
                               target_labels=target_labels)
        print(f'Saved correlation plot to {correlation_path}')
        
        # Heatmap
        heatmap_path = results_dir / f'{model_name.replace(" ", "_").lower()}_latent_heatmap.png'
        plot_similarity_heatmap(latent_cosine_sim, f'{model_name} Latent Space', 
                               heatmap_path, max_samples=max_samples_for_heatmap, 
                               original_indices=test_indices, is_kl=False)
        print(f'Saved latent space heatmap to {heatmap_path}')


def load_specific_model(model_name: str, model_type: str, config_dict: dict, full_dataset, device: torch.device,
                       test_features=None, test_labels=None):
    """
    Helper to construct a model architecture given its name and type.
    
    Args:
        model_name: Checkpoint/model name (e.g. 'mlp_embedding_frozen')
        model_type: One of {'simple', 'vae', 'wae'}
        config_dict: Dictionary returned from prepare_data_from_experiment
        full_dataset: Dataset object (used for decoder heads)
        device: torch.device
        test_features: Optional test features for TabPFN fit (numpy array)
        test_labels: Optional test labels for TabPFN fit (numpy array)
    """
    # Reuse existing helper to extract model configs
    model_configs = get_model_configs(config_dict)
    mlp_model_cfg = model_configs['mlp_model_cfg']
    tabm_model_cfg = model_configs['tabm_model_cfg']
    encoder_model_cfg = model_configs['encoder_model_cfg']
    wae_encoder_model_cfg = model_configs['wae_encoder_model_cfg']
    encoder_latent_dim = model_configs['encoder_latent_dim']
    wae_latent_dim = model_configs['wae_latent_dim']
    wae_hidden_dims = model_configs['wae_hidden_dims']
    
    input_dim = full_dataset.features.shape[1]
    
    # SIMPLE embedding models
    if model_type == 'simple':
        encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
        encoder_dropout = encoder_model_cfg.get('dropout', 0.1)
        
        if model_name.startswith('mlp'):
            encoder_batchnorm = encoder_model_cfg.get('batchnorm', True)
            encoder_activation = encoder_model_cfg.get('activation', 'ReLU')

            encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                dropout=encoder_dropout,
                batchnorm=encoder_batchnorm,
                activation=encoder_activation
            ).to(device)
            latent_dim = encoder_latent_dim
        elif model_name.startswith('tabnet'):
            tabnet_model_cfg = model_configs.get('tabnet_model_cfg', {})
            encoder = TabNetEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                n_d=tabnet_model_cfg.get('n_d', 8),
                n_a=tabnet_model_cfg.get('n_a', 8),
                n_steps=tabnet_model_cfg.get('n_steps', 3),
                gamma=tabnet_model_cfg.get('gamma', 1.5),
                dropout=encoder_dropout
            ).to(device)
            latent_dim = encoder_latent_dim
        # elif model_name.startswith('tabpfn'):
        #     tabpfn_model_cfg = model_configs.get('tabpfn_model_cfg', {})
        #     tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #     tabpfn_device = tabpfn_model_cfg.get('device', tabpfn_device)
        #     encoder = TabPFNEncoder(
        #         input_dim=input_dim,
        #         latent_dim=encoder_latent_dim,  # Not used, kept for interface consistency
        #         device=tabpfn_device,
        #         base_path=tabpfn_model_cfg.get('base_path', None)
        #     ).to(device)
        #     # TabPFN needs to be fitted to get actual embedding dimension
        #     # Use a subset of test data for fitting (will be re-fitted later with training data if available)
        #     if test_features is not None and test_labels is not None:
        #         print("  Fitting TabPFN encoder to determine embedding dimension...")
        #         fit_size = min(1000, len(test_features))
        #         encoder.fit(test_features[:fit_size], test_labels[:fit_size])
        #     else:
        #         # Fallback: use dummy data to get dimension
        #         print("  Warning: No test data provided. Using dummy data to determine TabPFN embedding dimension...")
        #         dummy_features = np.random.randn(10, input_dim)
        #         dummy_labels = np.random.randn(10)
        #         encoder.fit(dummy_features, dummy_labels)
        #     # Get actual embedding dimension
        #     with torch.no_grad():
        #         dummy_input = torch.zeros(1, input_dim, device=device)
        #         _ = encoder(dummy_input)  # Forward pass to cache embedding_dim
        #         actual_latent_dim = encoder.embedding_dim  # Use property instead of shape
        #     print(f"  TabPFN embedding dimension: {actual_latent_dim}")
        #     latent_dim = actual_latent_dim
        else:  # tabm
            encoder = TabMEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                k_heads=tabm_model_cfg.get('k_heads', 8),
                adapter_dim=tabm_model_cfg.get('adapter_dim', None),
                dropout=encoder_dropout
            ).to(device)
            latent_dim = encoder_latent_dim

        # Wrap with new EncoderEmbedding(encoder, latent_dim) structure
        return EncoderEmbedding(encoder, latent_dim).to(device)
    
    # VAE embedding models (encoder + VAE layers + regression head wrapped by VAEEncoderEmbedding)
    if model_type == 'vae':
        # if model_name.startswith('tabpfn'):
        #     raise ValueError("TabPFN does not support VAE models. Only frozen and gw cases are supported.")
        
        encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
        encoder_dropout = encoder_model_cfg.get('dropout', 0.1)

        if model_name.startswith('mlp'):
            encoder_batchnorm = encoder_model_cfg.get('batchnorm', True)
            encoder_activation = encoder_model_cfg.get('activation', 'ReLU')
            encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                dropout=encoder_dropout,
                batchnorm=encoder_batchnorm,
                activation=encoder_activation
            ).to(device)
        elif model_name.startswith('tabnet'):
            tabnet_model_cfg = model_configs.get('tabnet_model_cfg', {})
            encoder = TabNetEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                n_d=tabnet_model_cfg.get('n_d', 8),
                n_a=tabnet_model_cfg.get('n_a', 8),
                n_steps=tabnet_model_cfg.get('n_steps', 3),
                gamma=tabnet_model_cfg.get('gamma', 1.5),
                dropout=encoder_dropout
            ).to(device)
        else:  # tabm
            encoder = TabMEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=encoder_latent_dim,
                k_heads=tabm_model_cfg.get('k_heads', 8),
                adapter_dim=tabm_model_cfg.get('adapter_dim', None),
                dropout=encoder_dropout
            ).to(device)

        return VAEEncoderEmbedding(encoder, encoder_latent_dim).to(device)
    
    # WAE embedding models (TabAE-style encoder + regression head wrapped by EncoderEmbedding)
    if model_type == 'wae':
        encoder_hidden_dims = wae_hidden_dims
        encoder_dropout = wae_encoder_model_cfg.get('dropout', encoder_model_cfg.get('dropout', 0.1))

        if model_name.startswith('mlp'):
            encoder_batchnorm = wae_encoder_model_cfg.get('batchnorm', encoder_model_cfg.get('batchnorm', True))
            encoder_activation = wae_encoder_model_cfg.get('activation', encoder_model_cfg.get('activation', 'ReLU'))
            encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=wae_latent_dim,
                dropout=encoder_dropout,
                batchnorm=encoder_batchnorm,
                activation=encoder_activation
            ).to(device)
        elif model_name.startswith('tabnet'):
            tabnet_model_cfg = model_configs.get('tabnet_model_cfg', {})
            encoder = TabNetEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=wae_latent_dim,
                n_d=tabnet_model_cfg.get('n_d', 8),
                n_a=tabnet_model_cfg.get('n_a', 8),
                n_steps=tabnet_model_cfg.get('n_steps', 3),
                gamma=tabnet_model_cfg.get('gamma', 1.5),
                dropout=encoder_dropout
            ).to(device)
        else:  # tabm
            encoder = TabMEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=wae_latent_dim,
                k_heads=tabm_model_cfg.get('k_heads', 8),
                adapter_dim=tabm_model_cfg.get('adapter_dim', None),
                dropout=encoder_dropout
            ).to(device)

        return EncoderEmbedding(encoder, wae_latent_dim).to(device)
    
    raise NotImplementedError(f"load_specific_model not implemented for model_type='{model_type}'")


def process_model_group(model_list, model_type, checkpoints_dir, results_dir, 
                       test_features, test_labels, test_indices, 
                       input_cosine_values, max_samples, device, 
                       config_dict, full_dataset, train_features=None, train_labels=None):
    """
    Generic processor to reduce code duplication.
    
    Args:
        model_list: List of model names or tuples describing models
        model_type: 'simple', 'vae', 'wae', 'multi'
        train_features: Optional training features for TabPFN fit
        train_labels: Optional training labels for TabPFN fit
    """
    test_features_tensor = torch.from_numpy(test_features).float()
    n = test_features.shape[0]

    for model_info in model_list:
        # Unpack model info based on list structure
        if isinstance(model_info, str):
            model_name = model_info
            is_vae = ('vae' in model_name)
        else:
            model_name = model_info[0]
            is_vae = model_info[2]
            
        checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
        if not checkpoint_path.exists():
            continue
            
        print(f'Processing {model_name}...')
        
        try:
            # 1. Load Model (Logic abstracted)
            model = load_specific_model(model_name, model_type, config_dict, full_dataset, device,
                                       test_features=test_features, test_labels=test_labels)
            model = load_model_from_checkpoint(model, checkpoint_path, device)
            
            # # 2. Fit TabPFN encoder if needed
            # if model_name.startswith('tabpfn') and hasattr(model, 'encoder') and hasattr(model.encoder, 'fit'):
            #     if not model.encoder.is_fitted:
            #         print(f"  Fitting TabPFN encoder for {model_name}...")
            #         if train_features is not None and train_labels is not None:
            #             fit_size = min(1000, len(train_features))
            #             model.encoder.fit(train_features[:fit_size], train_labels[:fit_size])
            #         else:
            #             # Fallback: use test data subset
            #             print("  Warning: No training data provided. Using test data subset for TabPFN fit.")
            #             fit_size = min(1000, len(test_features))
            #             model.encoder.fit(test_features[:fit_size], test_labels[:fit_size])
            #     
            #     # Check if head dimension matches encoder output dimension (for TabPFN)
            #     if model_name.startswith('tabpfn') and hasattr(model, 'head'):
            #         with torch.no_grad():
            #             dummy_input = torch.zeros(1, test_features.shape[1], device=device)
            #             _ = model.encoder(dummy_input)  # Forward pass to cache embedding_dim
            #             actual_embedding_dim = model.encoder.embedding_dim  # Use property instead of shape
            #             head_input_dim = model.head.weight.shape[1]
            #             
            #             if head_input_dim != actual_embedding_dim:
            #                 print(f"  Warning: Head input dimension ({head_input_dim}) doesn't match encoder output ({actual_embedding_dim})")
            #                 print(f"  Recreating head with correct dimension...")
            #                 # Recreate head with correct dimension
            #                 model.head = torch.nn.Linear(actual_embedding_dim, 1).to(device)
            #                 # Try to load head weights if checkpoint has them (may fail, but that's ok)
            #                 try:
            #                     checkpoint = torch.load(checkpoint_path, map_location=device)
            #                     if 'head.weight' in checkpoint['model_state_dict']:
            #                         old_head_weight = checkpoint['model_state_dict']['head.weight']
            #                         old_head_bias = checkpoint['model_state_dict'].get('head.bias', None)
            #                         if old_head_weight.shape[0] == 1:  # Output dimension matches
            #                             # Only copy bias if it exists and matches
            #                             if old_head_bias is not None and old_head_bias.shape[0] == 1:
            #                                 model.head.bias.data = old_head_bias
            #                 except:
            #                     pass  # If loading fails, use random initialization
            
            # 3. Extract & Plot
            compute_and_plot_similarity(
                model, test_features_tensor, device, input_cosine_values,
                model_name, results_dir, test_labels, test_indices,
                max_samples, n, is_vae=is_vae
            )
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            import traceback
            traceback.print_exc()


def plot_similarity_distribution(cosine_sim: np.ndarray, save_path: Path, max_samples: int = None):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Limit to max_samples if provided (to match heatmap range)
    if max_samples is not None:
        n = min(max_samples, cosine_sim.shape[0])
        cosine_sim_subset = cosine_sim[:n, :n]
    else:
        n = cosine_sim.shape[0]
        cosine_sim_subset = cosine_sim
    
    # Extract upper triangle values (excluding diagonal) for cosine similarity
    cosine_values = cosine_sim_subset[np.triu_indices(n, k=1)]
    
    # Plot Cosine Similarity distribution
    ax.hist(cosine_values, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.axvline(cosine_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cosine_values.mean():.3f}')
    ax.axvline(np.median(cosine_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(cosine_values):.3f}')
    ax.set_xlabel('Cosine Similarity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Cosine Similarity Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_patient_similarity(experiment_dir: Path, max_samples_for_heatmap: int = 100):
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    print(f'\n{"="*80}')
    print(f'Starting patient similarity analysis: {experiment_dir.name}')
    print(f'Experiment directory: {experiment_dir}')
    print(f'{"="*80}\n')
    
    # Prepare data using the reusable function
    print("Preparing data...")
    result = prepare_data_from_experiment(
        experiment_dir=experiment_dir,
        max_samples=None,  # Use all samples for similarity analysis
        return_train=True  # Need training data for TabPFN fit
    )
    # Unpack result: test_features, test_labels, test_indices, full_dataset, config_dict, generator, train_features, train_labels, train_indices
    test_features, test_labels, test_indices, full_dataset, config_dict, generator = result[:6]
    if len(result) > 6:
        train_features, train_labels, train_indices = result[6:9]
    else:
        train_features, train_labels, train_indices = None, None, None
    
    # Get configurations
    full_config = config_dict['full_config']
    train_args = config_dict['train_args']
    training_cfg = config_dict['training_cfg']
    
    results_dir = experiment_dir / 'similarity_analysis_cosine'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n--- 1. Computing Cosine Similarity (Primary Metric) ---")
    # This captures the 'Angle' or 'Profile' similarity
    cosine_sim = compute_cosine_similarity_matrix(test_features)
    
    # --- 2. Visualization ---
    print("--- 2. Generating Plots ---")
    
    # Heatmap
    plot_similarity_heatmap(cosine_sim, "Cosine Similarity", results_dir / 'cosine_heatmap.png', 
                           max_samples=max_samples_for_heatmap, original_indices=test_indices)
    
    # Distribution Histogram (use same range as heatmap)
    plot_similarity_distribution(cosine_sim, results_dir / 'similarity_distributions.png', 
                                max_samples=max_samples_for_heatmap)
    
    # --- 3. Embedding Models Analysis ---
    print("\n--- 3. Analyzing Embedding Models ---")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get model configurations
    model_configs = get_model_configs(config_dict)
    mlp_model_cfg = model_configs['mlp_model_cfg']
    tabm_model_cfg = model_configs['tabm_model_cfg']
    encoder_model_cfg = model_configs['encoder_model_cfg']
    wae_encoder_model_cfg = model_configs['wae_encoder_model_cfg']
    encoder_latent_dim = model_configs['encoder_latent_dim']
    wae_latent_dim = model_configs['wae_latent_dim']
    wae_hidden_dims = model_configs['wae_hidden_dims']
    wae_regularization_type = model_configs['wae_regularization_type']
    input_dim = test_features.shape[1]
    
    print(f'WAE regularization type: {wae_regularization_type}')
    
    # Convert test features to tensor
    test_features_tensor = torch.from_numpy(test_features).float()
    
    # Extract upper triangle values for correlation plot
    n = cosine_sim.shape[0]
    input_cosine_values = cosine_sim[np.triu_indices(n, k=1)]
    
    checkpoints_dir = experiment_dir / 'checkpoints'
    
    # 1. Pretrained AE and VAE encoders (simple encoders only)
    print("\n--- 3.1. Pretrained Encoders (AE and VAE) ---")
    
    # Pretrained AE encoder
    encoder_checkpoint_path = checkpoints_dir / 'encoder_best.pt'
    if encoder_checkpoint_path.exists():
        try:
            print(f'Processing pretrained AE encoder...')
            encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
            encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
            encoder = create_tabae_model(input_dim, encoder_latent_dim, encoder_hidden_dims, 
                                        encoder_model_cfg, full_dataset, device, encoder_type='mlp', model_configs=model_configs)
            encoder = load_model_from_checkpoint(encoder, encoder_checkpoint_path, device)
            compute_and_plot_similarity(encoder, test_features_tensor, device, input_cosine_values,
                                       'Pretrained AE Encoder', results_dir, test_labels, test_indices,
                                       max_samples_for_heatmap, n, is_vae=False)
        except Exception as e:
            print(f'Error processing pretrained AE encoder: {str(e)}')
    
    # Pretrained VAE encoder
    vae_encoder_checkpoint_path = checkpoints_dir / 'vae_encoder_best.pt'
    if vae_encoder_checkpoint_path.exists():
        try:
            print(f'Processing pretrained VAE encoder...')
            encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
            vae_encoder = create_tabvae_model(input_dim, encoder_latent_dim, encoder_hidden_dims,
                                            encoder_model_cfg, full_dataset, device)
            vae_encoder = load_model_from_checkpoint(vae_encoder, vae_encoder_checkpoint_path, device)
            compute_and_plot_similarity(vae_encoder, test_features_tensor, device, input_cosine_values,
                                       'Pretrained VAE Encoder', results_dir, test_labels, test_indices,
                                       max_samples_for_heatmap, n, is_vae=True)
        except Exception as e:
            print(f'Error processing pretrained VAE encoder: {str(e)}')
    
    # Pretrained WAE encoder (check for both old and new naming conventions)
    wae_encoder_checkpoint_path = checkpoints_dir / f'wae_encoder_{wae_regularization_type}_best.pt'
    if not wae_encoder_checkpoint_path.exists():
        wae_encoder_checkpoint_path = checkpoints_dir / 'wae_encoder_best.pt'
    
    if wae_encoder_checkpoint_path.exists():
        try:
            print(f'Processing pretrained WAE encoder...')
            wae_encoder = create_tabae_model(input_dim, wae_latent_dim, wae_hidden_dims,
                                            wae_encoder_model_cfg, full_dataset, device)
            wae_encoder = load_model_from_checkpoint(wae_encoder, wae_encoder_checkpoint_path, device)
            model_name = f'Pretrained WAE Encoder ({wae_regularization_type.upper()})'
            compute_and_plot_similarity(wae_encoder, test_features_tensor, device, input_cosine_values,
                                       model_name, results_dir, test_labels, test_indices,
                                       max_samples_for_heatmap, n, is_vae=False)
        except Exception as e:
            print(f'Error processing pretrained WAE encoder: {str(e)}')
            import traceback
            traceback.print_exc()
    
    # 2. Simple embedding models (input cosine vs latent cosine) - Frozen and Fine-tuned versions
    print("\n--- 3.2. Simple Embedding Models (Frozen and Fine-tuned) ---")
    simple_embedding_models = [
        'mlp_embedding_frozen', 'tabm_embedding_frozen', 'tabnet_embedding_frozen',  # 'tabpfn_embedding_frozen',
        'mlp_embedding_finetuned', 'tabm_embedding_finetuned', 'tabnet_embedding_finetuned',  # 'tabpfn_embedding_finetuned'
    ]
    process_model_group(
        simple_embedding_models,
        model_type='simple',
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        input_cosine_values=input_cosine_values,
        max_samples=max_samples_for_heatmap,
        device=device,
        config_dict=config_dict,
        full_dataset=full_dataset,
        train_features=train_features,
        train_labels=train_labels
    )
    
    # 3. VAE embedding models (input cosine vs latent KL divergence) - Frozen and Fine-tuned versions
    print("\n--- 3.3. VAE Embedding Models (Frozen and Fine-tuned) ---")
    vae_embedding_models = [
        'mlp_vae_embedding_frozen', 'tabm_vae_embedding_frozen', 'tabnet_vae_embedding_frozen',
        'mlp_vae_embedding_finetuned', 'tabm_vae_embedding_finetuned', 'tabnet_vae_embedding_finetuned'
    ]
    process_model_group(
        vae_embedding_models,
        model_type='vae',
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        input_cosine_values=input_cosine_values,
        max_samples=max_samples_for_heatmap,
        device=device,
        config_dict=config_dict,
        full_dataset=full_dataset,
        train_features=train_features,
        train_labels=train_labels
    )
    
    # 4. WAE embedding models (input cosine vs latent cosine) - Frozen and Fine-tuned versions
    print("\n--- 3.4. WAE Embedding Models (Frozen and Fine-tuned) ---")
    wae_embedding_models = [
        'mlp_wae_embedding_frozen', 'tabm_wae_embedding_frozen', 'tabnet_wae_embedding_frozen',
        'mlp_wae_embedding_finetuned', 'tabm_wae_embedding_finetuned', 'tabnet_wae_embedding_finetuned'
    ]
    process_model_group(
        wae_embedding_models,
        model_type='wae',
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        input_cosine_values=input_cosine_values,
        max_samples=max_samples_for_heatmap,
        device=device,
        config_dict=config_dict,
        full_dataset=full_dataset,
        train_features=train_features,
        train_labels=train_labels
    )
    
    # 4.5. GW models (input cosine vs latent cosine) - Direct training, no frozen/finetuned versions
    print("\n--- 3.4.5. GW Models ---")
    gw_models = [
        'mlp_gw', 'tabm_gw', 'tabnet_gw',  # 'tabpfn_gw'
    ]
    process_model_group(
        gw_models,
        model_type='simple',  # GW uses same structure as simple embedding (encoder_latent_dim, EncoderEmbedding)
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        input_cosine_values=input_cosine_values,
        max_samples=max_samples_for_heatmap,
        device=device,
        config_dict=config_dict,
        full_dataset=full_dataset,
        train_features=train_features,
        train_labels=train_labels
    )
    
    # 5. Multi-task embedding models (need to load models including WAE, but not GW)
    print("\n--- 3.5. Multi-task Embedding Models ---")
    multitask_models = [
        ('mlp_multi_task_embedding', 'MulltiTaskEnecoderEmbedding', False, None),
        ('tabm_multi_task_embedding', 'MulltiTaskEnecoderEmbedding', False, None),
        ('tabnet_multi_task_embedding', 'MulltiTaskEnecoderEmbedding', False, None),
        
        ('mlp_multi_vae_task_embedding', 'MultitaskVAEEncoderEmbedding', True, None),
        ('tabm_multi_vae_task_embedding', 'MultitaskVAEEncoderEmbedding', True, None),
        ('tabnet_multi_vae_task_embedding', 'MultitaskVAEEncoderEmbedding', True, None),
        # Multi-task WAE models: try both with and without regularization type suffix
        ('mlp_multi_wae_task_embedding', 'MulltiTaskEnecoderEmbedding', False, 'wae'),
        ('tabm_multi_wae_task_embedding', 'MulltiTaskEnecoderEmbedding', False, 'wae'),
        ('tabnet_multi_wae_task_embedding', 'MulltiTaskEnecoderEmbedding', False, 'wae'),
        
        # Note: GW models don't have multi-task versions (they train directly with regression + GW loss)
    ]
    
    for model_name, model_type, is_vae, encoder_type in multitask_models:
        # For multi-task WAE models, try both naming conventions:
        # 1. Without suffix: mlp_multi_wae_task_embedding_best.pt (actual format from train.py)
        # 2. With suffix: mlp_multi_wae_task_embedding_{wae_regularization_type}_best.pt (alternative format)
        if encoder_type == 'wae':
            checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
            if not checkpoint_path.exists():
                # Try with regularization type suffix as fallback
                checkpoint_path_with_suffix = checkpoints_dir / f'{model_name}_{wae_regularization_type}_best.pt'
                if checkpoint_path_with_suffix.exists():
                    checkpoint_path = checkpoint_path_with_suffix
                else:
                    print(f'Warning: Checkpoint not found for {model_name} (tried both with and without suffix), skipping...')
                    continue
        else:
            checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
            if not checkpoint_path.exists():
                print(f'Warning: Checkpoint not found for {model_name}, skipping...')
                continue
        
        try:
            print(f'Processing {model_name}...')
            
          
            
            is_mlp = model_name.startswith('mlp')
            is_tabnet = model_name.startswith('tabnet')
            
            if is_mlp:
                encoder_prefix = 'mlp'
            elif is_tabnet:
                encoder_prefix = 'tabnet'
            else:
                encoder_prefix = 'tabm'
            
            # Note: GW models don't have pretrained checkpoints (they train directly)
            if encoder_type == 'gw':
                print(f'Warning: GW models don\'t have pretrained checkpoints. Skipping {model_name}')
                continue
            elif encoder_type == 'wae':
                pretrained_checkpoint_path = checkpoints_dir / f'{encoder_prefix}_tabwae_best.pt'
            elif is_vae:
                pretrained_checkpoint_path = checkpoints_dir / f'{encoder_prefix}_tabvae_best.pt'
            else:
                pretrained_checkpoint_path = checkpoints_dir / f'{encoder_prefix}_tabae_best.pt'
            
            if not pretrained_checkpoint_path.exists():
                model_type = 'GW' if encoder_type == 'gw' else 'WAE' if encoder_type == 'wae' else 'VAE' if is_vae else 'AE'
                print(f'Warning: {model_type} pretrained checkpoint not found: {pretrained_checkpoint_path.name}, skipping {model_name}')
                continue
            
            # Use current dataset dimensions for decoder
            # Note: If checkpoint was trained with different dimensions, filter_state_dict will handle mismatches
            decoder_n_continuous = full_dataset.n_continuous
            decoder_n_binary = full_dataset.n_binary
            decoder_cat_sizes = full_dataset.cat_sizes
            
            # Create model structure using pretrained model (only for architecture, not weights)
            if encoder_type == 'wae' or encoder_type == 'gw':
                wae_model = create_tabae_model(input_dim, wae_latent_dim, wae_hidden_dims,
                                              wae_encoder_model_cfg, full_dataset, device,
                                              encoder_type=encoder_prefix, model_configs=model_configs)
                wae_model = load_model_from_checkpoint(wae_model, pretrained_checkpoint_path, device)
                encoder = wae_model.encoder
                # Recreate decoder with correct dimensions from checkpoint
                decoder_hidden_dims = list(reversed(wae_hidden_dims))
                encoder_dropout = wae_encoder_model_cfg.get('dropout', encoder_model_cfg.get('dropout', 0.1))
                decoder = TabularDecoder(
                    latent_dim=wae_latent_dim,
                    hidden_dims=decoder_hidden_dims,
                    n_continuous=decoder_n_continuous,
                    n_binary=decoder_n_binary,
                    cat_sizes=decoder_cat_sizes,
                    dropout=encoder_dropout
                ).to(device)
                latent_dim = wae_latent_dim
               
            elif is_vae:
                encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
                vae_model = create_tabvae_model(input_dim, encoder_latent_dim, encoder_hidden_dims,
                                               encoder_model_cfg, full_dataset, device,
                                               encoder_type=encoder_prefix, model_configs=model_configs)
                vae_model = load_model_from_checkpoint(vae_model, pretrained_checkpoint_path, device)
                encoder = vae_model.encoder
                # Recreate decoder with correct dimensions from checkpoint
                decoder_hidden_dims = list(reversed(encoder_hidden_dims))
                encoder_dropout = encoder_model_cfg.get('dropout', 0.1)
                decoder = TabularDecoder(
                    latent_dim=encoder_latent_dim,
                    hidden_dims=decoder_hidden_dims,
                    n_continuous=decoder_n_continuous,
                    n_binary=decoder_n_binary,
                    cat_sizes=decoder_cat_sizes,
                    dropout=encoder_dropout
                ).to(device)
                latent_dim = encoder_latent_dim
            else:
                encoder_hidden_dims = encoder_model_cfg.get('hidden_dims', [32, 16])
                ae_model = create_tabae_model(input_dim, encoder_latent_dim, encoder_hidden_dims,
                                             encoder_model_cfg, full_dataset, device,
                                             encoder_type=encoder_prefix, model_configs=model_configs)
                ae_model = load_model_from_checkpoint(ae_model, pretrained_checkpoint_path, device)
                encoder = ae_model.encoder
                # Recreate decoder with correct dimensions from checkpoint
                decoder_hidden_dims = list(reversed(encoder_hidden_dims))
                encoder_dropout = encoder_model_cfg.get('dropout', 0.1)
                decoder = TabularDecoder(
                    latent_dim=encoder_latent_dim,
                    hidden_dims=decoder_hidden_dims,
                    n_continuous=decoder_n_continuous,
                    n_binary=decoder_n_binary,
                    cat_sizes=decoder_cat_sizes,
                    dropout=encoder_dropout
                ).to(device)
                latent_dim = encoder_latent_dim
            
            # Create multi-task model structure
            if is_vae:
                model = MultitaskVAEEncoderEmbedding(encoder, decoder, latent_dim).to(device)
            else:
                model = MulltiTaskEnecoderEmbedding(encoder, decoder, latent_dim).to(device)
            
            # Load weights from multitask model's own checkpoint (includes trained encoder, decoder, and head)
            # Use filter_state_dict to handle structure mismatches gracefully
            checkpoint = torch.load(checkpoint_path, map_location=device)
            filtered_state_dict = filter_state_dict(checkpoint['model_state_dict'], model, strict=False)
            model.load_state_dict(filtered_state_dict, strict=False)
            
            model.eval()
            
            # Extract latent and compute metric
            if is_vae:
                # For VAE: use KL divergence
                mu, log_var = extract_vae_latent_distribution(model, test_features_tensor, device)
                latent_kl = compute_kl_divergence_matrix_from_vae(mu, log_var)
                latent_kl_values = latent_kl[np.triu_indices(n, k=1)]
                # Also compute cosine similarity of mu for heatmap and correlation
                latent_cosine_sim = compute_cosine_similarity_matrix(mu)
                mu_cosine_values = latent_cosine_sim[np.triu_indices(n, k=1)]
                
                # Plot correlation with KL divergence
                correlation_path = results_dir / f'{model_name}_correlation_kl.png'
                plot_correlation_scatter(input_cosine_values, latent_kl_values, 
                                       f'{model_name.replace("_", " ").title()} (KL Divergence)', correlation_path, is_kl=True,
                                       target_labels=test_labels)
                print(f'Saved correlation plot (KL) to {correlation_path}')
                
                # Plot correlation with cosine similarity of mu
                correlation_path = results_dir / f'{model_name}_correlation_cosine.png'
                plot_correlation_scatter(input_cosine_values, mu_cosine_values, 
                                       f'{model_name.replace("_", " ").title()} (Cosine Similarity)', correlation_path, is_kl=False,
                                       target_labels=test_labels)
                print(f'Saved correlation plot (Cosine) to {correlation_path}')
            else:
                # For AE and WAE: use cosine similarity
                latent_features = extract_latent_representation(model, test_features_tensor, device)
                latent_cosine_sim = compute_cosine_similarity_matrix(latent_features)
                latent_metric_values = latent_cosine_sim[np.triu_indices(n, k=1)]
                
                # Determine title suffix based on encoder type
                if encoder_type == 'wae':
                    title_suffix = f' ({wae_regularization_type.upper()})'
                elif encoder_type == 'gw':
                    title_suffix = ' (GW)'
                else:
                    title_suffix = ''
                
                # Plot correlation
                correlation_path = results_dir / f'{model_name}_correlation.png'
                plot_correlation_scatter(input_cosine_values, latent_metric_values, 
                                       f'{model_name.replace("_", " ").title()}{title_suffix}', correlation_path, is_kl=False,
                                       target_labels=test_labels)
                print(f'Saved correlation plot to {correlation_path}')
            
            # Plot latent space heatmap
            heatmap_path = results_dir / f'{model_name}_latent_heatmap.png'
            if is_vae:
                heatmap_title_suffix = ' (mu)'
            elif encoder_type == 'wae':
                heatmap_title_suffix = f' ({wae_regularization_type.upper()})'
            elif encoder_type == 'gw':
                heatmap_title_suffix = ' (GW)'
            else:
                heatmap_title_suffix = ''
            plot_similarity_heatmap(latent_cosine_sim, f'{model_name.replace("_", " ").title()} Latent Space{heatmap_title_suffix}', 
                                   heatmap_path, max_samples=max_samples_for_heatmap, 
                                   original_indices=test_indices, is_kl=False)
            print(f'Saved latent space heatmap to {heatmap_path}')
            
        except Exception as e:
            print(f'Error processing {model_name}: {str(e)}')
            import traceback
            traceback.print_exc()
    
    print(f"\nDone. Results saved to {results_dir}")



def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Patient Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze single experiment
  python Similarity_analysis.py --experiment_dir output/20241105_001311
  
  # Specify maximum samples for heatmap
  python Similarity_analysis.py --experiment_dir output/20241105_001311 --max_samples 50
        """
    )
    
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment output directory path (must contain checkpoints and configs subdirectories)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples for heatmap (default: 100)')
    
    args = parser.parse_args()
    
    analyze_patient_similarity(
        experiment_dir=Path(args.experiment_dir),
        max_samples_for_heatmap=args.max_samples
    )


if __name__ == '__main__':
    main()

