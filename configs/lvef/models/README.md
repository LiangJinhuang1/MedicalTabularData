# Model Configuration Files

This directory contains unified configuration files for all model architectures.

## New Unified Structure

### `base_models.yaml`
Configuration for standalone base models (direct regression, no pre-training).

**Models:**
- **MLP**: MLPEncoder + EncoderEmbedding
- **TabM**: TabM-mini (local) architecture

**Structure:**
```yaml
base_models:
  mlp:
    encoder:
      hidden_dims: [256, 128, 64]
      latent_dim: 32
      dropout: 0.3
      batchnorm: true
      activation: 'ReLU'
  tabm:
    hidden_dims: [256, 128, 64]
    k_heads: 6
    dropout: 0.3
    activation: 'ReLU'
```

### `pretraining.yaml`
Configuration for all pre-training models (TabAE, TabVAE, TabWAE).

**Structure:**
```yaml
pretraining:
  encoder:          # Shared encoder config for all pre-training models
    latent_dim: 32
    hidden_dims: [32, 16]
    dropout: 0.1
    batchnorm: true
    activation: 'ReLU'
  tabm_encoder:     # Optional TabM-specific override (falls back to encoder)
    latent_dim: 32
    dropout: 0.1
  
  ae:               # Autoencoder (TabAE) - no additional params
    pass
  
  vae:              # Variational Autoencoder (TabVAE)
    beta: 1.0       # KL divergence weight
  
  wae:              # Wasserstein Autoencoder (TabWAE)
    latent_dim: null  # Optional override
    regularization_type: 'sinkhorn'
    lambda_ot: 10.0
    # ... other WAE params
```

## Configuration Hierarchy

1. **Base Models** (`base_models.yaml`): Standalone models for direct regression
   - MLP: Uses encoder config within base_models.mlp
   - TabM: Direct TabM architecture config

2. **Pre-training** (`pretraining.yaml`): All pre-training model configurations
   - **Shared encoder**: Used by TabAE, TabVAE, and TabWAE
   - **Model-specific**: Each model type (AE/VAE/WAE) has its own section

## Model Architecture Mapping

| Model Type | Config Source |
|------------|---------------|
| MLP (base) | `base_models.mlp.encoder` |
| TabM (base) | `base_models.tabm` |
| TabAE (MLP encoder) | `pretraining.encoder` |
| TabAE (TabM encoder) | `pretraining.encoder` + `base_models.tabm` |
| TabVAE (MLP encoder) | `pretraining.encoder` + `pretraining.vae` |
| TabVAE (TabM encoder) | `pretraining.encoder` + `base_models.tabm` + `pretraining.vae` |
| TabWAE (MLP encoder) | `pretraining.encoder` + `pretraining.wae` |
| TabWAE (TabM encoder) | `pretraining.encoder` + `base_models.tabm` + `pretraining.wae` |

## Benefits of New Structure

1. **Unified Organization**: All base models in one file, all pre-training configs in another
2. **Clear Separation**: Base models vs. pre-training models are clearly separated
3. **Shared Config**: Encoder config is shared across all pre-training models
4. **Easy Override**: WAE can override specific parameters (like latent_dim) if needed
5. **Less Duplication**: No need for separate vae_encoder.yaml since VAE uses same encoder as AE

## Migration from Old Structure

**Old files:**
- `mlp.yaml` → Now in `base_models.yaml` under `base_models.mlp`
- `tabm.yaml` → Now in `base_models.yaml` under `base_models.tabm`
- `encoder.yaml` → Now in `pretraining.yaml` under `pretraining.encoder`
- `vae_encoder.yaml` → Removed (uses `pretraining.encoder`)
- `wae_encoder.yaml` → Now in `pretraining.yaml` under `pretraining.wae`

**Code updates:**
- `main.py` now loads unified configs
- `train.py` extracts configs from unified structure
