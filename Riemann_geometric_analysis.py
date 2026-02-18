import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import warnings
import argparse
import gc
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr, pearsonr
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import logm as scipy_logm, expm as scipy_expm
from torch.func import jacrev, vmap
import umap

from src.data.prepare_data import prepare_data
from src.utils.config import get_variable_types, load_experiment_config
from src.utils.encoder_utils import encode_with_entropy
from src.utils import model_factory

warnings.filterwarnings('ignore')

def _load_analysis_config(full_config: dict):
    """Validate config sections needed for analysis."""
    train_args = full_config.get('train_args')
    if not train_args or not isinstance(train_args, dict):
        raise ValueError("train_args not found in full_config or is invalid")

    data_config = full_config.get('data_config')
    if not data_config or not isinstance(data_config, dict):
        raise ValueError("data_config not found in full_config or is invalid")

    paths_cfg = full_config.get('paths', {})
    train_file = paths_cfg.get('train_file')
    target_col = paths_cfg.get('target_col', 'LVEF_dis')
    if train_file is None:
        raise ValueError("train_file path not found in configuration file")

    exclude_cols = full_config.get('exclude_cols', [])
    return full_config, train_args, data_config, paths_cfg, train_file, target_col, exclude_cols


def _list_embedding_models(checkpoints_dir: Path) -> list[str]:
    """List candidate embedding models (exclude pretraining and TabPFN)."""
    available_models = []
    for cp in checkpoints_dir.glob('*_best.pt'):
        name = cp.stem.replace('_best', '')
        if (('embedding' in name or 'multi' in name or 'gw' in name) and 
            not any(x in name for x in ['encoder', 'vae_encoder', 'wae_encoder', 
                                       'tabae', 'tabvae', 'tabwae', 'tabgw']) and
            not name.startswith('tabpfn')):
            available_models.append(name)
    return sorted(available_models)


def set_plot_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_palette("colorblind")

def _save_figure(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")

def _safe_k_neighbors(k, n, min_k=None):
    """
    Return k for k-NN, capped by n. For Riemannian graph, use min_k=15..20
    so the graph stays well-connected and curvature is approximated well.
    """
    if n < 2:
        return 0
    if min_k is not None:
        k = max(k, min_k)
    return min(k, n)

def _select_latent_tensor(outputs):
    tensors = [item for item in outputs if isinstance(item, torch.Tensor)]
    if not tensors:
        return None
    for t in tensors:
        if t.dim() == 2 and t.shape[1] > 1:
            return t
    return tensors[0]

def _bin_labels(labels, threshold):
    if threshold is None:
        return None, None
    labels = np.asarray(labels).flatten()
    binned = (labels >= threshold).astype(int)
    t_str = f"{threshold:g}"
    bin_names = [f"< {t_str}", f"≥ {t_str}"]
    return binned, bin_names

def _bin_pair_categories(labels, threshold):
    if threshold is None:
        return None, None, None
    labels = np.asarray(labels).flatten()
    low = labels < threshold
    return low, ~low, threshold

def extract_regressor_func(model):
    """Extract regressor function."""
    head = getattr(model, 'head', None) or getattr(model, 'regressor', None)
    if head is None:
        return None
    return head

def extract_latent_z(model, x):
    """Extract latent representation z."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encoder'):
            z_features, _ = encode_with_entropy(model.encoder, x)
            if hasattr(model, 'mu'):
                try:
                    z = model.mu(z_features)
                except Exception:
                    z = z_features
            else:
                z = z_features
        else:
            out = model(x)
            if isinstance(out, tuple):
                z = _select_latent_tensor(out)
            else:
                z = out
        
        # Ensure z is a tensor, not a tuple (defensive check)
        if isinstance(z, tuple):
            # If z is still a tuple, extract the first tensor element
            z = next((item for item in z if isinstance(item, torch.Tensor)), z[0] if len(z) > 0 else None)
        if z is None or not isinstance(z, torch.Tensor):
            raise ValueError(f"Could not extract tensor from model output: {type(z)}")
        
        return z

def compute_metric_tensor_batch(z_batch, regressor_func, device, epsilon=1e-2):
    """Compute metric tensor G(z) = J_reg^T @ J_reg.
    """
    def func_reg(z_vec):
        return regressor_func(z_vec.unsqueeze(0)).squeeze(0)

    # 1. Compute Jacobian (Batch, 1, Latent_Dim)
    J_reg = vmap(jacrev(func_reg))(z_batch)
    
    # 2. Compute Metric (Batch, Latent, Latent)
    # G_task captures "Risk Distance"
    G_task = torch.einsum('bni,bnj->bij', J_reg, J_reg)
    
    # 3. Add Structural Regularization (Crucial for Decoder-less models)
    # Without this, distance between different patients with same risk is 0.
    # epsilon=1e-2: ~1 unit Risk change ≈ 10 units Euclidean
    G_struct = torch.eye(G_task.shape[-1], device=device).unsqueeze(0)
    
    G = G_task + epsilon * G_struct
    
    return G

def distance_correlation(D_z, D_y):
    """
    Distance Correlation between two pairwise distance matrices.
    dCor(D_z, D_y) = dCov(D_z, D_y) / sqrt(dVar(D_z) * dVar(D_y)).
    Uses double-centering of the distance matrices (Szekely et al.).
    """
    D_z = np.asarray(D_z, dtype=float)
    D_y = np.asarray(D_y, dtype=float)
    n = D_z.shape[0]
    if D_z.shape != (n, n) or D_y.shape != (n, n):
        raise ValueError("D_z and D_y must be (n, n) distance matrices with same n.")
    # Double-center: A_ij = D_ij - row_mean_i - col_mean_j + total_mean
    def double_center(D):
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        total_mean = D.mean()
        return D - row_mean - col_mean + total_mean
    A = double_center(D_z)
    B = double_center(D_y)
    # dCov^2 = (1/n^2) * sum_ij(A_ij * B_ij), dVar^2 = (1/n^2) * sum_ij(A_ij^2)
    dCov2 = (A * B).sum() / (n * n)
    dVar2_z = (A * A).sum() / (n * n)
    dVar2_y = (B * B).sum() / (n * n)
    if dVar2_z <= 0 or dVar2_y <= 0:
        return 0.0
    dCov = max(0.0, dCov2) ** 0.5
    dVar_z = dVar2_z ** 0.5
    dVar_y = dVar2_y ** 0.5
    dCor = dCov / (dVar_z * dVar_y)
    return float(np.clip(dCor, 0.0, 1.0))


def compute_feature_target_geometry_alignment(z_data, y_targets, geo_matrix=None, use_geodesic=False):
    """
    Feature-Target Geometry Alignment: correlation between latent distances and target distances.
    D_z: pairwise distance matrix of latent embeddings (Euclidean or geodesic).
    D_y: pairwise distance matrix of regression targets, i.e. |y_i - y_j|.
    Returns dict with distance_correlation, spearman, pearson, and the flat vectors for plotting.
    """
    if isinstance(z_data, torch.Tensor):
        z_np = z_data.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_data)
    y = np.asarray(y_targets).flatten()
    n = z_np.shape[0]
    if n != len(y):
        raise ValueError(f"z_data and y_targets length mismatch: {n} vs {len(y)}")
    # D_y: pairwise absolute difference of targets
    y_2d = y.reshape(-1, 1)
    D_y = euclidean_distances(y_2d)  # |y_i - y_j|
    if use_geodesic and geo_matrix is not None:
        D_z = np.asarray(geo_matrix, dtype=float)
    else:
        D_z = euclidean_distances(z_np)
    # Upper triangle indices (excluding diagonal)
    triu = np.triu_indices(n, k=1)
    flat_D_z = D_z[triu]
    flat_D_y = D_y[triu]
    dCor = distance_correlation(D_z, D_y)
    r_spearman, p_spearman = spearmanr(flat_D_z, flat_D_y)
    r_pearson, p_pearson = pearsonr(flat_D_z, flat_D_y)
    return {
        "distance_correlation": dCor,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "flat_D_z": flat_D_z,
        "flat_D_y": flat_D_y,
        "D_z": D_z,
        "D_y": D_y,
    }


def plot_feature_target_alignment(alignment_result, save_dir, model_name, use_geodesic=False):
    """Scatter plot: pairwise latent distance vs pairwise target distance, with dCor and Spearman."""
    set_plot_style()
    flat_D_z = alignment_result["flat_D_z"]
    flat_D_y = alignment_result["flat_D_y"]
    dCor = alignment_result["distance_correlation"]
    r_sp = alignment_result["spearman_r"]
    p_sp = alignment_result["spearman_p"]
    # Subsample for plotting if too many points
    max_pts = 5000
    if len(flat_D_z) > max_pts:
        idx = np.random.default_rng(42).choice(len(flat_D_z), max_pts, replace=False)
        flat_D_z = flat_D_z[idx]
        flat_D_y = flat_D_y[idx]
    # Keep only finite values for plotting/fit
    finite_mask = np.isfinite(flat_D_z) & np.isfinite(flat_D_y)
    flat_D_z = flat_D_z[finite_mask]
    flat_D_y = flat_D_y[finite_mask]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(flat_D_z, flat_D_y, s=4, alpha=0.35, c="#2ca02c", edgecolors="none")
    if flat_D_z.size >= 2 and np.nanstd(flat_D_z) > 0:
        try:
            m, b = np.polyfit(flat_D_z, flat_D_y, 1)
            x_line = np.linspace(flat_D_z.min(), flat_D_z.max(), 100)
            ax.plot(x_line, m * x_line + b, color="#d62728", linestyle="--", linewidth=2, alpha=0.85)
        except Exception as e:
            ax.text(0.05, 0.05, f"Fit skipped: {type(e).__name__}", transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))
    else:
        ax.text(0.05, 0.05, "Fit skipped: insufficient variation", transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))
    dist_label = "Geodesic (Riemannian)" if use_geodesic else "Euclidean (Latent)"
    ax.set_xlabel(f"Pairwise Distance in Latent Space ({dist_label})")
    ax.set_ylabel("Pairwise Distance in Target Space |y_i - y_j|")
    title = f"Feature-Target Geometry Alignment: {model_name.upper()}"
    ax.set_title(title, pad=10)
    p_text = "$p$ < 0.001" if p_sp < 0.001 else f"$p$ = {p_sp:.3f}"
    ax.text(0.05, 0.95, f"Distance Correlation: {dCor:.4f}\nSpearman $\\rho$: {r_sp:.4f}\n{p_text}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5))
    sns.despine(ax=ax)
    plt.tight_layout()
    suffix = "geodesic" if use_geodesic else "euclidean"
    _save_figure(fig, save_dir / f"{model_name}_feature_target_alignment_{suffix}.png")
    plt.close()


def compute_geodesic_matrix(z_data, G_all, k=15, use_log_euclidean_mean=False):
    """
    Compute geodesic distance matrix using Riemannian graph algorithm.
    
    This method uses a more efficient graph-based approach:
    1. Find k-nearest neighbors for each point
    2. Compute local Riemannian distances only between k-NN pairs (sparse matrix)
    3. Use Dijkstra's algorithm to compute global shortest paths
    
    Args:
        z_data: Latent representations (N, D) as torch.Tensor
        G_all: Metric tensors for all points (N, D, D) as torch.Tensor
        k: Number of nearest neighbors (default 15; for Riemannian graph use 15--20).
        use_log_euclidean_mean: If True, use log-Euclidean mean for G_avg (default False).
    
    Returns:
        geo_matrix: Geodesic distance matrix (N, N) as numpy array
    """
    N = z_data.shape[0]
    k = _safe_k_neighbors(k, N, min_k=15)  # ensure well-connected graph for curvature
    if k < 2:
        return np.zeros((N, N))
    z_numpy = z_data.detach().cpu().numpy()
    
    # 1. Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(z_numpy)
    distances, indices = nbrs.kneighbors(z_numpy)
    
    # 2. Build Riemannian sparse adjacency matrix
    adj_matrix = np.zeros((N, N))
    print("  Building Riemannian Graph...")
    
    for i in tqdm(range(N), desc="Graph Construction"):
        for j_idx, j in enumerate(indices[i]):
            if i == j:
                continue
            
            # Local metric: arithmetic mean is a good approximation for small edges
            if use_log_euclidean_mean:
                Gi = G_all[i].detach().cpu().numpy()
                Gj = G_all[j].detach().cpu().numpy()
                log_gi = scipy_logm(Gi)
                log_gj = scipy_logm(Gj)
                G_avg_np = scipy_expm(0.5 * (log_gi + log_gj))
                G_avg = torch.from_numpy(G_avg_np).to(device=G_all.device, dtype=G_all.dtype)
            else:
                G_avg = 0.5 * (G_all[i] + G_all[j])
            diff = (z_data[i] - z_data[j]).unsqueeze(1)
            dist_sq = diff.T @ G_avg @ diff
            dist = torch.sqrt(torch.clamp(dist_sq, min=1e-9)).item()
            
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist  # Ensure symmetry
    
    # 3. Compute global geodesic distances using graph search
    print("  Solving All-Pairs Shortest Paths (Dijkstra)...")
    geo_matrix = shortest_path(csgraph=adj_matrix, directed=False, method='D')
    
    # Handle isolated points (if any exist in the graph)
    if np.any(np.isinf(geo_matrix)):
        finite_values = geo_matrix[~np.isinf(geo_matrix)]
        if len(finite_values) > 0:
            max_finite = np.nanmax(finite_values)
            geo_matrix[np.isinf(geo_matrix)] = max_finite * 2
        else:
            # Fallback: use a large constant if all values are infinite
            geo_matrix[np.isinf(geo_matrix)] = 1e6
    
    return geo_matrix

def compute_geodesic_distance_matrix(z_data, regressor_func, device, k_neighbors=15):
    """
    Compute pairwise Riemannian geodesic distances.
    
    Args:
        z_data: Latent representations
        regressor_func: Regressor function for computing metric tensors
        device: Computation device
        k_neighbors: Number of neighbors for graph method (default: 15)
    
    Returns:
        dist_matrix: Geodesic distance matrix (N, N)
        G_all: Metric tensors (N, D, D)
    """
    # Ensure z_data is a tensor, not a tuple
    if isinstance(z_data, tuple):
        # Extract the first tensor element from tuple
        z_data = next((item for item in z_data if isinstance(item, torch.Tensor)), z_data[0] if len(z_data) > 0 else None)
        if z_data is None or not isinstance(z_data, torch.Tensor):
            raise ValueError(f"z_data must be a tensor, but got {type(z_data)}")
    
    z_data = z_data.to(device)
    N = z_data.shape[0]
    
    print(f"  Computing metric tensors (N={N})...")
    
    # 1. Compute Metric Tensors
    G_list = []
    batch_size = 64  # Larger batch size is safe since J is small (1D output)
    
    with torch.set_grad_enabled(True):
        for i in tqdm(range(0, N, batch_size), desc="Metric Calculation"):
            batch = z_data[i:i+batch_size].detach().clone().requires_grad_(True)
            G_batch = compute_metric_tensor_batch(batch, regressor_func, device, epsilon=0.01)
            G_list.append(G_batch)  # Keep on GPU for speed
            
    G_all = torch.cat(G_list, dim=0)  # (N, D, D)
    
    # 2. Compute geodesic distances using graph-based method
    print("  Using graph-based geodesic computation...")
    dist_matrix = compute_geodesic_matrix(z_data, G_all, k=k_neighbors)
            
    return dist_matrix, G_all


def compute_input_geodesic_matrix(features, k=15):
    """
    Geodesic distance matrix in input space: k-NN graph with Euclidean edge weights,
    then all-pairs shortest path (graph geodesic).
    """
    features = np.asarray(features)
    n = features.shape[0]
    k = _safe_k_neighbors(k, n, min_k=2)
    if k < 2:
        return euclidean_distances(features)
    euc = euclidean_distances(features)
    nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(euc)
    distances, indices = nbrs.kneighbors(euc)
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                adj[i, j] = euc[i, j]
    geo = shortest_path(csgraph=np.nan_to_num(adj, nan=np.inf, posinf=np.inf), directed=False, method="D")
    if np.any(np.isinf(geo)):
        finite = geo[~np.isinf(geo)]
        if len(finite) > 0:
            geo[np.isinf(geo)] = np.nanmax(finite) * 2
    return geo


def plot_distortion_validation(test_features, z_data, geo_matrix, regressor_func, save_dir, model_name, test_labels, max_pairs=None, input_geodesic_k=15, bin_threshold=None):
    """
    Validation Plot: three panels, all colored by true target difference.
    1) Input space geodesic vs Latent space (Riemannian) geodesic.
    2) Input cosine vs Latent cosine.
    3) Input cosine vs Latent geodesic.
    """
    print("  Generating Validation Plot...")

    set_plot_style()

    n = test_features.shape[0]
    labels = np.asarray(test_labels)[:n].flatten()
    indices = np.triu_indices(n, k=1)
    c_plot = np.abs(labels[indices[0]] - labels[indices[1]])
    pair_colors = None
    pair_handles = None
    if bin_threshold is not None:
        low_mask, high_mask, t_val = _bin_pair_categories(labels, bin_threshold)
        low_i = low_mask[indices[0]]
        low_j = low_mask[indices[1]]
        both_low = low_i & low_j
        both_high = (~low_i) & (~low_j)
        pair_cat = np.where(both_low, 0, np.where(both_high, 1, 2))
        palette = ["#1f77b4", "#d62728", "#7f7f7f"]  # low-low, high-high, mixed
        pair_colors = np.take(palette, pair_cat)
        t_str = f"{t_val:g}"
        pair_labels = [f"both < {t_str}", f"both ≥ {t_str}", "mixed"]
        pair_handles = [Patch(facecolor=palette[i], edgecolor="none", label=pair_labels[i]) for i in range(3)]

    # Panel 1: Input-space geodesic (x) vs Latent-space geodesic (y)
    geo_input = compute_input_geodesic_matrix(test_features, k=input_geodesic_k)
    flat_geo_input = geo_input[indices]
    flat_geo = geo_matrix[indices]

    # Panel 2: Input cosine vs Latent cosine
    input_cos = cosine_distances(test_features)
    flat_input_cos = input_cos[indices]
    if isinstance(z_data, torch.Tensor):
        z_np = z_data.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_data)
    latent_cos = cosine_distances(z_np)
    flat_latent_cos = latent_cos[indices]

    if max_pairs is not None and len(c_plot) > max_pairs:
        idx = np.random.choice(len(c_plot), max_pairs, replace=False)
        flat_geo_input = flat_geo_input[idx]
        flat_geo = flat_geo[idx]
        flat_input_cos = flat_input_cos[idx]
        flat_latent_cos = flat_latent_cos[idx]
        c_plot = c_plot[idx]
        if pair_colors is not None:
            pair_colors = pair_colors[idx]

    r_geo_geo, p_geo_geo = spearmanr(flat_geo_input, flat_geo)
    r_ic_lc, p_ic_lc = spearmanr(flat_input_cos, flat_latent_cos)
    r_ic_lg, p_ic_lg = spearmanr(flat_input_cos, flat_geo)

    def _p_text(p):
        return "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"

    vmin, vmax = np.nanpercentile(c_plot, [5, 95])
    norm = None if vmin >= vmax else Normalize(vmin=vmin, vmax=vmax)
    cbar_label = 'Target Difference $|y_i - y_j|$ (true)'

    def _panel(ax, x_flat, y_flat, r, p, xlabel, ylabel, title):
        finite_mask = np.isfinite(x_flat) & np.isfinite(y_flat)
        x_plot = x_flat[finite_mask]
        y_plot = y_flat[finite_mask]
        if pair_colors is None:
            ax.scatter(x_plot, y_plot, c=c_plot[finite_mask], s=6, alpha=0.35, cmap="viridis", norm=norm, linewidths=0.0)
        else:
            ax.scatter(x_plot, y_plot, c=pair_colors[finite_mask], s=6, alpha=0.35, linewidths=0.0)
        if x_plot.size >= 2 and np.nanstd(x_plot) > 0:
            try:
                m, b = np.polyfit(x_plot, y_plot, 1)
                x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
                ax.plot(x_line, m * x_line + b, color='#d62728', linestyle='--', linewidth=2, alpha=0.85)
            except Exception as e:
                ax.text(0.05, 0.05, f"Fit skipped: {type(e).__name__}", transform=ax.transAxes,
                        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))
        else:
            ax.text(0.05, 0.05, "Fit skipped: insufficient variation", transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.text(0.05, 0.95, f"Spearman $\\rho$: {r:.3f}\n{_p_text(p)}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
        sns.despine(ax=ax)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))
    _panel(ax1, flat_geo_input, flat_geo, r_geo_geo, p_geo_geo,
           "Input Space (Geodesic)", "Latent Space (Riemannian Geodesic)", "Input Geodesic vs Latent Geodesic")
    _panel(ax2, flat_input_cos, flat_latent_cos, r_ic_lc, p_ic_lc,
           "Input Space (Cosine)", "Latent Space (Cosine)", "Input Cosine vs Latent Cosine")
    _panel(ax3, flat_input_cos, flat_geo, r_ic_lg, p_ic_lg,
           "Input Space (Cosine)", "Latent Space (Riemannian Geodesic)", "Input Cosine vs Latent Geodesic")

    if bin_threshold is None:
        fig.suptitle(f"Distortion Validation: {model_name.upper()} (color = target difference)", fontsize=13, y=1.02)
        sm = ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax3, pad=0.02, shrink=0.7, location='right')
        cb.set_label(cbar_label, rotation=270, labelpad=18, fontsize=11)
        cb.outline.set_edgecolor('gray')
        cb.outline.set_linewidth(0.5)
    else:
        t_str = f"{bin_threshold:g}"
        fig.suptitle(f"Distortion Validation: {model_name.upper()} (color = target bins @ {t_str})", fontsize=13, y=1.02)
        if pair_handles is not None:
            ax3.legend(handles=pair_handles, loc="lower right", fontsize=9, frameon=True)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(direction='out', length=4, width=1)
    plt.tight_layout()
    if bin_threshold is None:
        _save_figure(fig, save_dir / f'{model_name}_validation_check.png')
    else:
        _save_figure(fig, save_dir / f'{model_name}_validation_check_binned.png')
    plt.close()
    print("  Saved Validation Plot.")


def plot_riemannian_manifold_pro(emb, labels, label_name, save_path, model_name, bin_threshold=None):

    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    binned, bin_names = _bin_labels(labels, bin_threshold)
    if binned is None:
        # Use same color settings as validation plot
        vmin, vmax = np.nanpercentile(labels, [5, 95])
        norm = None if vmin >= vmax else Normalize(vmin=vmin, vmax=vmax)
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1], c=labels, cmap="viridis",
            s=14, alpha=0.75, edgecolors='none', norm=norm
        )
    else:
        palette = ["#1f77b4", "#d62728"]
        cmap = ListedColormap(palette)
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1], c=binned, cmap=cmap,
            s=14, alpha=0.75, edgecolors='none', vmin=-0.5, vmax=1.5
        )
    
    # Set axis labels
    ax.set_xlabel('Latent Dimension 1 (UMAP)')
    ax.set_ylabel('Latent Dimension 2 (UMAP)')
    
    # Remove tick numbers (UMAP absolute values have no physical meaning)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if binned is None:
        # Place colorbar with enhanced label and formatting
        cb = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f'Target: {label_name}', weight='bold')
        cb.ax.tick_params(labelsize=10)
    else:
        handles = [Patch(facecolor=palette[i], edgecolor="none", label=bin_names[i]) for i in range(2)]
        ax.legend(handles=handles, title=label_name, fontsize=9, title_fontsize=9, frameon=True, loc="upper right")
    
    if binned is None:
        ax.set_title(f"Geometry of {model_name.upper()}", pad=15)
    else:
        t_str = f"{bin_threshold:g}"
        ax.set_title(f"Geometry of {model_name.upper()} (bins @ {t_str})", pad=15)
    
    # Remove spines for dimensionality reduction plots (typically no border lines needed)
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.close()


def plot_target_aware_latent_heatmaps(z_data, geo_matrix, labels, label_name, save_dir, model_name, n_neighbors_umap=15, n_neighbors_trust=12, bin_threshold=None):
    """
    Target-Aware Latent Heatmaps: 2D UMAP colored by regression target (y).
    Left = Baseline (UMAP on Euclidean latent); Right = GW (UMAP on geodesic).
    Trustworthiness measures how well local structure is retained in the 2D embedding.
    """
    set_plot_style()

    if isinstance(z_data, torch.Tensor):
        z_np = z_data.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_data)
    labels = np.asarray(labels).flatten()
    n = z_np.shape[0]
    if n < 3:
        print("  Skipping target-aware heatmaps: need at least 3 samples.")
        return

    n_neighbors_umap = min(n_neighbors_umap, n - 1)
    n_neighbors_trust = min(n_neighbors_trust, n - 1)
    if n_neighbors_umap < 2 or n_neighbors_trust < 2:
        print("  Skipping target-aware heatmaps: not enough neighbors.")
        return

    # UMAP on Euclidean latent (Baseline)
    reducer_eucl = umap.UMAP(metric='euclidean', random_state=42, n_neighbors=n_neighbors_umap, min_dist=0.1)
    emb_eucl = reducer_eucl.fit_transform(z_np)

    # UMAP on geodesic (GW / Riemannian)
    geo_clean = np.nan_to_num(geo_matrix, nan=0.0)
    reducer_geo = umap.UMAP(metric='precomputed', random_state=42, n_neighbors=n_neighbors_umap, min_dist=0.1)
    emb_geo = reducer_geo.fit_transform(geo_clean)

    # Trustworthiness: local structure retained in 2D
    tw_baseline = trustworthiness(z_np, emb_eucl, n_neighbors=n_neighbors_trust)
    tw_gw = trustworthiness(geo_clean, emb_geo, n_neighbors=n_neighbors_trust)

    binned, bin_names = _bin_labels(labels, bin_threshold)
    if binned is None:
        vmin, vmax = np.nanpercentile(labels, [5, 95])
        norm = None if vmin >= vmax else Normalize(vmin=vmin, vmax=vmax)
    else:
        palette = ["#1f77b4", "#d62728"]
        cmap = ListedColormap(palette)

    fig, (ax_baseline, ax_gw) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, emb, title, tw in [
        (ax_baseline, emb_eucl, "Baseline (Euclidean latent)", tw_baseline),
        (ax_gw, emb_geo, "GW / Riemannian (Geodesic)", tw_gw),
    ]:
        if binned is None:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=14, alpha=0.75, edgecolors='none', norm=norm)
        else:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=binned, cmap=cmap, s=14, alpha=0.75, edgecolors='none', vmin=-0.5, vmax=1.5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{title}\nTrustworthiness = {tw:.4f}", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(left=True, bottom=True)

    if binned is None:
        fig.suptitle(f"Target-Aware Latent Heatmaps: {model_name.upper()} (color = {label_name})", fontsize=13, y=1.02)
        cb = fig.colorbar(sc, ax=ax_gw, pad=0.02, shrink=0.7, location='right')
        cb.set_label(f'Target: {label_name}', weight='bold')
    else:
        t_str = f"{bin_threshold:g}"
        fig.suptitle(f"Target-Aware Latent Heatmaps: {model_name.upper()} (bins @ {t_str})", fontsize=13, y=1.02)
        handles = [Patch(facecolor=palette[i], edgecolor="none", label=bin_names[i]) for i in range(2)]
        ax_gw.legend(handles=handles, title=label_name, fontsize=9, title_fontsize=9, frameon=True, loc="upper right")
    plt.tight_layout()
    if binned is None:
        _save_figure(fig, save_dir / f'{model_name}_target_aware_heatmaps.png')
    else:
        _save_figure(fig, save_dir / f'{model_name}_target_aware_heatmaps_binned.png')
    plt.close()

    print("  Saved Target-Aware Latent Heatmaps.")

def run_riemannian_analysis(experiment_dir: Path, model_name: str = None, 
                           max_samples: int = None, device: str = None, bin_threshold: float = None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 1. Locate config (accept a direct full_config.yaml path or an experiment directory)
    full_config, experiment_dir, config_path = load_experiment_config(experiment_dir)
    print(f"\n=== Starting Riemannian Analysis: {experiment_dir.name} ===")
    print(f"Device: {device}")
    full_config, train_args, data_config, paths_cfg, train_file, target_col, exclude_cols = _load_analysis_config(full_config)
    continuous_cols, binary_cols, categorical_cols = get_variable_types(data_config)

    # 2. Prepare data directly
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
        max_samples=max_samples
    )
    full_dataset, train_dataset, test_dataset, generator, test_features, test_labels, test_indices = prepared

    # Training split (used for some models/debug)
    train_indices = train_dataset.indices
    train_features = full_dataset.features[train_indices].numpy()
    train_labels = full_dataset.label[train_indices].numpy()

    config_dict = {
        'full_config': full_config,
        'train_args': train_args,
        'data_config': data_config,
        'target_col': target_col,
        'paths_cfg': paths_cfg,
        'training_cfg': train_args.get('training', {}),
    }
    
    # 2. Identify Models to Run
    checkpoints_dir = experiment_dir / 'checkpoints'
    
    if model_name is None:
        available_models = _list_embedding_models(checkpoints_dir)
        
        if not available_models:
            raise ValueError("No suitable embedding model checkpoint found.")
        
        print(f"\nFound {len(available_models)} models: {available_models}")
        for mname in available_models:
            try:
                print(f"\n{'='*60}\nProcessing: {mname}\n{'='*60}")
                run_single_model_analysis(
                    experiment_dir, mname, max_samples, device,
                    test_features, test_labels, full_dataset, config_dict,
                    train_features, train_labels, bin_threshold
                )
            except Exception as e:
                print(f"Error processing {mname}: {e}")
                import traceback
                traceback.print_exc()
        return
    
    # Single model processing
    print(f"\nProcessing Model: {model_name}")
    checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    run_single_model_analysis(
        experiment_dir, model_name, max_samples, device,
        test_features, test_labels, full_dataset, config_dict,
        train_features, train_labels, bin_threshold
    )

def run_single_model_analysis(experiment_dir, model_name, max_samples, device,
                              test_features, test_labels, full_dataset, config_dict,
                              train_features=None, train_labels=None, bin_threshold=None):
    
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
    
    input_dim = test_features.shape[1]
    
    try:
        model = model_factory.load_model_from_checkpoint(
            checkpoint_path,
            model_name,
            input_dim,
            config_dict['full_config'],
            device,
            full_dataset=full_dataset,
            train_features=train_features,
            train_labels=train_labels,
            checkpoints_dir=checkpoints_dir,
            filter_mismatched=True,
        )
        print(f"  Loaded {type(model).__name__}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extraction
    print("  Extracting Regressor...")
    regressor_func = extract_regressor_func(model)
    if regressor_func is None:
        print(f"  Skipping {model_name}: no regression head found.")
        return
    
    # Extract Z
    x_tensor = torch.from_numpy(test_features).float().to(device)
    z_tensor = extract_latent_z(model, x_tensor)
    
    # Compute Matrix
    try:
        geo_matrix, _ = compute_geodesic_distance_matrix(z_tensor, regressor_func, device)
    except torch.cuda.OutOfMemoryError:
        print("  Error: OOM during geodesic calculation. Reduce batch size or max_samples.")
        return

    results_dir = experiment_dir / 'riemannian_analysis'
    results_dir.mkdir(exist_ok=True, parents=True)

    subset_size = geo_matrix.shape[0]
    subset_features = test_features[:subset_size]
    subset_z_tensor = z_tensor[:subset_size]

    print("  Generating distortion plot...")
    plot_distortion_validation(
        subset_features,
        subset_z_tensor,
        geo_matrix,
        regressor_func,
        results_dir,
        model_name,
        test_labels[:subset_size],
        bin_threshold=bin_threshold,
    )

    # Feature-Target Geometry Alignment: D_z vs D_y (Distance Correlation)
    print("  Computing Feature-Target Geometry Alignment...")
    subset_labels_flat = np.asarray(test_labels[:subset_size]).flatten()
    align_eucl = compute_feature_target_geometry_alignment(subset_z_tensor, subset_labels_flat, geo_matrix=None, use_geodesic=False)
    align_geo = compute_feature_target_geometry_alignment(subset_z_tensor, subset_labels_flat, geo_matrix=geo_matrix, use_geodesic=True)
    plot_feature_target_alignment(align_eucl, results_dir, model_name, use_geodesic=False)
    plot_feature_target_alignment(align_geo, results_dir, model_name, use_geodesic=True)
    print(f"  Feature-Target Alignment: dCor(Euclidean) = {align_eucl['distance_correlation']:.4f}, dCor(Geodesic) = {align_geo['distance_correlation']:.4f}")

    # Save Spearman values for latent-vs-target difference
    import csv as csv_module
    with open(results_dir / f'{model_name}_spearman_alignment.csv', 'w', newline='') as f:
        w = csv_module.writer(f)
        w.writerow(["space", "spearman_r", "spearman_p"])
        w.writerow(["euclidean_latent", round(align_eucl["spearman_r"], 6), round(align_eucl["spearman_p"], 6)])
        w.writerow(["geodesic", round(align_geo["spearman_r"], 6), round(align_geo["spearman_p"], 6)])
    
    del model, regressor_func, geo_matrix, z_tensor
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--bin_threshold', type=float, default=None,
                        help='If set, bin target into < threshold / >= threshold for plotting.')
    
    args = parser.parse_args()
    
    run_riemannian_analysis(Path(args.experiment_dir), args.model_name, args.max_samples, args.device, args.bin_threshold)
