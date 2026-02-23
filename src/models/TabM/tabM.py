import torch
import torch.nn as nn
from torch import Tensor

# --------------------------------------------------------------------------------------
# Minimal TabM-mini implementation (local, no pip). This follows the "mini ensemble"
# structure from the official TabM repo, but keeps a flat-input interface for this repo.
# --------------------------------------------------------------------------------------

def init_rsqrt_uniform_(tensor: Tensor, d: int) -> Tensor:
    d_rsqrt = d**-0.5
    return nn.init.uniform_(tensor, -d_rsqrt, d_rsqrt)

class ScaleEnsemble(nn.Module):
    """Simple per-head scaling used by the existing TabMEncoder."""

    def __init__(self, k: int, d: int):
        super().__init__()
        self.r = nn.Parameter(torch.empty(k, d))
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.r.unsqueeze(0)

    def reset_parameters(self) -> None:
        init_rsqrt_uniform_(self.r, self.r.shape[-1])


class ElementwiseAffine(nn.Module):
    def __init__(self, shape: tuple[int, ...], *, bias: bool, scaling_init: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None
        self._scaling_init = scaling_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_rsqrt_uniform_(self.weight, self.weight.shape[-1])
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, self.bias.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight if self.bias is None else torch.addcmul(self.bias, self.weight, x)


class EnsembleView(nn.Module):
    """Expand a 2D input (B, D) into (B, K, D) for the ensemble."""

    def __init__(self, *, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1).expand(-1, self.k, -1)
        if x.ndim == 3 and x.shape[1] == self.k:
            return x
        raise ValueError(f'Expected input of shape (B, D) or (B, {self.k}, D), got {x.shape}')


class LinearEnsemble(nn.Module):
    """K independent linear heads applied in parallel to a (B, K, D) tensor."""

    def __init__(self, in_features: int, out_features: int, *, k: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, D)
        x = x.transpose(0, 1)  # (K, B, D)
        x = x @ self.weight  # (K, B, out)
        x = x.transpose(0, 1)  # (B, K, out)
        if self.bias is not None:
            x = x + self.bias
        return x


class MLPBackboneMiniEnsemble(nn.Module):
    """TabM-mini backbone: shared MLP + one per-head affine at input."""

    def __init__(
        self,
        *,
        d_in: int,
        hidden_dims: list[int],
        dropout: float,
        batchnorm: bool = False,
        activation: str = 'ReLU',
        k: int,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError('hidden_dims must be non-empty for TabM-mini')
        Activation = getattr(nn, activation)
        self.k = k
        self.batchnorm = batchnorm
        self.affine = ElementwiseAffine((k, d_in), bias=False, scaling_init='random-signs')
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        [nn.Linear(d_in if i == 0 else hidden_dims[i - 1], h)]
                        + ([nn.BatchNorm1d(h)] if batchnorm else [])
                        + [Activation(), nn.Dropout(dropout)]
                    )
                )
                for i, h in enumerate(hidden_dims)
            ]
        )

    @property
    def out_dim(self) -> int:
        return self.blocks[-1][0].out_features

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, D)
        x = self.affine(x)
        for block in self.blocks:
            d = x.shape[-1]
            x = x.transpose(0, 1).reshape(-1, d)
            x = block(x)
            x = x.view(self.k, -1, x.shape[-1]).transpose(0, 1)
        return x


class TabM(nn.Module):
    """
    Local TabM-mini: minimal ensemble adapter + shared MLP backbone + K heads.
    Input is a flat (B, D) feature tensor (no internal num/cat handling).
    Output is (B, K, out_dim) to match existing training code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        k_heads: int = 5,
        dropout: float = 0.1,
        batchnorm: bool = False,
        activation: str = 'ReLU',
        task: str = 'regression',
    ):
        super().__init__()
        self.k = k_heads
        self.task = task
        self.ensemble_view = EnsembleView(k=self.k)
        self.backbone = MLPBackboneMiniEnsemble(
            d_in=in_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batchnorm=batchnorm,
            activation=activation,
            k=self.k,
        )
        self.output = LinearEnsemble(self.backbone.out_dim, out_dim, k=self.k)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensemble_view(x)
        x = self.backbone(x)
        x = self.output(x)
        if self.task == 'classification':
            x = torch.sigmoid(x)
        return x

        
