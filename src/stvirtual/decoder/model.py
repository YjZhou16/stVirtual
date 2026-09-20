from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DecoderOutput:
    """Parameters of the raw-count negative-binomial reconstruction."""

    expected_counts: torch.Tensor
    gene_proportions: torch.Tensor
    library_size: torch.Tensor
    log1p_library_size: torch.Tensor
    dispersion: torch.Tensor


@dataclass
class ReconstructionLoss:
    total: torch.Tensor
    nb_nll: torch.Tensor
    library: torch.Tensor
    library_target: torch.Tensor


def denormalize_latent(
    latent: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return Stage-1 normalized rollout latents to the source latent scale."""

    values = np.asarray(latent, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    scale = np.asarray(std, dtype=np.float32).reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != center.shape[1]:
        raise ValueError("rollout latent dimensions do not match normalization")
    if center.shape != scale.shape or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("latent normalization mean/std are invalid")
    return (values * scale + center).astype(np.float32, copy=False)


class ResidualBlock(nn.Module):
    """Pre-normalized 512 -> 1024 -> 512 residual MLP at default width."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, width * 2)
        self.linear_out = nn.Linear(width * 2, width)
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.linear_out(F.silu(self.linear_in(self.norm(features))))
        return features + F.dropout(residual, p=self.dropout, training=self.training)


class CountAwareDecoder(nn.Module):
    """Decode latent state into NB expected counts on the raw-count scale."""

    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        *,
        hidden_dim: int = 512,
        factor_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        skip_weight: float = 0.2,
        dispersion_eps: float = 1e-4,
    ):
        super().__init__()
        dimensions = (latent_dim, n_genes, hidden_dim, factor_dim, n_blocks)
        if min(dimensions) <= 0:
            raise ValueError("decoder dimensions and n_blocks must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if skip_weight < 0:
            raise ValueError("skip_weight must be non-negative")
        if dispersion_eps <= 0:
            raise ValueError("dispersion_eps must be positive")

        self.latent_dim = int(latent_dim)
        self.n_genes = int(n_genes)
        self.hidden_dim = int(hidden_dim)
        self.factor_dim = int(factor_dim)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.skip_weight = float(skip_weight)
        self.dispersion_eps = float(dispersion_eps)
        self.input_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(self.hidden_dim, self.dropout) for _ in range(self.n_blocks)
        )
        self.profile_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.factor_dim),
        )
        self.skip_gene_head = nn.Linear(self.latent_dim, self.n_genes)
        self.library_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.gene_factors = nn.Parameter(torch.empty(self.n_genes, self.factor_dim))
        self.gene_bias = nn.Parameter(torch.zeros(self.n_genes))
        self.raw_theta = nn.Parameter(torch.full((self.n_genes,), 2.0))
        nn.init.normal_(self.gene_factors, std=1.0 / math.sqrt(self.factor_dim))
        nn.init.zeros_(self.skip_gene_head.weight)
        nn.init.zeros_(self.skip_gene_head.bias)
        nn.init.zeros_(self.library_head[-1].weight)
        nn.init.zeros_(self.library_head[-1].bias)

        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))
        self.register_buffer("log_library_mean", torch.tensor(8.0))
        self.register_buffer("log_library_std", torch.tensor(1.0))

    def set_input_statistics(
        self,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        log_library_mean: torch.Tensor,
        log_library_std: torch.Tensor,
    ) -> None:
        latent_mean = torch.as_tensor(latent_mean).reshape(-1)
        latent_std = torch.as_tensor(latent_std).reshape(-1)
        if latent_mean.numel() != self.latent_dim or latent_std.numel() != self.latent_dim:
            raise ValueError("latent statistics have the wrong dimension")
        if torch.any(~torch.isfinite(latent_mean)):
            raise ValueError("latent mean must be finite")
        if torch.any(~torch.isfinite(latent_std)) or torch.any(latent_std <= 0):
            raise ValueError("latent std must be finite and positive")
        library_mean = torch.as_tensor(log_library_mean).reshape(())
        library_std = torch.as_tensor(log_library_std).reshape(())
        if not torch.isfinite(library_mean):
            raise ValueError("log-library mean must be finite")
        if not torch.isfinite(library_std) or library_std <= 0:
            raise ValueError("log-library std must be finite and positive")
        with torch.no_grad():
            self.latent_mean.copy_(latent_mean.to(self.latent_mean))
            self.latent_std.copy_(latent_std.to(self.latent_std))
            self.log_library_mean.copy_(library_mean.to(self.log_library_mean))
            self.log_library_std.copy_(library_std.to(self.log_library_std))

    def initialize_gene_bias(self, gene_count_sums: torch.Tensor) -> None:
        sums = torch.as_tensor(gene_count_sums, dtype=self.gene_bias.dtype).reshape(-1)
        if sums.numel() != self.n_genes or torch.any(sums < 0) or sums.sum() <= 0:
            raise ValueError("gene count sums are invalid")
        proportions = (sums + 1.0) / (sums.sum() + self.n_genes)
        with torch.no_grad():
            self.gene_bias.copy_(proportions.log().to(self.gene_bias.device))

    def forward(self, latent: torch.Tensor) -> DecoderOutput:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent batch has the wrong shape")

        normalized_latent = (latent - self.latent_mean) / self.latent_std
        hidden = F.silu(self.input_projection(normalized_latent))
        for block in self.blocks:
            hidden = block(hidden)

        factor_state = self.profile_head(hidden) / math.sqrt(self.factor_dim)
        main_gene_logits = F.linear(factor_state, self.gene_factors, self.gene_bias)
        skip_gene_logits = self.skip_gene_head(normalized_latent)
        gene_logits = main_gene_logits + self.skip_weight * skip_gene_logits
        gene_proportions = torch.softmax(gene_logits, dim=1)

        standardized_log_library = self.library_head(hidden).squeeze(1)
        log1p_library_size = (
            self.log_library_mean + self.log_library_std * standardized_log_library
        ).clamp(min=0.0, max=20.0)
        library_size = torch.expm1(log1p_library_size).clamp_min(1e-4)
        expected_counts = gene_proportions * library_size[:, None]
        dispersion = F.softplus(self.raw_theta) + self.dispersion_eps
        return DecoderOutput(
            expected_counts=expected_counts,
            gene_proportions=gene_proportions,
            library_size=library_size,
            log1p_library_size=log1p_library_size,
            dispersion=dispersion,
        )

    def architecture_config(self) -> dict[str, int | float]:
        return {
            "latent_dim": self.latent_dim,
            "n_genes": self.n_genes,
            "hidden_dim": self.hidden_dim,
            "factor_dim": self.factor_dim,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
            "skip_weight": self.skip_weight,
            "dispersion_eps": self.dispersion_eps,
        }


def validate_raw_count_tensor(counts: torch.Tensor) -> None:
    if counts.ndim != 2 or counts.numel() == 0:
        raise ValueError("raw counts must be a non-empty cells-by-genes matrix")
    if torch.any(~torch.isfinite(counts)) or torch.any(counts < 0):
        raise ValueError("raw counts must be finite and non-negative")
    if torch.any(counts != torch.round(counts)):
        raise ValueError("NB target must contain integer raw counts, not log1p counts")


def negative_binomial_nll(
    raw_counts: torch.Tensor,
    expected_counts: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    """Mean NB NLL on untransformed raw counts and raw-count-scale means."""

    counts = raw_counts.float()
    mean = expected_counts.float()
    theta = dispersion.float().reshape(1, -1)
    validate_raw_count_tensor(counts)
    if mean.shape != counts.shape or theta.shape[1] != counts.shape[1]:
        raise ValueError("NB counts, mean, and dispersion shapes do not match")
    if torch.any(~torch.isfinite(mean)) or torch.any(mean < 0):
        raise ValueError("NB expected counts must be finite and non-negative")
    if torch.any(~torch.isfinite(theta)) or torch.any(theta <= 0):
        raise ValueError("NB dispersion must be finite and positive")

    mean = mean.clamp_min(1e-8)
    log_theta_plus_mean = torch.log(theta + mean)
    log_probability = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta) - log_theta_plus_mean)
        + counts * (torch.log(mean) - log_theta_plus_mean)
    )
    return -log_probability.mean()


def reconstruction_loss(
    output: DecoderOutput,
    raw_counts: torch.Tensor,
    *,
    library_weight: float = 0.5,
) -> ReconstructionLoss:
    """NB reconstruction plus direct supervision of log1p raw library size."""

    if library_weight < 0:
        raise ValueError("library_weight must be non-negative")
    counts = raw_counts.float()
    nb_nll = negative_binomial_nll(counts, output.expected_counts, output.dispersion)
    library_target = torch.log1p(counts.sum(dim=1))
    library = F.smooth_l1_loss(output.log1p_library_size.float(), library_target)
    total = nb_nll + float(library_weight) * library
    return ReconstructionLoss(total, nb_nll, library, library_target)
