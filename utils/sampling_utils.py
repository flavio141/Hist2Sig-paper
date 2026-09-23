from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


SamplingStrategy = Literal[
    "none",
    "random",
    "energy_topk",
    "energy_gumbel",
]

EnergyMode = Literal[
    "l2",
    "l2_sq",
    "abs_mean",
]


@dataclass(frozen=True)
class BagSamplingConfig:
    bagsize: int
    strategy: SamplingStrategy = "random"
    energy_mode: EnergyMode = "l2_sq"
    energy_tau: float = 1.0
    seed: Optional[int] = None


def _generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def compute_energy(x: torch.Tensor, mode: EnergyMode = "l2_sq") -> torch.Tensor:
    """Compute a per-instance energy score.

    x: [L, D]
    returns: [L]
    """
    if mode == "l2":
        return torch.linalg.norm(x.float(), dim=1)
    if mode == "l2_sq":
        xf = x.float()
        return (xf * xf).sum(dim=1)
    if mode == "abs_mean":
        return x.float().abs().mean(dim=1)
    raise ValueError(f"Unknown energy mode: {mode}")


def sample_indices_random(num_instances: int, k: int, *, seed: Optional[int] = None, device: torch.device) -> torch.Tensor:
    if k <= 0 or num_instances <= k:
        return torch.arange(num_instances, device=device, dtype=torch.long)
    g = _generator(seed, device)
    return torch.randperm(num_instances, generator=g, device=device)[:k]


def sample_indices_energy_topk(energy: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic: take top-k by energy."""
    if k <= 0 or energy.numel() <= k:
        return torch.arange(energy.numel(), device=energy.device, dtype=torch.long)
    return torch.topk(energy, k=k, largest=True, sorted=False).indices


def sample_indices_energy_gumbel(energy: torch.Tensor, k: int, *, tau: float = 1.0, seed: Optional[int] = None) -> torch.Tensor:
    """Stochastic energy-based sampling without replacement using Gumbel-topk.

    We sample k indices by taking top-k of:
        logits = energy / tau + gumbel_noise
    This is fast and avoids explicit loops.
    """
    if k <= 0 or energy.numel() <= k:
        return torch.arange(energy.numel(), device=energy.device, dtype=torch.long)

    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    g = _generator(seed, energy.device)
    # Gumbel noise: -log(-log(U))
    u = torch.rand(energy.numel(), device=energy.device, generator=g).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(u))

    logits = energy.float() / tau + gumbel
    return torch.topk(logits, k=k, largest=True, sorted=False).indices


def sample_bag(
    x: torch.Tensor,
    *,
    config: BagSamplingConfig,
    per_item_seed: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Sample instances from a bag.

    Returns:
        x_sampled: [S, D]
        meta: dict with indices/energies/full_len
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x [L,D], got shape {tuple(x.shape)}")

    L = int(x.shape[0])
    k = int(config.bagsize)

    if config.strategy == "none" or k <= 0 or L <= k:
        idx = torch.arange(L, device=x.device, dtype=torch.long)
        energy = None
    elif config.strategy == "random":
        idx = sample_indices_random(L, k, seed=per_item_seed if per_item_seed is not None else config.seed, device=x.device)
        energy = None
    else:
        energy = compute_energy(x, mode=config.energy_mode)
        if config.strategy == "energy_topk":
            idx = sample_indices_energy_topk(energy, k)
        elif config.strategy == "energy_gumbel":
            idx = sample_indices_energy_gumbel(
                energy,
                k,
                tau=config.energy_tau,
                seed=per_item_seed if per_item_seed is not None else config.seed,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {config.strategy}")

    x_s = x.index_select(0, idx)
    meta = {
        "full_len": L,
        "sample_len": int(x_s.shape[0]),
        "indices": idx.detach().cpu(),
    }
    if energy is not None:
        meta["energy"] = energy.index_select(0, idx).detach().cpu()
    return x_s, meta
