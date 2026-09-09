from __future__ import annotations

import importlib.util
from typing import NamedTuple

import numpy as np
import pandas as pd

from annbatch.utils import check_lt_1


class WorkerInfo(NamedTuple):
    """Minimal worker info for RNG handling."""

    id: int
    num_workers: int


def get_torch_worker_info() -> WorkerInfo | None:
    """Get torch DataLoader worker info if available.

    Returns None if torch is not installed or not in a worker process.
    """
    if importlib.util.find_spec("torch"):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None:
            return WorkerInfo(id=info.id, num_workers=info.num_workers)
    return None


def validate_chunk_batch_preload_sizes(
    chunk_size: int,
    preload_nchunks: int,
    batch_size: int,
) -> None:
    check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preloaded chunks"])
    preload_size = chunk_size * preload_nchunks

    if batch_size > preload_size:
        raise ValueError(
            "batch_size cannot exceed chunk_size * preload_nchunks. "
            f"Got batch_size={batch_size}, but max is {preload_size}."
        )
    if preload_size % batch_size != 0:
        raise ValueError(
            "chunk_size * preload_nchunks must be divisible by batch_size. "
            f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
        )


def validate_mask_and_resolve(mask: slice) -> tuple[int, int]:
    """Validate a sampler mask against sanity checks then resolve the start and stop."""
    if mask.step is not None and mask.step != 1:
        raise ValueError(f"mask.step must be 1, but got {mask.step}")
    start, stop = mask.start or 0, mask.stop
    if start < 0:
        raise ValueError("mask.start must be >= 0")
    if stop is not None and start >= stop:
        raise ValueError("mask.start must be < mask.stop when mask.stop is specified")
    return start, stop


def validate_mask_n_obs_and_resolve(mask: slice, n_obs: int) -> tuple[int, int]:
    """Validate a sampler mask against n_obs then resolve the start and stop."""
    start, stop = validate_mask_and_resolve(mask)
    if stop is None:
        stop = n_obs
    if stop > n_obs:
        raise ValueError(
            f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
            "The sampler range must be within the loader's observations."
        )
    if start >= stop:
        raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")
    return start, stop


def resolve_class_weights(class_weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    if class_weights is None:
        weights = np.ones(n_classes, dtype=float)
    else:
        weights = np.array(class_weights, dtype=float)
        if weights.shape != (n_classes,):
            raise ValueError(
                f"class_weights must have one weight per class (expected shape ({n_classes},), got {weights.shape})."
            )
    if not (weights > 0).any():
        raise ValueError("class_weights must have at least one positive weight.")
    return weights


def _as_multiindex(index: pd.Index) -> pd.MultiIndex | None:
    if isinstance(index, pd.MultiIndex):
        return index
    if len(index) > 0 and isinstance(index[0], tuple):
        return pd.MultiIndex.from_tuples(index)
    return None


def to_level_arrays(index: pd.Index) -> list[pd.Index]:
    mi = _as_multiindex(index)
    if mi is None:
        return [index]
    return [mi.get_level_values(i) for i in range(mi.nlevels)]


def codes_of_categorical(categorical: pd.Categorical, name: str) -> np.ndarray:
    if not isinstance(categorical, pd.Categorical):
        raise TypeError(f"{name} must be a pandas.Categorical.")
    codes = categorical.codes
    if (codes == -1).any():
        raise ValueError(f"{name} contains NA values (codes == -1). Remove NAs before passing.")
    return codes


def grouped_weighted_choice(
    group_of_item: np.ndarray,
    weight_of_item: np.ndarray,
    group_of_draw: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.argsort(group_of_item, kind="stable")
    group, group_start = np.unique(group_of_item[order], return_index=True)
    group_end = np.append(group_start[1:], order.shape[0])
    cum_weight = np.cumsum(weight_of_item[order])
    hi = cum_weight[group_end - 1]
    lo = np.concatenate(([0.0], hi[:-1]))

    g = np.searchsorted(group, group_of_draw)
    target = lo[g] + rng.random(group_of_draw.shape[0]) * (hi[g] - lo[g])
    hit = np.clip(np.searchsorted(cum_weight, target, side="right"), group_start[g], group_end[g] - 1)
    return order[hit]


def project_index(labels: pd.Index, positions: tuple[int, ...] | None) -> pd.Index:
    if positions is None:
        return labels
    mi = _as_multiindex(labels)
    if mi is None:
        if positions != (0,):
            raise ValueError(
                f"Cannot project single-column labels onto positions {positions}; a single column has only position 0."
            )
        return labels
    if len(positions) == 1:
        return mi.get_level_values(positions[0])
    return pd.MultiIndex.from_arrays([mi.get_level_values(p) for p in positions])
