"""Utility functions for sampler operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from annbatch.utils import WorkerHandle

if TYPE_CHECKING:
    from collections.abc import Iterator


def indices_to_slices(indices: np.ndarray) -> list[slice]:
    """Convert sorted indices to slices for consecutive runs."""
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(indices)]])
    return [slice(int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, ends, strict=True)]


def shuffle_for_worker(arr: np.ndarray) -> np.ndarray:
    """Shuffle array and return this worker's partition."""
    worker = WorkerHandle()
    worker.shuffle(arr)
    return worker.get_part_for_worker(arr)


def generate_chunk_slices(n_obs: int, chunk_size: int, chunk_indices: np.ndarray) -> list[slice]:
    """Generate slices from chunk indices."""
    starts = chunk_indices * chunk_size
    stops = np.minimum(starts + chunk_size, n_obs)
    return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def batch_by_obs_count(slices: list[slice], batch_size: int) -> Iterator[list[slice]]:
    """Yield slices grouped into batches of approximately batch_size observations."""
    obs_count = 0
    batch: list[slice] = []

    for s in slices:
        batch.append(s)
        obs_count += s.stop - s.start
        if obs_count >= batch_size:
            yield batch
            batch, obs_count = [], 0

    if batch:
        yield batch


def is_sorted_categories(categories: np.ndarray) -> bool:
    """Check if categories form contiguous groups (data is sorted by category)."""
    if len(categories) == 0:
        return True
    # Categories are sorted if each unique value appears in one contiguous block
    seen = set()
    prev = categories[0]
    seen.add(prev)
    for cat in categories[1:]:
        if cat != prev:
            if cat in seen:
                return False
            seen.add(cat)
            prev = cat
    return True
