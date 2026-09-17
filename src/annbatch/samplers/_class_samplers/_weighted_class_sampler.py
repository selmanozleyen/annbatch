"""WeightedClassSampler -- class-weighted (but not class-coherent) batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.utils import split_given_size

from ._class_sampler import ClassSampler

if TYPE_CHECKING:
    import numpy as np


class WeightedClassSampler(ClassSampler):
    """Sample batches whose *class composition* follows ``class_weights``.

    Chunks are read exactly as :class:`~annbatch.samplers.ClassSampler` reads them --
    one class per chunk -- but the rows of a whole preload window are shuffled together
    before being split into batches, so each batch mixes classes in expectation
    proportionally to their weights instead of being drawn from a single class.
    """

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # No grouping: batches need not be class-coherent, so every chunk draws its own class.
        return self._class_rng.choice(self._rle_manager.n_classes, size=n_slices, p=self._rle_manager.weights)

    def _splits_for_window(self, ids: np.ndarray) -> list[np.ndarray]:
        # Shuffle across the whole window, not within each batch, so batches mix classes.
        self._split_rng.shuffle(ids)
        return split_given_size(ids, self._batch_size)
