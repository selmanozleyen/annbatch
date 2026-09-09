from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._class_sampler import _RunClassSampler
from annbatch.samplers._utils import (
    codes_of_categorical,
    grouped_weighted_choice,
    project_index,
    resolve_class_weights,
    to_level_arrays,
)

if TYPE_CHECKING:
    from annbatch.abc import BaseClassSampler


class BoundClassSampler(_RunClassSampler):
    _inner_sampler: BaseClassSampler
    _classes_to_bind_on: pd.Categorical

    def __init__(
        self,
        inner_sampler: BaseClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        classes_to_bind_on: pd.Categorical,
        on: dict[int, int] | None = None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        if batch_size % chunk_size != 0:
            raise ValueError(
                "batch_size must be a multiple of chunk_size so each batch replays one inner class as whole chunks. "
                f"Got chunk_size={chunk_size}, batch_size={batch_size}."
            )
        outer_codes = codes_of_categorical(classes_to_bind_on, "classes_to_bind_on")

        if on is None:
            inner_pos = outer_pos = None
        elif isinstance(on, dict):
            inner_pos, outer_pos = tuple(on.keys()), tuple(on.values())
        else:
            raise TypeError("on must be a dict[int, int] or None.")

        inner_proj = project_index(inner_sampler.vocab, inner_pos)
        outer_to_shared, shared_classes = pd.factorize(project_index(classes_to_bind_on.categories, outer_pos))
        shared_obs_codes = outer_to_shared[outer_codes]
        inner_to_shared = shared_classes.get_indexer(inner_proj)

        present = np.zeros(len(shared_classes), dtype=bool)
        present[shared_obs_codes] = True
        present_codes = np.flatnonzero(present)

        present_classes = shared_classes[present_codes]
        not_in_inner = inner_proj.unique().get_indexer(present_classes) < 0
        if not_in_inner.any():
            raise ValueError(
                f"classes_to_bind_on has classes {list(present_classes[not_in_inner])} not present in the inner "
                "sampler's classes; classes_to_bind_on must be a subset of the inner sampler's classes."
            )
        emittable_inner = inner_sampler.emittable_codes()
        emittable_inner_shared = inner_to_shared[emittable_inner]
        drawable = np.isin(emittable_inner_shared, present_codes)
        if not drawable.all():
            missing = inner_proj[emittable_inner][~drawable].unique()
            raise ValueError(f"The inner sampler can emit classes {list(missing)} absent from classes_to_bind_on.")
        emittable_shared = np.unique(emittable_inner_shared)

        self._inner_sampler = inner_sampler
        self._classes_to_bind_on = classes_to_bind_on
        self._on = on
        self._inner_to_shared = inner_to_shared
        self._shared_classes = shared_classes

        codes, weights, labels = self._build_joint(
            shared_obs_codes, shared_classes, emittable_shared, classes, class_weights
        )
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=inner_sampler.n_batches(0) * batch_size,
            drop_last=False,
            mask=mask,
            rng=rng,
            codes=codes,
            weights=weights,
            category_labels=labels,
        )

    def _build_joint(
        self,
        shared_obs_codes: np.ndarray,
        shared_classes: pd.Index,
        emittable_shared: np.ndarray,
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            self._joint_to_shared = np.arange(len(shared_classes))
            self._joint_weight = np.ones(len(shared_classes), dtype=float)
            drawable = np.isin(self._joint_to_shared, emittable_shared).astype(float)
            return shared_obs_codes, drawable, shared_classes

        sec_codes = codes_of_categorical(classes, "classes")
        if len(classes) != shared_obs_codes.shape[0]:
            raise ValueError(
                f"classes must be the same length as classes_to_bind_on ({shared_obs_codes.shape[0]}), got {len(classes)}."
            )
        n_sec = len(classes.categories)
        sec_weights = resolve_class_weights(class_weights, n_sec)

        joint_codes, joint_raw = pd.factorize(shared_obs_codes.astype(np.int64) * n_sec + sec_codes)
        j_shared = joint_raw // n_sec
        j_sec = joint_raw % n_sec
        self._joint_to_shared = j_shared
        self._joint_weight = sec_weights[j_sec]
        drawable = np.where(np.isin(j_shared, emittable_shared), sec_weights[j_sec], 0.0)

        labels = pd.MultiIndex.from_arrays(
            to_level_arrays(shared_classes.take(j_shared)) + to_level_arrays(classes.categories.take(j_sec))
        )
        return joint_codes, drawable, labels

    @property
    def classes_to_bind_on(self) -> pd.Categorical:
        return self._classes_to_bind_on

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        present_codes = self._per_class_sampling_info.index.to_numpy()
        present_shared = self._joint_to_shared[present_codes]
        present_weight = self._joint_weight[present_codes]

        shared_of_batch = self._inner_to_shared[self._inner_sampler.batch_codes()]
        drawable = np.zeros(len(self._shared_classes), dtype=bool)
        drawable[present_shared] = True
        undrawable = shared_of_batch[~drawable[shared_of_batch]]
        if undrawable.size:
            raise ValueError(
                f"Class {self._shared_classes[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )

        positions = grouped_weighted_choice(present_shared, present_weight, shared_of_batch, self._class_rng)
        group_chunks = self._batch_size // self._chunk_size
        return np.repeat(positions, group_chunks)
