"""BoundClassSampler -- class schedule bound to an inner ClassSampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._class_sampler import ClassSampler, _RunClassSampler
from annbatch.samplers._utils import resolve_class_weights

if TYPE_CHECKING:
    from collections.abc import Sequence


def _as_object_1d(values: Sequence) -> np.ndarray:
    # pack into a 1-D object array (numpy would otherwise coerce a list of tuples to 2-D)
    arr = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _project_labels(categories: pd.Index, positions: tuple[int, ...] | None) -> np.ndarray:
    labels = list(categories)
    if positions is None:
        return _as_object_1d(labels)
    if len(positions) == 1:
        p = positions[0]
        return _as_object_1d([label[p] for label in labels])
    return _as_object_1d([tuple(label[i] for i in positions) for label in labels])


class BoundClassSampler(_RunClassSampler):
    """Bind a :class:`ClassSampler`'s per-batch class schedule onto another obs table.

    Match on tuple *positions* with ``on`` -- a ``dict`` mapping inner tuple positions to
    condition tuple positions (e.g. ``{0: 0}``); ``None`` matches the whole label. Build
    the tuple categories with e.g. ``pd.Categorical(list(zip(cell_type, donor)))`` so a
    position selects one component (``pd.factorize`` on tuples works too; a bare
    :class:`pandas.Categorical` cannot carry column names, only positions).
    """

    _inner_sampler: ClassSampler
    _condition_classes: pd.Categorical

    def __init__(
        self,
        inner_sampler: ClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        condition_classes: pd.Categorical,
        on: dict[int, int] | None = None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not isinstance(inner_sampler, ClassSampler):
            raise TypeError(f"inner_sampler must be a ClassSampler, got {type(inner_sampler).__name__}.")
        if not isinstance(condition_classes, pd.Categorical):
            raise TypeError(f"condition_classes must be a pandas.Categorical, got {type(condition_classes).__name__}.")
        if batch_size % chunk_size != 0:
            raise ValueError(
                "batch_size must be a multiple of chunk_size so each batch replays one inner class as whole chunks. "
                f"Got chunk_size={chunk_size}, batch_size={batch_size}."
            )

        cond_codes = np.asarray(condition_classes.codes)
        if (cond_codes == -1).any():
            raise ValueError("condition_classes contains NA values (codes == -1). Remove NAs before passing.")

        if on is None:
            inner_pos = cond_pos = None
        elif isinstance(on, dict):
            inner_pos, cond_pos = tuple(on.keys()), tuple(on.values())
        else:
            raise TypeError(f"on must be a dict[int, int] or None, got {type(on).__name__}.")

        # project both sides onto the bind positions; the condition's projected key drives run-building
        inner_proj = _project_labels(inner_sampler.classes.categories, inner_pos)
        cond_proj_obs = _project_labels(condition_classes.categories, cond_pos)[cond_codes]
        cond_obs_codes, cond_uniques = pd.factorize(cond_proj_obs)

        # every class the inner sampler can emit must have a matching projected key here
        inner_to_cond = pd.Index(cond_uniques, tupleize_cols=False).get_indexer(inner_proj)
        emittable_inner = np.asarray(inner_sampler._per_class_sampling_info.index)
        emittable_cond = inner_to_cond[emittable_inner]
        if (emittable_cond < 0).any():
            missing = pd.unique(inner_proj[emittable_inner][emittable_cond < 0])
            raise ValueError(
                f"The inner sampler can emit classes {list(missing)} absent from condition_classes. "
                "condition_classes must contain every class the inner sampler can emit (after `on` projection)."
            )
        emittable_cond_set = {int(c) for c in np.unique(emittable_cond)}

        codes, weights, labels = self._build_joint(
            cond_obs_codes, cond_uniques, emittable_cond_set, classes, class_weights
        )

        self._inner_sampler = inner_sampler
        self._condition_classes = condition_classes
        self._on = on
        self._inner_to_cond = inner_to_cond
        self._cond_uniques = cond_uniques

        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=inner_sampler.n_batches(0) * batch_size,
            drop_last=False,  # num_samples is a multiple of batch_size, so every batch is full
            mask=mask,
            rng=rng,
            codes=codes,
            weights=weights,
            category_labels=labels,
        )

    def _build_joint(
        self,
        cond_obs_codes: np.ndarray,
        cond_uniques: np.ndarray,
        emittable_cond_set: set[int],
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        # caches _joint_to_cond (joint class -> condition class) and _joint_to_weight
        # (joint class -> secondary weight) for _draw_class_of_slice
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            self._joint_to_cond = np.arange(len(cond_uniques), dtype=np.int64)
            self._joint_to_weight = np.ones(len(cond_uniques), dtype=float)
            weights = np.array([1.0 if int(c) in emittable_cond_set else 0.0 for c in self._joint_to_cond])
            return cond_obs_codes, weights, pd.Index(cond_uniques, tupleize_cols=False)

        if not isinstance(classes, pd.Categorical):
            raise TypeError(f"classes must be a pandas.Categorical, got {type(classes).__name__}.")
        if len(classes) != cond_obs_codes.shape[0]:
            raise ValueError(
                f"classes must be the same length as condition_classes ({cond_obs_codes.shape[0]}), got {len(classes)}."
            )
        sec_codes = np.asarray(classes.codes)
        if (sec_codes == -1).any():
            raise ValueError("classes contains NA values (codes == -1). Remove NAs before passing.")
        n_sec = len(classes.categories)
        sec_weights = resolve_class_weights(class_weights, n_sec)

        # runs are cut on the joint (condition, secondary) key so each slice is coherent in both
        joint_obs_codes, joint_raw = pd.factorize(cond_obs_codes.astype(np.int64) * n_sec + sec_codes)
        j_cond = (joint_raw // n_sec).astype(np.int64)
        j_sec = (joint_raw % n_sec).astype(np.int64)
        self._joint_to_cond = j_cond
        self._joint_to_weight = sec_weights[j_sec]
        weights = np.array(
            [sec_weights[s] if int(c) in emittable_cond_set else 0.0 for c, s in zip(j_cond, j_sec, strict=True)]
        )
        labels = pd.Index([(cond_uniques[c], classes.categories[s]) for c, s in zip(j_cond, j_sec, strict=True)])
        return joint_obs_codes, weights, labels

    @property
    def classes(self) -> pd.Categorical:
        return self._condition_classes

    def _replay_inner_batch_classes(self) -> np.ndarray:
        inner = self._inner_sampler
        codes = np.asarray(inner.classes.codes)
        batch_size, chunk_size = inner.batch_size, None
        batch_classes: list[int] = []
        for load_request in inner.sample(codes.shape[0]):
            requests = load_request["requests"]  # chunk_size slices, in order
            if chunk_size is None:  # first slice is a full chunk (the short one, if any, is always last)
                chunk_size = requests[0].stop - requests[0].start
            # batch i occupies window rows [i*batch_size, (i+1)*batch_size) and is class-coherent,
            # so its class is that of the slice its first row falls in
            for i in range(len(load_request["splits"])):
                batch_classes.append(int(codes[requests[(i * batch_size) // chunk_size].start]))
        return np.asarray(batch_classes, dtype=np.int64)

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # group the joint classes present in the current range by their condition class
        info = self._per_class_sampling_info
        present_codes = info.index.to_numpy()
        present_positions = np.arange(len(info))
        present_cond = self._joint_to_cond[present_codes]
        present_weight = self._joint_to_weight[present_codes]
        by_cond = {
            int(c): (present_positions[present_cond == c], present_weight[present_cond == c])
            for c in np.unique(present_cond)
        }

        # replay the inner schedule -> a condition class per batch -> a (weighted) joint position per batch
        cond_of_batch = self._inner_to_cond[self._replay_inner_batch_classes()]
        positions = np.empty(cond_of_batch.shape[0], dtype=np.int64)
        for c in np.unique(cond_of_batch):
            batch_idx = np.flatnonzero(cond_of_batch == c)
            if int(c) not in by_cond:
                raise ValueError(
                    f"Class {self._cond_uniques[c]!r} emitted by the inner sampler has no drawable run "
                    "of at least chunk_size in the current range."
                )
            rows, weight = by_cond[int(c)]
            positions[batch_idx] = self._rng.choice(rows, size=batch_idx.shape[0], p=weight / weight.sum())

        group_chunks = self._batch_size // self._chunk_size
        return np.repeat(positions, group_chunks)
