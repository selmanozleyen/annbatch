"""BoundClassSampler -- class schedule bound to an inner class sampler."""

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
    """Bind a class sampler's per-batch class schedule onto another obs table.

    For every batch the inner sampler yields, this sampler yields one batch of its own
    ``batch_size`` observations of the matching class, drawn from ``classes_to_bind_on``
    (this sampler's own obs) and read as contiguous chunks. Classes are matched by *label*.

    When the inner balances over several columns (its categories are tuples) you can bind on
    a subset with ``on`` -- a ``dict`` mapping inner tuple positions to ``classes_to_bind_on``
    tuple positions (``{0: 0, 1: 1}``); ``None`` matches the whole label. ``classes_to_bind_on``
    (after ``on`` projection) must be a subset of the inner sampler's
    :attr:`~annbatch.abc.BaseClassSampler.vocab`.

    With no ``classes``, rows of the matched class are drawn uniformly. Passing a secondary
    ``classes`` (a single column, row-aligned with ``classes_to_bind_on``) plus ``class_weights``
    adds conditional sampling: which rows of the matched class are drawn is weighted by the
    secondary class (a non-positive weight excludes it), with ``class_weights`` one flat array
    in ``classes.categories`` order.

    ``batch_size`` must be a multiple of ``chunk_size``; the total is
    ``inner_sampler.n_batches() * batch_size`` (one full batch per inner batch). Every drawable
    run must be at least ``chunk_size`` long (stricter with a secondary ``classes``, since runs
    are then cut on the joint key). Multiple workers are not supported.

    Parameters
    ----------
    inner_sampler
        Any :class:`~annbatch.abc.BaseClassSampler` (typically a :class:`ClassSampler`) whose
        per-batch class schedule and batch count drive sampling.
    chunk_size, preload_nchunks, batch_size
        Sizing for this sampler; ``batch_size`` must be a multiple of ``chunk_size``.
    classes_to_bind_on
        A :class:`pandas.Categorical`, one entry per observation; matched (via ``on``) to the
        class the inner picks. Its (projected) categories must be a subset of the inner's
        :attr:`~annbatch.abc.BaseClassSampler.vocab`.
    on
        Optional ``dict`` mapping inner tuple positions to ``classes_to_bind_on`` positions.
        ``None`` matches the whole label.
    classes
        Optional secondary single-column :class:`pandas.Categorical`, the same length as
        ``classes_to_bind_on``, for conditional (within-class) sampling.
    class_weights
        Optional weights, one per ``classes.categories``; a non-positive weight excludes that class.
    mask, rng
        Optional observation range and random number generator (independent of the inner's).
    """

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
        bind_codes = codes_of_categorical(classes_to_bind_on, "classes_to_bind_on")

        if on is None:
            inner_pos = bind_pos = None
        elif isinstance(on, dict):
            inner_pos, bind_pos = tuple(on.keys()), tuple(on.values())
        else:
            raise TypeError("on must be a dict[int, int] or None.")

        # match key: project both sides onto the bound positions and match by label. Factorize the
        # (few) bound categories and map to obs by codes -- never the per-obs expansion -- and match
        # via MultiIndex.get_indexer, so construction stays vectorized even at ~100k categories.
        inner_proj = project_index(inner_sampler.vocab, inner_pos)
        cat_to_match, match_uniques = pd.factorize(project_index(classes_to_bind_on.categories, bind_pos))
        match_obs_codes = cat_to_match[bind_codes]  # per-obs match code
        inner_to_match = match_uniques.get_indexer(inner_proj)  # per inner category -> match code

        # Only match classes that actually occur in the obs matter for the subset/drawable rules:
        # factorizing over `.categories` also surfaces declared-but-unused categories (pandas keeps
        # these after subsetting), which must not trigger the checks.
        present = np.zeros(len(match_uniques), dtype=bool)
        present[match_obs_codes] = True
        present_codes = np.flatnonzero(present)

        # classes_to_bind_on (its present classes) must be a subset of the inner sampler's classes
        present_uniques = match_uniques[present_codes]
        not_in_inner = inner_proj.unique().get_indexer(present_uniques) < 0
        if not_in_inner.any():
            raise ValueError(
                f"classes_to_bind_on has classes {list(present_uniques[not_in_inner])} not present in the inner "
                "sampler's classes; classes_to_bind_on must be a subset of the inner sampler's classes."
            )
        # and every class the inner can emit must be present here, so it is drawable
        emittable_inner = inner_sampler.emittable_codes()
        # the match classes the inner will ever ask this bound to produce
        emittable_match = inner_to_match[emittable_inner]
        drawable = np.isin(emittable_match, present_codes)  # np.isin treats the -1 "absent" code as not present
        if not drawable.all():
            missing = inner_proj[emittable_inner][~drawable].unique()
            raise ValueError(f"The inner sampler can emit classes {list(missing)} absent from classes_to_bind_on.")

        codes, weights, labels = self._build_joint(
            match_obs_codes, match_uniques, np.unique(emittable_match), classes, class_weights
        )

        self._inner_sampler = inner_sampler
        self._classes_to_bind_on = classes_to_bind_on
        self._on = on
        self._inner_to_match = inner_to_match  # per inner category -> match code
        self._match_uniques = match_uniques  # projected match keys (for error messages)

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
        match_obs_codes: np.ndarray,
        match_uniques: pd.Index,
        emittable_match: np.ndarray,
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        # case 1: no secondary sampling
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            # identity mapping and uniform weighting
            self._match_of_joint = np.arange(len(match_uniques))
            self._weight_of_joint = np.ones(len(match_uniques), dtype=float)
            drawable = np.isin(self._match_of_joint, emittable_match).astype(float)
            return match_obs_codes, drawable, match_uniques

        sec_codes = codes_of_categorical(classes, "classes")
        if len(classes) != match_obs_codes.shape[0]:
            raise ValueError(
                f"classes must be the same length as classes_to_bind_on ({match_obs_codes.shape[0]}), got {len(classes)}."
            )
        n_sec = len(classes.categories)
        sec_weights = resolve_class_weights(class_weights, n_sec)

        # runs are cut on the joint (match, secondary) key so each slice is coherent in both
        joint_codes, joint_raw = pd.factorize(match_obs_codes.astype(np.int64) * n_sec + sec_codes)
        j_match = (joint_raw // n_sec).astype(np.int64)
        j_sec = (joint_raw % n_sec).astype(np.int64)
        self._match_of_joint = j_match
        self._weight_of_joint = sec_weights[j_sec]
        drawable = np.where(np.isin(j_match, emittable_match), sec_weights[j_sec], 0.0)

        # Flatten (match, secondary) into one flat label per joint class -- when the match is itself a
        # tuple (a multi-column inner), keeping it nested would hide those columns from a downstream
        # `on`, so a chain could not condition on them. Flat labels compose: (cellline, drug) + batch
        # -> (cellline, drug, batch), which the next level can project any position of. Built column-wise
        # (take + from_arrays) so it stays vectorized at ~200k joint classes -- no per-label loop.
        labels = pd.MultiIndex.from_arrays(
            to_level_arrays(match_uniques.take(j_match)) + to_level_arrays(classes.categories.take(j_sec))
        )
        return joint_codes, drawable, labels

    @property
    def classes_to_bind_on(self) -> pd.Categorical:
        return self._classes_to_bind_on

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # the joint classes present here, with the match class and secondary weight of each
        present_codes = self._per_class_sampling_info.index.to_numpy()
        present_match = self._match_of_joint[present_codes]
        present_weight = self._weight_of_joint[present_codes]

        # a mask can leave a class the inner still emits with no drawable run in the current range
        match_of_batch = self._inner_to_match[self._inner_sampler.batch_codes()]
        drawable = np.zeros(len(self._match_uniques), dtype=bool)
        drawable[present_match] = True
        undrawable = match_of_batch[~drawable[match_of_batch]]
        if undrawable.size:
            raise ValueError(
                f"Class {self._match_uniques[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )

        # pick a joint of each batch's match class, weighted by the secondary -- one vectorized grouped draw
        positions = grouped_weighted_choice(present_match, present_weight, match_of_batch, self._class_rng)
        group_chunks = self._batch_size // self._chunk_size
        return np.repeat(positions, group_chunks)
