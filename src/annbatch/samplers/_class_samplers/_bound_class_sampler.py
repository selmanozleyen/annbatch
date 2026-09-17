from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._utils import (
    codes_of_categorical,
    grouped_weighted_choice,
    project_index,
    resolve_class_weights,
    to_level_arrays,
)

from ._class_sampler import _RunClassSampler

if TYPE_CHECKING:
    from annbatch.abc import BaseClassSampler


class BoundClassSampler(_RunClassSampler):
    """Replay another class sampler's per-batch class schedule against a second annotation.

    The ``inner_sampler`` decides the class of each batch; this sampler then draws that
    batch's observations from ``classes_to_bind_on``, so two annotations are sampled in
    lockstep. Being a class sampler itself, a bound sampler can serve as another's
    ``inner_sampler``, so these chain.

    ``batch_size`` must be a multiple of ``chunk_size``: one class is drawn per batch and
    expanded into whole chunks, and a chunk shared by two batches could not serve two
    different inner classes.

    .. important::
        Asking the inner sampler for its schedule *draws* one, advancing its random stream.
        Give this sampler its own inner instance rather than one that is also being
        iterated elsewhere, and build the inner with ``drop_last=True`` so every replayed
        batch is full. The inner sampler's emittable classes and batch count must not
        change after binding (both raise if they do).

    Parameters
    ----------
    inner_sampler
        The sampler whose per-batch classes are replayed.
    chunk_size
        Number of observations in each slice yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch; must be a multiple of ``chunk_size``.
    classes_to_bind_on
        A :class:`pandas.Categorical` with one entry per observation, whose classes must
        be a subset of the inner sampler's. Multi-column labels can be built with
        ``pd.MultiIndex.from_arrays(...).to_flat_index()``.
    on
        Which label positions to match on, as ``{inner_position: outer_position}``.
        Defaults to matching whole labels.
    classes
        Optional second annotation subdividing each bound class, so a batch is coherent on
        the bound class *and* drawn from one of its subclasses.
    class_weights
        Weights over ``classes.categories``, controlling how often each subclass is drawn
        within its bound class. A non-positive weight excludes that subclass.
    mask
        Optional contiguous observation range to restrict sampling to.
    rng
        Random number generator used to pick a run and a start within the replayed class.
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
        # Unlike the other class samplers, this one draws one class per *batch* and expands it
        # into `batch_size // chunk_size` whole chunks, so a batch that is not a whole number of
        # chunks cannot replay the inner schedule: a shared chunk would need two classes at once.
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

        # scatter rather than np.unique: shared_obs_codes is n_obs long, so this stays O(n_obs)
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

        joint_classes, weights = self._build_joint(
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
            classes=joint_classes,
            class_weights=weights,
        )

    def _build_joint(
        self,
        shared_obs_codes: np.ndarray,
        shared_classes: pd.Index,
        emittable_shared: np.ndarray,
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[pd.Categorical, np.ndarray]:
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            self._joint_to_shared = np.arange(len(shared_classes))
            self._joint_weight = np.ones(len(shared_classes), dtype=float)
            drawable = np.isin(self._joint_to_shared, emittable_shared).astype(float)
            return pd.Categorical.from_codes(shared_obs_codes, categories=shared_classes.to_flat_index()), drawable

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
        return pd.Categorical.from_codes(joint_codes, categories=labels.to_flat_index()), drawable

    @property
    def classes_to_bind_on(self) -> pd.Categorical:
        return self._classes_to_bind_on

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        present_codes = self._rle_manager.emittable_codes
        present_shared = self._joint_to_shared[present_codes]
        present_weight = self._joint_weight[present_codes]

        inner_of_batch = self._inner_sampler.batch_codes()
        shared_of_batch = self._inner_to_shared[inner_of_batch]
        # get_indexer gives -1 for an inner class absent from classes_to_bind_on. Construction
        # rejects those, but the inner sampler may have gained emittable classes since (a wider
        # mask, say), and -1 would otherwise index the *last* shared class from the end.
        absent = shared_of_batch < 0
        if absent.any():
            raise ValueError(
                f"The inner sampler emits class {self._inner_sampler.vocab[inner_of_batch[absent][0]]!r}, "
                "which is absent from classes_to_bind_on; its emittable classes must not widen after binding."
            )
        undrawable = shared_of_batch[~np.isin(shared_of_batch, present_shared)]
        if undrawable.size:
            raise ValueError(
                f"Class {self._shared_classes[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )

        if shared_of_batch.shape[0] * self._group_chunks != n_slices:
            raise ValueError(
                f"The inner sampler scheduled {shared_of_batch.shape[0]} batches, but this pass covers "
                f"{n_slices // self._group_chunks}; its n_batches() must not change after binding."
            )
        positions = grouped_weighted_choice(present_shared, present_weight, shared_of_batch, self._class_rng)
        # one draw per batch, expanded over the batch's chunks (the ctor guarantees a whole number)
        return np.repeat(positions, self._group_chunks)
