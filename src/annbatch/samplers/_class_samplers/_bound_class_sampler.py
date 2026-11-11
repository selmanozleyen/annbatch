"""BoundClassSampler -- replay another class sampler's per-batch schedule on a second column."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._utils import codes_of_categorical

from ._class_sampler import ClassSampler, _RunClassSampler

if TYPE_CHECKING:
    from ._class_sampler import _ClassSchedule
    from ._rle_manager import RLEManager


def to_level_arrays(index: pd.Index) -> list[pd.Index]:
    """Split a (possibly flattened) MultiIndex into one array per level."""
    if not isinstance(index, pd.MultiIndex):
        if len(index) == 0 or not isinstance(index[0], tuple):
            return [index]
        index = pd.MultiIndex.from_tuples(index)
    return [index.get_level_values(i) for i in range(index.nlevels)]


class BoundClassSampler(_RunClassSampler):
    """Replay another class sampler's per-batch class schedule against a second annotation.

    The ``inner_sampler`` decides the class of each batch; this sampler then draws that
    batch's observations from ``classes_to_bind_on``, so one pass reads two annotations in step
    with each other. A bound sampler can itself serve as another's ``inner_sampler``, so these
    chain.

    ``batch_size`` must be a multiple of ``chunk_size``: the replayed class of a batch is read
    as whole chunks, and a chunk shared by two batches could not serve two different inner
    classes.

    .. important::
        The inner sampler is taken by value, not by reference: its schedule is copied at
        construction, so nothing you do to your own instance afterwards reaches this one.

        ``mask`` is *not* forwarded: the inner sampler supplies a class sequence, not rows, and
        need not even span the same observations. If it schedules a class that has no run left
        inside your range, the pass raises rather than substituting another.

        Build the inner with ``drop_last=True`` unless its ``batch_size`` divides its
        ``num_samples``: a trailing short batch of the inner's is replayed here as a *full*
        batch, so the pass yields the same number of batches but more rows.

    Parameters
    ----------
    inner_sampler
        The sampler whose per-batch classes are replayed: a
        :class:`~annbatch.samplers.ClassSampler` or another
        :class:`~annbatch.samplers.BoundClassSampler`. A
        :class:`~annbatch.samplers.WeightedClassSampler` mixes classes within a batch and so
        has no per-batch class to replay.
    chunk_size
        Number of observations in each slice yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch; must be a multiple of ``chunk_size``.
    classes_to_bind_on
        A :class:`pandas.Categorical` with one entry per observation, whose classes must
        be a subset of the inner sampler's. Multi-column labels can be built with
        ``pd.Categorical(pd.MultiIndex.from_arrays(...).to_flat_index())``.
    on
        Positions *within the inner sampler's own labels* to match on, for an inner built on
        multi-column labels. ``on=(0,)`` matches ``classes_to_bind_on`` against the first
        column of each inner label. Give one position per column of ``classes_to_bind_on``,
        whose labels must then be a subset of the *projected* inner labels. Defaults to
        matching whole labels.
    classes
        Optional second annotation subdividing each bound class, so a batch is coherent on
        the bound class *and* drawn from one of its secondary classes.
    class_weights
        Weights over ``classes.categories``, controlling how often each secondary class is drawn
        within its bound class. A non-positive weight excludes that secondary class.
    mask
        Optional contiguous observation range to restrict sampling to.
    rng
        Random number generator for everything this sampler draws: the schedule it replays,
        and the run and start within each replayed class. Passing it here is the same as
        assigning :attr:`rng` afterwards.
    """

    _schedule: _BoundSchedule

    def __init__(
        self,
        inner_sampler: ClassSampler | BoundClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        classes_to_bind_on: pd.Categorical,
        on: tuple[int, ...] | None = None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not isinstance(inner_sampler, ClassSampler | BoundClassSampler):
            raise TypeError(
                f"inner_sampler must be a ClassSampler or a BoundClassSampler, got "
                f"{type(inner_sampler).__name__}, which has no per-batch class to replay."
            )
        # Unlike the other class samplers, this one replays one class per *batch* as whole
        # chunks, so a batch that is not a whole number of chunks cannot be replayed: a shared
        # chunk would need two classes at once.
        if batch_size % chunk_size != 0:
            raise ValueError(
                "batch_size must be a multiple of chunk_size so each batch replays one inner class as whole chunks. "
                f"Got chunk_size={chunk_size}, batch_size={batch_size}."
            )
        bound_codes = codes_of_categorical(classes_to_bind_on, "classes_to_bind_on")
        # the one private member read from the inner sampler
        inner_schedule = inner_sampler._schedule

        # matched against whole labels, or with `on` against the label columns it picks
        inner_labels_on = inner_schedule.labels
        if on is not None:
            try:
                on = tuple(int(position) for position in on)
            except TypeError:
                raise TypeError(f"on must be a tuple of label positions or None, got {type(on).__name__}.") from None
            inner_levels = to_level_arrays(inner_schedule.labels)
            n_inner_columns = len(inner_levels)
            if not on or any(not 0 <= position < n_inner_columns for position in on):
                raise ValueError(
                    f"on must be a non-empty tuple of positions in [0, {n_inner_columns}), since the inner sampler's "
                    f"labels have {n_inner_columns} column(s). Got {on}. To sample without matching, use an "
                    "independent ClassSampler with the same number of batches."
                )
            n_bound_columns = len(to_level_arrays(classes_to_bind_on.categories))
            if len(on) != n_bound_columns:
                raise ValueError(
                    f"on selects {len(on)} of the inner sampler's label columns but classes_to_bind_on labels "
                    f"have {n_bound_columns}; they must match to be compared. Got on={on}."
                )
            inner_labels_on = (
                inner_levels[on[0]] if len(on) == 1 else pd.MultiIndex.from_arrays([inner_levels[p] for p in on])
            )

        # a Categorical's categories are unique, so bound_codes already index them directly
        bound_classes = classes_to_bind_on.categories
        inner_to_bound = bound_classes.get_indexer(inner_labels_on)

        # scatter rather than np.unique: bound_codes is n_obs long, so this stays O(n_obs)
        present = np.zeros(len(bound_classes), dtype=bool)
        present[bound_codes] = True
        present_codes = np.flatnonzero(present)

        present_classes = bound_classes[present_codes]
        not_in_inner = inner_labels_on.unique().get_indexer(present_classes) < 0
        if not_in_inner.any():
            raise ValueError(
                f"classes_to_bind_on has classes {list(present_classes[not_in_inner])} not present in the inner "
                "sampler's classes; classes_to_bind_on must be a subset of the inner sampler's classes."
            )
        bound_of_emittable = inner_to_bound[inner_schedule.codes]
        drawable = np.isin(bound_of_emittable, present_codes)
        if not drawable.all():
            missing = inner_labels_on[inner_schedule.codes][~drawable].unique()
            raise ValueError(f"The inner sampler can emit classes {list(missing)} absent from classes_to_bind_on.")
        emittable_bound_codes = np.unique(bound_of_emittable)

        # the categorical this sampler draws from: the bound class, optionally times a secondary class
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            joint_to_bound = np.arange(len(bound_classes))
            joint_weight = np.ones(len(bound_classes), dtype=float)
            joint_classes = pd.Categorical.from_codes(bound_codes, categories=bound_classes.to_flat_index())
        else:
            secondary_codes = codes_of_categorical(classes, "classes")
            if len(classes) != bound_codes.shape[0]:
                raise ValueError(
                    f"classes must be the same length as classes_to_bind_on ({bound_codes.shape[0]}), "
                    f"got {len(classes)}."
                )
            n_secondary = len(classes.categories)
            if hasattr(class_weights, "index"):
                raise TypeError(
                    "class_weights must be an array, not a pandas Series: a Series would be read in positional "
                    "order and its index ignored. Pass class_weights.reindex(classes.categories).to_numpy()."
                )
            secondary_weights = (
                np.ones(n_secondary) if class_weights is None else np.asarray(class_weights, dtype=float)
            )
            if secondary_weights.shape != (n_secondary,):
                raise ValueError(
                    f"class_weights must have one weight per class in classes.categories "
                    f"(expected shape ({n_secondary},), got {secondary_weights.shape})."
                )
            if not (secondary_weights > 0).any():
                raise ValueError("class_weights must have at least one positive weight.")

            joint_codes, joint_keys = pd.factorize(bound_codes.astype(np.int64) * n_secondary + secondary_codes)
            joint_to_bound, joint_to_secondary = joint_keys // n_secondary, joint_keys % n_secondary
            joint_weight = secondary_weights[joint_to_secondary]
            joint_labels = pd.MultiIndex.from_arrays(
                to_level_arrays(bound_classes.take(joint_to_bound))
                + to_level_arrays(classes.categories.take(joint_to_secondary))
            )
            joint_classes = pd.Categorical.from_codes(joint_codes, categories=joint_labels.to_flat_index())

        super().__init__(
            chunk_size,
            preload_nchunks,
            batch_size,
            classes=joint_classes,
            num_samples=inner_sampler.n_batches(0) * batch_size,
            # zero the joint classes whose bound class the inner sampler can never emit
            class_weights=np.where(np.isin(joint_to_bound, emittable_bound_codes), joint_weight, 0.0),
            mask=mask,
            drop_last=False,
            rng=rng,
        )
        self._schedule = _BoundSchedule(
            # copied, so nothing done to the caller's sampler afterwards reaches this one
            copy.deepcopy(inner_schedule),
            n_inner_batches=inner_sampler.n_batches(0),
            inner_to_bound=inner_to_bound,
            joint_to_bound=joint_to_bound,
            joint_weight=joint_weight,
            rle_manager=self._rle_manager,
            labels=joint_classes.categories,
            bound_labels=bound_classes,
        )

    def _chunk_schedule(self, n_chunks: int) -> np.ndarray:
        # a chunk is read from the class of the batch holding its first row
        return self._schedule.draw(self._rng)[(np.arange(n_chunks) * self._chunk_size) // self._batch_size]


class _BoundSchedule:
    """The class of every batch of a :class:`BoundClassSampler` pass.

    The inner schedule's class for each batch, mapped onto the bound annotation and then to one
    of its secondary classes. It has the same ``labels``, ``codes`` and :meth:`draw` as a
    :class:`ClassSampler`'s schedule, which is what lets a bound sampler be bound to in turn.
    """

    def __init__(
        self,
        inner: _ClassSchedule | _BoundSchedule,
        *,
        n_inner_batches: int,
        inner_to_bound: np.ndarray,
        joint_to_bound: np.ndarray,
        joint_weight: np.ndarray,
        rle_manager: RLEManager,
        labels: pd.Index,
        bound_labels: pd.Index,
    ):
        self.inner, self.n_inner_batches, self.labels = inner, n_inner_batches, labels
        self._inner_to_bound, self._joint_to_bound, self._joint_weight = inner_to_bound, joint_to_bound, joint_weight
        self._rle_manager, self._bound_labels = rle_manager, bound_labels

    @property
    def codes(self) -> np.ndarray:
        return self._rle_manager.codes

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One class code per batch of a pass. The inner schedule draws from ``rng`` first."""
        drawable_codes = self._rle_manager.codes
        drawable_bound = self._joint_to_bound[drawable_codes]
        drawable_weight = self._joint_weight[drawable_codes]

        # only the batches the inner sampler actually yields; with drop_last its schedule holds
        # one more slot, for the short batch it discards.
        inner_of_batch = self.inner.draw(rng)[: self.n_inner_batches]
        # Construction proved every emittable inner class maps to a real bound code, and the copy
        # is private, so no -1 from get_indexer can reach here.
        bound_of_batch = self._inner_to_bound[inner_of_batch]
        # the draw below would silently pick from the wrong group for a class that has no
        # drawable run at all, so check before rather than after.
        undrawable = bound_of_batch[~np.isin(bound_of_batch, drawable_bound)]
        if undrawable.size:
            raise ValueError(
                f"Class {self._bound_labels[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )

        # Draw each batch's joint class from within its bound class, weighted: sort the drawable
        # classes by bound class, so each bound class owns one slice of the cumulative weights,
        # then land a uniform point in that slice.
        order = np.argsort(drawable_bound, kind="stable")
        group, group_start = np.unique(drawable_bound[order], return_index=True)
        group_end = np.append(group_start[1:], order.shape[0])
        cum_weight = np.cumsum(drawable_weight[order])
        group_weight_end = cum_weight[group_end - 1]
        group_weight_start = np.concatenate(([0.0], group_weight_end[:-1]))
        group_of_batch = np.searchsorted(group, bound_of_batch)
        target = group_weight_start[group_of_batch] + rng.random(bound_of_batch.shape[0]) * (
            group_weight_end[group_of_batch] - group_weight_start[group_of_batch]
        )
        chosen = np.clip(
            np.searchsorted(cum_weight, target, side="right"),
            group_start[group_of_batch],
            group_end[group_of_batch] - 1,
        )
        return drawable_codes[order[chosen]]
