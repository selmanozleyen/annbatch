"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import itertools
import warnings
from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pandas import DataFrame

    from annbatch.utils import WorkerHandle


T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class LoadRequest[T_co]:
    """Load request from sampler."""

    # below the explanations are for when T_co = list[slice]
    # slices to load
    # a list of at most chunk_size ranged slices
    slices: T_co
    # how the concatenation of slices should be split into batches
    # a list of splits, last one may be partial (< batch_size)
    # the loader carries over partial batches to the next iteration
    splits: list[np.ndarray]


class Sampler[T_co](ABC):
    """Base sampler class."""

    @abstractmethod
    def __iter__(self) -> Iterator[LoadRequest[T_co]]:
        """Iterator over load requests."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of iterations to exhaust the sampler."""

    @property
    @abstractmethod
    def batch_size(self) -> int | None:
        """The batch size of the sampler if valid."""

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        """Set the worker handle if desired. If the sampler doesn't support workers, this is a no-op."""
        # this is a separate method because we'd want the LoaderBuilder itself
        # passing this to the sampler, this way Loader doesn't need to know about
        # the sampler's worker handle and knows every Sampler supports this
        # but we don't want to force every sampler to implement this
        del worker_handle  # to explicitly show that we don't use the worker handle
        return None

    def supports_workers(self) -> bool:
        """Return whether the sampler supports workers."""
        return False


class SliceSampler(Sampler[list[slice]]):
    """Chunk-based slice sampler for batched data access.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each slice.
    start_index
        Starting observation index (inclusive).
    end_index
        Ending observation index (exclusive).
    shuffle
        Whether to shuffle chunk and index order.
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling.
    """

    _batch_size: int
    _chunk_size: int
    _shuffle: bool
    _preload_nchunks: int
    _start_index: int
    _end_index: int
    _n_chunks: int
    _n_iters: int
    _drop_last: bool
    _rng: np.random.Generator

    def __init__(
        self,
        *,
        batch_size: int,
        chunk_size: int,
        start_index: int,
        end_index: int,
        shuffle: bool = False,
        preload_nchunks: int,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        if start_index < 0 or start_index >= end_index:
            raise ValueError("start_index must be >= 0 and < end_index")
        check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preload chunks"])
        preload_size = chunk_size * preload_nchunks

        # TODO: asses this: Should we make this check mandatory in the base Sampler class?
        # Because this is actually a requirement for any sampler to run in a Loader class
        if batch_size > preload_size:
            raise ValueError(
                "batch_size cannot exceed chunk_size * preload_nchunks. "
                f"Got batch_size={batch_size}, but max is {preload_size}."
            )

        n_batches = (
            math.floor((end_index - start_index) / batch_size)
            if drop_last
            else math.ceil((end_index - start_index) / batch_size)
        )
        total_yielded_obs = n_batches * batch_size

        self._rng = np.random.default_rng() if rng is None else rng
        self._n_iters = math.ceil(total_yielded_obs / (chunk_size * preload_nchunks))
        self._n_chunks = math.ceil((end_index - start_index) / chunk_size)
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks
        self._start_index = start_index
        self._end_index = end_index
        self._drop_last = drop_last
        self._worker_handle = None

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[LoadRequest[list[slice]]]:
        # Compute slices directly from index range
        slices = self._compute_slices()
        n_slices = len(slices)

        # Create slice indices for shuffling
        slice_indices = np.arange(n_slices)
        if self._shuffle:
            slice_indices = self._shuffle_integers(slice_indices)

        # Worker sharding: each worker gets a disjoint subset of slices
        if self._worker_handle is not None:
            slice_indices = self._worker_handle.get_part_for_worker(slice_indices)

        n_slices_for_worker = len(slice_indices)
        n_slice_iters = math.ceil(n_slices_for_worker / self._preload_nchunks) if n_slices_for_worker > 0 else 0

        n_leftover_indices = 0

        for i in range(n_slice_iters):
            start = i * self._preload_nchunks
            end = min(start + self._preload_nchunks, n_slices_for_worker)
            indices_to_load = slice_indices[start:end]

            # Compute total observations to load from selected slices
            total_obs_to_load = sum(slices[idx].stop - slices[idx].start for idx in indices_to_load)

            # Generate loaded indices with leftover from previous iteration
            loaded_indices = np.arange(total_obs_to_load + n_leftover_indices)
            if self._shuffle:
                loaded_indices = self._shuffle_integers(loaded_indices)
            splits = list(np.split(loaded_indices, np.arange(self._batch_size, len(loaded_indices), self._batch_size)))

            is_last_iter = i == n_slice_iters - 1
            last_is_partial = splits[-1].shape[0] < self._batch_size

            if last_is_partial:
                if is_last_iter and self._drop_last:
                    # Drop the final partial batch entirely
                    splits = splits[:-1]
                    n_leftover_indices = 0
                else:
                    # Track leftover count for next iteration's index generation
                    # splits[-1] is partial and will be carried over by Loader
                    n_leftover_indices = splits[-1].shape[0]
            else:
                n_leftover_indices = 0

            yield LoadRequest(
                slices=[slices[idx] for idx in indices_to_load],
                splits=splits,
            )

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        # Worker mode validation - only applies with multiple workers
        num_workers = worker_handle.num_workers
        preload_size = self._preload_nchunks * self._chunk_size
        if num_workers > 1 and preload_size % self._batch_size != 0:
            if self._drop_last:
                warnings.warn(
                    "With drop_last=True and multiple workers, up to "
                    "(batch_size - 1) * num_workers observations may be dropped "
                    "(one partial batch per worker).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    f"When using DataLoader workers with drop_last=False, "
                    f"(chunk_size * preload_nchunks) must be divisible by batch_size. "
                    f"Got {preload_size} % {self._batch_size} = {preload_size % self._batch_size}. "
                    f"Set drop_last=True to allow non-divisible configs."
                )
        # TODO: asses this: should we raise an error if worker handle is already set?
        self._worker_handle = worker_handle

    def supports_workers(self) -> bool:
        return True

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            # TODO: deal with generators later
            self._rng.shuffle(integers)
        else:
            # should we always have a worker handle? even if its no-op?
            self._worker_handle.shuffle(integers)
        return integers

    def _compute_slices(self) -> list[slice]:
        """Compute slices directly from start/end indices."""
        starts = list(range(self._start_index, self._end_index, self._chunk_size))
        stops = starts[1:] + [self._end_index]
        return [slice(start, stop) for start, stop in zip(starts, stops, strict=False)]

    @property
    def batch_size(self) -> int | None:
        return self._batch_size


@dataclass(frozen=True)
class GroupRange:
    """A named range of observations assigned to a specific distributed node.

    Parameters
    ----------
    name
        Identifier for this group (e.g., cell type name).
    start
        Starting observation index (inclusive).
    end
        Ending observation index (exclusive).
    node
        Which distributed node handles this group (default: 0).
    """

    name: str
    start: int
    end: int
    node: int = 0


class RangeGroupSampler(Sampler[list[slice]]):
    """Sampler yielding group-pure batches with distributed and multi-worker support.

    Each batch contains only elements from one group. Groups are assigned
    to distributed nodes; workers within a node shard the slices of their groups.

    Internally composes SliceSampler instances for each group to avoid code duplication.

    `drop_last=True` is enforced for simplicity.

    Parameters
    ----------
    groups
        Sequence of GroupRange objects defining observation ranges and node assignments.
    rank
        This node's distributed rank (0 if not using distributed training).
    world_size
        Total number of distributed nodes (1 if not using distributed training).
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk (range of each slice).
    preload_nchunks
        Number of chunks to load per iteration.
    shuffle_groups
        Whether to shuffle group order each epoch.
    shuffle_within
        Whether to shuffle indices within each group.
    interleave
        Whether to alternate batches between groups (True) or exhaust one group
        before moving to the next (False). Default is True.
    rng
        Random number generator for shuffling.
    """

    def __init__(
        self,
        *,
        groups: list[GroupRange],
        rank: int = 0,
        world_size: int = 1,
        batch_size: int,
        chunk_size: int,
        preload_nchunks: int,
        shuffle_groups: bool = True,
        shuffle_within: bool = True,
        interleave: bool = True,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([chunk_size, preload_nchunks, batch_size], ["Chunk size", "Preload chunks", "Batch size"])
        preload_size = chunk_size * preload_nchunks
        if preload_size % batch_size != 0:
            raise ValueError(
                f"(chunk_size * preload_nchunks) must be divisible by batch_size. "
                f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
            )

        # Validate group sizes
        for group in groups:
            group_size = group.end - group.start
            if group_size < preload_size:
                raise ValueError(
                    f"Group '{group.name}' has size {group_size} which is smaller than "
                    f"preload_size (chunk_size * preload_nchunks = {preload_size}). "
                    f"Each group must contain at least preload_size observations."
                )

        self._groups = groups
        self._rank = rank
        self._world_size = world_size
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle_groups = shuffle_groups
        self._shuffle_within = shuffle_within
        self._interleave = interleave
        self._rng = np.random.default_rng() if rng is None else rng
        self._worker_handle: WorkerHandle | None = None

        # Filter groups for this node
        self._my_groups = [g for g in groups if g.node == rank]

        # Create samplers for each group
        self._group_samplers = [
            SliceSampler(
                start_index=group.start,
                end_index=group.end,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                shuffle=shuffle_within,
                drop_last=True,  # Enforced for RangeGroupSampler
                rng=self._rng,
            )
            for group in self._my_groups
        ]

    @classmethod
    def from_obs(
        cls,
        obs: DataFrame,
        group_col: str,
        node_mapping: dict[str, int],
        *,
        rank: int = 0,
        world_size: int = 1,
        batch_size: int,
        chunk_size: int,
        preload_nchunks: int,
        **kwargs,
    ) -> "RangeGroupSampler":
        """Create sampler from sorted obs DataFrame.

        Parameters
        ----------
        obs
            DataFrame with observations (must be sorted by group_col).
        group_col
            Column name containing group labels.
        node_mapping
            Mapping from group name to distributed node id.
        rank
            This node's distributed rank.
        world_size
            Total number of distributed nodes.
        batch_size
            Number of observations per batch.
        chunk_size
            Size of each chunk.
        preload_nchunks
            Number of chunks to load per iteration.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        RangeGroupSampler
            Sampler configured with groups detected from obs.
        """

        # TODO: assert sorted
        group_values = obs[group_col].values

        # Get group keys and their sizes using groupby
        group_info = [(key, sum(1 for _ in g)) for key, g in itertools.groupby(group_values)]

        # Compute cumulative end positions
        ends = list(itertools.accumulate(size for _, size in group_info))
        starts = [0] + ends[:-1]

        groups = [
            GroupRange(
                name=str(key),
                start=start,
                end=end,
                node=node_mapping.get(str(key), 0),
            )
            for (key, _), start, end in zip(group_info, starts, ends, strict=True)
        ]

        return cls(
            groups=groups,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            **kwargs,
        )

    def __len__(self) -> int:
        # Count total batches across all groups for this node
        total_batches = 0
        for group in self._my_groups:
            group_size = group.end - group.start
            # drop_last is enforced
            total_batches += group_size // self._batch_size
        return total_batches

    def __iter__(self) -> Iterator[LoadRequest[list[slice]]]:
        # Get list of samplers, optionally shuffle order
        samplers = list(self._group_samplers)
        if self._shuffle_groups:
            self._rng.shuffle(samplers)

        if self._interleave:
            # Interleaved mode: reshuffle after each round
            iterators = [iter(s) for s in samplers]
            while iterators:
                if self._shuffle_groups:
                    self._rng.shuffle(iterators)
                still_active = []
                for it in iterators:
                    try:
                        yield next(it)
                        still_active.append(it)
                    except StopIteration:
                        pass
                iterators = still_active
        else:
            # Sequential mode: exhaust one group before moving to next
            for sampler in samplers:
                yield from sampler

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        """Set the worker handle for multi-worker data loading."""
        self._worker_handle = worker_handle
        # Propagate to cached group samplers
        for sampler in self._group_samplers:
            sampler.set_worker_handle(worker_handle)

    def supports_workers(self) -> bool:
        return True

    @property
    def batch_size(self) -> int | None:
        return self._batch_size


