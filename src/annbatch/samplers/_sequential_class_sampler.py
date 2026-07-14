"""SequentialClassSampler -- deterministic, exact-once, class-coherent full reads of chosen classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from annbatch.samplers._class_sampler import _RunClassSampler
from annbatch.samplers._utils import codes_of_categorical

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from annbatch.types import LoadRequest


class SequentialClassSampler(_RunClassSampler):
    """Read every observation of a supplied sequence of classes exactly once, one class per batch.

    The evaluation/inference counterpart to :class:`~annbatch.samplers.ClassSampler`: where that draws
    class-coherent batches *with replacement* (a random chunk-sized slice within a run), this reads each
    scheduled class *in full, once*, as contiguous slices straight off disk -- no shuffling, no RNG. It
    is what a downstream consumer uses when metrics need each class's whole cell set (e.g. all cells of a
    perturbation condition), grouped, without materializing the full matrix and boolean-masking it.

    Each scheduled class becomes exactly **one** batch = all of its observations, concatenated in obs
    order. The schedule (see :meth:`set_schedule`) may list classes in any order and may repeat them
    (each occurrence re-reads the class in full); with no schedule, every class present in the (masked)
    range is read once in ascending code order.

    Reuses :class:`_RunClassSampler`'s run-length machinery (``_class_runs`` /
    ``_per_class_sampling_info`` / :attr:`mask` / :attr:`vocab` / :meth:`emittable_codes`); the
    :attr:`~_RunClassSampler._enforce_run_length_rule` opt-out disables the ``chunk_size`` run-length
    check, which is meaningless for a full-run read. As a :class:`~annbatch.abc.BaseClassSampler` it is
    itself bindable: its :meth:`batch_codes` is exactly the schedule, so a
    :class:`~annbatch.samplers.BoundClassSampler` can replay it. Multiple workers are not supported.

    Parameters
    ----------
    classes
        A :class:`pandas.Categorical` with one entry per observation (the class/leaf code of each cell),
        the same input :class:`~annbatch.samplers.ClassSampler` takes. NA values (``codes == -1``) are
        not allowed.
    mask
        Optional contiguous observation range ``[start, stop)`` to restrict reads to. Defaults to the
        whole dataset.
    schedule
        Optional initial schedule; see :meth:`set_schedule`. If omitted, all classes present in the
        (masked) range are read once, in ascending code order.
    """

    # whole runs are read in full; the chunk-sized-run rule does not apply (see _RunClassSampler).
    _enforce_run_length_rule = False

    def __init__(
        self,
        classes: pd.Categorical,
        *,
        mask: slice | None = None,
        schedule: np.ndarray | None = None,
    ) -> None:
        codes = codes_of_categorical(classes, "classes")
        n_classes = len(classes.categories)
        # chunk/preload/batch and num_samples are vestigial here (the request stream is overridden to
        # emit whole-class reads); pass the minimal sizes the base validator accepts. All classes get a
        # positive weight so every one is emittable — selection is via the schedule, not weights.
        super().__init__(
            chunk_size=1,
            preload_nchunks=1,
            batch_size=1,
            num_samples=1,
            drop_last=False,
            mask=mask,
            rng=None,
            codes=codes,
            weights=np.ones(n_classes, dtype=float),
            category_labels=classes.categories,
        )
        self._schedule: np.ndarray | None = None
        if schedule is not None:
            self.set_schedule(schedule)

    def set_schedule(self, class_codes: np.ndarray | None) -> None:
        """Set (or clear) the ordered class codes to read; each is read fully, once. Repeats allowed.

        A code not present in the current (masked) range raises -- an evaluation condition with no cells
        is a bug, not something to silently skip. ``None`` restores the default (all present classes,
        once, ascending code order).
        """
        if class_codes is None:
            self._schedule = None
            return
        sched = np.asarray(class_codes, dtype=np.int64)
        emittable = set(self.emittable_codes().tolist())
        missing = sorted({int(c) for c in sched} - emittable)
        if missing:
            raise ValueError(f"scheduled class code(s) {missing} have no observations in the current range.")
        self._schedule = sched

    def _iter_codes(self) -> np.ndarray:
        return self._schedule if self._schedule is not None else self.emittable_codes()

    @property
    def shuffle(self) -> bool:
        return False

    def n_batches(self, n_obs: int) -> int:
        del n_obs  # one batch per scheduled class, not derived from num_samples/loader size
        return int(self._iter_codes().shape[0])

    def batch_codes(self) -> np.ndarray:
        """The class code of each batch a full pass yields -- exactly the schedule (one class per batch)."""
        return np.asarray(self._iter_codes(), dtype=np.int64)

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:  # pragma: no cover - request stream is overridden
        raise NotImplementedError("SequentialClassSampler generates its own whole-class request stream.")

    def _iter_requests(self) -> Iterator[LoadRequest]:
        # Runs are stored sorted by class (stable) with build_run_table's left-to-right order preserved
        # within a class, so a class's rows in _class_runs are already in ascending-start (obs) order.
        starts = self._class_runs["start"].to_numpy()
        ends = self._class_runs["end"].to_numpy()
        first = self._per_class_sampling_info["first_row_in_runs_of_class"]
        n_runs = self._per_class_sampling_info["n_runs"]
        for code in self._iter_codes():
            c = int(code)
            lo = int(first.loc[c])
            hi = lo + int(n_runs.loc[c])
            requests = [slice(int(s), int(e)) for s, e in zip(starts[lo:hi], ends[lo:hi], strict=True)]
            total = int((ends[lo:hi] - starts[lo:hi]).sum())
            # one whole-class split → the loader concatenates the runs (obs order) into a single batch.
            yield {"requests": requests, "splits": [np.arange(total)]}
