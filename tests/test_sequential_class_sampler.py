"""Tests for SequentialClassSampler.

The guarantees under test: each scheduled class yields exactly one batch = all its cells in obs order
(even across several non-contiguous runs), the schedule covers each scheduled cell exactly once, short
runs are read (the ``chunk_size`` run-length rule is opted out), schedule order/repeats and ``mask`` are
honored, ``batch_codes`` equals the schedule, and absent-class / NA inputs raise. One end-to-end test
reads through the real :class:`~annbatch.Loader`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import SequentialClassSampler


def _cat(codes: list[int], n_classes: int | None = None) -> pd.Categorical:
    k = (max(codes) + 1) if n_classes is None else n_classes
    return pd.Categorical.from_codes(codes, categories=[str(i) for i in range(k)])


def _read_indices(sampler: SequentialClassSampler) -> list[np.ndarray]:
    """The obs indices each batch loads, in order, reconstructed from the emitted LoadRequests.

    Mirrors the loader's contract for a single dataset: requests (slices) are concatenated in order into
    a buffer, then each split indexes into it.
    """
    out: list[np.ndarray] = []
    for lr in sampler.sample(sampler._n_obs):
        buf = np.concatenate([np.arange(s.start, s.stop) for s in lr["requests"]])
        out.extend(buf[split] for split in lr["splits"])
    return out


def test_multi_run_class_read_in_full_and_in_order():
    # class 0 spans two runs: rows [0,1,2] and [5,6]; class 1 is rows [3,4].
    s = SequentialClassSampler(_cat([0, 0, 0, 1, 1, 0, 0]))
    s.set_schedule([0, 1])
    batches = _read_indices(s)

    assert len(batches) == 2  # one batch per scheduled class
    np.testing.assert_array_equal(batches[0], [0, 1, 2, 5, 6])  # both runs of class 0, in obs order
    np.testing.assert_array_equal(batches[1], [3, 4])


def test_schedule_order_and_repeats_honored():
    s = SequentialClassSampler(_cat([0, 0, 1, 1, 2, 2]))
    s.set_schedule([2, 0, 2])  # out of order + a repeat
    batches = _read_indices(s)

    assert [list(b) for b in batches] == [[4, 5], [0, 1], [4, 5]]
    np.testing.assert_array_equal(s.batch_codes(), [2, 0, 2])
    assert s.n_batches(6) == 3


def test_default_schedule_reads_all_classes_once_ascending_incl_len1_run():
    # no schedule → all present classes, ascending code order. class 2 is a single length-1 run, read
    # fine because the chunk_size run-length rule is opted out for this sampler.
    s = SequentialClassSampler(_cat([1, 1, 0, 0, 0, 2]))
    batches = _read_indices(s)
    assert [list(b) for b in batches] == [[2, 3, 4], [0, 1], [5]]


def test_full_coverage_exactly_once_across_schedule():
    rng = np.random.default_rng(0)
    codes = list(rng.integers(0, 4, size=200))
    s = SequentialClassSampler(_cat(codes))
    s.set_schedule([0, 1, 2, 3])
    seen = np.concatenate(_read_indices(s))
    np.testing.assert_array_equal(np.sort(seen), np.arange(200))  # each row exactly once


def test_mask_restricts_and_reindexes_range():
    # rows 2..5 = [1, 1, 0, 0]; class 1 -> [2, 3], class 0 -> [4, 5] (global coords)
    s = SequentialClassSampler(_cat([0, 0, 1, 1, 0, 0]), mask=slice(2, 6))
    assert [list(b) for b in _read_indices(s)] == [[4, 5], [2, 3]]


def test_scheduling_absent_class_raises():
    s = SequentialClassSampler(_cat([0, 0, 1, 1], n_classes=3))  # class 2 has no cells
    with pytest.raises(ValueError, match="no observations"):
        s.set_schedule([0, 2])


def test_na_codes_rejected():
    with pytest.raises(ValueError, match="NA values"):
        SequentialClassSampler(pd.Categorical([np.nan, "a", "b"], categories=["a", "b"]))


def test_end_to_end_through_loader():
    from annbatch import Loader

    x = np.arange(7, dtype="float32").reshape(-1, 1)
    s = SequentialClassSampler(_cat([0, 0, 0, 1, 1, 0, 0]))
    s.set_schedule([1, 0])
    loader = Loader(batch_sampler=s, return_index=False, to=None, preload_to_gpu=False).add_datasets([x])
    got = [np.asarray(batch["X"]).ravel() for batch in loader]

    assert len(got) == 2
    np.testing.assert_array_equal(got[0], [3, 4])  # class 1
    np.testing.assert_array_equal(got[1], [0, 1, 2, 5, 6])  # class 0, both runs
