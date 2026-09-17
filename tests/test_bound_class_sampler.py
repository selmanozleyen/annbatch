"""Tests for BoundClassSampler.

The sampler replays another class sampler's per-batch class schedule against a second
categorical, so the tests check that the replay is faithful, that every batch stays a
full single-class read, and that the public knobs (``on``, ``classes`` /
``class_weights``, ``mask``) do what they say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import BoundClassSampler, ClassSampler
from annbatch.samplers._utils import project_index
from tests.conftest import make_class_sampler

CT = np.repeat(["B", "T"], 100)


def make_inner(labels, **kwargs) -> ClassSampler:
    """The shared ClassSampler factory, with drop_last on: a bound sampler replays whole batches."""
    classes = labels if isinstance(labels, pd.Categorical) else pd.Categorical(labels)
    return make_class_sampler(classes, drop_last=True, **kwargs)


def make_bound(inner, classes_to_bind_on, *, chunk_size=10, preload_nchunks=4, batch_size=10, seed=1, **kwargs):
    return BoundClassSampler(
        inner,
        chunk_size,
        preload_nchunks,
        batch_size,
        classes_to_bind_on=classes_to_bind_on
        if isinstance(classes_to_bind_on, pd.Categorical)
        else pd.Categorical(classes_to_bind_on),
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def batches(sampler, n_obs: int) -> list[np.ndarray]:
    """The obs indices of every batch a full pass yields."""
    out = []
    for load_request in sampler.sample(n_obs):
        window = np.concatenate([np.arange(s.start, s.stop) for s in load_request["requests"]])
        out.extend(window[split] for split in load_request["splits"])
    return out


def batch_keys(sampler, key_of_obs, n_obs: int) -> list:
    """The single key each batch is coherent on; asserts that coherence."""
    key = np.asarray(key_of_obs)
    keys = []
    for idx in batches(sampler, n_obs):
        unique = set(key[idx].tolist())
        assert len(unique) == 1, "each batch must be coherent on the bound key"
        keys.append(unique.pop())
    return keys


def expected_labels(sampler, positions: tuple[int, ...] | None = None) -> list:
    """The per-batch class labels a standalone pass of ``sampler`` would schedule."""
    return list(project_index(sampler.vocab, positions)[sampler.batch_codes()])


def joint(*columns) -> pd.Categorical:
    """A flat multi-column categorical, one tuple label per row."""
    return pd.Categorical(pd.MultiIndex.from_arrays(columns).to_flat_index())


@pytest.mark.parametrize(
    ("kwargs", "classes_to_bind_on", "match"),
    [
        pytest.param(
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            CT,
            "batch_size must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param({}, ["B"] * 200, "absent from classes_to_bind_on", id="inner_class_absent"),
        pytest.param({}, ["B"] * 100 + ["Z"] * 100, "subset of the inner sampler's classes", id="not_subset_of_inner"),
        pytest.param({}, (["B"] * 3 + ["T"] * 97) * 2, "at least chunk_size", id="run_too_short"),
    ],
)
def test_invalid_construction(kwargs, classes_to_bind_on, match):
    with pytest.raises(ValueError, match=match):
        make_bound(make_inner(CT, num_samples=100), classes_to_bind_on, **kwargs)


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(5, 10, 4, id="batch_two_chunks"),
        pytest.param(5, 20, 4, id="batch_four_chunks"),
    ],
)
def test_every_batch_is_full_and_coherent(chunk_size, batch_size, preload_nchunks):
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100))
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    sampler = make_bound(
        inner, condition, chunk_size=chunk_size, batch_size=batch_size, preload_nchunks=preload_nchunks
    )

    idxs = batches(sampler, len(condition))
    assert len(idxs) == inner.n_batches(0)
    assert all(idx.size == batch_size for idx in idxs), "every batch must be full"
    assert all(len(set(condition.codes[idx])) == 1 for idx in idxs), "every batch must be class-coherent"


def test_replays_inner_per_batch_classes():
    a_labels = pd.Categorical(np.repeat(["B", "T", "NK", "Mono"], 100))
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    sampler = make_bound(make_inner(a_labels, seed=7), condition)

    assert batch_keys(sampler, condition, len(condition)) == expected_labels(make_inner(a_labels, seed=7))


def test_chained_bound_replays_mid_schedule():
    def build_mid():
        return make_bound(make_inner(CT, num_samples=200), CT)

    outer = make_bound(build_mid(), CT, seed=2)
    assert outer.n_batches(0) == build_mid().n_batches(0)
    assert batch_keys(outer, CT, len(CT)) == expected_labels(build_mid()), "a bound sampler is itself bindable"


def test_on_binds_subset_by_position():
    # four contiguous blocks, so every run comfortably exceeds chunk_size
    inner_classes = joint(*(np.repeat(col, 40) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2, ["x", "y"] * 2)))
    condition = joint(*(np.repeat(col, 30) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2)))
    sampler = make_bound(make_inner(inner_classes, num_samples=2000), condition, on={0: 0, 1: 1})

    expected = expected_labels(make_inner(inner_classes, num_samples=2000), (0, 1))
    assert batch_keys(sampler, project_index(condition.categories, (0, 1))[condition.codes], len(condition)) == expected


def test_within_class_weights_shares():
    donor = pd.Categorical((["d1"] * 20 + ["d2"] * 20) * 10)
    sampler = make_bound(
        make_inner(pd.Categorical(["B"] * 200), num_samples=40_000),
        pd.Categorical(["B"] * 400),
        classes=donor,
        class_weights=np.array([3.0, 1.0]),
    )

    drawn = donor.codes[np.concatenate(batches(sampler, len(donor)))]
    shares = np.bincount(drawn, minlength=2) / drawn.size
    assert abs(shares[0] - 0.75) < 0.02 and abs(shares[1] - 0.25) < 0.02


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via):
    inner = make_inner(CT, num_samples=500, class_weights=np.array([1.0, 0.0]))
    condition = pd.Categorical(CT)
    if via == "constructor":
        sampler = make_bound(inner, condition, mask=slice(0, 100))
    else:
        sampler = make_bound(inner, condition)
        sampler.mask = slice(0, 100)
    chunks = [c for lr in sampler.sample(len(condition)) for c in lr["requests"]]
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"
