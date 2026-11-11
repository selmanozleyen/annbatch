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

from annbatch.samplers import BoundClassSampler, ClassSampler, WeightedClassSampler
from tests.test_class_sampler import make_sampler

CELL_TYPE = np.repeat(["B", "T"], 100)


def make_inner(labels, **kwargs) -> ClassSampler:
    """The shared ClassSampler factory, with drop_last on: a bound sampler replays whole batches."""
    classes = labels if isinstance(labels, pd.Categorical) else pd.Categorical(labels)
    return make_sampler(classes, drop_last=True, **kwargs)


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


def expected_labels(sampler) -> list:
    """The per-batch class labels a bound ``sampler``'s inner schedule would draw.

    That schedule is a private copy drawing from ``sampler``'s own generator, so the oracle must
    be an identically built twin -- asking the caller's own instance would draw different ones.
    """
    schedule = sampler._schedule
    return list(schedule.inner.labels[schedule.inner.draw(sampler.rng)[: schedule.n_inner_batches]])


def joint(*columns) -> pd.Categorical:
    """A flat multi-column categorical, one tuple label per row."""
    return pd.Categorical(pd.MultiIndex.from_arrays(columns).to_flat_index())


# (cell type, donor, batch) for an inner sampler and (cell type, donor) to bind on, in four
# contiguous blocks so every run comfortably exceeds chunk_size
THREE_COLUMNS = joint(*(np.repeat(col, 40) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2, ["x", "y"] * 2)))
TWO_COLUMNS = joint(*(np.repeat(col, 30) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2)))
DONOR = pd.Categorical((["d1"] * 20 + ["d2"] * 20) * 5)


@pytest.mark.parametrize(
    ("inner_cls", "bind_on", "kwargs", "exc", "match"),
    [
        pytest.param(
            ClassSampler,
            CELL_TYPE,
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            ValueError,
            "batch_size must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param(ClassSampler, ["B"] * 200, {}, ValueError, "absent from classes_to_bind_on", id="class_absent"),
        pytest.param(
            ClassSampler, ["B"] * 100 + ["Z"] * 100, {}, ValueError, "subset of the inner", id="not_subset_of_inner"
        ),
        pytest.param(ClassSampler, (["B"] * 3 + ["T"] * 97) * 2, {}, ValueError, "at least chunk_size", id="run_short"),
        # a Series would be read positionally and its index ignored
        pytest.param(
            ClassSampler,
            CELL_TYPE,
            {"classes": DONOR, "class_weights": pd.Series([3.0, 1.0], index=["d2", "d1"])},
            TypeError,
            "not a pandas Series",
            id="series_class_weights",
        ),
        # WeightedClassSampler mixes classes within a batch, so it has no per-batch class to replay
        pytest.param(WeightedClassSampler, CELL_TYPE, {}, TypeError, "must be a ClassSampler", id="weighted_inner"),
    ],
)
def test_invalid_construction(inner_cls, bind_on, kwargs, exc, match):
    inner = make_sampler(pd.Categorical(CELL_TYPE), cls=inner_cls, num_samples=100, drop_last=True)
    with pytest.raises(exc, match=match):
        make_bound(inner, bind_on, **kwargs)


def test_the_schedule_follows_our_rng_not_the_callers_instance():
    # the inner is copied at construction, so re-seeding the caller's own instance cannot reach
    # us -- but re-seeding *ours* must, since DistributedSampler does exactly that per rank
    inner = make_inner(CELL_TYPE, num_samples=200)
    sampler = make_bound(inner, CELL_TYPE)
    before = batch_keys(sampler, CELL_TYPE, len(CELL_TYPE))
    inner.mask, inner.rng = slice(0, 100), np.random.default_rng(999)
    assert (
        batch_keys(make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE), CELL_TYPE, len(CELL_TYPE)) == before
    ), "caller cannot reach us"

    def keys_after_reseeding(seed):
        s = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
        s.rng = np.random.default_rng(seed)
        return batch_keys(s, CELL_TYPE, len(CELL_TYPE))

    assert keys_after_reseeding(0) != keys_after_reseeding(1), "re-seeding us must change the schedule"
    assert keys_after_reseeding(0) == keys_after_reseeding(0), "and must stay reproducible"


def test_two_bound_samplers_off_one_inner_are_independent():
    # the copy is handed our generator, so an inner shared between two bound samplers does not
    # lock them to the same schedule -- and a seeded bound is reproducible even if the inner is not
    inner = make_inner(CELL_TYPE, num_samples=200)
    x, y = make_bound(inner, CELL_TYPE, seed=1), make_bound(inner, CELL_TYPE, seed=2)
    assert batch_keys(x, CELL_TYPE, len(CELL_TYPE)) != batch_keys(y, CELL_TYPE, len(CELL_TYPE)), (
        "different seeds, different schedules"
    )

    unseeded = ClassSampler(10, 4, 10, classes=pd.Categorical(CELL_TYPE), num_samples=200, drop_last=True)
    assert batch_keys(make_bound(unseeded, CELL_TYPE, seed=3), CELL_TYPE, len(CELL_TYPE)) == batch_keys(
        make_bound(unseeded, CELL_TYPE, seed=3), CELL_TYPE, len(CELL_TYPE)
    ), "our seed alone must fix the schedule"


def test_mask_hiding_a_scheduled_class_is_rejected():
    # mask is not forwarded to the inner, so our range can lose a class it still schedules;
    # that must raise rather than quietly read some other class's rows
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
    sampler.mask = slice(0, 100)  # only B survives here, but the inner still emits T

    with pytest.raises(ValueError, match="no drawable run"):
        batches(sampler, len(CELL_TYPE))


def test_a_schedule_naming_an_undrawable_class_is_rejected():
    # searchsorted maps a code that is not emittable onto a *neighbouring* class, so a schedule
    # that names one must raise rather than quietly sample something else
    class AlwaysSecond(ClassSampler):
        def _chunk_schedule(self, n_chunks):
            return np.full(n_chunks, 1, dtype=np.int64)

    sampler = make_sampler(
        pd.Categorical(np.repeat(["A", "B", "C"], 100)),
        cls=AlwaysSecond,
        num_samples=200,
        class_weights=np.array([1.0, 0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="is not drawable in the current range"):
        batches(sampler, 300)


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(5, 10, 4, id="batch_two_chunks"),
        pytest.param(5, 20, 4, id="batch_four_chunks"),
    ],
)
def test_every_batch_is_full_coherent_and_replays_the_inner(chunk_size, batch_size, preload_nchunks):
    # the same four classes as the inner's, ordered differently
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    sizes = {"chunk_size": chunk_size, "batch_size": batch_size, "preload_nchunks": preload_nchunks}
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), seed=7)
    sampler = make_bound(inner, condition, **sizes)

    batch_rows = batches(sampler, len(condition))
    assert len(batch_rows) == inner.n_batches(0)
    assert all(idx.size == batch_size for idx in batch_rows), "every batch must be full"
    assert all(len(set(condition.codes[idx])) == 1 for idx in batch_rows), "every batch must be class-coherent"
    keys = [condition[idx[0]] for idx in batch_rows]
    twin = make_bound(make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), seed=7), condition, **sizes)
    assert keys == expected_labels(twin), "and must replay the inner's per-batch classes"


def test_chained_bound_replays_mid_schedule():
    def build_mid():
        return make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)

    outer = make_bound(build_mid(), CELL_TYPE, seed=2)
    assert outer.n_batches(0) == build_mid().n_batches(0)
    expected = expected_labels(make_bound(build_mid(), CELL_TYPE, seed=2))
    assert batch_keys(outer, CELL_TYPE, len(CELL_TYPE)) == expected, "a bound sampler is itself bindable"


def test_on_binds_subset_by_position():
    sampler = make_bound(make_inner(THREE_COLUMNS, num_samples=2000), TWO_COLUMNS, on=(0, 1))
    twin = make_bound(make_inner(THREE_COLUMNS, num_samples=2000), TWO_COLUMNS, on=(0, 1))
    expected = [label[:2] for label in expected_labels(twin)]
    assert batch_keys(sampler, TWO_COLUMNS, len(TWO_COLUMNS)) == expected


@pytest.mark.parametrize(
    ("on", "exc", "match"),
    [
        pytest.param((0, 1, 2), ValueError, "but classes_to_bind_on labels have 2", id="too_many_positions"),
        pytest.param((5,), ValueError, r"positions in \[0, 3\)", id="position_out_of_range"),
        pytest.param((), ValueError, "non-empty tuple", id="empty"),
        pytest.param(0, TypeError, "must be a tuple of label positions", id="bare_int"),
    ],
)
def test_invalid_on_is_rejected(on, exc, match):
    # every one of these used to surface as a subset error, a pandas AssertionError, or
    # `len()` on an int -- none of which named `on`
    with pytest.raises(exc, match=match):
        make_bound(make_inner(THREE_COLUMNS, num_samples=200), TWO_COLUMNS, on=on)


def test_within_class_weights_shares():
    sampler = make_bound(
        make_inner(["B"] * 200, num_samples=40_000), ["B"] * 200, classes=DONOR, class_weights=np.array([3.0, 1.0])
    )

    drawn = DONOR.codes[np.concatenate(batches(sampler, len(DONOR)))]
    shares = np.bincount(drawn, minlength=2) / drawn.size
    assert abs(shares[0] - 0.75) < 0.02 and abs(shares[1] - 0.25) < 0.02


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via):
    inner = make_inner(CELL_TYPE, num_samples=500, class_weights=np.array([1.0, 0.0]))
    if via == "constructor":
        sampler = make_bound(inner, CELL_TYPE, mask=slice(0, 100))
    else:
        sampler = make_bound(inner, CELL_TYPE)
        sampler.mask = slice(0, 100)
    chunks = [c for lr in sampler.sample(len(CELL_TYPE)) for c in lr["requests"]]
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"
