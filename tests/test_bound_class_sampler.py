"""Tests for BoundClassSampler.

The sampler replays an inner :class:`~annbatch.samplers.ClassSampler`'s per-batch class
schedule onto its own observations: every batch is class-coherent and full, the class of
each batch matches the inner sampler's corresponding batch (even across different category
orderings and obs lengths), an optional secondary class weights which rows are drawn, and
the whole thing is reproducible and picklable (both the inner and outer RNGs round-trip).
"""

from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import BoundClassSampler, ClassSampler
from annbatch.samplers._utils import WorkerInfo


def make_inner(
    labels,
    *,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    num_samples: int = 1000,
    drop_last: bool = True,
    seed: int = 0,
    **kwargs,
) -> ClassSampler:
    classes = labels if isinstance(labels, pd.Categorical) else pd.Categorical(labels)
    return ClassSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        classes=classes,
        num_samples=num_samples,
        drop_last=drop_last,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def make_bound(
    inner: ClassSampler,
    condition,
    *,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    seed: int = 1,
    **kwargs,
) -> BoundClassSampler:
    condition_classes = condition if isinstance(condition, pd.Categorical) else pd.Categorical(condition)
    return BoundClassSampler(
        inner,
        chunk_size,
        preload_nchunks,
        batch_size,
        condition_classes=condition_classes,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def batch_infos(sampler: BoundClassSampler, condition_codes: np.ndarray, n_obs: int) -> list[tuple[int, int, int]]:
    """For each yielded batch: ``(class_code, size, n_unique_classes)``."""
    infos = []
    for load_request in sampler.sample(n_obs):
        concat = np.concatenate([condition_codes[s.start : s.stop] for s in load_request["requests"]])
        for split in load_request["splits"]:
            unique = np.unique(concat[split])
            infos.append((int(unique[0]), int(split.size), int(unique.size)))
    return infos


def inner_batch_labels(inner: ClassSampler) -> list:
    """The class *label* of each batch a full pass of the inner sampler yields."""
    codes = np.asarray(inner.classes.codes)
    labels = []
    for load_request in inner.sample(codes.shape[0]):
        concat = np.concatenate([codes[s.start : s.stop] for s in load_request["requests"]])
        labels.extend(inner.classes.categories[np.unique(concat[split])[0]] for split in load_request["splits"])
    return labels


# =============================================================================
# Construction / validation
# =============================================================================


@pytest.mark.parametrize(
    ("kwargs", "condition", "extra", "error_type", "match"),
    [
        pytest.param(
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            np.repeat([0, 1], 100),
            {},
            ValueError,
            "batch_size must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param(
            {},
            np.repeat(["B", "X"], 100),
            {},
            ValueError,
            "absent from condition_classes",
            id="inner_class_absent",
        ),
        pytest.param(
            {},
            pd.Categorical.from_codes([-1, 0] * 100, categories=["B", "T"]),
            {},
            ValueError,
            "NA values",
            id="condition_na",
        ),
        pytest.param(
            {},
            np.repeat(["B", "T"], 100),
            {"classes": pd.Categorical(["d"] * 50)},
            ValueError,
            "same length as condition_classes",
            id="secondary_length_mismatch",
        ),
        pytest.param(
            {},
            np.repeat(["B", "T"], 100),
            {"class_weights": np.array([1.0])},
            ValueError,
            "class_weights was given but classes is None",
            id="weights_without_classes",
        ),
        pytest.param(
            {},
            (["B"] * 3 + ["T"] * 97) * 2,
            {},
            ValueError,
            "at least chunk_size",
            id="run_too_short",
        ),
    ],
)
def test_invalid_construction(kwargs, condition, extra, error_type, match):
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    with pytest.raises(error_type, match=match):
        make_bound(inner, condition, **kwargs, **extra)


def test_inner_must_be_class_sampler():
    with pytest.raises(TypeError, match="inner_sampler must be a ClassSampler"):
        make_bound("not a sampler", np.repeat(["B", "T"], 100))


def test_condition_must_be_categorical():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    with pytest.raises(TypeError, match="condition_classes must be a pandas.Categorical"):
        BoundClassSampler(inner, 10, 4, 10, condition_classes=np.repeat([0, 1], 100))


def test_validate_rejects_n_obs_mismatch():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        sampler.validate(n_obs=999)


def test_multiple_workers_not_supported():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    with (
        patch(
            "annbatch.samplers._class_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(200))


# =============================================================================
# Core behavior
# =============================================================================


def test_n_batches_matches_inner():
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), num_samples=1000)
    sampler = make_bound(inner, np.repeat(["B", "T", "NK", "Mono"], 50))
    assert sampler.n_batches(200) == inner.n_batches(0)


def test_shuffle_is_true():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    assert make_bound(inner, np.repeat(["B", "T"], 100)).shuffle is True


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(5, 10, 4, id="batch_two_chunks"),
        pytest.param(5, 20, 4, id="batch_four_chunks"),
    ],
)
def test_every_batch_is_full_and_coherent(chunk_size, batch_size, preload_nchunks):
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), num_samples=1000)
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    sampler = make_bound(
        inner, condition, chunk_size=chunk_size, batch_size=batch_size, preload_nchunks=preload_nchunks
    )
    infos = batch_infos(sampler, np.asarray(condition.codes), len(condition))
    assert len(infos) == inner.n_batches(0)
    assert all(n_unique == 1 for _, _, n_unique in infos), "every batch must be class-coherent"
    assert all(size == batch_size for _, size, _ in infos), "every batch must be full"


def test_replays_inner_per_batch_classes():
    # different category *orderings* and different obs *lengths* -> matched purely by label
    a_labels = pd.Categorical(np.repeat(["B", "T", "NK", "Mono"], 100))
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    inner_for_bound = make_inner(a_labels, num_samples=1000, seed=7)
    sampler = make_bound(inner_for_bound, condition)

    infos = batch_infos(sampler, np.asarray(condition.codes), len(condition))
    bound_labels = [condition.categories[code] for code, _, _ in infos]

    expected = inner_batch_labels(make_inner(a_labels, num_samples=1000, seed=7))
    assert bound_labels == expected


def test_reproducible_with_same_seeds():
    def build():
        inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600, seed=3)
        return make_bound(
            inner,
            np.repeat(
                ["NK", "T", "B"],
                60,
            ),
            seed=5,
        )

    codes = np.asarray(pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes)
    assert batch_infos(build(), codes, 180) == batch_infos(build(), codes, 180)


def test_two_passes_differ():
    # both rngs advance across sample() calls, like ClassSampler
    inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600)
    sampler = make_bound(inner, np.repeat(["NK", "T", "B"], 60))
    codes = np.asarray(pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes)
    first = batch_infos(sampler, codes, 180)
    second = batch_infos(sampler, codes, 180)
    assert first != second


# =============================================================================
# Positional binding (`on`)
# =============================================================================


def _projected_per_obs(categorical: pd.Categorical, position: int) -> np.ndarray:
    return np.array([label[position] for label in categorical.categories], dtype=object)[np.asarray(categorical.codes)]


def test_on_binds_on_tuple_position():
    # inner classes are (cell_type, donor); condition is (cell_type, batch); bind on cell_type (pos 0 both)
    inner_pairs = [("B", "d1")] * 40 + [("B", "d2")] * 40 + [("T", "d1")] * 40 + [("T", "d2")] * 40
    inner = make_inner(pd.Categorical(inner_pairs), num_samples=1000, seed=4)
    condition = pd.Categorical([("B", "x")] * 50 + [("T", "y")] * 50 + [("B", "y")] * 50 + [("T", "x")] * 50)
    sampler = make_bound(inner, condition, on={0: 0}, seed=6)

    cond_cell = _projected_per_obs(condition, 0)
    bound_cell = []
    for lr in sampler.sample(len(condition)):
        concat = cond_cell[np.concatenate([np.arange(s.start, s.stop) for s in lr["requests"]])]
        for split in lr["splits"]:
            unique = np.unique(concat[split])
            assert unique.size == 1, "each batch must be coherent on the bound cell_type"
            bound_cell.append(unique[0])

    expected = [lbl[0] for lbl in inner_batch_labels(make_inner(pd.Categorical(inner_pairs), num_samples=1000, seed=4))]
    assert bound_cell == expected


def test_on_maps_different_positions():
    # cell_type sits at inner position 0 but condition position 1 -> on={0: 1}
    inner_pairs = [("B", "d1")] * 40 + [("T", "d1")] * 40 + [("B", "d2")] * 40 + [("T", "d2")] * 40
    inner = make_inner(pd.Categorical(inner_pairs), num_samples=800, seed=1)
    condition = pd.Categorical([("x", "B")] * 50 + [("y", "T")] * 50 + [("y", "B")] * 50 + [("x", "T")] * 50)
    sampler = make_bound(inner, condition, on={0: 1}, seed=2)

    cond_cell = _projected_per_obs(condition, 1)
    for lr in sampler.sample(len(condition)):
        concat = cond_cell[np.concatenate([np.arange(s.start, s.stop) for s in lr["requests"]])]
        for split in lr["splits"]:
            assert np.unique(concat[split]).size == 1, "each batch must be coherent on the bound cell_type"
    assert sampler.n_batches(len(condition)) == inner.n_batches(0)


def test_on_pickle_roundtrip():
    def build():
        inner = make_inner(pd.Categorical([("B", "d1")] * 40 + [("T", "d2")] * 40), num_samples=400, seed=3)
        condition = pd.Categorical([("B", "x")] * 40 + [("T", "y")] * 40)
        return make_bound(inner, condition, on={0: 0}, seed=8)

    codes = np.asarray(pd.Categorical([("B", "x")] * 40 + [("T", "y")] * 40).codes)
    original = build()
    restored = pickle.loads(pickle.dumps(original))
    assert batch_infos(original, codes, 80) == batch_infos(restored, codes, 80)


def test_on_rejects_non_dict():
    inner = make_inner(pd.Categorical([("B", "d1")] * 40 + [("T", "d2")] * 40), num_samples=100)
    with pytest.raises(TypeError, match="on must be a dict"):
        make_bound(inner, pd.Categorical([("B", "x")] * 40 + [("T", "y")] * 40), on=(0, 0))


# =============================================================================
# Secondary (conditional) class
# =============================================================================


def test_secondary_class_weights_shares():
    # inner always emits "B"; within B, weight donors d1:d2 = 3:1
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=40_000)
    condition = pd.Categorical(["B"] * 400)
    donor = pd.Categorical((["d1"] * 20 + ["d2"] * 20) * 10)
    sampler = make_bound(inner, condition, classes=donor, class_weights=np.array([3.0, 1.0]))

    donor_codes = np.asarray(donor.codes)
    counts = np.zeros(2)
    for load_request in sampler.sample(len(condition)):
        for s in load_request["requests"]:
            for code in donor_codes[s.start : s.stop]:
                counts[int(code)] += 1
    shares = counts / counts.sum()
    assert abs(shares[0] - 0.75) < 0.02 and abs(shares[1] - 0.25) < 0.02


def test_secondary_zero_weight_excludes_and_exempts_run_length():
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=2000)
    condition = pd.Categorical(["B"] * 400)
    # d2 lives only in short (3-row) runs; excluding it with weight 0 must exempt those runs
    donor = pd.Categorical((["d1"] * 37 + ["d2"] * 3) * 10)
    sampler = make_bound(inner, condition, classes=donor, class_weights=np.array([1.0, 0.0]))

    donor_codes = np.asarray(donor.codes)
    drawn = {
        int(code)
        for load_request in sampler.sample(len(condition))
        for s in load_request["requests"]
        for code in donor_codes[s.start : s.stop]
    }
    assert drawn == {0}, "only the positive-weight secondary class should be drawn"

    # giving the short-run class a positive weight -> run-length rule now applies and fails
    with pytest.raises(ValueError, match="at least chunk_size"):
        make_bound(inner, condition, classes=donor, class_weights=np.array([1.0, 1.0]))


# =============================================================================
# Mask
# =============================================================================


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via):
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=500)
    condition = pd.Categorical(["B"] * 100 + ["T"] * 100)
    if via == "constructor":
        sampler = make_bound(inner, condition, mask=slice(0, 100))
    else:
        sampler = make_bound(inner, condition)
        sampler.mask = slice(0, 100)
    chunks = [c for lr in sampler.sample(len(condition)) for c in lr["requests"]]
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"


def test_mask_with_emittable_class_absent_raises():
    # inner emits B and T; masking to the T-only region leaves B with no drawable run,
    # which surfaces when the (B-emitting) inner schedule is replayed during sampling
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=200)
    condition = pd.Categorical(["B"] * 100 + ["T"] * 100)
    sampler = make_bound(inner, condition)
    sampler.mask = slice(100, 200)
    with pytest.raises(ValueError, match="no drawable run"):
        list(sampler.sample(len(condition)))


# =============================================================================
# Serialization / RNG
# =============================================================================


def test_pickle_roundtrip_continues_identically():
    def build():
        inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600, seed=2)
        return make_bound(inner, np.repeat(["NK", "T", "B"], 60), seed=9)

    codes = np.asarray(pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes)
    original = build()
    restored = pickle.loads(pickle.dumps(original))
    # both the inner and outer RNG states must survive the round-trip
    assert batch_infos(original, codes, 180) == batch_infos(restored, codes, 180)


def test_deepcopy_is_independent():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=200)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    clone = copy.deepcopy(sampler)
    codes = np.asarray(pd.Categorical(np.repeat(["B", "T"], 100)).codes)
    # a clone taken before consuming reproduces the same stream
    assert batch_infos(clone, codes, 200) == batch_infos(sampler, codes, 200)


# =============================================================================
# Loader integration
# =============================================================================


def test_bound_class_sampler_from_collection(simple_collection):
    from annbatch import Loader

    _, collection = simple_collection

    condition = collection.obs(columns=["src_path"])["src_path"].values
    categories = condition.categories
    # inner over a synthetic dataset A with the SAME category labels but a different length/order
    inner = ClassSampler(
        chunk_size=1,
        preload_nchunks=4,
        batch_size=4,
        classes=pd.Categorical(np.repeat(categories[::-1], 20), categories=categories),
        num_samples=100,
        drop_last=True,
        rng=np.random.default_rng(0),
    )
    sampler = BoundClassSampler(
        inner,
        1,
        4,
        4,
        condition_classes=condition,
        rng=np.random.default_rng(0),
    )

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None)
    loader.use_collection(collection)

    batches = list(loader)
    assert len(batches) == inner.n_batches(0)
    for batch in batches:
        assert batch["X"].shape == (4, 100)
        assert len(np.unique(batch["obs"]["src_path"])) == 1, "every batch must be class-coherent"
