from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from annbatch import Loader
from annbatch.abc import BaseClassSampler
from annbatch.samplers import BoundClassSampler, ClassSampler
from annbatch.samplers._utils import WorkerInfo, grouped_weighted_choice, project_index
from tests.conftest import load_x_obs_var, make_class_sampler

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


def batches(sampler: BaseClassSampler, n_obs: int) -> list[np.ndarray]:
    """The obs indices of every batch a full pass yields."""
    out = []
    for load_request in sampler.sample(n_obs):
        window = np.concatenate([np.arange(s.start, s.stop) for s in load_request["requests"]])
        out.extend(window[split] for split in load_request["splits"])
    return out


def batch_keys(sampler: BaseClassSampler, key_of_obs, n_obs: int) -> list:
    """The single key each batch is coherent on; asserts that coherence."""
    key = np.asarray(key_of_obs)
    keys = []
    for idx in batches(sampler, n_obs):
        unique = set(key[idx].tolist())
        assert len(unique) == 1, "each batch must be coherent on the bound key"
        keys.append(unique.pop())
    return keys


def expected_labels(sampler: BaseClassSampler, positions: tuple[int, ...] | None = None) -> list:
    """The per-batch class labels a standalone pass of ``sampler`` would schedule."""
    return list(project_index(sampler.vocab, positions)[sampler.batch_codes()])


def joint(*columns) -> pd.Categorical:
    """A flat multi-column categorical, one tuple label per row."""
    return pd.Categorical(pd.MultiIndex.from_arrays(columns).to_flat_index())


@pytest.mark.parametrize(
    ("kwargs", "classes_to_bind_on", "error_type", "match"),
    [
        pytest.param(
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            np.repeat([0, 1], 100),
            ValueError,
            "batch_size must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param({}, ["B"] * 200, ValueError, "absent from classes_to_bind_on", id="inner_class_absent"),
        pytest.param(
            {}, ["B"] * 100 + ["Z"] * 100, ValueError, "subset of the inner sampler's classes", id="not_subset_of_inner"
        ),
        pytest.param(
            {},
            pd.Categorical.from_codes([-1, 0] * 100, categories=["B", "T"]),
            ValueError,
            "NA values",
            id="classes_na",
        ),
        pytest.param({}, (["B"] * 3 + ["T"] * 97) * 2, ValueError, "at least chunk_size", id="run_too_short"),
        pytest.param({"on": [0, 1]}, CT, TypeError, "on must be a dict", id="on_not_dict"),
        pytest.param(
            {"class_weights": np.array([1.0, 1.0])},
            CT,
            ValueError,
            "class_weights was given but classes is None",
            id="class_weights_without_classes",
        ),
        pytest.param(
            {"classes": pd.Categorical(["d1", "d2"] * 5)},
            CT,
            ValueError,
            "classes must be the same length",
            id="classes_length_mismatch",
        ),
    ],
)
def test_invalid_construction(kwargs, classes_to_bind_on, error_type, match):
    with pytest.raises(error_type, match=match):
        make_bound(make_inner(CT, num_samples=100), classes_to_bind_on, **kwargs)


def test_classes_to_bind_on_must_be_categorical():
    with pytest.raises(TypeError, match="classes_to_bind_on must be a pandas.Categorical"):
        BoundClassSampler(make_inner(CT, num_samples=100), 10, 4, 10, classes_to_bind_on=np.repeat([0, 1], 100))


def test_validate_rejects_n_obs_mismatch():
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        make_bound(make_inner(CT, num_samples=100), CT).validate(n_obs=999)


def test_multiple_workers_not_supported():
    sampler = make_bound(make_inner(CT, num_samples=100), CT)
    with (
        patch(
            "annbatch.samplers._class_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(200))


def test_exposes_input_and_shuffles():
    bind = pd.Categorical(CT)
    sampler = make_bound(make_inner(CT, num_samples=100), bind)
    assert sampler.classes_to_bind_on is bind, "the property returns the caller's object, not a copy"
    assert sampler.shuffle is True


@pytest.mark.parametrize(
    ("inner_labels", "classes_to_bind_on"),
    [
        pytest.param(
            np.repeat(["B", "T", "NK", "Mono"], 100),
            np.repeat(["B", "T", "NK", "Mono"], 50),
            id="shorter_than_inner",
        ),
        pytest.param(CT, pd.Categorical(CT, categories=["B", "T", "Ghost"]), id="unused_category"),
    ],
)
def test_n_batches_matches_inner(inner_labels, classes_to_bind_on):
    inner = make_inner(inner_labels)
    assert make_bound(inner, classes_to_bind_on).n_batches(len(classes_to_bind_on)) == inner.n_batches(0)


def test_grouped_weighted_choice():
    group_of_item = np.array([0, 0, 1])
    weight_of_item = np.array([3.0, 1.0, 1.0])
    rng = np.random.default_rng(0)

    picks0 = grouped_weighted_choice(group_of_item, weight_of_item, np.zeros(4000, dtype=int), rng)
    assert set(picks0.tolist()) <= {0, 1}, "group-0 draws pick only group-0 items"
    assert abs((picks0 == 0).mean() - 0.75) < 0.03, "within group 0, items are weighted 3:1"

    picks1 = grouped_weighted_choice(group_of_item, weight_of_item, np.ones(10, dtype=int), rng)
    assert (picks1 == 2).all(), "group-1 draws pick its only item"


def test_project_index():
    mi = pd.MultiIndex.from_tuples([("c1", "b1"), ("c1", "b2"), ("c2", "b1")])
    assert isinstance(project_index(mi, None), pd.MultiIndex), "a MultiIndex is preserved, not flattened"
    obj = pd.Index([("c2", "b1"), ("c1", "b1")], tupleize_cols=False)
    assert project_index(mi, None).get_indexer(obj).tolist() == [2, 0], "MultiIndex <-> object-tuple cross-match"
    assert project_index(mi, (1,)).tolist() == ["b1", "b2", "b1"]

    idx = pd.Index(["B", "T", "NK"])
    assert project_index(idx, (0,)) is idx, "position 0 is the whole label -> returned unchanged"
    with pytest.raises(ValueError, match="only position 0"):
        project_index(idx, (1,))


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


@pytest.mark.parametrize(
    ("mid_kwargs", "inner_classes", "outer_bind", "outer_on", "expected_mid_vocab"),
    [
        pytest.param({}, CT, CT, None, None, id="plain"),
        pytest.param(
            {"classes": pd.Categorical(np.tile(np.repeat(["d1", "d2"], 50), 2)), "class_weights": np.ones(2)},
            CT,
            CT,
            {0: 0},
            {("B", "d1"), ("B", "d2"), ("T", "d1"), ("T", "d2")},
            id="coarsens_joint_onto_match",
        ),
        pytest.param(
            {"classes": pd.Categorical(np.tile(np.repeat(["b1", "b2"], 25), 4)), "class_weights": np.ones(2)},
            joint(np.repeat(["c1", "c2"], 100), np.tile(np.repeat(["dA", "dB"], 50), 2)),
            joint(
                np.repeat(["c1", "c2"], 100),
                np.tile(np.repeat(["dA", "dB"], 50), 2),
                np.tile(np.repeat(["b1", "b2"], 25), 4),
            ),
            {0: 0, 1: 1, 2: 2},
            {(c, d, b) for c in ("c1", "c2") for d in ("dA", "dB") for b in ("b1", "b2")},
            id="composes_columns",
        ),
        pytest.param(
            {
                "on": {0: 0},
                "classes": pd.Categorical(np.tile(np.repeat(["b1", "b2"], 50), 2)),
                "class_weights": np.ones(2),
            },
            np.repeat(["c1", "c2"], 100),
            joint(np.repeat(["c1", "c2"], 100), np.tile(np.repeat(["b1", "b2"], 50), 2)),
            None,
            {("c1", "b1"), ("c1", "b2"), ("c2", "b1"), ("c2", "b2")},
            id="matches_whole_multiindex_vocab",
        ),
    ],
)
def test_chained_bound_replays_mid_schedule(mid_kwargs, inner_classes, outer_bind, outer_on, expected_mid_vocab):
    def build_mid():
        return make_bound(make_inner(inner_classes, num_samples=200), inner_classes, **mid_kwargs)

    mid = build_mid()
    assert isinstance(mid, BaseClassSampler), "a bound sampler is itself bindable"
    if expected_mid_vocab is not None:
        assert set(mid.vocab) == expected_mid_vocab

    outer = make_bound(mid, outer_bind, on=outer_on, seed=2)
    assert outer.n_batches(0) == mid.n_batches(0)
    positions = tuple(outer_on) if outer_on else None
    assert batch_keys(outer, outer_bind, len(outer_bind)) == expected_labels(build_mid(), positions)


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


def test_within_class_zero_weight_excludes_and_exempts_run_length():
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=2000)
    classes_to_bind_on = pd.Categorical(["B"] * 400)
    donor = pd.Categorical((["d1"] * 37 + ["d2"] * 3) * 10)
    sampler = make_bound(inner, classes_to_bind_on, classes=donor, class_weights=np.array([1.0, 0.0]))

    drawn = set(donor.codes[np.concatenate(batches(sampler, len(donor)))].tolist())
    assert drawn == {0}, "only the positive-weight secondary class should be drawn"

    with pytest.raises(ValueError, match="at least chunk_size"):
        make_bound(inner, classes_to_bind_on, classes=donor, class_weights=np.array([1.0, 1.0]))


def test_covers_all_drawable_obs_and_respects_weights():
    cell_types, donors, sizes = zip(
        ("B", "d1", 30),
        ("B", "d2", 30),
        ("T", "d1", 30),
        ("T", "d3", 20),
        ("NK", "d2", 30),
        ("NK", "d3", 1),
        ("Mono", "d1", 30),
        ("B", "d1", 30),
        ("NK", "d2", 30),
        strict=True,
    )
    obs = pd.DataFrame({"cell_type": np.repeat(cell_types, sizes), "donor": np.repeat(donors, sizes)})
    donor = pd.Categorical(obs["donor"])
    donor_weights = pd.Series({"d1": 3.0, "d2": 1.0, "d3": 0.0})

    inner_cell = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 60))
    inner = make_inner(
        inner_cell,
        num_samples=60_000,
        class_weights=np.where(inner_cell.categories == "Mono", 0.0, 1.0),
    )
    sampler = make_bound(
        inner,
        pd.Categorical(obs["cell_type"]),
        chunk_size=3,
        batch_size=3,
        classes=donor,
        class_weights=donor_weights.reindex(donor.categories).to_numpy(),
    )

    drawn_idx = np.concatenate(batches(sampler, len(obs)))
    drawable = obs[obs["cell_type"].isin(["B", "T", "NK"]) & obs["donor"].map(donor_weights).gt(0)]
    assert set(drawn_idx.tolist()) == set(drawable.index)

    observed = obs.iloc[drawn_idx].groupby("cell_type")["donor"].value_counts(normalize=True)
    expected = drawable.drop_duplicates(["cell_type", "donor"]).copy()
    expected["w"] = expected["donor"].map(donor_weights)
    expected["share"] = expected["w"] / expected.groupby("cell_type")["w"].transform("sum")
    for row in expected.itertuples():
        assert abs(observed[row.cell_type, row.donor] - row.share) < 0.02, (
            f"{row.cell_type}/{row.donor}: {observed[row.cell_type, row.donor]:.3f} vs {row.share:.3f}"
        )


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


def test_mask_with_emittable_class_absent_raises():
    sampler = make_bound(make_inner(CT, num_samples=200), CT)
    sampler.mask = slice(100, 200)
    with pytest.raises(ValueError, match="no drawable run"):
        list(sampler.sample(200))


@pytest.mark.parametrize(
    "clone",
    [
        pytest.param(None, id="rebuilt_with_same_seeds"),
        pytest.param(lambda sampler: pickle.loads(pickle.dumps(sampler)), id="pickle"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_same_seeds_yield_the_same_pass(clone):
    labels, bind = np.repeat(["B", "T", "NK"], 100), np.repeat(["NK", "T", "B"], 60)

    def build():
        return make_bound(make_inner(labels, num_samples=600, seed=2), bind, seed=9)

    original = build()
    twin = build() if clone is None else clone(original)
    assert [idx.tolist() for idx in batches(twin, len(bind))] == [idx.tolist() for idx in batches(original, len(bind))]


def test_two_passes_differ():
    bind = np.repeat(["NK", "T", "B"], 60)
    sampler = make_bound(make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600), bind)
    first = [idx.tolist() for idx in batches(sampler, len(bind))]
    assert [idx.tolist() for idx in batches(sampler, len(bind))] != first


@pytest.mark.parametrize("bind", [False, True], ids=["class_sampler", "bound_class_sampler"])
def test_multi_dataset_batches_stay_pure(*, bind):
    """Chunks regrouped across datasets must not leak rows between class-coherent batches (#256)."""
    labels = np.repeat(np.arange(4), 20)
    adatas = [
        ad.AnnData(
            X=np.stack([labels, np.full(labels.size, d)], axis=1).astype("f4"),
            obs=pd.DataFrame({"label": labels.astype(str)}),
        )
        for d in range(3)
    ]
    classes = pd.Categorical(np.tile(labels.astype(str), len(adatas)))
    sizing = {"chunk_size": 5, "preload_nchunks": 8, "batch_size": 20}
    inner = make_inner(classes, num_samples=1200, seed=1, **sizing)
    sampler = make_bound(inner, classes, **sizing) if bind else inner
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None).add_adatas(adatas)

    datasets_per_batch = []
    for batch in loader:
        X = np.asarray(batch["X"])
        assert len(np.unique(X[:, 0])) == 1, "every batch must be class-coherent"
        assert np.array_equal(np.asarray(batch["obs"]["label"]).astype(float), X[:, 0]), "obs must track X row-for-row"
        datasets_per_batch.append(len(np.unique(X[:, 1])))
    assert max(datasets_per_batch) > 1, "test must exercise batches whose chunks span more than one dataset"


def test_from_collection(simple_collection):
    _, collection = simple_collection
    condition = collection.obs(columns=["src_path"])["src_path"].values
    sizing = {"chunk_size": 1, "preload_nchunks": 4, "batch_size": 4}
    inner = make_inner(
        pd.Categorical(np.repeat(condition.categories[::-1], 20), categories=condition.categories),
        num_samples=100,
        **sizing,
    )
    sampler = make_bound(inner, condition, seed=0, **sizing)

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None)
    loader.use_collection(collection, load_adata=load_x_obs_var)

    batch_list = list(loader)
    assert len(batch_list) == inner.n_batches(0)
    for batch in batch_list:
        assert batch["X"].shape == (4, 100)
        assert len(np.unique(batch["obs"]["src_path"])) == 1, "every batch must be class-coherent"
