"""Regression: class-coherent batches must stay pure when a batch's chunks span several datasets.

The loader fills its read buffer in *dataset* order, but ``splits`` are expressed in the sampler's
*request* order, so the request->buffer index map must be the INVERSE permutation (a scatter). A gather
(``inv = positions[order]``) is only correct when ``order`` is self-inverse -- e.g. a single dataset, or
a collection concatenated into one store -- and silently pulls rows from the wrong dataset once chunks
regroup across several separately-added datasets, yielding impure batches (scverse/annbatch#256).

This exercises the ``add_adatas`` path (several distinct datasets), which is what surfaces the bug; the
existing ``*_from_collection`` tests concatenate into a single store and so never regroup across datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import anndata as ad
from annbatch import Loader
from annbatch.samplers import BoundClassSampler, ClassSampler

N_CLASSES = 4
PER_CLASS_PER_DS = 20
N_DS = 3


def _adatas() -> list[ad.AnnData]:
    """One AnnData per dataset; each holds a contiguous run of every class, so a class's cells span
    all datasets. Column 0 of X carries the class code and column 1 the dataset id, so the rows the
    loader actually returns can be checked against the batch it placed them in."""
    out = []
    for d in range(N_DS):
        labels = np.repeat(np.arange(N_CLASSES), PER_CLASS_PER_DS)
        X = np.zeros((labels.size, 2), dtype="f4")
        X[:, 0] = labels
        X[:, 1] = d
        out.append(ad.AnnData(X=X, obs=pd.DataFrame({"label": labels.astype(str)})))
    return out


def _all_labels(adatas) -> pd.Categorical:
    return pd.Categorical(np.concatenate([a.obs["label"].to_numpy() for a in adatas]))


def test_class_sampler_multi_dataset_batches_stay_pure():
    adatas = _adatas()
    sampler = ClassSampler(
        chunk_size=5,
        preload_nchunks=8,
        batch_size=20,  # 4 chunks/batch; a class spans 3 datasets, so a coherent batch regroups across them
        classes=_all_labels(adatas),
        num_samples=20 * 60,
        drop_last=True,
        rng=np.random.default_rng(1),
    )
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None).add_adatas(adatas)

    saw_cross_dataset_batch = False
    for batch in loader:
        X = np.asarray(batch["X"])
        assert len(np.unique(X[:, 0])) == 1, "every batch must be class-coherent"
        # obs stays row-aligned with X
        assert np.array_equal(np.asarray(batch["obs"]["label"]).astype(float), X[:, 0])
        if len(np.unique(X[:, 1])) > 1:
            saw_cross_dataset_batch = True
    assert saw_cross_dataset_batch, "test must exercise batches whose chunks span more than one dataset"


def test_bound_class_sampler_multi_dataset_batches_stay_pure():
    """The same guarantee via a BoundClassSampler inner (the eval/inference read path)."""
    adatas = _adatas()
    labels = _all_labels(adatas)
    inner = ClassSampler(
        chunk_size=5, preload_nchunks=8, batch_size=20, classes=labels,
        num_samples=20 * 60, drop_last=True, rng=np.random.default_rng(0),
    )
    sampler = BoundClassSampler(
        inner, 5, 8, 20, classes_to_bind_on=labels, rng=np.random.default_rng(1),
    )
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None).add_adatas(adatas)

    saw_cross_dataset_batch = False
    for batch in loader:
        X = np.asarray(batch["X"])
        assert len(np.unique(X[:, 0])) == 1, "every bound batch must be class-coherent"
        if len(np.unique(X[:, 1])) > 1:
            saw_cross_dataset_batch = True
    assert saw_cross_dataset_batch, "test must exercise batches whose chunks span more than one dataset"
