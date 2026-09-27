from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial, singledispatchmethod
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, Self, cast
from warnings import warn

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from packaging.version import Version
from scipy import sparse as sp
from zarr import Array as ZarrArray

from annbatch.samplers import RandomSampler, SequentialSampler
from annbatch.types import BackingArray_T, LoaderOutput, OutputInMemoryArray_T
from annbatch.utils import (
    CSRContainer,
    as_runs,
    check_lt_1,
    check_var_shapes,
    convert,
    load_all_aligned,
    ramp,
    validate_sampler,
    warn_ignored_obs_aligned,
)

from .compat import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from annbatch.abc import Sampler
    from annbatch.io import DatasetCollection

    # TODO: remove after sphinx 9 - myst compat
    BackingArray = BackingArray_T
    OutputInMemoryArray = OutputInMemoryArray_T

zarr_version = Version(version("zarr"))


_FETCH_POOL: ThreadPoolExecutor | None = None


def _fetch_pool() -> ThreadPoolExecutor:
    """The pool the synchronous fetch runs on. A WIDTH, so a knob rather than an arm.

    `ANNBATCH_FETCH_THREADS` sets it. The default is `min(32, cpus + 4)`, which is what
    `asyncio`'s own default executor would have given, so leaving it unset keeps the
    concurrency the gather-based version had.

    Sized once, on first use. Resizing mid-run would put two widths inside one measurement.
    """
    global _FETCH_POOL
    if _FETCH_POOL is None:
        default = min(32, (os.cpu_count() or 1) + 4)
        width = int(os.environ.get("ANNBATCH_FETCH_THREADS", default))
        _FETCH_POOL = ThreadPoolExecutor(max_workers=width, thread_name_prefix="annbatch-fetch")
        print(f"[annbatch] synchronous fetch pool: {width} threads", flush=True)
    return _FETCH_POOL


class _Runs:
    """Runs of rows in one dataset, ``starts[i] : starts[i] + lengths[i]``, in buffer order.

    What a loader's chunks are, and what a range read takes: its rows exist only for the
    callers that need one entry per row (obs, indices, an in-memory source), built once.
    """

    def __init__(self, starts: np.ndarray, lengths: np.ndarray) -> None:
        self.starts, self.lengths = starts, lengths
        self._rows: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.lengths.sum())

    @property
    def rows(self) -> np.ndarray:
        if self._rows is None:
            self._rows = ramp(self.starts, self.lengths)
        return self._rows

    def nnz(self, indptr: np.ndarray) -> int:
        """Non-zeros in the runs, off a CSR's `indptr`: one lookup per run, not per row."""
        return int((indptr[self.starts + self.lengths] - indptr[self.starts]).sum())


def _csr_parts(dataset: BackingArray_T) -> tuple[np.ndarray, np.dtype, np.dtype]:
    """``(indptr, data dtype, indices dtype)`` for a backed or in-memory CSR.

    Sizing the output buffer needs all three before any read happens, and a backed
    dataset answers without IO: its `indptr` is held in memory and the dtypes come off
    the group metadata.
    """
    if isinstance(dataset, ad.abc.CSRDataset):
        return dataset.indptr, dataset.dtype, dataset.indices_dtype
    return dataset.indptr, dataset.data.dtype, dataset.indices.dtype


if find_spec("numba"):
    import numba

    @numba.njit(parallel=True, cache=True, nogil=True)
    def _csr_subset_rows(src_data, src_indices, src_indptr, rows, out_data, out_indices):  # type: ignore
        n_rows = rows.shape[0]
        row_nnz = np.empty(n_rows, dtype=np.int64)
        for i in range(n_rows):
            r = rows[i]
            row_nnz[i] = src_indptr[r + 1] - src_indptr[r]
        out_offsets = np.empty(n_rows + 1, dtype=np.int64)
        out_offsets[0] = 0
        for i in range(n_rows):
            out_offsets[i + 1] = out_offsets[i] + row_nnz[i]
        for i in numba.prange(n_rows):
            r = rows[i]
            src_start = src_indptr[r]
            dst_start = out_offsets[i]
            n = row_nnz[i]
            for j in range(n):
                out_data[dst_start + j] = src_data[src_start + j]
                out_indices[dst_start + j] = src_indices[src_start + j]
else:  # pragma: no cover

    def _csr_subset_rows(src_data, src_indices, src_indptr, rows, out_data, out_indices):
        raise ImportError("numba must be installed for in-memory sparse data: `pip install annbatch[numba]`")


def _cupy_dtype(dtype: np.dtype) -> np.dtype:
    if dtype in {np.dtype("float32"), np.dtype("float64"), np.dtype("bool")}:
        return dtype
    if dtype.itemsize < 4:
        return np.dtype("float32")
    return np.dtype("float64")


class Loader[
    BackingArray: BackingArray_T,
    OutputInMemoryArray: OutputInMemoryArray_T,
](IterableDataset):
    """A loader for on-disk data anndata stores.

    This loader by default batches together slice requests (`chunk_size` parameter) to the underlying stores to achieve higher performance.
    You can also use `chunk_size==1` for perfect random sampling (for relevant samplers), although this comes at a performance penalty for on-disk (and likely in-memory) data as well.
    Custom samplers are supported via the `batch_sampler` argument.
    We thus recommend using :class:`~annbatch.DatasetCollection` to preshuffle your data (or pre-shuffling in-memory).
    The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

    When providing a custom sampler, `chunk_size`, `preload_nchunks`, `batch_size`,
    `shuffle`, `drop_last`, and `rng` must not be set (they are controlled by the `batch_sampler` instead).
    When providing these arguments and no `batch_sampler`, they are used to construct a :class:`~annbatch.samplers.RandomSampler` (if ``shuffle=True``) or :class:`~annbatch.samplers.SequentialSampler`.

    Parameters
    ----------
        batch_sampler
            If not provided, a default :class:`~annbatch.samplers.SequentialSampler` or :class:`~annbatch.samplers.RandomSampler` will be used with the same defaults below.
        chunk_size
            The obs size (i.e., axis 0) of contiguous array data to fetch. Mutually exclusive with `batch_sampler`. Defaults to 512.
        preload_nchunks
            The number of chunks of contiguous array data to fetch. Mutually exclusive with `batch_sampler`. Defaults to 32.
        shuffle
            Whether or not to shuffle the data. Mutually exclusive with `batch_sampler`. Defaults to False.
        batch_size
            Batch size to yield from the dataset. Mutually exclusive with `batch_sampler`. Defaults to 1.
        drop_last
            Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            Leave as False when using in conjunction with a :class:`torch.utils.data.DataLoader`.
            Mutually exclusive with `batch_sampler`. Defaults to False.
        rng
            Random number generator for shuffling. Mutually exclusive with `batch_sampler`. Defaults to `np.random.default_rng()` if not provided.
        return_index
            Whether or not to yield the index on each iteration.
        preload_to_gpu
            Whether or not to use cupy for non-io array operations like vstack and indexing once the data is in memory internally (and batches yielded if `to` is not set).
            This option entails greater GPU memory usage, but is faster at least for sparse operations.
            :func:`torch.vstack` does not support CSR sparse matrices, hence the current use of `cupy` internally (which also means `torch` is an optional dep).
            Furthermore, there is no way to allocate pinned memory for jax arrays.
            Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data due to memory pressure.
            For top performance, this should be used in conjunction with `to="torch"` and then :meth:`torch.Tensor.to_dense` if you wish to densify (or similar for jax, although jax may be very slow for sparse->dense transform).
            :meth:`cupy.cuda.MemoryPool.free_all_blocks` (i.e., the method of the pool of :func:`cupy.get_default_memory_pool()`) is called aggressively to keep memory usage low.
            If you are using your own memory pool or allocator, you may have to free blocks on your own.
        to
            The output library for which you would like your array output.


    Examples
    --------
        >>> from annbatch import Loader
        >>> ds = Loader(
                batch_size=4096,
                chunk_size=32,
                preload_nchunks=512,
            ).add_adata(my_anndata)
        >>> for batch in ds:
                # optionally convert to dense
                # batch = batch.to_dense()
                do_fit(batch)
    """

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 512,
        "preload_nchunks": 32,
        "batch_size": 1,
        "drop_last": False,
    }
    # TODO(selmanozleyen): these should be also presented in the documentation
    # but this is not ideal since they are hardcoded into the docstrings
    # maybe we should make _COMMON_SAMPLER_ARGS a public class field?

    _train_datasets: list[BackingArray]
    _obs: list[pd.DataFrame] | None = None
    _var: pd.DataFrame | None = None
    _return_index: bool = False
    _shapes: list[tuple[int, int]]
    _preload_to_gpu: bool = True
    _to: Literal["torch", "jax"] | None = None
    _batch_sampler: Sampler
    _collection_added: bool = False
    _dtypes_homogeneous: bool = True

    def __init__(
        self,
        *,
        batch_sampler: Sampler | None = None,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        shuffle: bool | None = None,
        return_index: bool = False,
        batch_size: int | None = None,
        preload_to_gpu: bool = find_spec("cupy") is not None,
        drop_last: bool | None = None,
        to: Literal["torch", "jax"] | None = None,
        rng: np.random.Generator | None = None,
    ):
        # args that are passed after resolving defaults
        core_sampler_args = {
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks,
            "batch_size": batch_size,
            "drop_last": drop_last,
        }
        sampler_args = {**core_sampler_args, "rng": rng, "shuffle": shuffle}
        if batch_sampler is not None:
            if any(v is not None for v in sampler_args.values()):
                provided_args = [name for name, val in sampler_args.items() if val is not None]
                raise ValueError(
                    f"Cannot specify {', '.join(provided_args)} when providing a custom sampler. "
                    "These parameters are controlled by the sampler."
                )
            self._batch_sampler = batch_sampler
        else:
            resolved_core_args = {
                k: Loader._COMMON_SAMPLER_ARGS[k] if v is None else v for k, v in core_sampler_args.items()
            }
            if shuffle is not None and shuffle:
                self._batch_sampler = RandomSampler(
                    **resolved_core_args,
                    rng=rng if rng is not None else np.random.default_rng(),
                )
            else:
                self._batch_sampler = SequentialSampler(**resolved_core_args)
        for pkg in ["torch", "jax"]:
            if to == pkg and not find_spec(pkg):
                raise ImportError(f"Could not find {pkg} dependency. Try `pip install {pkg}`.")
        if preload_to_gpu and not find_spec("cupy"):
            raise ImportError(
                "Could not find cupy dependency. Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy."
            )

        self._return_index = return_index
        self._preload_to_gpu = preload_to_gpu
        self._to = to
        self._train_datasets = []
        self._shapes = []

    def __len__(self) -> int:
        return self._batch_sampler.n_batches(self.n_obs)

    @property
    def _sp_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupyx.scipy.sparse as cpx  # pragma: no cover

                return cpx
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None
        return sp

    @property
    def _np_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupy as cp

                return cp
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None

        return np

    @property
    def dataset_type(self) -> type[BackingArray]:
        """The type of on-disk data used in this loader.

        Returns
        -------
            The type used.
        """
        return type(self._train_datasets[0])

    @property
    def n_obs(self) -> int:
        """The total number of observations in this instance i.e., the sum of the first axis of all added datasets.

        Returns
        -------
            The number of observations.
        """
        return sum(shape[0] for shape in self._shapes)

    @property
    def n_var(self) -> int:
        """The total number of variables in this instance i.e., the second axis (which is the same) across all datasets.

        Returns
        -------
            The number of variables.
        """
        if len(self._shapes) == 0:
            raise ValueError("No datasets added yet")
        return self._shapes[0][1]

    @property
    def var(self) -> pd.DataFrame | None:
        """The var annotations for the variables in this loader.

        Returns
        -------
            The var DataFrame or None if no var annotations were provided.
        """
        return self._var

    @property
    def batch_sampler(self) -> Sampler:
        """The sampler used to generate batches.

        Returns
        -------
            The sampler.
        """
        return self._batch_sampler

    def use_collection(
        self,
        collection: DatasetCollection,
        *,
        load_adata: Callable[[zarr.Group], ad.AnnData] = load_all_aligned,
    ) -> Self:
        """Load from an existing :class:`annbatch.DatasetCollection`.

        This function can only be called once. If you want to manually add more data, use :meth:`Loader.add_adatas` or open an issue.

        Parameters
        ----------
        collection
            The collection whose on-disk datasets should be used in this loader.
        load_adata
            A custom load function - recall that whatever is found in :attr:`~anndata.AnnData.X` and :attr:`~anndata.AnnData.obs` will be yielded in batches.
            Only `X`, `obs`, and `var` are yielded for now; a future release will additionally yield observation-aligned
            :attr:`~anndata.AnnData.obsm` and :attr:`~anndata.AnnData.layers` elements.
        """
        if collection.is_empty:
            raise ValueError("DatasetCollection is empty")
        if self._collection_added:
            raise RuntimeError(
                "You should not add multiple collections, independently shuffled - please preshuffle multiple collections, use `add_adatas` manually if you know what you are doing, or open an issue if you believe that this should be supported at an API level higher than `add_adatas`."
            )
        adatas = [load_adata(g) for g in collection]
        self.add_adatas(adatas)
        self._collection_added = True
        return self

    @validate_sampler
    def add_adatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append adatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array`, :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray`, or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
                Only `X`, `obs`, and `var` are kept for now: any :attr:`~anndata.AnnData.obsm` and :attr:`~anndata.AnnData.layers` elements are ignored and a :class:`FutureWarning` is emitted (a future release will additionally load and yield them).
        """
        check_lt_1([len(adatas)], ["Number of adatas"])
        for adata in adatas:
            self._add_adata_unchecked(adata)
        return self

    @validate_sampler
    def add_adata(self, adata: ad.AnnData) -> Self:
        """Append an adata to this dataset.

        Parameters
        ----------
            adata
                A :class:`anndata.AnnData` object, with :class:`zarr.Array`, :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray`, or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
                :attr:`~anndata.AnnData.var` must match the ``var`` of any previously added datasets.
                Only `X`, `obs`, and `var` are kept for now: any :attr:`~anndata.AnnData.obsm` and :attr:`~anndata.AnnData.layers` elements are ignored and a :class:`FutureWarning` is emitted (a future release will additionally load and yield them if present in the passed in `adata`).
        """
        self._add_adata_unchecked(adata)
        return self

    def _add_adata_unchecked(self, adata: ad.AnnData) -> Self:
        # TODO(obsm): drop this call - and `warn_ignored_obs_aligned` - once these elements are yielded
        warn_ignored_obs_aligned(adata, stacklevel=3)
        dataset, obs, var = self._prepare_dataset_obs_and_var(adata)
        self._add_dataset_unchecked(dataset, obs, var)
        return self

    def _prepare_dataset_obs_and_var(
        self, adata: ad.AnnData
    ) -> tuple[BackingArray, pd.DataFrame | None, pd.DataFrame | None]:
        dataset = adata.X
        obs = adata.obs
        var = adata.var
        if len(obs.columns) == 0:
            obs = None
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Found {type(dataset)} but only {BackingArray_T.__value__} are usable")

        return cast("BackingArray", dataset), obs, var

    @validate_sampler
    def add_datasets(
        self,
        datasets: list[BackingArray],
        obs: list[pd.DataFrame] | None = None,
        var: list[pd.DataFrame] | None = None,
    ) -> Self:
        """Append datasets to this dataset.

        Parameters
        ----------
            datasets
                List of :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` objects, generally from :attr:`anndata.AnnData.X`.
                They must all be of the same type and match that of any already added datasets.
            obs
                List of :class:`~pandas.DataFrame` for annotating observations (i.e., samples), generally from :attr:`anndata.AnnData.obs`.
            var
                List of :class:`~pandas.DataFrame` for annotating features, generally from :attr:`anndata.AnnData.var`.
                All var DataFrames must be identical.
        """
        if obs is None:
            obs = [None] * len(datasets)
        if var is None:
            var = [None] * len(datasets)
        for ds, o, v in zip(datasets, obs, var, strict=True):
            self._add_dataset_unchecked(ds, o, v)
        return self

    @validate_sampler
    def add_dataset(
        self,
        dataset: BackingArray,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
    ) -> Self:
        """Append a dataset to this dataset.

        Parameters
        ----------
            dataset
                A :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` object, generally from :attr:`anndata.AnnData.X`.
            obs
                :class:`~pandas.DataFrame` obs, generally from :attr:`anndata.AnnData.obs`.
            var
                :class:`~pandas.DataFrame` var, generally from :attr:`anndata.AnnData.var`.
                :attr:`~anndata.AnnData.var` must match the ``var`` of any previously added datasets.
        """
        self._add_dataset_unchecked(dataset, obs, var)
        return self

    def _add_dataset_unchecked(
        self,
        dataset: BackingArray,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
    ) -> Self:
        if len(self._train_datasets) > 0:
            if self._obs is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without obs"
                )
            if self._obs is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without obs"
                )
            if self._var is None and var is not None:
                raise ValueError(
                    "Cannot add a dataset with var when training datasets have already been added without var"
                )
            if self._var is not None and var is None:
                raise ValueError(
                    "Cannot add a dataset without var when training datasets have already been added with var"
                )
            if not isinstance(dataset, self.dataset_type):
                raise ValueError(
                    f"All datasets on a given loader must be of the same type {self.dataset_type} but got {type(dataset)}"
                )
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Cannot add dataset of type {type(dataset)}")
        if isinstance(dataset, ad.abc.CSRDataset) and not dataset.backend == "zarr":
            raise TypeError(
                "Cannot add CSRDataset backed by h5ad at the moment: see https://github.com/zarr-developers/VirtualiZarr/pull/790"
            )
        if isinstance(dataset, sp.csr_matrix | sp.csr_array) and not find_spec("numba"):
            raise ImportError("numba must be installed for in-memory sparse data: `pip install annbatch[numba]`")
        if not isinstance(obs, pd.DataFrame) and obs is not None:
            raise TypeError("obs must be a pandas DataFrame")
        if not isinstance(var, pd.DataFrame) and var is not None:
            raise TypeError("var must be a pandas DataFrame")
        datasets = self._train_datasets + [dataset]
        check_var_shapes(datasets)
        self._dtypes_homogeneous = self._datasets_share_dtype(datasets)
        if self._train_datasets and not self._dtypes_homogeneous:
            warn(
                f"Adding dataset with dtype {dataset.dtype!r} that differs from the existing dataset dtype(s) "
                f"(first dataset: {self._train_datasets[0].dtype!r}). Heterogeneous dtypes incur extra per-batch "
                "allocation and dtype promotion in the loader; consider casting all datasets to a common dtype.",
                stacklevel=2,
            )
        if isinstance(dataset, ad.abc.CSRDataset):
            # Sizing a batch reads all three of these, and a backed dataset resolves
            # them lazily through synchronous store calls. Batches run on zarr's event
            # loop, which refuses those. Registration is the loader's one synchronous
            # phase -- it already resolves `shape` and `backend` above -- so the whole
            # set is resolved here, once, rather than warmed per batch.
            _ = (dataset.indptr, dataset.dtype, dataset.indices_dtype)
        self._shapes = self._shapes + [dataset.shape]
        self._train_datasets = datasets
        if self._obs is not None:  # obs exist
            self._obs += [obs]
        elif obs is not None:  # obs dont exist yet, but are being added for the first time
            self._obs = [obs]
        # var is the same across all datasets (describes variables/features)
        if self._var is None and var is not None:
            self._var = var
        elif self._var is not None and var is not None and not self._var.equals(var):
            raise ValueError(
                "All datasets must have identical var DataFrames. "
                "The var of the new dataset does not match the existing var."
            )
        return self

    def _requests_to_dataset_rows(
        self, requests: list[slice] | np.ndarray
    ) -> tuple[OrderedDict[int, _Runs], np.ndarray]:
        """Given a ndarray or list of slices, the runs of rows each on-disk dataset is asked for.

        Parameters
        ----------
            requests
                Runs of rows, slices, or an array of row indices, relative to the on-disk datasets.

        Returns
        -------
            A lookup between the dataset and its runs of rows, ordered by keys, and the
            permutation ``order`` mapping each in-memory buffer position to its index in the
            original request order (the buffer is filled in dataset order, so ``order`` is what
            undoes that reordering).
        """
        if isinstance(requests, np.ndarray) and requests.ndim == 1:
            run_starts = requests.astype(np.int64, copy=False)
            run_lengths = np.ones_like(run_starts)
        else:
            runs = as_runs(requests)
            run_starts, run_lengths = runs[:, 0], runs[:, 1] - runs[:, 0]
        keep = run_lengths > 0
        run_starts, run_lengths = run_starts[keep], run_lengths[keep]
        # Where each run's rows land in request order.
        request_at = np.cumsum(run_lengths) - run_lengths

        sizes = np.fromiter((shape[0] for shape in self._shapes), dtype=np.int64, count=len(self._shapes))
        ends = np.cumsum(sizes)
        starts = ends - sizes
        dataset_of_run = np.searchsorted(ends, run_starts, side="right")
        # A run crossing a dataset boundary is cut there. Rare, so the cut is a loop over
        # just those runs.
        last = np.searchsorted(ends, run_starts + run_lengths - 1, side="right")
        if (cross := np.flatnonzero(last != dataset_of_run)).size:
            pieces = [(run_starts, run_lengths, request_at, dataset_of_run)]
            parts = []
            for i in cross:
                at, left, where = run_starts[i], run_lengths[i], request_at[i]
                for ds in range(dataset_of_run[i], last[i] + 1):
                    n = min(left, int(ends[ds]) - at)
                    parts.append((at, n, where, ds))
                    at, left, where = at + n, left - n, where + n
            whole = np.ones(run_starts.size, dtype=bool)
            whole[cross] = False
            cut = np.array(parts, dtype=np.int64).T
            run_starts, run_lengths, request_at, dataset_of_run = (
                np.concatenate([p[whole], c]) for p, c in zip(pieces[0], cut, strict=True)
            )

        # Runs grouped by dataset and ascending within it -- a sort of RUNS, not rows. Ascending
        # keeps runs that touch next to each other, so a range read merges them, and keeps rows
        # of one inner chunk together, so it is read and decoded once for all of them.
        run_order = np.lexsort((run_starts, dataset_of_run))
        grouped = dataset_of_run[run_order]
        group_start = np.concatenate([[0], np.flatnonzero(np.diff(grouped)) + 1])
        group_end = np.append(group_start[1:], grouped.size)

        result: OrderedDict[int, _Runs] = OrderedDict()
        for gs, ge in zip(group_start, group_end, strict=True):
            ds = int(grouped[gs])
            picked = run_order[gs:ge]
            result[ds] = _Runs(run_starts[picked] - starts[ds], run_lengths[picked])
        order = ramp(request_at[run_order], run_lengths[run_order])
        return result, order

    def _alloc(self, shape: tuple[int, ...], dtype: np.dtype, *, use_pinned: bool) -> np.ndarray:
        if use_pinned:
            import cupyx as cpx

            return cpx.empty_pinned(shape, dtype)
        return np.empty(shape, dtype)

    def _allocate_out(self, dataset_index_to_rows: OrderedDict[int, _Runs]) -> CSRContainer | np.ndarray:
        """Preallocate a single contiguous output buffer covering all datasets and rows.

        For sparse data the buffer is a :class:`~annbatch.utils.CSRContainer` whose ``data``
        and ``indices`` arrays span the total number of non-zeros (derived from the cached
        ``indptr``) and whose ``indptr`` array spans the total number of rows + 1.
        For dense data it is a plain :class:`numpy.ndarray` of shape
        ``(total_rows, n_var)``.

        """
        total_rows = sum(len(rows) for rows in dataset_index_to_rows.values())

        if issubclass(self.dataset_type, ad.abc.CSRDataset | sp.csr_array | sp.csr_matrix):
            parts = {idx: _csr_parts(self._train_datasets[idx]) for idx in dataset_index_to_rows}
            total_nnz = sum(runs.nnz(parts[idx][0]) for idx, runs in dataset_index_to_rows.items())
            first_indptr, data_dtype, indices_dtype = parts[next(iter(dataset_index_to_rows))]
            indptr_dtype = first_indptr.dtype
            return CSRContainer(
                elems=(
                    self._alloc((total_nnz,), data_dtype, use_pinned=self._preload_to_gpu),
                    self._alloc((total_nnz,), indices_dtype, use_pinned=self._preload_to_gpu),
                    np.empty(total_rows + 1, dtype=indptr_dtype),
                ),
                shape=(total_rows, self.n_var),
                dtype=data_dtype,
            )
        else:
            first_idx = next(iter(dataset_index_to_rows))
            dtype = self._train_datasets[first_idx].dtype
            shape_res = self._train_datasets[first_idx].shape[1:]
            return self._alloc((total_rows, *shape_res), dtype, use_pinned=self._preload_to_gpu)

    @staticmethod
    def _datasets_share_dtype(datasets: list[BackingArray]) -> bool:
        """Whether all given dataset-like objects share the same dtype(s)."""
        if len(datasets) <= 1:
            return True

        def dtypes_of(d):
            if isinstance(d, ad.abc.CSRDataset):
                return (d.dtype, d.indices_dtype)
            if hasattr(d, "data") and hasattr(d, "indices"):
                return (d.data.dtype, d.indices.dtype)
            return (d.dtype,)

        first = dtypes_of(datasets[0])
        return all(dtypes_of(d) == first for d in datasets[1:])

    def _allocate_per_dataset_outs(
        self, dataset_index_to_rows: OrderedDict[int, _Runs]
    ) -> OrderedDict[int, CSRContainer | np.ndarray]:
        """Allocate one output buffer per dataset, each using that dataset's native dtype(s).

        Used when datasets have differing dtypes — the per-dataset buffers are concatenated
        into a final buffer of the promoted dtype by :meth:`_concatenate_outs`.
        """
        is_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset | sp.csr_array | sp.csr_matrix)
        outs: OrderedDict[int, CSRContainer | np.ndarray] = OrderedDict()
        if is_sparse:
            for idx, rows in dataset_index_to_rows.items():
                indptr, data_dtype, indices_dtype = _csr_parts(self._train_datasets[idx])
                nnz = rows.nnz(indptr)
                outs[idx] = CSRContainer(
                    elems=(
                        self._alloc((nnz,), data_dtype, use_pinned=False),
                        self._alloc((nnz,), indices_dtype, use_pinned=False),
                        self._alloc((len(rows) + 1,), np.min_scalar_type(nnz), use_pinned=False),
                    ),
                    shape=(len(rows), self.n_var),
                    dtype=data_dtype,
                )
        else:
            for idx, rows in dataset_index_to_rows.items():
                ds = self._train_datasets[idx]
                outs[idx] = self._alloc((len(rows), *ds.shape[1:]), ds.dtype, use_pinned=False)
        return outs

    def _concatenate_outs(self, outs: OrderedDict[int, CSRContainer | np.ndarray]) -> CSRContainer | np.ndarray:
        """Concatenate per-dataset buffers into a single buffer with promoted dtype(s)."""
        values = list(outs.values())
        if isinstance(values[0], CSRContainer):
            data_dtype = np.result_type(*[o.elems[0].dtype for o in values])
            indices_dtype = np.result_type(*[o.elems[1].dtype for o in values])
            total_nnz = sum(o.elems[0].size for o in values)
            total_rows = sum(o.shape[0] for o in values)
            data = self._alloc((total_nnz,), data_dtype, use_pinned=self._preload_to_gpu)
            indices = self._alloc((total_nnz,), indices_dtype, use_pinned=self._preload_to_gpu)
            indptr = self._alloc((total_rows + 1,), np.min_scalar_type(total_nnz), use_pinned=self._preload_to_gpu)
            indptr[0] = 0
            nnz_offset = 0
            row_offset = 0
            for o in values:
                n = o.elems[0].size
                r = o.shape[0]
                data[nnz_offset : nnz_offset + n] = o.elems[0]
                indices[nnz_offset : nnz_offset + n] = o.elems[1]
                indptr[row_offset + 1 : row_offset + r + 1] = o.elems[2][1:] + nnz_offset
                nnz_offset += n
                row_offset += r
            return CSRContainer(
                elems=(data, indices, indptr),
                shape=(total_rows, self.n_var),
                dtype=data_dtype,
            )
        dtype = np.result_type(*[o.dtype for o in values])
        total_rows = sum(o.shape[0] for o in values)
        out = self._alloc((total_rows, *values[0].shape[1:]), dtype, use_pinned=self._preload_to_gpu)
        offset = 0
        for o in values:
            out[offset : offset + o.shape[0]] = o
            offset += o.shape[0]
        return out

    @singledispatchmethod
    def _fetch_data(
        self,
        dataset: ZarrArray | ad.abc.CSRDataset,
        rows: np.ndarray,
        out: CSRContainer | np.ndarray,
    ) -> None:
        """Fetch data from an on-disk store into a preallocated buffer.

        Parameters
        ----------
        dataset
            The underlying store.
        rows
            Array of integer row indices within this dataset to fetch.
        out
            Preallocated buffer to write into — a contiguous view of the full
            output buffer allocated by :meth:`_allocate_out`.

        Raises
        ------
        NotImplementedError
            If the dataset type is not recognised.
        """
        raise NotImplementedError(f"Cannot fetch data for type {type(dataset)}")

    @_fetch_data.register
    def _fetch_data_dense(self, dataset: ZarrArray, rows: _Runs, out: np.ndarray) -> None:
        """The runs, handed to the codec pipeline whole.

        A zarr with a range selection takes them as runs, and a pipeline that implements its
        hook reads them without an indexer. Older zarr gets one orthogonal selection of the
        rows, which it groups by chunk itself: a range read built from `BasicIndexer`s, one per
        run, measured up to 2.3x slower than that on a fragmented draw.
        """
        prototype = zarr.core.buffer.default_buffer_prototype()
        if hasattr(dataset, "get_range_selection"):
            dataset.get_range_selection(rows.starts, rows.lengths, out=prototype.nd_buffer(out))
            return
        dataset.get_orthogonal_selection((rows.rows, slice(None)), out=prototype.nd_buffer(out))

    @_fetch_data.register
    def _fetch_data_numpy_matrix(
        self,
        dataset: np.ndarray,
        rows: _Runs,
        out: np.ndarray,
    ) -> None:
        out[:] = dataset[rows.rows]

    @_fetch_data.register
    def _fetch_data_csr_matrix(
        self,
        dataset: sp.csr_matrix | sp.csr_array,
        rows: _Runs,
        out: CSRContainer,
    ) -> None:
        _csr_subset_rows(
            dataset.data,
            dataset.indices,
            dataset.indptr,
            np.ascontiguousarray(rows.rows),
            out.elems[0],
            out.elems[1],
        )

    @_fetch_data.register
    def _fetch_data_sparse(
        self,
        dataset: ad.abc.CSRDataset,
        rows: _Runs,
        out: CSRContainer,
    ) -> None:
        """Read the rows through anndata's synchronous interface. Nothing else.

        `read_rows(out=)` -- no async entry point, no zarr privates. It is safe because
        NOTHING here runs on zarr's event loop any more: `_index_datasets` is an ordinary
        method and its fetches run on pool workers, so a worker is free to block on the
        sync bridge. That was the one thing `aread_rows` existed to work around.

        Concurrency is unaffected: it comes from the pipeline's own workers WITHIN each
        read, and from :meth:`_run_fetches` ACROSS datasets.

        `read_rows` keeps `out=` without keeping asyncio: the decode lands straight in
        the caller's buffers, which under `preload_to_gpu` are PINNED host memory. The
        `__getitem__` arm copies the whole batch into pinned afterwards instead -- under
        1% of a CPU-bound batch, and a much larger share of a GPU-bound one, which is
        what this third arm exists to price.

        `indptr` is not filled here. It spans every dataset in the batch, so only
        :meth:`_index_datasets` knows the offsets, and it writes it afterwards.
        """
        # An anndata that reads runs takes them as they are, one range each.
        if hasattr(dataset, "read_row_ranges"):
            dataset.read_row_ranges(rows.starts, rows.lengths, out=(out.elems[0], out.elems[1]))
            return
        dataset.read_rows(rows.rows, out=(out.elems[0], out.elems[1]))

    def _run_fetches(self, tasks: list) -> None:
        """Run the per-dataset fetches concurrently, on threads instead of a loop.

        One task per dataset, the same shape as the `asyncio.gather` this replaces. A pool
        worker is not zarr's loop thread, so each task is free to make ordinary synchronous
        anndata calls -- which is the entire reason the async fetch interface existed.

        `map` is consumed rather than left lazy: an exception in a worker surfaces on
        iteration, and an unconsumed generator would swallow it.
        """
        if len(tasks) == 1:
            tasks[0]()  # no pool hop for the single-dataset case
            return
        list(_fetch_pool().map(lambda task: task(), tasks))

    def _index_datasets(
        self,
        dataset_index_to_rows: OrderedDict[int, _Runs],
    ) -> CSRContainer | np.ndarray:
        """Preallocate one output buffer, dispatch concurrent fetches into per-dataset views, then return the buffer.

        Parameters
        ----------
            dataset_index_to_rows
                A lookup of the list-placement index of a dataset to the sorted row indices to fetch.
        """
        is_backed_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset)
        is_sparse = is_backed_sparse or issubclass(self.dataset_type, sp.csr_array | sp.csr_matrix)
        if not self._dtypes_homogeneous:
            per_dataset_outs = self._allocate_per_dataset_outs(dataset_index_to_rows)
            tasks = [
                partial(
                    self._fetch_data,
                    self._train_datasets[dataset_idx],
                    rows,
                    per_dataset_outs[dataset_idx],
                )
                for dataset_idx, rows in dataset_index_to_rows.items()
            ]
            self._run_fetches(tasks)
            if is_sparse:
                datasets = self._train_datasets
                for dataset_idx, rows in dataset_index_to_rows.items():
                    sub_out = per_dataset_outs[dataset_idx]
                    cached_indptr = datasets[dataset_idx].indptr
                    per_row_nnz = cached_indptr[rows.rows + 1] - cached_indptr[rows.rows]
                    sub_out.elems[2][0] = 0
                    np.cumsum(per_row_nnz, out=sub_out.elems[2][1:])
            return self._concatenate_outs(per_dataset_outs)

        out = self._allocate_out(dataset_index_to_rows)

        tasks = []
        row_offset = 0
        nnz_offset = 0

        for dataset_idx, rows in dataset_index_to_rows.items():
            nrows = len(rows)
            if is_sparse:
                datasets = self._train_datasets
                nnz = rows.nnz(datasets[dataset_idx].indptr)
                out_view: CSRContainer | np.ndarray = CSRContainer(
                    elems=(
                        out.elems[0][nnz_offset : nnz_offset + nnz],
                        out.elems[1][nnz_offset : nnz_offset + nnz],
                        out.elems[2][row_offset : row_offset + nrows + 1],
                    ),
                    shape=(nrows, self.n_var),
                    dtype=out.dtype,
                )
                nnz_offset += nnz
            else:
                out_view = out[row_offset : row_offset + nrows]

            tasks.append(partial(self._fetch_data, self._train_datasets[dataset_idx], rows, out_view))
            row_offset += nrows

        self._run_fetches(tasks)

        if is_sparse:
            datasets = self._train_datasets
            running_nnz = 0
            row_pos = 0
            out.elems[2][0] = 0
            for dataset_idx, rows in dataset_index_to_rows.items():
                cached_indptr = datasets[dataset_idx].indptr
                per_row_nnz = cached_indptr[rows.rows + 1] - cached_indptr[rows.rows]
                dest = out.elems[2][row_pos + 1 : row_pos + len(rows) + 1]
                np.cumsum(per_row_nnz, out=dest)
                dest += running_nnz
                running_nnz = dest[-1]
                row_pos += len(rows)

        return out

    def __iter__(
        self,
    ) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Iterate over the on-disk datasets.

        Data for all requested datasets is fetched concurrently into a single preallocated
        buffer, converted to the output format once, and then yielded as direct row-index
        subsets — no vstack or intermediate concatenation is required.

        Yields
        ------
            A batch of data along with its obs and index (both optional).
        """
        check_lt_1(
            [len(self._train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        is_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset | sp.csr_matrix | sp.csr_array)
        # Create `positions` variable so we don't need to run `np.arange` (O(n)) every time
        positions = np.empty(0, dtype=np.intp)
        for load_request in self._batch_sampler.sample(self.n_obs):
            requests_to_load = load_request.get("requests", None)
            if requests_to_load is None:
                raise KeyError("load_request must contain either 'requests'.")
            splits = load_request["splits"]

            dataset_index_to_rows, order = self._requests_to_dataset_rows(requests_to_load)

            # The buffer below is filled in dataset order, but ``splits`` are expressed in the
            # sampler's `LoadRequest.request` order. ``inv`` maps a request-order position to its buffer position so
            # the split semantics are independent of how chunks were regrouped across datasets.
            # ``order`` is a permutation of ``range(n)``, so every used slot is overwritten -- the
            # reused buffer never carries stale values from a previous request.
            n = order.size
            inv_buffer = np.empty(n, dtype=np.intp)
            if n > positions.size:
                positions = np.arange(n, dtype=np.intp)
            inv = inv_buffer[:n]
            inv[order] = positions[:n]

            raw_out: CSRContainer | np.ndarray = self._index_datasets(dataset_index_to_rows)

            if is_sparse:
                in_memory_data = self._sp_module.csr_matrix(
                    tuple(self._np_module.asarray(e) for e in raw_out.elems),
                    shape=raw_out.shape,
                    dtype=_cupy_dtype(raw_out.dtype) if self._preload_to_gpu else raw_out.dtype,
                )
            else:
                in_memory_data = self._np_module.asarray(raw_out)

            concatenated_obs: None | pd.DataFrame = self._maybe_accumulate_obs(dataset_index_to_rows)
            in_memory_indices: None | np.ndarray = self._maybe_accumulate_indices(dataset_index_to_rows)
            for split in splits:
                sel = inv[split]
                data = in_memory_data[sel]
                yield {
                    "X": data if self._to is None else convert(data, self._preload_to_gpu, self._to),
                    "obs": concatenated_obs.iloc[sel] if concatenated_obs is not None else None,
                    "var": self._var,
                    "index": in_memory_indices[sel] if in_memory_indices is not None else None,
                }

            # https://github.com/cupy/cupy/issues/9625
            if self._preload_to_gpu and is_sparse:
                self._np_module.get_default_memory_pool().free_all_blocks()

    def _maybe_accumulate_obs(self, dataset_index_to_rows: OrderedDict[int, _Runs]) -> pd.DataFrame | None:
        """Gather obs labels for the loaded rows if possible."""
        if self._obs is None:
            return None
        return pd.concat([self._obs[idx].iloc[rows.rows] for idx, rows in dataset_index_to_rows.items()])

    def _maybe_accumulate_indices(self, dataset_index_to_rows: OrderedDict[int, _Runs]) -> np.ndarray | None:
        """Gather original indices for the loaded rows if possible."""
        if self._return_index is False:
            return None
        dataset_offsets = np.concatenate(([0], np.cumsum([shape[0] for shape in self._shapes])))
        return np.concatenate([rows.rows + dataset_offsets[idx] for idx, rows in dataset_index_to_rows.items()])
