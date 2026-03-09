from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict
from functools import singledispatchmethod
from importlib.util import find_spec
from itertools import accumulate, chain, pairwise
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, cast

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from scipy import sparse as sp
from zarr import Array as ZarrArray

from annbatch.samplers import ChunkSampler
from annbatch.types import (
    BackingArray_T,
    InputInMemoryArray_T,
    LoadRequest,
    LoaderOutput,
    OutputInMemoryArray_T,
    _multi_arange,
)
from annbatch.utils import (
    CSRContainer,
    MultiBasicIndexer,
    check_lt_1,
    check_var_shapes,
    load_x_and_obs_and_var,
    to_torch,
    validate_sampler,
)
from annbatch._direct_read import read_direct_dense, read_direct_1d

from .compat import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import ModuleType

    from annbatch.abc import Sampler
    from annbatch.io import DatasetCollection

    # TODO: remove after sphinx 9 - myst compat
    BackingArray = BackingArray_T
    OutputInMemoryArray = OutputInMemoryArray_T
    InputInMemoryArray = InputInMemoryArray_T

type concat_strategies = Literal["concat-shuffle", "shuffle-concat"]


class CSRDatasetElems(NamedTuple):
    """Container for cached objects that will be indexed into to generate CSR matrices"""

    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


def _cupy_dtype(dtype: np.dtype) -> np.dtype:
    if dtype in {np.dtype("float32"), np.dtype("float64"), np.dtype("bool")}:
        return dtype
    if dtype.itemsize < 4:
        return np.dtype("float32")
    return np.dtype("float64")


class Loader[
    BackingArray: BackingArray_T,
    InputInMemoryArray: InputInMemoryArray_T,
    OutputInMemoryArray: OutputInMemoryArray_T,
](IterableDataset):
    """A loader for on-disk data anndata stores.

    This loader batches together slice requests to the underlying stores to achieve higher performance.
    This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
    The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

    The dataset class on its own is quite performant for "chunked loading" i.e., `chunk_size > 1`.
    When `chunk_size == 1`, a :class:`torch.utils.data.DataLoader` should wrap the dataset object.
    In this case, be sure to use `spawn` multiprocessing in the wrapping loader.

    If `preload_to_gpu` to True and `to_torch` is False, the yielded type is a `cupy` matrix.
    If `to_torch` is True, the yielded type is a :class:`torch.Tensor`.
    If both `preload_to_gpu` and `to_torch` are False, then the return type is the CPU class for the given data type.
    When providing a custom sampler, `chunk_size`, `preload_nchunks`, `batch_size`,
    `shuffle`, `drop_last`, and `rng` must not be set (they are controlled by the `batch_sampler` instead).
    When providing these arguments and no `batch_sampler`, they are used to construct a :class:`annbatch.ChunkSampler`.

    Parameters
    ----------
        batch_sampler
            If not provided, a default :class:`annbatch.ChunkSampler` will be used with the same defaults below.
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
            Whether or not to use cupy for non-io array operations like vstack and indexing once the data is in memory internally.
            This option entails greater GPU memory usage, but is faster at least for sparse operations.
            :func:`torch.vstack` does not support CSR sparse matrices, hence the current use of `cupy` internally (which also means `torch` is an optional dep).
            Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data due to memory pressure.
            For top performance, this should be used in conjuction with `to_torch` and then :meth:`torch.Tensor.to_dense` if you wish to densify.
            :meth:`cupy.cuda.MemoryPool.free_all_blocks` (i.e., the method of the pool of :func:`cupy.get_default_memory_pool()`) is called aggresively to keep memory usage low.
            If you are using your own memory pool or allocator, you may have to free blocks on your own.
        to_torch
            Whether to return `torch.Tensor` as the output.
            Data transferred should be 0-copy independent of source, and transfer to cuda when applicable is non-blocking.
            Defaults to True if `torch` is installed.
        concat_strategy
            The strategy for how in-memory, preloaded data should be concatenated and yielded.
            With `concat-shuffle`, preloaded data is concatenated and then subsetted/shuffled (higher memory usage, but faster, at least for sparse data)
            With `shuffle-concat`, preloaded data is first shuffled/subsetted chunk-by-chunk and then concatenated (lower memory usage, potentially faster for dense data)
            The default is automatically chosen - `concat-shuffle` if the data added to the loader is sparse and otherwise `shuffle-concat`.
            See


    Examples
    --------
        >>> from annbatch import Loader
        >>> ds = Loader(
                batch_size=4096,
                chunk_size=32,
                preload_nchunks=512,
            ).add_anndata(my_anndata)
        >>> for batch in ds:
                # optionally convert to dense
                # batch = batch.to_dense()
                do_fit(batch)
    """

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 512,
        "preload_nchunks": 32,
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "rng": np.random.default_rng(),
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
    _to_torch: bool = True
    _dataset_elem_cache: dict[int, CSRDatasetElems]
    _batch_sampler: Sampler
    _concat_strategy: None | concat_strategies = None
    _dataset_intervals: pd.IntervalIndex | None = None

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
        to_torch: bool = find_spec("torch") is not None,
        concat_strategy: None | concat_strategies = None,
        rng: np.random.Generator | None = None,
    ):
        sampler_args = {
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "rng": rng,
        }
        if batch_sampler is not None:
            if any(v is not None for v in sampler_args.values()):
                provided_args = [name for name, val in sampler_args.items() if val is not None]
                raise ValueError(
                    f"Cannot specify {', '.join(provided_args)} when providing a custom sampler. "
                    "These parameters are controlled by the sampler."
                )
            self._batch_sampler = batch_sampler
        else:
            sampler_args_processed = {
                k: (v if v is not None else Loader._COMMON_SAMPLER_ARGS[k]) for k, v in sampler_args.items()
            }
            self._batch_sampler = ChunkSampler(**sampler_args_processed)

        if to_torch and not find_spec("torch"):
            raise ImportError("Could not find torch dependency. Try `pip install torch`.")
        if preload_to_gpu and not find_spec("cupy"):
            raise ImportError("Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy.")

        self._return_index = return_index
        self._preload_to_gpu = preload_to_gpu
        self._to_torch = to_torch
        self._train_datasets = []
        self._shapes = []
        self._dataset_elem_cache = {}
        self._direct_sparse_cache: dict = {}
        self._concat_strategy = concat_strategy

    def __len__(self) -> int:
        return self._batch_sampler.n_iters(self.n_obs)

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
        self, collection: DatasetCollection, *, load_adata: Callable[[zarr.Group], ad.AnnData] = load_x_and_obs_and_var
    ) -> Self:
        """Load from an existing :class:`annbatch.DatasetCollection`.

        This function can only be called once. If you want to manually add more data, use :meth:`Loader.add_anndatas` or open an issue.

        Parameters
        ----------
        collection
            The collection who on-disk datasets should be used in this loader.
        load_adata
            A custom load function - recall that whatever is found in :attr:`~anndata.AnnData.X` and :attr:`~anndata.AnnData.obs` will be yielded in batches.
            Default is to just load `X` and all of `obs`.
            This default behavior can degrade performance if you don't need all columns in `obs` - it is recommended to use the `load_adata` argument.
        """
        if collection.is_empty:
            raise ValueError("DatasetCollection is empty")
        if getattr(self, "_collection_added", False):
            raise RuntimeError(
                "You should not add multiple collections, independently shuffled - please preshuffle multiple collections, use `add_anndatas` manually if you know what you are doing, or open an issue if you believe that this should be supported at an API level higher than `add_anndatas`."
            )
        adatas = [load_adata(g) for g in collection]
        self.add_anndatas(adatas)
        self._collection_added = True
        return self

    @validate_sampler
    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append anndatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
        """
        check_lt_1([len(adatas)], ["Number of anndatas"])
        for adata in adatas:
            dataset, obs, var = self._prepare_dataset_obs_and_var(adata)
            self._add_dataset_unchecked(dataset, obs, var)
        return self

    def add_anndata(self, adata: ad.AnnData) -> Self:
        """Append an anndata to this dataset.

        Parameters
        ----------
            adata
                A :class:`anndata.AnnData` object, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
                :attr:`~anndata.AnnData.var` must match the ``var`` of any previously added datasets.
        """
        dataset, obs, var = self._prepare_dataset_obs_and_var(adata)
        self.add_dataset(dataset, obs, var)
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
        self, dataset: BackingArray, obs: pd.DataFrame | None = None, var: pd.DataFrame | None = None
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
        self, dataset: BackingArray, obs: pd.DataFrame | None = None, var: pd.DataFrame | None = None
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
        if (is_sparse := isinstance(dataset, ad.abc.CSRDataset)) and not dataset.backend == "zarr":
            raise TypeError(
                "Cannot add CSRDataset backed by h5ad at the moment: see https://github.com/zarr-developers/VirtualiZarr/pull/790"
            )
        if not isinstance(obs, pd.DataFrame) and obs is not None:
            raise TypeError("obs must be a pandas DataFrame")
        if not isinstance(var, pd.DataFrame) and var is not None:
            raise TypeError("var must be a pandas DataFrame")
        datasets = self._train_datasets + [dataset]
        check_var_shapes(datasets)
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
        if self._concat_strategy is None:
            if is_sparse:
                self._concat_strategy = "concat-shuffle"
            else:
                self._concat_strategy = "shuffle-concat"
        self._update_dataset_intervals()
        return self

    def _update_dataset_intervals(self) -> None:
        if len(self._shapes) == 0:
            self._dataset_intervals = None
            return
        # Build intervals [start, end) for each dataset
        cumsum = list(accumulate(shape[0] for shape in self._shapes))
        starts = [0] + cumsum[:-1]
        ends = cumsum
        self._dataset_intervals = pd.IntervalIndex.from_arrays(starts, ends, closed="left")

    def _load_request_to_dataset_boundaries(
        self, lr: LoadRequest, *, use_original_space: bool = False
    ) -> OrderedDict[int, np.ndarray]:
        """Map a LoadRequest to per-dataset boundary arrays.

        Each value is a flat interleaved 1-D int array
        ``[s0, e0, s1, e1, ...]`` of length ``2 * n_chunks``.
        Starts are ``arr[::2]``, stops are ``arr[1::2]`` -- zero-copy views,
        no ``slice`` objects created.

        For the single-dataset fast path (the common case), all interval
        overlap checks are skipped entirely.
        """
        if self._dataset_intervals is None:
            return OrderedDict()

        chunk_starts = lr["starts"]
        chunk_stops = lr["stops"]

        # Fast path: single dataset -- every chunk belongs to dataset 0
        if len(self._shapes) == 1:
            interleaved = np.empty(2 * len(chunk_starts), dtype=np.intp)
            interleaved[::2] = chunk_starts
            interleaved[1::2] = chunk_stops
            return OrderedDict([(0, interleaved)])

        # Multi-dataset path
        ds_starts = np.array([iv.left for iv in self._dataset_intervals])
        ds_ends = np.array([iv.right for iv in self._dataset_intervals])

        result_s: defaultdict[int, list[int]] = defaultdict(list)
        result_e: defaultdict[int, list[int]] = defaultdict(list)
        for ci in range(len(chunk_starts)):
            c_start, c_stop = int(chunk_starts[ci]), int(chunk_stops[ci])
            overlapping = np.flatnonzero((ds_starts < c_stop) & (ds_ends > c_start))
            for di in overlapping:
                a_start, a_end = int(ds_starts[di]), int(ds_ends[di])
                s = max(c_start, a_start)
                e = min(c_stop, a_end)
                if use_original_space:
                    result_s[int(di)].append(s)
                    result_e[int(di)].append(e)
                else:
                    result_s[int(di)].append(s - a_start)
                    result_e[int(di)].append(e - a_start)

        out: OrderedDict[int, np.ndarray] = OrderedDict()
        for k in sorted(result_s.keys()):
            sa = np.array(result_s[k])
            ea = np.array(result_e[k])
            interleaved = np.empty(2 * len(sa), dtype=np.intp)
            interleaved[::2] = sa
            interleaved[1::2] = ea
            out[k] = interleaved
        return out

    @singledispatchmethod
    async def _fetch_data(self, dataset: ZarrArray | CSRDatasetElems, boundaries: np.ndarray) -> InputInMemoryArray:
        """Fetch data from an on-disk store.

        Parameters
        ----------
        dataset
            The underlying store.
        boundaries
            1-D int array of length ``n_chunks + 1``.
            ``slice`` objects are built only at the zarr API call site.

        Returns
        -------
            The sparse or dense fetched data.

        Raises
        ------
        NotImplementedError
            If the dataset is not recognized.
        """
        raise NotImplementedError(f"Cannot fetch data for type {type(dataset)}")

    @_fetch_data.register
    async def _fetch_data_dense(self, dataset: ZarrArray, boundaries: np.ndarray) -> np.ndarray:
        indexer = MultiBasicIndexer.from_boundaries(
            boundaries, dataset.metadata.shape, dataset.metadata.chunk_grid
        )
        res = cast(
            "np.ndarray",
            await dataset._async_array._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
        )
        return res

    async def _create_sparse_elems(self, idx: int) -> CSRDatasetElems:
        """Fetch the in-memory indptr, and backed indices and data for a given dataset index.

        Parameters
        ----------
            idx
                The index

        Returns
        -------
            The constituent elems of the CSR dataset.
        """
        if isinstance(ds := self._train_datasets[idx], ZarrArray):
            raise ValueError(f"Requested sparse dataset at idx {idx} of {self._train_datasets} but found dense array")
        indptr = await ds.group._async_group.getitem("indptr")
        return CSRDatasetElems(
            *(
                await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    ds.group._async_group.getitem("indices"),
                    ds.group._async_group.getitem("data"),
                )
            )
        )

    async def _ensure_sparse_cache(self) -> None:
        """Build up the cache of datasets i.e., in-memory indptr, and backed indices and data."""
        arr_idxs = [idx for idx in range(len(self._train_datasets)) if idx not in self._dataset_elem_cache]
        all_elems: list[CSRDatasetElems] = await asyncio.gather(
            *(
                self._create_sparse_elems(idx)
                for idx in range(len(self._train_datasets))
                if idx not in self._dataset_elem_cache
            )
        )
        for idx, elems in zip(arr_idxs, all_elems, strict=True):
            self._dataset_elem_cache[idx] = elems

    def _get_elem_from_cache(self, dataset_idx: int) -> CSRDatasetElems | ZarrArray:
        """Return the arrays (zarr or otherwise) needed to represent on-disk data at a given index.

        Parameters
        ----------
            dataset_idx
                The index of the dataset whose arrays are sought.

        Returns
        -------
            The arrays representing the sparse data.
        """
        if dataset_idx not in self._dataset_elem_cache:
            raise ValueError("Cache not prepared")
        return self._dataset_elem_cache[dataset_idx]

    @_fetch_data.register
    async def _fetch_data_sparse(
        self,
        dataset: CSRDatasetElems,
        boundaries: np.ndarray,
    ) -> CSRContainer:
        # See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        indptr, indices, data = dataset
        starts, stops = boundaries[::2], boundaries[1::2]
        n = len(starts)
        indptr_indices = [indptr[int(s):int(e) + 1] for s, e in zip(starts, stops, strict=True)]

        # Build an interleaved boundaries array for the 1-D data/indices arrays
        ip_starts = np.array([int(ip[0]) for ip in indptr_indices])
        ip_stops = np.array([int(ip[-1]) for ip in indptr_indices])
        ip_boundaries = np.empty(2 * n, dtype=np.intp)
        ip_boundaries[::2] = ip_starts
        ip_boundaries[1::2] = ip_stops
        indexer = MultiBasicIndexer.from_boundaries(
            ip_boundaries, data.metadata.shape, data.metadata.chunk_grid
        )

        data_np, indices_np = await asyncio.gather(
            data._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
            indices._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
        )

        gaps = (int(ip_starts[i + 1]) - int(ip_stops[i]) for i in range(n - 1))
        offsets = accumulate(chain([int(ip_starts[0])], gaps))
        start_indptr = indptr_indices[0] - next(offsets)
        if n < 2:
            return CSRContainer(
                elems=(data_np, indices_np, start_indptr),
                shape=(start_indptr.shape[0] - 1, self.n_var),
                dtype=data_np.dtype,
            )
        end_indptr = np.concatenate([s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)])
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return CSRContainer(
            elems=(data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self.n_var),
            dtype=data_np.dtype,
        )

    async def _index_datasets(
        self,
        dataset_boundaries: OrderedDict[int, np.ndarray],
    ) -> list[InputInMemoryArray]:
        """Fetch data from each dataset using boundary arrays.

        Parameters
        ----------
            dataset_boundaries
                Mapping from dataset index to a boundaries array (length
                ``n_chunks + 1``).
        """
        tasks = []
        if is_sparse := issubclass(self.dataset_type, ad.abc.CSRDataset):
            await self._ensure_sparse_cache()
        for dataset_idx, boundaries in dataset_boundaries.items():
            tasks.append(
                self._fetch_data(
                    self._get_elem_from_cache(dataset_idx) if is_sparse else self._train_datasets[dataset_idx],
                    boundaries,
                )
            )
        return await asyncio.gather(*tasks)

    def _index_datasets_direct(
        self,
        dataset_boundaries: OrderedDict[int, np.ndarray],
    ) -> list[InputInMemoryArray]:
        """Synchronous direct-read path for local sharded stores.

        Uses mmap + C extension instead of zarr's async pipeline.
        Works for both single and multiple datasets.
        """
        is_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset)
        results: list[InputInMemoryArray] = []
        for dataset_idx, boundaries in dataset_boundaries.items():
            ds = self._train_datasets[dataset_idx]
            if is_sparse:
                results.append(self._fetch_direct_sparse(ds, boundaries))
            else:
                results.append(read_direct_dense(ds, boundaries))
        return results

    def _fetch_direct_sparse(
        self,
        dataset: ad.abc.CSRDataset,
        boundaries: np.ndarray,
    ) -> CSRContainer:
        """Synchronous direct-read for a single sparse dataset."""
        grp = dataset.group
        if dataset not in self._direct_sparse_cache:
            indptr = zarr.open(grp.store, path=grp.path + "/indptr")[:]
            data_arr = zarr.open(grp.store, path=grp.path + "/data")
            indices_arr = zarr.open(grp.store, path=grp.path + "/indices")
            self._direct_sparse_cache[dataset] = (indptr, data_arr, indices_arr)
        indptr, data_arr, indices_arr = self._direct_sparse_cache[dataset]

        starts, stops = boundaries[::2], boundaries[1::2]
        n = len(starts)
        indptr_indices = [indptr[int(s):int(e) + 1] for s, e in zip(starts, stops, strict=True)]

        ip_starts = np.array([int(ip[0]) for ip in indptr_indices])
        ip_stops = np.array([int(ip[-1]) for ip in indptr_indices])
        ip_bounds = np.empty(2 * n, dtype=np.intp)
        ip_bounds[::2] = ip_starts
        ip_bounds[1::2] = ip_stops

        data_np = read_direct_1d(data_arr, ip_bounds)
        indices_np = read_direct_1d(indices_arr, ip_bounds)

        gaps = (int(ip_starts[i + 1]) - int(ip_stops[i]) for i in range(n - 1))
        offsets = accumulate(chain([int(ip_starts[0])], gaps))
        start_indptr = indptr_indices[0] - next(offsets)
        if n < 2:
            return CSRContainer(
                elems=(data_np, indices_np, start_indptr),
                shape=(start_indptr.shape[0] - 1, self.n_var),
                dtype=data_np.dtype,
            )
        end_indptr = np.concatenate([s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)])
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return CSRContainer(
            elems=(data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self.n_var),
            dtype=data_np.dtype,
        )

    def __iter__(
        self,
    ) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Iterate over the on-disk datasets.

        Data is fetched from `N` on-disk anndata objects, returning `N` blocks which are then either concatenated immediately and then yieled/shuffled, or subsetted to shuffled subsets and then concatenated/yielded.
        See `concat_strategy` initialization argument for more information.

        Yields
        ------
            A batch of data along with its obs and index (both optional).
        """
        check_lt_1(
            [len(self._train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        all_direct = all(self._can_direct_read(ds) for ds in self._train_datasets)
        if len(self._shapes) == 1 and all_direct:
            if isinstance(self._train_datasets[0], ZarrArray):
                yield from self._iter_direct_dense()
            else:
                yield from self._iter_direct_sparse()
        elif len(self._shapes) == 1 and isinstance(self._train_datasets[0], ZarrArray):
            yield from self._iter_single_dense()
        else:
            yield from self._iter_generic(use_direct=all_direct)

    @staticmethod
    def _can_direct_read(dataset: BackingArray_T) -> bool:
        """Check if the dataset is backed by a local sharded zarr store."""
        try:
            if isinstance(dataset, ZarrArray):
                arr = dataset
            else:
                arr = dataset.group["data"]
            store = arr.store
            if not isinstance(store, zarr.storage.LocalStore):
                return False
            m = arr.metadata
            if not m.codecs or not isinstance(m.codecs[0], zarr.codecs.sharding.ShardingCodec):
                return False
            return True
        except Exception:
            return False

    def _iter_direct_dense(self) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Fastest path: single dense dataset on local sharded store.

        Bypasses zarr's entire async pipeline -- reads shard files directly,
        parses indices, decompresses with blosc synchronously.
        """
        dataset = self._train_datasets[0]
        obs0 = self._obs[0] if self._obs is not None else None
        var = self._var
        has_obs = obs0 is not None
        has_idx = self._return_index
        do_torch = self._to_torch
        gpu = self._preload_to_gpu

        for load_request in self._batch_sampler.sample(self.n_obs):
            starts = load_request["starts"]
            stops = load_request["stops"]
            interleaved = np.empty(2 * len(starts), dtype=np.intp)
            interleaved[::2] = starts
            interleaved[1::2] = stops

            in_memory_data = read_direct_dense(dataset, interleaved)

            if has_obs or has_idx:
                flat_indices = _multi_arange(starts, stops)
                concatenated_obs = obs0.iloc[flat_indices] if has_obs else None
                in_memory_indices = flat_indices if has_idx else None
            else:
                concatenated_obs = None
                in_memory_indices = None

            for split in load_request["splits"]:
                data = in_memory_data[split]
                yield {
                    "X": data if not do_torch else to_torch(data, gpu),
                    "obs": concatenated_obs.iloc[split] if concatenated_obs is not None else None,
                    "var": var,
                    "index": in_memory_indices[split] if in_memory_indices is not None else None,
                }

    def _iter_direct_sparse(self) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Fastest path: single sparse dataset on local sharded store.

        Reads indptr from memory, then fetches CSR data/indices via direct
        shard reads -- no async overhead.
        """
        csr_ds = self._train_datasets[0]
        obs0 = self._obs[0] if self._obs is not None else None
        var = self._var
        has_obs = obs0 is not None
        has_idx = self._return_index
        do_torch = self._to_torch
        gpu = self._preload_to_gpu
        sp_mod = self._sp_module
        np_mod = self._np_module

        for load_request in self._batch_sampler.sample(self.n_obs):
            starts = load_request["starts"]
            stops = load_request["stops"]
            interleaved = np.empty(2 * len(starts), dtype=np.intp)
            interleaved[::2] = starts
            interleaved[1::2] = stops

            csr = self._fetch_direct_sparse(csr_ds, interleaved)
            in_memory_data = sp_mod.csr_matrix(
                tuple(np_mod.asarray(e) for e in csr.elems),
                shape=csr.shape,
                dtype=csr.dtype,
            )

            if has_obs or has_idx:
                flat_indices = _multi_arange(starts, stops)
                concatenated_obs = obs0.iloc[flat_indices] if has_obs else None
                in_memory_indices = flat_indices if has_idx else None
            else:
                concatenated_obs = None
                in_memory_indices = None

            for split in load_request["splits"]:
                data = in_memory_data[split]
                yield {
                    "X": data if not do_torch else to_torch(data, gpu),
                    "obs": concatenated_obs.iloc[split] if concatenated_obs is not None else None,
                    "var": var,
                    "index": in_memory_indices[split] if in_memory_indices is not None else None,
                }

    def _iter_single_dense(self) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Fast path for single dense dataset (non-local or non-sharded stores).

        Uses MultiBasicIndexer.from_boundaries through zarr's async pipeline.
        """
        dataset = self._train_datasets[0]
        obs0 = self._obs[0] if self._obs is not None else None
        var = self._var
        has_obs = obs0 is not None
        has_idx = self._return_index
        do_torch = self._to_torch
        gpu = self._preload_to_gpu
        shape = dataset.metadata.shape
        chunk_grid = dataset.metadata.chunk_grid

        for load_request in self._batch_sampler.sample(self.n_obs):
            starts = load_request["starts"]
            stops = load_request["stops"]
            interleaved = np.empty(2 * len(starts), dtype=np.intp)
            interleaved[::2] = starts
            interleaved[1::2] = stops

            indexer = MultiBasicIndexer.from_boundaries(interleaved, shape, chunk_grid)
            in_memory_data = cast(
                "np.ndarray",
                zsync.sync(
                    dataset._async_array._get_selection(
                        indexer, prototype=zarr.core.buffer.default_buffer_prototype()
                    )
                ),
            )

            if has_obs or has_idx:
                flat_indices = _multi_arange(starts, stops)
                concatenated_obs = obs0.iloc[flat_indices] if has_obs else None
                in_memory_indices = flat_indices if has_idx else None
            else:
                concatenated_obs = None
                in_memory_indices = None

            for split in load_request["splits"]:
                data = in_memory_data[split]
                yield {
                    "X": data if not do_torch else to_torch(data, gpu),
                    "obs": concatenated_obs.iloc[split] if concatenated_obs is not None else None,
                    "var": var,
                    "index": in_memory_indices[split] if in_memory_indices is not None else None,
                }

    def _iter_generic(self, use_direct: bool = False) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """General path supporting multiple datasets and sparse data."""
        mod = self._sp_module if issubclass(self.dataset_type, ad.abc.CSRDataset) else np
        for load_request in self._batch_sampler.sample(self.n_obs):
            splits = load_request["splits"]
            ds_boundaries = self._load_request_to_dataset_boundaries(load_request, use_original_space=False)
            if use_direct:
                chunks: list[InputInMemoryArray] = self._index_datasets_direct(ds_boundaries)
            else:
                chunks = zsync.sync(self._index_datasets(ds_boundaries))
            in_memory_data = self._accumulate_chunks(chunks)
            concatenated_obs: None | pd.DataFrame = self._maybe_accumulate_obs(load_request, ds_boundaries)
            in_memory_indices: None | np.ndarray = self._maybe_accumulate_indices(load_request)
            if self._concat_strategy == "concat-shuffle":
                in_memory_data = mod.vstack(in_memory_data)
                for split in splits:
                    data = in_memory_data[split]
                    yield {
                        "X": data if not self._to_torch else to_torch(data, self._preload_to_gpu),
                        "obs": concatenated_obs.iloc[split] if concatenated_obs is not None else None,
                        "var": self._var,
                        "index": in_memory_indices[split] if in_memory_indices is not None else None,
                    }
            elif self._concat_strategy == "shuffle-concat":
                dataset_interval_indexer = self._interval_indexer_from_boundaries(ds_boundaries.values())
                for split in splits:
                    sorted_split = np.sort(split)
                    dataset_locs = dataset_interval_indexer.get_indexer_for(sorted_split)
                    offsets = dataset_interval_indexer.left[dataset_locs]
                    data = mod.vstack(
                        [
                            chunk[sorted_split[dataset_locs == i] - offsets[dataset_locs == i]]
                            for i, chunk in enumerate(in_memory_data)
                        ]
                    )
                    yield {
                        "X": data if not self._to_torch else to_torch(data, self._preload_to_gpu),
                        "obs": concatenated_obs.iloc[sorted_split] if concatenated_obs is not None else None,
                        "var": self._var,
                        "index": in_memory_indices[sorted_split] if in_memory_indices is not None else None,
                    }
            else:  # pragma: no cover
                raise RuntimeError(
                    f"Found unrecognized concatenation strategy at iteration time {self._concat_strategy}.  Please open an issue"
                )
            # https://github.com/cupy/cupy/issues/9625
            if self._preload_to_gpu and issubclass(self.dataset_type, ad.abc.CSRDataset):
                self._np_module.get_default_memory_pool().free_all_blocks()

    @staticmethod
    def _interval_indexer_from_boundaries(boundaries_iter: Iterable[np.ndarray]) -> pd.IntervalIndex:
        """Build an IntervalIndex from per-dataset interleaved boundary arrays."""
        totals = [int((b[1::2] - b[::2]).sum()) for b in boundaries_iter]
        ends = list(accumulate(totals))
        starts = [0] + ends[:-1]
        return pd.IntervalIndex.from_arrays(starts, ends, closed="left")

    def _accumulate_chunks(self, chunks: list[InputInMemoryArray]) -> list[OutputInMemoryArray_T]:
        """Convert fetched chunks to output array format (CSR or ndarray)."""
        result: list[OutputInMemoryArray_T] = []
        for chunk in chunks:
            if isinstance(chunk, CSRContainer):
                result.append(
                    self._sp_module.csr_matrix(
                        tuple(self._np_module.asarray(e) for e in chunk.elems),
                        shape=chunk.shape,
                        dtype=_cupy_dtype(chunk.dtype) if self._preload_to_gpu else chunk.dtype,
                    )
                )
            else:
                result.append(self._np_module.asarray(chunk))
        return result

    def _maybe_accumulate_obs(
        self, lr: LoadRequest, ds_boundaries: OrderedDict[int, np.ndarray]
    ) -> pd.DataFrame | None:
        """Gather obs labels for the loaded chunks."""
        if self._obs is None:
            return None
        if len(self._shapes) == 1:
            return self._obs[0].iloc[_multi_arange(lr["starts"], lr["stops"])]
        return pd.concat(
            [
                self._obs[idx].iloc[_multi_arange(b[::2], b[1::2])]
                for idx, b in ds_boundaries.items()
            ]
        )

    def _maybe_accumulate_indices(self, lr: LoadRequest) -> np.ndarray | None:
        """Gather original indices for the loaded chunks."""
        if self._return_index is False:
            return None
        if len(self._shapes) == 1:
            return _multi_arange(lr["starts"], lr["stops"])
        ds_boundaries = self._load_request_to_dataset_boundaries(lr, use_original_space=True)
        return np.concatenate(
            [_multi_arange(b[::2], b[1::2]) for b in ds_boundaries.values()]
        )
