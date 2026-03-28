from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Self

import anndata as ad
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata._core.sparse_dataset import BaseCompressedSparseDataset
from anndata.experimental.backed import Dataset2D
from dask.array.core import Array as DaskArray
from humanfriendly import parse_size
from tqdm.auto import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

from annbatch.utils import split_given_size

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
    from os import PathLike
    from typing import Any, Literal

    from zarr.abc.codec import BytesBytesCodec

V1_ENCODING = {"encoding-type": "annbatch-preshuffled", "encoding-version": "0.1.0"}
V1_GROUPED_ENCODING = {"encoding-type": "annbatch-grouped", "encoding-version": "0.1.0"}
GROUP_INDEX_KEY = "group_index"


def _default_load_adata[T: zarr.Group | h5py.Group | PathLike[str] | str](x: T) -> ad.AnnData:
    adata = ad.experimental.read_lazy(x, load_annotation_index=False)
    if not isinstance(x, zarr.Group | h5py.Group):
        group = (
            h5py.File(adata.file.filename, mode="r")
            if adata.file.filename is not None
            else zarr.open_group(x, mode="r")
        )
    else:
        group = x
    # -1 indicates that all of each `obs` column should just be loaded, but this is probably fine since it goes column by column and discards.
    # TODO: Bug with empty columns: https://github.com/scverse/anndata/pull/2307
    for attr in ["obs", "var"]:
        # Only one column at a time will be loaded so we will hopefully pick up the benefit of loading into memory by the cache without having memory pressure.
        if len(getattr(adata, attr).columns) > 0:
            setattr(adata, attr, ad.experimental.read_elem_lazy(group[attr], chunks=(-1,), use_range_index=True))
            for col in getattr(adata, attr).columns:
                # Nullables / categoricals have bad perforamnce characteristics when concatenating using dask
                if pd.api.types.is_extension_array_dtype(getattr(adata, attr)[col].dtype):
                    getattr(adata, attr)[col] = getattr(adata, attr)[col].data
    return adata


def _round_down(num: int, divisor: int):
    return num - (num % divisor)


def _shard_size_param_to_n_obs(shard_size: int | str, elem) -> int:
    """Convert `shard_size` to a number of observations given the size of an element from the anndata object.

    If *shard_size* is already an int, it is interpreted as `n_obs`.  When it is a
    size string the target byte budget is divided by the element's
    uncompressed bytes-per-observation-row.
    """
    if isinstance(shard_size, int):
        return shard_size
    target_bytes = parse_size(shard_size, binary=True)

    def _cs_bytes(x) -> int:
        return int(x.data.nbytes + x.indptr.nbytes + x.indices.nbytes)

    n_obs = elem.shape[0] if hasattr(elem, "shape") else len(elem)
    if n_obs == 0:
        return 1

    if isinstance(elem, h5py.Dataset):
        total_bytes = int(np.array(elem.shape).prod() * elem.dtype.itemsize)
    elif isinstance(elem, BaseCompressedSparseDataset):
        total_bytes = _cs_bytes(elem._to_backed())
    elif sp.issparse(elem):
        total_bytes = _cs_bytes(elem)
    else:
        total_bytes = elem.__sizeof__()

    bytes_per_row = total_bytes / n_obs
    return max(1, int(target_bytes / bytes_per_row)) if bytes_per_row > 0 else 1


def write_sharded(
    group: zarr.Group,
    adata: ad.AnnData,
    *,
    n_obs_per_chunk: int = 64,
    shard_size: int | str = "1GB",
    compressors: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    key: str | None = None,
):
    """Write a sharded zarr store from a single AnnData object.

    Parameters
    ----------
        group
            The destination group, must be zarr v3
        adata
            The source anndata object
        n_obs_per_chunk
            Number of observations per chunk. For dense arrays this directly sets the first-axis chunk size.
            For sparse arrays it is converted to element counts using the average non-zero elements per row of the matrix being written.
        shard_size
            Number of observations per shard, or a size string (e.g. ``'1GB'``, ``'512MB'``).
            If a size string is provided, the observation count is derived independently for each array element from its uncompressed bytes-per-row so that every shard stays close to the target size.
            For dense arrays the resolved count directly sets the first-axis shard size.
            For sparse arrays it is converted to element counts using the average non-zero elements per row of the matrix being written.
        compressors
            The compressors to pass to `zarr`.
        key
            The key to which this object should be written - by default the root, in which case the *entire* store (not just the group) is cleared first.
    """
    with ad.settings.override(zarr_write_format=3, write_csr_csc_indices_with_min_possible_dtype=True):

        def callback(
            write_func: ad.experimental.Write,
            store: zarr.Group,
            elem_name: str,
            elem: ad.typing.RWAble,
            dataset_kwargs: Mapping[str, Any],
            *,
            iospec: ad.experimental.IOSpec,
        ):
            # Ensure we're not overriding anything here
            dataset_kwargs = dict(dataset_kwargs)
            if iospec.encoding_type in {"array"} and (
                any(n in store.name for n in {"obsm", "layers", "obsp"}) or "X" == elem_name
            ):
                obs_per_shard = _shard_size_param_to_n_obs(shard_size, elem)
                # Clamp chunk/shard to the element size for small datasets
                dense_chunk = min(n_obs_per_chunk, elem.shape[0])
                if dense_chunk == 0:
                    raise ValueError(f"Cannot write sharded array {elem_name!r} with 0 observations.")
                dense_shard = min(obs_per_shard, elem.shape[0])
                dense_shard = max(dense_chunk, _round_down(dense_shard, dense_chunk))
                dataset_kwargs = {
                    **dataset_kwargs,
                    "shards": (dense_shard,) + elem.shape[1:],  # only shard over 1st dim
                    "chunks": (dense_chunk,) + elem.shape[1:],  # only chunk over 1st dim
                    "compressors": compressors,
                }
            elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
                obs_per_shard = _shard_size_param_to_n_obs(shard_size, elem)
                nnz = elem.nnz
                if elem.shape[0] == 0:
                    raise ValueError(f"Cannot write sharded sparse matrix {elem_name!r} with 0 observations.")
                avg_nnz_per_obs = nnz / elem.shape[0]
                sparse_chunk = max(1, int(n_obs_per_chunk * avg_nnz_per_obs))
                sparse_chunk = min(sparse_chunk, nnz) if nnz > 0 else sparse_chunk
                sparse_shard = max(1, int(obs_per_shard * avg_nnz_per_obs))
                sparse_shard = min(sparse_shard, nnz) if nnz > 0 else sparse_shard
                sparse_shard = max(sparse_chunk, _round_down(sparse_shard, sparse_chunk))
                dataset_kwargs = {
                    **dataset_kwargs,
                    "shards": (sparse_shard,),
                    "chunks": (sparse_chunk,),
                    "compressors": compressors,
                }
            write_func(store, elem_name, elem, dataset_kwargs=dataset_kwargs)

        ad.experimental.write_dispatched(group, "/" if key is None else key, adata, callback=callback)
        zarr.consolidate_metadata(group.store)


def _estimate_bytes_per_obs_row(
    adata: ad.AnnData,
    backing: zarr.Group | h5py.Group,
) -> float:
    """Estimate uncompressed bytes per observation row from on-disk metadata.

    Uses the lazy-loaded *adata* to determine which array keys are present, then
    reads shapes and dtypes from *backing* (the on-disk h5py/zarr group) to
    compute the per-row byte budget without materialising any data.
    """
    n_obs = adata.shape[0]
    if n_obs == 0:
        return 0.0

    elem_paths: list[str] = []
    if adata.X is not None:
        elem_paths.append("X")
    for k in adata.layers.keys():
        elem_paths.append(f"layers/{k}")
    for k in adata.obsm.keys():
        elem_paths.append(f"obsm/{k}")
    elem_paths.append("obs")

    mean_bytes_per_row = 0.0
    for elem_path in elem_paths:
        if elem_path not in backing:
            raise KeyError(f"Could not find {elem_path} on AnnData object in backing store")
        node = backing[elem_path]
        encoding = dict(node.attrs).get("encoding-type", "")
        if encoding in {"csr_matrix", "csc_matrix"}:
            data, indices, indptr = node["data"], node["indices"], node["indptr"]
            mean_bytes_per_row += (
                data.shape[0] * (data.dtype.itemsize + indices.dtype.itemsize) + indptr.shape[0] * indptr.dtype.itemsize
            ) / n_obs
        elif encoding in {"array", ""}:
            mean_bytes_per_row += int(np.prod(node.shape[1:])) * node.dtype.itemsize
        elif encoding == "dataframe":
            for col_key in node:
                if col_key == "_index":
                    continue
                col_node = node[col_key]
                col_encoding = dict(col_node.attrs).get("encoding-type", "")
                if col_encoding == "categorical":
                    col_node = col_node["codes"]
                if hasattr(col_node, "shape") and hasattr(col_node, "dtype"):
                    mean_bytes_per_row += col_node.shape[0] * col_node.dtype.itemsize / n_obs
        elif encoding == "awkward-array":
            for buf_key in node:
                buf = node[buf_key]
                if hasattr(buf, "shape") and hasattr(buf, "dtype"):
                    mean_bytes_per_row += buf.shape[0] * buf.dtype.itemsize / n_obs
        else:
            raise ValueError(
                f"Unsupported encoding-type {encoding!r} for element {elem_path!r}. Cannot estimate per-row byte size."
            )

    return mean_bytes_per_row


def _validate_anndatas_and_maybe_get_bytes_per_row[T: zarr.Group | h5py.Group | PathLike[str] | str](
    paths_or_anndatas: Iterable[T | ad.AnnData],
    *,
    load_adata: Callable[[T], ad.AnnData] = lambda x: ad.experimental.read_lazy(x, load_annotation_index=False),
    estimate_bytes_per_obs_row: bool = False,
) -> float | None:
    """Validate that all datasets share the same keys and optionally estimate bytes per observation row.

    Parameters
    ----------
    paths_or_anndatas
        Paths or AnnData objects to validate.
    load_adata
        Function to lazy-load an AnnData from a path.
    estimate_bytes_per_obs_row
        If ``True``, estimate the average uncompressed bytes per observation row from the on-disk data.
        All entries must be paths or groups (not AnnData objects) in this case.

    Returns
    -------
    The average bytes per observation row when *estimate_bytes_per_obs_row* is ``True``, otherwise ``None``.
    """
    num_raw_in_adata = 0
    found_keys: dict[str, defaultdict[str, int]] = {
        "layers": defaultdict(lambda: 0),
        "obsm": defaultdict(lambda: 0),
        "obs": defaultdict(lambda: 0),
    }
    bytes_per_obs_samples: list[float] = []
    for path_or_anndata in tqdm(paths_or_anndatas, desc="Validating anndatas"):
        if not isinstance(path_or_anndata, ad.AnnData):
            adata = load_adata(path_or_anndata)
            if estimate_bytes_per_obs_row:
                if isinstance(path_or_anndata, zarr.Group | h5py.Group):
                    backing = path_or_anndata
                else:
                    p = Path(str(path_or_anndata))
                    backing = h5py.File(str(p), "r") if p.is_file() else zarr.open_group(str(p), mode="r")
                bytes_per_obs_samples.append(_estimate_bytes_per_obs_row(adata, backing=backing))
        else:
            if estimate_bytes_per_obs_row:
                raise NotImplementedError(
                    "Cannot estimate bytes per observation row from an AnnData object. "
                    "Provide file paths or groups instead, or pass an integer for dataset_size."
                )
            adata = path_or_anndata
        for elem_name, key_count in found_keys.items():
            curr_keys = set(getattr(adata, elem_name).keys())
            for key in curr_keys:
                if not (elem_name in {"var", "obs"} and key == "_index"):
                    key_count[key] += 1
        if adata.raw is not None:
            num_raw_in_adata += 1
    if num_raw_in_adata != (num_anndatas := len(list(paths_or_anndatas))) and num_raw_in_adata != 0:
        warnings.warn(
            f"Found raw keys not present in all anndatas {paths_or_anndatas}, consider deleting raw or moving it to a shared layer/X location via `load_adata`",
            stacklevel=2,
        )
    for elem_name, key_count in found_keys.items():
        elem_keys_mismatched = [key for key, count in key_count.items() if (count != num_anndatas and count != 0)]
        if len(elem_keys_mismatched) > 0:
            warnings.warn(
                f"Found {elem_name} keys {elem_keys_mismatched} not present in all anndatas {paths_or_anndatas}, consider stopping and using the `load_adata` argument to alter {elem_name} accordingly.",
                stacklevel=2,
            )
    return float(np.mean(bytes_per_obs_samples)) if bytes_per_obs_samples else None


def _lazy_load_adata[T: zarr.Group | h5py.Group | PathLike[str] | str](
    paths: Iterable[T],
    load_adata: Callable[[T], ad.AnnData] = _default_load_adata,
):
    adatas = []
    categoricals_in_all_adatas: dict[str, pd.Index] = {}
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Lazy loading anndatas"):
        adata = load_adata(path)
        # Track the source file for this given anndata object
        adata.obs["src_path"] = pd.Categorical.from_codes(
            np.ones((adata.shape[0],), dtype="int") * i, categories=pd.Index([str(p) for p in paths])
        )
        # Concatenating Dataset2D drops categoricals so we need to track them
        if isinstance(adata.obs, Dataset2D):
            categorical_cols_in_this_adata = {
                col: adata.obs[col].dtype.categories for col in adata.obs.columns if adata.obs[col].dtype == "category"
            }
            if not categoricals_in_all_adatas:
                categoricals_in_all_adatas = {
                    **categorical_cols_in_this_adata,
                    "src_path": adata.obs["src_path"].dtype.categories,
                }
            else:
                for k in categoricals_in_all_adatas.keys() & categorical_cols_in_this_adata.keys():
                    categoricals_in_all_adatas[k] = categoricals_in_all_adatas[k].union(
                        categorical_cols_in_this_adata[k]
                    )
        # TODO: Probably bug in anndata, need the true index for proper outer joins (can't skirt this with fake indexes, at least not in the mixed-type regime).
        # See: https://github.com/scverse/anndata/pull/2299
        if isinstance(adata.var, Dataset2D):
            adata.var.index = adata.var.true_index
        if adata.raw is not None and isinstance(adata.raw.var, Dataset2D):
            adata.raw.var.index = adata.raw.var.true_index
        adatas.append(adata)
    if len(adatas) == 1:
        return adatas[0]
    adata = ad.concat(adatas, join="outer")
    if len(categoricals_in_all_adatas) > 0:
        adata.uns["dataset2d_categoricals_to_convert"] = categoricals_in_all_adatas
    return adata


def _create_chunks_for_shuffling(
    n_obs: int,
    rng: np.random.Generator,
    shuffle_chunk_size: int = 1000,
    shuffle: bool = True,
    *,
    shuffle_n_obs_per_dataset: int | None = None,
    n_chunkings: int | None = None,
) -> list[np.ndarray]:
    # this splits the array up into `shuffle_chunk_size` contiguous runs
    idxs = split_given_size(np.arange(n_obs), shuffle_chunk_size)
    if shuffle:
        rng.shuffle(idxs)
    match shuffle_n_obs_per_dataset is not None, n_chunkings is not None:
        case True, False:
            n_slices_per_dataset = int(shuffle_n_obs_per_dataset // shuffle_chunk_size)
            use_single_chunking = n_obs <= shuffle_n_obs_per_dataset or n_slices_per_dataset <= 1
        case False, True:
            n_slices_per_dataset = (n_obs // n_chunkings) // shuffle_chunk_size
            use_single_chunking = n_chunkings == 1
        case _, _:
            raise ValueError("Cannot provide both shuffle_n_obs_per_dataset and n_chunkings or neither")
    # In this case `shuffle_n_obs_per_dataset` is bigger than the size of the dataset or the slice size is probably too big.
    if use_single_chunking:
        return [np.concatenate(idxs)]
    # unfortunately, this is the only way to prevent numpy.split from trying to np.array the idxs list, which can have uneven elements.
    idxs_as_slices = np.array([slice(int(idx[0]), int(idx[-1] + 1)) for idx in idxs])
    return [
        np.concatenate([np.arange(s.start, s.stop) for s in idx])
        for idx in (
            split_given_size(idxs_as_slices, n_slices_per_dataset)
            if n_chunkings is None
            else np.array_split(idxs_as_slices, n_chunkings)
        )
    ]


def _compute_blockwise(x: DaskArray) -> sp.spmatrix:
    """.compute() for large datasets is bad: https://github.com/scverse/annbatch/pull/75"""
    if isinstance(x._meta, sp.csr_matrix | sp.csr_array):
        return sp.vstack(da.compute(*list(x.blocks)))
    return x.compute()


def _to_categorical_obs(adata: ad.AnnData) -> ad.AnnData:
    """Convert columns marked as categorical in `uns` to categories, accounting for `concat` on `Dataset2D` lost dtypes"""
    if "dataset2d_categoricals_to_convert" in adata.uns:
        for col, categories in adata.uns["dataset2d_categoricals_to_convert"].items():
            adata.obs[col] = pd.Categorical(np.array(adata.obs[col]), categories=categories)
        del adata.uns["dataset2d_categoricals_to_convert"]
    return adata


def _persist_adata_in_memory(adata: ad.AnnData) -> ad.AnnData:
    if isinstance(adata.X, DaskArray):
        adata.X = _compute_blockwise(adata.X)
    if isinstance(adata.obs, Dataset2D):
        adata.obs = adata.obs.to_memory()
        # TODO: This is a bug in anndata?
        if "_index" in adata.obs.columns:
            adata.obs.index = adata.obs["_index"]
            del adata.obs["_index"]
    adata = _to_categorical_obs(adata)
    if isinstance(adata.var, Dataset2D):
        adata.var = adata.var.to_memory()
        if "_index" in adata.var.columns:
            del adata.var["_index"]

    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
        if isinstance(adata_raw.X, DaskArray):
            adata_raw.X = _compute_blockwise(adata_raw.X)
        if isinstance(adata_raw.var, Dataset2D):
            adata_raw.var = adata_raw.var.to_memory()
            if "_index" in adata_raw.var.columns:
                del adata_raw.var["_index"]
        if isinstance(adata_raw.obs, Dataset2D):
            adata_raw.obs = adata_raw.obs.to_memory()
        del adata.raw
        adata.raw = adata_raw

    for axis_name in ["layers", "obsm", "varm", "obsp", "varp"]:
        for k, elem in getattr(adata, axis_name).items():
            # TODO: handle `Dataset2D` in `obsm` and `varm` that are
            if isinstance(elem, DaskArray):
                getattr(adata, axis_name)[k] = _compute_blockwise(elem)
            if isinstance(elem, Dataset2D):
                elem = elem.to_memory()
                if "_index" in elem.columns:
                    del elem["_index"]
                # TODO: Bug in anndata
                if "obs" in axis_name:
                    elem.index = adata.obs_names
                getattr(adata, axis_name)[k] = elem

    return adata.to_memory()


DATASET_PREFIX = "dataset"


def _load_and_check(
    adata_paths: Iterable[zarr.Group | h5py.Group | PathLike[str] | str],
    *,
    load_adata: Callable[[zarr.Group | h5py.Group | PathLike[str] | str], ad.AnnData],
    var_subset: Iterable[str] | None,
) -> tuple[ad.AnnData, pd.Index]:
    """Validate keys, lazy-load, make unique, resolve var_subset."""
    _validate_anndatas_and_maybe_get_bytes_per_row(adata_paths, load_adata=load_adata)
    adata_concat = _lazy_load_adata(adata_paths, load_adata=load_adata)
    adata_concat.obs_names_make_unique()
    if var_subset is None:
        var_subset = adata_concat.var_names
    var_mask = adata_concat.var_names.isin(var_subset)
    return adata_concat, var_mask


def _with_settings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ad.settings.override(zarr_write_format=3, remove_unused_categories=False):
            return func(*args, **kwargs)

    return wrapper


class BaseCollection:
    """Base class providing shared initialization for collection types."""

    _group: zarr.Group | Path

    def __init__(
        self, group: zarr.Group | str | Path, *, mode: Literal["a", "r", "r+"] = "a", is_collection_h5ad: bool = False
    ):
        if is_collection_h5ad:
            if isinstance(group, zarr.Group):
                raise ValueError("Do not set `is_collection_h5ad` to True when also passing in a zarr Group.")
            warnings.warn(
                "Loading h5ad is currently not supported and thus we cannot guarantee the funcionality of the ecosystem with h5ad files."
                "DatasetCollection should be able to handle shuffling but we guarantee little else."
                "Proceed with caution.",
                stacklevel=2,
            )
            self._group = Path(group)
            self._group.mkdir(exist_ok=True)
        elif isinstance(group, zarr.Group):
            self._group = group
        elif isinstance(group, str | Path):
            if not str(group).endswith(".zarr"):
                warnings.warn(
                    f"It is highly recommended to make your collections have the `.zarr` suffix, got: {group}.",
                    stacklevel=2,
                )
            self._group = zarr.open_group(group, mode=mode)
        else:
            raise TypeError("Group must either be a zarr group or a path")


class DatasetCollection(BaseCollection):
    """A preshuffled collection object including functionality for creating, adding to, and loading collections shuffled by `annbatch`."""

    def __init__(
        self, group: zarr.Group | str | Path, *, mode: Literal["a", "r", "r+"] = "a", is_collection_h5ad: bool = False
    ):
        """Initialization of the object at a given location.

        Note that if the group is a h5py/zarr object, it must have the correct permissions for any subsequent operations you plan to do.
        Otherwise, the store will be opened according to the mode argument.


        Parameters
        ----------
            group
                The base location for a preshuffled collection.
                A :class:`zarr.Group` or path ending in `.zarr` indicates zarr as the shuffled format and otherwise a directory of `h5ad` files will be created.
        """
        super().__init__(group, mode=mode, is_collection_h5ad=is_collection_h5ad)

    @property
    def _dataset_keys(self) -> list[str]:
        if isinstance(self._group, zarr.Group):
            return sorted(
                [k for k in self._group.keys() if re.match(rf"{DATASET_PREFIX}_([0-9]*)", k) is not None],
                key=lambda x: int(x.split("_")[1]),
            )
        else:
            raise ValueError("Cannot iterate through folder of h5ad files")

    def __iter__(self) -> Generator[zarr.Group]:
        if isinstance(self._group, zarr.Group):
            for k in self._dataset_keys:
                yield self._group[k]
        else:
            raise ValueError("Cannot iterate through folder of h5ad files")

    @property
    def is_empty(self) -> bool:
        """Whether or not there is an existing store at the group location."""
        return (
            (not (V1_ENCODING.items() <= self._group.attrs.items()) or len(self._dataset_keys) == 0)
            if isinstance(self._group, zarr.Group)
            else (len(list(self._group.iterdir())) == 0)
        )

    @_with_settings
    def add_adatas(
        self,
        adata_paths: Iterable[zarr.Group | h5py.Group | PathLike[str] | str],
        *,
        load_adata: Callable[[zarr.Group | h5py.Group | PathLike[str] | str], ad.AnnData] = _default_load_adata,
        var_subset: Iterable[str] | None = None,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        dataset_size: int | str = "20GB",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Take AnnData paths and create or add to an on-disk set of AnnData datasets with uniform var spaces at the desired path (with `dataset_size` rows per dataset if running for the first time).

        The set of AnnData datasets is collectively referred to as a "collection" where each dataset is called `dataset_i{.h5ad}`.
        The main purpose of this function is to create shuffled sharded zarr datasets, which is the default behavior of this function.
        However, this function can also output h5 datasets and also unshuffled datasets as well.
        The var space is by default outer-joined initially, and then subsequently added datasets (i.e., on second calls to this function) are subsetted, but this behavior can be controlled by `var_subset`.
        A key `src_path` is added to `obs` to indicate where individual row came from.
        We highly recommend making your indexes unique across files, and this function will call `AnnData.obs_names_make_unique`.
        Memory usage should be controlled by `dataset_size` + `shuffle_chunk_size` as so many rows will be read into memory before writing to disk.
        After the dataset completes, a marker is added to the group's `attrs` to note that this dataset has been shuffled by `annbatch`.
        This is only for internal purposes at the moment so that we can recognize datasets that have been shuffled by an instance of this class.

        Parameters
        ----------
            adata_paths
                Paths to the AnnData files used to create the zarr store.
            load_adata
                Function to customize (lazy-)loading the invidiual input anndata files. By default, :func:`anndata.experimental.read_lazy` is used with categoricals/nullables read into memory and `(-1)` chunks for `obs`.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and certain `obs` columns), you can provide a custom function here to speed up loading and harmonize your data.
                Beware that concatenating nullables/categoricals (i.e., what happens if `len(adata_paths) > 1` internally in this function) from {class}`anndata.experimental.backed.Dataset2D` `obs` is very time consuming - consider loading these into memory if you use this argument.
            var_subset
                Subset of gene names to include in the store. If None, all genes are included.
                Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                String sizes get parsed using the humanfriendly package.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            h5ad_compressor
                Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
            dataset_size
                Number of observations to load into memory at once for shuffling / pre-processing, or a size string (e.g. ``'2GB'``, ``'512MB'``).
                When a size string is provided, the observation count is derived from the estimated uncompressed bytes per row of the input data.
                String sizes get parsed using the humanfriendly package.
                The higher this number, the more memory is used, but the better the shuffling.
                This corresponds to the size of the dataset level shards created.
                Only applicable when adding datasets for the first time, otherwise ignored.
            shuffle
                Whether to shuffle the data before writing it to the store.
                Ignored once the store is non-empty.
            shuffle_chunk_size
                How many contiguous rows to load into memory before shuffling at once.
                `(shuffle_chunk_size // dataset_size)` slices will be loaded of size `shuffle_chunk_size`.
            rng
                Random number generator for shuffling.

        Examples
        --------
            >>> import anndata as ad
            >>> from annbatch import DatasetCollection
            # create a custom load function to only keep `.X`, `.obs` and `.var` in the output store
            >>> def read_lazy_x_and_obs_only(path):
            ...     adata = ad.experimental.read_lazy(path)
            ...     return ad.AnnData(
            ...         X=adata.X,
            ...         obs=adata.obs.to_memory(),
            ...         var=adata.var.to_memory(),
            ...)
            >>> datasets = [
            ...     "path/to/first_adata.h5ad",
            ...     "path/to/second_adata.h5ad",
            ...     "path/to/third_adata.h5ad",
            ... ]
            >>> DatasetCollection("path/to/output/zarr_store.zarr").add_adatas(
            ...    datasets,
            ...    load_adata=read_lazy_x_and_obs_only,
            ...)
        """
        if rng is None:
            rng = np.random.default_rng()
        shared_kwargs = {
            "adata_paths": adata_paths,
            "load_adata": load_adata,
            "n_obs_per_chunk": n_obs_per_chunk,
            "shard_size": shard_size,
            "zarr_compressor": zarr_compressor,
            "h5ad_compressor": h5ad_compressor,
            "shuffle_chunk_size": shuffle_chunk_size,
            "shuffle": shuffle,
            "rng": rng,
        }
        if self.is_empty:
            self._create_collection(**shared_kwargs, dataset_size=dataset_size, var_subset=var_subset)
        else:
            self._add_to_collection(**shared_kwargs)
        return self

    def _create_collection(
        self,
        *,
        adata_paths: Iterable[PathLike[str]] | Iterable[str],
        load_adata: Callable[[PathLike[str] | str], ad.AnnData] = _default_load_adata,
        var_subset: Iterable[str] | None = None,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        dataset_size: int | str = "20GB",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator,
    ) -> None:
        """Take AnnData paths, create an on-disk set of AnnData datasets with uniform var spaces at the desired path with `dataset_size` rows per dataset.

        The set of AnnData datasets is collectively referred to as a "collection" where each dataset is called `dataset_i.{zarr,h5ad}`.
        The main purpose of this function is to create shuffled sharded zarr datasets, which is the default behavior of this function.
        However, this function can also output h5 datasets and also unshuffled datasets as well.
        The var space is by default outer-joined, but can be subsetted by `var_subset`.
        A key `src_path` is added to `obs` to indicate where individual row came from.
        We highly recommend making your indexes unique across files, and this function will call `AnnData.obs_names_make_unique`.
        Memory usage should be controlled by `dataset_size` as so many rows will be read into memory before writing to disk.

        Parameters
        ----------
            adata_paths
                Paths to the AnnData files used to create the zarr store.
            load_adata
                Function to customize lazy-loading the invidiual input anndata files. By default, :func:`anndata.experimental.read_lazy` is used.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and `obs`), you can provide a custom function here to speed up loading and harmonize your data.
                The input to the function is a path to an anndata file, and the output is an anndata object which has `X` as a :class:`dask.array.Array`.
            var_subset
                Subset of gene names to include in the store. If None, all genes are included.
                Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
                Only applicable when adding datasets for the first time, otherwise ignored and the incoming data's var space is subsetted to that of the existing collection.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            h5ad_compressor
                Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
            dataset_size
                Number of observations to load into memory at once for shuffling / pre-processing, or a size string (e.g. ``'2GB'``, ``'512MB'``).
                When a size string is provided, the observation count is derived from the estimated uncompressed bytes per row of the input data.
                The higher this number, the more memory is used, but the better the shuffling.
                This corresponds to the size of the shards created.
                Only applicable when adding datasets for the first time, otherwise ignored.
            shuffle
                Whether to shuffle the data before writing it to the store.
            shuffle_chunk_size
                How many contiguous rows to load into memory before shuffling at once.
                `(shuffle_chunk_size // dataset_size)` slices will be loaded of size `shuffle_chunk_size`.
            rng
                Random number generator for shuffling.
        """
        if not self.is_empty:
            raise RuntimeError("Cannot create a collection at a location that already has a shuffled collection")
        needs_estimate = isinstance(dataset_size, str)
        estimated_bytes_per_row = _validate_anndatas_and_maybe_get_bytes_per_row(
            adata_paths, load_adata=load_adata, estimate_bytes_per_obs_row=needs_estimate
        )

        if needs_estimate:
            target_bytes = parse_size(dataset_size, binary=True)
            dataset_size = max(1, int(target_bytes / estimated_bytes_per_row))

        if shuffle_chunk_size > dataset_size:
            raise ValueError(
                "Cannot have a larger slice size than observations per dataset. Reduce `shuffle_chunk_size` or increase `dataset_size`."
            )

        adata_concat = _lazy_load_adata(adata_paths, load_adata=load_adata)
        adata_concat.obs_names_make_unique()
        dataset_size = min(adata_concat.shape[0], dataset_size)
        chunks = _create_chunks_for_shuffling(
            adata_concat.shape[0],
            rng=rng,
            shuffle_chunk_size=shuffle_chunk_size,
            shuffle=shuffle,
            shuffle_n_obs_per_dataset=dataset_size,
        )

        if var_subset is None:
            var_subset = adata_concat.var_names
        for i, chunk in enumerate(tqdm(chunks, desc="Creating dataset collection")):
            var_mask = adata_concat.var_names.isin(var_subset)
            # np.sort: It's more efficient to access elements sequentially from dask arrays
            # The data will be shuffled later on, we just want the elements at this point
            adata_chunk = adata_concat[np.sort(chunk), :][:, var_mask].copy()
            adata_chunk = _persist_adata_in_memory(adata_chunk)
            if shuffle:
                # shuffle adata in memory to break up individual chunks
                idxs = rng.permutation(np.arange(len(adata_chunk)))
                adata_chunk = adata_chunk[idxs]
            if isinstance(self._group, zarr.Group):
                write_sharded(
                    self._group,
                    adata_chunk,
                    n_obs_per_chunk=min(n_obs_per_chunk, adata_chunk.shape[0]),
                    shard_size=shard_size,
                    compressors=zarr_compressor,
                    key=f"{DATASET_PREFIX}_{i}",
                )
            else:
                ad.io.write_h5ad(
                    self._group / f"{DATASET_PREFIX}_{i}.h5ad",
                    adata_chunk,
                    dataset_kwargs={"compression": h5ad_compressor},
                )
        if isinstance(self._group, zarr.Group):
            self._group.update_attributes(V1_ENCODING)

    def _add_to_collection(
        self,
        *,
        adata_paths: Iterable[PathLike[str]] | Iterable[str],
        load_adata: Callable[[PathLike[str] | str], ad.AnnData] = ad.read_h5ad,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator,
    ) -> None:
        """Add anndata files to an existing collection of sharded anndata zarr datasets.

        The var space of the source anndata files will be adapted to the target store.

        Parameters
        ----------
            adata_paths
                Paths to the anndata files to be appended to the collection of output chunks.
            rng
                Random number generator for shuffling.
            load_adata
                Function to customize loading the invidiual input anndata files. By default, :func:`anndata.read_h5ad` is used.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and `obs`), you can provide a custom function here to speed up loading and harmonize your data.
                The input to the function is a path to an anndata file, and the output is an anndata object.
                If the input data is too large to fit into memory, you should use :func:`annndata.experimental.read_lazy` instead.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            should_sparsify_output_in_memory
                This option is for testing only appending sparse files to dense stores.
                To save memory, the blocks of a dense on-disk store can be sparsified for in-memory processing.
            shuffle_chunk_size
                How many contiguous rows to load into memory of the input data for pseudo-blockshuffling into the existing datasets.
            shuffle
                Whether or not to shuffle when adding.  Otherwise, the incoming data will just be split up and appended.
        """
        if self.is_empty:
            raise ValueError("Store is empty. Please run `DatasetCollection.add_adatas` first.")
        # Check for mismatched keys among the inputs.
        adata_concat = _lazy_load_adata(adata_paths, load_adata=load_adata)
        if math.ceil(adata_concat.shape[0] / shuffle_chunk_size) < len(self._dataset_keys):
            raise ValueError(
                f"Use a shuffle size small enough to distribute the input data with {adata_concat.shape[0]} obs across {len(self._dataset_keys)} anndata stores."
                "Open an issue if the incoming anndata is so small it cannot be distributed across the on-disk data"
            )
        # Check for mismatched keys between datasets and the inputs.
        _validate_anndatas_and_maybe_get_bytes_per_row([adata_concat] + [self._group[k] for k in self._dataset_keys])
        chunks = _create_chunks_for_shuffling(
            adata_concat.shape[0],
            rng=rng,
            shuffle_chunk_size=shuffle_chunk_size,
            shuffle=shuffle,
            n_chunkings=len(self._dataset_keys),
        )

        adata_concat.obs_names_make_unique()
        for dataset, chunk in tqdm(
            zip(self._dataset_keys, chunks, strict=True),
            total=len(self._dataset_keys),
            desc="Extending dataset collection",
        ):
            adata_dataset = ad.io.read_elem(self._group[dataset])
            subset_adata = _to_categorical_obs(
                adata_concat[chunk, :][:, adata_concat.var.index.isin(adata_dataset.var.index)]
            )
            adata = ad.concat([adata_dataset, subset_adata], join="outer")
            if shuffle:
                idxs = rng.permutation(adata.shape[0])
            else:
                idxs = np.arange(adata.shape[0])
            adata = _persist_adata_in_memory(adata[idxs, :].copy())
            if isinstance(self._group, zarr.Group):
                write_sharded(
                    self._group,
                    adata,
                    n_obs_per_chunk=min(n_obs_per_chunk, adata.shape[0]),
                    shard_size=shard_size,
                    compressors=zarr_compressor,
                    key=dataset,
                )
            else:
                ad.io.write_h5ad(
                    self._group / f"{dataset}.h5ad",
                    adata,
                    dataset_kwargs={"compression": h5ad_compressor},
                )


_SCAN_BLOCK_SIZE = 500_000


def _sequential_scan_and_write(
    *,
    adata_concat: ad.AnnData,
    obs_full: pd.DataFrame,
    var_mask: np.ndarray,
    chunks: list[np.ndarray],
    group: zarr.Group | Path,
    n_obs_per_chunk: int,
    zarr_shard_size: int | str,
    zarr_compressor,
    h5ad_compressor,
    scan_block_size: int = _SCAN_BLOCK_SIZE,
    max_buffered_rows: int | None = None,
) -> None:
    """Read source data in one sequential pass and scatter rows into per-chunk buffers.

    Instead of indexing into the lazy-backed AnnData once per output chunk
    (which causes random I/O on the source file), this reads the source in
    contiguous blocks and dispatches each row to the correct output chunk.
    When all rows for a chunk have been collected the chunk is written to
    disk and its buffer is freed.

    This turns O(n_chunks) random-access passes over the source file into a
    single sequential scan -- critical for large h5ad files on network
    filesystems where random I/O is orders of magnitude slower.

    Each chunk buffer collects ``(position, AnnData-slice)`` fragments.
    When a chunk is complete the fragments are concatenated in position
    order, producing an AnnData identical to what the old random-access
    code path would have built.

    Parameters
    ----------
    max_buffered_rows
        Maximum total rows held in fragment buffers across all chunks
        before an incomplete-chunk flush is triggered.  When exceeded,
        the chunk with the most accumulated rows is flushed early by
        re-reading its missing rows from the source via random access.
        Defaults to ``2 * max(chunk_sizes)`` which guarantees at most
        ~2 chunks' worth of data in memory at any time.
    """
    n_obs = adata_concat.shape[0]
    n_chunks = len(chunks)

    row_to_chunk = np.empty(n_obs, dtype=np.int64)
    row_to_pos = np.empty(n_obs, dtype=np.int64)
    for cid, chunk_indices in enumerate(chunks):
        row_to_chunk[chunk_indices] = cid
        row_to_pos[chunk_indices] = np.arange(len(chunk_indices), dtype=np.int64)

    chunk_sizes = np.array([len(c) for c in chunks], dtype=np.int64)
    chunk_filled = np.zeros(n_chunks, dtype=np.int64)

    if max_buffered_rows is None:
        max_buffered_rows = int(2 * chunk_sizes.max())
    total_buffered_rows = 0

    chunk_fragments: list[list[tuple[int, ad.AnnData]]] = [[] for _ in range(n_chunks)]
    chunk_done = np.zeros(n_chunks, dtype=bool)

    var_out = adata_concat.var
    if isinstance(var_out, Dataset2D):
        var_out = var_out.to_memory()
    var_out = var_out[var_mask]

    def _flush_chunk(cid: int) -> None:
        """Assemble fragments in position order, write, and free."""
        frags = chunk_fragments[cid]
        frags.sort(key=lambda t: t[0])
        adata_out = ad.concat([f for _, f in frags], join="outer")
        adata_out.var = var_out.copy()

        key = f"{DATASET_PREFIX}_{cid}"
        if isinstance(group, zarr.Group):
            write_sharded(
                group,
                adata_out,
                n_obs_per_chunk=n_obs_per_chunk,
                shard_size=zarr_shard_size,
                compressors=zarr_compressor,
                key=key,
            )
        else:
            ad.io.write_h5ad(
                group / f"{key}.h5ad",
                adata_out,
                dataset_kwargs={"compression": h5ad_compressor},
            )
        chunk_fragments[cid] = []
        chunk_done[cid] = True

    def _force_flush_largest() -> int:
        """Flush the incomplete chunk with the most buffered rows.

        The missing rows are fetched via random access from adata_concat,
        which is slower but keeps memory bounded.  Returns number of
        rows freed.
        """
        buffered_per_chunk = np.array([
            int(chunk_filled[c]) if not chunk_done[c] else 0
            for c in range(n_chunks)
        ], dtype=np.int64)
        cid = int(np.argmax(buffered_per_chunk))
        rows_held = int(buffered_per_chunk[cid])
        if rows_held == 0:
            return 0

        have_positions = set()
        for pos_min, frag in chunk_fragments[cid]:
            have_positions.update(range(pos_min, pos_min + frag.n_obs))

        chunk_indices = chunks[cid]
        need_positions = sorted(set(range(len(chunk_indices))) - have_positions)
        if need_positions:
            source_rows = chunk_indices[np.array(need_positions, dtype=np.int64)]
            missing_adata = _materialize_rows(source_rows)
            chunk_fragments[cid].append((int(min(need_positions)), missing_adata))
            chunk_filled[cid] = chunk_sizes[cid]

        _flush_chunk(cid)
        return rows_held

    def _materialize_block(block_slice: slice) -> ad.AnnData:
        """Read a contiguous block from the source, materializing X but using pre-loaded obs."""
        block_adata = adata_concat[block_slice, :][:, var_mask].copy()
        block_adata = _persist_adata_in_memory(block_adata)
        block_adata.obs = obs_full.iloc[block_slice.start : block_slice.stop].copy()
        block_adata.obs.index = block_adata.obs_names
        return block_adata

    def _materialize_rows(source_rows: np.ndarray) -> ad.AnnData:
        """Read specific rows from the source (random access fallback)."""
        sort_order = np.argsort(source_rows)
        adata_subset = adata_concat[source_rows[sort_order], :][:, var_mask].copy()
        adata_subset = _persist_adata_in_memory(adata_subset)
        unsort_order = np.argsort(sort_order)
        adata_subset = adata_subset[unsort_order]
        adata_subset.obs = obs_full.iloc[source_rows].copy()
        adata_subset.obs.index = adata_subset.obs_names
        return adata_subset

    for block_start in tqdm(range(0, n_obs, scan_block_size), desc="sequential scan"):
        block_end = min(block_start + scan_block_size, n_obs)
        block_adata = _materialize_block(slice(block_start, block_end))

        global_rows = np.arange(block_start, block_end, dtype=np.int64)
        cids = row_to_chunk[global_rows]
        positions = row_to_pos[global_rows]

        for cid_val in np.unique(cids):
            cid = int(cid_val)
            if chunk_done[cid]:
                continue

            mask = cids == cid_val
            local_idx = np.where(mask)[0]
            pos = positions[mask]

            frag = block_adata[local_idx].copy()
            chunk_fragments[cid].append((int(pos.min()), frag))
            n_new = len(local_idx)
            chunk_filled[cid] += n_new
            total_buffered_rows += n_new

            if chunk_filled[cid] == chunk_sizes[cid]:
                _flush_chunk(cid)
                total_buffered_rows -= int(chunk_sizes[cid])

        while total_buffered_rows > max_buffered_rows:
            freed = _force_flush_largest()
            if freed == 0:
                break
            total_buffered_rows -= freed

    for cid in range(n_chunks):
        if chunk_done[cid]:
            continue
        if chunk_filled[cid] < chunk_sizes[cid]:
            chunk_indices = chunks[cid]
            have_positions = set()
            for pos_min, frag in chunk_fragments[cid]:
                have_positions.update(range(pos_min, pos_min + frag.n_obs))
            need_positions = sorted(set(range(len(chunk_indices))) - have_positions)
            if need_positions:
                source_rows = chunk_indices[np.array(need_positions, dtype=np.int64)]
                missing_adata = _materialize_rows(source_rows)
                chunk_fragments[cid].append((int(min(need_positions)), missing_adata))
        if chunk_fragments[cid]:
            _flush_chunk(cid)


def _group_obs_rows(
    obs: pd.DataFrame,
    *,
    groupby: list[str],
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reorder observation indices so that rows are contiguous by group, and return a group index."""
    g = obs[groupby].groupby(groupby, dropna=False, sort=True, observed=False)
    group_ids = g.ngroup().to_numpy(dtype=np.int64)

    if shuffle:
        order = np.lexsort((rng.random(len(obs)), group_ids))
    else:
        order = np.argsort(group_ids, kind="stable")

    ordered_positions = np.arange(len(obs), dtype=np.int64)[order]

    group_index = g.size().reset_index(name="count")
    counts = group_index["count"].to_numpy(dtype=np.int64)
    stops = np.cumsum(counts)
    group_index["stop"] = stops
    group_index["start"] = stops - counts

    return ordered_positions, group_index


def _split_positions_by_dataset_groupby(
    ordered_positions: np.ndarray,
    group_index: pd.DataFrame,
    *,
    dataset_groupby_keys: list[str],
) -> list[np.ndarray]:
    """Split already-ordered positions into one chunk per unique ``dataset_groupby_keys`` combination.

    ``group_index`` is produced by ``_group_obs_rows`` and has ``start``/``stop``
    columns describing contiguous ranges in ``ordered_positions``.  Groups that
    share the same ``dataset_groupby_keys`` values are consecutive (because
    ``_group_obs_rows`` sorts by all ``groupby`` columns and
    ``dataset_groupby_keys`` is a prefix), so we merge adjacent ranges.
    """
    dataset_groups = group_index.groupby(dataset_groupby_keys, dropna=False, sort=True, observed=False)
    chunks: list[np.ndarray] = []
    for _, df in dataset_groups:
        start = int(df["start"].iloc[0])
        stop = int(df["stop"].iloc[-1])
        chunks.append(ordered_positions[start:stop])
    return chunks


class GroupedCollection(BaseCollection):
    """A grouped zarr collection where data is written sequentially with group boundaries stored as metadata.

    Observations are reordered so that each group is contiguous on disk.
    A ``group_index`` DataFrame records the ``start``/``stop``/``count``
    per group, allowing a :class:`~annbatch.CategoricalSampler` to be
    constructed via :meth:`~annbatch.CategoricalSampler.from_collection`.
    """

    @property
    def _dataset_keys(self) -> list[str]:
        if isinstance(self._group, zarr.Group):
            return sorted(
                [k for k in self._group.keys() if re.fullmatch(rf"{DATASET_PREFIX}_([0-9]+)", k) is not None],
                key=lambda x: int(x.split("_")[1]),
            )
        raise ValueError("Cannot list dataset keys for a folder-based collection")

    @property
    def is_empty(self) -> bool:
        return not (V1_GROUPED_ENCODING.items() <= self._group.attrs.items()) or len(self._dataset_keys) == 0

    def __iter__(self) -> Generator[zarr.Group]:
        for k in self._dataset_keys:
            yield self._group[k]

    @property
    def group_index(self) -> pd.DataFrame:
        """The group boundary metadata for this collection."""
        if GROUP_INDEX_KEY not in self._group:
            raise ValueError("Grouped collection is missing `group_index` metadata.")
        return ad.io.read_elem(self._group[GROUP_INDEX_KEY])

    @_with_settings
    def add_adatas(
        self,
        adata_paths: Iterable[zarr.Group | h5py.Group | PathLike[str] | str],
        *,
        groupby: str | Iterable[str],
        dataset_groupby: str | Iterable[str] | None = None,
        load_adata: Callable[[zarr.Group | h5py.Group | PathLike[str] | str], ad.AnnData] = _default_load_adata,
        var_subset: Iterable[str] | None = None,
        n_obs_per_chunk: int = 64,
        zarr_shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        n_obs_per_dataset: int = 2_097_152,
        shuffle: bool = True,
        random_seed: int | None = None,
    ) -> Self:
        """Create a grouped collection from AnnData paths.

        Observations are reordered so that each group is contiguous, then
        written sequentially as a flat series of datasets.  A
        ``group_index`` DataFrame is persisted as metadata to record per-group
        boundaries.

        Parameters
        ----------
        groupby
            One or more obs column names to group by.  The full set of
            columns defines the finest-grained groups whose boundaries
            are recorded in ``group_index``.
        dataset_groupby
            A subset (prefix) of ``groupby`` columns that determines on-disk
            dataset boundaries.  Each unique combination of these columns
            becomes its own ``dataset_i`` on disk instead of splitting by
            ``n_obs_per_dataset``.

            For example, with ``groupby=["cell_line", "drug"]``:

            * ``dataset_groupby=None`` (default) -- datasets are split
              by ``n_obs_per_dataset`` (current behavior).
            * ``dataset_groupby="cell_line"`` -- one dataset per cell
              line; within each dataset, observations are still
              contiguous by ``(cell_line, drug)`` pair.
            * ``dataset_groupby=["cell_line", "drug"]`` -- one dataset
              per ``(cell_line, drug)`` combination.

            When set, ``n_obs_per_dataset`` is ignored.
        n_obs_per_dataset
            Maximum number of observations per dataset.  Ignored when
            ``dataset_groupby`` is set.
        shuffle
            Whether to shuffle observations within each group.
        random_seed
            Seed for the random number generator used when ``shuffle=True``.
        """
        if not shuffle and random_seed is not None:
            raise ValueError("`random_seed` must be None when `shuffle` is False.")
        if not self.is_empty:
            raise RuntimeError("Cannot create a grouped collection at a non-empty location.")
        groupby_keys = [groupby] if isinstance(groupby, str) else list(groupby)
        if len(groupby_keys) == 0:
            raise ValueError("`groupby` must contain at least one obs column name.")
        if len(set(groupby_keys)) != len(groupby_keys):
            raise ValueError("`groupby` must not contain duplicate column names.")

        if dataset_groupby is not None:
            dataset_groupby_keys = [dataset_groupby] if isinstance(dataset_groupby, str) else list(dataset_groupby)
            if len(dataset_groupby_keys) == 0:
                raise ValueError("`dataset_groupby` must contain at least one obs column name when set.")
            if not all(k in groupby_keys for k in dataset_groupby_keys):
                raise ValueError(
                    f"`dataset_groupby` columns {dataset_groupby_keys} must be a subset of "
                    f"`groupby` columns {groupby_keys}."
                )
            if dataset_groupby_keys != groupby_keys[: len(dataset_groupby_keys)]:
                raise ValueError(
                    f"`dataset_groupby` columns {dataset_groupby_keys} must be a prefix of "
                    f"`groupby` columns {groupby_keys} to preserve contiguous group ordering."
                )
        else:
            dataset_groupby_keys = None

        adata_concat, var_mask = _load_and_check(adata_paths, load_adata=load_adata, var_subset=var_subset)
        missing_group_keys = [k for k in groupby_keys if k not in adata_concat.obs.columns]
        if len(missing_group_keys) > 0:
            raise ValueError(f"Could not find groupby key(s) in obs: {missing_group_keys}.")

        # Materialize obs once -- it is small enough to fit in memory
        # and avoids re-reading it from the backing store on every
        # scan block / chunk.
        obs_full = adata_concat.obs
        if isinstance(obs_full, Dataset2D):
            obs_full = obs_full.to_memory()
            if "_index" in obs_full.columns:
                obs_full.index = obs_full["_index"]
                obs_full = obs_full.drop(columns=["_index"])
        obs_full = obs_full.copy()

        obs_for_grouping = obs_full[groupby_keys]
        rng = np.random.default_rng(random_seed)
        ordered_positions, group_index = _group_obs_rows(
            obs_for_grouping,
            groupby=groupby_keys,
            shuffle=shuffle,
            rng=rng,
        )

        if dataset_groupby_keys is not None:
            chunks = _split_positions_by_dataset_groupby(
                ordered_positions, group_index, dataset_groupby_keys=dataset_groupby_keys
            )
        else:
            chunks = split_given_size(ordered_positions, n_obs_per_dataset)

        _sequential_scan_and_write(
            adata_concat=adata_concat,
            obs_full=obs_full,
            var_mask=var_mask,
            chunks=chunks,
            group=self._group,
            n_obs_per_chunk=n_obs_per_chunk,
            zarr_shard_size=zarr_shard_size,
            zarr_compressor=zarr_compressor,
            h5ad_compressor=h5ad_compressor,
        )

        ad.io.write_elem(self._group, GROUP_INDEX_KEY, group_index)
        self._group.update_attributes({**V1_GROUPED_ENCODING, "groupby_keys": groupby_keys})
        return self
