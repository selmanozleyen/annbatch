from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, Protocol

import anndata as ad
import numpy as np
import pandas as pd
import scipy as sp
import zarr

from .compat import CupyArray, CupyCSRMatrix, Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from annbatch.loader import Loader
    from annbatch.types import OutputInMemoryArray_T


def validate_sampler[**Param, RetType](
    method: Callable[Concatenate[Loader, Param], RetType],
) -> Callable[Concatenate[Loader, Param], RetType]:
    """Decorator that validates n_obs before modifying state.

    Expects the first positional argument to be either:
    - A single object with a `.shape` attribute
    - A list of objects with `.shape` attributes

    The total n_obs is computed as sum of shape[0] values for a list of objects or the shape[0] value for a single object.
    """
    sig = inspect.signature(method)
    if len(sig.parameters) < 2:
        raise ValueError("validate_sampler decorator expects at least two positional arguments after 'self'")
    first_param_name = list(sig.parameters.keys())[1]

    @wraps(method)
    def wrapper(self: Loader, *args: Param.args, **kwargs: Param.kwargs) -> RetType:
        if len(args) > 0:
            first_arg = args[0]
        else:
            first_arg = kwargs[first_param_name]

        n_obs = sum(item.shape[0] for item in first_arg) if isinstance(first_arg, list) else first_arg.shape[0]
        self.batch_sampler.validate(n_obs)
        return method(self, *args, **kwargs)

    return wrapper


def split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    """Wrapper around `np.split` to split up an array into `size` chunks"""
    return np.split(a, np.arange(size, len(a), size))


@dataclass
class CSRContainer:
    """A low-cost container for moving around the buffers of a CSR object"""

    elems: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int]
    dtype: np.dtype


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
# Another "solution" would be for zarrs to support integer indexing properly, if that pipeline works,
# or make this an "experimental setting" and to use integer indexing for the zarr-python pipeline.
# See: https://github.com/zarr-developers/zarr-python/issues/3175 for why this is better than simpler alternatives.
class MultiBasicIndexer(zarr.core.indexing.Indexer):
    """Custom indexer to enable joint fetching of disparate slices.

    Two construction modes:

    1. ``MultiBasicIndexer(indexers)`` -- list of pre-built indexers
       (used by the sparse path where indptr-derived slices are irregular).
    2. ``MultiBasicIndexer.from_boundaries(boundaries, shape, chunk_grid)``
       -- accepts the interleaved boundaries array and computes
       ``ChunkProjection`` tuples directly in ``__iter__``, completely
       bypassing ``BasicIndexer`` / ``SliceDimIndexer`` construction.
    """

    def __init__(self, indexers: list[zarr.core.indexing.Indexer]):
        self.shape = (sum(i.shape[0] for i in indexers), *indexers[0].shape[1:])
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers
        self._boundaries = None

    @classmethod
    def from_boundaries(
        cls,
        boundaries: np.ndarray,
        shape: tuple[int, ...],
        chunk_grid: zarr.core.chunk_grids.ChunkGrid,
    ) -> MultiBasicIndexer:
        """Build from an interleaved ``[s0, e0, s1, e1, ...]`` array.

        Iteration produces ``ChunkProjection`` tuples via direct arithmetic
        on the chunk grid -- no ``slice``, ``BasicIndexer``, or
        ``SliceDimIndexer`` objects are ever created for the axis-0
        dimension.
        """
        starts, stops = boundaries[::2], boundaries[1::2]
        total_rows = (stops - starts).sum()
        inst = cls.__new__(cls)
        inst._boundaries = boundaries
        inst._arr_shape = shape
        inst._chunk_grid = chunk_grid
        inst.shape = (total_rows, *shape[1:])
        inst.drop_axes = ()
        inst.indexers = None
        return inst

    def __iter__(self):
        total = 0
        if self._boundaries is not None:
            yield from self._iter_from_boundaries()
        else:
            for i in self.indexers:
                for c in i:
                    out_selection = c[2]
                    gap = out_selection[0].stop - out_selection[0].start
                    yield type(c)(c[0], c[1], (slice(total, total + gap), *out_selection[1:]), c[3])
                    total += gap

    def _iter_from_boundaries(self):
        """Yield ChunkProjection tuples for axis-0 range accesses.

        Reproduces the logic of SliceDimIndexer.__iter__ (step=1 only)
        inlined into a tight loop over the interleaved boundaries array.
        Remaining dimensions are assumed to be full-chunk (Ellipsis).
        """
        boundaries = self._boundaries
        arr_shape = self._arr_shape
        chunk_grid = self._chunk_grid

        chunk_shape = zarr.core.indexing.get_chunk_shape(chunk_grid)
        dim0_chunk_len = chunk_shape[0]
        dim0_len = arr_shape[0]
        ndim = len(arr_shape)

        # Pre-build the "full dim" selections for dims 1..N-1
        # (these are the same for every projection and reused)
        if ndim > 1:
            tail_chunk_coords = []
            tail_chunk_sels = []
            tail_out_sels = []
            tail_is_complete = []
            for d in range(1, ndim):
                tail_chunk_coords.append(0)
                tail_chunk_sels.append(slice(0, chunk_shape[d], 1))
                tail_out_sels.append(slice(0, arr_shape[d]))
                tail_is_complete.append(True)
            tail_chunk_coords_t = tuple(tail_chunk_coords)
            tail_chunk_sels_t = tuple(tail_chunk_sels)
            tail_out_sels_t = tuple(tail_out_sels)
            tail_all_complete = all(tail_is_complete)
        else:
            tail_chunk_coords_t = ()
            tail_chunk_sels_t = ()
            tail_out_sels_t = ()
            tail_all_complete = True

        CP = zarr.core.indexing.ChunkProjection
        out_offset = 0

        for i in range(0, len(boundaries), 2):
            sel_start = boundaries[i]
            sel_stop = boundaries[i + 1]
            if sel_start >= sel_stop:
                continue

            cix_from = sel_start // dim0_chunk_len
            cix_to = -(-sel_stop // dim0_chunk_len)  # ceildiv

            for cix in range(cix_from, cix_to):
                dim_offset = cix * dim0_chunk_len
                dim_limit = min(dim0_len, dim_offset + dim0_chunk_len)

                cs_start = max(sel_start - dim_offset, 0)
                cs_stop = min(sel_stop - dim_offset, dim_limit - dim_offset)
                nitems = cs_stop - cs_start
                if nitems <= 0:
                    continue

                is_complete = cs_start == 0 and sel_stop >= dim_limit

                yield CP(
                    (cix, *tail_chunk_coords_t),
                    (slice(cs_start, cs_stop, 1), *tail_chunk_sels_t),
                    (slice(out_offset, out_offset + nitems), *tail_out_sels_t),
                    is_complete and tail_all_complete,
                )
                out_offset += nitems


def _spawn_worker_rng(rng: np.random.Generator, worker_id: int) -> np.random.Generator:
    """Create a worker-specific RNG using the sequence-of-integers seeding pattern.

    Uses NumPy's recommended approach for multi-process RNG. See:
    https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds
    """
    root_seed = rng.integers(np.iinfo(np.int64).max)
    return np.random.default_rng([worker_id, root_seed])


def check_lt_1(vals: list[int], obs: list[str]) -> None:
    """Raise a ValueError if any of the values are less than one.

    The format of the error is "{obs[i]} must be greater than 1, got {values[i]}"
    and is raised based on the first found less than one value.

    Parameters
    ----------
        vals
            The values to check < 1
        obs
            The label for the value in the error if the value is less than one.

    Raises
    ------
        ValueError: _description_
    """
    if any(is_lt_1 := [v < 1 for v in vals]):
        label, value = next(
            (label, value)
            for label, value, check in zip(
                obs,
                vals,
                is_lt_1,
                strict=True,
            )
            if check
        )
        raise ValueError(f"{label} must be greater than 1, got {value}")


class SupportsShape(Protocol):  # noqa: D101
    @property
    def shape(self) -> tuple[int, int] | list[int]: ...  # noqa: D102


def check_var_shapes(objs: list[SupportsShape]) -> None:
    """Small utility function to check that all objects have the same shape along the second axis"""
    if not all(objs[0].shape[1] == d.shape[1] for d in objs):
        raise ValueError("TODO: All datasets must have same shape along the var axis.")


def to_torch(input: OutputInMemoryArray_T, preload_to_gpu: bool) -> Tensor:
    """Send the input data to a torch.Tensor"""
    import torch

    if isinstance(input, torch.Tensor):
        return input
    if isinstance(input, sp.sparse.csr_matrix):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
            tensor = torch.sparse_csr_tensor(
                torch.from_numpy(input.indptr),
                torch.from_numpy(input.indices),
                torch.from_numpy(input.data),
                input.shape,
            )
        if preload_to_gpu:
            return tensor.cuda(non_blocking=True)
        return tensor
    if isinstance(input, np.ndarray):
        tensor = torch.from_numpy(input)
        if preload_to_gpu:
            return tensor.cuda(non_blocking=True)
        return tensor
    if isinstance(input, CupyArray):
        return torch.from_dlpack(input)
    if isinstance(input, CupyCSRMatrix):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
            return torch.sparse_csr_tensor(
                torch.from_dlpack(input.indptr),
                torch.from_dlpack(input.indices),
                torch.from_dlpack(input.data),
                input.shape,
            )
    raise TypeError(f"Cannot convert {type(input)} to torch.Tensor")


def load_x_and_obs_and_var(g: zarr.Group) -> ad.AnnData:
    """Load X as a sparse array or dense zarr array and obs from a group"""
    return ad.AnnData(
        X=g["X"] if isinstance(g["X"], zarr.Array) else ad.io.sparse_dataset(g["X"]),
        obs=ad.io.read_elem(g["obs"]),
        var=pd.DataFrame(
            index=pd.Index(g[f"var/{g['var'].attrs.get('_index')}"][:], name=g["var"].attrs.get("_index"))
        ),
    )
