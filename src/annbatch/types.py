from __future__ import annotations

from typing import NotRequired, TypedDict

import anndata as ad
import numpy as np
import pandas as pd  # noqa: TC002
from scipy import sparse as sp
from zarr import Array as ZarrArray

from .compat import CupyArray, CupyCSRMatrix, Tensor
from .utils import CSRContainer

type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray | Tensor


class LoadRequest(TypedDict):
    """Load request from sampler.

    This is the request format Loader will expect from the sampler.
    Not satisfying the constrains documented here may result in unexpected behavior.

    Since chunk_size is uniform, chunks are represented compactly as starting
    offsets plus the chunk_size. Only when the very last chunk of the dataset
    is incomplete does ``remainder`` differ from 0, giving its actual size.

    Attributes
    ----------
    chunk_size
        The uniform size of every chunk in this request.
    starts
        1-D array of chunk starting offsets.
    remainder
        Size of the last chunk when it is smaller than ``chunk_size``.
        0 means all chunks (including the last) are full-sized.
    splits
        How the in-memory data should be split into batches after it is read off disk and concatenated in-memory.
        A list of splits, last one may be partial but not empty i.e. 1 <= len(last_split) <= batch_size.
        If not provided, the sampler's batch_size property will be used to automatically generate splits.

    Notes
    -----
    **In-memory data ordering**: When chunks span multiple datasets, the loader groups and fetches
    chunks by dataset index for efficiency. The resulting in-memory data is ordered by dataset index,
    not by the original order of starts in the ``starts`` array. Within each dataset, chunks maintain
    their relative order from the original array.

    For example, given two datasets (dataset 0: rows 0-99, dataset 1: rows 100-199) and starts
    ``[100, 0, 110]`` with chunk_size=10, the in-memory data will be ordered as:
    ``[rows 0-10 from ds0, rows 100-110 from ds1, rows 110-120 from ds1]`` i.e., sorted by dataset index.

    The ``splits`` indices must account for this ordering. For a single dataset, the in-memory order
    naturally matches the chunk order since there is only one dataset to fetch from.

    """

    chunk_size: int
    starts: np.ndarray
    remainder: int
    splits: NotRequired[list[np.ndarray]]


def load_request_stops(lr: LoadRequest) -> np.ndarray:
    """Compute the stop offset for every chunk described by a LoadRequest."""
    starts = lr["starts"]
    chunk_size = lr["chunk_size"]
    remainder = lr["remainder"]
    stops = starts + chunk_size
    if remainder:
        stops[-1] = starts[-1] + remainder
    return stops


def load_request_to_slices(lr: LoadRequest) -> list[slice]:
    """Expand a LoadRequest into a list of slices for indexing."""
    stops = load_request_stops(lr)
    return [slice(int(s), int(e)) for s, e in zip(lr["starts"], stops, strict=True)]


def load_request_total_obs(lr: LoadRequest) -> int:
    """Total number of observations described by a LoadRequest."""
    n = len(lr["starts"])
    if n == 0:
        return 0
    remainder = lr["remainder"]
    if remainder == 0:
        return n * lr["chunk_size"]
    return (n - 1) * lr["chunk_size"] + remainder


def _multi_arange(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Vectorized multi-range: equivalent to np.concatenate([np.arange(a,b) for a,b in zip(starts,stops)])."""
    lengths = stops - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=starts.dtype)
    ones = np.ones(total, dtype=starts.dtype)
    ones[0] = starts[0]
    if len(starts) > 1:
        cumlen = np.cumsum(lengths[:-1])
        ones[cumlen] = starts[1:] - stops[:-1] + 1
    return np.cumsum(ones)


class LoaderOutput[OutputInMemoryArray: OutputInMemoryArray_T](TypedDict):
    """The output of the loader, the "data matrix" with its obs, optional, var, optional, and index, also optional."""

    X: OutputInMemoryArray_T.__value__  # TODO: remove after sphinx 9 - myst compat
    obs: pd.DataFrame | None
    var: pd.DataFrame | None
    index: np.ndarray | None
