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

    Attributes
    ----------
    starts
        1-D int array of chunk starting offsets.
    stops
        1-D int array of chunk stop offsets (same length as ``starts``).
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
    ``[100, 0, 110]`` with stops ``[110, 10, 120]``, the in-memory data will be ordered as:
    ``[rows 0-10 from ds0, rows 100-110 from ds1, rows 110-120 from ds1]`` i.e., sorted by dataset index.

    The ``splits`` indices must account for this ordering. For a single dataset, the in-memory order
    naturally matches the chunk order since there is only one dataset to fetch from.

    """

    starts: np.ndarray
    stops: np.ndarray
    splits: NotRequired[list[np.ndarray]]


def load_request_total_obs(lr: LoadRequest) -> int:
    """Total number of observations described by a LoadRequest."""
    return int((lr["stops"] - lr["starts"]).sum())


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
