"""Convenience wrappers for the zarrs_direct Rust extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarr_direct._zarrs_direct import ShardedArrayReader

if TYPE_CHECKING:
    import zarr

_reader_cache: dict[tuple[str, bool], ShardedArrayReader] = {}


def _get_reader(store_root: str, use_mmap: bool) -> ShardedArrayReader:
    key = (store_root, use_mmap)
    reader = _reader_cache.get(key)
    if reader is None:
        reader = ShardedArrayReader(store_root, use_mmap=use_mmap)
        _reader_cache[key] = reader
    return reader


def _parse_store_path(arr: zarr.Array) -> tuple[str, str]:
    """Extract the filesystem store root and array node path."""
    store = arr.store
    if not isinstance(store, __import__("zarr").storage.LocalStore):
        raise TypeError("only LocalStore is supported")
    store_root = str(store.root)
    array_path = "/" + arr.store_path.path if arr.store_path.path else "/"
    return store_root, array_path


def read_dense(
    arr: zarr.Array,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    use_mmap: bool = True,
) -> np.ndarray:
    """Read axis-0 row ranges from a sharded zarr array.

    Parameters
    ----------
    arr
        A zarr v3 Array backed by LocalStore with ShardingCodec.
    starts
        Start indices for each range along axis 0.
    stops
        Stop indices for each range along axis 0.
    use_mmap
        If True use mmap store (zero-copy), else use pread store.

    Returns
    -------
    np.ndarray
        Dense array with all requested rows concatenated.
    """
    store_root, array_path = _parse_store_path(arr)
    shape = tuple(arr.shape)
    dtype = arr.dtype

    reader = _get_reader(store_root, use_mmap)
    raw = reader.read_raw(
        array_path,
        np.ascontiguousarray(starts, dtype=np.int64),
        np.ascontiguousarray(stops, dtype=np.int64),
    )

    total_rows = int((stops - starts).sum())
    out_shape = (total_rows, *shape[1:])
    return np.frombuffer(raw, dtype=dtype).reshape(out_shape)


def read_1d(
    arr: zarr.Array,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    use_mmap: bool = True,
) -> np.ndarray:
    """Read ranges from a 1-D sharded zarr array (e.g. CSR data/indices).

    Parameters
    ----------
    arr
        A zarr v3 1-D Array backed by LocalStore with ShardingCodec.
    starts
        Start indices for each range.
    stops
        Stop indices for each range.
    use_mmap
        If True use mmap store (zero-copy), else use pread store.

    Returns
    -------
    np.ndarray
        1-D array of concatenated elements.
    """
    store_root, array_path = _parse_store_path(arr)
    dtype = arr.dtype

    reader = _get_reader(store_root, use_mmap)
    raw = reader.read_raw(
        array_path,
        np.ascontiguousarray(starts, dtype=np.int64),
        np.ascontiguousarray(stops, dtype=np.int64),
    )
    return np.frombuffer(raw, dtype=dtype)
