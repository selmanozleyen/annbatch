"""Direct shard reader that bypasses zarr's async machinery.

Uses mmap for zero-copy file access and an optional C extension for the
hot inner loop (shard index parse + blosc decompress + memcpy).  Falls
back to a pure-Python path when the C extension is unavailable.

Only supports the codec chain produced by ``write_sharded``:
``ShardingCodec(BytesCodec + BloscCodec, index_codecs=BytesCodec + Crc32cCodec)``.
"""

from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import zarr
import zarr.core.chunk_grids

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Resolve blosc C library and (optionally) our C extension
# ---------------------------------------------------------------------------
def _find_blosc_lib() -> ctypes.CDLL:
    nc_dir = os.path.dirname(numcodecs.blosc.__file__)
    for f in os.listdir(nc_dir):
        if f.startswith("blosc.") and (f.endswith(".so") or f.endswith(".pyd") or ".dylib" in f):
            lib = ctypes.CDLL(os.path.join(nc_dir, f))
            lib.blosc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.blosc_decompress.restype = ctypes.c_int
            return lib
    raise RuntimeError("Cannot find blosc shared library in numcodecs")

_blosc_lib = _find_blosc_lib()

try:
    from annbatch._shard_reader import _init_blosc, read_into_dense, read_into_1d
    _addr = ctypes.cast(_blosc_lib.blosc_decompress, ctypes.c_void_p).value
    _init_blosc(_addr)
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False


# ---------------------------------------------------------------------------
# Metadata parsing (shared by all paths)
# ---------------------------------------------------------------------------
def _parse_sharding_metadata(
    arr: zarr.Array,
) -> tuple[
    Path,             # disk_root
    tuple[int, ...],  # array shape
    tuple[int, ...],  # shard shape
    tuple[int, ...],  # inner chunk shape
    tuple[int, ...],  # chunks_per_shard
    int,              # shard_index_size (incl. crc32c)
    np.dtype,         # dtype
    numcodecs.Blosc,  # blosc codec (for Python fallback)
]:
    m = arr.metadata
    store = arr.store
    if not isinstance(store, zarr.storage.LocalStore):
        raise TypeError("direct_read only supports LocalStore")

    disk_root = store.root / arr.store_path.path
    shape = tuple(m.shape)
    shard_shape = tuple(m.chunk_grid.chunk_shape)
    sharding_codec = m.codecs[0]
    inner_chunk_shape = tuple(sharding_codec.chunk_shape)
    chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, inner_chunk_shape))

    n_inner = 1
    for d in chunks_per_shard:
        n_inner *= d
    shard_index_size = 16 * n_inner + 4  # BytesCodec + Crc32cCodec

    dtype = m.data_type.to_native_dtype()

    blosc_codec = None
    for c in sharding_codec.codecs:
        if isinstance(c, zarr.codecs.blosc.BloscCodec):
            blosc_codec = c._blosc_codec
            break
    if blosc_codec is None:
        raise ValueError("No BloscCodec found in sharding inner codecs")

    return disk_root, shape, shard_shape, inner_chunk_shape, chunks_per_shard, shard_index_size, dtype, blosc_codec


# ---------------------------------------------------------------------------
# mmap cache
# ---------------------------------------------------------------------------
class _MmapCache:
    """Cache of mmap'd shard files + parsed shard indices."""

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, tuple[mmap.mmap, np.ndarray, np.ndarray]] = {}

    def get(
        self, shard_path: Path, shard_index_size: int, chunks_per_shard: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (shard_np_view, shard_index) for the given shard file."""
        key = str(shard_path)
        if key not in self._cache:
            fd = os.open(key, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            shard_np = np.frombuffer(mm, dtype=np.uint8)
            idx_raw = shard_np[-shard_index_size:-4]
            shard_index = idx_raw.view("<u8").reshape(chunks_per_shard + (2,))
            self._cache[key] = (mm, shard_np, shard_index)
        return self._cache[key][1], self._cache[key][2]

    def clear(self) -> None:
        for mm, _, _ in self._cache.values():
            try:
                mm.close()
            except BufferError:
                pass
        self._cache.clear()


# ---------------------------------------------------------------------------
# Dense reader
# ---------------------------------------------------------------------------
def read_direct_dense(arr: zarr.Array, boundaries: np.ndarray) -> np.ndarray:
    """Read axis-0 ranges from a sharded zarr array, bypassing async machinery.

    Parameters
    ----------
    arr
        A zarr v3 ``Array`` backed by ``LocalStore`` with ``ShardingCodec``.
    boundaries
        Interleaved ``[s0, e0, s1, e1, ...]`` int array of axis-0 ranges.

    Returns
    -------
        Dense ndarray with all requested rows concatenated.
    """
    (
        disk_root, shape, shard_shape, inner_chunk_shape,
        chunks_per_shard, shard_index_size, dtype, blosc_codec,
    ) = _parse_sharding_metadata(arr)

    starts = boundaries[::2]
    stops = boundaries[1::2]
    total_rows = (stops - starts).sum()
    ndim = len(shape)
    out = np.empty((total_rows, *shape[1:]), dtype=dtype)

    inner_row_size = inner_chunk_shape[0]
    shard_row_size = shard_shape[0]
    inner_chunk_nbytes = np.prod(inner_chunk_shape) * dtype.itemsize
    row_nbytes = np.prod(shape[1:]) * dtype.itemsize if ndim > 1 else dtype.itemsize
    ndim_extra = ndim - 1
    # Flat stride between consecutive axis-0 chunks in the shard index.
    # Index shape is (*chunks_per_shard, 2); axis-0 stride in the
    # flattened view = product(chunks_per_shard[1:]) * 2.
    idx_stride = np.prod(chunks_per_shard[1:]) * 2 if ndim > 1 else 2

    cache = _MmapCache()
    try:
        out_offset = 0
        for i in range(len(starts)):
            sel_start, sel_stop = starts[i], stops[i]
            row = sel_start

            while row < sel_stop:
                shard_row_idx = row // shard_row_size
                shard_key = (shard_row_idx,) + (0,) * ndim_extra
                parts = "/".join(str(k) for k in shard_key)
                shard_path = disk_root / "c" / parts

                shard_np, shard_index = cache.get(shard_path, shard_index_size, chunks_per_shard)
                shard_base = shard_row_idx * shard_row_size
                # Flatten the index once for this shard
                idx_flat = shard_index.ravel()

                if _HAS_C_EXT:
                    rows_written, row = read_into_dense(
                        shard_np, idx_flat, out,
                        inner_chunk_nbytes, inner_row_size, shard_row_size,
                        shard_base, row, sel_stop, out_offset,
                        row_nbytes, idx_stride,
                    )
                    out_offset += rows_written
                else:
                    row, out_offset = _dense_python_fallback(
                        shard_np, idx_flat, out, blosc_codec, dtype,
                        inner_chunk_shape, inner_row_size, shard_row_size,
                        shard_base, row, sel_stop, out_offset, idx_stride,
                    )
    finally:
        cache.clear()

    return out


def _dense_python_fallback(
    shard_np, idx_flat, out, blosc_codec, dtype,
    inner_chunk_shape, inner_row_size, shard_row_size,
    shard_base, row, sel_stop, out_offset, idx_stride,
):
    """Pure-Python fallback for dense shard reading."""
    while row < sel_stop and row < shard_base + shard_row_size:
        inner_idx = (row - shard_base) // inner_row_size
        inner_base = shard_base + inner_idx * inner_row_size
        inner_end = inner_base + inner_row_size

        flat_idx = inner_idx * idx_stride
        offset_val = idx_flat[flat_idx]
        length_val = idx_flat[flat_idx + 1]

        take_start = max(row, inner_base) - inner_base
        take_end = min(sel_stop, inner_end) - inner_base
        n_take = take_end - take_start

        chunk_bytes = bytes(shard_np[offset_val:offset_val + length_val])
        decoded = blosc_codec.decode(chunk_bytes)
        chunk_arr = np.frombuffer(decoded, dtype=dtype).reshape(inner_chunk_shape)

        out[out_offset:out_offset + n_take] = chunk_arr[take_start:take_end]
        out_offset += n_take
        row = inner_base + take_end

    return row, out_offset


# ---------------------------------------------------------------------------
# 1-D reader (for CSR data/indices arrays)
# ---------------------------------------------------------------------------
def read_direct_1d(arr: zarr.Array, boundaries: np.ndarray) -> np.ndarray:
    """Read ranges from a 1-D sharded zarr array (e.g. CSR data/indices).

    Same as ``read_direct_dense`` but specialized for 1-D arrays.
    """
    if isinstance(arr, zarr.AsyncArray):
        arr = zarr.Array(arr)

    (
        disk_root, shape, shard_shape, inner_chunk_shape,
        chunks_per_shard, shard_index_size, dtype, blosc_codec,
    ) = _parse_sharding_metadata(arr)

    starts = boundaries[::2]
    stops = boundaries[1::2]
    total = (stops - starts).sum()
    out = np.empty(total, dtype=dtype)

    inner_size = inner_chunk_shape[0]
    shard_size = shard_shape[0]
    inner_chunk_nbytes = inner_size * dtype.itemsize
    elem_nbytes = dtype.itemsize

    cache = _MmapCache()
    try:
        out_offset = 0
        for i in range(len(starts)):
            sel_start, sel_stop = starts[i], stops[i]
            pos = sel_start

            while pos < sel_stop:
                shard_idx = pos // shard_size
                shard_path = disk_root / "c" / str(shard_idx)

                shard_np, shard_index = cache.get(shard_path, shard_index_size, chunks_per_shard)
                shard_base = shard_idx * shard_size

                if _HAS_C_EXT:
                    elems_written, pos = read_into_1d(
                        shard_np, shard_index, out,
                        inner_chunk_nbytes, inner_size, shard_size,
                        shard_base, pos, sel_stop, out_offset,
                        elem_nbytes,
                    )
                    out_offset += elems_written
                else:
                    pos, out_offset = _1d_python_fallback(
                        shard_np, shard_index, out, blosc_codec, dtype,
                        inner_size, shard_size,
                        shard_base, pos, sel_stop, out_offset,
                    )
    finally:
        cache.clear()

    return out


def _1d_python_fallback(
    shard_np, shard_index, out, blosc_codec, dtype,
    inner_size, shard_size,
    shard_base, pos, sel_stop, out_offset,
):
    """Pure-Python fallback for 1-D shard reading."""
    while pos < sel_stop and pos < shard_base + shard_size:
        inner_idx = (pos - shard_base) // inner_size
        inner_base = shard_base + inner_idx * inner_size
        inner_end = inner_base + inner_size

        offset_val = shard_index[(inner_idx, 0)]
        length_val = shard_index[(inner_idx, 1)]

        take_start = max(pos, inner_base) - inner_base
        take_end = min(sel_stop, inner_end) - inner_base
        n_take = take_end - take_start

        chunk_bytes = bytes(shard_np[offset_val:offset_val + length_val])
        decoded = blosc_codec.decode(chunk_bytes)
        chunk_arr = np.frombuffer(decoded, dtype=dtype)

        out[out_offset:out_offset + n_take] = chunk_arr[take_start:take_end]
        out_offset += n_take
        pos = inner_base + take_end

    return pos, out_offset
