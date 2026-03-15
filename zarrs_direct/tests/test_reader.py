"""Correctness tests for zarr_direct against zarr-python.

Requires: pip install zarr-direct (maturin develop), zarr, numpy
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from zarr_direct import read_dense, read_1d, ShardedArrayReader


def _create_dense_store(path: Path, n_rows: int = 100, n_cols: int = 20) -> zarr.Array:
    """Create a small sharded float32 array and return an open handle."""
    store = zarr.storage.LocalStore(path)
    g = zarr.open_group(store, mode="w", zarr_format=3)

    arr = g.create_array(
        "X",
        shape=(n_rows, n_cols),
        dtype="float32",
        chunks=(10, n_cols),
        shards=(50, n_cols),
        compressors=(BloscCodec(cname="lz4", clevel=3),),
    )
    data = np.arange(n_rows * n_cols, dtype="float32").reshape(n_rows, n_cols)
    arr[:] = data
    return zarr.open(store)["X"]


def _create_1d_store(path: Path, n_elems: int = 200) -> zarr.Array:
    """Create a 1D sharded float32 array."""
    store = zarr.storage.LocalStore(path)
    g = zarr.open_group(store, mode="w", zarr_format=3)

    arr = g.create_array(
        "data",
        shape=(n_elems,),
        dtype="float32",
        chunks=(20,),
        shards=(100,),
        compressors=(BloscCodec(cname="lz4", clevel=3),),
    )
    data = np.arange(n_elems, dtype="float32")
    arr[:] = data
    return zarr.open(store)["data"]


class TestReadDense:
    def test_single_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_dense_store(Path(tmp))
            starts = np.array([0], dtype=np.int64)
            stops = np.array([10], dtype=np.int64)

            result = read_dense(arr, starts, stops)
            expected = arr[0:10]

            np.testing.assert_array_equal(result, expected)

    def test_multiple_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_dense_store(Path(tmp))
            starts = np.array([0, 50, 90], dtype=np.int64)
            stops = np.array([5, 60, 100], dtype=np.int64)

            result = read_dense(arr, starts, stops)

            expected = np.concatenate([arr[0:5], arr[50:60], arr[90:100]])
            np.testing.assert_array_equal(result, expected)

    def test_pread_mode_raises(self):
        import pytest
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_dense_store(Path(tmp))
            starts = np.array([10], dtype=np.int64)
            stops = np.array([30], dtype=np.int64)

            read_dense(arr, starts, stops, use_mmap=True)
            with pytest.raises(ValueError, match="use_mmap=True"):
                read_dense(arr, starts, stops, use_mmap=False)

    def test_matches_zarr_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_dense_store(Path(tmp))
            starts = np.array([5, 40], dtype=np.int64)
            stops = np.array([15, 50], dtype=np.int64)

            result = read_dense(arr, starts, stops)
            expected = np.concatenate([arr[5:15], arr[40:50]])
            np.testing.assert_array_equal(result, expected)


class TestRead1D:
    def test_single_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_1d_store(Path(tmp))
            starts = np.array([0], dtype=np.int64)
            stops = np.array([20], dtype=np.int64)

            result = read_1d(arr, starts, stops)
            expected = arr[0:20]

            np.testing.assert_array_equal(result, expected)

    def test_multiple_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_1d_store(Path(tmp))
            starts = np.array([10, 150], dtype=np.int64)
            stops = np.array([30, 170], dtype=np.int64)

            result = read_1d(arr, starts, stops)
            expected = np.concatenate([arr[10:30], arr[150:170]])
            np.testing.assert_array_equal(result, expected)


class TestShardedArrayReader:
    def test_low_level_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = _create_dense_store(Path(tmp))
            store_root = str(arr.store.root)

            reader = ShardedArrayReader(store_root, use_mmap=True)
            shape = reader.array_shape("/X")
            assert shape == [100, 20]

            raw = reader.read_raw(
                "/X",
                np.array([0], dtype=np.int64),
                np.array([10], dtype=np.int64),
            )
            result = np.frombuffer(raw, dtype="float32").reshape(10, 20)
            expected = arr[0:10]
            np.testing.assert_array_equal(result, expected)
