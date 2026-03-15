"""Fast synchronous Zarr v3 sharded array reader."""

from zarr_direct.reader import read_dense, read_1d
from zarr_direct._zarrs_direct import ShardedArrayReader

__all__ = ["read_dense", "read_1d", "ShardedArrayReader"]
