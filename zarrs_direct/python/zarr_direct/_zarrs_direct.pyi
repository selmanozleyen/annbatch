import numpy as np
import numpy.typing as npt

class ShardedArrayReader:
    def __init__(self, store_path: str, use_mmap: bool = True, fuse_ranges: bool = True) -> None: ...
    def read_raw(
        self,
        array_path: str,
        starts: npt.NDArray[np.int64],
        stops: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.uint8]: ...
    def array_shape(self, array_path: str) -> list[int]: ...
