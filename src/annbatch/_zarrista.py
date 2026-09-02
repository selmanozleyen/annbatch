"""Read the bulk CSR arrays through `zarrista` instead of `zarr-python`.

`zarrista` (developmentseed/zarrista) is a separate low-level binding to the Rust `zarrs`
crate. It exposes zarrs' own array API rather than plugging into zarr-python's codec
pipeline, so it cannot be selected with `zarr.config.set` -- it has to be called directly.

Only the two BULK arrays go through it: `data` and `indices`. Everything else -- opening the
AnnData, validating shapes, reading `indptr` into memory -- stays on anndata and zarr-python,
because none of it is on the hot path and all of it is where the two libraries differ most.

What this costs today, and why it is worth measuring anyway
-----------------------------------------------------------
zarr-python's `_get_selection(out=...)` decodes STRAIGHT INTO the caller's buffer. zarrista
has no equivalent: every read allocates, and `retrieve_array_subset` hands back a `Tensor`
that owns its bytes. `Tensor.to_numpy()` is a zero-copy view of those bytes (verified:
`owndata` is False and repeated calls return the same pointer), so the extra cost is exactly
ONE memcpy per run -- from zarrista's allocation into annbatch's preallocated buffer.

That copy is the price of the missing `out=`. Measuring it is the point: it says what an
upstream `retrieve_array_subset_into` would be worth before anyone writes one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import zarrista


class ZarristaArray:
    """A zarrista `Array` that reports a numpy dtype.

    The loader sizes its output buffer from `data.dtype` and `indices.dtype`. zarrista answers
    with its own `DataType`, which `np.empty` cannot interpret -- so the arm converts once,
    here, rather than teaching the loader a second dtype vocabulary. `DataType.name` is the
    zarr dtype string, which numpy already speaks.
    """

    __slots__ = ("_array", "dtype")

    def __init__(self, array) -> None:  # noqa: ANN001
        self._array = array
        self.dtype = np.dtype(array.dtype.name)

    def retrieve_array_subset(self, selection):  # noqa: ANN001, ANN201
        return self._array.retrieve_array_subset(selection)


class ZarristaCSRElems(NamedTuple):
    """The `CSRDatasetElems` shape, with the bulk arrays opened by zarrista.

    `indptr` is an ordinary in-memory array, exactly as on the zarr-python path -- it is read
    once per dataset and indexed per batch, so it never touches either library's read path.
    `indices` and `data` are zarrista arrays, which is what makes this a different arm rather
    than a different spelling of the same one.
    """

    indptr: np.ndarray
    indices: ZarristaArray
    data: ZarristaArray


def store_root(group) -> Path:  # noqa: ANN001
    """The filesystem path behind a zarr group, which is what zarrista needs to open it.

    zarrista takes a store and a path, not a zarr object; there is no adapter between the two
    libraries' store abstractions, so the only thing they can share is a location on disk.
    Anything that is not a `LocalStore` cannot be handed over at all -- and saying so here is
    better than a `NoneType has no attribute` three frames down.
    """
    store = group.store
    root = getattr(store, "root", None)
    if root is None:
        raise TypeError(
            f"zarrista can only open a local filesystem store, got {type(store).__name__}. "
            "The two libraries share no store abstraction, only a path."
        )
    return root


async def open_csr_elems(group, indptr: np.ndarray) -> ZarristaCSRElems:
    """Open one CSR group's `data` and `indices` with zarrista.

    `indptr` is passed in rather than read here: the caller already has it from the
    zarr-python path, and re-reading it would make the two arms differ in setup as well as in
    the thing under test.
    """
    import zarrista
    from zarrista.store import FilesystemStore

    root = store_root(group)
    # zarrista wants the store root and a path within it; a zarr group knows its own name
    # relative to that root.
    prefix = group.name.strip("/")
    store = FilesystemStore(str(root))

    # The SYNC array, deliberately. zarrista's async API takes only an obstore `ObjectStore`,
    # an icechunk `Session` or an `AsyncZipStore` -- there is no async filesystem store, so
    # the async path for local data means adding obstore and reading through an object-store
    # abstraction. The sync methods release the GIL (`crate::py::detach`), so running them on
    # a thread gives real parallelism against the native filesystem store instead.
    data = ZarristaArray(zarrista.Array.open(store, f"/{prefix}/data" if prefix else "/data"))
    indices = ZarristaArray(
        zarrista.Array.open(store, f"/{prefix}/indices" if prefix else "/indices")
    )
    return ZarristaCSRElems(indptr=indptr, indices=indices, data=data)


def read_runs_into(arr, limits: list[slice], out: np.ndarray) -> None:
    """Read each 1-D slice of `arr` and lay the results end to end in `out`.

    This is the shape the CSR path always has: scattered ROWS become contiguous RANGES in the
    flat `data`/`indices` arrays, because `indptr` says where each row's values live. So even
    a fully scattered row draw asks zarrista only for contiguous subsets, which is the one
    thing its API is good at.

    The copy here is the missing `out=`. `to_numpy()` does not copy -- it views zarrista's own
    allocation -- so this is one memcpy of exactly the bytes wanted, not two.
    """
    at = 0
    for limit in limits:
        tensor = arr.retrieve_array_subset(limit)
        chunk = tensor.to_numpy()
        out[at : at + chunk.size] = chunk
        at += chunk.size
    if at != out.size:
        raise ValueError(f"read {at} elements into a buffer of {out.size}")


async def read_runs_into_async(arr, limits: list[slice], out: np.ndarray) -> None:
    """`read_runs_into` on a worker thread, so two of these actually overlap.

    annbatch gathers the `data` and `indices` reads, and on the zarr-python arm those are
    genuine coroutines that interleave. Calling a blocking function inside a coroutine would
    serialise them and the arm would look slow for a reason that is not zarrista. zarrista's
    sync methods release the GIL, so a thread is a real thread here rather than a queue.
    """
    import asyncio

    await asyncio.to_thread(read_runs_into, arr, limits, out)
