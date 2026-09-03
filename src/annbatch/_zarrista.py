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

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
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
        name = array.dtype.name
        if name is None:
            # `DataType.name` is `str | None`, and `np.dtype(None)` is float64 -- a buffer
            # sized from that would be silently wrong rather than absent.
            raise TypeError(f"zarrista dtype {array.dtype} has no numpy spelling")
        self.dtype = np.dtype(name)

    def retrieve_array_subset(self, selection):  # noqa: ANN001, ANN201
        """Delegate to the wrapped zarrista array."""
        return self._array.retrieve_array_subset(selection)

    @property
    def effective_subchunk_shape(self):  # noqa: ANN201
        """The inner chunk shape when sharded, else None."""
        return self._array.effective_subchunk_shape

    def chunk_shape(self, indices):  # noqa: ANN001, ANN201
        """The chunk shape at `indices`."""
        return self._array.chunk_shape(indices)


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
    # On a thread, like the reads: `Array.open` blocks, and `_ensure_sparse_cache` gathers
    # one of these per plate expecting them to interleave. Fourteen plates is 28 serialised
    # metadata round trips on Lustre if they do not.
    def _open(name: str):  # noqa: ANN202
        return ZarristaArray(zarrista.Array.open(store, f"/{prefix}/{name}" if prefix else f"/{name}"))

    data, indices = await asyncio.gather(
        asyncio.to_thread(_open, "data"), asyncio.to_thread(_open, "indices")
    )
    return ZarristaCSRElems(indptr=indptr, indices=indices, data=data)


def decode_unit(array) -> int:  # noqa: ANN001
    """How many elements one decode covers, which is what runs can share.

    A sharded array decodes an INNER chunk, not the shard, so that is the unit two runs can
    share; an unsharded one decodes the chunk. Asked of the array rather than assumed, because
    getting it wrong in either direction only shows up as a slow arm.
    """
    subchunk = array.effective_subchunk_shape
    if subchunk is not None:
        return int(subchunk[0])
    return int(array.chunk_shape([0])[0])


def coalesce(runs: list[tuple[int, slice]], unit: int) -> list[tuple[slice, list[tuple[int, slice]]]]:
    """Group runs that land in the same decode unit into one read each.

    zarrista has no equivalent of zarr-python's `MultiBasicIndexer`: every
    `retrieve_array_subset` is its own call, and a call decodes every unit it touches with no
    reuse across calls. So N short runs inside one chunk cost N decodes of that chunk --
    measured at ~0.5 ms per call once runs are smaller than a chunk, dead linear in the call
    count. Merging costs nothing: a merged range touches exactly the units its members already
    needed, so no extra unit is ever decoded.

    `runs` is `(output_offset, slice)`, and the offsets are what make this safe for input in
    ANY order. Rows within a dataset are NOT sorted -- `_group_rows` orders by which dataset a
    row belongs to, not by the row -- so the runs arrive shuffled, and a merged range that
    assumed ascending input produced a negative index into its own result and silently read
    nothing.
    """
    ordered = sorted((r for r in runs if r[1].stop > r[1].start), key=lambda r: r[1].start)
    groups: list[tuple[slice, list[tuple[int, slice]]]] = []
    for offset, limit in ordered:
        if groups:
            merged, members = groups[-1]
            if limit.start // unit <= (merged.stop - 1) // unit:
                groups[-1] = (
                    slice(merged.start, max(merged.stop, limit.stop)),
                    [*members, (offset, limit)],
                )
                continue
        groups.append((limit, [(offset, limit)]))
    return groups


#: How many zarrista calls are in flight at once. A bench knob, because the right value is a
#: property of the machine and the draw, not of this code: at chunk_size=1 a batch is ~2,700
#: calls and at 64 it is ~35, and a width tuned for one is wrong for the other. Measured
#: rather than guessed -- an arm compared at an arbitrary width is not a fair comparison.
FANOUT_WIDTH = int(os.environ.get("ZARRISTA_WIDTH", "8"))


def read_runs_into(arr, limits: list[slice], out: np.ndarray, width: int = FANOUT_WIDTH) -> None:  # noqa: ANN001
    """Read each 1-D run of `arr` and lay the results out at their own offsets.

    This is the shape the CSR path always has: scattered ROWS become contiguous RANGES in the
    flat `data`/`indices` arrays, because `indptr` says where each row's values live. Runs
    sharing a decode unit are read once (see `coalesce`) and scattered here.

    Offsets are computed from the run lengths in INPUT order -- that is the order
    `_allocate_out` sized the buffer for -- while the reads happen in position order. The two
    orders are different and keeping them apart is the whole point.

    The copy is the missing `out=`. `to_numpy()` does not copy -- it views zarrista's own
    allocation -- so this is one memcpy of exactly the bytes wanted, not two.
    """
    runs: list[tuple[int, slice]] = []
    at = 0
    for limit in limits:
        runs.append((at, limit))
        at += max(0, limit.stop - limit.start)
    if at != out.size:
        raise ValueError(f"runs cover {at} elements, buffer holds {out.size}")

    groups = coalesce(runs, decode_unit(arr))

    def serve(group: tuple[slice, list[tuple[int, slice]]]) -> None:
        merged, members = group
        block = arr.retrieve_array_subset(merged).to_numpy()
        if block.dtype != out.dtype:
            # numpy `__setitem__` would cast silently; the zarr-python arm would not.
            raise TypeError(f"zarrista returned {block.dtype} for a buffer of {out.dtype}")
        for offset, limit in members:
            size = limit.stop - limit.start
            begin = limit.start - merged.start
            out[offset : offset + size] = block[begin : begin + size]

    # Fanned out, because the call COUNT is the cost and nothing in zarrista overlaps them.
    # Every call decodes an entire unit to extract one row's values -- 65,536 elements for
    # ~1,450 wanted at chunk_size=1 -- and shares nothing with the next call. Coalescing does
    # not help there: rows drawn at random over a plate of millions almost never land in the
    # same unit, measured at 1024 runs collapsing to 1019 calls. So the only lever left is
    # doing them at the same time. zarrista releases the GIL for the whole read and decode,
    # so these threads are real threads.
    #
    # Writes are disjoint by construction -- each group owns the output offsets of its own
    # members, and `coalesce` puts every run in exactly one group -- so no lock is needed.
    if len(groups) > 1 and width > 1:
        with ThreadPoolExecutor(max_workers=min(width, len(groups))) as pool:
            for _ in pool.map(serve, groups):
                pass
    else:
        for group in groups:
            serve(group)


async def read_runs_into_async(arr, limits: list[slice], out: np.ndarray, width: int = FANOUT_WIDTH) -> None:
    """`read_runs_into` on a worker thread, so two of these actually overlap.

    annbatch gathers the `data` and `indices` reads, and on the zarr-python arm those are
    genuine coroutines that interleave. Calling a blocking function inside a coroutine would
    serialise them and the arm would look slow for a reason that is not zarrista. zarrista's
    sync methods release the GIL, so a thread is a real thread here rather than a queue.
    """
    await asyncio.to_thread(read_runs_into, arr, limits, out, width)
