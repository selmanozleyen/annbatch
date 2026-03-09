"""Benchmark: LoadRequest vectorised helpers vs naive slice-based equivalents.

Run with:
    python tests/bench_load_request.py
"""

from __future__ import annotations

import timeit

import numpy as np

from annbatch.types import (
    LoadRequest,
    _multi_arange,
    load_request_stops,
    load_request_to_slices,
    load_request_total_obs,
)


def _make_load_request(n_chunks: int, chunk_size: int, remainder: int) -> LoadRequest:
    starts = np.arange(n_chunks) * chunk_size
    return {
        "chunk_size": chunk_size,
        "starts": starts,
        "remainder": remainder,
        "splits": [],
    }


def bench_total_obs(lr: LoadRequest, n: int) -> dict[str, float]:
    slices = load_request_to_slices(lr)

    def via_slices():
        return sum(s.stop - s.start for s in slices)

    def via_lr():
        return load_request_total_obs(lr)

    assert via_slices() == via_lr()
    t_slices = timeit.timeit(via_slices, number=n)
    t_lr = timeit.timeit(via_lr, number=n)
    return {"slices": t_slices, "lr": t_lr}


def bench_stops(lr: LoadRequest, n: int) -> dict[str, float]:
    slices = load_request_to_slices(lr)

    def via_slices():
        return np.array([s.stop for s in slices])

    def via_lr():
        return load_request_stops(lr)

    np.testing.assert_array_equal(via_slices(), via_lr())
    t_slices = timeit.timeit(via_slices, number=n)
    t_lr = timeit.timeit(via_lr, number=n)
    return {"slices": t_slices, "lr": t_lr}


def bench_multi_arange(lr: LoadRequest, n: int) -> dict[str, float]:
    slices = load_request_to_slices(lr)
    starts = lr["starts"]
    stops = load_request_stops(lr)

    def via_slices():
        return np.concatenate([np.arange(s.start, s.stop) for s in slices])

    def via_lr():
        return _multi_arange(starts, stops)

    np.testing.assert_array_equal(via_slices(), via_lr())
    t_slices = timeit.timeit(via_slices, number=n)
    t_lr = timeit.timeit(via_lr, number=n)
    return {"slices": t_slices, "lr": t_lr}


def bench_to_slices(lr: LoadRequest, n: int) -> dict[str, float]:
    """Time the cost of materializing slice objects themselves."""
    starts = lr["starts"]
    stops = load_request_stops(lr)

    def to_slices():
        return [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]

    # baseline: how fast is the no-op if we already have slices cached
    cached = to_slices()

    def noop():
        return cached

    t_build = timeit.timeit(to_slices, number=n)
    t_noop = timeit.timeit(noop, number=n)
    return {"build_slices": t_build, "cached_noop": t_noop}


def run_suite(label: str, n_chunks: int, chunk_size: int, remainder: int, n: int = 50_000):
    lr = _make_load_request(n_chunks, chunk_size, remainder)
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  n_chunks={n_chunks}  chunk_size={chunk_size}  remainder={remainder}")
    print(f"  iterations={n}")
    print(f"{'=' * 60}")

    for name, bench_fn in [
        ("total_obs", bench_total_obs),
        ("stops", bench_stops),
        ("multi_arange", bench_multi_arange),
        ("to_slices_cost", bench_to_slices),
    ]:
        res = bench_fn(lr, n)
        parts = "  |  ".join(f"{k}: {v*1e6/n:.2f} us/call" for k, v in res.items())
        keys = list(res.keys())
        if len(keys) == 2:
            speedup = res[keys[0]] / res[keys[1]]
            parts += f"  |  speedup: {speedup:.1f}x"
        print(f"  {name:20s}  {parts}")


if __name__ == "__main__":
    run_suite("Small (typical per-request)", n_chunks=32, chunk_size=512, remainder=0)
    run_suite("Small with remainder", n_chunks=32, chunk_size=512, remainder=37)
    run_suite("Medium", n_chunks=256, chunk_size=512, remainder=0)
    run_suite("Medium with remainder", n_chunks=256, chunk_size=512, remainder=100)
    run_suite("Large", n_chunks=2048, chunk_size=512, remainder=0)
    run_suite("Large with remainder", n_chunks=2048, chunk_size=512, remainder=200)
    # Also test single chunk (batch_size == chunk_size scenario)
    run_suite("Single chunk", n_chunks=1, chunk_size=4096, remainder=0)
