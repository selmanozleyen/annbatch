"""Does the zarrista arm return the SAME BYTES as the zarr-python arm?

Correctness before speed, and correctness stated as equality against the arm already trusted
-- not against a hand-computed expectation, which is a second thing that can be wrong.

Both arms read the same store, with the same sampler seed, so a difference is the read path.
"""

import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import zarr
from annbatch import Loader
from annbatch.samplers import RandomSampler

N_ROWS, N_COLS, DENSITY = 4000, 200, 0.05


def build_store(root: Path) -> list[Path]:
    """Three small CSR plates, written the way the real ones are: sharded zarr v3."""
    paths = []
    rng = np.random.default_rng(0)
    for plate in range(3):
        x = sp.random(N_ROWS, N_COLS, density=DENSITY, format="csr", random_state=plate,
                      dtype=np.float32)
        path = root / f"plate{plate}.zarr"
        adata = ad.AnnData(X=x, obs=None, var=None)
        adata.write_zarr(path)
        paths.append(path)
    assert rng is not None
    return paths


def read_all(paths: list[Path], *, use_zarrista: bool, seed: int) -> tuple[np.ndarray, int]:
    """One pass over a fixed row draw, returning the concatenated batch values."""
    datasets = [ad.io.sparse_dataset(zarr.open_group(p, mode="r")["X"]) for p in paths]
    sampler = RandomSampler(chunk_size=4, preload_nchunks=16, batch_size=64,
                            replacement=False, rng=np.random.default_rng(seed))
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None,
                    use_zarrista=use_zarrista)
    loader.add_datasets(datasets)
    out, n = [], 0
    for batch in loader:
        x = batch["X"] if isinstance(batch, dict) else batch
        arr = x.toarray() if sp.issparse(x) else np.asarray(x)
        out.append(arr)
        n += arr.shape[0]
        if n >= 512:
            break
    return np.concatenate(out, axis=0), n


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = build_store(root)
        print(f"built {len(paths)} plates of {N_ROWS}x{N_COLS} at density {DENSITY}")

        zp, n_zp = read_all(paths, use_zarrista=False, seed=17)
        print(f"zarr-python arm: {n_zp} rows, {zp.shape}, sum {zp.sum():.4f}")

        zt, n_zt = read_all(paths, use_zarrista=True, seed=17)
        print(f"zarrista arm:    {n_zt} rows, {zt.shape}, sum {zt.sum():.4f}")

        if zp.shape != zt.shape:
            print(f"FAIL: shapes differ {zp.shape} vs {zt.shape}")
            return 1
        if not np.array_equal(zp, zt):
            bad = int((zp != zt).sum())
            print(f"FAIL: {bad} of {zp.size} elements differ")
            return 1
        print(f"OK: both arms returned identical bytes over {zp.size:,} elements")
        return 0


if __name__ == "__main__":
    sys.exit(main())
