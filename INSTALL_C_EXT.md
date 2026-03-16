# C Shard Reader Extension

A Python C extension (`_shard_reader`) that accelerates zarr v3 shard reading
by performing mmap index parsing, blosc decompression, and memcpy entirely in C
with the GIL released.

## Prerequisites

- Python >= 3.12
- A C compiler (`gcc` or `clang`)
- NumPy (for `numpy/arrayobject.h`)
- numcodecs (provides the blosc shared library at runtime)

All Python dependencies are already pulled in by `annbatch` itself.

## Building the extension

From the repository root:

```bash
python build_ext.py
```

This compiles `src/annbatch/_shard_reader.c` with `-O3` and places the
resulting `.so` (or `.pyd` on Windows) into `src/annbatch/`.

## Verifying the build

```python
from annbatch._shard_reader import read_into_dense, read_into_1d
print("C extension loaded successfully")
```

If the import fails, `_direct_read.py` will automatically fall back to a
pure-Python path -- the extension is optional.

## How it works

`_direct_read.py` orchestrates reading:

1. Opens shard files via `mmap` (zero-copy).
2. Parses the shard index to locate inner chunks.
3. Calls into the C extension (`read_into_dense` / `read_into_1d`) which
   decompresses via blosc and copies rows directly into a pre-allocated
   NumPy output buffer -- all without holding the GIL.

The blosc function pointer is resolved once at import time through
`ctypes` and handed to the C side via `_init_blosc()`.

## Files

| File | Purpose |
|------|---------|
| `src/annbatch/_shard_reader.c` | C extension module |
| `src/annbatch/_direct_read.py` | Python orchestration + pure-Python fallback |
| `build_ext.py` | Build script (`setuptools.Extension`) |
