/*
 * _shard_reader.c -- Fast shard reader for zarr v3 sharded arrays.
 *
 * Provides two I/O strategies:
 *   - mmap: reads from an mmap'd numpy uint8 view (read_into_dense / read_into_1d)
 *   - pread: reads compressed chunks on demand via pread(2) (pread_into_dense / pread_into_1d)
 *
 * Both decompress via blosc_decompress and copy into a pre-allocated
 * output buffer with the GIL released.
 *
 * Build as a Python C extension module.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* blosc_decompress: int blosc_decompress(const void *src, void *dest, size_t destsize) */
typedef int (*blosc_decompress_fn)(const void *, void *, size_t);
static blosc_decompress_fn blosc_decompress_ptr = NULL;

/*
 * _init_blosc(address: int) -> None
 *
 * Store the function pointer for blosc_decompress.
 * Called once from Python with the ctypes-resolved address.
 */
static PyObject *
_init_blosc(PyObject *self, PyObject *args)
{
    unsigned long long addr;
    if (!PyArg_ParseTuple(args, "K", &addr))
        return NULL;
    blosc_decompress_ptr = (blosc_decompress_fn)(uintptr_t)addr;
    Py_RETURN_NONE;
}

/*
 * read_into_dense(
 *     shard_data: numpy uint8 array (mmap'd shard bytes),
 *     shard_index_flat: numpy uint64 1-D array [off0, len0, off1, len1, ...],
 *     out: numpy output array (contiguous, correct dtype),
 *     inner_chunk_nbytes: int,
 *     inner_row_size: int,
 *     shard_row_size: int,
 *     shard_base: int,
 *     sel_start: int,
 *     sel_stop: int,
 *     out_offset: int (row offset into out),
 *     row_nbytes: int,
 *     idx_stride: int (flat stride between consecutive inner chunks in the index,
 *                       = product(chunks_per_shard[1:]) * 2),
 * ) -> (rows_written: int, next_row: int)
 */
static PyObject *
read_into_dense(PyObject *self, PyObject *args)
{
    PyArrayObject *shard_data_obj, *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_row_size, shard_row_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long row_nbytes, idx_stride;

    if (!PyArg_ParseTuple(args, "O!O!O!LLLLLLLLL",
            &PyArray_Type, &shard_data_obj,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_row_size, &shard_row_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &row_nbytes, &idx_stride))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized; call _init_blosc first");
        return NULL;
    }

    const uint8_t *shard_bytes = (const uint8_t *)PyArray_DATA(shard_data_obj);
    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    uint8_t *tmp_buf = NULL;

    long long row = sel_start;
    long long rows_written = 0;

    Py_BEGIN_ALLOW_THREADS

    while (row < sel_stop && row < shard_base + shard_row_size) {
        long long inner_idx = (row - shard_base) / inner_row_size;
        long long inner_base = shard_base + inner_idx * inner_row_size;
        long long inner_end = inner_base + inner_row_size;

        long long flat_idx = inner_idx * idx_stride;
        uint64_t offset_val = idx_data[flat_idx];

        long long take_start = (row > inner_base ? row - inner_base : 0);
        long long take_end_row = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_row - inner_base;
        long long n_take = take_end - take_start;

        if (take_start == 0 && n_take == inner_row_size) {
            blosc_decompress_ptr(
                shard_bytes + offset_val,
                out_ptr + out_offset * row_nbytes,
                (size_t)inner_chunk_nbytes
            );
        } else {
            if (!tmp_buf) {
                tmp_buf = (uint8_t *)malloc((size_t)inner_chunk_nbytes);
            }
            blosc_decompress_ptr(
                shard_bytes + offset_val,
                tmp_buf,
                (size_t)inner_chunk_nbytes
            );
            memcpy(
                out_ptr + out_offset * row_nbytes,
                tmp_buf + take_start * row_nbytes,
                (size_t)(n_take * row_nbytes)
            );
        }

        out_offset += n_take;
        rows_written += n_take;
        row = inner_base + take_end;
    }

    free(tmp_buf);

    Py_END_ALLOW_THREADS

    return Py_BuildValue("LL", rows_written, row);
}

/*
 * read_into_1d(
 *     shard_data: numpy uint8 array,
 *     shard_index_flat: numpy uint64 1-D array,
 *     out: numpy 1d output array,
 *     inner_chunk_nbytes: int,
 *     inner_size: int,
 *     shard_size: int,
 *     shard_base: int,
 *     sel_start: int,
 *     sel_stop: int,
 *     out_offset: int,
 *     elem_nbytes: int,
 * ) -> (elems_written: int, next_pos: int)
 */
static PyObject *
read_into_1d(PyObject *self, PyObject *args)
{
    PyArrayObject *shard_data_obj, *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_size, shard_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long elem_nbytes;

    if (!PyArg_ParseTuple(args, "O!O!O!LLLLLLLL",
            &PyArray_Type, &shard_data_obj,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_size, &shard_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &elem_nbytes))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized");
        return NULL;
    }

    const uint8_t *shard_bytes = (const uint8_t *)PyArray_DATA(shard_data_obj);
    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    uint8_t *tmp_buf = NULL;
    long long pos = sel_start;
    long long elems_written = 0;

    Py_BEGIN_ALLOW_THREADS

    while (pos < sel_stop && pos < shard_base + shard_size) {
        long long inner_idx = (pos - shard_base) / inner_size;
        long long inner_base = shard_base + inner_idx * inner_size;
        long long inner_end = inner_base + inner_size;

        uint64_t offset_val = idx_data[inner_idx * 2];

        long long take_start = (pos > inner_base ? pos - inner_base : 0);
        long long take_end_pos = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_pos - inner_base;
        long long n_take = take_end - take_start;

        if (take_start == 0 && n_take == inner_size) {
            blosc_decompress_ptr(
                shard_bytes + offset_val,
                out_ptr + out_offset * elem_nbytes,
                (size_t)inner_chunk_nbytes
            );
        } else {
            if (!tmp_buf) {
                tmp_buf = (uint8_t *)malloc((size_t)inner_chunk_nbytes);
            }
            blosc_decompress_ptr(
                shard_bytes + offset_val,
                tmp_buf,
                (size_t)inner_chunk_nbytes
            );
            memcpy(
                out_ptr + out_offset * elem_nbytes,
                tmp_buf + take_start * elem_nbytes,
                (size_t)(n_take * elem_nbytes)
            );
        }

        out_offset += n_take;
        elems_written += n_take;
        pos = inner_base + take_end;
    }

    free(tmp_buf);

    Py_END_ALLOW_THREADS

    return Py_BuildValue("LL", elems_written, pos);
}

/* -----------------------------------------------------------------------
 * pread variants -- read compressed chunks from fd via pread(2)
 * -----------------------------------------------------------------------*/

static ssize_t
_full_pread(int fd, void *buf, size_t count, off_t offset)
{
    size_t total = 0;
    while (total < count) {
        ssize_t n = pread(fd, (char *)buf + total, count - total, offset + (off_t)total);
        if (n <= 0) return n == 0 ? (ssize_t)total : n;
        total += (size_t)n;
    }
    return (ssize_t)total;
}

/*
 * pread_into_dense(
 *     fd: int,
 *     shard_index_flat: numpy uint64 1-D array,
 *     out: numpy output array,
 *     inner_chunk_nbytes: int,
 *     inner_row_size: int,
 *     shard_row_size: int,
 *     shard_base: int,
 *     sel_start: int,
 *     sel_stop: int,
 *     out_offset: int,
 *     row_nbytes: int,
 *     idx_stride: int,
 *     max_compressed: int (upper bound on compressed chunk size for read buffer),
 * ) -> (rows_written: int, next_row: int)
 */
static PyObject *
pread_into_dense(PyObject *self, PyObject *args)
{
    int fd;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_row_size, shard_row_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long row_nbytes, idx_stride, max_compressed;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLLL",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_row_size, &shard_row_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &row_nbytes, &idx_stride, &max_compressed))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized; call _init_blosc first");
        return NULL;
    }

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    uint8_t *tmp_buf = NULL;
    uint8_t *read_buf = NULL;
    long long row = sel_start;
    long long rows_written = 0;
    int err = 0;

    Py_BEGIN_ALLOW_THREADS

    read_buf = (uint8_t *)malloc((size_t)max_compressed);
    if (!read_buf) { err = 1; goto done_dense; }

    while (row < sel_stop && row < shard_base + shard_row_size) {
        long long inner_idx = (row - shard_base) / inner_row_size;
        long long inner_base = shard_base + inner_idx * inner_row_size;
        long long inner_end = inner_base + inner_row_size;

        long long flat_idx = inner_idx * idx_stride;
        uint64_t offset_val = idx_data[flat_idx];
        uint64_t length_val = idx_data[flat_idx + 1];

        ssize_t nr = _full_pread(fd, read_buf, (size_t)length_val, (off_t)offset_val);
        if (nr < (ssize_t)length_val) { err = 2; goto done_dense; }

        long long take_start = (row > inner_base ? row - inner_base : 0);
        long long take_end_row = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_row - inner_base;
        long long n_take = take_end - take_start;

        if (take_start == 0 && n_take == inner_row_size) {
            blosc_decompress_ptr(read_buf, out_ptr + out_offset * row_nbytes,
                                 (size_t)inner_chunk_nbytes);
        } else {
            if (!tmp_buf)
                tmp_buf = (uint8_t *)malloc((size_t)inner_chunk_nbytes);
            blosc_decompress_ptr(read_buf, tmp_buf, (size_t)inner_chunk_nbytes);
            memcpy(out_ptr + out_offset * row_nbytes,
                   tmp_buf + take_start * row_nbytes,
                   (size_t)(n_take * row_nbytes));
        }

        out_offset += n_take;
        rows_written += n_take;
        row = inner_base + take_end;
    }

done_dense:
    free(tmp_buf);
    free(read_buf);

    Py_END_ALLOW_THREADS

    if (err == 1) {
        PyErr_NoMemory();
        return NULL;
    }
    if (err == 2) {
        PyErr_SetString(PyExc_OSError, "pread: short read on shard file");
        return NULL;
    }

    return Py_BuildValue("LL", rows_written, row);
}

/*
 * pread_into_1d(
 *     fd: int,
 *     shard_index_flat: numpy uint64 1-D array,
 *     out: numpy 1d output array,
 *     inner_chunk_nbytes: int,
 *     inner_size: int,
 *     shard_size: int,
 *     shard_base: int,
 *     sel_start: int,
 *     sel_stop: int,
 *     out_offset: int,
 *     elem_nbytes: int,
 *     max_compressed: int,
 * ) -> (elems_written: int, next_pos: int)
 */
static PyObject *
pread_into_1d(PyObject *self, PyObject *args)
{
    int fd;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_size, shard_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long elem_nbytes, max_compressed;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLL",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_size, &shard_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &elem_nbytes, &max_compressed))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized");
        return NULL;
    }

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    uint8_t *tmp_buf = NULL;
    uint8_t *read_buf = NULL;
    long long pos = sel_start;
    long long elems_written = 0;
    int err = 0;

    Py_BEGIN_ALLOW_THREADS

    read_buf = (uint8_t *)malloc((size_t)max_compressed);
    if (!read_buf) { err = 1; goto done_1d; }

    while (pos < sel_stop && pos < shard_base + shard_size) {
        long long inner_idx = (pos - shard_base) / inner_size;
        long long inner_base = shard_base + inner_idx * inner_size;
        long long inner_end = inner_base + inner_size;

        uint64_t offset_val = idx_data[inner_idx * 2];
        uint64_t length_val = idx_data[inner_idx * 2 + 1];

        ssize_t nr = _full_pread(fd, read_buf, (size_t)length_val, (off_t)offset_val);
        if (nr < (ssize_t)length_val) { err = 2; goto done_1d; }

        long long take_start = (pos > inner_base ? pos - inner_base : 0);
        long long take_end_pos = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_pos - inner_base;
        long long n_take = take_end - take_start;

        if (take_start == 0 && n_take == inner_size) {
            blosc_decompress_ptr(read_buf, out_ptr + out_offset * elem_nbytes,
                                 (size_t)inner_chunk_nbytes);
        } else {
            if (!tmp_buf)
                tmp_buf = (uint8_t *)malloc((size_t)inner_chunk_nbytes);
            blosc_decompress_ptr(read_buf, tmp_buf, (size_t)inner_chunk_nbytes);
            memcpy(out_ptr + out_offset * elem_nbytes,
                   tmp_buf + take_start * elem_nbytes,
                   (size_t)(n_take * elem_nbytes));
        }

        out_offset += n_take;
        elems_written += n_take;
        pos = inner_base + take_end;
    }

done_1d:
    free(tmp_buf);
    free(read_buf);

    Py_END_ALLOW_THREADS

    if (err == 1) {
        PyErr_NoMemory();
        return NULL;
    }
    if (err == 2) {
        PyErr_SetString(PyExc_OSError, "pread: short read on shard file");
        return NULL;
    }

    return Py_BuildValue("LL", elems_written, pos);
}

static PyMethodDef methods[] = {
    {"_init_blosc", _init_blosc, METH_VARARGS, "Initialize blosc function pointer"},
    {"read_into_dense", read_into_dense, METH_VARARGS, "Read chunks from mmap'd shard into dense output"},
    {"read_into_1d", read_into_1d, METH_VARARGS, "Read chunks from mmap'd shard into 1d output"},
    {"pread_into_dense", pread_into_dense, METH_VARARGS, "Read chunks from shard fd via pread into dense output"},
    {"pread_into_1d", pread_into_1d, METH_VARARGS, "Read chunks from shard fd via pread into 1d output"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_shard_reader",
    "Fast shard reader C extension",
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__shard_reader(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;
    import_array();
    return m;
}
