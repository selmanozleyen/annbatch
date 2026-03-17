/*
 * _shard_reader.c -- Fast shard reader for zarr v3 sharded arrays.
 *
 * Provides three I/O strategies:
 *   - mmap: reads from an mmap'd numpy uint8 view (read_into_dense / read_into_1d)
 *   - pread: reads compressed chunks on demand via pread(2) (pread_into_dense / pread_into_1d)
 *   - pread_mt: multi-threaded pread with concurrent I/O and decompression
 *               (pread_mt_into_dense / pread_mt_into_1d)
 *
 * All decompress via blosc_decompress and copy into a pre-allocated
 * output buffer with the GIL released.
 *
 * Build as a Python C extension module.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* blosc function pointers */
typedef int (*blosc_decompress_fn)(const void *, void *, size_t);
typedef int (*blosc_set_nthreads_fn)(int);

static blosc_decompress_fn blosc_decompress_ptr = NULL;
static blosc_set_nthreads_fn blosc_set_nthreads_ptr = NULL;

/*
 * _init_blosc(decompress_addr: int, set_nthreads_addr: int) -> None
 *
 * Store function pointers for blosc_decompress and blosc_set_nthreads.
 * Called once from Python with ctypes-resolved addresses.
 */
static PyObject *
_init_blosc(PyObject *self, PyObject *args)
{
    unsigned long long addr, nthreads_addr;
    if (!PyArg_ParseTuple(args, "KK", &addr, &nthreads_addr))
        return NULL;
    blosc_decompress_ptr = (blosc_decompress_fn)(uintptr_t)addr;
    blosc_set_nthreads_ptr = (blosc_set_nthreads_fn)(uintptr_t)nthreads_addr;
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

/* -----------------------------------------------------------------------
 * Multi-threaded pread -- concurrent I/O + decompression via pthreads
 * -----------------------------------------------------------------------*/

typedef struct {
    int fd;
    uint64_t file_offset;
    uint64_t compressed_len;
    uint8_t *out_dest;
    size_t decompressed_size;
    long long take_start;
    long long n_take;
    long long row_nbytes;
    int needs_partial;
    int error;
} _chunk_work_t;

typedef struct {
    _chunk_work_t *work;
    int start;
    int stride;
    int count;
    int error;
} _strided_arg_t;

static void
_do_one_chunk(_chunk_work_t *w)
{
    w->error = 0;

    uint8_t *read_buf = (uint8_t *)malloc((size_t)w->compressed_len);
    if (!read_buf) { w->error = 1; return; }

    size_t total = 0;
    while (total < (size_t)w->compressed_len) {
        ssize_t n = pread(w->fd, read_buf + total,
                          (size_t)w->compressed_len - total,
                          (off_t)(w->file_offset + total));
        if (n <= 0) { w->error = 2; free(read_buf); return; }
        total += (size_t)n;
    }

    if (!w->needs_partial) {
        blosc_decompress_ptr(read_buf, w->out_dest, w->decompressed_size);
    } else {
        uint8_t *tmp = (uint8_t *)malloc(w->decompressed_size);
        if (!tmp) { w->error = 1; free(read_buf); return; }
        blosc_decompress_ptr(read_buf, tmp, w->decompressed_size);
        memcpy(w->out_dest,
               tmp + w->take_start * w->row_nbytes,
               (size_t)(w->n_take * w->row_nbytes));
        free(tmp);
    }

    free(read_buf);
}

static void *
_strided_thread_fn(void *arg)
{
    _strided_arg_t *ta = (_strided_arg_t *)arg;
    for (int i = ta->start; i < ta->count; i += ta->stride) {
        _do_one_chunk(&ta->work[i]);
        if (ta->work[i].error) { ta->error = ta->work[i].error; return NULL; }
    }
    return NULL;
}

/*
 * pread_mt_into_dense(
 *     fd, shard_index_flat, out,
 *     inner_chunk_nbytes, inner_row_size, shard_row_size,
 *     shard_base, sel_start, sel_stop, out_offset,
 *     row_nbytes, idx_stride, n_threads,
 * ) -> (rows_written, next_row)
 */
static PyObject *
pread_mt_into_dense(PyObject *self, PyObject *args)
{
    int fd, n_threads;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_row_size, shard_row_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long row_nbytes, idx_stride;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLLi",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_row_size, &shard_row_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &row_nbytes, &idx_stride, &n_threads))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized; call _init_blosc first");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    /* Phase 1: pre-compute work items (single-threaded, cheap) */
    long long shard_end = shard_base + shard_row_size;
    long long eff_stop = sel_stop < shard_end ? sel_stop : shard_end;
    int max_items = (int)((eff_stop - sel_start + inner_row_size - 1) / inner_row_size) + 1;
    _chunk_work_t *work = (_chunk_work_t *)calloc((size_t)max_items, sizeof(_chunk_work_t));
    if (!work) { PyErr_NoMemory(); return NULL; }

    int n_items = 0;
    long long row = sel_start;
    long long cur_offset = out_offset;
    long long total_rows = 0;

    while (row < eff_stop) {
        long long inner_idx = (row - shard_base) / inner_row_size;
        long long inner_base = shard_base + inner_idx * inner_row_size;
        long long inner_end = inner_base + inner_row_size;
        long long flat_idx = inner_idx * idx_stride;

        long long take_start = (row > inner_base ? row - inner_base : 0);
        long long take_end_row = (eff_stop < inner_end ? eff_stop : inner_end);
        long long n_take = take_end_row - inner_base - take_start;

        _chunk_work_t *w = &work[n_items];
        w->fd = fd;
        w->file_offset = idx_data[flat_idx];
        w->compressed_len = idx_data[flat_idx + 1];
        w->out_dest = out_ptr + cur_offset * row_nbytes;
        w->decompressed_size = (size_t)inner_chunk_nbytes;
        w->take_start = take_start;
        w->n_take = n_take;
        w->row_nbytes = row_nbytes;
        w->needs_partial = !(take_start == 0 && n_take == inner_row_size);

        cur_offset += n_take;
        total_rows += n_take;
        row = inner_base + take_start + n_take;
        n_items++;
    }

    /* Phase 2: dispatch work across threads */
    int err = 0;
    int old_nthreads = 0;

    Py_BEGIN_ALLOW_THREADS

    if (blosc_set_nthreads_ptr)
        old_nthreads = blosc_set_nthreads_ptr(1);

    if (n_threads > n_items) n_threads = n_items;

    if (n_threads <= 1) {
        for (int i = 0; i < n_items; i++) {
            _do_one_chunk(&work[i]);
            if (work[i].error) { err = work[i].error; break; }
        }
    } else {
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        if (!threads) { err = 1; goto restore; }

        _strided_arg_t *targs = (_strided_arg_t *)calloc((size_t)n_threads, sizeof(_strided_arg_t));
        if (!targs) { free(threads); err = 1; goto restore; }

        for (int t = 0; t < n_threads; t++) {
            targs[t].work = work;
            targs[t].start = t;
            targs[t].stride = n_threads;
            targs[t].count = n_items;
            targs[t].error = 0;
        }

        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, _strided_thread_fn, &targs[t]);
        for (int t = 0; t < n_threads; t++) {
            pthread_join(threads[t], NULL);
            if (targs[t].error && !err) err = targs[t].error;
        }

        free(targs);
        free(threads);
    }

restore:
    if (blosc_set_nthreads_ptr && old_nthreads > 0)
        blosc_set_nthreads_ptr(old_nthreads);

    Py_END_ALLOW_THREADS

    free(work);

    if (err == 1) { PyErr_NoMemory(); return NULL; }
    if (err == 2) { PyErr_SetString(PyExc_OSError, "pread_mt: short read"); return NULL; }

    return Py_BuildValue("LL", total_rows, row);
}

/*
 * pread_mt_into_1d -- same idea for 1-D arrays (CSR data/indices)
 */
static PyObject *
pread_mt_into_1d(PyObject *self, PyObject *args)
{
    int fd, n_threads;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_size, shard_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long elem_nbytes;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLi",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_size, &shard_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &elem_nbytes, &n_threads))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    long long shard_end = shard_base + shard_size;
    long long eff_stop = sel_stop < shard_end ? sel_stop : shard_end;
    int max_items = (int)((eff_stop - sel_start + inner_size - 1) / inner_size) + 1;
    _chunk_work_t *work = (_chunk_work_t *)calloc((size_t)max_items, sizeof(_chunk_work_t));
    if (!work) { PyErr_NoMemory(); return NULL; }

    int n_items = 0;
    long long pos = sel_start;
    long long cur_offset = out_offset;
    long long total_elems = 0;

    while (pos < eff_stop) {
        long long inner_idx = (pos - shard_base) / inner_size;
        long long inner_base = shard_base + inner_idx * inner_size;
        long long inner_end = inner_base + inner_size;

        long long take_start = (pos > inner_base ? pos - inner_base : 0);
        long long take_end_pos = (eff_stop < inner_end ? eff_stop : inner_end);
        long long n_take = take_end_pos - inner_base - take_start;

        _chunk_work_t *w = &work[n_items];
        w->fd = fd;
        w->file_offset = idx_data[inner_idx * 2];
        w->compressed_len = idx_data[inner_idx * 2 + 1];
        w->out_dest = out_ptr + cur_offset * elem_nbytes;
        w->decompressed_size = (size_t)inner_chunk_nbytes;
        w->take_start = take_start;
        w->n_take = n_take;
        w->row_nbytes = elem_nbytes;
        w->needs_partial = !(take_start == 0 && n_take == inner_size);

        cur_offset += n_take;
        total_elems += n_take;
        pos = inner_base + take_start + n_take;
        n_items++;
    }

    int err = 0;
    int old_nthreads = 0;

    Py_BEGIN_ALLOW_THREADS

    if (blosc_set_nthreads_ptr)
        old_nthreads = blosc_set_nthreads_ptr(1);

    if (n_threads > n_items) n_threads = n_items;

    if (n_threads <= 1) {
        for (int i = 0; i < n_items; i++) {
            _do_one_chunk(&work[i]);
            if (work[i].error) { err = work[i].error; break; }
        }
    } else {
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        if (!threads) { err = 1; goto restore_1d; }

        _strided_arg_t *targs = (_strided_arg_t *)calloc((size_t)n_threads, sizeof(_strided_arg_t));
        if (!targs) { free(threads); err = 1; goto restore_1d; }

        for (int t = 0; t < n_threads; t++) {
            targs[t].work = work;
            targs[t].start = t;
            targs[t].stride = n_threads;
            targs[t].count = n_items;
            targs[t].error = 0;
        }

        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, _strided_thread_fn, &targs[t]);
        for (int t = 0; t < n_threads; t++) {
            pthread_join(threads[t], NULL);
            if (targs[t].error && !err) err = targs[t].error;
        }

        free(targs);
        free(threads);
    }

restore_1d:
    if (blosc_set_nthreads_ptr && old_nthreads > 0)
        blosc_set_nthreads_ptr(old_nthreads);

    Py_END_ALLOW_THREADS

    free(work);

    if (err == 1) { PyErr_NoMemory(); return NULL; }
    if (err == 2) { PyErr_SetString(PyExc_OSError, "pread_mt: short read"); return NULL; }

    return Py_BuildValue("LL", total_elems, pos);
}

static PyMethodDef methods[] = {
    {"_init_blosc", _init_blosc, METH_VARARGS, "Initialize blosc function pointer"},
    {"read_into_dense", read_into_dense, METH_VARARGS, "Read chunks from mmap'd shard into dense output"},
    {"read_into_1d", read_into_1d, METH_VARARGS, "Read chunks from mmap'd shard into 1d output"},
    {"pread_into_dense", pread_into_dense, METH_VARARGS, "Read chunks from shard fd via pread into dense output"},
    {"pread_into_1d", pread_into_1d, METH_VARARGS, "Read chunks from shard fd via pread into 1d output"},
    {"pread_mt_into_dense", pread_mt_into_dense, METH_VARARGS, "Multi-threaded pread into dense output"},
    {"pread_mt_into_1d", pread_mt_into_1d, METH_VARARGS, "Multi-threaded pread into 1d output"},
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
