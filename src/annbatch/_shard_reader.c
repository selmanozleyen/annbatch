/*
 * _shard_reader.c -- Fast shard reader for zarr v3 sharded arrays.
 *
 * Provides mmap-based and pread-based readers.  The pread path opens
 * shard files, reads the shard index + compressed chunks via pread(),
 * decompresses with blosc, and writes into a pre-allocated output
 * buffer -- all with GIL released.
 *
 * pread functions accept an nthreads parameter; when > 1 they use
 * pthreads to issue concurrent pread+decompress operations.
 *
 * Build as a Python C extension module.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <pthread.h>

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

/* ------------------------------------------------------------------ */
/* pread-based readers (multi-threaded)                               */
/*                                                                    */
/* fd and shard index are cached on the Python side.  These functions */
/* take (fd, shard_index_flat, out, ..., nthreads) and pread chunks   */
/* directly from the fd, decompress, and write to the output buffer.  */
/*                                                                    */
/* When nthreads > 1 a first pass builds a work-item list, then       */
/* pthreads process the items concurrently.  Each work item writes to */
/* a disjoint region of `out` so no locking is needed for the output. */
/* ------------------------------------------------------------------ */

static ssize_t
_full_pread(int fd, void *buf, size_t count, off_t offset)
{
    size_t done = 0;
    while (done < count) {
        ssize_t n = pread(fd, (char *)buf + done, count - done, offset + (off_t)done);
        if (n <= 0) return n == 0 ? (ssize_t)done : n;
        done += (size_t)n;
    }
    return (ssize_t)done;
}

/* Work item for one chunk read+decompress. */
typedef struct {
    int         fd;
    uint64_t    chunk_off;
    uint64_t    chunk_len;
    uint8_t    *out_dst;         /* where decompressed data starts in out */
    long long   take_start;      /* offset within decompressed chunk */
    long long   n_take;          /* number of elements/rows to copy */
    long long   elem_nbytes;     /* bytes per element/row */
    long long   inner_chunk_nbytes;
    int         need_partial;    /* 1 if partial chunk copy needed */
} work_item_t;

typedef struct {
    work_item_t *items;
    int          start;
    int          end;
    int          err;            /* 0 = ok, nonzero = error */
} thread_arg_t;

static void *
_worker(void *arg)
{
    thread_arg_t *ta = (thread_arg_t *)arg;
    uint8_t *comp_buf = NULL;
    size_t comp_cap = 0;
    uint8_t *tmp_buf = NULL;

    for (int i = ta->start; i < ta->end; i++) {
        work_item_t *w = &ta->items[i];

        if ((size_t)w->chunk_len > comp_cap) {
            free(comp_buf);
            comp_cap = (size_t)w->chunk_len;
            comp_buf = (uint8_t *)malloc(comp_cap);
            if (!comp_buf) { ta->err = 1; break; }
        }

        if (_full_pread(w->fd, comp_buf, (size_t)w->chunk_len,
                        (off_t)w->chunk_off) != (ssize_t)w->chunk_len) {
            ta->err = 2; break;
        }

        if (!w->need_partial) {
            blosc_decompress_ptr(comp_buf, w->out_dst,
                                 (size_t)w->inner_chunk_nbytes);
        } else {
            if (!tmp_buf) {
                tmp_buf = (uint8_t *)malloc((size_t)w->inner_chunk_nbytes);
                if (!tmp_buf) { ta->err = 3; break; }
            }
            blosc_decompress_ptr(comp_buf, tmp_buf,
                                 (size_t)w->inner_chunk_nbytes);
            memcpy(w->out_dst,
                   tmp_buf + w->take_start * w->elem_nbytes,
                   (size_t)(w->n_take * w->elem_nbytes));
        }
    }

    free(comp_buf);
    free(tmp_buf);
    return NULL;
}

/*
 * pread_into_dense(
 *     fd, shard_index_flat, out,
 *     inner_chunk_nbytes, inner_row_size, shard_row_size,
 *     shard_base, sel_start, sel_stop, out_offset,
 *     row_nbytes, idx_stride, nthreads
 * ) -> (rows_written, next_row)
 */
static PyObject *
pread_into_dense(PyObject *self, PyObject *args)
{
    int fd;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_row_size, shard_row_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long row_nbytes, idx_stride;
    int nthreads;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLLi",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_row_size, &shard_row_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &row_nbytes, &idx_stride, &nthreads))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized");
        return NULL;
    }
    if (nthreads < 1) nthreads = 1;

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    /* --- Build work list (serial scan) --- */
    int work_cap = 64;
    int work_n = 0;
    work_item_t *work = (work_item_t *)malloc(work_cap * sizeof(work_item_t));
    if (!work) {
        PyErr_NoMemory();
        return NULL;
    }

    long long row = sel_start;
    long long cur_out = out_offset;
    while (row < sel_stop && row < shard_base + shard_row_size) {
        long long inner_idx = (row - shard_base) / inner_row_size;
        long long inner_base = shard_base + inner_idx * inner_row_size;
        long long inner_end = inner_base + inner_row_size;

        long long flat_idx = inner_idx * idx_stride;
        uint64_t chunk_off = idx_data[flat_idx];
        uint64_t chunk_len = idx_data[flat_idx + 1];

        long long take_start = (row > inner_base ? row - inner_base : 0);
        long long take_end_row = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_row - inner_base;
        long long n_take = take_end - take_start;

        if (work_n >= work_cap) {
            work_cap *= 2;
            work = (work_item_t *)realloc(work, work_cap * sizeof(work_item_t));
            if (!work) { PyErr_NoMemory(); return NULL; }
        }

        work_item_t *w = &work[work_n++];
        w->fd = fd;
        w->chunk_off = chunk_off;
        w->chunk_len = chunk_len;
        w->out_dst = out_ptr + cur_out * row_nbytes;
        w->take_start = take_start;
        w->n_take = n_take;
        w->elem_nbytes = row_nbytes;
        w->inner_chunk_nbytes = inner_chunk_nbytes;
        w->need_partial = !(take_start == 0 && n_take == inner_row_size);

        cur_out += n_take;
        row = inner_base + take_end;
    }

    long long rows_written = cur_out - out_offset;
    int err = 0;

    Py_BEGIN_ALLOW_THREADS

    if (nthreads <= 1 || work_n <= 1) {
        thread_arg_t ta = { work, 0, work_n, 0 };
        _worker(&ta);
        err = ta.err;
    } else {
        if (nthreads > work_n) nthreads = work_n;
        pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
        thread_arg_t *targs = (thread_arg_t *)malloc(nthreads * sizeof(thread_arg_t));

        int per = work_n / nthreads;
        int rem = work_n % nthreads;
        int offset = 0;
        for (int t = 0; t < nthreads; t++) {
            int count = per + (t < rem ? 1 : 0);
            targs[t].items = work;
            targs[t].start = offset;
            targs[t].end = offset + count;
            targs[t].err = 0;
            offset += count;
        }

        for (int t = 0; t < nthreads; t++)
            pthread_create(&threads[t], NULL, _worker, &targs[t]);
        for (int t = 0; t < nthreads; t++)
            pthread_join(threads[t], NULL);

        for (int t = 0; t < nthreads; t++) {
            if (targs[t].err) { err = targs[t].err; break; }
        }

        free(threads);
        free(targs);
    }

    Py_END_ALLOW_THREADS

    free(work);

    if (err) {
        PyErr_Format(PyExc_OSError, "pread_into_dense: error (code %d)", err);
        return NULL;
    }
    return Py_BuildValue("LL", rows_written, row);
}

/*
 * pread_into_1d(
 *     fd, shard_index_flat, out,
 *     inner_chunk_nbytes, inner_size, shard_size,
 *     shard_base, sel_start, sel_stop, out_offset,
 *     elem_nbytes, nthreads
 * ) -> (elems_written, next_pos)
 */
static PyObject *
pread_into_1d(PyObject *self, PyObject *args)
{
    int fd;
    PyArrayObject *shard_index_obj, *out_obj;
    long long inner_chunk_nbytes, inner_size, shard_size;
    long long shard_base, sel_start, sel_stop, out_offset;
    long long elem_nbytes;
    int nthreads;

    if (!PyArg_ParseTuple(args, "iO!O!LLLLLLLLi",
            &fd,
            &PyArray_Type, &shard_index_obj,
            &PyArray_Type, &out_obj,
            &inner_chunk_nbytes, &inner_size, &shard_size,
            &shard_base, &sel_start, &sel_stop, &out_offset,
            &elem_nbytes, &nthreads))
        return NULL;

    if (!blosc_decompress_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "blosc not initialized");
        return NULL;
    }
    if (nthreads < 1) nthreads = 1;

    const uint64_t *idx_data = (const uint64_t *)PyArray_DATA(shard_index_obj);
    uint8_t *out_ptr = (uint8_t *)PyArray_DATA(out_obj);

    /* --- Build work list (serial scan) --- */
    int work_cap = 64;
    int work_n = 0;
    work_item_t *work = (work_item_t *)malloc(work_cap * sizeof(work_item_t));
    if (!work) {
        PyErr_NoMemory();
        return NULL;
    }

    long long pos = sel_start;
    long long cur_out = out_offset;
    while (pos < sel_stop && pos < shard_base + shard_size) {
        long long inner_idx = (pos - shard_base) / inner_size;
        long long inner_base = shard_base + inner_idx * inner_size;
        long long inner_end = inner_base + inner_size;

        uint64_t chunk_off = idx_data[inner_idx * 2];
        uint64_t chunk_len = idx_data[inner_idx * 2 + 1];

        long long take_start = (pos > inner_base ? pos - inner_base : 0);
        long long take_end_pos = (sel_stop < inner_end ? sel_stop : inner_end);
        long long take_end = take_end_pos - inner_base;
        long long n_take = take_end - take_start;

        if (work_n >= work_cap) {
            work_cap *= 2;
            work = (work_item_t *)realloc(work, work_cap * sizeof(work_item_t));
            if (!work) { PyErr_NoMemory(); return NULL; }
        }

        work_item_t *w = &work[work_n++];
        w->fd = fd;
        w->chunk_off = chunk_off;
        w->chunk_len = chunk_len;
        w->out_dst = out_ptr + cur_out * elem_nbytes;
        w->take_start = take_start;
        w->n_take = n_take;
        w->elem_nbytes = elem_nbytes;
        w->inner_chunk_nbytes = inner_chunk_nbytes;
        w->need_partial = !(take_start == 0 && n_take == inner_size);

        cur_out += n_take;
        pos = inner_base + take_end;
    }

    long long elems_written = cur_out - out_offset;
    int err = 0;

    Py_BEGIN_ALLOW_THREADS

    if (nthreads <= 1 || work_n <= 1) {
        thread_arg_t ta = { work, 0, work_n, 0 };
        _worker(&ta);
        err = ta.err;
    } else {
        if (nthreads > work_n) nthreads = work_n;
        pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
        thread_arg_t *targs = (thread_arg_t *)malloc(nthreads * sizeof(thread_arg_t));

        int per = work_n / nthreads;
        int rem = work_n % nthreads;
        int offset = 0;
        for (int t = 0; t < nthreads; t++) {
            int count = per + (t < rem ? 1 : 0);
            targs[t].items = work;
            targs[t].start = offset;
            targs[t].end = offset + count;
            targs[t].err = 0;
            offset += count;
        }

        for (int t = 0; t < nthreads; t++)
            pthread_create(&threads[t], NULL, _worker, &targs[t]);
        for (int t = 0; t < nthreads; t++)
            pthread_join(threads[t], NULL);

        for (int t = 0; t < nthreads; t++) {
            if (targs[t].err) { err = targs[t].err; break; }
        }

        free(threads);
        free(targs);
    }

    Py_END_ALLOW_THREADS

    free(work);

    if (err) {
        PyErr_Format(PyExc_OSError, "pread_into_1d: error (code %d)", err);
        return NULL;
    }
    return Py_BuildValue("LL", elems_written, pos);
}

static PyMethodDef methods[] = {
    {"_init_blosc", _init_blosc, METH_VARARGS, "Initialize blosc function pointer"},
    {"read_into_dense", read_into_dense, METH_VARARGS, "Read chunks from mmap'd shard into dense output"},
    {"read_into_1d", read_into_1d, METH_VARARGS, "Read chunks from mmap'd shard into 1d output"},
    {"pread_into_dense", pread_into_dense, METH_VARARGS, "Read chunks via pread into dense output (multi-threaded)"},
    {"pread_into_1d", pread_into_1d, METH_VARARGS, "Read chunks via pread into 1d output (multi-threaded)"},
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
