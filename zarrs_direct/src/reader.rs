use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::mmap_store::MmapStore;
use crate::shard_meta::ShardMeta;

struct FusedGroup {
    fused_start: u64,
    fused_stop: u64,
    // (original_range_index, byte_offset_within_fused_read, byte_length)
    slices: Vec<(usize, usize, usize)>,
}

fn fuse_sorted_ranges(
    order: &[usize], starts: &[i64], stops: &[i64], row_bytes: usize,
) -> Vec<FusedGroup> {
    if order.is_empty() { return Vec::new(); }
    let first = order[0];
    let mut result = Vec::new();
    let mut cur_start = starts[first] as u64;
    let mut cur_stop = stops[first] as u64;
    let mut slices = vec![(first, 0usize, (stops[first] - starts[first]) as usize * row_bytes)];

    for &idx in &order[1..] {
        let (s, e) = (starts[idx] as u64, stops[idx] as u64);
        if s <= cur_stop {
            slices.push((idx, (s - cur_start) as usize * row_bytes, (e - s) as usize * row_bytes));
            if e > cur_stop { cur_stop = e; }
        } else {
            result.push(FusedGroup { fused_start: cur_start, fused_stop: cur_stop, slices });
            cur_start = s;
            cur_stop = e;
            slices = vec![(idx, 0, (e - s) as usize * row_bytes)];
        }
    }
    result.push(FusedGroup { fused_start: cur_start, fused_stop: cur_stop, slices });
    result
}

fn shard_index<'a>(mmap: &'a Mmap, meta: &ShardMeta) -> &'a [u64] {
    let len = mmap.len();
    let bytes = &mmap[len - meta.shard_index_size..len - 4];
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, bytes.len() / 8) }
}

unsafe fn blosc_decompress(src: *const u8, dst: *mut u8, dst_size: usize) {
    blosc_src::blosc_decompress(
        src as *const std::ffi::c_void,
        dst as *mut std::ffi::c_void,
        dst_size,
    );
}

pub fn read_direct(
    store: &MmapStore, meta: &ShardMeta, starts: &[i64], stops: &[i64],
) -> PyResult<Vec<u8>> {
    let n = starts.len();
    let rb = meta.row_bytes;

    let range_bytes: Vec<usize> = starts.iter().zip(stops).map(|(&s, &e)| (e - s) as usize * rb).collect();
    let total: usize = range_bytes.iter().sum();
    let mut out = vec![0u8; total];

    // output offset for each original range (in input order)
    let mut out_off = vec![0usize; n];
    let mut acc = 0usize;
    for i in 0..n { out_off[i] = acc; acc += range_bytes[i]; }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| starts[i]);
    let fused = fuse_sorted_ranges(&order, starts, stops, rb);

    let chunk_rows = meta.chunk_shape[0];
    let shard_rows = meta.shard_shape[0];
    let chunk_bytes = meta.inner_chunk_nbytes();
    let ndim_extra = meta.ndim - 1;
    let mut tmp: Option<Vec<u8>> = None;

    for g in &fused {
        let (sel0, sel1) = (g.fused_start, g.fused_stop);
        let mut fbuf = vec![0u8; (sel1 - sel0) as usize * rb];
        let mut row = sel0;
        let mut boff = 0usize;

        while row < sel1 {
            let si = row / shard_rows;
            let mut idx = vec![si];
            for _ in 0..ndim_extra { idx.push(0); }

            let key = meta.shard_key(&idx)?;
            let mmap = store.get_mmap_direct(&key)
                .map_err(|e| PyValueError::new_err(format!("store error: {e}")))?
                .ok_or_else(|| PyValueError::new_err(format!("shard not found: {}", key.as_str())))?;
            let sidx = shard_index(&mmap, meta);
            let sbase = si * shard_rows;

            while row < sel1 && row < sbase + shard_rows {
                let ci = ((row - sbase) / chunk_rows) as usize;
                let cbase = sbase + ci as u64 * chunk_rows;
                let cend = cbase + chunk_rows;
                let off = sidx[ci * meta.idx_stride] as usize;

                let t0 = if row > cbase { (row - cbase) as usize } else { 0 };
                let t1 = (sel1.min(cend) - cbase) as usize;
                let nt = t1 - t0;

                if t0 == 0 && nt == chunk_rows as usize {
                    unsafe { blosc_decompress(mmap.as_ptr().add(off), fbuf.as_mut_ptr().add(boff), chunk_bytes); }
                } else {
                    let tb = tmp.get_or_insert_with(|| vec![0u8; chunk_bytes]);
                    if tb.len() < chunk_bytes { tb.resize(chunk_bytes, 0); }
                    unsafe { blosc_decompress(mmap.as_ptr().add(off), tb.as_mut_ptr(), chunk_bytes); }
                    let s = t0 * rb;
                    fbuf[boff..boff + nt * rb].copy_from_slice(&tb[s..s + nt * rb]);
                }
                boff += nt * rb;
                row = cbase + t1 as u64;
            }
        }

        for &(oi, so, sl) in &g.slices {
            out[out_off[oi]..out_off[oi] + sl].copy_from_slice(&fbuf[so..so + sl]);
        }
    }
    Ok(out)
}
