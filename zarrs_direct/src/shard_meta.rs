use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use zarrs::array::{Array, ArrayShardedExt, chunk_shape_to_array_shape};
use zarrs_storage::StoreKey;

pub struct ShardMeta {
    pub shape: Vec<u64>,
    pub dtype_size: usize,
    pub ndim: usize,
    pub shard_shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub shard_index_size: usize,
    pub row_bytes: usize,
    pub idx_stride: usize,
    pub array_path_prefix: String,
}

impl ShardMeta {
    pub fn from_array<S: ?Sized>(array: &Array<S>, array_path: &str) -> PyResult<Self> {
        if !array.is_sharded() {
            return Err(PyValueError::new_err("direct reader only supports sharded arrays"));
        }
        let subchunk = array.subchunk_shape()
            .ok_or_else(|| PyValueError::new_err("could not determine subchunk shape"))?;
        let chunk_shape = chunk_shape_to_array_shape(&subchunk);

        let origin = vec![0u64; array.shape().len()];
        let shard_cs = array.chunk_shape(&origin)
            .map_err(|e| PyValueError::new_err(format!("could not get shard shape: {e}")))?;
        let shard_shape = chunk_shape_to_array_shape(&shard_cs);

        let shape = array.shape().to_vec();
        let ndim = shape.len();
        let dtype_size = array.data_type().fixed_size()
            .ok_or_else(|| PyValueError::new_err("variable-length dtypes not supported"))?;

        let chunks_per_shard: Vec<u64> = shard_shape.iter()
            .zip(chunk_shape.iter())
            .map(|(&s, &c)| s / c)
            .collect();
        let n_inner: u64 = chunks_per_shard.iter().product();

        let row_bytes = if ndim > 1 {
            shape[1..].iter().product::<u64>() as usize * dtype_size
        } else {
            dtype_size
        };
        let idx_stride = if ndim > 1 {
            chunks_per_shard[1..].iter().product::<u64>() as usize * 2
        } else {
            2
        };

        let prefix = array_path.strip_prefix('/').unwrap_or(array_path);
        let array_path_prefix = if prefix.is_empty() {
            "c".to_string()
        } else {
            format!("{prefix}/c")
        };

        Ok(Self {
            shape, dtype_size, ndim, shard_shape, chunk_shape,
            shard_index_size: (16 * n_inner + 4) as usize,
            row_bytes, idx_stride, array_path_prefix,
        })
    }

    pub fn inner_chunk_nbytes(&self) -> usize {
        self.chunk_shape.iter().product::<u64>() as usize * self.dtype_size
    }

    pub fn shard_key(&self, shard_indices: &[u64]) -> PyResult<StoreKey> {
        let parts: Vec<String> = shard_indices.iter().map(|i| i.to_string()).collect();
        StoreKey::new(format!("{}/{}", self.array_path_prefix, parts.join("/")))
            .map_err(|e| PyValueError::new_err(format!("invalid shard key: {e}")))
    }
}
