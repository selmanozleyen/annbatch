mod mmap_store;
mod reader;
mod shard_meta;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zarrs::array::Array;
use zarrs::storage::{ListableStorageTraits, ReadableStorageTraits};

use mmap_store::MmapStore;
use shard_meta::ShardMeta;

trait Store: ReadableStorageTraits + ListableStorageTraits + Send + Sync {}
impl<T: ReadableStorageTraits + ListableStorageTraits + Send + Sync> Store for T {}

#[pyclass]
struct ShardedArrayReader {
    store: Arc<MmapStore>,
    meta_cache: Mutex<HashMap<String, Arc<ShardMeta>>>,
}

#[pymethods]
impl ShardedArrayReader {
    #[new]
    #[pyo3(signature = (store_path, use_mmap = true))]
    fn new(store_path: String, use_mmap: bool) -> PyResult<Self> {
        if !use_mmap {
            return Err(PyValueError::new_err(
                "direct reader requires use_mmap=True (pread mode removed in direct path)",
            ));
        }
        let store = Arc::new(
            MmapStore::new(&store_path)
                .map_err(|e| PyValueError::new_err(format!("store error: {e}")))?,
        );
        Ok(Self { store, meta_cache: Mutex::new(HashMap::new()) })
    }

    fn read_raw<'py>(
        &self,
        py: Python<'py>,
        array_path: &str,
        starts: PyReadonlyArray1<'py, i64>,
        stops: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let s = starts.as_slice().map_err(|_| PyValueError::new_err("starts must be contiguous i64"))?;
        let e = stops.as_slice().map_err(|_| PyValueError::new_err("stops must be contiguous i64"))?;
        if s.len() != e.len() {
            return Err(PyValueError::new_err("starts and stops must have equal length"));
        }

        let meta = self.get_or_parse_meta(array_path)?;
        let store = self.store.clone();
        let (sv, ev) = (s.to_vec(), e.to_vec());

        let bytes = py.allow_threads(move || reader::read_direct(&store, &meta, &sv, &ev))?;

        let arr = ArrayD::from_shape_vec(numpy::ndarray::IxDyn(&[bytes.len()]), bytes)
            .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))?;
        Ok(arr.into_pyarray(py))
    }

    fn array_shape(&self, array_path: &str) -> PyResult<Vec<u64>> {
        Ok(self.get_or_parse_meta(array_path)?.shape.clone())
    }
}

impl ShardedArrayReader {
    fn get_or_parse_meta(&self, array_path: &str) -> PyResult<Arc<ShardMeta>> {
        let mut cache = self.meta_cache.lock().unwrap();
        if let Some(m) = cache.get(array_path) {
            return Ok(m.clone());
        }
        let store_dyn: Arc<dyn Store> = self.store.clone();
        let array = Array::open(store_dyn, array_path)
            .map_err(|e| PyValueError::new_err(format!("failed to open array: {e}")))?;
        let meta = Arc::new(ShardMeta::from_array(&array, array_path)?);
        cache.insert(array_path.to_owned(), meta.clone());
        Ok(meta)
    }
}

#[pymodule]
fn _zarrs_direct(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShardedArrayReader>()?;
    Ok(())
}
