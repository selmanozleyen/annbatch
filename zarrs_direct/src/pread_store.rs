// this file exists only for ablation studies
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use zarrs_storage::{StorageError, StoreKey};

#[derive(Debug)]
pub struct PreadStore {
    pub base_path: PathBuf,
    files: RwLock<HashMap<StoreKey, Arc<File>>>,
}

impl PreadStore {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            files: RwLock::default(),
        }
    }

    fn key_to_fspath(&self, key: &StoreKey) -> PathBuf {
        let mut path = self.base_path.clone();
        if !key.as_str().is_empty() {
            path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        path
    }

    fn get_file(&self, key: &StoreKey) -> Result<Option<Arc<File>>, StorageError> {
        {
            let cache = self.files.read().unwrap();
            if let Some(f) = cache.get(key) {
                return Ok(Some(Arc::clone(f)));
            }
        }
        let fspath = self.key_to_fspath(key);
        let file = match File::open(&fspath) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let file = Arc::new(file);
        let mut cache = self.files.write().unwrap();
        let entry = cache.entry(key.clone()).or_insert(Arc::clone(&file));
        Ok(Some(Arc::clone(entry)))
    }

    pub fn read_shard(
        &self, key: &StoreKey, index_size: usize,
    ) -> Result<Option<ShardFile>, StorageError> {
        let file = match self.get_file(key)? {
            Some(f) => f,
            None => return Ok(None),
        };
        let file_len = file.metadata().map_err(|e| StorageError::from(e))?.len() as usize;
        Ok(Some(ShardFile { file, file_len, index_size }))
    }
}

pub struct ShardFile {
    file: Arc<File>,
    file_len: usize,
    index_size: usize,
}

impl ShardFile {
    pub fn read_index(&self) -> Vec<u64> {
        let idx_offset = self.file_len - self.index_size;
        let idx_bytes_len = self.index_size - 4;
        let mut buf = vec![0u8; idx_bytes_len];
        self.file.read_at(&mut buf, idx_offset as u64).unwrap();
        let n_u64 = idx_bytes_len / 8;
        let mut out = vec![0u64; n_u64];
        // reinterpret as little-endian u64
        for i in 0..n_u64 {
            out[i] = u64::from_le_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        out
    }

    pub fn read_chunk(&self, offset: usize, dest: &mut [u8]) {
        // Read the blosc header (16 bytes) to get compressed size
        let mut header = [0u8; 16];
        self.file.read_at(&mut header, offset as u64).unwrap();
        let cbytes = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let mut compressed = vec![0u8; cbytes];
        self.file.read_at(&mut compressed, offset as u64).unwrap();
        unsafe {
            blosc_src::blosc_decompress(
                compressed.as_ptr() as *const std::ffi::c_void,
                dest.as_mut_ptr() as *mut std::ffi::c_void,
                dest.len(),
            );
        }
    }
}
