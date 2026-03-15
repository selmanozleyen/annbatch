use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use walkdir::WalkDir;
use zarrs_storage::byte_range::{ByteRangeIterator, InvalidByteRangeError};
use zarrs_storage::{
    Bytes, ListableStorageTraits, MaybeBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StoreKeyError, StoreKeys, StoreKeysPrefixes, StorePrefix, StorePrefixes,
};

struct MmapSlice {
    mmap: Arc<Mmap>,
    offset: usize,
    len: usize,
}

impl AsRef<[u8]> for MmapSlice {
    fn as_ref(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.len]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MmapStoreCreateError {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("base path {0} is not valid")]
    InvalidBasePath(PathBuf),
}

/// A read-only, memory-mapped filesystem store for zarrs.
///
/// Uses `mmap(2)` for file access. Mapped files are cached for the lifetime
/// of the store, so repeated reads of the same file (e.g. many chunks inside
/// one shard) hit the OS page cache with zero syscall and zero copy overhead.
///
/// Only [`ReadableStorageTraits`] and [`ListableStorageTraits`] are
/// implemented; this store is intentionally read-only.
#[derive(Debug)]
pub struct MmapStore {
    base_path: PathBuf,
    sort: bool,
    cache: RwLock<HashMap<StoreKey, Arc<Mmap>>>,
}

impl MmapStore {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self, MmapStoreCreateError> {
        let base_path = base_path.as_ref().to_path_buf();
        if base_path.to_str().is_none() {
            return Err(MmapStoreCreateError::InvalidBasePath(base_path));
        }
        Ok(Self {
            base_path,
            sort: false,
            cache: RwLock::default(),
        })
    }

    #[must_use]
    pub const fn sorted(mut self) -> Self {
        self.sort = true;
        self
    }

    #[must_use]
    pub fn key_to_fspath(&self, key: &StoreKey) -> PathBuf {
        let mut path = self.base_path.clone();
        if !key.as_str().is_empty() {
            path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        path
    }

    fn fspath_to_key(&self, path: &std::path::Path) -> Result<StoreKey, StoreKeyError> {
        let path = pathdiff::diff_paths(path, &self.base_path)
            .ok_or_else(|| StoreKeyError::from(path.to_str().unwrap_or_default().to_string()))?;
        let path_str = path.to_string_lossy();
        #[cfg(target_os = "windows")]
        {
            StoreKey::new(path_str.replace('\\', "/"))
        }
        #[cfg(not(target_os = "windows"))]
        {
            StoreKey::new(path_str)
        }
    }

    #[must_use]
    pub fn prefix_to_fs_path(&self, prefix: &StorePrefix) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(prefix.as_str());
        path
    }

    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
    }

    pub fn get_mmap_direct(&self, key: &StoreKey) -> Result<Option<Arc<Mmap>>, StorageError> {
        self.get_mmap(key)
    }

    fn get_mmap(&self, key: &StoreKey) -> Result<Option<Arc<Mmap>>, StorageError> {
        {
            let cache = self.cache.read().unwrap();
            if let Some(mmap) = cache.get(key) {
                return Ok(Some(Arc::clone(mmap)));
            }
        }

        let fspath = self.key_to_fspath(key);
        let file = match std::fs::File::open(&fspath) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // SAFETY: file is opened read-only; zarr stores are immutable during reads.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });

        let mut cache = self.cache.write().unwrap();
        let entry = cache.entry(key.clone()).or_insert(Arc::clone(&mmap));
        Ok(Some(Arc::clone(entry)))
    }
}

impl ReadableStorageTraits for MmapStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(mmap) = self.get_mmap(key)? else {
            return Ok(None);
        };

        let file_size = mmap.len() as u64;

        let out = byte_ranges
            .map(move |byte_range| {
                let start = byte_range.start(file_size);
                let length = byte_range.length(file_size);
                let end = start + length;

                if end > file_size {
                    return Err(InvalidByteRangeError::new(byte_range, file_size).into());
                }

                let offset = usize::try_from(start).unwrap();
                let len = usize::try_from(length).unwrap();
                let slice = MmapSlice {
                    mmap: Arc::clone(&mmap),
                    offset,
                    len,
                };
                Ok(Bytes::from_owner(slice))
            })
            .collect::<Vec<_>>();

        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let key_path = self.key_to_fspath(key);
        std::fs::metadata(key_path).map_or_else(|_| Ok(None), |metadata| Ok(Some(metadata.len())))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for MmapStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        let mut walker = WalkDir::new(&self.base_path);
        if self.sort {
            walker = walker.sort_by_file_name();
        }
        Ok(walker
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|v| v.path().is_file())
            .filter_map(|v| self.fspath_to_key(v.path()).ok())
            .collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut walker = WalkDir::new(self.prefix_to_fs_path(prefix));
        if self.sort {
            walker = walker.sort_by_file_name();
        }
        Ok(walker
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|v| v.path().is_file())
            .filter_map(|v| self.fspath_to_key(v.path()).ok())
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let prefix_path = self.prefix_to_fs_path(prefix);
        let mut keys: StoreKeys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        let dir = std::fs::read_dir(prefix_path);
        if let Ok(dir) = dir {
            for entry in dir {
                let entry = entry?;
                let fs_path = entry.path();
                let path = fs_path.file_name().unwrap();
                if fs_path.is_dir() {
                    prefixes.push(StorePrefix::new(
                        prefix.as_str().to_string() + path.to_str().unwrap() + "/",
                    )?);
                } else {
                    keys.push(StoreKey::new(
                        prefix.as_str().to_owned() + path.to_str().unwrap(),
                    )?);
                }
            }
        }
        if self.sort {
            keys.sort();
            prefixes.sort();
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut size = 0;
        for key in self.list_prefix(prefix)? {
            if let Some(size_key) = self.size_key(&key)? {
                size += size_key;
            }
        }
        Ok(size)
    }
}
