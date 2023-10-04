//! A zip store.

use crate::{
    byte_range::ByteRange,
    storage::{
        storage_value_io::StorageValueIO, ListableStorageTraits, ReadableStorageTraits,
        StorageError, StoreKey, StoreKeyRange, StoreKeys, StoreKeysPrefixes, StorePrefix,
        StorePrefixes,
    },
};

use itertools::Itertools;
use parking_lot::Mutex;
use thiserror::Error;
use zip::{result::ZipError, ZipArchive};

use std::{io::Read, path::PathBuf, sync::Arc};

/// A zip storage adapter.
pub struct ZipStorageAdapter<TStorage: ?Sized + ReadableStorageTraits> {
    size: u64,
    zip_archive: Mutex<ZipArchive<StorageValueIO<TStorage>>>,
    zip_path: PathBuf,
}

impl<TStorage: ?Sized + ReadableStorageTraits> ZipStorageAdapter<TStorage> {
    /// Create a new zip storage adapter.
    ///
    /// # Errors
    ///
    /// Returns a [`ZipStorageAdapterCreateError`] if `zip_path` is not valid zip file.
    pub fn new(
        storage: Arc<TStorage>,
    ) -> Result<ZipStorageAdapter<TStorage>, ZipStorageAdapterCreateError> {
        Self::new_with_path(storage, "")
    }

    /// Create a new zip storage adapter to `path` within the zip file.
    ///
    /// # Errors
    ///
    /// Returns a [`ZipStorageAdapterCreateError`] if `zip_path` is not valid zip file.
    pub fn new_with_path<T: Into<PathBuf>>(
        storage: Arc<TStorage>,
        path: T,
    ) -> Result<ZipStorageAdapter<TStorage>, ZipStorageAdapterCreateError> {
        let zip_path = path.into();
        let key = unsafe { StoreKey::new_unchecked(String::new()) };
        let size = storage.size_key(&key)?;
        let storage_io = StorageValueIO::new(storage, key)?;
        let zip_archive = Mutex::new(ZipArchive::new(storage_io)?);
        Ok(ZipStorageAdapter {
            size,
            zip_archive,
            zip_path,
        })
    }

    fn get_impl(&self, key: &StoreKey, byte_range: &ByteRange) -> Result<Vec<u8>, StorageError> {
        let mut zip_archive = self.zip_archive.lock();
        let mut zip_name = self.zip_path.clone();
        zip_name.push(key.as_str());
        let mut file =
            zip_archive
                .by_name(&zip_name.to_string_lossy())
                .map_err(|err| match err {
                    ZipError::FileNotFound => StorageError::KeyNotFound(key.clone()),
                    _ => StorageError::Other(err.to_string()),
                })?;
        let size = file.size();

        let buffer = match byte_range {
            ByteRange::FromStart(offset, None) => {
                std::io::copy(&mut file.by_ref().take(*offset), &mut std::io::sink()).unwrap();
                let mut buffer = Vec::with_capacity(usize::try_from(size - *offset).unwrap());
                file.read_to_end(&mut buffer)?;
                buffer
            }
            ByteRange::FromStart(offset, Some(length)) => {
                std::io::copy(&mut file.by_ref().take(*offset), &mut std::io::sink()).unwrap();
                let mut buffer = Vec::with_capacity(usize::try_from(*length).unwrap());
                file.take(*length).read_to_end(&mut buffer)?;
                buffer
            }
            ByteRange::FromEnd(offset, None) => {
                let mut buffer = Vec::with_capacity(usize::try_from(size - *offset).unwrap());
                file.take(size - offset).read_to_end(&mut buffer)?;
                buffer
            }
            ByteRange::FromEnd(offset, Some(length)) => {
                std::io::copy(
                    &mut file.by_ref().take(size - length - offset),
                    &mut std::io::sink(),
                )
                .unwrap();
                let mut buffer = Vec::with_capacity(usize::try_from(*length).unwrap());
                file.take(*length).read_to_end(&mut buffer)?;
                buffer
            }
        };

        Ok(buffer)
    }

    fn zip_file_strip_prefix<'a>(&self, name: &'a str) -> Option<&'a str> {
        name.strip_prefix(self.zip_path.to_str().unwrap())
            .filter(|&name| !name.is_empty())
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits
    for ZipStorageAdapter<TStorage>
{
    fn get(&self, key: &StoreKey) -> Result<Vec<u8>, StorageError> {
        self.get_impl(key, &ByteRange::FromStart(0, None))
    }

    fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Vec<Result<Vec<u8>, StorageError>> {
        let mut out = Vec::with_capacity(key_ranges.len());
        for key_range in key_ranges {
            out.push(self.get_impl(&key_range.key, &key_range.byte_range));
        }
        out
    }

    fn size(&self) -> Result<u64, StorageError> {
        Ok(self.size)
    }

    fn size_key(&self, key: &StoreKey) -> Result<u64, StorageError> {
        Ok(self
            .zip_archive
            .lock()
            .by_name(key.as_str())
            .map_err(|err| StorageError::Other(err.to_string()))?
            .compressed_size())
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ListableStorageTraits
    for ZipStorageAdapter<TStorage>
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(self
            .zip_archive
            .lock()
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
            .filter_map(|v| StoreKey::try_from(v).ok())
            .sorted()
            .collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut zip_archive = self.zip_archive.lock();
        let file_names: Vec<String> = zip_archive
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
            .map(std::string::ToString::to_string)
            .collect();
        Ok(file_names
            .into_iter()
            .filter_map(|name| {
                if name.starts_with(prefix.as_str()) {
                    let mut zip_name = self.zip_path.clone();
                    zip_name.push(&name);
                    if let Ok(file) = zip_archive.by_name(&zip_name.to_string_lossy()) {
                        if file.is_file() {
                            let name = name.strip_suffix('/').unwrap_or(&name);
                            if let Ok(store_key) = StoreKey::try_from(name) {
                                return Some(store_key);
                            }
                        }
                    }
                }
                None
            })
            .sorted()
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let zip_archive = self.zip_archive.lock();
        let mut keys: StoreKeys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        for name in zip_archive
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
        {
            if name.starts_with(prefix.as_str()) {
                if name.ends_with('/') {
                    if let Ok(store_prefix) = StorePrefix::try_from(name) {
                        if let Some(parent) = store_prefix.parent() {
                            if &parent == prefix {
                                prefixes.push(store_prefix);
                            }
                        }
                    }
                } else if let Ok(store_key) = StoreKey::try_from(name) {
                    if let Some(parent) = store_key.parent() {
                        if &parent == prefix {
                            keys.push(store_key);
                        }
                    }
                }
            }
        }
        keys.sort();
        prefixes.sort();

        Ok(StoreKeysPrefixes { keys, prefixes })
    }
}

/// A zip store creation error.
#[derive(Debug, Error)]
pub enum ZipStorageAdapterCreateError {
    /// An IO error.
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    /// An existing directory.
    #[error("{0} is an existing directory, not a zip file")]
    ExistingDir(PathBuf),
    /// A zip error.
    #[error(transparent)]
    ZipError(#[from] ZipError),
    /// A storage error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
}

#[cfg(test)]
mod tests {
    use walkdir::WalkDir;

    use crate::storage::{store::FilesystemStore, WritableStorageTraits};

    use super::*;
    use std::{
        error::Error,
        fs::File,
        io::{Read, Write},
        path::Path,
    };

    // https://github.com/zip-rs/zip/blob/master/examples/write_dir.rs
    fn zip_dir(
        it: &mut dyn Iterator<Item = walkdir::DirEntry>,
        prefix: &str,
        writer: File,
        method: zip::CompressionMethod,
    ) -> zip::result::ZipResult<()> {
        let mut zip = zip::ZipWriter::new(writer);
        let options = zip::write::FileOptions::default().compression_method(method);
        let mut buffer = Vec::new();
        for entry in it {
            let path = entry.path();
            let name = path.strip_prefix(Path::new(prefix)).unwrap();
            if path.is_file() {
                #[allow(deprecated)]
                zip.start_file_from_path(name, options)?;
                let mut f = File::open(path)?;
                f.read_to_end(&mut buffer)?;
                zip.write_all(&buffer)?;
                buffer.clear();
            } else if !name.as_os_str().is_empty() {
                #[allow(deprecated)]
                zip.add_directory_from_path(name, options)?;
            }
        }
        zip.finish()?;
        Result::Ok(())
    }

    fn zip_write(path: &Path) -> Result<(), Box<dyn Error>> {
        let tmp_path = tempfile::TempDir::new()?;
        let tmp_path = tmp_path.path();
        let store = FilesystemStore::new(tmp_path)?.sorted();
        store.set(&"a/b".try_into()?, &[0, 1, 2, 3])?;
        store.set(&"a/c".try_into()?, &[])?;
        store.set(&"a/d/e".try_into()?, &[])?;
        store.set(&"a/f/g".try_into()?, &[])?;
        store.set(&"a/f/h".try_into()?, &[])?;
        store.set(&"b/c/d".try_into()?, &[])?;
        store.set(&"c".try_into()?, &[])?;

        let walkdir = WalkDir::new(tmp_path);

        let file = File::create(path).unwrap();
        zip_dir(
            &mut walkdir.into_iter().filter_map(|e| e.ok()),
            tmp_path.to_str().unwrap(),
            file,
            zip::CompressionMethod::Stored,
        )?;

        Ok(())
    }

    #[test]
    fn zip_root() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let mut path = path.path().to_path_buf();
        path.push("test.zip");
        zip_write(&path).unwrap();

        println!("{path:?}");

        let store = FilesystemStore::new(path)?;
        let store = ZipStorageAdapter::new(store.into())?;

        assert_eq!(
            store.list()?,
            &[
                "a/b".try_into()?,
                "a/c".try_into()?,
                "a/d/e".try_into()?,
                "a/f/g".try_into()?,
                "a/f/h".try_into()?,
                "b/c/d".try_into()?,
                "c".try_into()?,
            ]
        );
        assert_eq!(
            store.list_prefix(&"a/".try_into()?)?,
            &[
                "a/b".try_into()?,
                "a/c".try_into()?,
                "a/d/e".try_into()?,
                "a/f/g".try_into()?,
                "a/f/h".try_into()?,
            ]
        );
        assert_eq!(
            store.list_prefix(&"a/d/".try_into()?)?,
            &["a/d/e".try_into()?]
        );
        assert_eq!(
            store.list_prefix(&"".try_into()?)?,
            &[
                "a/b".try_into()?,
                "a/c".try_into()?,
                "a/d/e".try_into()?,
                "a/f/g".try_into()?,
                "a/f/h".try_into()?,
                "b/c/d".try_into()?,
                "c".try_into()?,
            ]
        );

        let list = store.list_dir(&"a/".try_into()?)?;
        assert_eq!(list.keys(), &["a/b".try_into()?, "a/c".try_into()?]);
        assert_eq!(list.prefixes(), &["a/d/".try_into()?, "a/f/".try_into()?,]);

        assert!(crate::storage::node_exists(&store, &"/a/b".try_into()?)?);
        assert!(crate::storage::node_exists_listable(
            &store,
            &"/a/b".try_into()?
        )?);

        assert_eq!(store.get(&"a/b".try_into()?)?, &[0, 1, 2, 3]);
        assert_eq!(store.get(&"a/c".try_into()?)?, Vec::<u8>::new().as_slice());

        Ok(())
    }

    #[test]
    fn zip_path() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let mut path = path.path().to_path_buf();
        path.push("test.zip");
        zip_write(&path).unwrap();

        println!("{path:?}");

        let store = FilesystemStore::new(path)?;
        let store = ZipStorageAdapter::new_with_path(store.into(), "a/")?;

        assert_eq!(
            store.list()?,
            &[
                "b".try_into()?,
                "c".try_into()?,
                "d/e".try_into()?,
                "f/g".try_into()?,
                "f/h".try_into()?,
            ]
        );
        assert_eq!(store.list_prefix(&"a/".try_into()?)?, &[]);
        assert_eq!(store.list_prefix(&"d/".try_into()?)?, &["d/e".try_into()?]);
        assert_eq!(
            store.list_prefix(&"".try_into()?)?,
            &[
                "b".try_into()?,
                "c".try_into()?,
                "d/e".try_into()?,
                "f/g".try_into()?,
                "f/h".try_into()?,
            ]
        );

        let list = store.list_dir(&"".try_into()?)?;
        assert_eq!(list.keys(), &["b".try_into()?, "c".try_into()?]);
        assert_eq!(list.prefixes(), &["d/".try_into()?, "f/".try_into()?,]);

        assert!(crate::storage::node_exists(&store, &"/b".try_into()?)?);
        assert!(crate::storage::node_exists_listable(
            &store,
            &"/b".try_into()?
        )?);

        assert_eq!(store.get(&"b".try_into()?)?, &[0, 1, 2, 3]);
        // assert_eq!(store.get(&"c".try_into()?)?, Vec::<u8>::new().as_slice());

        Ok(())
    }
}
