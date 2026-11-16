// SPDX-License-Identifier: Apache-2.0

//! Connection-scoped parsed footers. Each bind still opens and validates a live file handle;
//! cached entries own neither file handles nor data readers.

use std::fs::Metadata;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex::utils::aliases::hash_map::HashMap;
use vortex_parquet::PreparedParquetMetadata;

const MAX_ENTRIES: usize = 128;
const MAX_METADATA_BYTES: usize = 64 * 1024 * 1024;

/// An opened file's identity and change stamps. Size/mtime alone miss same-size rewrites whose
/// modification time was restored, and inode identity distinguishes atomic replacements.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FileVersion {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl FileVersion {
    pub(crate) fn from_metadata(metadata: &Metadata) -> Option<Self> {
        #[cfg(unix)]
        {
            Some(Self {
                device: metadata.dev(),
                inode: metadata.ino(),
                size: metadata.len(),
                modified: (metadata.mtime(), metadata.mtime_nsec()),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            })
        }
        #[cfg(not(unix))]
        {
            // Bypass caching where this adapter does not have a reliable file-version key.
            let _ = metadata;
            None
        }
    }
}

/// Footer-derived bounds contain no readers, file handles, or decoded data.
pub(crate) struct CachedMetadata {
    pub(crate) prepared: PreparedParquetMetadata,
    pub(crate) bounds: Vec<OnceLock<Option<(i128, i128)>>>,
}

impl CachedMetadata {
    pub(crate) fn new(prepared: PreparedParquetMetadata) -> Self {
        let columns = prepared
            .metadata()
            .file_metadata()
            .schema_descr()
            .root_schema()
            .get_fields()
            .len();
        Self {
            prepared,
            bounds: (0..columns).map(|_| OnceLock::new()).collect(),
        }
    }

    fn memory_size(&self) -> usize {
        self.prepared.memory_size()
            + self.bounds.capacity() * size_of::<OnceLock<Option<(i128, i128)>>>()
    }
}

struct Entry {
    metadata: Arc<CachedMetadata>,
    bytes: usize,
    accessed: u64,
}

#[derive(Default)]
pub(crate) struct MetadataCache {
    entries: HashMap<FileVersion, Entry>,
    bytes: usize,
    clock: u64,
}

impl MetadataCache {
    pub(crate) fn get(&mut self, version: FileVersion) -> Option<Arc<CachedMetadata>> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&version)?;
        entry.accessed = self.clock;
        Some(Arc::clone(&entry.metadata))
    }

    pub(crate) fn insert(&mut self, version: FileVersion, metadata: Arc<CachedMetadata>) {
        let bytes = metadata.memory_size();
        if bytes > MAX_METADATA_BYTES {
            return;
        }
        if let Some(old) = self.entries.remove(&version) {
            self.bytes -= old.bytes;
        }
        while self.entries.len() >= MAX_ENTRIES || self.bytes + bytes > MAX_METADATA_BYTES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.accessed)
                .map(|(version, _)| *version)
            else {
                break;
            };
            if let Some(old) = self.entries.remove(&oldest) {
                self.bytes -= old.bytes;
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.bytes += bytes;
        self.entries.insert(
            version,
            Entry {
                metadata,
                bytes,
                accessed: self.clock,
            },
        );
    }
}
