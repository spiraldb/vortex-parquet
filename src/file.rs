// SPDX-License-Identifier: Apache-2.0

//! The file to be opened, and how to get one.
//!
//! [`FileHandle`] carries the three things opening a Parquet file needs: a random-access reader, the
//! file's length, and the session whose runtime drives the I/O. It exists as a type rather than as
//! three arguments so that the eager footer reads and the lazy segment reads are expressed against
//! the same object — the first through exact footer-range reads, the second through the V2 scan's
//! placed segment reader over [`read`](FileHandle::read).

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::filesystem::FileSystem;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::StdFileSystem;
use vortex::metrics::MetricsRegistry;
use vortex::session::VortexSession;

/// A [`FileSystem`] over the local filesystem, driven on `session`'s runtime.
///
/// A convenience for tests, benchmarks, and CLIs. Anything production-shaped should build its own
/// [`FileSystem`] and pass it to [`file_handle`], so that credentials, retry policy, and concurrency
/// limits stay under the caller's control.
pub fn local_filesystem(session: &VortexSession) -> Arc<dyn FileSystem> {
    Arc::new(StdFileSystem::new(session.handle()))
}

/// Resolves `path` on `fs` to a handle, asking for the file's size.
pub async fn file_handle(
    path: &str,
    fs: &dyn FileSystem,
    session: &VortexSession,
) -> VortexResult<FileHandle> {
    let listing = fs
        .head(path)
        .await?
        .ok_or_else(|| vortex_err!("no such file: {path:?}"))?;
    let read: Arc<dyn VortexReadAt> = fs.open_read(path).await?;

    // `head` is allowed to omit the size; fall back to asking the reader directly.
    let size = match listing.size {
        Some(size) => size,
        None => read.size().await?,
    };

    Ok(FileHandle::new(read, size, session.clone()))
}

/// A file to be opened, plus the context needed to read it.
///
/// Cheap to clone: the reader and session are reference-counted.
#[derive(Clone)]
pub struct FileHandle {
    read: Arc<dyn VortexReadAt>,
    size: u64,
    session: VortexSession,
    metrics: Option<Arc<dyn MetricsRegistry>>,
}

impl Debug for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHandle")
            .field("uri", &self.uri())
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl FileHandle {
    /// Creates a handle over `read`, a file of `size` bytes.
    ///
    /// Prefer [`file_handle`], which resolves the size for you. Construct a handle directly when the
    /// bytes come from somewhere a filesystem cannot address — an in-memory buffer in a test, or a
    /// bespoke storage layer.
    pub fn new(read: Arc<dyn VortexReadAt>, size: u64, session: VortexSession) -> Self {
        Self {
            read,
            size,
            session,
            metrics: None,
        }
    }

    /// Attaches a metrics registry, which the segment source will use to report I/O counters.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The underlying random-access reader.
    pub fn read(&self) -> &Arc<dyn VortexReadAt> {
        &self.read
    }

    /// The size of the file in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The URI of the file, if the reader knows one.
    pub fn uri(&self) -> Option<&Arc<str>> {
        self.read.uri()
    }

    /// The session this file is being opened in.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// The metrics registry, if one was attached.
    pub fn metrics_registry(&self) -> Option<&Arc<dyn MetricsRegistry>> {
        self.metrics.as_ref()
    }

    /// Reads an exact byte range eagerly.
    ///
    /// This bypasses segment coalescing and is for the metadata reads that must complete before the
    /// file can be described at all — the footer. Data reads belong in a [`SegmentTable`] so the
    /// driver can coalesce and pipeline them.
    pub async fn read_range(&self, byte_range: Range<u64>) -> VortexResult<ByteBuffer> {
        if byte_range.end < byte_range.start {
            vortex_bail!(
                "read_range: inverted byte range {}..{}",
                byte_range.start,
                byte_range.end
            );
        }
        if byte_range.end > self.size {
            vortex_bail!(
                "read_range: {}..{} extends past the end of the {}-byte file{}",
                byte_range.start,
                byte_range.end,
                self.size,
                self.uri()
                    .map(|uri| format!(" at {uri}"))
                    .unwrap_or_default()
            );
        }

        let length = usize::try_from(byte_range.end - byte_range.start)?;
        Ok(self
            .read
            .read_at(byte_range.start, length, Alignment::none())
            .await?
            .unwrap_host())
    }

    /// Reads the last `length` bytes of the file, or the whole file if it is shorter.
    ///
    /// The usual first move for a footer-oriented format: grab a suffix large enough to contain the
    /// footer and parse it from the buffer rather than issuing a size read followed by a second,
    /// exact read.
    pub async fn read_suffix(&self, length: u64) -> VortexResult<ByteBuffer> {
        let length = length.min(self.size);
        self.read_range(self.size - length..self.size).await
    }
}
