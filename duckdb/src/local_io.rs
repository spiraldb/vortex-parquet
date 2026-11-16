// SPDX-License-Identifier: Apache-2.0

//! Small positional local reads driven by DuckDB's scan workers.
//!
//! The bridge runs a current-thread Vortex runtime on each DuckDB worker. Local reads can run on
//! that same worker rather than parking it while a second thread pool performs the system call.
//! Larger ranges retain Vortex's I/O pool so column reads overlap decoding on the scan workers.

use std::fs::File;
use std::fs::Metadata;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::BufferMut;
use vortex::error::VortexResult;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::DEFAULT_CONCURRENCY;
use vortex::io::std_file::read_exact_at;

const INLINE_READ_BYTES: usize = 128 * 1024;

pub(crate) struct WorkerFileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
}

impl WorkerFileReadAt {
    pub(crate) fn metadata(&self) -> std::io::Result<Metadata> {
        self.file.metadata()
    }

    pub(crate) fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let path = path.as_ref();
        Ok(Self {
            uri: path.to_string_lossy().into_owned().into(),
            file: Arc::new(File::open(path)?),
            handle,
        })
    }
}

impl VortexReadAt for WorkerFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        // Match DuckDB's local Parquet prefetch gap: widely separated projected columns should
        // not pull megabytes of unrelated data into each read. Keep large adjacent reads possible.
        Some(CoalesceConfig::new(16 * 1024, 4 * 1024 * 1024))
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(&self.file);
        async move { Ok(file.metadata()?.len()) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        async move {
            let read = move || -> VortexResult<BufferHandle> {
                let mut buffer = BufferMut::<u8>::with_capacity_aligned(length, alignment);
                // SAFETY: as in Vortex's FileReadAt, read_exact_at initializes all bytes before the
                // buffer is exposed. An error only drops the u8 allocation without reading its bytes.
                unsafe { buffer.set_len(length) };
                read_exact_at(&file, buffer.as_mut_slice(), offset)?;
                Ok(BufferHandle::new_host(buffer.freeze()))
            };
            if length <= INLINE_READ_BYTES {
                read()
            } else {
                handle.spawn_blocking(read).await
            }
        }
        .boxed()
    }
}
