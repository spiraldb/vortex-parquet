// SPDX-License-Identifier: Apache-2.0

//! Reader selection, request concurrency, and coalescing for benchmark I/O.
//!
//! Local paths use `FileReadAt`; remote paths use object storage. The legacy `object-store` option
//! delegates to the source filesystem, whose local implementation also uses `FileReadAt` at the
//! pinned Vortex revision. It is not a distinct local backend.
//!
//! Concurrency and coalescing overrides are reported with each run. I/O arms interleave different
//! settings within one run so comparisons are less sensitive to machine drift.

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use clap::ValueEnum;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::session::VortexSession;

use crate::source::Source;

/// Environment variable setting the maximum gap bridged when coalescing, in KiB.
pub const COALESCE_DISTANCE_ENV: &str = "BENCHMARKS_COALESCE_DISTANCE_KB";

/// Environment variable setting the maximum coalesced span, in KiB.
pub const COALESCE_MAX_ENV: &str = "BENCHMARKS_COALESCE_MAX_KB";

/// Environment variable setting the maximum number of in-flight read requests.
pub const CONCURRENCY_ENV: &str = "BENCHMARKS_IO_CONCURRENCY";

/// Which reader a local file is opened through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    /// [`Backend::File`] for local paths, [`Backend::ObjectStore`] for remote ones.
    ///
    /// The default, and the only setting that is right for both: a local file wants one descriptor
    /// and a modest queue, a bucket wants a deep one.
    #[default]
    Auto,
    /// `FileReadAt`: one open descriptor, `pread`, `CoalesceConfig::file()`, 32 in flight.
    ///
    /// Local paths only — there is no descriptor to hold for an object store.
    File,
    /// Delegate to [`Source::filesystem`]: object storage remotely, `StdFileSystem` locally.
    /// At the pinned Vortex revision, local files still use `FileReadAt`.
    ObjectStore,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::File => "file",
            Self::ObjectStore => "object-store",
        })
    }
}

/// The I/O configuration a run measures under.
#[derive(Clone, Copy, Debug, Default)]
pub struct IoConfig {
    /// Which reader backend to open with.
    pub backend: Backend,
    /// Maximum in-flight read requests, or the backend's own default.
    pub concurrency: Option<usize>,
    /// Maximum gap bridged when coalescing, in bytes, or the backend's own default.
    pub coalesce_distance: Option<u64>,
    /// Maximum coalesced span, in bytes, or the backend's own default.
    pub coalesce_max: Option<u64>,
}

impl IoConfig {
    /// Fills unset knobs from the environment, so an existing sweep script keeps working.
    ///
    /// Resolution order is **flag, then environment, then the backend's own default**. An explicit
    /// flag wins because it is the more visible of the two, and a value that cannot be parsed is an
    /// error rather than a silent fallback — the same rule [`crate::run::parse_env`] applies, and for
    /// the same reason: a knob that reads as `default` in the header while a typo'd value was ignored
    /// is a run claiming a configuration it did not use.
    pub fn resolve_env(mut self) -> Result<Self> {
        let kib = |name: &str| -> Result<Option<u64>> {
            crate::run::parse_env::<u64>(name)?
                .map(|kb| {
                    kb.checked_mul(1 << 10)
                        .with_context(|| format!("{name}={kb} KiB overflows"))
                })
                .transpose()
        };
        if self.coalesce_distance.is_none() {
            self.coalesce_distance = kib(COALESCE_DISTANCE_ENV)?;
        }
        if self.coalesce_max.is_none() {
            self.coalesce_max = kib(COALESCE_MAX_ENV)?;
        }
        if self.concurrency.is_none() {
            self.concurrency = crate::run::parse_env::<usize>(CONCURRENCY_ENV)?;
        }
        // Zero would stall I/O outright — `FileSegmentSource` panics on it rather than deadlocking,
        // which is better, but a benchmark should refuse it here with a sentence instead.
        if self.concurrency == Some(0) {
            anyhow::bail!("an I/O concurrency of 0 would stall every read");
        }
        Ok(self)
    }

    /// Whether to open [`FileReadAt`] directly rather than delegate to the source's filesystem.
    fn uses_file_backend(&self, source: &Source) -> bool {
        match self.backend {
            Backend::ObjectStore => false,
            Backend::File | Backend::Auto => source.local_path().is_some(),
        }
    }

    /// Opens `source` for reading, applying the backend choice and any overrides.
    pub async fn open_read(
        &self,
        source: &Source,
        session: &VortexSession,
    ) -> Result<Arc<dyn VortexReadAt>> {
        let read: Arc<dyn VortexReadAt> = if self.uses_file_backend(source) {
            let path = source
                .local_path()
                .ok_or_else(|| anyhow::anyhow!("{source} is not a local path"))?;
            Arc::new(
                FileReadAt::open(path, session.handle())
                    .with_context(|| format!("failed to open {}", path.display()))?,
            )
        } else {
            if self.backend == Backend::File {
                anyhow::bail!(
                    "--io file reads a local file through one descriptor; {source} is not local. Use \
                     --io auto, which picks the object store for remote paths."
                );
            }
            let fs = source.filesystem(session)?;
            let path = source.open_path()?;
            fs.open_read(&path)
                .await
                .with_context(|| format!("failed to open {path}"))?
        };
        Ok(self.with_overrides(read))
    }

    /// Wraps `read` to declare overridden coalescing and concurrency, if any were set.
    ///
    /// Whichever knob is unset keeps the reader's own value, so a sweep of one moves one thing. If
    /// none is set the reader is returned untouched rather than rewrapped with its own values, so the
    /// unswept path stays byte-identical to no wrapper at all.
    fn with_overrides(&self, read: Arc<dyn VortexReadAt>) -> Arc<dyn VortexReadAt> {
        if self.coalesce_distance.is_none()
            && self.coalesce_max.is_none()
            && self.concurrency.is_none()
        {
            return read;
        }
        // The reader's own choice is the baseline an unset knob falls back to. A reader that declines
        // to declare a coalesce config is treated as the object-storage case rather than silently
        // given something else.
        let base = read
            .coalesce_config()
            .unwrap_or_else(CoalesceConfig::object_storage);
        let coalesce = CoalesceConfig::new(
            self.coalesce_distance.unwrap_or(base.distance),
            self.coalesce_max.unwrap_or(base.max_size),
        );
        let concurrency = self.concurrency.unwrap_or_else(|| read.concurrency());
        Arc::new(Overridden {
            inner: read,
            coalesce,
            concurrency,
        })
    }

    /// A one-line description of the I/O configuration, for the report header.
    ///
    /// Resolved rather than echoed: `default` would leave a reader unable to tell which backend was
    /// measured, and the backend is now the setting most likely to make two runs incomparable.
    pub fn describe(&self, source: &Source) -> String {
        let effective = if self.uses_file_backend(source) {
            "file"
        } else {
            "object-store"
        };
        let render = |value: Option<u64>| match value {
            Some(bytes) => format!("{}KiB", bytes / (1 << 10)),
            None => "backend-default".to_string(),
        };
        format!(
            "backend={} (--io {}) concurrency={} coalesce_distance={} coalesce_max={}",
            effective,
            self.backend,
            self.concurrency
                .map_or_else(|| "backend-default".to_string(), |c| c.to_string()),
            render(self.coalesce_distance),
            render(self.coalesce_max),
        )
    }
}

/// A reader that declares different coalescing and concurrency from the one it wraps.
struct Overridden {
    inner: Arc<dyn VortexReadAt>,
    coalesce: CoalesceConfig,
    concurrency: usize,
}

impl VortexReadAt for Overridden {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(self.coalesce)
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> futures::future::BoxFuture<'static, VortexResult<u64>> {
        self.inner.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> futures::future::BoxFuture<'static, VortexResult<BufferHandle>> {
        self.inner.read_at(offset, length, alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `auto` selects the direct file path locally and the source filesystem remotely.
    #[test]
    fn auto_picks_the_file_backend_only_for_local_paths() {
        let local = Source::Local("/tmp/hits.parquet".into());
        let remote = Source::parse("gs://bucket/hits.parquet").expect("parses");

        let auto = IoConfig::default();
        assert!(auto.uses_file_backend(&local));
        assert!(!auto.uses_file_backend(&remote));
    }

    /// The legacy object-store option delegates local paths to the source filesystem too.
    #[test]
    fn object_store_can_be_forced_for_a_local_file() {
        let local = Source::Local("/tmp/hits.parquet".into());
        let forced = IoConfig {
            backend: Backend::ObjectStore,
            ..IoConfig::default()
        };
        assert!(!forced.uses_file_backend(&local));
    }

    /// An unconfigured run must not wrap the reader at all, so the default path is byte-identical to
    /// having no override machinery.
    #[test]
    fn an_unset_configuration_does_not_wrap_the_reader() {
        let io = IoConfig::default();
        assert!(io.coalesce_distance.is_none() && io.coalesce_max.is_none());
        assert!(io.concurrency.is_none());
    }

    /// The header must name the backend that actually ran, not the flag that was typed: `auto` is
    /// two different configurations depending on the path, and a run that reports `auto` for both is
    /// unreadable later.
    #[test]
    fn the_description_resolves_auto_to_what_actually_ran() {
        let io = IoConfig::default();
        let local = io.describe(&Source::Local("/tmp/hits.parquet".into()));
        let remote = io.describe(&Source::parse("gs://bucket/hits.parquet").expect("parses"));

        assert!(local.contains("backend=file"), "{local}");
        assert!(remote.contains("backend=object-store"), "{remote}");
        // The flag as typed is kept alongside it, so the command can be reconstructed.
        assert!(local.contains("--io auto"));
    }

    /// A concurrency of zero stalls every read, so it is refused rather than passed down to the
    /// panic inside `FileSegmentSource`.
    #[test]
    fn a_zero_concurrency_is_refused() {
        let io = IoConfig {
            concurrency: Some(0),
            ..IoConfig::default()
        };
        assert!(io.resolve_env().is_err());
    }
}
