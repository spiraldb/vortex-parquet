// SPDX-License-Identifier: Apache-2.0

//! Where a run reads its data from: a local file, or an object store.
//!
//! # Why this is a type rather than a path
//!
//! Every measurement in this benchmark harness has been taken against a local file with the data resident in the
//! page cache, which makes it a *compute* benchmark. The one design in the reader that exists purely
//! for object storage — the coalescing I/O driver, and the `CoalesceConfig::object_storage()` it
//! defaults to — has therefore never run in the regime it was written for. Reading `gs://…` is what
//! closes that gap, and it needs more than a different string: a different filesystem for the Vortex
//! arms, a registered object store for the DataFusion arm, and an honest answer for the operations
//! that only mean something locally.

use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use datafusion::prelude::SessionContext;
use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;
use url::Url;
use vortex::io::filesystem::FileSystem;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::StdFileSystem;
use vortex::session::VortexSession;

/// The location of one file a run reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A path on the local filesystem.
    Local(PathBuf),
    /// An object in a store, addressed by URL.
    Remote {
        /// The full URL, e.g. `gs://bucket/hits.parquet`.
        url: Url,
        /// The bucket or container name.
        bucket: String,
        /// The object's key within the bucket, with no leading slash.
        key: String,
    },
}

impl Source {
    /// Parses a command-line location.
    ///
    /// Anything without a `://` is a local path, which keeps every existing invocation working and
    /// means a Windows drive letter or a relative path is never mistaken for a scheme.
    pub fn parse(raw: &str) -> Result<Self> {
        if !raw.contains("://") {
            return Ok(Self::Local(PathBuf::from(raw)));
        }

        let url = Url::parse(raw).with_context(|| format!("{raw:?} is not a valid URL"))?;
        match url.scheme() {
            "file" => Ok(Self::Local(PathBuf::from(url.path()))),
            "gs" => {
                let bucket = url
                    .host_str()
                    .with_context(|| format!("{raw:?} has no bucket"))?
                    .to_string();
                let key = url.path().trim_start_matches('/').to_string();
                if key.is_empty() {
                    bail!("{raw:?} names a bucket but no object");
                }
                Ok(Self::Remote { url, bucket, key })
            }
            other => bail!(
                "unsupported scheme {other:?}; this benchmark harness reads local paths and gs://"
            ),
        }
    }

    /// The same location with its extension replaced, for finding a file's `.vortex` sibling.
    pub fn with_extension(&self, extension: &str) -> Self {
        match self {
            Self::Local(path) => Self::Local(path.with_extension(extension)),
            Self::Remote { url, bucket, key } => {
                let stem = key.rsplit_once('.').map_or(key.as_str(), |(stem, _)| stem);
                let key = format!("{stem}.{extension}");
                let mut url = url.clone();
                url.set_path(&key);
                Self::Remote {
                    url,
                    bucket: bucket.clone(),
                    key,
                }
            }
        }
    }

    /// The path to hand [`FileSystem::open_read`], which is store-relative for an object store.
    pub fn open_path(&self) -> Result<String> {
        match self {
            Self::Local(path) => Ok(path
                .to_str()
                .with_context(|| format!("{} is not valid UTF-8", path.display()))?
                .to_string()),
            Self::Remote { key, .. } => Ok(key.clone()),
        }
    }

    /// The filesystem this location is read through.
    ///
    /// Credentials for `gs://` come from the ambient environment — `GOOGLE_APPLICATION_CREDENTIALS`,
    /// gcloud application-default credentials, or a GCE instance's metadata server. Nothing is read
    /// from a flag or a config file here, so a credential cannot end up in a run's recorded command.
    pub fn filesystem(&self, session: &VortexSession) -> Result<Arc<dyn FileSystem>> {
        match self {
            Self::Local(_) => Ok(Arc::new(StdFileSystem::new(session.handle()))),
            Self::Remote { bucket, .. } => {
                let store = self.object_store()?;
                let _ = bucket;
                Ok(Arc::new(ObjectStoreFileSystem::new(
                    store,
                    session.handle(),
                )))
            }
        }
    }

    /// The `object_store` client for this location, or an error if it is local.
    pub fn object_store(&self) -> Result<Arc<dyn ObjectStore>> {
        match self {
            Self::Local(_) => bail!("a local path has no object store"),
            Self::Remote { bucket, .. } => Ok(Arc::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket)
                    .build()
                    .with_context(|| format!("failed to open the GCS bucket {bucket:?}"))?,
            )),
        }
    }

    /// Registers this location's object store with DataFusion, so `gs://` resolves in a plan.
    ///
    /// A no-op for a local path, which DataFusion already handles.
    pub fn register_object_store(&self, ctx: &SessionContext) -> Result<()> {
        if let Self::Remote { url, .. } = self {
            // Registered against the bucket root rather than the object: DataFusion matches a store
            // by scheme and authority, and registering per object would miss the sibling files the
            // reference arms read.
            let mut root = url.clone();
            root.set_path("");
            ctx.register_object_store(&root, self.object_store()?);
        }
        Ok(())
    }

    /// The URL DataFusion's `ListingTable` should be pointed at.
    pub fn listing_url(&self) -> Result<String> {
        match self {
            Self::Local(path) => Ok(format!(
                "file://{}",
                path.to_str()
                    .with_context(|| format!("{} is not valid UTF-8", path.display()))?
            )),
            Self::Remote { url, .. } => Ok(url.to_string()),
        }
    }

    /// The local path, if this is one.
    ///
    /// Used by the operations that are only meaningful against a real file — evicting it from the
    /// page cache, and reading its footer with a plain `File`.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            Self::Local(path) => Some(path),
            Self::Remote { .. } => None,
        }
    }

    /// Whether the file exists, asking the store rather than assuming.
    ///
    /// Errors are propagated rather than folded into `false`. A missing credential, a denied bucket
    /// or a transport failure is not the same as a missing file, and reporting them as "no such file"
    /// sends the reader looking for a typo in a path that is correct.
    pub async fn exists(&self, session: &VortexSession) -> Result<bool> {
        match self {
            // `Path::exists` folds a permission or I/O error into `false`, which is the same fault
            // this method was changed to stop committing on the remote side.
            Self::Local(path) => match std::fs::metadata(path) {
                Ok(_) => Ok(true),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(e).with_context(|| format!("failed to look up {}", path.display())),
            },
            Self::Remote { .. } => {
                let fs = self.filesystem(session)?;
                let path = self.open_path()?;
                let found = fs
                    .head(&path)
                    .await
                    .with_context(|| format!("failed to look up {self}"))?;
                Ok(found.is_some())
            }
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local(path) => write!(f, "{}", path.display()),
            Self::Remote { url, .. } => write!(f, "{url}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare path is local, whatever it contains.
    #[test]
    fn a_path_without_a_scheme_is_local() {
        assert_eq!(
            Source::parse("data/hits.parquet").unwrap(),
            Source::Local(PathBuf::from("data/hits.parquet"))
        );
    }

    /// A `gs://` URL splits into bucket and key, with no leading slash on the key.
    #[test]
    fn a_gs_url_splits_into_bucket_and_key() {
        let source = Source::parse("gs://example-bucket/clickbench/hits.parquet").unwrap();
        let Source::Remote { bucket, key, .. } = &source else {
            panic!("expected a remote source");
        };
        assert_eq!(bucket, "example-bucket");
        assert_eq!(key, "clickbench/hits.parquet");
    }

    /// The sibling lookup the reference arms depend on must work in a bucket too.
    ///
    /// Getting this wrong is not an error, it is a *missing file* — and the reference arms would then
    /// be skipped rather than fail, quietly removing the comparison they exist to provide.
    #[test]
    fn a_remote_sibling_replaces_the_extension() {
        let source = Source::parse("gs://example-bucket/clickbench/hits.parquet").unwrap();
        assert_eq!(
            source.with_extension("vortex").to_string(),
            "gs://example-bucket/clickbench/hits.vortex"
        );
    }

    /// A bucket with no object is a mistake worth catching at parse time.
    #[test]
    fn a_bucket_without_an_object_is_rejected() {
        assert!(Source::parse("gs://example-bucket").is_err());
        assert!(Source::parse("gs://example-bucket/").is_err());
    }

    /// An unknown scheme is rejected rather than silently treated as a path.
    #[test]
    fn an_unsupported_scheme_is_rejected() {
        assert!(Source::parse("s3://bucket/key").is_err());
    }
}
