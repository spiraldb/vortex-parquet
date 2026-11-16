// SPDX-License-Identifier: Apache-2.0

//! The thing being compared: a way to make a Parquet file queryable in DataFusion.
//!
//! An arm is deliberately thin — "register table `hits` from this file into this DataFusion
//! context" — because that is the entire difference between the things being compared. Everything
//! downstream (planning, filtering, aggregation, timing) is identical across arms, so a difference in
//! the numbers is a difference in how bytes became Arrow batches.
//!
//! # The ladder
//!
//! Reading a Parquet file into a query engine has two ownership questions, and they are independent:
//!
//! * **who schedules** — splits, row-group selection, mask refinement, predicate pushdown
//! * **who decodes** — a column chunk's compressed pages into an in-memory array, and at what
//!   granularity
//!
//! A single "how internalized is it" ladder conflates the two, and then a difference between the ends
//! cannot be attributed to either. The benchmark therefore compares the two complete readers:
//!
//! | arm | schedules | decodes | isolates |
//! |---|---|---|---|
//! | [`DataFusionParquet`] | arrow-rs | arrow-rs | the baseline |
//! | [`VortexParquet`] | Vortex | lazy Parquet page arrays; arrow-rs for nested Dremel fields | the canonical adapter |
//!
//! # The reference arms
//!
//! [`VortexScanApi`] and [`VortexListing`] read the **same rows stored as a Vortex file**. They are
//! not rungs on the ladder — they are the ceiling it is being measured against, and they answer a
//! question no Parquet arm can: how much of what a Vortex arm pays is the *format*, and how much is
//! this crate's adapter?
//!
//! [`VortexScanApi`] is deliberately the same table provider a [`VortexParquet`] arm uses, so with
//! the provider held constant its difference is attributable to the file format. [`VortexListing`] is
//! the older `VortexFormat` + `ListingTable` integration, which is the path Vortex's own published
//! numbers use. Running both separates "how fast is the format" from "how fast is this particular
//! DataFusion glue" — alone, either one conflates them.
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::prelude::SessionContext;
use vortex::VortexSessionDefault;
use vortex::arrow::ArrowSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexTableOptions;
use vortex_datafusion::v2::VortexTable;
use vortex_parquet::ParquetFormat;
use vortex_parquet::file::FileHandle;

use crate::io::IoConfig;
use crate::source::Source;

/// Something that can make a Parquet file queryable in DataFusion under a given table name.
#[async_trait]
pub trait Arm: Send + Sync {
    /// A short label used in reports, e.g. `"vortex-parquet"`.
    fn name(&self) -> &str;

    /// Registers `table` in `ctx`, reading from `path`.
    async fn register(&self, ctx: &SessionContext, table: &str, source: &Source) -> Result<()>;

    /// The file this arm actually reads, given the Parquet file the run was pointed at.
    ///
    /// Almost every arm reads that same file — that is what makes the comparison a comparison. The
    /// reference arms are the exception: they read the *same rows* stored as a Vortex file, which is
    /// the whole point of having them, so they map the path to its `.vortex` sibling.
    fn data_path(&self, source: &Source) -> Source {
        source.clone()
    }

    /// Drops any state this arm caches between queries.
    ///
    /// Only meaningful for arms that cache across scans. The default does nothing, which is right for
    /// an arm whose caching all lives inside DataFusion — those are reset by re-registering instead.
    fn flush(&self) {}

    /// Which optional code paths this arm took, if it counts any.
    ///
    /// In the trait rather than left to each arm because the question generalises: an arm's
    /// interesting decisions are *invisible* from the outside, since every path returns the same
    /// rows. A benchmark arm that turns one on therefore proves nothing about whether it ran, and a
    /// correctness test over it passes either way.
    ///
    /// Reported once after a run rather than per query, because the counts are cumulative and answer
    /// "did the path run" rather than anything comparable across queries.
    fn counters(&self) -> Option<String> {
        None
    }
}

/// A shared arm.
pub type ArmRef = Arc<dyn Arm>;

// --- Baseline ------------------------------------------------------------------------------------

/// DataFusion's own Parquet reader. **The baseline.**
///
/// Without it every other number is unanchored: "so many milliseconds" means nothing without the
/// number you would otherwise have got.
#[derive(Debug, Default)]
pub struct DataFusionParquet;

#[async_trait]
impl Arm for DataFusionParquet {
    fn name(&self) -> &str {
        "datafusion-parquet"
    }

    async fn register(&self, ctx: &SessionContext, table: &str, source: &Source) -> Result<()> {
        source.register_object_store(ctx)?;
        ctx.register_parquet(table, source.listing_url()?, Default::default())
            .await?;
        Ok(())
    }
}

// --- Vortex schedules ----------------------------------------------------------------------------

/// This crate: Vortex's scan engine over a Parquet file, decoding one top-level column at a time.
///
/// The chain is `ParquetFormat -> LayoutReader -> LayoutReaderDataSource -> VortexTable`, all of it
/// upstream machinery except the leaf.
pub struct VortexParquet {
    format: ParquetFormat,
    label: String,
    io: IoConfig,
}

impl VortexParquet {
    /// Wraps `format` as an arm labelled `label`, reading under `io`.
    pub fn new(format: ParquetFormat, label: impl Into<String>, io: IoConfig) -> Self {
        Self {
            format,
            label: label.into(),
            io,
        }
    }

    /// The default arm, reading under `io`.
    pub fn with_io(io: IoConfig) -> Self {
        Self::new(ParquetFormat::new(), "vortex-parquet", io)
    }
}

#[async_trait]
impl Arm for VortexParquet {
    fn name(&self) -> &str {
        &self.label
    }

    fn flush(&self) {
        self.format.flush_caches();
    }

    fn counters(&self) -> Option<String> {
        Some(self.format.metrics().to_string())
    }

    async fn register(&self, ctx: &SessionContext, table: &str, source: &Source) -> Result<()> {
        let session = VortexSession::default().with_tokio();
        let path = source.open_path()?;

        let read = self.io.open_read(source, &session).await?;
        let size = read.size().await?;
        let data_source = self
            .format
            .open_data_source(FileHandle::new(read, size, session.clone()))
            .await
            .with_context(|| format!("vortex-parquet failed to open {path}"))?;

        // The schema DataFusion observes, derived from the data source's own dtype rather than
        // re-read from the file, so the two cannot disagree.
        let arrow_schema = Arc::new(session.arrow().to_arrow_schema(data_source.dtype())?);
        ctx.register_table(
            table,
            Arc::new(VortexTable::new(data_source, session, arrow_schema)),
        )?;
        Ok(())
    }
}

// --- Reference: the same rows, stored as Vortex --------------------------------------------------

/// Maps a Parquet path to the Vortex file holding the same rows.
fn vortex_sibling(source: &Source) -> Source {
    source.with_extension("vortex")
}

/// A Vortex file read through the **v2 Scan API**, behind the same [`VortexTable`] the Parquet arms
/// use.
///
/// The file is opened once at registration and handed to DataFusion as an already-open `DataSource` —
/// the same machinery a foreign-format adapter is forced onto. With the table provider held constant,
/// a difference against [`VortexParquet`] is attributable to the file format rather than to the
/// integration.
#[derive(Debug, Default)]
pub struct VortexScanApi {
    io: IoConfig,
}

impl VortexScanApi {
    /// The arm, reading under `io`.
    pub fn with_io(io: IoConfig) -> Self {
        Self { io }
    }
}

#[async_trait]
impl Arm for VortexScanApi {
    fn name(&self) -> &str {
        "vortex-scan-api"
    }

    fn data_path(&self, source: &Source) -> Source {
        vortex_sibling(source)
    }

    async fn register(&self, ctx: &SessionContext, table: &str, source: &Source) -> Result<()> {
        use vortex::file::OpenOptionsSessionExt;

        let session = VortexSession::default().with_tokio();
        // Opened through the same configuration the Parquet arms use, so the I/O paths stay
        // comparable and a backend change moves both arms together.
        let read = self.io.open_read(source, &session).await?;
        let file = session.open_options().open(read).await?;
        // Through the wrapper, so the planner sees the statistics the file already carries. Without
        // it TPC-DS q72 plans a hash join from the wrong side and never finishes; see
        // [`crate::file_stats`].
        let data_source = crate::file_stats::FooterStatistics::wrap(&file)?;
        let arrow_schema = Arc::new(session.arrow().to_arrow_schema(data_source.dtype())?);
        ctx.register_table(
            table,
            Arc::new(VortexTable::new(data_source, session, arrow_schema)),
        )?;
        Ok(())
    }
}

/// The same Vortex file read through **`VortexFormat` + `ListingTable`**.
///
/// Here DataFusion owns the file: it does the listing, opens the file itself, and asks the format for
/// statistics. This is the older and more developed integration, and the path Vortex's published
/// benchmarks use. Identical bytes to [`VortexScanApi`] — only the glue differs.
#[derive(Debug, Default)]
pub struct VortexListing;

#[async_trait]
impl Arm for VortexListing {
    fn name(&self) -> &str {
        "vortex-listing"
    }

    fn data_path(&self, source: &Source) -> Source {
        vortex_sibling(source)
    }

    async fn register(&self, ctx: &SessionContext, table: &str, source: &Source) -> Result<()> {
        let session = VortexSession::default().with_tokio();

        // Both pushdowns are enabled explicitly. `projection_pushdown` defaults to *false*, so a
        // default `VortexFormat` evaluates every projection after the scan — benchmarking it that way
        // measures a configuration nobody would deploy.
        let mut options = VortexTableOptions::default();
        options.projection_pushdown = true;
        options.predicate_pushdown = true;
        let format = Arc::new(VortexFormat::new_with_options(session, options));

        // `collect_stat` defaults to *false*, and without it a `ListingTable` never asks the format
        // for statistics. The table then reports none at all, so `COUNT(*)` and `MIN`/`MAX` fall back
        // to scanning — which presents as the format being slow when the harness is what is
        // misconfigured.
        let listing_options = ListingOptions::new(format).with_file_extension(".vortex");
        source.register_object_store(ctx)?;
        let url = ListingTableUrl::parse(source.listing_url()?)?;
        let resolved = listing_options
            .infer_schema(&ctx.state(), &url)
            .await
            .with_context(|| format!("failed to infer a schema for {source}"))?;

        let config = ListingTableConfig::new(url)
            .with_listing_options(listing_options)
            .with_schema(resolved);
        ctx.register_table(table, Arc::new(ListingTable::try_new(config)?))?;
        Ok(())
    }
}
