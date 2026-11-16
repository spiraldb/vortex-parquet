// SPDX-License-Identifier: Apache-2.0

//! Read Apache Parquet files through the Vortex reader.
//!
//! # Why this can exist
//!
//! Vortex's reader is not welded to the Vortex file format. Its scan entry point,
//! [`ScanBuilder::new`](vortex::layout::scan::scan_builder::ScanBuilder::new), takes a
//! [`LayoutReader`](vortex::layout::LayoutReader) — not a file, not a footer, not a path.
//! `VortexFile::scan()` is sugar over exactly that call. Everything the reader is good at — expression
//! pushdown, split scheduling, mask refinement, coalesced I/O, projection, limits — is expressed
//! against that one trait.
//!
//! So reading Parquet with the Vortex reader means neither converting the file nor forking the
//! reader. It means implementing `LayoutReader` over the Parquet bytes.
//!
//! # How it works
//!
//! The in-memory layout preserves both Parquet's physical hierarchy and Vortex's logical
//! composition:
//!
//! ```text
//! ParquetFile
//!   └─ ChunkedLayout                 row groups
//!      └─ ParquetRowGroup
//!         └─ StructLayout            top-level logical fields
//!            └─ ParquetField         Dremel reconstruction boundary
//!               └─ ParquetColumnChunk
//!                  └─ ParquetIndexedPages or ParquetHeaderPages
//! ```
//!
//! Nothing is rewritten and no Vortex footer is serialized. Generic `ChunkedLayout` and
//! `StructLayout` nodes retain Vortex's split and expression machinery. Open reads only the footer;
//! page indexes and chained page headers are discovered through distinct lazy page-set layouts.
//!
//! # Status
//!
//! A flat data page is exposed as a lazy `vortex.parquet.page` array. It retains the compressed wire
//! payload, header-derived physical encoding, dictionary page, and column descriptor. Outer
//! decompression and parquet-rs value decoding occur only when a consuming kernel forces that page.
//! Nested fields currently fall back to arrow-rs at the `ParquetField` boundary, where repetition
//! and definition levels are reconstructed.
//!
//! # Usage
//!
//! ```ignore
//! let session = VortexSession::default().with_handle(handle);
//! let reader = open_path("data/events.parquet", &session).await?;
//! let stream = ScanBuilder::new(session, reader)
//!     .with_filter(gt(get_item("ts", root()), lit(cutoff)))
//!     .into_array_stream()?;
//! ```

#![deny(missing_docs)]

// Parquet's PLAIN fixed-width values are little-endian on the wire, and the page decoders
// reinterpret those bytes directly as a primitive buffer — which is correct only where the host is
// little-endian too. On a big-endian target that produces byte-swapped values *silently*: no error,
// no wrong length, just wrong numbers. Refused at compile time rather than documented as a caveat,
// because a caveat cannot stop a build.
#[cfg(target_endian = "big")]
compile_error!(
    "vortex-parquet decodes Parquet's little-endian values by reinterpreting them, which is wrong on \
     a big-endian target. The page and native decoders would need byte-swapping before this can be \
     built here."
);

pub mod arrow_bridge;
pub mod cache;
pub mod chunk_reader;
mod data_source;
pub mod file;
pub mod hierarchy;
pub mod layout;
pub mod metrics;
pub mod options;
pub mod pages;
mod reader;
pub mod segments;
mod statistics;

use std::sync::Arc;

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::metadata::FooterTail;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::metadata::ParquetOffsetIndex;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::filesystem::FileSystem;
use vortex::layout::LayoutReaderRef;
use vortex::layout::LayoutRef;
use vortex::layout::layout_children;
use vortex::layout::layouts::chunked::ChunkedLayout;
use vortex::layout::layouts::struct_::StructLayout;
use vortex::scan::DataSourceRef;
use vortex::session::VortexSession;

use crate::cache::CachePolicy;
use crate::cache::DecodeCache;
use crate::cache::FileId;
use crate::data_source::ParquetDataSource;
use crate::file::FileHandle;
use crate::file::file_handle;
use crate::file::local_filesystem;
use crate::hierarchy::ColumnChunkData;
use crate::hierarchy::PageSetData;
use crate::hierarchy::column_chunk_layout;
use crate::hierarchy::file_layout;
use crate::hierarchy::header_pages_layout;
use crate::hierarchy::indexed_pages_layout;
use crate::hierarchy::row_group_layout;
use crate::layout::BloomFilterSegment;
use crate::layout::ColumnIndexSegment;
use crate::layout::FileContext;
use crate::layout::OffsetIndexSegment;
use crate::layout::PageExecutionData;
use crate::layout::ParquetFieldData;
use crate::layout::field_layout;
use crate::metrics::ScanMetrics;
use crate::options::ReaderOptions;
use crate::reader::SharedArrayFuture;
use crate::segments::SegmentTable;

/// The bytes at both ends of every Parquet file.
const MAGIC: &[u8; 4] = b"PAR1";

/// The fixed-size tail holding the footer length and the trailing magic.
const FOOTER_TAIL_LEN: u64 = 8;

/// The Parquet reader.
///
/// Holds decode-sharing state, so one `ParquetFormat` is a small data provider. The default shares
/// only in-flight work and retains no completed arrays; retained cache policies are explicit.
#[derive(Clone, Debug)]
pub struct ParquetFormat {
    decode_cache: Arc<DecodeCache<SharedArrayFuture>>,
    metrics: Arc<ScanMetrics>,
    options: ReaderOptions,
}

/// An opened Parquet file and the precise layout that produced its reader.
///
/// Most callers only need [`ParquetFormat::open`]. This form exists for planners, diagnostics, and
/// tests that need to inspect physical column chunks and pages before constructing a scan.
#[derive(Clone)]
pub struct OpenedParquetFile {
    /// The root `vortex.parquet.file` layout.
    pub layout: LayoutRef,
    /// A reader over `layout` and the file's registered byte segments.
    pub reader: LayoutReaderRef,
}

impl Default for ParquetFormat {
    fn default() -> Self {
        Self {
            decode_cache: Arc::new(DecodeCache::new(CachePolicy::default())),
            metrics: Arc::default(),
            options: ReaderOptions::from_env(),
        }
    }
}

impl ParquetFormat {
    /// The reader with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much decoded data is kept, discarding anything already cached. See [`CachePolicy`].
    pub fn with_cache_policy(mut self, cache_policy: CachePolicy) -> Self {
        self.decode_cache = Arc::new(DecodeCache::new(cache_policy));
        self
    }

    /// Sets performance policy explicitly, replacing environment-derived defaults.
    pub fn with_options(mut self, options: ReaderOptions) -> Self {
        self.options = options;
        self
    }

    /// The performance policy captured by this format.
    pub fn options(&self) -> &ReaderOptions {
        &self.options
    }

    /// Which decode path this reader's chunks have taken, cumulative over its lifetime.
    ///
    /// This is how a test or a benchmark arm asserts that the page path actually ran — see
    /// [`metrics`] for why that needs asserting rather than assuming.
    pub fn metrics(&self) -> &Arc<ScanMetrics> {
        &self.metrics
    }

    /// The decode cache this reader holds, shared by every file it opens.
    pub fn decode_cache(&self) -> &Arc<DecodeCache<SharedArrayFuture>> {
        &self.decode_cache
    }

    /// Drops every decoded chunk currently held.
    ///
    /// A benchmark needs this so that a "cold" measurement is actually cold, and so that a warm number
    /// can be labelled as warm rather than quietly reported as a cold one.
    pub fn flush_caches(&self) {
        self.decode_cache.clear();
    }

    /// Opens `file`, returning a reader Vortex can scan.
    pub async fn open(&self, file: FileHandle) -> VortexResult<LayoutReaderRef> {
        Ok(open_inner(
            file,
            Arc::clone(&self.decode_cache),
            Arc::clone(&self.metrics),
            self.options,
        )
        .await?
        .reader)
    }

    /// Opens `file` while retaining its inspectable physical layout tree.
    pub async fn open_file(&self, file: FileHandle) -> VortexResult<OpenedParquetFile> {
        open_inner(
            file,
            Arc::clone(&self.decode_cache),
            Arc::clone(&self.metrics),
            self.options,
        )
        .await
    }

    /// Opens `file` as a Vortex [`DataSource`](vortex::scan::DataSource).
    ///
    /// This is the bridge that makes the reader usable by anything speaking the Vortex Scan API,
    /// including `vortex-datafusion`. Footer-derived statistics are available at bind; layout
    /// readers are built when a scan supplies its projection and filter.
    pub async fn open_data_source(&self, file: FileHandle) -> VortexResult<DataSourceRef> {
        let metadata = read_footer(&file).await?;
        self.open_data_source_with_metadata(file, metadata)
    }

    /// Opens a fresh data source using a previously parsed footer.
    ///
    /// The caller must ensure `metadata` belongs to the same file version as `file`. This avoids
    /// footer I/O and parsing while creating fresh per-file state. Layouts and readers are built
    /// when a scan supplies its referenced columns.
    pub fn open_data_source_with_metadata(
        &self,
        file: FileHandle,
        metadata: Arc<ParquetMetaData>,
    ) -> VortexResult<DataSourceRef> {
        let prepared = PreparedParquetMetadata::new(&file, metadata)?;
        self.open_data_source_with_prepared_metadata(file, &prepared)
    }

    /// Opens a fresh source with an already validated footer and converted schema.
    ///
    /// The caller must ensure `metadata` belongs to the same file version and was prepared with
    /// compatible Arrow type-conversion settings. File length is checked again. This shares only
    /// immutable metadata; file handles, decode identities, and scan state are fresh.
    pub fn open_data_source_with_prepared_metadata(
        &self,
        file: FileHandle,
        metadata: &PreparedParquetMetadata,
    ) -> VortexResult<DataSourceRef> {
        let prepared = PreparedFile::from_metadata(
            file,
            metadata,
            Arc::clone(&self.decode_cache),
            Arc::clone(&self.metrics),
            self.options,
        )?;
        Ok(Arc::new(ParquetDataSource::new(prepared)?))
    }

    /// Opens `path` on `fs`.
    pub async fn open_path(
        &self,
        path: &str,
        fs: &dyn FileSystem,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        self.open(file_handle(path, fs, session).await?).await
    }
}

/// Opens a Parquet file on the local filesystem, returning a reader Vortex can scan.
///
/// A convenience for tests, benchmarks, and CLIs: it builds a local filesystem and a fresh decode
/// state per call. Anything running more than one query should hold a [`ParquetFormat`] so in-flight
/// work and metrics are shared.
pub async fn open_path(path: &str, session: &VortexSession) -> VortexResult<LayoutReaderRef> {
    let fs = local_filesystem(session);
    ParquetFormat::new()
        .open_path(path, fs.as_ref(), session)
        .await
}

/// Opens a Parquet file, returning a reader Vortex can scan.
///
/// Uses fresh decode-sharing state; go through [`ParquetFormat`] to share one across files.
pub async fn open(file: FileHandle) -> VortexResult<LayoutReaderRef> {
    ParquetFormat::new().open(file).await
}

/// Parses `file`'s footer and returns it, without building a reader.
///
/// Exposed for callers that need to describe the *shape* of a file rather than read it — how many row
/// groups, how many rows, which encodings. A benchmark should report that shape, because row-group
/// count is the setting most likely to silently invalidate a comparison: too few of them and what is
/// being measured is the engines' parallelism rather than their readers.
pub async fn file_metadata(file: &FileHandle) -> VortexResult<Arc<ParquetMetaData>> {
    read_footer(file).await
}

/// Builds the layout tree over a Parquet file's bytes.
async fn open_inner(
    file: FileHandle,
    decode_cache: Arc<DecodeCache<SharedArrayFuture>>,
    metrics: Arc<ScanMetrics>,
    options: ReaderOptions,
) -> VortexResult<OpenedParquetFile> {
    let metadata = read_footer(&file).await?;
    open_inner_with_metadata(file, metadata, decode_cache, metrics, options)
}

fn open_inner_with_metadata(
    file: FileHandle,
    metadata: Arc<ParquetMetaData>,
    decode_cache: Arc<DecodeCache<SharedArrayFuture>>,
    metrics: Arc<ScanMetrics>,
    options: ReaderOptions,
) -> VortexResult<OpenedParquetFile> {
    let prepared = PreparedFile::new(file, metadata, decode_cache, metrics, options)?;
    open_prepared(&prepared, None)
}

/// A validated footer and converted schema, reusable across binds of the same file version.
///
/// This owns no file handles, readers, runtime tasks, or decoded data. Callers caching it must
/// validate file identity and change stamps before reuse; matching length alone is insufficient.
pub struct PreparedParquetMetadata {
    metadata: Arc<ParquetMetaData>,
    arrow_metadata: ArrowReaderMetadata,
    dtype: DType,
    root_leaves: Arc<[Vec<usize>]>,
    file_len: u64,
}

impl PreparedParquetMetadata {
    /// Validates all footer ranges against `file` and converts its schema with the file's session.
    /// The caller must supply a footer belonging to this file version.
    pub fn new(file: &FileHandle, metadata: Arc<ParquetMetaData>) -> VortexResult<Self> {
        let arrow_metadata = arrow_metadata_with_view_types(&metadata)?;

        // The Arrow schema arrow-rs derives is the authority on the logical types, because it applies the
        // logical-type annotations and any arrow schema stored in the Parquet key-value metadata. Deriving
        // types from the Parquet physical schema directly would disagree with what the decoder produces.
        let dtype = file
            .session()
            .arrow()
            .from_arrow_schema(&arrow_bridge::import_schema(arrow_metadata.schema())?)?;
        let DType::Struct(fields, _) = dtype.clone() else {
            vortex_bail!("a Parquet schema must map to a struct dtype, got {dtype}")
        };
        let nfields = fields.nfields();
        // Schema membership is identical in every row group. Discover it once instead of scanning
        // every leaf twice per field per row group, which is quadratic for wide, flat schemas.
        let schema = metadata.file_metadata().schema_descr();
        let mut root_leaves = vec![Vec::new(); nfields];
        for leaf in 0..schema.num_columns() {
            let root = schema.get_column_root_idx(leaf);
            root_leaves
                .get_mut(root)
                .ok_or_else(|| vortex_err!("Parquet leaf {leaf} has unknown root column {root}"))?
                .push(leaf);
        }

        // Validate every footer range at bind, including columns a later projection omits. Layout
        // allocation is deferred, but malformed or external-file metadata remains an open error.
        for (rg_idx, row_group) in metadata.row_groups().iter().enumerate() {
            u64::try_from(row_group.num_rows())
                .map_err(|_| vortex_err!("row group {rg_idx} reports a negative row count"))?;
            for (root, leaves) in root_leaves.iter().enumerate() {
                root_column_byte_range(&metadata, rg_idx, root, leaves, file.size())?;
                for &leaf in leaves {
                    let column = row_group.column(leaf);
                    for (offset, length, kind) in [
                        (
                            column.offset_index_offset(),
                            column.offset_index_length(),
                            "offset",
                        ),
                        (
                            column.column_index_offset(),
                            column.column_index_length(),
                            "column",
                        ),
                        (
                            column.bloom_filter_offset(),
                            column.bloom_filter_length(),
                            "Bloom-filter",
                        ),
                    ] {
                        checked_index_range(offset, length, file.size(), rg_idx, leaf, kind)?;
                    }
                }
            }
        }
        Ok(Self {
            metadata,
            arrow_metadata,
            dtype,
            root_leaves: root_leaves.into(),
            file_len: file.size(),
        })
    }

    /// The parsed footer used for this preparation.
    pub fn metadata(&self) -> &Arc<ParquetMetaData> {
        &self.metadata
    }

    /// An approximate memory charge for bounded metadata caches, excluding shared runtime state.
    /// Schema fields are charged twice to account for both Arrow and Vortex representations.
    pub fn memory_size(&self) -> usize {
        size_of::<Self>()
            + self.metadata.memory_size()
            + 2 * self.arrow_metadata.schema().fields().size()
            + self
                .arrow_metadata
                .schema()
                .metadata()
                .iter()
                .map(|(key, value)| {
                    size_of::<(String, String)>() + key.capacity() + value.capacity()
                })
                .sum::<usize>()
            + self.root_leaves.len() * size_of::<Vec<usize>>()
            + self
                .root_leaves
                .iter()
                .map(|leaves| leaves.capacity() * size_of::<usize>())
                .sum::<usize>()
    }
}

/// Fresh file identity and reader context backed by reusable immutable metadata.
pub(crate) struct PreparedFile {
    ctx: Arc<FileContext>,
    dtype: DType,
    root_leaves: Arc<[Vec<usize>]>,
}

impl PreparedFile {
    fn new(
        file: FileHandle,
        metadata: Arc<ParquetMetaData>,
        decode_cache: Arc<DecodeCache<SharedArrayFuture>>,
        metrics: Arc<ScanMetrics>,
        options: ReaderOptions,
    ) -> VortexResult<Self> {
        let metadata = PreparedParquetMetadata::new(&file, metadata)?;
        Self::from_metadata(file, &metadata, decode_cache, metrics, options)
    }

    fn from_metadata(
        file: FileHandle,
        metadata: &PreparedParquetMetadata,
        decode_cache: Arc<DecodeCache<SharedArrayFuture>>,
        metrics: Arc<ScanMetrics>,
        options: ReaderOptions,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            file.size() == metadata.file_len,
            "prepared Parquet metadata has a different file length"
        );
        let ctx = Arc::new(FileContext {
            file,
            decode_cache,
            file_id: FileId::next(),
            metadata: Arc::clone(&metadata.metadata),
            arrow_metadata: metadata.arrow_metadata.clone(),
            file_len: metadata.file_len,
            metrics,
            options,
        });
        Ok(Self {
            ctx,
            dtype: metadata.dtype.clone(),
            root_leaves: Arc::clone(&metadata.root_leaves),
        })
    }
}

fn open_prepared(
    prepared: &PreparedFile,
    selected: Option<&[bool]>,
) -> VortexResult<OpenedParquetFile> {
    let ctx = &prepared.ctx;
    let file = &ctx.file;
    let metadata = &ctx.metadata;
    pages::register_kernels(file.session());
    let DType::Struct(fields, nullability) = &prepared.dtype else {
        vortex_bail!("Parquet root must be a struct");
    };
    if let Some(selected) = selected {
        vortex_ensure!(
            selected.len() == fields.nfields(),
            "invalid column selection length"
        );
    }
    let dtype = if let Some(selected) = selected {
        DType::Struct(
            vortex::dtype::StructFields::from_iter(
                fields
                    .names()
                    .iter()
                    .cloned()
                    .zip(fields.fields())
                    .enumerate()
                    .filter_map(|(index, field)| selected[index].then_some(field)),
            ),
            *nullability,
        )
    } else {
        prepared.dtype.clone()
    };
    let selected_count = dtype
        .as_struct_fields_opt()
        .map_or(0, |fields| fields.nfields());
    let schema = metadata.file_metadata().schema_descr();

    // Footer-known physical objects and lazy discovery inputs are represented explicitly. Page
    // children are not enumerated here: open is permitted to read only the footer.
    let row_group_count = metadata.row_groups().len();
    // Bounded as well as saturating. Saturating stops the multiply from wrapping; the bound stops a
    // footer claiming an absurd row-group count from turning a capacity *hint* into an arbitrarily
    // large allocation. Both numbers are footer claims, and the loop below grows these vectors
    // correctly whatever the hint was.
    let capacity_hint = row_group_count.saturating_mul(selected_count).min(1 << 20);
    let mut segments = SegmentTable::with_capacity(capacity_hint);
    let mut row_group_layouts: Vec<LayoutRef> = Vec::with_capacity(row_group_count.min(1 << 20));
    let mut total_rows: u64 = 0;

    for rg_idx in 0..row_group_count {
        let rg_rows = u64::try_from(metadata.row_groups()[rg_idx].num_rows())
            .map_err(|_| vortex_err!("row group {rg_idx} reports a negative row count"))?;

        let mut column_layouts: Vec<LayoutRef> = Vec::with_capacity(selected_count);
        for (root_column, leaves) in prepared.root_leaves.iter().enumerate() {
            if selected.is_some_and(|selected| !selected[root_column]) {
                continue;
            }
            let field_dtype = fields.field_by_index(root_column).ok_or_else(|| {
                vortex_err!("no top-level column {root_column} in the Parquet schema")
            })?;
            let (byte_offset, byte_length) =
                root_column_byte_range(metadata, rg_idx, root_column, leaves, file.size())?;
            let field_segment = segments.push(byte_offset..byte_offset + byte_length)?;

            let leaf_count = leaves.len();
            let mut physical_layouts: Vec<LayoutRef> = Vec::with_capacity(leaves.len());
            let mut physical_dtypes: Vec<DType> = Vec::with_capacity(leaves.len());
            let mut offset_indexes = Vec::new();
            let mut column_indexes = Vec::new();
            let mut bloom_filters = Vec::new();

            for &leaf in leaves {
                let column = metadata.row_group(rg_idx).column(leaf);
                let descriptor = schema.column(leaf);
                let physical_dtype =
                    physical_dtype(descriptor.physical_type(), descriptor.max_def_level() > 0);
                // A flat field's column chunk produces the declared logical values directly; its
                // Parquet physical type remains explicit in `ColumnChunkData`. Nested leaves do not
                // independently have the field's logical dtype and are reconstructed at the field
                // boundary, so their layouts retain wire dtypes.
                let chunk_dtype = if leaf_count == 1 {
                    field_dtype.clone()
                } else {
                    physical_dtype
                };
                let (chunk_offset, chunk_length) = checked_byte_range(column, rg_idx, leaf)?;
                let chunk_range = chunk_offset..chunk_offset + chunk_length;
                let chunk_segment = segments.push(chunk_range.clone())?;
                let offset_index = checked_index_range(
                    column.offset_index_offset(),
                    column.offset_index_length(),
                    file.size(),
                    rg_idx,
                    leaf,
                    "offset",
                )?;
                let column_index = checked_index_range(
                    column.column_index_offset(),
                    column.column_index_length(),
                    file.size(),
                    rg_idx,
                    leaf,
                    "column",
                )?;
                let bloom_filter = checked_index_range(
                    column.bloom_filter_offset(),
                    column.bloom_filter_length(),
                    file.size(),
                    rg_idx,
                    leaf,
                    "Bloom-filter",
                )?;
                let mut page_segment_ids = vec![chunk_segment];
                if let Some(range) = &offset_index {
                    let id = segments.push(range.clone())?;
                    page_segment_ids.push(id);
                    offset_indexes.push(OffsetIndexSegment {
                        leaf,
                        id,
                        range: range.clone(),
                    });
                }
                if let Some(range) = &column_index {
                    let id = segments.push(range.clone())?;
                    page_segment_ids.push(id);
                    column_indexes.push(ColumnIndexSegment {
                        leaf,
                        id,
                        range: range.clone(),
                    });
                }
                if let Some(range) = &bloom_filter {
                    let id = segments.push(range.clone())?;
                    page_segment_ids.push(id);
                    bloom_filters.push(BloomFilterSegment {
                        leaf,
                        id,
                        range: range.clone(),
                    });
                }
                let page_data = PageSetData {
                    chunk_range: chunk_range.clone(),
                    offset_index: offset_index.clone(),
                    column_index: column_index.clone(),
                    bloom_filter: bloom_filter.clone(),
                    execution: PageExecutionData {
                        ctx: Arc::clone(ctx),
                        row_group: rg_idx,
                        root_column,
                        byte_offset: chunk_offset,
                        byte_length: chunk_length,
                        offset_indexes: offset_indexes
                            .iter()
                            .filter(|index| index.leaf == leaf)
                            .cloned()
                            .collect(),
                        column_indexes: column_indexes
                            .iter()
                            .filter(|index| index.leaf == leaf)
                            .cloned()
                            .collect(),
                        bloom_filters: bloom_filters
                            .iter()
                            .filter(|filter| filter.leaf == leaf)
                            .cloned()
                            .collect(),
                    },
                };
                let pages = if offset_index.is_some() {
                    Arc::new(indexed_pages_layout(
                        chunk_dtype.clone(),
                        rg_rows,
                        page_segment_ids,
                        page_data,
                    )) as LayoutRef
                } else {
                    Arc::new(header_pages_layout(
                        chunk_dtype.clone(),
                        rg_rows,
                        page_segment_ids,
                        page_data,
                    )) as LayoutRef
                };

                physical_layouts.push(Arc::new(column_chunk_layout(
                    chunk_dtype.clone(),
                    rg_rows,
                    chunk_segment,
                    ColumnChunkData {
                        leaf,
                        physical_type: descriptor.physical_type(),
                        compression: column.compression(),
                        byte_range: chunk_range,
                        offset_index,
                        column_index,
                        bloom_filter,
                    },
                    pages,
                )) as LayoutRef);
                physical_dtypes.push(chunk_dtype);
            }

            column_layouts.push(Arc::new(field_layout(
                field_dtype,
                rg_rows,
                field_segment,
                ParquetFieldData {
                    ctx: Arc::clone(ctx),
                    row_group: rg_idx,
                    root_column,
                    byte_offset,
                    byte_length,
                    offset_indexes,
                    column_indexes,
                    bloom_filters,
                    physical_dtypes,
                },
                physical_layouts,
            )) as LayoutRef);
        }

        let fields_layout = Arc::new(StructLayout::new(rg_rows, dtype.clone(), column_layouts));
        row_group_layouts.push(Arc::new(row_group_layout(
            dtype.clone(),
            rg_rows,
            rg_idx,
            fields_layout,
        )) as LayoutRef);
        total_rows = total_rows.checked_add(rg_rows).ok_or_else(|| {
            vortex_err!("the row counts of this file's row groups sum to more than u64::MAX")
        })?;
    }

    let source = file.segment_source(segments)?;

    // A chunked layout even for a single row group, so the reader's shape does not depend on the
    // file's chunking. A one-child chunked layout costs a pointer hop and removes a special case.
    let row_groups = Arc::new(ChunkedLayout::new(
        total_rows,
        dtype.clone(),
        layout_children(row_group_layouts),
    )) as LayoutRef;
    let root = Arc::new(file_layout(dtype, total_rows, row_groups)) as LayoutRef;
    let reader = root.new_reader(
        "parquet".into(),
        source,
        file.session(),
        &Default::default(),
    )?;
    Ok(OpenedParquetFile {
        layout: root,
        reader,
    })
}

/// The dtype on the Parquet wire, before logical annotations and Dremel reconstruction.
fn physical_dtype(physical: parquet::basic::Type, nullable: bool) -> DType {
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    match physical {
        parquet::basic::Type::BOOLEAN => DType::Bool(nullability),
        parquet::basic::Type::INT32 => DType::Primitive(PType::I32, nullability),
        parquet::basic::Type::INT64 => DType::Primitive(PType::I64, nullability),
        parquet::basic::Type::FLOAT => DType::Primitive(PType::F32, nullability),
        parquet::basic::Type::DOUBLE => DType::Primitive(PType::F64, nullability),
        parquet::basic::Type::INT96
        | parquet::basic::Type::BYTE_ARRAY
        | parquet::basic::Type::FIXED_LEN_BYTE_ARRAY => DType::Binary(nullability),
    }
}

/// Derives arrow-rs's view of the footer, asking for Arrow *view* types for strings and binary.
///
/// This is the single highest-leverage line in the reader. Vortex's canonical string array is
/// `VarBinViewArray`, which is view-shaped: converting an Arrow `StringViewArray` into it reuses the
/// views buffer directly, while converting a plain `StringArray` has to rebuild every view from
/// offsets — an allocation and a full pass over the column.
///
/// arrow-rs will decode straight into `StringViewArray` if the target schema says `Utf8View`, so
/// asking for it moves the conversion from O(rows) work to a pointer handoff. On ClickBench the wide
/// string columns (`URL`, `Title`, `Referer`, `SearchPhrase`) dominate this reader's cost, and this
/// is what that cost mostly was.
///
/// Falls back to the default schema if the coercion is rejected: a slower read beats a failed one.
fn arrow_metadata_with_view_types(
    metadata: &Arc<ParquetMetaData>,
) -> VortexResult<ArrowReaderMetadata> {
    let plain = ArrowReaderMetadata::try_new(Arc::clone(metadata), ArrowReaderOptions::default())
        .map_err(|e| {
        vortex_err!("failed to derive an Arrow schema from the Parquet footer: {e}")
    })?;

    let viewed = Arc::new(view_typed_schema(plain.schema().as_ref()));
    match ArrowReaderMetadata::try_new(
        Arc::clone(metadata),
        ArrowReaderOptions::default().with_schema(viewed),
    ) {
        Ok(with_views) => Ok(with_views),
        Err(_) => Ok(plain),
    }
}

/// Rewrites a schema's string and binary types to their view equivalents, recursively.
///
/// The reader and benchmark control arms use this to agree on their Arrow output schema.
/// Parquet dictionary encoding does not change the schema produced here.
pub fn view_typed_schema(schema: &arrow_schema::Schema) -> arrow_schema::Schema {
    let fields: arrow_schema::Fields = schema
        .fields()
        .iter()
        .map(|field| Arc::new(view_typed_field(field)))
        .collect();
    arrow_schema::Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// Rewrites one field's type, descending into nested types.
fn view_typed_field(field: &arrow_schema::Field) -> arrow_schema::Field {
    use arrow_schema::DataType;

    let data_type = match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
        DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Arc::new(view_typed_field(f)))
                .collect(),
        ),
        DataType::List(inner) => DataType::List(Arc::new(view_typed_field(inner))),
        DataType::LargeList(inner) => DataType::LargeList(Arc::new(view_typed_field(inner))),
        other => other.clone(),
    };

    arrow_schema::Field::new(field.name(), data_type, field.is_nullable())
        .with_metadata(field.metadata().clone())
}

/// Reads and parses the Parquet footer.
///
/// Reads exactly the fixed tail and serialized footer metadata. Data pages and page indexes are
/// outside this range and are never touched by open.
async fn read_footer(file: &FileHandle) -> VortexResult<Arc<ParquetMetaData>> {
    if file.size() < FOOTER_TAIL_LEN + MAGIC.len() as u64 {
        vortex_bail!(
            "not a Parquet file: {} bytes is too short to hold a footer",
            file.size()
        );
    }

    let tail = file
        .read_range(file.size() - FOOTER_TAIL_LEN..file.size())
        .await?;
    let footer_tail: [u8; 8] = tail
        .as_ref()
        .try_into()
        .map_err(|_| vortex_err!("internal error: the footer tail was not 8 bytes"))?;
    if &footer_tail[4..] != MAGIC {
        vortex_bail!("not a Parquet file: the trailing magic bytes are not PAR1");
    }

    let metadata_len = u64::try_from(
        FooterTail::try_new(&footer_tail)
            .map_err(|e| vortex_err!("failed to decode the Parquet footer length: {e}"))?
            .metadata_length(),
    )?;

    let footer_end = file.size() - FOOTER_TAIL_LEN;
    let footer_start = footer_end
        .checked_sub(metadata_len)
        .ok_or_else(|| vortex_err!("the Parquet footer claims to be longer than the file"))?;

    let footer_bytes = file.read_range(footer_start..footer_end).await?;

    Ok(Arc::new(
        ParquetMetaDataReader::decode_metadata(footer_bytes.as_ref())
            .map_err(|e| vortex_err!("failed to decode the Parquet footer: {e}"))?,
    ))
}

/// Whether every offset index describes pages that could exist in `metadata`.
///
/// The indexes are borrowed separately so validating lazily loaded indexes does not require
/// cloning the file's footer just to attach them.
///
/// An offset index is *data*: it decides which bytes are read and which rows they are believed to
/// hold, and a malformed one does not fail loudly — it yields the expected number of rows over the
/// wrong values. Anything inconsistent with the column chunk it describes disqualifies the whole
/// index, because a reader that trusts it for one column and not another is harder to reason about
/// than one that does not trust it at all.
pub(crate) fn offset_index_is_sound(
    metadata: &ParquetMetaData,
    offset_index: &ParquetOffsetIndex,
) -> bool {
    if offset_index.len() != metadata.num_row_groups() {
        return false;
    }

    for (rg_idx, columns) in offset_index.iter().enumerate() {
        let rg = metadata.row_group(rg_idx);
        if columns.len() != rg.num_columns() {
            return false;
        }
        let Ok(rows) = u64::try_from(rg.num_rows()) else {
            return false;
        };

        for (leaf, index) in columns.iter().enumerate() {
            let locations = &index.page_locations;
            if locations.is_empty() {
                // A column with no page locations is not malformed; it simply cannot be page-addressed.
                continue;
            }
            let column = rg.column(leaf);
            // `byte_range` panics on negative offsets; this validator's whole job is to survive a
            // hostile footer, so it must not be the thing that crashes on one.
            let Ok((chunk_start, chunk_len)) = checked_byte_range(column, rg_idx, leaf) else {
                return false;
            };
            let chunk_end = chunk_start.saturating_add(chunk_len);

            let mut previous: Option<(u64, u64)> = None;
            for location in locations {
                let (Ok(offset), Ok(first_row)) = (
                    u64::try_from(location.offset),
                    u64::try_from(location.first_row_index),
                ) else {
                    return false;
                };
                if location.compressed_page_size <= 0 {
                    return false;
                }
                let Ok(page_len) = u64::try_from(location.compressed_page_size) else {
                    return false;
                };
                let Some(page_end) = offset.checked_add(page_len) else {
                    return false;
                };
                // Inside the chunk it claims to belong to, and leaving room for at least one byte.
                if offset < chunk_start
                    || offset >= chunk_end
                    || page_end > chunk_end
                    || first_row >= rows
                {
                    return false;
                }
                // Strictly increasing offsets and rows, AND no overlap: `previous` carries the last
                // page's END, not its start. Comparing starts alone accepts a page that begins after
                // its predecessor and still runs into it, which makes the row-to-page mapping
                // ambiguous — two pages claiming the same bytes, so a row lookup has two answers.
                if let Some((last_end, last_row)) = previous
                    && (offset < last_end || first_row <= last_row)
                {
                    return false;
                }
                previous = Some((page_end, first_row));
            }

            // The first data page must start the row group. A nonzero first row means the index does
            // not describe this chunk from its beginning, and every row-to-page mapping derived from
            // it would be shifted.
            if u64::try_from(locations[0].first_row_index).is_ok_and(|first| first != 0) {
                return false;
            }
        }
    }
    true
}

/// The byte range spanned by every column chunk belonging to one top-level column.
///
/// A nested top-level column occupies several Parquet leaf columns, which Parquet writes contiguously
/// within a row group in schema order. Taking the span from the first leaf's start to the last leaf's
/// end therefore yields one range rather than several — and it is the same range a coalesced read
/// would have fetched anyway.
fn root_column_byte_range(
    metadata: &ParquetMetaData,
    row_group: usize,
    root_column: usize,
    leaves: &[usize],
    file_len: u64,
) -> VortexResult<(u64, u64)> {
    let rg = metadata
        .row_groups()
        .get(row_group)
        .ok_or_else(|| vortex_err!("no row group {row_group} in this file"))?;

    let mut start = u64::MAX;
    let mut end = 0u64;
    for &leaf_idx in leaves {
        let column = rg.column(leaf_idx);

        // `file_path` means this chunk's data lives in a *different* file. Reading the range out of
        // this one would silently return whatever bytes happen to sit there, so it is refused rather
        // than ignored. Rare — Parquet's spec allows it and few writers emit it — but a wrong answer
        // is not an acceptable response to a file we merely do not support.
        if let Some(path) = column.file_path() {
            vortex_bail!(
                "row group {row_group} column {leaf_idx} stores its data in an external file \
                 ({path:?}), which is not supported"
            );
        }

        let (chunk_start, chunk_len) = checked_byte_range(column, row_group, leaf_idx)?;
        // Unreachable while both come from non-negative `i64`s, whose sum is at most `2^64 - 2` — kept
        // because it costs nothing and the day that provenance changes it stops being unreachable.
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| vortex_err!(
                "row group {row_group} column {leaf_idx} claims a byte range {chunk_start}..+{chunk_len} that overflows"
            ))?;
        start = start.min(chunk_start);
        end = end.max(chunk_end);
    }

    if start == u64::MAX {
        vortex_bail!(
            "top-level column {root_column} has no Parquet leaf columns in row group {row_group}"
        );
    }

    // Checked against the file rather than trusted. A footer is data, and a range past the end of the
    // file would otherwise become a read error much later, in the I/O driver, with no indication that
    // the metadata was the problem.
    if end > file_len {
        vortex_bail!(
            "row group {row_group} column {root_column} spans {start}..{end}, past the end of a \
             {file_len}-byte file"
        );
    }

    Ok((start, end - start))
}

/// [`ColumnChunkMetaData::byte_range`] without the panic.
///
/// Upstream's `byte_range` asserts that the offset and compressed size are non-negative, so a footer
/// with a negative value there aborts the process instead of failing the open. The values are `i64` on
/// the wire and nothing has validated them at this point, so they are checked here first.
fn checked_byte_range(
    column: &ColumnChunkMetaData,
    row_group: usize,
    leaf_idx: usize,
) -> VortexResult<(u64, u64)> {
    let offset = column
        .dictionary_page_offset()
        .unwrap_or_else(|| column.data_page_offset());
    let length = column.compressed_size();
    if offset < 0 || length < 0 {
        vortex_bail!(
            "row group {row_group} column {leaf_idx} has a negative byte range \
             (offset {offset}, compressed size {length})"
        );
    }
    Ok((offset as u64, length as u64))
}

/// A footer-referenced page-index range, validated without reading it.
fn checked_index_range(
    offset: Option<i64>,
    length: Option<i32>,
    file_len: u64,
    row_group: usize,
    leaf: usize,
    kind: &str,
) -> VortexResult<Option<std::ops::Range<u64>>> {
    let (Some(offset), Some(length)) = (offset, length) else {
        return Ok(None);
    };
    if offset < 0 || length < 0 {
        vortex_bail!("row group {row_group} column {leaf} has a negative {kind}-index range");
    }
    let start = u64::try_from(offset)?;
    let end = start
        .checked_add(u64::try_from(length)?)
        .ok_or_else(|| vortex_err!("{kind}-index range overflows"))?;
    if end > file_len {
        vortex_bail!(
            "row group {row_group} column {leaf} {kind} index ends at {end}, past the end of a \
             {file_len}-byte file"
        );
    }
    Ok(Some(start..end))
}

#[cfg(test)]
mod tests {
    use parquet::basic::Type as PhysicalType;
    use parquet::file::metadata::ColumnChunkMetaDataBuilder;
    use parquet::file::metadata::FileMetaData;
    use parquet::file::metadata::RowGroupMetaData;
    use parquet::schema::types::SchemaDescriptor;
    use parquet::schema::types::Type as SchemaType;

    use super::*;

    /// Reading the footer retains page-index locations without loading the indexes.
    #[test]
    fn the_footer_only_references_the_page_index() {
        use arrow_array::Int64Array;
        use arrow_array::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use vortex::VortexSessionDefault;
        use vortex::io::runtime::single::block_on;
        use vortex::io::session::RuntimeSessionExt;

        let ids = Int64Array::from_iter_values(0..4000);
        let batch = RecordBatch::try_from_iter([("id", Arc::new(ids) as arrow_array::ArrayRef)])
            .expect("the batch builds");
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(500))
            .set_data_page_row_count_limit(100)
            .build();
        let mut bytes: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(properties))
            .expect("the writer builds");
        writer.write(&batch).expect("the batch writes");
        writer.close().expect("the writer closes");
        let bytes = vortex::buffer::ByteBuffer::copy_from(bytes);
        let size = bytes.len() as u64;

        block_on(|handle| async move {
            let session = VortexSession::default().with_handle(handle);
            let file = FileHandle::new(Arc::new(bytes), size, session);
            let metadata = read_footer(&file).await.expect("the footer reads");
            assert!(metadata.offset_index().is_none());
            assert!(
                metadata
                    .row_group(0)
                    .column(0)
                    .offset_index_offset()
                    .is_some(),
                "the footer should retain the lazy index location"
            );
        });
    }

    /// A one-column footer whose single chunk is described by `chunk`.
    ///
    /// Built by hand rather than written by `ArrowWriter`, because every case below is metadata a
    /// writer would never produce — that is the point. A footer is *data*, and these are the shapes a
    /// corrupt or hostile one takes.
    fn metadata_with(
        chunk: impl FnOnce(ColumnChunkMetaDataBuilder) -> ColumnChunkMetaDataBuilder,
        num_rows: i64,
    ) -> ParquetMetaData {
        let field = Arc::new(
            SchemaType::primitive_type_builder("a", PhysicalType::INT64)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()
                .expect("the fixture field is valid"),
        );
        let message = Arc::new(
            SchemaType::group_type_builder("schema")
                .with_fields(vec![field])
                .build()
                .expect("the fixture schema is valid"),
        );
        let descr = Arc::new(SchemaDescriptor::new(message));

        let column = chunk(ColumnChunkMetaData::builder(descr.column(0)))
            .build()
            .expect("the fixture chunk metadata is well-formed");
        let row_group = RowGroupMetaData::builder(Arc::clone(&descr))
            .set_num_rows(num_rows)
            .set_column_metadata(vec![column])
            .build()
            .expect("the fixture row group is well-formed");

        let file_metadata = FileMetaData::new(2, num_rows, None, None, descr, None);
        ParquetMetaData::new(file_metadata, vec![row_group])
    }

    /// A chunk claiming to live in another file must be refused, not silently read from this one.
    ///
    /// `file_path` is legal Parquet and rare in practice. Ignoring it means reading whatever bytes
    /// happen to sit at that offset in the wrong file — a wrong answer, which is not an acceptable
    /// response to a file we do not support.
    #[test]
    fn an_external_column_chunk_is_rejected() {
        let metadata = metadata_with(
            |b| {
                b.set_file_path("elsewhere.parquet".to_string())
                    .set_data_page_offset(4)
                    .set_total_compressed_size(64)
            },
            10,
        );
        let err = root_column_byte_range(&metadata, 0, 0, &[0], 1024)
            .expect_err("an external chunk must not be accepted");
        assert!(
            err.to_string().contains("external file"),
            "the error should name the cause, got: {err}"
        );
    }

    /// A negative offset must fail the open rather than abort the process.
    ///
    /// `ColumnChunkMetaData::byte_range` asserts that these are non-negative, so calling it on a footer
    /// like this panics. They are `i64` on the wire and nothing validates them before this point.
    #[test]
    fn a_negative_byte_range_is_rejected_rather_than_asserted() {
        let metadata = metadata_with(
            |b| b.set_data_page_offset(-8).set_total_compressed_size(64),
            10,
        );
        let err = root_column_byte_range(&metadata, 0, 0, &[0], 1024)
            .expect_err("a negative offset must not be accepted");
        assert!(
            err.to_string().contains("negative"),
            "the error should name the cause, got: {err}"
        );
    }

    /// Likewise a negative compressed size.
    #[test]
    fn a_negative_compressed_size_is_rejected() {
        let metadata = metadata_with(
            |b| b.set_data_page_offset(4).set_total_compressed_size(-64),
            10,
        );
        assert!(root_column_byte_range(&metadata, 0, 0, &[0], 1024).is_err());
    }

    /// A range extending past the end of the file must be caught here, where the footer can be blamed.
    ///
    /// Otherwise it becomes a read error deep in the I/O driver with nothing pointing at the metadata.
    #[test]
    fn a_range_past_the_end_of_the_file_is_rejected() {
        let metadata = metadata_with(
            |b| b.set_data_page_offset(900).set_total_compressed_size(500),
            10,
        );
        let err = root_column_byte_range(&metadata, 0, 0, &[0], 1024)
            .expect_err("a range past the end of the file must not be accepted");
        assert!(
            err.to_string().contains("past the end"),
            "the error should name the cause, got: {err}"
        );
        // The same range inside a large enough file is fine, so the check is not simply always failing.
        assert!(root_column_byte_range(&metadata, 0, 0, &[0], 4096).is_ok());
    }

    /// An absurd but *representable* range must be rejected by the file-length check.
    ///
    /// This started out asserting that offset + length overflows `u64` and is caught. It does not:
    /// both come from non-negative `i64`s, so their sum is at most `2^64 - 2` and the `checked_add`
    /// guarding it is unreachable. Which means the file-length check is the only thing standing between
    /// a hostile footer and a nonsense range, so that is what gets asserted.
    #[test]
    fn an_absurd_byte_range_is_rejected_by_the_file_length_check() {
        let metadata = metadata_with(
            |b| {
                b.set_data_page_offset(i64::MAX)
                    .set_total_compressed_size(i64::MAX)
            },
            10,
        );
        let err = root_column_byte_range(&metadata, 0, 0, &[0], 4096)
            .expect_err("a range far past the end of the file must not be accepted");
        assert!(
            err.to_string().contains("past the end"),
            "the error should name the cause, got: {err}"
        );
    }

    /// A column chunk over 4 GiB cannot be one segment, and must say so actionably.
    ///
    /// `SegmentSpec` stores a length as `u32`. This is reachable on a genuinely large row group rather
    /// than only on corrupt metadata, so the boundary is asserted rather than assumed — and the message
    /// has to say what to do, because the answer is to split the range, not to give up.
    #[test]
    fn a_column_chunk_over_four_gibibytes_is_rejected_actionably() {
        let over = u64::from(u32::MAX) + 1;
        let mut segments = SegmentTable::with_capacity(1);
        let err = segments
            .push(0..over)
            .expect_err("a segment over 4 GiB must not be accepted");
        assert!(
            err.to_string().contains("split it into multiple segments"),
            "the error should say what to do, got: {err}"
        );
        // One byte under the limit is fine, so the boundary is where it claims to be.
        assert!(segments.push(0..(over - 1)).is_ok());
    }

    /// A footer whose single chunk spans bytes 100..200 of a 10-row row group, described by `pages`.
    ///
    /// The chunk is fixed so the cases below vary only the index, which is the thing under test.
    fn metadata_with_pages(pages: Vec<(i64, i32, i64)>) -> ParquetMetaData {
        use parquet::file::page_index::offset_index::OffsetIndexMetaData;
        use parquet::file::page_index::offset_index::PageLocation;

        let metadata = metadata_with(
            |b| b.set_data_page_offset(100).set_total_compressed_size(100),
            10,
        );
        let index = OffsetIndexMetaData {
            page_locations: pages
                .into_iter()
                .map(
                    |(offset, compressed_page_size, first_row_index)| PageLocation {
                        offset,
                        compressed_page_size,
                        first_row_index,
                    },
                )
                .collect(),
            unencoded_byte_array_data_bytes: None,
        };
        metadata
            .into_builder()
            .set_offset_index(Some(vec![vec![index]]))
            .build()
    }

    /// The index a real writer emits must be accepted, or every check below is passing vacuously.
    #[test]
    fn a_well_formed_offset_index_is_accepted() {
        let metadata = metadata_with_pages(vec![(100, 40, 0), (140, 30, 4), (170, 30, 7)]);
        assert!(offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }

    /// A page running off the end of its chunk must be refused.
    ///
    /// The check this pins was missing: only a page's *start* was tested against the chunk, so a
    /// page could declare a length carrying it past the chunk end and into the next column's bytes.
    /// The last page here starts legally at 170 and claims 60 bytes, ending at 230 against a chunk
    /// ending at 200.
    #[test]
    fn a_page_extending_past_its_chunk_is_rejected() {
        let metadata = metadata_with_pages(vec![(100, 40, 0), (140, 30, 4), (170, 60, 7)]);
        assert!(!offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }

    /// A zero-length page is not a page, and admitting one would let extents stop advancing.
    #[test]
    fn a_zero_length_page_is_rejected() {
        for page_size in [0, -8] {
            let metadata = metadata_with_pages(vec![(100, page_size, 0)]);
            assert!(!offset_index_is_sound(
                &metadata,
                metadata.offset_index().unwrap()
            ));
        }
    }

    /// A page starting before its chunk must be refused.
    #[test]
    fn a_page_before_its_chunk_is_rejected() {
        let metadata = metadata_with_pages(vec![(80, 40, 0)]);
        assert!(!offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }

    /// The documented limit of this validator, asserted so it is not mistaken for a proof.
    ///
    /// Every offset and row index here is shifted off the real page framing by a constant, which is
    /// exactly the fault the check cannot see: the result is ordered, in bounds, non-overlapping and
    /// starts at row zero, so it passes — and it still points at bytes that are not page headers.
    /// Catching this needs the page headers parsed, which means reading the data region, which this
    /// deliberately does not do. See `offset_index_is_sound`.
    #[test]
    fn a_consistently_wrong_index_passes_and_that_is_the_known_limit() {
        let metadata = metadata_with_pages(vec![(100, 33, 0), (133, 33, 4), (166, 33, 7)]);
        assert!(offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }

    /// Pages that overlap must be refused.
    ///
    /// The check this pins was missing: offsets were compared start-to-start, so a page could begin
    /// after its predecessor's start and still run into its bytes. Page one spans 100..160 and page two
    /// starts at 140, inside it -- two pages claiming the same bytes, so a row lookup has two answers
    /// and neither is provably right.
    #[test]
    fn overlapping_pages_are_rejected() {
        let metadata = metadata_with_pages(vec![(100, 60, 0), (140, 30, 4)]);
        assert!(!offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }

    /// Pages that merely abut are fine: one page's end is the next page's start.
    ///
    /// The boundary case for the check above, and the reason it compares `<` rather than `<=`.
    #[test]
    fn abutting_pages_are_accepted() {
        let metadata = metadata_with_pages(vec![(100, 40, 0), (140, 30, 4), (170, 30, 7)]);
        assert!(offset_index_is_sound(
            &metadata,
            metadata.offset_index().unwrap()
        ));
    }
}
