// SPDX-License-Identifier: Apache-2.0

//! The logical-field layout: one top-level field of one row group.
//!
//! Everything above the leaf is Vortex's. A Parquet file becomes
//!
//! ```text
//! ChunkedLayout          one child per row group
//!   └─ StructLayout      one child per top-level column
//!        └─ ParquetField  this module
//! ```
//!
//! built in memory — no footer is ever serialized — so Vortex's own `ChunkedReader` and
//! `StructReader` compose these leaves. That is what buys per-field expression partitioning, split
//! registration across row groups, and cross-chunk mask plumbing without this crate implementing any
//! of it.
//!
//! # Why a *top-level* column rather than a Parquet leaf column
//!
//! Parquet stores nested data as one column chunk per primitive leaf, with repetition and definition
//! levels to reconstruct the nesting. Reconstructing that is genuinely hard and easy to get subtly
//! wrong. arrow-rs already does it correctly, and it does it for a whole top-level column at once.
//!
//! So the unit here is a top-level column: `ProjectionMask::roots` selects it, arrow-rs reassembles
//! whatever nesting it contains, and the result converts to a Vortex array in one step. A struct,
//! list, or map column is one leaf holding a nested Vortex array, not a subtree of leaves. Vortex's
//! `StructReader` can still project *into* it, because expression evaluation happens on the decoded
//! array.

use std::sync::Arc;

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::file::metadata::ParquetMetaData;
use vortex::array::EmptyMetadata;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::layout::Layout;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutDeserializeArgs;
use vortex::layout::LayoutId;
use vortex::layout::LayoutParts;
use vortex::layout::LayoutReaderContext;
use vortex::layout::LayoutReaderRef;
use vortex::layout::VTable;
use vortex::layout::layout_children;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::scan::plan::PlanRef;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;

use crate::cache::DecodeCache;
use crate::cache::FileId;
use crate::file::FileHandle;
use crate::metrics::ScanMetrics;
use crate::options::ReaderOptions;
use crate::plan::DecodedChunk;

/// State shared by every leaf of one Parquet file.
///
/// Held behind an `Arc` because each leaf needs the same metadata, and a Parquet footer for a wide
/// file is not something to clone per column per row group.
#[derive(Debug)]
pub struct FileContext {
    /// The random-access file handle, retained so a lazily decoded offset index can construct a
    /// page-specific segment source after open has completed.
    pub file: FileHandle,
    /// Decoded column chunks, shared by every leaf of this file.
    ///
    /// On the context rather than on each leaf because a budget is only meaningful in aggregate — see
    /// [`crate::cache`].
    pub decode_cache: Arc<DecodeCache<DecodedChunk>>,
    /// Which file this is, so cached chunks of different files cannot collide.
    pub file_id: FileId,
    /// The parsed Parquet footer.
    pub metadata: Arc<ParquetMetaData>,
    /// arrow-rs's view of the same footer, with string columns requested as Arrow view types.
    pub arrow_metadata: ArrowReaderMetadata,
    /// The total length of the file, needed to satisfy arrow-rs's absolute offsets.
    pub file_len: u64,
    /// Which decode path this file's chunks took. See [`crate::metrics`].
    pub metrics: Arc<ScanMetrics>,
    /// Filter-evaluation and decoder-skipping policy captured when the file was opened.
    pub options: ReaderOptions,
}

/// A footer-referenced offset index registered for lazy I/O.
#[derive(Clone, Debug)]
pub struct OffsetIndexSegment {
    /// Physical leaf ordinal in the Parquet schema.
    pub leaf: usize,
    /// Segment containing the serialized offset index.
    pub id: SegmentId,
    /// Exact serialized-index range.
    pub range: std::ops::Range<u64>,
}

/// A footer-referenced column index registered for lazy I/O.
#[derive(Clone, Debug)]
pub struct ColumnIndexSegment {
    /// Physical leaf ordinal in the Parquet schema.
    pub leaf: usize,
    /// Segment containing the serialized column index.
    pub id: SegmentId,
    /// Exact serialized-index range.
    pub range: std::ops::Range<u64>,
}

/// A footer-referenced split-block Bloom filter registered for lazy pruning I/O.
#[derive(Clone, Debug)]
pub struct BloomFilterSegment {
    /// Physical leaf ordinal in the Parquet schema.
    pub leaf: usize,
    /// Segment containing the serialized Bloom header and bitset.
    pub id: SegmentId,
    /// Exact serialized Bloom-filter range.
    pub range: std::ops::Range<u64>,
}

/// Which column chunk a leaf covers.
#[derive(Clone, Debug)]
pub struct ParquetFieldData {
    /// Shared per-file metadata.
    pub ctx: Arc<FileContext>,
    /// The row group index.
    pub row_group: usize,
    /// The index of the top-level column within the Arrow schema.
    pub root_column: usize,
    /// The first byte of the column chunk, in file coordinates.
    pub byte_offset: u64,
    /// The length of the column chunk in bytes.
    pub byte_length: u64,
    /// Offset indexes for every physical leaf of this logical field that supplied one.
    pub offset_indexes: Vec<OffsetIndexSegment>,
    /// Column indexes for every physical leaf of this logical field that supplied one.
    pub column_indexes: Vec<ColumnIndexSegment>,
    /// Bloom filters for physical leaves that supplied one.
    pub bloom_filters: Vec<BloomFilterSegment>,
    /// Wire dtype of each physical column-chunk child.
    pub physical_dtypes: Vec<DType>,
}

impl ParquetFieldData {
    /// Extracts everything a decode of this chunk needs, leaving [`FileContext`] behind.
    ///
    /// See [`ChunkDecode`] for why the distinction matters.
    pub fn decode_inputs(&self) -> ChunkDecode {
        ChunkDecode {
            file: self.ctx.file.clone(),
            metadata: Arc::clone(&self.ctx.metadata),
            arrow_metadata: self.ctx.arrow_metadata.clone(),
            file_len: self.ctx.file_len,
            metrics: Arc::clone(&self.ctx.metrics),
            options: self.ctx.options,
            row_group: self.row_group,
            root_column: self.root_column,
            byte_offset: self.byte_offset,
            byte_length: self.byte_length,
            offset_indexes: self.offset_indexes.clone(),
            column_indexes: self.column_indexes.clone(),
            bloom_filters: self.bloom_filters.clone(),
        }
    }
}

/// Everything a decode of one column chunk needs, and deliberately nothing more.
///
/// # Why this is not just `ParquetFieldData`
///
/// A decode future is stored in the [`DecodeCache`], and the cache is owned by
/// [`ParquetFormat`](crate::ParquetFormat) rather than by any one file — so it outlives every file it
/// holds entries for. Capturing [`ParquetFieldData`] in that future captured `Arc<FileContext>`, which
/// holds the `Arc<DecodeCache>` the future is stored in: a cycle. The visible cost was retention
/// rather than a leak of the usual kind — the `ParquetMetaData` of every file ever opened stayed
/// resident until `flush_caches`, and a wide file's footer is large.
///
/// So the future captures this instead. Every field is either a `Copy` scalar or an `Arc` into
/// something the footer already owns, which is why constructing one per decode is cheap; what it does
/// not hold is a route back to the cache.
#[derive(Clone, Debug)]
pub struct ChunkDecode {
    /// The file backing this decode. Unlike `FileContext`, this does not point back to the decode
    /// cache and therefore does not create a retention cycle.
    pub file: FileHandle,
    /// The parsed Parquet footer.
    pub metadata: Arc<ParquetMetaData>,
    /// arrow-rs's view of the same footer.
    pub arrow_metadata: ArrowReaderMetadata,
    /// The total length of the file, needed to satisfy arrow-rs's absolute offsets.
    pub file_len: u64,
    /// Which decode path this file's chunks took.
    ///
    /// An `Arc` to the counters rather than the [`FileContext`] holding them, for the reason this
    /// whole type exists: the context would drag the decode cache into the future stored in it.
    pub metrics: Arc<ScanMetrics>,
    /// Filter-evaluation and decoder-skipping policy.
    pub options: ReaderOptions,
    /// The row group index.
    pub row_group: usize,
    /// The index of the top-level column within the Arrow schema.
    pub root_column: usize,
    /// The first byte of the column chunk, in file coordinates.
    pub byte_offset: u64,
    /// The length of the column chunk in bytes.
    pub byte_length: u64,
    /// Lazily requested offset indexes for physical leaves.
    pub offset_indexes: Vec<OffsetIndexSegment>,
    /// Lazily requested column indexes for physical leaves.
    pub column_indexes: Vec<ColumnIndexSegment>,
    /// Lazily requested Bloom filters for physical leaves.
    pub bloom_filters: Vec<BloomFilterSegment>,
}

/// The error every Parquet layout returns for a [`LayoutReader`](vortex::layout::LayoutReader).
///
/// Parquet files scan through Vortex's planner-driven executor only: each layout lowers to a
/// [`PlanRef`] (see [`crate::plan`]). Open a file with [`crate::ParquetFormat::open`] and pass the
/// returned [`ScanFile`](vortex::layout::scan::v2::ScanFile) to a V2 scan.
pub(crate) fn no_layout_reader<T>(id: LayoutId) -> VortexResult<T> {
    vortex_bail!("layout '{id}' has no LayoutReader; scan Parquet files through the V2 scan plan")
}

/// The logical-field layout vtable.
#[derive(Clone, Debug)]
pub struct ParquetField;

/// A layout over one top-level column of one Parquet row group.
pub type ParquetFieldLayout = Layout<ParquetField>;

impl VTable for ParquetField {
    type LayoutData = ParquetFieldData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.parquet.field");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        _args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        // These layouts describe bytes in a Parquet file and are built in memory by [`crate::open`];
        // they are never written into a Vortex footer, so there is nothing to deserialize them from.
        // Reaching here means someone registered this vtable in a session and tried to read a Vortex
        // file that claims to contain it.
        vortex_bail!(
            "vortex.parquet.field layouts are constructed in memory from a Parquet footer and \
             cannot be deserialized from a Vortex file"
        )
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        layout
            .data()
            .physical_dtypes
            .get(idx)
            .cloned()
            .ok_or_else(|| vortex_err!("vortex.parquet.field child {idx} out of bounds"))
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        LayoutChildType::Auxiliary(format!("column-chunk[{idx}]").into())
    }

    fn new_reader(
        layout: &Layout<Self>,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        no_layout_reader(layout.vtable().id())
    }

    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        crate::plan::field_plan(layout)
    }
}

/// Constructs a logical-field layout over the physical column chunks described by `data`.
///
/// `segment_id` must be the id returned when the chunk's byte range was registered in the file's
/// [`SegmentTable`](crate::segments::SegmentTable). A free function rather than an inherent
/// constructor, because `Layout` is Vortex's type and Rust reserves inherent impls for the defining
/// crate.
pub fn field_layout(
    dtype: DType,
    row_count: u64,
    segment_id: SegmentId,
    data: ParquetFieldData,
    column_chunks: Vec<Arc<dyn vortex::layout::DynLayout>>,
) -> ParquetFieldLayout {
    LayoutParts::new(
        ParquetField,
        dtype,
        row_count,
        vec![segment_id],
        layout_children(column_chunks),
        data,
    )
    .into_typed()
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use arrow_array::Int32Array;
    use arrow_array::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ArrowReaderOptions;
    use parquet::file::metadata::ParquetMetaDataReader;

    use super::*;
    use crate::cache::CachePolicy;

    /// A one-column, one-row-group Parquet file, parsed the way [`crate::open`] parses one.
    ///
    /// Real metadata rather than a stub because [`ArrowReaderMetadata`] has no constructor that does
    /// not go through a footer, and a fake would not tell us anything about what the real one retains.
    fn file_context() -> Arc<FileContext> {
        let batch = RecordBatch::try_from_iter([(
            "a",
            Arc::new(Int32Array::from_iter_values(0..8)) as arrow_array::ArrayRef,
        )])
        .expect("the fixture batch is well-formed");

        let mut bytes: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None)
            .expect("the fixture schema is writable");
        writer.write(&batch).expect("the fixture batch is writable");
        writer.close().expect("the fixture file closes");

        let footer_len = u32::from_le_bytes(
            bytes[bytes.len() - 8..bytes.len() - 4]
                .try_into()
                .expect("the footer length is four bytes"),
        ) as usize;
        let footer_start = bytes.len() - 8 - footer_len;
        let metadata = Arc::new(
            ParquetMetaDataReader::decode_metadata(&bytes[footer_start..bytes.len() - 8])
                .expect("the fixture footer decodes"),
        );
        let arrow_metadata =
            ArrowReaderMetadata::try_new(Arc::clone(&metadata), ArrowReaderOptions::default())
                .expect("the fixture footer has an Arrow schema");

        Arc::new(FileContext {
            file: FileHandle::new(
                Arc::new(vortex::buffer::ByteBuffer::copy_from(bytes.clone())),
                bytes.len() as u64,
                VortexSession::empty(),
            ),
            decode_cache: Arc::new(DecodeCache::new(CachePolicy::default())),
            file_id: FileId::next(),
            metadata,
            arrow_metadata,
            file_len: bytes.len() as u64,
            metrics: Arc::default(),
            options: ReaderOptions::default(),
        })
    }

    /// A [`ChunkDecode`] must not keep its [`FileContext`] alive.
    ///
    /// This is the whole reason the type exists. A cached decode future captures a `ChunkDecode` and
    /// the cache outlives every file it holds entries for, so if this retained the context it would
    /// also retain the `Arc<DecodeCache>` the future is stored in — a cycle, and the visible symptom is
    /// that every opened file's `ParquetMetaData` stays resident until `flush_caches`.
    ///
    /// The rest of the invariant is the compiler's: [`crate::reader`] hands the decode future nothing
    /// but a `ChunkDecode`, a `DType`, a row count and a segment future, none of which can reach a
    /// `FileContext`. So this test covers the one thing a future change could quietly break.
    #[test]
    fn decode_inputs_do_not_retain_the_file_context() {
        let ctx = file_context();
        let weak: Weak<FileContext> = Arc::downgrade(&ctx);

        let data = ParquetFieldData {
            ctx,
            row_group: 0,
            root_column: 0,
            byte_offset: 4,
            byte_length: 32,
            offset_indexes: Vec::new(),
            column_indexes: Vec::new(),
            bloom_filters: Vec::new(),
            physical_dtypes: Vec::new(),
        };
        let inputs = data.decode_inputs();

        // Everything holding the context strongly is now gone except, possibly, `inputs`.
        drop(data);
        assert!(
            weak.upgrade().is_none(),
            "a decode's inputs retain the file context, so the decode cache retains every \
             opened file's footer"
        );

        // Held past the assertion, so the compiler cannot drop it early and make this vacuous.
        assert_eq!(inputs.row_group, 0);
    }
}
