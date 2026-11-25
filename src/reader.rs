// SPDX-License-Identifier: Apache-2.0

//! The leaf reader: decodes one column chunk and evaluates expressions over it.
//!
//! The evaluation methods follow the same shape as Vortex's own `FlatReader`, which is the reference
//! for what the [`LayoutReader`] contract means in practice. Deviating from that shape is how a
//! foreign-format reader ends up subtly wrong, so the differences here are only the ones Parquet
//! forces.
//!
//! Two post-conditions are load-bearing and neither fails loudly when broken: a filter result must be
//! intersected with the input mask, and a projection must filter *before* applying the projection
//! expression. Both are marked at their call sites below.

use std::any::Any;
use std::io::Cursor;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::arrow::arrow_reader::RowSelector;
use parquet::basic::Type as PhysicalType;
use parquet::bloom_filter::Sbbf;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::metadata::OffsetIndexBuilder;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataBuilder;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
#[allow(deprecated)]
use parquet::file::page_index::index_reader::read_columns_indexes;
#[allow(deprecated)]
use parquet::file::page_index::index_reader::read_offset_indexes;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
#[allow(deprecated)]
use parquet::format::OffsetIndex;
#[allow(deprecated)]
use parquet::format::PageHeader;
#[allow(deprecated)]
use parquet::format::PageLocation;
#[allow(deprecated)]
use parquet::format::PageType;
use parquet::thrift::TSerializable;
use thrift::protocol::TCompactInputProtocol;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::MaskFuture;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::builders::builder_with_capacity_in;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::memory::MemorySessionExt;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::DType;
use vortex::dtype::FieldMask;
use vortex::dtype::PType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::ArrayExprExt;
use vortex::expr::BoundExpression;
use vortex::functions::dynamic::DynamicExprUpdates;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutReader;
use vortex::layout::RowSplits;
use vortex::layout::SplitRange;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar_fn::builtin::compare::Compare;
use vortex::scalar_fn::builtin::literal::Literal;
use vortex::scalar_fn::builtin::operators::CompareOperator;
use vortex::session::VortexSession;

use crate::cache::ChunkKey;
use crate::cache::DecodeCache;
use crate::cache::DecodeUnit;
use crate::chunk_reader::SegmentChunkReader;
use crate::layout::ChunkDecode;
use crate::layout::PageExecutionData;
use crate::layout::ParquetFieldLayout;
use crate::offset_index_is_sound;
use crate::pages;
use crate::segments::SegmentTable;
use crate::statistics::ParquetStats;
use crate::statistics::can_prune;
use crate::statistics::page_index_statistics;

/// A field decode with lazily loaded offset indexes attached to parquet-rs and retained in their
/// wire form so the same metadata can drive page-granular I/O.
#[allow(deprecated)]
struct IndexedDecode {
    inputs: ChunkDecode,
    indexes: Vec<(usize, OffsetIndex)>,
    /// This row group's offset indexes, already converted to arrow-rs's metadata form.
    ///
    /// One row group's worth, not the file's: the nested fallback needs a `ParquetMetaData` carrying
    /// an offset index, and keeping the converted row group lets that be assembled on demand without
    /// retaining a clone of the whole footer. See [`augment_metadata`].
    row_group_indexes: Vec<OffsetIndexMetaData>,
}

type SharedIndexedDecode =
    Shared<BoxFuture<'static, Result<Option<Arc<IndexedDecode>>, Arc<VortexError>>>>;

/// A decoded Parquet page-index pair, retained independently of the predicate evaluated over it.
///
/// Dynamic filters may tighten several times while a row group is in flight. The index bytes and
/// Thrift structures are immutable; only the Vortex falsification expression needs reevaluation.
/// This is the Parquet analogue of ZonedLayout loading one zone map and deriving a new mask for
/// each dynamic-expression revision.
#[allow(deprecated)]
struct PagePruningIndexes {
    offset: OffsetIndexMetaData,
    column: ColumnIndexMetaData,
}

type SharedPagePruningIndexes =
    Shared<BoxFuture<'static, Result<Option<Arc<PagePruningIndexes>>, Arc<VortexError>>>>;

/// The largest batch arrow-rs is asked for, whatever the footer claims the chunk holds.
///
/// A footer is data. Handing an unchecked row count to `with_batch_size` lets a hostile file size
/// arrow-rs's buffers, so the request is bounded and a genuinely larger chunk simply arrives as
/// several batches.
pub(crate) const MAX_BATCH_ROWS: usize = 1 << 20;

/// Shared so that the several evaluations of one split decode the column chunk once.
pub type SharedArrayFuture = Shared<BoxFuture<'static, Result<ArrayRef, Arc<VortexError>>>>;

/// Local alias, kept short at the many use sites below.
type SharedArray = SharedArrayFuture;

/// A decode shared between live evaluations of the same chunk.
///
/// Sharing is the point. Vortex calls `filter_evaluation` and `projection_evaluation` on the same reader
/// for the same split, so `SELECT MIN("URL") ... WHERE "URL" LIKE ...` — ClickBench has several — would
/// otherwise decode that column chunk twice, and `SplitBy::Layout` multiplies that by the splits per row
/// group. Lifetime is [`crate::cache::DecodeCache`]'s problem: the default holds only a weak reference
/// once consumers release it, while explicitly retained policies enforce their budget file-wide.
type CachedArray = Arc<SharedArray>;

/// A reader over one top-level column of one Parquet row group.
pub struct ParquetFieldReader {
    layout: ParquetFieldLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    physical: Option<Arc<dyn LayoutReader>>,
    page_pruning_indexes: Arc<OnceLock<SharedPagePruningIndexes>>,
    offset_indexes: Arc<OnceLock<SharedIndexedDecode>>,
    /// Dictionary values are tiny and immutable; native `DictReader` retains the same thing.
    dictionary_values: Arc<OnceLock<ArrayRef>>,
}

impl ParquetFieldReader {
    /// Creates a reader for `layout`.
    pub fn new(
        layout: ParquetFieldLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        physical: Option<Arc<dyn LayoutReader>>,
    ) -> VortexResult<Self> {
        if layout.segment_ids().len() != 1 {
            return Err(vortex_err!(
                "a Parquet field layout must reference exactly one segment, found {}",
                layout.segment_ids().len()
            ));
        }
        Ok(Self {
            layout,
            name,
            segment_source,
            session,
            physical,
            page_pruning_indexes: Arc::new(OnceLock::new()),
            offset_indexes: Arc::new(OnceLock::new()),
            dictionary_values: Arc::new(OnceLock::new()),
        })
    }

    /// A `'static` handle that can fetch or start this chunk's decode from inside a future.
    ///
    /// Needed because [`LayoutReader::projection_evaluation`] must defer that decision until after it
    /// has awaited the mask, by which point `&self` is long gone.
    fn cached_decode(&self) -> CachedDecode {
        let data = self.layout.data();
        CachedDecode {
            cache: Arc::clone(&data.ctx.decode_cache),
            key: (
                data.ctx.file_id,
                data.row_group,
                DecodeUnit::Column(data.root_column),
            ),
            inputs: data.decode_inputs(),
            dtype: self.layout.dtype().clone(),
            row_count: self.layout.row_count(),
            segment_source: Arc::clone(&self.segment_source),
            segment_id: self.layout.segment_ids()[0],
            offset_indexes: Arc::clone(&self.offset_indexes),
            dictionary_values: Arc::clone(&self.dictionary_values),
        }
    }
}

/// Executable reader for a column chunk's lazy page-set child.
///
/// Indexed and header-discovered page layouts share execution semantics; only their boundary
/// discovery metadata differs. The reader is deliberately attached below `ParquetColumnChunk`, so
/// flat scans actually traverse the hierarchy represented in the layout instead of treating it as
/// explanatory metadata beside an unrelated field decoder.
pub struct ParquetPageSetReader {
    name: Arc<str>,
    dtype: DType,
    row_count: u64,
    segment_id: SegmentId,
    data: PageExecutionData,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    offset_indexes: Arc<OnceLock<SharedIndexedDecode>>,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
}

impl ParquetPageSetReader {
    /// Creates an executable lazy page-set reader.
    pub fn new(
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        segment_id: SegmentId,
        data: PageExecutionData,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        vortex_ensure_page_range(&data)?;
        Ok(Self {
            name,
            dtype,
            row_count,
            segment_id,
            data,
            segment_source,
            session,
            offset_indexes: Arc::new(OnceLock::new()),
            dictionary_values: Arc::new(OnceLock::new()),
        })
    }

    fn cached_decode(&self) -> CachedDecode {
        CachedDecode {
            cache: Arc::clone(&self.data.ctx.decode_cache),
            key: (
                self.data.ctx.file_id,
                self.data.row_group,
                DecodeUnit::Column(self.data.root_column),
            ),
            inputs: self.data.decode_inputs(),
            dtype: self.dtype.clone(),
            row_count: self.row_count,
            segment_source: Arc::clone(&self.segment_source),
            segment_id: self.segment_id,
            offset_indexes: Arc::clone(&self.offset_indexes),
            dictionary_values: Arc::clone(&self.dictionary_values),
        }
    }
}

fn evaluate_filter(
    array: ArrayRef,
    expr: &BoundExpression,
    row_range: Range<usize>,
    mask: Mask,
    session: &VortexSession,
    threshold: f64,
) -> VortexResult<Mask> {
    if mask.all_false() {
        return Ok(mask);
    }

    // Apply the expression before slicing the split. This keeps casts and predicates directly above
    // encoded pages, where structural and execute-parent kernels can run. Reversing the order makes
    // `Cast(Slice(Chunked(...)))`, forcing Slice to canonicalize bytes before cast-on-decode sees
    // them.
    let mut evaluated = array.apply_bound(expr)?;
    if row_range.start > 0 || row_range.end < evaluated.len() {
        evaluated = evaluated.slice(row_range)?;
    }
    if mask.density() < threshold {
        // Keep the expression above the encoded page so its own structural/execute-parent kernels
        // get first refusal, then retain the filter in the array tree. Vortex's filter rules commute
        // it through safe scalar functions and eventually reach the Parquet page FilterKernel.
        let evaluated = evaluated.filter(mask.clone())?;
        let mut ctx = session.create_execution_ctx();
        let evaluated = evaluated.fill_null(false)?.execute::<Mask>(&mut ctx)?;
        Ok(mask.intersect_by_rank(&evaluated))
    } else {
        let mut ctx = session.create_execution_ctx();
        let evaluated = evaluated.fill_null(false)?.execute::<Mask>(&mut ctx)?;
        Ok(mask.bitand(&evaluated))
    }
}

/// Evaluates a predicate over an array that already contains only the rows selected by `mask`.
///
/// This is the mask-aware page path: pages containing no selected row were skipped while still
/// compressed, so applying `mask` to the resulting array again would be both wrong and wasteful.
fn evaluate_selected_filter(
    array: ArrayRef,
    expr: &BoundExpression,
    mask: Mask,
    session: &VortexSession,
) -> VortexResult<Mask> {
    vortex_ensure!(
        array.len() == mask.true_count(),
        "selective filter decode returned {} rows for a {}-row mask",
        array.len(),
        mask.true_count()
    );
    let mut ctx = session.create_execution_ctx();
    let evaluated = array
        .apply_bound(expr)?
        .fill_null(false)?
        .execute::<Mask>(&mut ctx)?;
    Ok(mask.intersect_by_rank(&evaluated))
}

fn vortex_ensure_page_range(data: &PageExecutionData) -> VortexResult<()> {
    let end = data
        .byte_offset
        .checked_add(data.byte_length)
        .ok_or_else(|| vortex_err!("Parquet page-set byte range overflows u64"))?;
    if end > data.ctx.file_len {
        vortex_bail!(
            "Parquet page-set range {}..{} extends past its {}-byte file",
            data.byte_offset,
            end,
            data.ctx.file_len
        );
    }
    Ok(())
}

/// Everything needed to fetch a chunk's decode, detached from the reader that described it.
///
/// Deliberately holds no `Arc<FileContext>`: see [`ChunkDecode`] for the cycle that caused.
#[derive(Clone)]
struct CachedDecode {
    cache: Arc<DecodeCache<SharedArrayFuture>>,
    key: ChunkKey,
    inputs: ChunkDecode,
    dtype: DType,
    row_count: u64,
    segment_source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    offset_indexes: Arc<OnceLock<SharedIndexedDecode>>,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
}

/// Translates `mask`, which covers `row_range` within the chunk, into a Parquet `RowSelection`.
///
/// The selection is expressed over the whole row group, because that is the unit
/// [`decode_arrow`] hands arrow-rs — so it opens by skipping everything before `row_range`.
fn row_selection(row_range: &Range<usize>, mask: &Mask) -> RowSelection {
    let mut selectors: Vec<RowSelector> = Vec::new();
    if row_range.start > 0 {
        selectors.push(RowSelector::skip(row_range.start));
    }

    // Run-length encoded from the mask's bools: a selection is runs, not indices, and building it
    // per-index would produce one selector per row.
    let mut run = 0usize;
    let mut selected = false;
    for value in mask.iter() {
        if value == selected {
            run += 1;
            continue;
        }
        if run > 0 {
            selectors.push(if selected {
                RowSelector::select(run)
            } else {
                RowSelector::skip(run)
            });
        }
        selected = value;
        run = 1;
    }
    if run > 0 {
        selectors.push(if selected {
            RowSelector::select(run)
        } else {
            RowSelector::skip(run)
        });
    }

    RowSelection::from(selectors)
}

/// Places a split-local mask back into the column chunk's row domain.
///
/// Sparse masks stay sparse: offsetting their cached indices avoids constructing and walking a full
/// row-group bit buffer once per projected column.
fn chunk_mask(row_range: &Range<usize>, mask: &Mask, chunk_len: usize) -> VortexResult<Mask> {
    vortex_ensure!(
        row_range.end >= row_range.start
            && row_range.end - row_range.start == mask.len()
            && row_range.end <= chunk_len,
        "split-local mask does not fit its column chunk"
    );
    if row_range.start == 0 && row_range.end == chunk_len {
        return Ok(mask.clone());
    }
    Ok(match mask.indices() {
        vortex::mask::AllOr::All => Mask::from_slices(
            chunk_len,
            (row_range.start != row_range.end)
                .then_some((row_range.start, row_range.end))
                .into_iter()
                .collect(),
        ),
        vortex::mask::AllOr::None => Mask::new_false(chunk_len),
        vortex::mask::AllOr::Some(indices) => Mask::from_indices(
            chunk_len,
            indices.iter().map(|index| row_range.start + index),
        ),
    })
}

impl CachedDecode {
    fn logical_is_decimal(&self) -> bool {
        matches!(
            self.inputs
                .arrow_metadata
                .schema()
                .field(self.inputs.root_column)
                .data_type(),
            arrow_schema::DataType::Decimal32(_, _)
                | arrow_schema::DataType::Decimal64(_, _)
                | arrow_schema::DataType::Decimal128(_, _)
                | arrow_schema::DataType::Decimal256(_, _)
        )
    }

    /// Whether this chunk should be decoded through `mask` rather than in full.
    ///
    /// The result cannot be reused for another mask, so selection is reserved for masks retaining at
    /// at most the configured fraction of rows. The offset-index path applies a second byte-cost
    /// check after it knows page boundaries, so this gate primarily avoids non-reusable selective
    /// decodes for dense masks while still admitting useful page and decoder skipping.
    fn worth_selecting(&self, mask: &Mask) -> bool {
        !mask.all_true()
            && mask.true_count() != 0
            && mask.density() <= self.inputs.options.selective_decode_threshold()
    }

    /// Whether the current mask leaves at least one indexed data page completely unused.
    ///
    /// This decision is deliberately independent of row density.  A 95%-dense mask can still omit
    /// an entire large page, and downloading that page only because the generic selective threshold
    /// is 20% violates page-level late materialization.  The offset-index future is shared, so a
    /// filter and its projected siblings pay this metadata read at most once.
    async fn skips_indexed_page(
        &self,
        row_range: &Range<usize>,
        mask: &Mask,
    ) -> VortexResult<bool> {
        // Sparse decoding can discover headers later; a dense-cost probe need not.
        if self.inputs.offset_indexes.is_empty()
            || (mask.all_true()
                && row_range.start == 0
                && u64::try_from(row_range.end)? == self.row_count)
        {
            return Ok(false);
        }
        let Some(indexed) = self.with_offset_indexes().await? else {
            return Ok(false);
        };
        let Some(selected) = selected_page_ranges(
            &indexed.inputs,
            &indexed.indexes,
            row_range,
            mask,
            self.row_count,
        )?
        else {
            return Ok(false);
        };
        let selected_bytes = selected
            .ranges
            .iter()
            .try_fold(0u64, |total, range| {
                total.checked_add(range.end - range.start)
            })
            .ok_or_else(|| vortex_err!("selected Parquet page bytes overflow u64"))?;
        Ok(selected_bytes < indexed.inputs.byte_length)
    }

    /// Decodes **only** the rows `mask` selects, bypassing the cache.
    ///
    /// The result is valid for this mask alone, so it is deliberately not cached. It uses the
    /// Flat indexed columns remain Parquet page arrays: only selected page spans are fetched, then
    /// their native filter kernels skip false runs inside those pages. Nested fields fall back to
    /// arrow-rs with the same `RowSelection` because Dremel reconstruction belongs there.
    ///
    /// The returned array already holds exactly the selected rows, so the caller must **not** filter
    /// it again — that is the post-condition `projection_evaluation` owes its caller, satisfied by
    /// construction rather than afterwards.
    async fn decode_selected(
        &self,
        row_range: &Range<usize>,
        mask: &Mask,
    ) -> VortexResult<ArrayRef> {
        let indexed = self.with_offset_indexes().await?;
        let rows = u64::try_from(mask.true_count())?;
        let mut used_page_decoder = false;
        let array = if let Some(indexed) = indexed {
            let selected = selected_page_ranges(
                &indexed.inputs,
                &indexed.indexes,
                row_range,
                mask,
                self.row_count,
            )?;
            let selected_bytes = selected
                .as_ref()
                .map(|selection| {
                    selection
                        .ranges
                        .iter()
                        .map(|range| range.end - range.start)
                        .sum::<u64>()
                })
                .unwrap_or(indexed.inputs.byte_length);
            if let Some(selected_pages) = selected.filter(|selection| !selection.ranges.is_empty())
            {
                let SelectedPages {
                    ranges,
                    all_locations,
                    selected_locations,
                } = selected_pages;
                let reader = fetch_page_ranges(&indexed.inputs, ranges).await?;
                let schema = indexed.inputs.metadata.file_metadata().schema_descr();
                let direct = if let Some(leaf) =
                    pages::single_leaf_index(schema, indexed.inputs.root_column)
                {
                    pages::decode_selected_chunk(
                        reader.clone(),
                        indexed
                            .inputs
                            .metadata
                            .row_group(indexed.inputs.row_group)
                            .column(leaf),
                        &schema.column(leaf),
                        &self.dtype,
                        self.row_count,
                        &all_locations,
                        selected_locations,
                        row_range,
                        mask,
                        indexed.inputs.options,
                        Arc::clone(&indexed.inputs.metrics),
                        Arc::clone(&self.dictionary_values),
                    )?
                } else {
                    None
                };
                if let Some(array) = direct {
                    indexed.inputs.metrics.record_page_decode();
                    used_page_decoder = true;
                    array
                } else if selected_bytes.saturating_mul(5)
                    < indexed.inputs.byte_length.saturating_mul(4)
                {
                    // Page requests trade extra bookkeeping and smaller coalescing opportunities
                    // for bytes. For nested/Dremel fields, use page-index I/O whenever it removes
                    // at least a fifth of the field. Flat supported fields never reach this policy:
                    // preserving their native page/Dict representation is more important than the
                    // I/O shape and the page decoder above handles both dense and sparse masks.
                    decode_arrow_reader(
                        &indexed.inputs,
                        &self.dtype,
                        rows,
                        reader,
                        Some(row_selection(row_range, mask)),
                        Some(indexed_arrow_metadata(indexed.as_ref())?),
                    )?
                } else {
                    let bytes = self
                        .segment_source
                        .request(self.segment_id)
                        .await?
                        .unwrap_host();
                    let range = indexed.inputs.byte_offset
                        ..indexed.inputs.byte_offset + indexed.inputs.byte_length;
                    if let Some(array) = decode_full_page_selection(
                        &indexed.inputs,
                        &self.dtype,
                        self.row_count,
                        bytes.clone(),
                        range.clone(),
                        row_range,
                        mask,
                        Arc::clone(&self.dictionary_values),
                    )? {
                        indexed.inputs.metrics.record_page_decode();
                        used_page_decoder = true;
                        array
                    } else {
                        decode_arrow(
                            &indexed.inputs,
                            &self.dtype,
                            rows,
                            bytes,
                            range,
                            Some(row_selection(row_range, mask)),
                        )?
                    }
                }
            } else {
                let bytes = self
                    .segment_source
                    .request(self.segment_id)
                    .await?
                    .unwrap_host();
                let range = indexed.inputs.byte_offset
                    ..indexed.inputs.byte_offset + indexed.inputs.byte_length;
                if let Some(array) = decode_full_page_selection(
                    &indexed.inputs,
                    &self.dtype,
                    self.row_count,
                    bytes.clone(),
                    range.clone(),
                    row_range,
                    mask,
                    Arc::clone(&self.dictionary_values),
                )? {
                    indexed.inputs.metrics.record_page_decode();
                    used_page_decoder = true;
                    array
                } else {
                    decode_arrow(
                        &indexed.inputs,
                        &self.dtype,
                        rows,
                        bytes,
                        range,
                        Some(row_selection(row_range, mask)),
                    )?
                }
            }
        } else {
            let bytes = self
                .segment_source
                .request(self.segment_id)
                .await?
                .unwrap_host();
            let range = self.inputs.byte_offset..self.inputs.byte_offset + self.inputs.byte_length;
            let schema = self.inputs.metadata.file_metadata().schema_descr();
            let direct = if let Some(leaf) =
                pages::single_leaf_index(schema, self.inputs.root_column)
            {
                let reader =
                    SegmentChunkReader::new(bytes.clone(), range.clone(), self.inputs.file_len)?;
                pages::decode_selected_records(
                    reader,
                    self.inputs
                        .metadata
                        .row_group(self.inputs.row_group)
                        .column(leaf),
                    &self.dtype,
                    self.row_count,
                    row_range,
                    mask,
                    &mut self.inputs.file.session().create_execution_ctx(),
                )?
            } else {
                None
            };
            if let Some(array) = direct {
                self.inputs.metrics.record_page_decode();
                used_page_decoder = true;
                array
            } else {
                decode_arrow(
                    &self.inputs,
                    &self.dtype,
                    rows,
                    bytes,
                    range,
                    Some(row_selection(row_range, mask)),
                )?
            }
        };
        if !used_page_decoder {
            self.inputs.metrics.record_arrow_decode();
        }
        self.inputs
            .metrics
            .record_selective_decode(rows, u64::try_from(mask.len())?);

        if array.len() != mask.true_count() {
            vortex_bail!(
                "a selective decode of {} returned {} rows, expected {}",
                self.inputs.root_column,
                array.len(),
                mask.true_count()
            );
        }
        Ok(array)
    }

    /// Loads this field's offset indexes only after a sparse mask chooses the selective path.
    #[allow(deprecated)]
    async fn with_offset_indexes(&self) -> VortexResult<Option<Arc<IndexedDecode>>> {
        self.offset_indexes
            .get_or_init(|| {
                let decode = self.clone();
                async move {
                    decode
                        .load_offset_indexes()
                        .await
                        .map(|indexed| indexed.map(Arc::new))
                        .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
            .await
            .map_err(|error| vortex_err!("failed to load Parquet offset indexes: {error}"))
    }

    /// Performs the one lazy index read shared by all selective decodes of this field.
    #[allow(deprecated)]
    async fn load_offset_indexes(&self) -> VortexResult<Option<IndexedDecode>> {
        let schema = self.inputs.metadata.file_metadata().schema_descr();
        let leaves: Vec<usize> = (0..schema.num_columns())
            .filter(|leaf| schema.get_column_root_idx(*leaf) == self.inputs.root_column)
            .collect();
        let decoded_indexes = if self.inputs.offset_indexes.is_empty() && leaves.len() == 1 {
            let leaf = leaves[0];
            let descriptor = schema.column(leaf);
            if descriptor.max_rep_level() != 0 {
                return Ok(None);
            }
            let column = self
                .inputs
                .metadata
                .row_group(self.inputs.row_group)
                .column(leaf);
            let Some(index) = discover_header_index(&self.inputs, column, self.row_count).await?
            else {
                return Ok(None);
            };
            vec![(leaf, index)]
        } else {
            if leaves.len() != self.inputs.offset_indexes.len()
                || leaves.iter().any(|leaf| {
                    !self
                        .inputs
                        .offset_indexes
                        .iter()
                        .any(|index| index.leaf == *leaf)
                })
            {
                return Ok(None);
            }

            // Construct every future before polling any of them, so sibling index reads enter the
            // I/O driver's coalescing window together.
            let requested = self
                .inputs
                .offset_indexes
                .iter()
                .map(|index| (index, self.segment_source.request(index.id)))
                .collect::<Vec<_>>();
            let mut decoded_indexes = Vec::with_capacity(requested.len());
            for (index, future) in requested {
                let bytes = future.await?.unwrap_host();
                let mut cursor = Cursor::new(bytes.as_ref());
                let decoded = {
                    let mut protocol = TCompactInputProtocol::new(&mut cursor);
                    OffsetIndex::read_from_in_protocol(&mut protocol)
                }
                .map_err(|error| {
                    vortex_err!(
                        "failed to decode row group {} column {} offset index {:?}: {error}",
                        self.inputs.row_group,
                        index.leaf,
                        index.range
                    )
                })?;
                self.inputs.metrics.record_offset_index_decode();
                decoded_indexes.push((index.leaf, decoded));
            }
            decoded_indexes
        };

        let mut row_group_indexes = (0..self
            .inputs
            .metadata
            .row_group(self.inputs.row_group)
            .num_columns())
            .map(|_| OffsetIndexBuilder::new().build())
            .collect::<Vec<_>>();
        for (leaf, decoded) in &decoded_indexes {
            let mut builder = OffsetIndexBuilder::new();
            let page_count = decoded.page_locations.len();
            for (page, location) in decoded.page_locations.iter().enumerate() {
                let next_row = decoded
                    .page_locations
                    .get(page + 1)
                    .map_or(i64::try_from(self.row_count)?, |next| next.first_row_index);
                let row_count =
                    next_row
                        .checked_sub(location.first_row_index)
                        .ok_or_else(|| {
                            vortex_err!("offset-index rows are not ordered for leaf {leaf}")
                        })?;
                builder.append_offset_and_size(location.offset, location.compressed_page_size);
                builder.append_row_count(row_count);
                builder.append_unencoded_byte_array_data_bytes(
                    decoded
                        .unencoded_byte_array_data_bytes
                        .as_ref()
                        .and_then(|values| values.get(page))
                        .copied(),
                );
            }
            if decoded
                .unencoded_byte_array_data_bytes
                .as_ref()
                .is_some_and(|values| values.len() != page_count)
            {
                return Ok(None);
            }
            row_group_indexes[*leaf] = builder.build();
        }

        let mut indexes = self
            .inputs
            .metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                (0..row_group.num_columns())
                    .map(|_| OffsetIndexBuilder::new().build())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        indexes[self.inputs.row_group] = row_group_indexes.clone();
        // Validate against the shared footer without cloning all row groups and column metadata.
        // Only the nested Arrow fallback needs an augmented metadata object.
        if !offset_index_is_sound(&self.inputs.metadata, &indexes) {
            return Ok(None);
        }
        Ok(Some(IndexedDecode {
            // The *original* `ChunkDecode`, so its `metadata` stays the one `Arc` every leaf shares.
            inputs: self.inputs.clone(),
            indexes: decoded_indexes,
            row_group_indexes,
        }))
    }

    /// Fetches the chunk's decode, starting one if the cache does not already hold it.
    ///
    /// `segment` is a byte request *already made* for this chunk, handed over rather than remade so
    /// that the I/O driver still saw it inside the coalescing window. `None` means "request one now",
    /// which is correct but late: the caller either knew the cache held this chunk — the common case —
    /// or is on the eviction-between-check-and-use path, which costs a round trip rather than an error.
    fn get(self, segment: Option<SegmentFuture>) -> CachedArray {
        let Self {
            cache,
            key,
            inputs,
            dtype,
            row_count,
            segment_source,
            segment_id,
            offset_indexes: _,
            dictionary_values,
        } = self;
        let charge = inputs.byte_length;
        cache.get_or_insert(key, charge, move || {
            decode_chunk(
                inputs,
                dtype,
                row_count,
                segment,
                segment_source,
                segment_id,
                dictionary_values,
            )
        })
    }
}

/// Computes the compressed page spans needed by `mask` for every physical leaf of a field.
///
/// `None` means the offset index is not trustworthy enough for page-granular I/O. That is a
/// performance fallback, not a corrupt-file error: arrow-rs can still decode the whole chunk and
/// apply the same row selection. Footer bounds are nevertheless enforced before any range is used.
#[allow(deprecated)]
struct SelectedPages {
    ranges: Vec<Range<u64>>,
    all_locations: Vec<PageLocation>,
    selected_locations: Vec<PageLocation>,
}

/// Discovers flat-column page boundaries after a sparse mask asks for them.
///
/// A footer without an offset index does not force a whole-column read.  We probe only the compact
/// Thrift header at each page offset, skip its compressed payload by length, and turn the result into
/// the same in-memory offset index used by the indexed layout.  The probes are intentionally small;
/// unusually large headers (for example, long byte-array statistics) retry geometrically.
#[allow(deprecated)]
async fn discover_header_index(
    inputs: &ChunkDecode,
    column: &ColumnChunkMetaData,
    row_count: u64,
) -> VortexResult<Option<OffsetIndex>> {
    let (mut offset, length) = column.byte_range();
    let end = offset
        .checked_add(length)
        .ok_or_else(|| vortex_err!("Parquet column chunk range overflows u64"))?;
    let mut first_row = 0u64;
    let mut locations = Vec::new();
    while offset < end {
        let remaining = end - offset;
        let mut probe = 64u64.min(remaining);
        let (header, header_len) = loop {
            let mut table = SegmentTable::with_capacity(1);
            let id = table.push(offset..offset + probe)?;
            let source = inputs.file.segment_source(table)?;
            let bytes = source.request(id).await?.unwrap_host();
            let mut cursor = Cursor::new(bytes.as_ref());
            let decoded = {
                let mut protocol = TCompactInputProtocol::new(&mut cursor);
                PageHeader::read_from_in_protocol(&mut protocol)
            };
            match decoded {
                Ok(header) => break (header, cursor.position()),
                Err(error) if probe < remaining => {
                    probe = probe.saturating_mul(2).min(remaining);
                    if probe == 0 {
                        return Ok(None);
                    }
                    drop(error);
                }
                Err(error) => {
                    return Err(vortex_err!(
                        "failed to discover Parquet page header at {offset}: {error}"
                    ));
                }
            }
        };
        let payload_len = match u64::try_from(header.compressed_page_size) {
            Ok(length) => length,
            Err(_) => return Ok(None),
        };
        let total_len = match header_len.checked_add(payload_len) {
            Some(length) if length != 0 && length <= remaining => length,
            _ => return Ok(None),
        };
        if header.type_ == PageType::DATA_PAGE || header.type_ == PageType::DATA_PAGE_V2 {
            let page_rows = if header.type_ == PageType::DATA_PAGE_V2 {
                header
                    .data_page_header_v2
                    .as_ref()
                    .and_then(|header| u64::try_from(header.num_rows).ok())
            } else {
                header
                    .data_page_header
                    .as_ref()
                    .and_then(|header| u64::try_from(header.num_values).ok())
            };
            let Some(page_rows) = page_rows else {
                return Ok(None);
            };
            locations.push(PageLocation::new(
                i64::try_from(offset)?,
                i32::try_from(total_len)?,
                i64::try_from(first_row)?,
            ));
            first_row = match first_row.checked_add(page_rows) {
                Some(rows) if rows <= row_count => rows,
                _ => return Ok(None),
            };
        }
        offset += total_len;
    }
    if first_row != row_count || locations.is_empty() {
        return Ok(None);
    }
    Ok(Some(OffsetIndex::new(locations, None)))
}

#[allow(deprecated)]
fn selected_page_ranges(
    inputs: &ChunkDecode,
    indexes: &[(usize, OffsetIndex)],
    row_range: &Range<usize>,
    mask: &Mask,
    row_count: u64,
) -> VortexResult<Option<SelectedPages>> {
    if row_range.end < row_range.start || row_range.end - row_range.start != mask.len() {
        vortex_bail!(
            "selection range {}..{} does not match its {}-row mask",
            row_range.start,
            row_range.end,
            mask.len()
        );
    }
    if u64::try_from(row_range.end)? > row_count {
        vortex_bail!(
            "selection range ends at {}, past the {row_count}-row field",
            row_range.end
        );
    }

    let row_group = inputs.metadata.row_group(inputs.row_group);
    let mut ranges = Vec::new();
    let mut all_locations = Vec::new();
    let mut selected_locations = Vec::new();
    for (leaf, index) in indexes {
        let Some(column) = row_group.columns().get(*leaf) else {
            return Ok(None);
        };
        let data_offset = match u64::try_from(column.data_page_offset()) {
            Ok(offset) => offset,
            Err(_) => return Ok(None),
        };
        let chunk_start = match column.dictionary_page_offset() {
            Some(offset) => match u64::try_from(offset) {
                Ok(offset) => offset.min(data_offset),
                Err(_) => return Ok(None),
            },
            None => data_offset,
        };
        let chunk_length = match u64::try_from(column.compressed_size()) {
            Ok(length) => length,
            Err(_) => return Ok(None),
        };
        let Some(chunk_end) = chunk_start.checked_add(chunk_length) else {
            return Ok(None);
        };
        if chunk_start < inputs.byte_offset
            || chunk_end > inputs.byte_offset.saturating_add(inputs.byte_length)
            || chunk_end > inputs.file_len
        {
            return Ok(None);
        }

        let locations = &index.page_locations;
        if indexes.len() == 1 {
            all_locations = locations.clone();
        }
        let Some(first) = locations.first() else {
            return Ok(None);
        };
        if first.first_row_index != 0 {
            return Ok(None);
        }
        let first_page = match u64::try_from(first.offset) {
            Ok(offset) if offset >= chunk_start && offset <= chunk_end => offset,
            _ => return Ok(None),
        };
        let mut selected_leaf_page = false;
        for (page, location) in locations.iter().enumerate() {
            let start_row = match u64::try_from(location.first_row_index) {
                Ok(row) => row,
                Err(_) => return Ok(None),
            };
            let end_row = match locations.get(page + 1) {
                Some(next) => match u64::try_from(next.first_row_index) {
                    Ok(row) => row,
                    Err(_) => return Ok(None),
                },
                None => row_count,
            };
            if end_row < start_row || end_row > row_count {
                return Ok(None);
            }

            let overlap_start = start_row.max(u64::try_from(row_range.start)?);
            let overlap_end = end_row.min(u64::try_from(row_range.end)?);
            if overlap_start >= overlap_end {
                continue;
            }
            let local_start = usize::try_from(overlap_start)? - row_range.start;
            let local_end = usize::try_from(overlap_end)? - row_range.start;
            if mask.slice(local_start..local_end).all_false() {
                continue;
            }

            let page_start = match u64::try_from(location.offset) {
                Ok(offset) => offset,
                Err(_) => return Ok(None),
            };
            let page_length = match u64::try_from(location.compressed_page_size) {
                Ok(length) => length,
                Err(_) => return Ok(None),
            };
            let Some(page_end) = page_start.checked_add(page_length) else {
                return Ok(None);
            };
            if page_start < first_page || page_end > chunk_end || page_start >= page_end {
                return Ok(None);
            }
            ranges.push(page_start..page_end);
            if indexes.len() == 1 {
                selected_locations.push(location.clone());
            }
            selected_leaf_page = true;
        }

        // A dictionary page has no offset-index entry. When any data page of this leaf is selected,
        // retain exactly the dictionary prefix.  The page array parser does not need the hole
        // between the dictionary and the first selected data page, so unselected data pages remain
        // unfetched even when the first match is late in the chunk.
        if selected_leaf_page && chunk_start < first_page {
            ranges.push(chunk_start..first_page);
        }
    }

    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start <= previous.end
        {
            previous.end = previous.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    Ok(Some(SelectedPages {
        ranges: merged,
        all_locations,
        selected_locations,
    }))
}

/// Fetches disjoint page spans through a temporary coalescing segment source.
async fn fetch_page_ranges(
    inputs: &ChunkDecode,
    ranges: Vec<Range<u64>>,
) -> VortexResult<SegmentChunkReader> {
    let mut table = SegmentTable::with_capacity(ranges.len());
    let mut registered = Vec::with_capacity(ranges.len());
    for range in ranges {
        let id = table.push(range.clone())?;
        registered.push((range, id));
    }
    let source = inputs.file.segment_source(table)?;

    // Request every segment before awaiting the first. FileSegmentSource batches requests that are
    // simultaneously visible, so adjacent selected pages become one physical read while distant
    // pages remain genuinely sparse.
    let requested = registered
        .into_iter()
        .map(|(range, id)| (range, source.request(id)))
        .collect::<Vec<_>>();
    let mut fetched = Vec::with_capacity(requested.len());
    for (range, future) in requested {
        fetched.push((range, future.await?.unwrap_host()));
    }
    SegmentChunkReader::new_multi(fetched, inputs.file_len)
}

/// Decodes one column chunk into a Vortex array once `segment` resolves.
fn decode_chunk(
    inputs: ChunkDecode,
    dtype: DType,
    row_count: u64,
    segment: Option<SegmentFuture>,
    segment_source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
) -> SharedArray {
    async move {
        // Constructing this shared future is metadata-only.  The segment request is created on its
        // first poll, allowing filter and projection graphs to capture the same decode without
        // fetching a page before their input masks resolve.
        let segment = segment.unwrap_or_else(|| segment_source.request(segment_id));
        let bytes = segment.await.map_err(Arc::new)?.unwrap_host();
        let range = inputs.byte_offset..inputs.byte_offset + inputs.byte_length;
        let parquet_schema = inputs.metadata.file_metadata().schema_descr();

        // The page path, when asked for and when this chunk is one it handles. It returns `None` for
        // anything it declines — a nested column, a non-dictionary encoding — and the Arrow path
        // below is the fallback. An `Err` is different: that is our decoder being wrong, and it
        // propagates rather than silently falling back, because a silent fallback would turn a bug
        // into a performance mystery.
        if let Some(leaf) = pages::single_leaf_index(parquet_schema, inputs.root_column) {
            // Cloned rather than moved: this path may decline, and the Arrow fallback below needs
            // the same bytes. Both clones are refcount bumps.
            let reader = SegmentChunkReader::new(bytes.clone(), range.clone(), inputs.file_len)?;
            let column_metadata = inputs.metadata.row_group(inputs.row_group).column(leaf);
            let descr = parquet_schema.column(leaf);
            let decoded = pages::decode_chunk(
                reader,
                column_metadata,
                &descr,
                &dtype,
                row_count,
                inputs.options,
                Arc::clone(&inputs.metrics),
                dictionary_values,
            );
            if let Some(array) = decoded.map_err(Arc::new)? {
                inputs.metrics.record_page_decode();
                return Ok(array);
            }
        }

        inputs.metrics.record_arrow_decode();
        decode_arrow(&inputs, &dtype, row_count, bytes, range, None).map_err(Arc::new)
    }
    .boxed()
    .shared()
}

/// Decodes a flat chunk through its native page arrays and applies a row selection afterwards.
///
/// This is the full-chunk fallback when lazy header discovery declines malformed or unsupported
/// metadata. It still avoids routing a supported flat encoding through arrow-rs merely because an
/// optional index is absent.
#[expect(clippy::too_many_arguments)]
fn decode_full_page_selection(
    inputs: &ChunkDecode,
    dtype: &DType,
    row_count: u64,
    bytes: vortex::buffer::ByteBuffer,
    range: Range<u64>,
    row_range: &Range<usize>,
    mask: &Mask,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
) -> VortexResult<Option<ArrayRef>> {
    let parquet_schema = inputs.metadata.file_metadata().schema_descr();
    let Some(leaf) = pages::single_leaf_index(parquet_schema, inputs.root_column) else {
        return Ok(None);
    };
    let reader = SegmentChunkReader::new(bytes, range, inputs.file_len)?;
    let decoded = pages::decode_chunk(
        reader,
        inputs.metadata.row_group(inputs.row_group).column(leaf),
        &parquet_schema.column(leaf),
        dtype,
        row_count,
        inputs.options,
        Arc::clone(&inputs.metrics),
        dictionary_values,
    )?;
    let Some(mut array) = decoded else {
        return Ok(None);
    };
    if row_range.start != 0 || row_range.end != array.len() {
        array = array.slice(row_range.clone())?;
    }
    Ok(Some(pages::filter_page_chunks(array, mask.clone())?))
}

/// `Vec::remove(0)` reads badly at the call site; this says what it means.
trait RemoveFirst<T> {
    fn remove_first(self) -> T;
}

impl RemoveFirst<ArrayRef> for Vec<ArrayRef> {
    fn remove_first(mut self) -> ArrayRef {
        self.swap_remove(0)
    }
}

#[allow(deprecated)]
fn page_pruning_mask(
    indexes: &PagePruningIndexes,
    dtype: &DType,
    row_count: u64,
    row_range: &Range<u64>,
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<Mask>> {
    let offset_index = &indexes.offset;
    let column_index = &indexes.column;

    let page_locations = offset_index.page_locations();
    let page_count = page_locations.len();
    if page_count == 0
        || usize::try_from(column_index.num_pages()).ok() != Some(page_count)
        || column_index
            .null_counts()
            .is_some_and(|counts| counts.len() != page_count)
        || page_locations[0].first_row_index != 0
    {
        return Ok(None);
    }

    let len = usize::try_from(row_range.end - row_range.start)?;
    let mut pruned_ranges = Vec::new();
    for page in 0..page_count {
        let start = page_locations[page].first_row_index;
        let end = page_locations
            .get(page + 1)
            .map_or(i64::try_from(row_count)?, |location| {
                location.first_row_index
            });
        if start < 0 || end < start || u64::try_from(end)? > row_count {
            return Ok(None);
        }
        let start = u64::try_from(start)?;
        let end = u64::try_from(end)?;
        let page_rows = end - start;
        let null_count = match column_index.null_counts() {
            Some(counts) => match u64::try_from(counts[page]) {
                Ok(count) if count <= page_rows => Some(count),
                _ => return Ok(None),
            },
            None if column_index.is_null_page(page) => Some(page_rows),
            None => None,
        };
        let Some(statistics) = page_index_statistics(column_index, page, null_count) else {
            return Ok(None);
        };
        let prune = can_prune(
            expr,
            &ParquetStats {
                dtype,
                statistics: &statistics,
                row_count: page_rows,
                // Column indexes always use the type-defined order. Deprecated footer statistics
                // are handled separately below.
                min_max_safe: true,
            },
            session,
        )?;
        if !prune {
            continue;
        }

        let overlap_start = start.max(row_range.start);
        let overlap_end = end.min(row_range.end);
        if overlap_start < overlap_end {
            pruned_ranges.push((
                usize::try_from(overlap_start - row_range.start)?,
                usize::try_from(overlap_end - row_range.start)?,
            ));
        }
    }

    // Most unsorted row groups cannot prune any page. Avoid allocating and packing one boolean
    // per row in that case; otherwise copy whole page ranges into the bitmap in bulk.
    if pruned_ranges.is_empty() {
        return Ok(Some(Mask::new_true(len)));
    }
    let mut keep = Vec::with_capacity(pruned_ranges.len() + 1);
    let mut next = 0;
    for (start, end) in pruned_ranges {
        if next < start {
            keep.push((next, start));
        }
        next = end;
    }
    if next < len {
        keep.push((next, len));
    }
    Ok(Some(Mask::from_slices(len, keep)))
}

/// Requests and decodes one immutable page-index pair, sharing it across dynamic re-pruning.
///
/// This is called from inside the pruning future, after footer statistics have had first refusal.
/// Consequently a row-group footer miss issues no page-index I/O. Both segment futures are still
/// constructed together, before either is polled, preserving the I/O driver's coalescing window.
#[allow(deprecated)]
#[expect(clippy::too_many_arguments)]
fn page_pruning_indexes(
    cache: &OnceLock<SharedPagePruningIndexes>,
    segment_source: Arc<dyn SegmentSource>,
    offset_id: SegmentId,
    column_id: SegmentId,
    offset_range: Range<u64>,
    column_range: Range<u64>,
    column_metadata: ColumnChunkMetaData,
    file_len: u64,
    metrics: Arc<crate::metrics::ScanMetrics>,
) -> SharedPagePruningIndexes {
    cache
        .get_or_init(move || {
            let offset_bytes = segment_source.request(offset_id);
            let column_bytes = segment_source.request(column_id);
            async move {
                let offset_bytes = offset_bytes.await.map_err(Arc::new)?.unwrap_host();
                let column_bytes = column_bytes.await.map_err(Arc::new)?.unwrap_host();
                metrics.record_page_index_decode();

                let offset = {
                    let reader = SegmentChunkReader::new(offset_bytes, offset_range, file_len)
                        .map_err(Arc::new)?;
                    let mut indexes =
                        read_offset_indexes(&reader, std::slice::from_ref(&column_metadata))
                            .map_err(|error| {
                                Arc::new(vortex_err!("failed to decode offset index: {error}"))
                            })?
                            .unwrap_or_default();
                    if indexes.len() != 1 {
                        return Ok(None);
                    }
                    indexes.swap_remove(0)
                };
                let column = {
                    let reader = SegmentChunkReader::new(column_bytes, column_range, file_len)
                        .map_err(Arc::new)?;
                    let mut indexes =
                        read_columns_indexes(&reader, std::slice::from_ref(&column_metadata))
                            .map_err(|error| {
                                Arc::new(vortex_err!("failed to decode column index: {error}"))
                            })?
                            .unwrap_or_default();
                    if indexes.len() != 1 || matches!(indexes[0], ColumnIndexMetaData::NONE) {
                        return Ok(None);
                    }
                    indexes.swap_remove(0)
                };
                Ok(Some(Arc::new(PagePruningIndexes { offset, column })))
            }
            .boxed()
            .shared()
        })
        .clone()
}

/// The non-null literal tested by a root-level equality, if this expression has that shape.
///
/// Bloom filters only answer membership questions. Matching the scalar function's comparison
/// property here, rather than field names or SQL syntax, also covers commuted equalities produced by
/// expression normalization.
fn equality_literal(expr: &BoundExpression) -> Option<&Scalar> {
    if expr.as_opt::<Compare>().copied() != Some(CompareOperator::Eq) {
        return None;
    }
    let lhs = expr.child(0);
    let rhs = expr.child(1);
    if lhs.is_root() {
        rhs.as_opt::<Literal>().filter(|value| value.is_valid())
    } else if rhs.is_root() {
        lhs.as_opt::<Literal>().filter(|value| value.is_valid())
    } else {
        None
    }
}

/// Checks a logical Vortex scalar using Parquet's physical Bloom-filter representation.
fn bloom_might_contain(
    bloom: &Sbbf,
    value: &Scalar,
    physical_type: PhysicalType,
) -> VortexResult<Option<bool>> {
    let storage;
    let mut value = if value.dtype().is_extension() {
        storage = value.as_extension().to_storage_scalar();
        &storage
    } else {
        value
    };

    // Unsigned Parquet logical integers retain their bit pattern in signed INT32/INT64 physical
    // storage. A numerical cast would reject the upper half and disable useful pruning.
    let reinterpreted;
    match (value.dtype(), physical_type) {
        (DType::Primitive(PType::U32, _), PhysicalType::INT32) => {
            reinterpreted = value.primitive_reinterpret_cast(PType::I32)?;
            value = &reinterpreted;
        }
        (DType::Primitive(PType::U64, _), PhysicalType::INT64) => {
            reinterpreted = value.primitive_reinterpret_cast(PType::I64)?;
            value = &reinterpreted;
        }
        _ => {}
    }

    Ok(match physical_type {
        PhysicalType::BOOLEAN => bool::try_from(value).ok().map(|value| bloom.check(&value)),
        PhysicalType::INT32 => i32::try_from(value).ok().map(|value| bloom.check(&value)),
        PhysicalType::INT64 => i64::try_from(value).ok().map(|value| bloom.check(&value)),
        PhysicalType::FLOAT => f32::try_from(value).ok().map(|value| bloom.check(&value)),
        PhysicalType::DOUBLE => f64::try_from(value).ok().map(|value| bloom.check(&value)),
        PhysicalType::BYTE_ARRAY => match value.dtype() {
            DType::Utf8(_) => String::try_from(value)
                .ok()
                .map(|value| bloom.check(value.as_str())),
            DType::Binary(_) => vortex::buffer::ByteBuffer::try_from(value)
                .ok()
                .map(|value| bloom.check(value.as_slice())),
            _ => None,
        },
        PhysicalType::INT96 | PhysicalType::FIXED_LEN_BYTE_ARRAY => None,
    })
}

impl LayoutReader for ParquetPageSetReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.row_count)?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // ParquetField owns logical statistics interpretation. Keeping that responsibility at the
        // reconstruction boundary avoids applying one physical leaf's bounds to a multi-leaf field.
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let decode = self.cached_decode();
        let expr = expr.clone();
        let session = self.session.clone();
        let threshold = self.data.ctx.options.filter_eval_threshold();
        let full = decode.clone().get(None);

        Ok(MaskFuture::new(mask.len(), async move {
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(mask);
            }
            if full.peek().is_none()
                && !decode.logical_is_decimal()
                && !mask.all_true()
                && (mask.density() < threshold
                    || decode.skips_indexed_page(&row_range, &mask).await?)
            {
                let array = decode.decode_selected(&row_range, &mask).await?;
                return evaluate_selected_filter(array, &expr, mask, &session);
            }
            let array = SharedArray::clone(&full).await?;
            evaluate_filter(array, &expr, row_range, mask, &session, threshold)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let expr = expr.clone();
        let decode = self.cached_decode();
        let full = decode.clone().get(None);
        let handle = self.session.handle();

        Ok(async move {
            let mask = mask.await?;
            if mask.true_count() == 0 {
                return builder_with_capacity_in(
                    &decode.dtype,
                    0,
                    &decode.inputs.file.session().allocator(),
                )
                .finish()
                .apply_bound(&expr);
            }
            handle
                .spawn(async move {
                    if full.peek().is_none()
                        && (decode.worth_selecting(&mask)
                            || decode.skips_indexed_page(&row_range, &mask).await?)
                    {
                        let array = decode.decode_selected(&row_range, &mask).await?;
                        return array.apply_bound(&expr);
                    }

                    let mut array = SharedArray::clone(&full).await?;
                    if row_range.start > 0 || row_range.end < array.len() {
                        array = array.slice(row_range)?;
                    }
                    if !mask.all_true() {
                        array = pages::filter_page_chunks(array, mask)?;
                    }
                    array.apply_bound(&expr)
                })
                .await
        }
        .boxed())
    }
}

impl LayoutReader for ParquetFieldReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        if let Some(physical) = &self.physical {
            return physical.register_splits(field_mask, split_range, splits);
        }
        // A column chunk is read as a unit, so the only boundary this leaf contributes is its end.
        // Row-group boundaries come from the chunked layout above it, which is why they are absent
        // here.
        split_range.check_bounds(self.row_count())?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let data = self.layout.data();
        let schema = data.ctx.metadata.file_metadata().schema_descr();
        let Some(leaf) = pages::single_leaf_index(schema, data.root_column) else {
            return Ok(MaskFuture::ready(mask));
        };
        let physical_type = schema.column(leaf).physical_type();
        if !pages::direct_dtype(self.layout.dtype(), physical_type) {
            return Ok(MaskFuture::ready(mask));
        }
        let footer_statistics = data
            .ctx
            .metadata
            .row_group(data.row_group)
            .column(leaf)
            .statistics()
            .cloned()
            .map(|statistics| {
                let min_max_safe = crate::statistics::min_max_trustworthy(
                    &statistics,
                    data.ctx
                        .metadata
                        .file_metadata()
                        .schema_descr()
                        .column(leaf)
                        .as_ref(),
                );
                (statistics, min_max_safe)
            });
        let dynamic = DynamicExprUpdates::new(expr).is_some();
        if !dynamic && let Some((statistics, min_max_safe)) = &footer_statistics {
            let stats = ParquetStats {
                dtype: self.layout.dtype(),
                statistics,
                row_count: self.layout.row_count(),
                min_max_safe: *min_max_safe,
            };
            if can_prune(expr, &stats, &self.session)? {
                return Ok(MaskFuture::ready(Mask::new_false(mask.len())));
            }
        }

        let bloom = equality_literal(expr).and_then(|value| {
            data.bloom_filters
                .iter()
                .find(|filter| filter.leaf == leaf)
                .map(|filter| (value.clone(), self.segment_source.request(filter.id)))
        });
        let page_index_inputs = data
            .offset_indexes
            .iter()
            .find(|index| index.leaf == leaf)
            .zip(data.column_indexes.iter().find(|index| index.leaf == leaf))
            .map(|(offset, column)| {
                (
                    offset.id,
                    column.id,
                    offset.range.clone(),
                    column.range.clone(),
                )
            });
        if bloom.is_none() && page_index_inputs.is_none() && !dynamic {
            return Ok(MaskFuture::ready(mask));
        }
        let dtype = self.layout.dtype().clone();
        let row_count = self.layout.row_count();
        let row_range = row_range.clone();
        let expr = expr.clone();
        let session = self.session.clone();
        let metrics = Arc::clone(&data.ctx.metrics);
        let page_pruning_cache = Arc::clone(&self.page_pruning_indexes);
        let segment_source = Arc::clone(&self.segment_source);
        let column_metadata = data
            .ctx
            .metadata
            .row_group(data.row_group)
            .column(leaf)
            .clone();
        let file_len = data.ctx.file_len;

        Ok(MaskFuture::new(mask.len(), async move {
            // A dynamic RHS is read here, not when `pruning_evaluation` constructs this future.
            // Besides handling an update between construction and polling, this gives footer
            // statistics first refusal before any page-index request is registered.
            if dynamic
                && let Some((statistics, min_max_safe)) = &footer_statistics
                && can_prune(
                    &expr,
                    &ParquetStats {
                        dtype: &dtype,
                        statistics,
                        row_count,
                        min_max_safe: *min_max_safe,
                    },
                    &session,
                )?
            {
                return Ok(Mask::new_false(mask.len()));
            }

            // Poll the tiny Bloom filter first; a definite miss makes both the data pages and
            // page-index I/O unnecessary.
            if let Some((value, bloom_bytes)) = bloom {
                let bloom_bytes = bloom_bytes.await?.unwrap_host();
                metrics.record_bloom_filter_decode();
                if let Ok(filter) = Sbbf::from_bytes(&bloom_bytes)
                    && bloom_might_contain(&filter, &value, physical_type)? == Some(false)
                {
                    return Ok(Mask::new_false(mask.len()));
                }
            }

            if let Some((offset_id, column_id, offset_range, column_range)) = page_index_inputs {
                let indexes = page_pruning_indexes(
                    &page_pruning_cache,
                    segment_source,
                    offset_id,
                    column_id,
                    offset_range,
                    column_range,
                    column_metadata,
                    file_len,
                    metrics,
                )
                .await?;
                if let Some(indexes) = indexes
                    && let Some(page_mask) =
                        page_pruning_mask(&indexes, &dtype, row_count, &row_range, &expr, &session)?
                {
                    return Ok(mask.bitand(&page_mask));
                }
            }
            Ok(mask)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if let Some(physical) = &self.physical {
            return physical.filter_evaluation(row_range, expr, mask);
        }
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let decode = self.cached_decode();
        let expr = expr.clone();
        let session = self.session.clone();
        let threshold = self.layout.data().ctx.options.filter_eval_threshold();
        let full = decode.clone().get(None);

        Ok(MaskFuture::new(mask.len(), async move {
            // The request above is registered early for prefetch, but CPU decoding waits for prior
            // conjuncts and any revised dynamic pruning mask. A newly empty mask does no decode.
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(mask);
            }
            if full.peek().is_none()
                && !decode.logical_is_decimal()
                && !mask.all_true()
                && (mask.density() < threshold
                    || decode.skips_indexed_page(&row_range, &mask).await?)
            {
                let array = decode.decode_selected(&row_range, &mask).await?;
                return evaluate_selected_filter(array, &expr, mask, &session);
            }
            let array = SharedArray::clone(&full).await?;

            // Post-condition: intersect with the input mask. Sparse masks use rank intersection
            // because the predicate is evaluated only for surviving rows.
            evaluate_filter(array, &expr, row_range, mask, &session, threshold)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        if let Some(physical) = &self.physical {
            return physical.projection_evaluation(row_range, expr, mask);
        }
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        let expr = expr.clone();

        let decode = self.cached_decode();
        // Capturing the shared future here is safe: its SegmentFuture is constructed only when
        // polled, after the mask.  A filter over this same field captures the identical handle, so a
        // full decode (including outer decompression) is shared without making empty masks eager.
        let full = decode.clone().get(None);
        let handle = self.session.handle();

        Ok(async move {
            let mask = mask.await?;

            // A fully-pruned projection must not fetch or decode its column. Applying the expression
            // to an empty array preserves expression semantics (including casts and aliases) while
            // keeping the footer-only/open and pruning contracts intact.
            if mask.true_count() == 0 {
                return builder_with_capacity_in(
                    &decode.dtype,
                    0,
                    &decode.inputs.file.session().allocator(),
                )
                .finish()
                .apply_bound(&expr);
            }

            handle
                .spawn(async move {
                    // The selective path, when this file asked for it and when the mask is sparse
                    // enough to be worth it. Spawning at the field boundary lets sibling projected
                    // columns resolve I/O and decode concurrently while preserving the row group as
                    // the only row-splitting unit.
                    if full.peek().is_none()
                        && (decode.worth_selecting(&mask)
                            || decode.skips_indexed_page(&row_range, &mask).await?)
                    {
                        let array = decode.decode_selected(&row_range, &mask).await?;
                        return array.apply_bound(&expr);
                    }

                    let mut array = SharedArray::clone(&full).await?;

                    // Post-condition: filter before projecting. The projection may only be valid on
                    // the rows that survive the filter, as in `CAST(a, u8) WHERE a < 256`.
                    if !mask.all_true() {
                        let mask = chunk_mask(&row_range, &mask, array.len())?;
                        array = pages::filter_page_chunks(array, mask)?;
                    } else if row_range.start > 0 || row_range.end < array.len() {
                        array = array.slice(row_range)?;
                    }

                    array.apply_bound(&expr)
                })
                .await
        }
        .boxed())
    }
}

/// Decodes a column chunk through arrow-rs.
///
/// This is the bridge: every value is materialized into an Arrow array and then converted to Vortex.
/// It is why the reader is correct, and it is the cost the next stage is meant to remove.
fn decode_arrow(
    inputs: &ChunkDecode,
    dtype: &DType,
    row_count: u64,
    bytes: vortex::buffer::ByteBuffer,
    range: Range<u64>,
    selection: Option<RowSelection>,
) -> VortexResult<ArrayRef> {
    let chunk_reader = SegmentChunkReader::new(bytes, range, inputs.file_len)?;
    decode_arrow_reader(inputs, dtype, row_count, chunk_reader, selection, None)
}

/// A copy of the file metadata carrying `indexes` as its offset index.
///
/// # Why this is built and thrown away rather than kept
///
/// arrow-rs addresses pages through the offset index attached to its `ParquetMetaData`, so the
/// nested/Dremel fallback needs a metadata object that carries one. The obvious implementation
/// would retain a copy beside each field reader's decoded indexes. However,
/// `ParquetMetaDataBuilder::new_from_metadata` clones the entire footer to attach one row group's
/// offset index. Retaining those copies would multiply footer memory by the number of field readers.
///
/// That directly contradicted the invariant [`crate::layout::FileContext`] states: the footer is held
/// behind one `Arc` precisely because it "is not something to clone per column per row group".
///
/// So it is built where it is needed and dropped immediately. The flat page decoder — every ClickBench
/// column — consumes `IndexedDecode::indexes` directly and never calls this at all, so the common path
/// allocates nothing. The nested path pays one transient clone per decode, which is bounded by
/// in-flight decodes rather than by their total count.
fn augment_metadata(
    inputs: &ChunkDecode,
    indexes: Vec<Vec<OffsetIndexMetaData>>,
) -> ParquetMetaData {
    ParquetMetaDataBuilder::new_from_metadata(inputs.metadata.as_ref().clone())
        .set_offset_index(Some(indexes))
        .build()
}

/// An empty offset-index matrix shaped like `metadata`, for one row group to be filled in.
fn empty_offset_indexes(metadata: &ParquetMetaData) -> Vec<Vec<OffsetIndexMetaData>> {
    metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            (0..row_group.num_columns())
                .map(|_| OffsetIndexBuilder::new().build())
                .collect()
        })
        .collect()
}

/// Arrow reader metadata carrying `indexes`, for the nested fallback, built transiently.
///
/// See [`augment_metadata`] for why this is not cached.
fn indexed_arrow_metadata(indexed: &IndexedDecode) -> VortexResult<ArrowReaderMetadata> {
    let inputs = &indexed.inputs;
    let mut matrix = empty_offset_indexes(&inputs.metadata);
    let slot = matrix.get_mut(inputs.row_group).ok_or_else(|| {
        vortex_err!(
            "row group {} is out of range while attaching an offset index",
            inputs.row_group
        )
    })?;
    slot.clone_from(&indexed.row_group_indexes);
    ArrowReaderMetadata::try_new(
        Arc::new(augment_metadata(inputs, matrix)),
        ArrowReaderOptions::default().with_schema(Arc::clone(inputs.arrow_metadata.schema())),
    )
    .map_err(|error| vortex_err!("failed to attach lazy offset index: {error}"))
}

/// As [`decode_arrow`], over a reader that may hold several disjoint spans.
///
/// Separate because a page-granular fetch holds one buffer per page rather than one per chunk, and
/// arrow-rs does not care which — it addresses bytes by absolute file offset either way.
fn decode_arrow_reader(
    inputs: &ChunkDecode,
    dtype: &DType,
    row_count: u64,
    chunk_reader: SegmentChunkReader,
    selection: Option<RowSelection>,
    // `Some` only when reading page ranges, where arrow-rs must resolve pages through an offset
    // index. Passed in rather than read off `inputs` because that metadata is now built transiently
    // for exactly this call and dropped afterwards -- see [`augment_metadata`].
    indexed_metadata: Option<ArrowReaderMetadata>,
) -> VortexResult<ArrayRef> {
    let selection_rows = selection.as_ref().map(RowSelection::row_count);
    let parquet_schema = inputs.metadata.file_metadata().schema_descr();
    let projection = ProjectionMask::roots(parquet_schema, [inputs.root_column]);

    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
        chunk_reader,
        indexed_metadata.unwrap_or_else(|| inputs.arrow_metadata.clone()),
    )
    .with_row_groups(vec![inputs.row_group])
    .with_projection(projection);
    let builder = match selection {
        Some(selection) => builder.with_row_selection(selection),
        None => builder,
    };
    let builder = builder
        // One batch for the whole chunk where possible: it is already in memory, so batching buys nothing
        // and only forces a concatenation afterwards.
        //
        // Bounded because `row_count` is a footer claim that sizes arrow-rs's own allocations. A chunk
        // genuinely larger than the bound is emitted as several batches and concatenated, which is the
        // pre-existing multi-batch path rather than a new one.
        .with_batch_size(
            usize::try_from(row_count.max(1))
                .unwrap_or(usize::MAX)
                .min(MAX_BATCH_ROWS),
        );
    let batch_reader = builder
        .build()
        .map_err(|e| vortex_err!("failed to build a Parquet record reader: {e}"))?;

    // Whether `nullable` is set follows the Arrow field rather than the physical nulls, so an
    // all-present chunk of a nullable column keeps the column's declared type. Getting this backwards
    // makes the leaf's dtype disagree with the struct layout above it.
    let nullable = dtype.is_nullable();

    let mut chunks: Vec<ArrayRef> = Vec::new();
    for batch in batch_reader {
        let batch = batch.map_err(|e| {
            vortex_err!(
                "failed to decode row group {} column {}: {e}",
                inputs.row_group,
                inputs.root_column
            )
        })?;
        if batch.num_columns() != 1 {
            return Err(vortex_err!(
                "expected a single projected column for row group {} column {}, got {}",
                inputs.row_group,
                inputs.root_column,
                batch.num_columns()
            ));
        }
        chunks.push(inputs.file.session().arrow().from_arrow_array(
            crate::arrow_bridge::import_array(batch.column(0).as_ref())?,
            nullable,
        )?);
    }

    let array = match chunks.len() {
        // An empty row group still needs an array of the right type — and the *declared* type, not its
        // nullable form. Getting it wrong makes a non-nullable column decode as nullable and the leaf's
        // dtype disagree with the struct layout above it.
        0 => builder_with_capacity_in(dtype, 0, &inputs.file.session().allocator()).finish(),
        1 => chunks.remove_first(),
        _ => ChunkedArray::try_new(chunks, dtype.clone())?.into_array(),
    };

    // The row count this decode was asked for is what the layout above has already told Vortex to
    // expect. A short or empty result is not a smaller answer, it is a *wrong* one: the struct layout
    // zips this column against its siblings by position, so a column that comes back short silently
    // shifts every field beside it. The selective paths check this; the full decode did not, which
    // left the invariant enforced only where the risk was newest rather than where it was greatest.
    //
    // Only checked when a selection was not supplied: with one, the expected length is the selection's
    // and the caller — which knows the mask — checks it there.
    if selection_rows.is_none() && u64::try_from(array.len())? != row_count {
        vortex_bail!(
            "decoding row group {} column {} produced {} rows, but the footer declares {row_count}",
            inputs.row_group,
            inputs.root_column,
            array.len()
        );
    }
    Ok(array)
}
