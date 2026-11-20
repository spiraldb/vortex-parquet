// SPDX-License-Identifier: Apache-2.0

//! Decodes one field of one row group from the bytes of its column chunks.
//!
//! The scan plan's leaf ([`crate::plan`]) hands this the bytes of a field's column-chunk segment.
//! Flat fields with a supported encoding decode through the page path ([`crate::pages`]) into lazy
//! page arrays; anything it declines decodes through arrow-rs and is converted to Vortex.

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::builders::builder_with_capacity_in;
use vortex::array::memory::MemorySessionExt;
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::chunk_reader::SegmentChunkReader;
use crate::layout::ChunkDecode;
use crate::pages;

/// The largest batch arrow-rs is asked for, whatever the footer claims the chunk holds.
///
/// A footer is data. Handing an unchecked row count to `with_batch_size` lets a hostile file size
/// arrow-rs's buffers, so the request is bounded and a genuinely larger chunk simply arrives as
/// several batches.
pub(crate) const MAX_BATCH_ROWS: usize = 1 << 20;

/// Decodes one field of one row group from the bytes of its whole column-chunk range.
///
/// The page path runs when this chunk is one it handles. It returns `None` for anything it
/// declines — a nested column, a non-dictionary encoding — and the Arrow path below is the
/// fallback. An `Err` is different: that is our decoder being wrong, and it propagates rather than
/// silently falling back, because a silent fallback would turn a bug into a performance mystery.
pub(crate) fn decode_field_bytes(
    inputs: &ChunkDecode,
    dtype: &DType,
    row_count: u64,
    bytes: ByteBuffer,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let range = inputs.byte_offset..inputs.byte_offset + inputs.byte_length;
    let parquet_schema = inputs.metadata.file_metadata().schema_descr();
    if let Some(leaf) = pages::single_leaf_index(parquet_schema, inputs.root_column) {
        // Cloned rather than moved: this path may decline, and the Arrow fallback below needs the
        // same bytes. Both clones are refcount bumps.
        let reader = SegmentChunkReader::new(bytes.clone(), range.clone(), inputs.file_len)?;
        let column_metadata = inputs.metadata.row_group(inputs.row_group).column(leaf);
        let descr = parquet_schema.column(leaf);
        let decoded = pages::decode_chunk(
            reader,
            column_metadata,
            &descr,
            dtype,
            row_count,
            inputs.options,
            Arc::clone(&inputs.metrics),
            dictionary_values,
        )?;
        if let Some(array) = decoded {
            inputs.metrics.record_page_decode();
            return Ok(array);
        }
    }

    inputs.metrics.record_arrow_decode();
    decode_arrow(inputs, dtype, row_count, bytes, range)
}

/// Decodes a column chunk through arrow-rs.
///
/// This is the bridge: every value is materialized into an Arrow array and then converted to
/// Vortex. It handles everything the page path declines, nested fields above all.
fn decode_arrow(
    inputs: &ChunkDecode,
    dtype: &DType,
    row_count: u64,
    bytes: ByteBuffer,
    range: Range<u64>,
) -> VortexResult<ArrayRef> {
    let chunk_reader = SegmentChunkReader::new(bytes, range, inputs.file_len)?;
    let parquet_schema = inputs.metadata.file_metadata().schema_descr();
    let projection = ProjectionMask::roots(parquet_schema, [inputs.root_column]);

    let batch_reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
        chunk_reader,
        inputs.arrow_metadata.clone(),
    )
    .with_row_groups(vec![inputs.row_group])
    .with_projection(projection)
    // One batch for the whole chunk where possible: it is already in memory, so batching
    // buys nothing and only forces a concatenation afterwards. Bounded because `row_count`
    // is a footer claim that sizes arrow-rs's own allocations.
    .with_batch_size(
        usize::try_from(row_count.max(1))
            .unwrap_or(usize::MAX)
            .min(MAX_BATCH_ROWS),
    )
    .build()
    .map_err(|e| vortex_err!("failed to build a Parquet record reader: {e}"))?;

    // Whether `nullable` is set follows the Arrow field rather than the physical nulls, so an
    // all-present chunk of a nullable column keeps the column's declared type. Getting this
    // backwards makes the leaf's dtype disagree with the struct layout above it.
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
            vortex_bail!(
                "expected a single projected column for row group {} column {}, got {}",
                inputs.row_group,
                inputs.root_column,
                batch.num_columns()
            );
        }
        chunks.push(inputs.file.session().arrow().from_arrow_array(
            crate::arrow_bridge::import_array(batch.column(0).as_ref())?,
            nullable,
        )?);
    }

    let array = match chunks.len() {
        // An empty row group still needs an array of the declared type, not its nullable form.
        0 => builder_with_capacity_in(dtype, 0, &inputs.file.session().allocator()).finish(),
        1 => chunks.swap_remove(0),
        _ => ChunkedArray::try_new(chunks, dtype.clone())?.into_array(),
    };

    // The struct above zips this column against its siblings by position, so a column that comes
    // back short would silently shift every field beside it.
    if u64::try_from(array.len())? != row_count {
        vortex_bail!(
            "decoding row group {} column {} produced {} rows, but the footer declares {row_count}",
            inputs.row_group,
            inputs.root_column,
            array.len()
        );
    }
    Ok(array)
}
