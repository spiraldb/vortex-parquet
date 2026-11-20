// SPDX-License-Identifier: Apache-2.0

//! Physical plans for Vortex's planner-driven scan.
//!
//! A Parquet file lowers to the plan tree its layout already describes:
//!
//! ```text
//! Concat                       row groups (from the file's ChunkedLayout)
//!   └─ Pack                    top-level fields (from each row group's StructLayout)
//!      └─ LegacyStats          footer min/max/null-count, when the field has trustworthy ones
//!         ├─ ParquetFieldScan  reads and decodes the field's column-chunk segment
//!         └─ FooterZones       a one-zone statistics table, built from the footer without I/O
//! ```
//!
//! The file and row-group layouts lower to their children's plans: they carry no rows of their
//! own. Each field's leaf names one segment, the byte range of its column chunks, so the
//! session's [`IoScheduler`](vortex::io::request::IoScheduler) coalesces, deduplicates and
//! retains those reads across every scan of the file.
//!
//! # What is not yet expressed
//!
//! The leaf reads whole column chunks. Page-granular reads need either segments discovered after
//! open (page locations live in the lazily read offset index) or a scan operator that can ask for
//! byte ranges within a segment; Vortex's scan context exposes neither yet. Page-index pruning,
//! Bloom-filter pruning and selective decoding therefore do not run on this path; see
//! spiraldb/vortex-parquet#6.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Stat;
use vortex::io::segments::SegmentId;
use vortex::layout::Layout;
use vortex::layout::lowering::LegacyStats;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scan::execution::operator::Cx;
use vortex::scan::execution::operator::Input;
use vortex::scan::execution::operator::Operator;
use vortex::scan::execution::operator::Source;
use vortex::scan::execution::operator::SourceBlocked;
use vortex::scan::execution::operator::SourceReady;
use vortex::scan::execution::operator::Step;
use vortex::scan::execution::pipeline::Chain;
use vortex::scan::execution::pipeline::Compiler;
use vortex::scan::plan::Plan;
use vortex::scan::plan::PlanId;
use vortex::scan::plan::PlanParts;
use vortex::scan::plan::PlanRef;
use vortex::scan::plan::PlanSlots;
use vortex::scan::plan::PlanVTable;
use vortex::session::registry::CachedId;

use crate::cache::DecodeUnit;
use crate::layout::ChunkDecode;
use crate::layout::ParquetField;
use crate::pages;
use crate::reader::decode_field_bytes;
use crate::statistics::max_scalar;
use crate::statistics::min_max_trustworthy;
use crate::statistics::min_scalar;

/// Lowers a field layout to its leaf scan, under its footer statistics when it has any.
pub(crate) fn field_plan(layout: &Layout<ParquetField>) -> VortexResult<PlanRef> {
    let data = layout.data();
    let segment = layout
        .segment_ids()
        .first()
        .copied()
        .ok_or_else(|| vortex_err!("vortex.parquet.field has no column-chunk segment"))?;
    let scan = ParquetFieldScan::new_plan(
        layout.dtype().clone(),
        layout.row_count(),
        segment,
        data.decode_inputs(),
        data.ctx.decode_cache.get_or_insert(
            (
                data.ctx.file_id,
                data.row_group,
                DecodeUnit::Column(data.root_column),
            ),
            data.byte_length,
            DecodedChunk::default,
        ),
    )
    .into_plan();

    let Some((zones, stats)) =
        footer_zones(data.decode_inputs(), layout.dtype(), layout.row_count())?
    else {
        return Ok(scan);
    };
    Ok(LegacyStats::new_plan(scan, zones, layout.row_count(), stats).into_plan())
}

/// Reads one field of one row group: the bytes of its column chunks, decoded once per scan.
#[derive(Clone, Debug)]
pub struct ParquetFieldScan;

/// What a field scan reads and how it decodes it.
#[derive(Clone)]
pub struct ParquetFieldScanData {
    decode: ChunkDecode,
    segment: SegmentId,
    chunk: Arc<DecodedChunk>,
}

impl fmt::Debug for ParquetFieldScanData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetFieldScanData")
            .field("row_group", &self.decode.row_group)
            .field("root_column", &self.decode.root_column)
            .field("segment", &self.segment)
            .finish_non_exhaustive()
    }
}

/// A field's decode, shared by every plan over the same field of the same row group.
///
/// The scan executor compiles a plan once per chunk of at most a few blocks of 8,192 rows, while a
/// Parquet row group is usually far larger, and a scan lowers a field once for each filter
/// conjunct and once more for the projection. Without sharing, each of those would decode the
/// whole column chunk again. The decoded array is lazy page arrays, so holding it holds compressed
/// pages and the dictionary, not decoded values.
///
/// Held in the file's [`DecodeCache`](crate::cache::DecodeCache), so its
/// [`CachePolicy`](crate::CachePolicy) decides how long it lives: by default, as long as some
/// plan over the field does.
#[derive(Default)]
pub struct DecodedChunk {
    decoded: Mutex<Option<ArrayRef>>,
    dictionary: Arc<OnceLock<ArrayRef>>,
}

impl fmt::Debug for DecodedChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodedChunk")
            .field("decoded", &self.decoded.lock().is_some())
            .finish()
    }
}

/// A plan reading one Parquet field of one row group.
pub type ParquetFieldScanPlan = Plan<ParquetFieldScan>;

impl ParquetFieldScan {
    /// Creates a scan of the field `decode` describes, whose bytes are `segment`, sharing `chunk`'s
    /// decode with other plans over the same field.
    pub(crate) fn new_plan(
        dtype: DType,
        row_count: u64,
        segment: SegmentId,
        decode: ChunkDecode,
        chunk: Arc<DecodedChunk>,
    ) -> ParquetFieldScanPlan {
        PlanParts {
            vtable: ParquetFieldScan,
            dtype,
            row_count,
            slots: PlanSlots::default(),
            data: ParquetFieldScanData {
                decode,
                segment,
                chunk,
            },
        }
        .into_typed()
    }
}

impl PlanVTable for ParquetFieldScan {
    type PlanData = ParquetFieldScanData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.parquet.plan.field_scan");
        *ID
    }

    fn fmt(plan: &Plan<Self>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = plan.data();
        write!(
            formatter,
            " row_group={} column={} segment={}",
            data.decode.row_group, data.decode.root_column, data.segment
        )
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // A field scan holds the parsed footer; it is rebuilt from the file, not serialized.
        None
    }

    fn with_slots(
        _plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("ParquetFieldScan", 0)
    }

    fn is_data_source(_plan: &Plan<Self>) -> bool {
        true
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        compiler.register_segment(plan.data().segment);
        Ok(Some(Chain::new(FieldSource {
            plan: plan.clone(),
            rows,
            mask: mask.clone(),
            decoded: None,
            requested: false,
            done: false,
        })))
    }
}

/// Reads a field's segment, or reuses the row group's decode, and emits the masked rows.
struct FieldSource {
    plan: ParquetFieldScanPlan,
    rows: Range<u64>,
    mask: Mask,
    decoded: Option<ArrayRef>,
    requested: bool,
    done: bool,
}

impl FieldSource {
    /// Decodes `bytes`, unless a concurrent chunk of the same row group already has.
    fn decode(&self, bytes: vortex::buffer::ByteBuffer) -> VortexResult<ArrayRef> {
        let data = self.plan.data();
        // Held across the decode, so concurrent chunks of one row group wait for one decode rather
        // than each running their own.
        let mut decoded = data.chunk.decoded.lock();
        if let Some(array) = &*decoded {
            return Ok(array.clone());
        }
        let array = decode_field_bytes(
            &data.decode,
            self.plan.dtype(),
            self.plan.row_count(),
            bytes,
            Arc::clone(&data.chunk.dictionary),
        )?;
        *decoded = Some(array.clone());
        Ok(array)
    }
}

impl Source for FieldSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.done || self.decoded.is_some() {
            return Ok(SourceReady::Ready);
        }
        if !self.requested {
            if let Some(array) = self.plan.data().chunk.decoded.lock().clone() {
                self.decoded = Some(array);
                return Ok(SourceReady::Ready);
            }
            cx.request_read(self.plan.data().segment);
            self.requested = true;
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if cx.has_bytes() {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Io)
        })
    }
}

impl Operator for FieldSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        let array = match self.decoded.take() {
            Some(array) => array,
            None => {
                let (_, bytes) = cx
                    .take_segment()
                    .ok_or_else(|| vortex_err!("Parquet field scan ran without its bytes"))?;
                self.decode(bytes.into_host_sync())?
            }
        };
        self.done = true;

        let rows = usize::try_from(self.rows.start)?..usize::try_from(self.rows.end)?;
        let array = if rows.start == 0 && rows.end == array.len() {
            array
        } else {
            array.slice(rows)?
        };
        let array = if self.mask.all_true() {
            array
        } else {
            pages::filter_page_chunks(array, self.mask.clone())?
        };
        Ok(Step::Final(array))
    }
}

/// The stats a footer zone table holds, in field order.
const FOOTER_STATS: [Stat; 3] = [Stat::Max, Stat::Min, Stat::NullCount];

/// A one-zone statistics table for a flat field, built from its row group's footer statistics.
///
/// `None` when the footer has nothing to prune with: a nested field, a type the page decoder does
/// not produce directly, or no statistics. Min and max are left out when their ordering is not
/// trustworthy; the null count is kept.
fn footer_zones(
    decode: ChunkDecode,
    dtype: &DType,
    row_count: u64,
) -> VortexResult<Option<(PlanRef, Arc<[Stat]>)>> {
    if row_count == 0 {
        return Ok(None);
    }
    let schema = decode.metadata.file_metadata().schema_descr();
    let Some(leaf) = pages::single_leaf_index(schema, decode.root_column) else {
        return Ok(None);
    };
    let descriptor = schema.column(leaf);
    if !pages::direct_dtype(dtype, descriptor.physical_type()) {
        return Ok(None);
    }
    let Some(statistics) = decode
        .metadata
        .row_group(decode.row_group)
        .column(leaf)
        .statistics()
    else {
        return Ok(None);
    };
    let trusted = min_max_trustworthy(statistics, descriptor.as_ref());

    let mut names: Vec<FieldName> = Vec::with_capacity(FOOTER_STATS.len());
    let mut fields: Vec<ArrayRef> = Vec::with_capacity(FOOTER_STATS.len());
    let mut stats: Vec<Stat> = Vec::with_capacity(FOOTER_STATS.len());
    for stat in FOOTER_STATS {
        let value = match stat {
            Stat::Max if trusted => max_scalar(statistics, dtype),
            Stat::Min if trusted => min_scalar(statistics, dtype),
            Stat::NullCount => statistics
                .null_count_opt()
                .map(|count| Scalar::primitive(count, Nullability::Nullable)),
            _ => None,
        };
        // Zone tables hold nullable states: a zone may lack a statistic another zone has.
        let value = value
            .map(|value| {
                let nullable = value.dtype().as_nullable();
                value.cast(&nullable)
            })
            .transpose()?;
        let Some(value) = value else { continue };
        names.push(stat.name().into());
        fields.push(ConstantArray::new(value, 1).into_array());
        stats.push(stat);
    }
    if stats.is_empty() {
        return Ok(None);
    }
    let table = StructArray::try_new(names.into(), fields, 1, Validity::NonNullable)?.into_array();
    Ok(Some((
        FooterZones::new_plan(table).into_plan(),
        stats.into(),
    )))
}

/// Emits a statistics table held in memory: zones read from a Parquet footer, not from the file.
#[derive(Clone, Debug)]
pub struct FooterZones;

/// A plan emitting an in-memory zone table.
pub type FooterZonesPlan = Plan<FooterZones>;

impl FooterZones {
    /// Creates a plan emitting `table`, one row per zone.
    pub fn new_plan(table: ArrayRef) -> FooterZonesPlan {
        PlanParts {
            vtable: FooterZones,
            dtype: table.dtype().clone(),
            row_count: table.len() as u64,
            slots: PlanSlots::default(),
            data: table,
        }
        .into_typed()
    }
}

impl PlanVTable for FooterZones {
    type PlanData = ArrayRef;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.parquet.plan.footer_zones");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn with_slots(
        _plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("FooterZones", 0)
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        _compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        let table = plan
            .data()
            .slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?;
        let table = if mask.all_true() {
            table
        } else {
            table.filter(mask.clone())?
        };
        Ok(Some(Chain::new(OnceSource(Some(table)))))
    }
}

/// Emits one batch it was built with.
struct OnceSource(Option<ArrayRef>);

impl Source for OnceSource {}

impl Operator for OnceSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match self.0.take() {
            Some(batch) => Step::Final(batch),
            None => Step::End,
        })
    }
}
