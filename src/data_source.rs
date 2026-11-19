// SPDX-License-Identifier: Apache-2.0

//! Row-group-aligned Scan API adapter with Parquet footer statistics.

use std::cmp::Ordering;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::extension::datetime::AnyTemporal;
use vortex::array::stats::StatsSet;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::extension::Matcher;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder as V2ScanBuilder;
use vortex::layout::scan::v2::ScanFile;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scan::DataSource;
use vortex::scan::DataSourceScan;
use vortex::scan::DataSourceScanRef;
use vortex::scan::Partition;
use vortex::scan::PartitionRef;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::scan::execution::driver::ScanDriver;
use vortex::scan::selection::Selection;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::parallelism::get_available_parallelism;

use crate::PreparedFile;
use crate::pages::single_leaf_index;
use crate::statistics::max_scalar;
use crate::statistics::min_scalar;

/// A layout-backed Parquet source with footer-derived file statistics.
pub(crate) struct ParquetDataSource {
    prepared: PreparedFile,
    session: VortexSession,
    dtype: DType,
    row_groups: Arc<[Range<u64>]>,
    /// The file's layout, segment table and placed reader, built by the first scan.
    ///
    /// Built once and shared by every scan, because the segment ids in the layout must index the
    /// same table every time: the session's I/O scheduler keeps one registration per placed
    /// reader, with the locations it was first registered with. Its plan cache is `None`, so plans
    /// are lowered per partition and a row group's decodes are released when its partition ends.
    file: Mutex<Option<ScanFile>>,
}

impl ParquetDataSource {
    pub(crate) fn new(prepared: PreparedFile) -> VortexResult<Self> {
        let metadata = &prepared.ctx.metadata;
        let mut start = 0u64;
        let mut row_groups = Vec::with_capacity(metadata.row_groups().len());
        for (index, row_group) in metadata.row_groups().iter().enumerate() {
            let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
                vortex::error::vortex_err!(
                    "Parquet row group {index} has negative row count {}",
                    row_group.num_rows()
                )
            })?;
            let end = start.checked_add(rows).ok_or_else(|| {
                vortex::error::vortex_err!("Parquet row-group row ranges overflow u64")
            })?;
            row_groups.push(start..end);
            start = end;
        }
        Ok(Self {
            session: prepared.ctx.file.session().clone(),
            dtype: prepared.dtype.clone(),
            prepared,
            row_groups: row_groups.into(),
            file: Mutex::new(None),
        })
    }

    /// The file's scan inputs, built on first use. Building them visits every column chunk in
    /// the footer, which binding the source does not need to pay for.
    fn scan_file(&self) -> VortexResult<ScanFile> {
        let mut file = self.file.lock();
        if let Some(scan_file) = &*file {
            return Ok(scan_file.clone());
        }
        let scan_file = crate::scan_file(&self.prepared)?;
        *file = Some(scan_file.clone());
        Ok(scan_file)
    }

    /// Drop only static comparisons proven true for every row, including null rows. Doing this
    /// before choosing columns avoids opening and decoding columns used only by redundant filters.
    fn unsatisfied_filter(
        &self,
        expression: Expression,
        statistics: &mut HashMap<FieldName, StatsSet>,
    ) -> VortexResult<Option<Expression>> {
        if expression.as_opt::<Binary>() == Some(&Operator::And) {
            let children = expression
                .children()
                .iter()
                .cloned()
                .map(|child| self.unsatisfied_filter(child, statistics))
                .collect::<VortexResult<Vec<_>>>()?;
            return Ok(and_collect(children.into_iter().flatten()));
        }
        if self.comparison_satisfied(&expression, statistics)? {
            Ok(None)
        } else {
            Ok(Some(expression))
        }
    }

    fn comparison_satisfied(
        &self,
        expression: &Expression,
        statistics: &mut HashMap<FieldName, StatsSet>,
    ) -> VortexResult<bool> {
        let Some(operator) = expression
            .as_opt::<Binary>()
            .and_then(|op| op.to_compare_operator())
        else {
            return Ok(false);
        };
        let (column, value, operator) = if let Some(value) = expression.child(1).as_opt::<Literal>()
        {
            (expression.child(0), value, operator)
        } else if let Some(value) = expression.child(0).as_opt::<Literal>() {
            (expression.child(1), value, operator.swap())
        } else {
            // In particular, never replace a dynamic comparison with its current snapshot.
            return Ok(false);
        };
        let Some(name) = column
            .as_opt::<GetItem>()
            .filter(|_| column.child(0).is_root())
        else {
            return Ok(false);
        };
        let Some(dtype) = self
            .dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.field(name))
        else {
            return Ok(false);
        };
        let ordered = match &dtype {
            DType::Primitive(ptype, _) => ptype.is_int(),
            DType::Bool(_) | DType::Utf8(_) | DType::Binary(_) | DType::Decimal(..) => true,
            DType::Extension(ext) => AnyTemporal::try_match(ext).is_some(),
            _ => false,
        };
        if !ordered || value.is_null() || !value.dtype().eq_ignore_nullability(&dtype) {
            return Ok(false);
        }
        if !statistics.contains_key(name) {
            statistics.insert(name.clone(), self.top_level_statistics(name.as_ref())?);
        }
        let stats = &statistics[name];
        if stats
            .get_as::<u64>(
                Stat::NullCount,
                &DType::Primitive(PType::U64, Nullability::NonNullable),
            )
            .as_exact()
            != Some(0)
        {
            return Ok(false);
        }
        let endpoint = |stat| {
            let scalar = Scalar::try_new(dtype.clone(), Some(stats.get(stat).as_exact()?)).ok()?;
            (!scalar.is_null())
                .then(|| scalar.partial_cmp(value))
                .flatten()
        };
        let minimum = || endpoint(Stat::Min);
        let maximum = || endpoint(Stat::Max);
        Ok(match operator {
            CompareOperator::Eq => {
                minimum() == Some(Ordering::Equal) && maximum() == Some(Ordering::Equal)
            }
            CompareOperator::NotEq => {
                minimum() == Some(Ordering::Greater) || maximum() == Some(Ordering::Less)
            }
            CompareOperator::Lt => maximum() == Some(Ordering::Less),
            CompareOperator::Lte => maximum().is_some_and(|order| order != Ordering::Greater),
            CompareOperator::Gt => minimum() == Some(Ordering::Greater),
            CompareOperator::Gte => minimum().is_some_and(|order| order != Ordering::Less),
        })
    }

    fn top_level_statistics(&self, name: &str) -> VortexResult<StatsSet> {
        let DType::Struct(fields, _) = &self.dtype else {
            return Ok(StatsSet::default());
        };
        let Some(root_column) = fields.find(name) else {
            return Ok(StatsSet::default());
        };
        let Some(dtype) = fields.field_by_index(root_column) else {
            return Ok(StatsSet::default());
        };
        let schema = self.prepared.ctx.metadata.file_metadata().schema_descr();
        let Some(leaf) = single_leaf_index(schema, root_column) else {
            // A nested logical field's null count and bounds are not any one Dremel leaf's stats.
            return Ok(StatsSet::default());
        };

        let mut minimum: Option<(Scalar, bool)> = None;
        let mut maximum: Option<(Scalar, bool)> = None;
        let mut null_count = Some(0u64);
        let mut parquet_uncompressed_size = Some(0u64);
        let mut bounds_complete = true;
        for row_group in self.prepared.ctx.metadata.row_groups() {
            let column = row_group.column(leaf);
            parquet_uncompressed_size = parquet_uncompressed_size
                .zip(u64::try_from(column.uncompressed_size()).ok())
                .and_then(|(total, size)| total.checked_add(size));
            let Some(statistics) = column.statistics() else {
                null_count = None;
                bounds_complete = false;
                continue;
            };
            null_count = null_count
                .zip(statistics.null_count_opt())
                .and_then(|(total, count)| total.checked_add(count));
            let safe =
                crate::statistics::min_max_trustworthy(statistics, schema.column(leaf).as_ref());
            let (Some(group_min), Some(group_max)) = (
                safe.then(|| min_scalar(statistics, &dtype)).flatten(),
                safe.then(|| max_scalar(statistics, &dtype)).flatten(),
            ) else {
                bounds_complete = false;
                continue;
            };
            bounds_complete &= update_min(&mut minimum, group_min, statistics.min_is_exact());
            bounds_complete &= update_max(&mut maximum, group_max, statistics.max_is_exact());
        }

        let mut stats = StatsSet::default();
        if let Some(count) = null_count {
            stats.set(Stat::NullCount, Precision::exact(count));
        }
        let num_rows = u64::try_from(self.prepared.ctx.metadata.file_metadata().num_rows()).ok();
        let arrow_dtype = self
            .prepared
            .ctx
            .arrow_metadata
            .schema()
            .field(root_column)
            .data_type();
        if let Some(size) = arrow_dtype
            .primitive_width()
            .and_then(|width| num_rows?.checked_mul(u64::try_from(width).ok()?))
        {
            stats.set(Stat::UncompressedSizeInBytes, Precision::exact(size));
        } else if let Some(size) = parquet_uncompressed_size {
            // This is precisely DataFusion's Parquet convention: for variable-width columns the
            // footer's uncompressed Parquet bytes are an estimate of Arrow memory, not an exact
            // size. Vortex's stats precision survives conversion into DataFusion.
            stats.set(Stat::UncompressedSizeInBytes, Precision::inexact(size));
        }
        if bounds_complete {
            if let Some((minimum, exact)) = minimum
                && let Some(minimum) = minimum.into_value()
            {
                stats.set(
                    Stat::Min,
                    if exact {
                        Precision::exact(minimum)
                    } else {
                        Precision::inexact(minimum)
                    },
                );
            }
            if let Some((maximum, exact)) = maximum
                && let Some(maximum) = maximum.into_value()
            {
                stats.set(
                    Stat::Max,
                    if exact {
                        Precision::exact(maximum)
                    } else {
                        Precision::inexact(maximum)
                    },
                );
            }
        }
        Ok(stats)
    }
}

fn update_min(current: &mut Option<(Scalar, bool)>, candidate: Scalar, exact: bool) -> bool {
    let Some((value, current_exact)) = current else {
        *current = Some((candidate, exact));
        return true;
    };
    match candidate.partial_cmp(value) {
        Some(Ordering::Less) => *current = Some((candidate, exact)),
        Some(Ordering::Equal) => *current_exact |= exact,
        Some(Ordering::Greater) => {}
        None => return false,
    }
    true
}

fn update_max(current: &mut Option<(Scalar, bool)>, candidate: Scalar, exact: bool) -> bool {
    let Some((value, current_exact)) = current else {
        *current = Some((candidate, exact));
        return true;
    };
    match candidate.partial_cmp(value) {
        Some(Ordering::Greater) => *current = Some((candidate, exact)),
        Some(Ordering::Equal) => *current_exact |= exact,
        Some(Ordering::Less) => {}
        None => return false,
    }
    true
}

#[async_trait]
impl DataSource for ParquetDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_groups.last().map_or(0, |group| group.end))
    }

    fn byte_size(&self) -> Precision<u64> {
        // DataFusion's native Parquet metadata adapter leaves file-level total_byte_size absent.
        // Its per-column Arrow memory estimates below are what the cost model consumes.
        Precision::Absent
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("Parquet row-group partitions are not serializable")
    }

    async fn scan(&self, mut scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.row_groups.last().map_or(0, |group| group.end);
        let requested = scan_request.row_range.unwrap_or(0..total_rows);
        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && scan_request.filter.is_none()
        {
            let selected = scan_request
                .selection
                .row_mask(&requested)
                .mask()
                .true_count() as u64;
            return Ok(Box::new(EmptyScan {
                dtype,
                row_count: scan_request
                    .limit
                    .map_or(selected, |limit| selected.min(limit)),
            }));
        }

        if let Some(filter) = scan_request.filter.take() {
            scan_request.filter = self.unsatisfied_filter(filter, &mut HashMap::new())?;
        }

        let parts = self.scan_file()?;
        // Bound once in the file scope and shared by every row group's scan. Optimized as Vortex's
        // own data sources do: engines hand over expressions over their projected row, such as
        // DataFusion's `pack(id: $.id).id >= 4000`, and zone pruning only sees through the
        // simplified `$.id >= 4000`.
        let projection = scan_request
            .projection
            .bind(&self.dtype)?
            .optimize_recursive()?;
        let filter = scan_request
            .filter
            .map(|filter| filter.bind(&self.dtype)?.optimize_recursive())
            .transpose()?;

        let partition_domain = 0..u64::try_from(self.row_groups.len())?;
        let partition_range = scan_request.partition_range.as_ref().map_or_else(
            || partition_domain.clone(),
            |range| intersect(range, &partition_domain).unwrap_or(0..0),
        );
        let partition_mask = scan_request.partition_selection.row_mask(&partition_range);
        let row_groups = self
            .row_groups
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let idx = u64::try_from(*idx).unwrap_or(u64::MAX);
                idx >= partition_range.start
                    && idx < partition_range.end
                    && usize::try_from(idx - partition_range.start)
                        .is_ok_and(|idx| partition_mask.mask().value(idx))
            })
            .filter_map(|(idx, row_group)| {
                intersect(row_group, &requested).map(|range| (idx, range))
            })
            .collect::<Vec<_>>();
        // The driver's limit is per worker thread; a request's is for the whole partition.
        let concurrency = scan_request.split_concurrency.map_or(1, |splits| {
            splits
                .get()
                .div_ceil(get_available_parallelism().unwrap_or(1))
                .max(1)
        });
        Ok(Box::new(RowGroupScan {
            parts,
            session: self.session.clone(),
            dtype,
            projection,
            filter,
            limit: scan_request.limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            concurrency,
            row_groups,
            next: 0,
        }))
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        let [field] = field_path.parts() else {
            return Ok(StatsSet::default());
        };
        let Some(name) = field.as_name() else {
            return Ok(StatsSet::default());
        };
        self.top_level_statistics(name)
    }
}

fn intersect(left: &Range<u64>, right: &Range<u64>) -> Option<Range<u64>> {
    let start = left.start.max(right.start);
    let end = left.end.min(right.end);
    (start < end).then_some(start..end)
}

struct RowGroupScan {
    parts: ScanFile,
    session: VortexSession,
    dtype: DType,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    limit: Option<u64>,
    selection: Selection,
    ordered: bool,
    concurrency: usize,
    row_groups: Vec<(usize, Range<u64>)>,
    next: usize,
}

impl DataSourceScan for RowGroupScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(self.row_groups.len())
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        (*self).boxed()
    }
}

impl Stream for RowGroupScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.next >= self.row_groups.len() || self.limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }
        let (row_group, row_range) = self.row_groups[self.next].clone();
        self.next += 1;
        let selected_rows = self.selection.row_mask(&row_range).mask().true_count() as u64;
        let split_limit = self.limit;
        if self.filter.is_none()
            && let Some(limit) = &mut self.limit
        {
            *limit = limit.saturating_sub(selected_rows);
        }
        Poll::Ready(Some(Ok(Box::new(RowGroupPartition {
            parts: self.parts.clone(),
            session: self.session.clone(),
            dtype: self.dtype.clone(),
            projection: self.projection.clone(),
            filter: self.filter.clone(),
            limit: split_limit,
            selection: self.selection.clone(),
            ordered: self.ordered,
            concurrency: self.concurrency,
            row_group,
            row_range,
        }))))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.row_groups.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

/// One row group, scanned through the V2 plan its layout lowers to.
struct RowGroupPartition {
    parts: ScanFile,
    session: VortexSession,
    dtype: DType,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    limit: Option<u64>,
    selection: Selection,
    ordered: bool,
    concurrency: usize,
    row_group: usize,
    row_range: Range<u64>,
}

impl Partition for RowGroupPartition {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index(&self) -> usize {
        self.row_group
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = self.selection.row_mask(&self.row_range).mask().true_count() as u64;
        let rows = self.limit.map_or(rows, |limit| rows.min(limit));
        if self.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let builder = V2ScanBuilder::new(self.session.clone(), self.parts.clone())
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_row_range(self.row_range)
            .with_selection(self.selection);
        let driver = ScanDriver::new(self.session.handle(), self.concurrency, self.ordered);
        let stream = driver.run(builder);
        let stream = match self.limit {
            Some(limit) => limit_rows(stream, limit),
            None => stream,
        };
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            self.dtype, stream,
        )))
    }
}

/// Ends `stream` after `limit` rows, slicing the batch that crosses it.
fn limit_rows(
    stream: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: u64,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    stream
        .scan(limit, |remaining, batch| {
            if *remaining == 0 {
                return future::ready(None);
            }
            let batch = batch.and_then(|batch| {
                let len = batch.len() as u64;
                if len <= *remaining {
                    *remaining -= len;
                    Ok(batch)
                } else {
                    let keep = usize::try_from(*remaining)?;
                    *remaining = 0;
                    batch.slice(0..keep)
                }
            });
            future::ready(Some(batch))
        })
        .boxed()
}

struct EmptyScan {
    dtype: DType,
    row_count: u64,
}

impl DataSourceScan for EmptyScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as PartitionRef)]).boxed()
    }
}

impl Partition for EmptyScan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::exact(0u64)
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let array = ConstantArray::new(
            Scalar::struct_(self.dtype.clone(), vec![]),
            usize::try_from(self.row_count)?,
        )
        .into_array();
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            self.dtype,
            stream::iter([Ok(array)]),
        )))
    }
}
