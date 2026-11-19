// SPDX-License-Identifier: Apache-2.0

//! Parquet statistics translated into Vortex's generic falsification predicates.
//!
//! Trusted bounds can directly disprove simple comparisons in totally ordered scalar domains.
//! Other expressions use the aggregate properties requested by [`BoundExpression::falsify`],
//! leaving unsupported expressions, dynamic values, and floating-point semantics to Vortex.

#[cfg(test)]
use std::cmp::Ordering;

#[cfg(test)]
use parquet::data_type::ByteArray;
#[cfg(test)]
use parquet::data_type::FixedLenByteArray;
#[cfg(test)]
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnDescriptor;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex::aggregate_fn::fns::all_null::AllNull;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::scalar_fn::internal::row_count::contains_row_count;
use vortex::array::scalar_fn::internal::row_count::substitute_row_count;
use vortex::buffer::ByteBuffer;
use parquet::file::metadata::ParquetMetaData;
use vortex::dtype::DType;
use vortex::dtype::StructFields;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
#[cfg(test)]
use vortex::dtype::extension::Matcher;
use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::expr::bound::lit;
use vortex::expr::stats::bind::StatBinder;
use vortex::expr::stats::bind::bind_stats;
use vortex::expr::stats::rewrite::BoundExpressionPruneExt;
#[cfg(test)]
use vortex::extension::datetime::AnyTemporal;
use vortex::mask::Mask;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
#[cfg(test)]
use vortex::scalar_fn::fns::between::Between;
#[cfg(test)]
use vortex::scalar_fn::fns::binary::Binary;
#[cfg(test)]
use vortex::scalar_fn::fns::literal::Literal;
#[cfg(test)]
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::session::VortexSession;

/// Footer or page statistics over one flat physical Parquet column.
pub(crate) struct ParquetStats<'a> {
    pub(crate) dtype: &'a DType,
    pub(crate) statistics: &'a Statistics,
    pub(crate) row_count: u64,
    /// Whether the min/max bytes use an ordering compatible with `dtype`.
    ///
    /// Null counts remain useful when this is false. Older Parquet files wrote min/max using the
    /// deprecated fields, whose ordering is undefined for byte arrays and unsigned values; binding
    /// those bounds can prune rows that actually match.
    pub(crate) min_max_safe: bool,
}

impl StatBinder for ParquetStats<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        if !input.is_root() {
            return Ok(None);
        }
        Ok(self.bind_leaf(aggregate_fn))
    }
}

impl ParquetStats<'_> {
    /// Binds `aggregate_fn` of this column to its statistic, when the footer has it.
    fn bind_leaf(&self, aggregate_fn: &AggregateFnRef) -> Option<BoundExpression> {
        if aggregate_fn.is::<NullCount>() {
            return self.statistics.null_count_opt().map(lit);
        }
        if aggregate_fn.is::<AllNull>() {
            return self
                .statistics
                .null_count_opt()
                .map(|count| lit(count == self.row_count));
        }
        if aggregate_fn.is::<AllNonNull>() {
            return self
                .statistics
                .null_count_opt()
                .map(|count| lit(count == 0));
        }
        if aggregate_fn.is::<Min>() {
            return self
                .min_max_safe
                .then(|| stat_scalar(self.statistics, self.dtype, Bound::Min))
                .flatten()
                .map(lit);
        }
        if aggregate_fn.is::<Max>() {
            return self
                .min_max_safe
                .then(|| stat_scalar(self.statistics, self.dtype, Bound::Max))
                .flatten()
                .map(lit);
        }

        None
    }
}

#[cfg(test)]
/// Returns whether Parquet's statistics prove `expr` false for this entire statistics scope.
pub(crate) fn can_prune(
    expr: &BoundExpression,
    stats: &ParquetStats<'_>,
    session: &VortexSession,
) -> VortexResult<bool> {
    if let Some(prunes) = direct_comparison_prunes(expr, stats) {
        return Ok(prunes);
    }
    if let Some(prunes) = direct_between_prunes(expr, stats) {
        return Ok(prunes);
    }
    falsified(expr, stats, stats.dtype, stats.row_count, session)
}

/// Whether `binder`'s statistics prove `expr`, over `row_count` rows of `dtype`, false for every
/// row. Dynamic expressions are evaluated at their current values.
fn falsified(
    expr: &BoundExpression,
    binder: &dyn StatBinder,
    dtype: &DType,
    row_count: u64,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(predicate) = expr.falsify(session)? else {
        return Ok(false);
    };
    let predicate = bind_stats(predicate, binder)?;
    let input = ConstantArray::new(Scalar::null(dtype.as_nullable()), 1).into_array();
    let mut applied = input.apply_bound(&predicate)?;
    if contains_row_count(&applied) {
        let count = ConstantArray::new(row_count, 1).into_array();
        applied = substitute_row_count(applied, &count)?;
    }
    let mut ctx = session.create_execution_ctx();
    let mask = applied.fill_null(false)?.execute::<Mask>(&mut ctx)?;
    Ok(mask.value(0))
}

/// The footer statistics of one row group, for a filter over the whole row.
///
/// Binds the statistics of each flat top-level field the filter reads, under the same trust rules
/// as the scan's footer zone maps.
pub(crate) struct RowGroupStats<'a> {
    pub(crate) fields: &'a StructFields,
    pub(crate) metadata: &'a ParquetMetaData,
    pub(crate) row_group: usize,
}

impl StatBinder for RowGroupStats<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(name) = input
            .as_opt::<GetItem>()
            .filter(|_| input.child(0).is_root())
        else {
            return Ok(None);
        };
        let Some(index) = self.fields.find(name) else {
            return Ok(None);
        };
        let Some(dtype) = self.fields.field_by_index(index) else {
            return Ok(None);
        };
        let schema = self.metadata.file_metadata().schema_descr();
        let Some(leaf) = crate::pages::single_leaf_index(schema, index) else {
            return Ok(None);
        };
        let descriptor = schema.column(leaf);
        if !crate::pages::direct_dtype(&dtype, descriptor.physical_type()) {
            return Ok(None);
        }
        let row_group = self.metadata.row_group(self.row_group);
        let Some(statistics) = row_group.column(leaf).statistics() else {
            return Ok(None);
        };
        Ok(ParquetStats {
            dtype: &dtype,
            statistics,
            row_count: u64::try_from(row_group.num_rows())?,
            min_max_safe: min_max_trustworthy(statistics, descriptor.as_ref()),
        }
        .bind_leaf(aggregate_fn))
    }
}

/// Whether row group `row_group`'s footer statistics prove `filter`, over a row of `dtype`, false
/// for every row, evaluating any dynamic expression at its current value.
pub(crate) fn row_group_falsified(
    filter: &BoundExpression,
    dtype: &DType,
    metadata: &ParquetMetaData,
    row_group: usize,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(fields) = dtype.as_struct_fields_opt() else {
        return Ok(false);
    };
    let row_count = u64::try_from(metadata.row_group(row_group).num_rows())?;
    let binder = RowGroupStats {
        fields,
        metadata,
        row_group,
    };
    falsified(filter, &binder, dtype, row_count, session)
}

#[cfg(test)]
/// Evaluate the bounds proof for supported root/literal comparisons without building a generic
/// stats expression. Those rewrites repeat these same endpoint comparisons, including when the
/// bounds cannot disprove the predicate. Null rows cannot satisfy these comparisons either.
/// Temporal values use their integer storage order after checking that units and timezone match.
/// Floats, other extensions and other expressions keep the generic rewrite path.
fn direct_comparison_prunes(expr: &BoundExpression, stats: &ParquetStats<'_>) -> Option<bool> {
    if !stats.min_max_safe
        || !match stats.dtype {
            DType::Primitive(ptype, _) => ptype.is_int(),
            DType::Bool(_) | DType::Utf8(_) | DType::Binary(_) | DType::Decimal(..) => true,
            DType::Extension(ext) => AnyTemporal::try_match(ext).is_some(),
            _ => false,
        }
    {
        return None;
    }
    let operator = expr.as_opt::<Binary>()?.to_compare_operator()?;
    let (operator, value) = if expr.child(0).is_root() {
        (operator, expr.child(1).as_opt::<Literal>()?)
    } else if expr.child(1).is_root() {
        (operator.swap(), expr.child(0).as_opt::<Literal>()?)
    } else {
        return None;
    };
    if value.is_null() || !value.dtype().eq_ignore_nullability(stats.dtype) {
        return None;
    }
    let minimum = || min_scalar(stats.statistics, stats.dtype)?.partial_cmp(value);
    let maximum = || max_scalar(stats.statistics, stats.dtype)?.partial_cmp(value);
    Some(match operator {
        CompareOperator::Eq => {
            minimum() == Some(Ordering::Greater) || maximum() == Some(Ordering::Less)
        }
        CompareOperator::NotEq => {
            minimum() == Some(Ordering::Equal) && maximum() == Some(Ordering::Equal)
        }
        CompareOperator::Lt => minimum().is_some_and(|order| order != Ordering::Less),
        CompareOperator::Lte => minimum() == Some(Ordering::Greater),
        CompareOperator::Gt => maximum().is_some_and(|order| order != Ordering::Greater),
        CompareOperator::Gte => maximum() == Some(Ordering::Less),
    })
}

#[cfg(test)]
/// Check a root/literal interval directly, preserving the generic path for Boolean expressions.
#[inline(never)]
fn direct_between_prunes(expr: &BoundExpression, stats: &ParquetStats<'_>) -> Option<bool> {
    if !stats.min_max_safe
        || !match stats.dtype {
            DType::Primitive(ptype, _) => ptype.is_int(),
            DType::Bool(_) | DType::Utf8(_) | DType::Binary(_) | DType::Decimal(..) => true,
            DType::Extension(ext) => AnyTemporal::try_match(ext).is_some(),
            _ => false,
        }
    {
        return None;
    }
    if let Some(options) = expr.as_opt::<Between>() {
        if !expr.child(0).is_root() {
            return None;
        }
        let lower = expr.child(1).as_opt::<Literal>()?;
        let upper = expr.child(2).as_opt::<Literal>()?;
        if lower.is_null()
            || upper.is_null()
            || !lower.dtype().eq_ignore_nullability(stats.dtype)
            || !upper.dtype().eq_ignore_nullability(stats.dtype)
        {
            return None;
        }
        let below = max_scalar(stats.statistics, stats.dtype)?.partial_cmp(lower);
        let above = min_scalar(stats.statistics, stats.dtype)?.partial_cmp(upper);
        return Some(
            below == Some(Ordering::Less)
                || (options.lower_strict.is_strict() && below == Some(Ordering::Equal))
                || above == Some(Ordering::Greater)
                || (options.upper_strict.is_strict() && above == Some(Ordering::Equal)),
        );
    }
    None
}

#[derive(Clone, Copy)]
enum Bound {
    Min,
    Max,
}

/// Decodes a footer minimum into the declared logical dtype.
pub(crate) fn min_scalar(statistics: &Statistics, dtype: &DType) -> Option<Scalar> {
    stat_scalar(statistics, dtype, Bound::Min)
}

/// Decodes a footer maximum into the declared logical dtype.
pub(crate) fn max_scalar(statistics: &Statistics, dtype: &DType) -> Option<Scalar> {
    stat_scalar(statistics, dtype, Bound::Max)
}

fn stat_scalar(statistics: &Statistics, dtype: &DType, bound: Bound) -> Option<Scalar> {
    // Parquet's float statistics do not provide the all-non-NaN proof that Vortex's comparison
    // rewrites require, so binding their bounds cannot by itself cause unsafe pruning. They remain
    // useful once a writer supplies the additional proof through another statistics source.
    let nullability = Nullability::Nullable;
    // A decimal bound outside the declared precision is dropped rather than trusted: the data it
    // describes is itself invalid, and decoding reports that as an error.
    if let DType::Decimal(decimal_dtype, _) = dtype {
        return match statistics {
            Statistics::Int32(values) => selected(values.min_opt(), values.max_opt(), bound)
                .copied()
                .and_then(|value| {
                    Scalar::try_decimal(DecimalValue::I32(value), *decimal_dtype, nullability).ok()
                }),
            Statistics::Int64(values) => selected(values.min_opt(), values.max_opt(), bound)
                .copied()
                .and_then(|value| {
                    Scalar::try_decimal(DecimalValue::I64(value), *decimal_dtype, nullability).ok()
                }),
            _ => None,
        };
    }
    let physical = match statistics {
        Statistics::Boolean(values) => selected(values.min_opt(), values.max_opt(), bound)
            .copied()
            .map(|value| Scalar::bool(value, nullability)),
        Statistics::Int32(values) => selected(values.min_opt(), values.max_opt(), bound)
            .copied()
            .map(|value| Scalar::primitive(value, nullability)),
        Statistics::Int64(values) => selected(values.min_opt(), values.max_opt(), bound)
            .copied()
            .map(|value| Scalar::primitive(value, nullability)),
        Statistics::Float(values) => selected(values.min_opt(), values.max_opt(), bound)
            .copied()
            .map(|value| Scalar::primitive(value, nullability)),
        Statistics::Double(values) => selected(values.min_opt(), values.max_opt(), bound)
            .copied()
            .map(|value| Scalar::primitive(value, nullability)),
        Statistics::ByteArray(values) => {
            let bytes = selected(values.min_opt(), values.max_opt(), bound)
                .map(|value| ByteBuffer::copy_from(value.data()));
            match dtype {
                DType::Utf8(_) => bytes.and_then(|bytes| Scalar::try_utf8(bytes, nullability).ok()),
                DType::Binary(_) => bytes.map(|bytes| Scalar::binary(bytes, nullability)),
                _ => None,
            }
        }
        Statistics::Int96(_) | Statistics::FixedLenByteArray(_) => None,
    }?;

    // Statistics are decoded according to the Parquet physical type. The expression being
    // falsified is typed according to the Arrow/Vortex logical schema, which may narrow INT32 to
    // i8/i16 or wrap INT32/INT64 in a temporal extension. Bind a scalar of that declared dtype; a
    // physical scalar in its place either prevents pruning or, worse, gives the optimizer bounds in
    // a different comparison domain. A malformed out-of-range bound disables pruning rather than
    // turning metadata into a scan failure.
    match dtype {
        DType::Primitive(PType::U32, _) if physical.dtype().as_ptype() == PType::I32 => physical
            .primitive_reinterpret_cast(PType::U32)
            .and_then(|value| value.cast(dtype))
            .ok(),
        DType::Primitive(PType::U64, _) if physical.dtype().as_ptype() == PType::I64 => physical
            .primitive_reinterpret_cast(PType::U64)
            .and_then(|value| value.cast(dtype))
            .ok(),
        _ => physical.cast(dtype).ok(),
    }
}

fn selected<'a, T>(min: Option<&'a T>, max: Option<&'a T>, bound: Bound) -> Option<&'a T> {
    match bound {
        Bound::Min => min,
        Bound::Max => max,
    }
}

#[cfg(test)]
/// Builds typed statistics for one decoded parquet-rs column-index entry.
///
/// parquet-rs uses its own compact-Thrift decoder for column indexes (not the generated `thrift`
/// crate protocol used by older APIs), so pruning consumes its typed metadata directly. Besides
/// avoiding a second conversion, this handles compact boolean-list encodings that the generated
/// protocol rejects as malformed.
pub(crate) fn page_index_statistics(
    index: &ColumnIndexMetaData,
    page: usize,
    null_count: Option<u64>,
) -> Option<Statistics> {
    Some(match index {
        ColumnIndexMetaData::NONE => return None,
        ColumnIndexMetaData::BOOLEAN(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::INT32(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::INT64(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::INT96(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::FLOAT(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::DOUBLE(index) => Statistics::new(
            index.min_value(page).copied(),
            index.max_value(page).copied(),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::BYTE_ARRAY(index) => Statistics::new(
            index
                .min_value(page)
                .map(|value| ByteArray::from(value.to_vec())),
            index
                .max_value(page)
                .map(|value| ByteArray::from(value.to_vec())),
            None,
            null_count,
            false,
        ),
        ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(index) => Statistics::new(
            index
                .min_value(page)
                .map(|value| FixedLenByteArray::from(value.to_vec())),
            index
                .max_value(page)
                .map(|value| FixedLenByteArray::from(value.to_vec())),
            None,
            null_count,
            false,
        ),
    })
}

/// Whether a column chunk's `min`/`max` may be trusted as bounds in the column's **logical** domain.
///
/// # Why the obvious check is not enough
///
/// Parquet's pre-2.0 `min`/`max` fields were written using the *signed* comparison order of the
/// physical type. arrow-rs exposes `is_min_max_backwards_compatible()` for this, and the natural
/// reading — "deprecated bounds are fine when they are backwards compatible" — is what both callers
/// used. It is wrong for unsigned logical types.
///
/// For a `UINT32` column holding the values `2^31` and `0`, the deprecated bounds decode in *signed*
/// order, so the logical bounds come back as `min = 2^31, max = 0` — inverted. A predicate `x = 0`
/// then falls outside `[min, max]` and the page is pruned, **discarding a row that matched**. No
/// error is raised and the query simply returns the wrong answer.
///
/// That is the one direction of pruning error that is never acceptable: under-pruning costs time,
/// over-pruning costs correctness. So deprecated bounds are additionally required to come from a
/// column whose schema sort order is signed, which is exactly the domain they were written in.
///
/// Centralised because it was duplicated at the two footer-statistics call sites, and a rule that
/// decides what may be skipped must not be able to disagree with itself.
pub(crate) fn min_max_trustworthy(statistics: &Statistics, descriptor: &ColumnDescriptor) -> bool {
    !statistics.is_min_max_deprecated()
        || (statistics.is_min_max_backwards_compatible() && descriptor.sort_order().is_signed())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use bytes::Bytes;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::LogicalType;
    use parquet::basic::Type as PhysicalType;
    use parquet::file::properties::WriterProperties;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::ReadOptionsBuilder;
    use parquet::file::serialized_reader::SerializedFileReader;
    use parquet::file::statistics::ValueStatistics;
    use parquet::schema::types::ColumnPath;
    use parquet::schema::types::Type as SchemaType;
    use vortex::VortexSessionDefault;
    use vortex::dtype::DecimalDType;
    use vortex::error::VortexResult;
    use vortex::expr::Expression;
    use vortex::expr::between;
    use vortex::expr::eq;
    use vortex::expr::gt;
    use vortex::expr::gt_eq;
    use vortex::expr::is_not_null;
    use vortex::expr::is_null;
    use vortex::expr::lit;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::expr::not_eq;
    use vortex::expr::root;
    use vortex::extension::datetime::Date;
    use vortex::extension::datetime::Time;
    use vortex::extension::datetime::TimeUnit;
    use vortex::extension::datetime::Timestamp;
    use vortex::scalar_fn::fns::between::BetweenOptions;
    use vortex::scalar_fn::fns::between::StrictComparison;

    use super::*;

    fn prunes(
        expr: &Expression,
        dtype: &DType,
        statistics: &Statistics,
        row_count: u64,
        min_max_safe: bool,
    ) -> VortexResult<bool> {
        can_prune(
            &expr.bind(dtype)?,
            &ParquetStats {
                dtype,
                statistics,
                row_count,
                min_max_safe,
            },
            &VortexSession::default(),
        )
    }

    fn utf8_statistics(min: &str, max: &str, min_is_exact: bool, max_is_exact: bool) -> Statistics {
        Statistics::ByteArray(
            ValueStatistics::new(
                Some(ByteArray::from(min.as_bytes().to_vec())),
                Some(ByteArray::from(max.as_bytes().to_vec())),
                None,
                Some(0),
                false,
            )
            .with_min_is_exact(min_is_exact)
            .with_max_is_exact(max_is_exact),
        )
    }

    fn string_page_statistics(truncation_length: Option<usize>) -> Statistics {
        let values = Arc::new(StringArray::from(vec!["bookkeeper", "bookshelf"])) as ArrayRef;
        let batch = RecordBatch::try_from_iter([("word", values)]).expect("the batch builds");
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_data_page_row_count_limit(2)
            .set_column_index_truncate_length(truncation_length)
            .build();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(properties))
            .expect("the writer builds");
        writer.write(&batch).expect("the batch writes");
        writer.close().expect("the writer closes");

        let options = ReadOptionsBuilder::new().with_page_index().build();
        let reader = SerializedFileReader::new_with_options(Bytes::from(bytes), options)
            .expect("the page index reads");
        let indexes = reader
            .metadata()
            .column_index()
            .expect("the writer emitted a column index");
        let index = &indexes[0][0];
        assert_eq!(
            index.num_pages(),
            1,
            "the fixture should have one data page"
        );
        page_index_statistics(index, 0, Some(0)).expect("the page has typed statistics")
    }

    /// Exact extrema are values in the page, so inclusive predicates at either boundary must stay.
    ///
    /// The strict neighbors are asserted too: they prove this reaches both bound binders instead of
    /// merely disabling pruning for the fixture's dtype.
    #[test]
    fn exact_scalar_extrema_keep_their_boundary_rows() -> VortexResult<()> {
        let nullable = Nullability::Nullable;
        // Temporal comparisons use integer storage in one fixed unit and timezone. Dates and
        // timestamps also exercise values before the epoch; time-of-day values stay positive.
        let dtypes = [
            (DType::Primitive(PType::I32, nullable), PType::I32, 0),
            (
                DType::Extension(Date::new(TimeUnit::Days, nullable).erased()),
                PType::I32,
                -30,
            ),
            (
                DType::Extension(Date::new(TimeUnit::Milliseconds, nullable).erased()),
                PType::I64,
                -30,
            ),
            (
                DType::Extension(Time::new(TimeUnit::Seconds, nullable).erased()),
                PType::I32,
                0,
            ),
            (
                DType::Extension(Time::new(TimeUnit::Milliseconds, nullable).erased()),
                PType::I32,
                0,
            ),
            (
                DType::Extension(Time::new(TimeUnit::Microseconds, nullable).erased()),
                PType::I64,
                0,
            ),
            (
                DType::Extension(Time::new(TimeUnit::Nanoseconds, nullable).erased()),
                PType::I64,
                0,
            ),
            (
                DType::Extension(Timestamp::new(TimeUnit::Seconds, nullable).erased()),
                PType::I64,
                -30,
            ),
            (
                DType::Extension(Timestamp::new(TimeUnit::Milliseconds, nullable).erased()),
                PType::I64,
                -30,
            ),
            (
                DType::Extension(Timestamp::new(TimeUnit::Microseconds, nullable).erased()),
                PType::I64,
                -30,
            ),
            (
                DType::Extension(
                    Timestamp::new_with_tz(
                        TimeUnit::Nanoseconds,
                        Some("Europe/London".into()),
                        nullable,
                    )
                    .erased(),
                ),
                PType::I64,
                -30,
            ),
        ];
        for (dtype, physical, offset) in dtypes {
            let literal = |value: i64| lit(Scalar::from(value + offset).cast(&dtype).unwrap());
            let stats = |minimum: i64, maximum: i64, null_count| match physical {
                PType::I32 => Statistics::int32(
                    Some(i32::try_from(minimum + offset).unwrap()),
                    Some(i32::try_from(maximum + offset).unwrap()),
                    None,
                    Some(null_count),
                    false,
                ),
                PType::I64 => Statistics::int64(
                    Some(minimum + offset),
                    Some(maximum + offset),
                    None,
                    Some(null_count),
                    false,
                ),
                _ => unreachable!("the fixture uses signed integer storage"),
            };
            let statistics = stats(10, 20, 0);
            for (expr, expected) in [
                (eq(root(), literal(10)), false),
                (eq(root(), literal(20)), false),
                (eq(root(), literal(9)), true),
                (eq(root(), literal(21)), true),
                (lt(root(), literal(10)), true),
                (lt_eq(root(), literal(10)), false),
                (gt(root(), literal(20)), true),
                (gt_eq(root(), literal(20)), false),
                (lt(literal(10), root()), false),
                (lt_eq(literal(9), root()), false),
                (gt(literal(20), root()), false),
                (gt_eq(literal(21), root()), false),
                (eq(literal(10), root()), false),
                (eq(literal(21), root()), true),
                (lt(literal(20), root()), true),
                (lt_eq(literal(20), root()), false),
                (gt(literal(10), root()), true),
                (gt_eq(literal(10), root()), false),
                (not_eq(root(), literal(10)), false),
                (not_eq(literal(15), root()), false),
            ] {
                assert_eq!(
                    prunes(&expr, &dtype, &statistics, 8, true)?,
                    expected,
                    "{dtype}: {expr}"
                );
            }
            for (lower, upper, lower_strict, upper_strict, expected) in [
                (10, 20, false, false, false),
                (10, 20, true, true, false),
                (0, 10, false, false, false),
                (0, 10, false, true, true),
                (20, 30, false, false, false),
                (20, 30, true, false, true),
                (0, 9, false, false, true),
                (21, 30, false, false, true),
            ] {
                let strictness = |strict| {
                    if strict {
                        StrictComparison::Strict
                    } else {
                        StrictComparison::NonStrict
                    }
                };
                let expr = between(
                    root(),
                    literal(lower),
                    literal(upper),
                    BetweenOptions {
                        lower_strict: strictness(lower_strict),
                        upper_strict: strictness(upper_strict),
                    },
                );
                assert_eq!(
                    prunes(&expr, &dtype, &statistics, 8, true)?,
                    expected,
                    "{dtype}: {expr}"
                );
                assert!(
                    !prunes(&expr, &dtype, &statistics, 8, false)?,
                    "{dtype}: untrusted {expr}"
                );
            }
            assert!(
                !prunes(&eq(root(), literal(9)), &dtype, &statistics, 8, false)?,
                "{dtype}: untrusted bounds"
            );
            // All non-null values equal the minimum; NULL cannot satisfy either inequality.
            let singleton = stats(10, 10, 3);
            for (expr, expected) in [
                (not_eq(root(), literal(10)), true),
                (not_eq(literal(10), root()), true),
                (not_eq(root(), literal(11)), false),
                (not_eq(literal(11), root()), false),
            ] {
                assert_eq!(
                    prunes(&expr, &dtype, &singleton, 8, true)?,
                    expected,
                    "{dtype}: {expr}"
                );
            }
        }
        Ok(())
    }

    /// A truncated minimum is a lower bound and a truncated maximum is an upper bound, not values
    /// known to occur. In particular, equality with either bound must remain inconclusive.
    ///
    /// Treating the inexact maximum as a shortened exact value can prune `bookshelf`, even though it
    /// lies below the safe `bp` upper bound.
    #[test]
    fn inexact_string_bounds_remain_one_sided() -> VortexResult<()> {
        let dtype = DType::Utf8(Nullability::Nullable);
        let exact = utf8_statistics("bookkeeper", "bookshelf", true, true);
        let truncated = utf8_statistics("bo", "bp", false, false);
        assert!(exact.min_is_exact() && exact.max_is_exact());
        assert!(!truncated.min_is_exact() && !truncated.max_is_exact());

        for statistics in [&exact, &truncated] {
            assert!(!prunes(
                &eq(root(), lit("bookshelf")),
                &dtype,
                statistics,
                2,
                true,
            )?);
        }
        assert!(prunes(&gt(root(), lit("bp")), &dtype, &truncated, 2, true,)?);
        assert!(!prunes(
            &gt_eq(root(), lit("bp")),
            &dtype,
            &truncated,
            2,
            true,
        )?);
        assert!(!prunes(
            &lt_eq(root(), lit("bo")),
            &dtype,
            &truncated,
            2,
            true,
        )?);
        Ok(())
    }

    /// Page indexes have no exactness bits, so their truncated strings must survive conversion as
    /// directional bounds. This uses a real parquet-rs column index rather than footer-shaped stats.
    ///
    /// Reversing the page-index maximum and minimum makes `bookshelf` look greater than the maximum
    /// and reproduces the over-prune this test guards against.
    #[test]
    fn truncated_page_index_maximum_keeps_matching_strings() -> VortexResult<()> {
        let dtype = DType::Utf8(Nullability::Nullable);
        let exact = string_page_statistics(None);
        let truncated = string_page_statistics(Some(2));
        assert_eq!(exact.min_bytes_opt(), Some(&b"bookkeeper"[..]));
        assert_eq!(exact.max_bytes_opt(), Some(&b"bookshelf"[..]));
        assert_eq!(truncated.min_bytes_opt(), Some(&b"bo"[..]));
        assert_eq!(truncated.max_bytes_opt(), Some(&b"bp"[..]));

        assert!(!prunes(
            &eq(root(), lit("bookshelf")),
            &dtype,
            &truncated,
            2,
            true,
        )?);
        assert!(prunes(&gt(root(), lit("bp")), &dtype, &truncated, 2, true,)?);
        assert!(!prunes(
            &gt_eq(root(), lit("bp")),
            &dtype,
            &truncated,
            2,
            true,
        )?);
        Ok(())
    }

    /// Deprecated bounds are usable only when their signed ordering matches the logical dtype.
    /// Turning bounds off must not also discard independent null-count evidence.
    ///
    /// This covers both outcomes of parquet-rs's compatibility bit: signed legacy bounds still
    /// prune, while explicitly incompatible bounds do not.
    #[test]
    fn deprecated_bounds_are_gated_without_losing_null_counts() -> VortexResult<()> {
        let signed_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let signed = Statistics::Int32(
            ValueStatistics::new(Some(-5), Some(5), None, Some(0), true)
                .with_backwards_compatible_min_max(true),
        );
        assert!(signed.is_min_max_deprecated());
        assert!(signed.is_min_max_backwards_compatible());
        let signed_safe =
            !signed.is_min_max_deprecated() || signed.is_min_max_backwards_compatible();
        assert!(prunes(
            &eq(root(), lit(6i32)),
            &signed_dtype,
            &signed,
            4,
            signed_safe,
        )?);
        assert!(!prunes(
            &eq(root(), lit(5i32)),
            &signed_dtype,
            &signed,
            4,
            signed_safe,
        )?);

        let unsigned_dtype = DType::Primitive(PType::U32, Nullability::Nullable);
        let incompatible = Statistics::Int32(
            ValueStatistics::new(Some(i32::MIN), Some(0), None, Some(0), true)
                .with_backwards_compatible_min_max(false),
        );
        assert!(incompatible.is_min_max_deprecated());
        assert!(!incompatible.is_min_max_backwards_compatible());
        let unsigned_safe =
            !incompatible.is_min_max_deprecated() || incompatible.is_min_max_backwards_compatible();
        assert!(!prunes(
            &eq(root(), lit(0u32)),
            &unsigned_dtype,
            &incompatible,
            2,
            unsigned_safe,
        )?);
        assert!(prunes(
            &is_null(root()),
            &unsigned_dtype,
            &incompatible,
            2,
            unsigned_safe,
        )?);
        Ok(())
    }

    /// Deprecated `min`/`max` on an **unsigned** column must not be trusted as logical bounds.
    ///
    /// The rows represented here are `2^31` and `0`. Written pre-2.0, their bounds are in *signed*
    /// order, so read as unsigned they inverted to `min = 2^31, max = 0` — and a predicate `x = 0`
    /// fell outside that range and pruned the page holding the matching row. A wrong answer with no
    /// error, on the one column type where `is_min_max_backwards_compatible()` alone is not enough.
    ///
    /// This asserts the gate through [`min_max_trustworthy`] rather than restating the rule, so the
    /// test fails if a caller ever reverts to the looser check.
    #[test]
    fn deprecated_unsigned_bounds_are_not_trusted() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::U32, Nullability::Nullable);
        let statistics = Statistics::int32(Some(i32::MIN), Some(0), None, Some(0), true);
        assert!(statistics.is_min_max_deprecated());
        // arrow-rs says these are backwards compatible, which is true *in the signed domain*.
        assert!(statistics.is_min_max_backwards_compatible());

        assert!(
            !min_max_trustworthy(&statistics, &unsigned_descriptor()),
            "deprecated signed bounds must not be trusted for an unsigned column"
        );
        assert!(
            !prunes(
                &eq(root(), lit(0u32)),
                &dtype,
                &statistics,
                2,
                min_max_trustworthy(&statistics, &unsigned_descriptor()),
            )?,
            "a statistics scope containing 0 must not be pruned for x = 0"
        );

        // The same deprecated bounds on a *signed* column remain usable: the fix must not throw away
        // every deprecated statistic, only the ones read in the wrong domain.
        assert!(
            min_max_trustworthy(&statistics, &signed_descriptor()),
            "deprecated bounds on a signed column are still in their own domain"
        );
        Ok(())
    }

    /// A `UINT32` column descriptor, whose schema sort order is unsigned.
    fn unsigned_descriptor() -> ColumnDescriptor {
        descriptor(Some(LogicalType::Integer {
            bit_width: 32,
            is_signed: false,
        }))
    }

    /// A plain `INT32` column descriptor, whose sort order is signed.
    fn signed_descriptor() -> ColumnDescriptor {
        descriptor(None)
    }

    fn descriptor(logical: Option<LogicalType>) -> ColumnDescriptor {
        let primitive = SchemaType::primitive_type_builder("c", PhysicalType::INT32)
            .with_logical_type(logical)
            .build()
            .expect("a primitive column type");
        ColumnDescriptor::new(
            Arc::new(primitive),
            0,
            0,
            ColumnPath::new(vec!["c".to_string()]),
        )
    }

    /// Null counts are proofs in their own right: zero nulls, some nulls, all nulls, and an unknown
    /// count each permit different pruning. Min/max still describe only the non-null subset.
    ///
    /// The partial-null boundary assertion catches treating nullable extrema as if null itself were
    /// ordered below the minimum.
    #[test]
    fn null_counts_only_prune_impossible_nullable_predicates() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let no_nulls = Statistics::int32(Some(1), Some(9), None, Some(0), false);
        assert!(prunes(&is_null(root()), &dtype, &no_nulls, 10, true)?);
        assert!(!prunes(&is_not_null(root()), &dtype, &no_nulls, 10, true,)?);

        let partial_nulls = Statistics::int32(Some(1), Some(9), None, Some(3), false);
        assert!(!prunes(&is_null(root()), &dtype, &partial_nulls, 10, true,)?);
        assert!(!prunes(
            &is_not_null(root()),
            &dtype,
            &partial_nulls,
            10,
            true,
        )?);
        assert!(!prunes(
            &eq(root(), lit(1i32)),
            &dtype,
            &partial_nulls,
            10,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(10i32)),
            &dtype,
            &partial_nulls,
            10,
            true,
        )?);

        let all_null = Statistics::int32(None, None, None, Some(10), false);
        assert!(!prunes(&is_null(root()), &dtype, &all_null, 10, true)?);
        assert!(prunes(&is_not_null(root()), &dtype, &all_null, 10, true,)?);

        let unknown = Statistics::int32(Some(1), Some(9), None, None, false);
        assert!(!prunes(&is_null(root()), &dtype, &unknown, 10, true)?);
        assert!(!prunes(&is_not_null(root()), &dtype, &unknown, 10, true,)?);
        Ok(())
    }

    /// INT32/INT64 statistics carry unsigned logical values in their bit patterns. Casting the
    /// signed physical numbers would drop the bounds or compare them in the wrong domain.
    ///
    /// Both endpoints are known rows; the below-range probes prove the reinterpretation remained
    /// available for pruning.
    #[test]
    fn unsigned_bounds_are_reinterpreted_in_the_logical_domain() -> VortexResult<()> {
        let u32_dtype = DType::Primitive(PType::U32, Nullability::Nullable);
        let u32_statistics = Statistics::int32(Some(i32::MIN), Some(-1), None, Some(0), false);
        assert!(!prunes(
            &eq(root(), lit(1u32 << 31)),
            &u32_dtype,
            &u32_statistics,
            2,
            true,
        )?);
        assert!(!prunes(
            &eq(root(), lit(u32::MAX)),
            &u32_dtype,
            &u32_statistics,
            2,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(0u32)),
            &u32_dtype,
            &u32_statistics,
            2,
            true,
        )?);

        let u64_dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        let u64_statistics = Statistics::int64(Some(i64::MIN), Some(-1), None, Some(0), false);
        assert!(!prunes(
            &eq(root(), lit(1u64 << 63)),
            &u64_dtype,
            &u64_statistics,
            2,
            true,
        )?);
        assert!(!prunes(
            &eq(root(), lit(u64::MAX)),
            &u64_dtype,
            &u64_statistics,
            2,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(0u64)),
            &u64_dtype,
            &u64_statistics,
            2,
            true,
        )?);
        Ok(())
    }

    /// Decimal footer bounds are unscaled integers. Applying the scale while decoding statistics
    /// shifts the pruning domain and can discard a row exactly at an encoded endpoint.
    #[test]
    fn decimal_bounds_stay_in_the_unscaled_domain() -> VortexResult<()> {
        let decimal32 = DecimalDType::new(8, 2);
        let dtype32 = DType::Decimal(decimal32, Nullability::Nullable);
        let statistics32 = Statistics::int32(Some(-125), Some(450), None, Some(0), false);
        let min32 = Scalar::decimal(DecimalValue::I32(-125), decimal32, Nullability::NonNullable);
        let above32 = Scalar::decimal(DecimalValue::I32(451), decimal32, Nullability::NonNullable);
        assert!(!prunes(
            &eq(root(), lit(min32)),
            &dtype32,
            &statistics32,
            5,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(above32)),
            &dtype32,
            &statistics32,
            5,
            true,
        )?);

        let decimal64 = DecimalDType::new(15, 4);
        let dtype64 = DType::Decimal(decimal64, Nullability::Nullable);
        let statistics64 = Statistics::int64(
            Some(-5_000_000_000),
            Some(9_000_000_000),
            None,
            Some(0),
            false,
        );
        let max64 = Scalar::decimal(
            DecimalValue::I64(9_000_000_000),
            decimal64,
            Nullability::NonNullable,
        );
        let above64 = Scalar::decimal(
            DecimalValue::I64(9_000_000_001),
            decimal64,
            Nullability::NonNullable,
        );
        assert!(!prunes(
            &eq(root(), lit(max64)),
            &dtype64,
            &statistics64,
            5,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(above64)),
            &dtype64,
            &statistics64,
            5,
            true,
        )?);
        Ok(())
    }

    /// Float min/max cannot prove a comparison false without proof that every value is non-NaN.
    /// parquet-rs also canonicalizes a zero-only range to `-0.0..=0.0`; decoding must preserve both
    /// signs even though Vortex deliberately declines to prune from those bounds alone.
    #[test]
    fn float_bounds_preserve_signed_zero_but_do_not_prune() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let statistics = Statistics::float(Some(-0.0), Some(0.0), None, Some(0), false);
        let min = min_scalar(&statistics, &dtype).expect("the float minimum decodes");
        let max = max_scalar(&statistics, &dtype).expect("the float maximum decodes");
        assert_eq!(
            min.as_primitive()
                .as_::<f32>()
                .expect("the minimum is f32")
                .to_bits(),
            (-0.0f32).to_bits(),
        );
        assert_eq!(
            max.as_primitive()
                .as_::<f32>()
                .expect("the maximum is f32")
                .to_bits(),
            0.0f32.to_bits(),
        );

        assert!(!prunes(
            &eq(root(), lit(1.0f32)),
            &dtype,
            &statistics,
            2,
            true,
        )?);
        assert!(!prunes(
            &eq(root(), lit(f32::NAN)),
            &dtype,
            &statistics,
            2,
            true,
        )?);
        Ok(())
    }

    /// Binary statistics compare unsigned bytes, including across the signed-byte boundary. A cast
    /// through `i8` would reverse `0x7f < 0x80` and make a present high-byte value disappear.
    #[test]
    fn binary_bounds_use_unsigned_byte_order() -> VortexResult<()> {
        let dtype = DType::Binary(Nullability::Nullable);
        let statistics = Statistics::byte_array(
            Some(ByteArray::from(vec![0x7f])),
            Some(ByteArray::from(vec![0x80])),
            None,
            Some(0),
            false,
        );
        let at_max = Scalar::binary(ByteBuffer::copy_from([0x80]), Nullability::NonNullable);
        let above = Scalar::binary(ByteBuffer::copy_from([0x81]), Nullability::NonNullable);
        assert!(!prunes(
            &eq(root(), lit(at_max)),
            &dtype,
            &statistics,
            2,
            true,
        )?);
        assert!(prunes(
            &eq(root(), lit(above)),
            &dtype,
            &statistics,
            2,
            true,
        )?);
        Ok(())
    }
}
