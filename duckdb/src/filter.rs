// SPDX-License-Identifier: Apache-2.0
//! Convert native DuckDB scan-filter snapshots into owned Vortex predicates.

use std::ffi::c_void;

use duckdb::core::LogicalTypeId;
use duckdb::ffi;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::gt;
use vortex::expr::gt_eq;
use vortex::expr::in_list;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::like;
use vortex::expr::lit;
use vortex::expr::lt;
use vortex::expr::lt_eq;
use vortex::expr::not_eq;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::CompareOperator;

use crate::BindData;
use crate::Result;

unsafe extern "C" {
    pub(crate) fn vp_scan_partition(index: u64) -> bool;
    fn vp_projection_count() -> u64;
    fn vp_is_empty_column(column: u64) -> bool;
    fn vp_projection_column(index: u64) -> u64;
    fn vp_filter_count() -> u64;
    fn vp_filter_get(index: u64, column: *mut u64) -> *const c_void;
    fn vp_filter_kind(node: *const c_void) -> u32;
    fn vp_filter_comparison(node: *const c_void) -> u32;
    fn vp_filter_dynamic(node: *const c_void) -> *mut c_void;
    fn vp_dynamic_free(handle: *mut c_void);
    fn vp_dynamic_value(handle: *mut c_void) -> ffi::duckdb_value;
    fn vp_filter_children(node: *const c_void) -> u64;
    fn vp_filter_child(node: *const c_void, index: u64) -> *const c_void;
    fn vp_filter_field(node: *const c_void, len: *mut u64) -> *const u8;
    fn vp_filter_value(node: *const c_void) -> ffi::duckdb_value;
    fn vp_value_type(value: ffi::duckdb_value) -> u32;
    fn vp_value_bytes(value: ffi::duckdb_value, len: *mut u64) -> *const u8;
}

struct DynamicFilter(*mut c_void);
// SAFETY: owns a C++ shared_ptr; the mutable filter data is protected by its own mutex.
unsafe impl Send for DynamicFilter {}
// SAFETY: all reads copy the Value under the C++ filter's mutex.
unsafe impl Sync for DynamicFilter {}
impl DynamicFilter {
    fn latest(&self, dtype: &DType) -> Option<ScalarValue> {
        // SAFETY: our handle stays alive through this call. The returned Value is an owned copy.
        unsafe {
            let mut value = vp_dynamic_value(self.0);
            if value.is_null() {
                return None;
            }
            let result = scalar(value, dtype).ok().and_then(Scalar::into_value);
            ffi::duckdb_destroy_value(&raw mut value);
            result
        }
    }
}
impl Drop for DynamicFilter {
    fn drop(&mut self) {
        // SAFETY: releases our single owning shared_ptr wrapper, after all closures are dropped.
        unsafe { vp_dynamic_free(self.0) };
    }
}

// SAFETY: installed only for ParquetVTab; DuckDB retains BindData throughout planning. This
// callback does no allocation or fallible operations and must not unwind across C++.
#[unsafe(no_mangle)]
unsafe extern "C" fn vp_supports_filter_column(bind_data: *const c_void, column: u64) -> bool {
    let bind = unsafe { &*bind_data.cast::<BindData>() };
    bind.source
        .dtype()
        .as_struct_fields_opt()
        .and_then(|fields| fields.field_by_index(usize::try_from(column).ok()?))
        .is_some_and(|dtype| supported_type(&dtype))
}

fn supported_type(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Decimal(..)
        | DType::Extension(_) => true,
        // DuckDB's NaN ordering differs from ordinary scalar comparisons. Keep float filters
        // in DuckDB until we have an explicit compatible conversion.
        DType::Primitive(p, _) => !matches!(p, PType::F16 | PType::F32 | PType::F64),
        DType::Struct(fields, _) => fields.fields().all(|d| supported_type(&d)),
        _ => false,
    }
}

/// Output columns after DuckDB removes filter-only columns from the scan result.
pub(crate) fn projection() -> Vec<u64> {
    // SAFETY: called synchronously inside the native init hook, while its input is alive.
    unsafe {
        (0..vp_projection_count())
            .map(|i| vp_projection_column(i))
            .filter(|&column| !vp_is_empty_column(column))
            .collect()
    }
}

/// Called synchronously by the native init hook. No snapshot pointer escapes this call.
pub(crate) fn current(dtype: &DType) -> Result<Option<Expression>> {
    let fields = dtype
        .as_struct_fields_opt()
        .ok_or("scan root must be a struct")?;
    let mut expressions = Vec::new();
    // SAFETY: the C++ init scope owns all nodes for the duration of this callback.
    unsafe {
        for i in 0..vp_filter_count() {
            let mut column = 0;
            let node = vp_filter_get(i, &raw mut column);
            let index = usize::try_from(column)?;
            let name = fields.names().get(index).ok_or("invalid filter column")?;
            let column_dtype = fields.field_by_index(index).ok_or("invalid filter dtype")?;
            expressions.push(convert(
                node,
                get_item(name.clone(), root()),
                &column_dtype,
            )?);
        }
    }
    Ok(and_collect(expressions))
}

fn convert(node: *const c_void, col: Expression, dtype: &DType) -> Result<Expression> {
    // SAFETY: private callers only pass nodes borrowed from the current native init scope.
    unsafe {
        Ok(match vp_filter_kind(node) {
            0 => {
                let value = scalar(vp_filter_value(node), dtype)?;
                let value = lit(value);
                match vp_filter_comparison(node) {
                    25 => eq(col, value),
                    26 => not_eq(col, value),
                    27 => lt(col, value),
                    28 => gt(col, value),
                    29 => lt_eq(col, value),
                    30 => gt_eq(col, value),
                    _ => return Err("unsupported DuckDB comparison".into()),
                }
            }
            1 => is_null(col),
            2 => is_not_null(col),
            kind @ (3 | 4) => {
                let children = (0..vp_filter_children(node))
                    .map(|i| convert(vp_filter_child(node, i), col.clone(), dtype))
                    .collect::<Result<Vec<_>>>()?;
                if kind == 3 {
                    or_collect(children).unwrap_or_else(|| lit(false))
                } else {
                    and_collect(children).unwrap_or_else(|| lit(true))
                }
            }
            5 => {
                let mut len = 0;
                let ptr = vp_filter_field(node, &raw mut len);
                let name =
                    std::str::from_utf8(std::slice::from_raw_parts(ptr, usize::try_from(len)?))?;
                let child_dtype = dtype
                    .as_struct_fields_opt()
                    .and_then(|fields| fields.field(name))
                    .ok_or("invalid struct filter field")?;
                convert(vp_filter_child(node, 0), get_item(name, col), &child_dtype)?
            }
            // Optional zonemap, bloom, and join hints may be omitted if unsupported.
            6 => convert(vp_filter_child(node, 0), col, dtype).unwrap_or_else(|_| lit(true)),
            10 => crate::bloom::expression(node, col),
            100 => lit(true),
            104 => lit(false),
            103 => {
                let count = vp_filter_children(node);
                let has_null = (0..count)
                    .any(|i| ffi::duckdb_is_null_value(vp_filter_value(vp_filter_child(node, i))));
                let element_dtype = dtype.with_nullability(Nullability::from(has_null));
                let values = (0..count)
                    .map(|i| {
                        let value =
                            scalar(vp_filter_value(vp_filter_child(node, i)), &element_dtype)?;
                        Ok(value.cast(&element_dtype)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                in_list(
                    col,
                    lit(Scalar::list(
                        element_dtype,
                        values,
                        Nullability::NonNullable,
                    )),
                )
            }
            101 => like(col, lit(scalar(vp_filter_value(node), dtype)?)),
            102 => {
                let op = match vp_filter_comparison(node) {
                    25 => CompareOperator::Eq,
                    26 => CompareOperator::NotEq,
                    27 => CompareOperator::Lt,
                    28 => CompareOperator::Gt,
                    29 => CompareOperator::Lte,
                    30 => CompareOperator::Gte,
                    _ => return Ok(lit(true)),
                };
                let data = DynamicFilter(vp_filter_dynamic(node));
                let dtype = dtype.as_nullable();
                let rhs_dtype = dtype.clone();
                vortex::expr::dynamic(op, move || data.latest(&dtype), rhs_dtype, true, col)
            }
            kind => return Err(format!("unsupported mandatory DuckDB filter {kind}").into()),
        })
    }
}

fn scalar(value: ffi::duckdb_value, dtype: &DType) -> Result<Scalar> {
    use LogicalTypeId as L;
    // SAFETY: value is borrowed from a native snapshot; its logical type determines each getter.
    unsafe {
        if ffi::duckdb_is_null_value(value) {
            return Ok(Scalar::null(dtype.as_nullable()));
        }
        let scalar = match L::from(vp_value_type(value)) {
            L::Boolean => Scalar::from(ffi::duckdb_get_bool(value)),
            L::Tinyint => Scalar::from(ffi::duckdb_get_int8(value)),
            L::Smallint => Scalar::from(ffi::duckdb_get_int16(value)),
            L::Integer => Scalar::from(ffi::duckdb_get_int32(value)),
            L::Bigint => Scalar::from(ffi::duckdb_get_int64(value)),
            L::UTinyint => Scalar::from(ffi::duckdb_get_uint8(value)),
            L::USmallint => Scalar::from(ffi::duckdb_get_uint16(value)),
            L::UInteger => Scalar::from(ffi::duckdb_get_uint32(value)),
            L::UBigint => Scalar::from(ffi::duckdb_get_uint64(value)),
            kind @ (L::Varchar | L::Blob) => {
                let mut len = 0;
                let ptr = vp_value_bytes(value, &raw mut len);
                let bytes = std::slice::from_raw_parts(ptr, usize::try_from(len)?);
                if kind == L::Varchar {
                    Scalar::utf8(
                        std::str::from_utf8(bytes)?.to_owned(),
                        Nullability::NonNullable,
                    )
                } else {
                    Scalar::binary(bytes.to_vec(), Nullability::NonNullable)
                }
            }
            L::Decimal => {
                let decimal = ffi::duckdb_get_decimal(value);
                let DType::Decimal(decimal_dtype, _) = dtype else {
                    return Err("decimal constant for non-decimal column".into());
                };
                if decimal.scale as i8 != decimal_dtype.scale() {
                    return Err("decimal filter scale mismatch".into());
                }
                let value =
                    (i128::from(decimal.value.upper) << 64) | i128::from(decimal.value.lower);
                Scalar::decimal(
                    DecimalValue::I128(value),
                    *decimal_dtype,
                    Nullability::NonNullable,
                )
            }
            kind @ (L::Date
            | L::Time
            | L::TimeNs
            | L::Timestamp
            | L::TimestampS
            | L::TimestampMs
            | L::TimestampNs
            | L::TimestampTZ) => {
                let storage = match kind {
                    L::Date => ScalarValue::from(ffi::duckdb_get_date(value).days),
                    L::Time => ScalarValue::from(ffi::duckdb_get_time(value).micros),
                    L::TimeNs => ScalarValue::from(ffi::duckdb_get_time_ns(value).nanos),
                    L::Timestamp => ScalarValue::from(ffi::duckdb_get_timestamp(value).micros),
                    L::TimestampTZ => ScalarValue::from(ffi::duckdb_get_timestamp_tz(value).micros),
                    L::TimestampS => ScalarValue::from(ffi::duckdb_get_timestamp_s(value).seconds),
                    L::TimestampMs => ScalarValue::from(ffi::duckdb_get_timestamp_ms(value).millis),
                    L::TimestampNs => ScalarValue::from(ffi::duckdb_get_timestamp_ns(value).nanos),
                    _ => unreachable!(),
                };
                Scalar::try_new(dtype.as_nonnullable(), Some(storage))?
            }
            other => return Err(format!("unsupported DuckDB filter scalar {other:?}").into()),
        };
        Ok(scalar.cast(&dtype.as_nullable())?)
    }
}
