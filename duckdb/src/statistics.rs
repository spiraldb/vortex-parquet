// SPDX-License-Identifier: Apache-2.0
//! Footer bounds exposed to DuckDB's optimizer without reading column payloads.

use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use duckdb::ffi;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::PType;
use vortex::expr::stats::Stat;
use vortex::io::runtime::BlockingRuntime;

use crate::BindData;

// SAFETY: DuckDB holds BindData alive during planning and supplies writable output pointers.
// Missing, unsupported, or failed statistics stay unknown. Never unwind across the C++ boundary.
#[unsafe(no_mangle)]
unsafe extern "C" fn vp_column_bounds(
    bind_data: *const c_void,
    column: u64,
    minimum: *mut ffi::duckdb_value,
    maximum: *mut ffi::duckdb_value,
) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let bind = unsafe { &*bind_data.cast::<BindData>() };
        let Some((min, max)) = integer_bounds(bind, column) else {
            return false;
        };
        // SAFETY: both handles are owned values; ownership transfers to the C++ caller.
        unsafe {
            *minimum = min;
            *maximum = max;
        }
        true
    }))
    .unwrap_or(false)
}

fn integer_bounds(bind: &BindData, column: u64) -> Option<(ffi::duckdb_value, ffi::duckdb_value)> {
    let fields = bind.source.dtype().as_struct_fields_opt()?;
    let column = usize::try_from(column).ok()?;
    let dtype = fields.field_by_index(column)?;
    let DType::Primitive(ptype, _) = &dtype else {
        return None;
    };
    if !ptype.is_int() {
        return None;
    }
    let name = fields.names().get(column)?.clone();
    let path = FieldPath::from_name(name);
    let mut combined: Option<(i128, i128)> = None;
    for (source, metadata) in std::iter::once(&bind.source)
        .chain(&bind.additional_sources)
        .zip(&bind.metadata)
    {
        let (low, high) = (*metadata.bounds.get(column)?.get_or_init(|| {
            let stats = bind.runtime.block_on(source.field_statistics(&path)).ok()?;
            exact_integer_bounds(&stats, &dtype)
        }))?;
        combined = Some(combined.map_or((low, high), |(min, max)| (min.min(low), max.max(high))));
    }
    let (low, high) = combined?;
    macro_rules! bounds {
        ($native:ty, $create:path) => {{
            let min = <$native>::try_from(low).ok()?;
            let max = <$native>::try_from(high).ok()?;
            // SAFETY: each C API constructor receives a value of its declared native type.
            unsafe { ($create(min), $create(max)) }
        }};
    }
    let (mut min, mut max) = match ptype {
        PType::I8 => bounds!(i8, ffi::duckdb_create_int8),
        PType::I16 => bounds!(i16, ffi::duckdb_create_int16),
        PType::I32 => bounds!(i32, ffi::duckdb_create_int32),
        PType::I64 => bounds!(i64, ffi::duckdb_create_int64),
        PType::U8 => bounds!(u8, ffi::duckdb_create_uint8),
        PType::U16 => bounds!(u16, ffi::duckdb_create_uint16),
        PType::U32 => bounds!(u32, ffi::duckdb_create_uint32),
        PType::U64 => bounds!(u64, ffi::duckdb_create_uint64),
        _ => return None,
    };
    if min.is_null() || max.is_null() {
        // SAFETY: destroy_value accepts null handles and releases each owned allocation once.
        unsafe {
            ffi::duckdb_destroy_value(&raw mut min);
            ffi::duckdb_destroy_value(&raw mut max);
        }
        return None;
    }
    Some((min, max))
}

fn exact_integer_bounds(stats: &StatsSet, dtype: &DType) -> Option<(i128, i128)> {
    macro_rules! bounds {
        ($native:ty) => {{
            let min = stats.get_as::<$native>(Stat::Min, dtype).as_exact()?;
            let max = stats.get_as::<$native>(Stat::Max, dtype).as_exact()?;
            Some((i128::from(min), i128::from(max)))
        }};
    }
    // Widen only after conversion to the actual physical integer type. Inexact or missing
    // bounds must remain unknown, including for empty files and mixed-statistics row groups.
    match dtype {
        DType::Primitive(PType::I8, _) => bounds!(i8),
        DType::Primitive(PType::I16, _) => bounds!(i16),
        DType::Primitive(PType::I32, _) => bounds!(i32),
        DType::Primitive(PType::I64, _) => bounds!(i64),
        DType::Primitive(PType::U8, _) => bounds!(u8),
        DType::Primitive(PType::U16, _) => bounds!(u16),
        DType::Primitive(PType::U32, _) => bounds!(u32),
        DType::Primitive(PType::U64, _) => bounds!(u64),
        _ => None,
    }
}
