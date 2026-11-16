// SPDX-License-Identifier: Apache-2.0
//! Rust callback ownership and panic containment for the native registration bridge.

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use duckdb::ffi;
use duckdb::vtab::BindInfo;
use duckdb::vtab::InitInfo;

use crate::ParquetVTab;
use crate::ReaderContext;
use crate::Result;

unsafe extern "C" {
    fn vp_register(
        database: ffi::duckdb_database,
        context: *mut c_void,
        destroy: ffi::duckdb_delete_callback_t,
        bind: ffi::duckdb_table_function_bind_t,
        init: ffi::duckdb_table_function_init_t,
        local_init: ffi::duckdb_table_function_init_t,
        scan: ffi::duckdb_table_function_t,
    ) -> bool;
}

// A panic payload can itself panic on drop. Contain destructor panics without recursively
// destroying their payloads; only the exceptional path leaks a payload.
#[expect(
    clippy::mem_forget,
    reason = "panic payload destructors must not unwind across FFI"
)]
fn discard<T>(value: T) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(value))) {
        std::mem::forget(payload);
    }
}

pub(crate) fn callback(report: impl FnOnce(&CStr), run: impl FnOnce() -> Result<()>) {
    let message = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(())) => return,
        Ok(Err(error)) => {
            let message = catch_unwind(AssertUnwindSafe(|| error.to_string()));
            discard(error);
            match message {
                Ok(message) => message,
                Err(payload) => {
                    discard(payload);
                    "Rust callback error formatting panicked".into()
                }
            }
        }
        Err(payload) => {
            discard(payload);
            "Rust callback panicked".into()
        }
    };
    match CString::new(message.replace('\0', "\\0")) {
        Ok(message) => report(&message),
        Err(_) => report(c"Rust callback failed"),
    }
}

unsafe extern "C" fn destroy<T>(data: *mut c_void) {
    // SAFETY: every caller transfers a Box<T> through the corresponding DuckDB owner.
    discard(unsafe { Box::from_raw(data.cast::<T>()) });
}

unsafe extern "C" fn bind(info: ffi::duckdb_bind_info) {
    callback(
        |error| unsafe { ffi::duckdb_bind_set_error(info, error.as_ptr()) },
        || {
            let info = BindInfo::from(info);
            let data = ParquetVTab::bind(&info)?;
            // SAFETY: DuckDB owns this box until the bound plan is released.
            unsafe {
                info.set_bind_data(
                    Box::into_raw(Box::new(data)).cast(),
                    Some(destroy::<crate::BindData>),
                );
            }
            Ok(())
        },
    );
}

unsafe extern "C" fn init(info: ffi::duckdb_init_info) {
    callback(
        |error| unsafe { ffi::duckdb_init_set_error(info, error.as_ptr()) },
        || {
            let info = InitInfo::from(info);
            let data = ParquetVTab::init(&info)?;
            // SAFETY: DuckDB owns this box until execution finishes.
            unsafe {
                info.set_init_data(
                    Box::into_raw(Box::new(data)).cast(),
                    Some(destroy::<crate::InitData>),
                );
            }
            Ok(())
        },
    );
}

// DuckDB gives each worker exclusive use of its own scan cursor.
unsafe extern "C" fn init_local(info: ffi::duckdb_init_info) {
    callback(
        |error| unsafe { ffi::duckdb_init_set_error(info, error.as_ptr()) },
        || {
            let info = InitInfo::from(info);
            // SAFETY: DuckDB takes ownership of this cursor until the worker finishes.
            unsafe {
                info.set_init_data(
                    Box::into_raw(Box::new(crate::Scan::default())).cast(),
                    Some(destroy::<crate::Scan>),
                );
            }
            Ok(())
        },
    );
}

unsafe extern "C" fn scan(info: ffi::duckdb_function_info, output: ffi::duckdb_data_chunk) {
    callback(
        |error| unsafe { ffi::duckdb_function_set_error(info, error.as_ptr()) },
        || {
            // SAFETY: DuckDB lends these handles exclusively for this callback.
            unsafe { ParquetVTab::func(info, output) }
        },
    );
}

pub(super) unsafe fn register(
    database: ffi::duckdb_database,
    context: ReaderContext,
) -> duckdb::Result<()> {
    // SAFETY: the caller supplies a live database; the native bridge takes ownership of the
    // boxed context even on failure, and preserves each callback's corresponding Rust type.
    let registered = unsafe {
        vp_register(
            database,
            Box::into_raw(Box::new(context)).cast(),
            Some(destroy::<ReaderContext>),
            Some(bind),
            Some(init),
            Some(init_local),
            Some(scan),
        )
    };
    if !registered {
        return Err(duckdb::Error::DuckDBFailure(
            ffi::Error::new(ffi::DuckDBError),
            Some("failed to register read_parquet_vortex in DuckDB".into()),
        ));
    }
    Ok(())
}
