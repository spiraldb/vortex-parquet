// SPDX-License-Identifier: Apache-2.0
//! Loadable DuckDB extension; see duckdb/README.md for packaging and loading.

/// # Safety
/// Called only by DuckDB's loader with its live extension-access table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_parquet_init_c_api(
    info: duckdb::ffi::duckdb_extension_info,
    access: *const duckdb::ffi::duckdb_extension_access,
) -> bool {
    // SAFETY: forward the host-provided handles; the adapter contains callback panics.
    unsafe { vortex_parquet_duckdb::extension_init(info, access) }
}
