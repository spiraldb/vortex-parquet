// SPDX-License-Identifier: Apache-2.0
#include "vortex_parquet.hpp"
#include <memory>
#include <stdexcept>

extern "C" const char *vp_library_version() noexcept { return duckdb_library_version(); }

ParquetTableFunction::ParquetTableFunction() : function(duckdb_create_table_function()) {
    if (!function) throw std::runtime_error("could not allocate DuckDB table function");
}

ParquetTableFunction::~ParquetTableFunction() {
    duckdb_destroy_table_function(&function);
}

bool ParquetTableFunction::Register(duckdb_database database) {
    duckdb_connection connection = nullptr;
    if (duckdb_connect(database, &connection) != DuckDBSuccess) return false;
    const auto result = duckdb_register_table_function(connection, function);
    duckdb_disconnect(&connection);
    return result == DuckDBSuccess;
}

extern "C" bool vp_register(duckdb_database database, void *context, duckdb_delete_callback_t destroy,
                            duckdb_table_function_bind_t bind, duckdb_table_function_init_t init,
                            duckdb_table_function_init_t local_init, duckdb_table_function_t scan) noexcept {
    std::unique_ptr<void, duckdb_delete_callback_t> owner(context, destroy);
    try {
        ParquetTableFunction function;
        auto handle = function.Handle();
        duckdb_table_function_set_extra_info(handle, owner.get(), destroy);
        owner.release();
        duckdb_table_function_set_name(handle, "read_parquet_vortex");
        auto path_type = duckdb_create_logical_type(DUCKDB_TYPE_ANY);
        duckdb_table_function_add_parameter(handle, path_type);
        duckdb_destroy_logical_type(&path_type);
        duckdb_table_function_supports_projection_pushdown(handle, true);
        duckdb_table_function_set_bind(handle, bind);
        duckdb_table_function_set_init(handle, init);
        duckdb_table_function_set_local_init(handle, local_init);
        duckdb_table_function_set_function(handle, scan);
        vp_enable_filters(handle);
        return function.Register(database);
    } catch (...) {
        return false;
    }
}
