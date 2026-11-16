// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "duckdb.h"

// Owns only the adapter's table-function descriptor, never the host database.
class ParquetTableFunction final {
public:
    ParquetTableFunction();
    ~ParquetTableFunction();
    ParquetTableFunction(const ParquetTableFunction &) = delete;
    ParquetTableFunction &operator=(const ParquetTableFunction &) = delete;

    duckdb_table_function Handle() const { return function; }
    bool Register(duckdb_database database);

private:
    duckdb_table_function function;
};

extern "C" {
// Stable host symbol, usable before initializing the Rust C API dispatch table.
const char *vp_library_version() noexcept;
void vp_enable_filters(duckdb_table_function function);

// Takes ownership of context on both success and failure. The host retains it after registration.
bool vp_register(duckdb_database database, void *context, duckdb_delete_callback_t destroy,
                 duckdb_table_function_bind_t bind, duckdb_table_function_init_t init,
                 duckdb_table_function_init_t local_init, duckdb_table_function_t scan) noexcept;
}
