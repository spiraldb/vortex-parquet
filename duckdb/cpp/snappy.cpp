// SPDX-License-Identifier: Apache-2.0
// DuckDB v2.0.0's bundled Snappy implementation, using its public raw-buffer interface.
#include "snappy.h"

extern "C" bool vp_snappy_decode(const char *input, size_t input_length,
                                  char *output, size_t output_length) noexcept {
    try {
        size_t length = 0;
        // RawUncompress requires a buffer at least as large as the Snappy header declares.
        // Check against the Parquet header before handing it the Rust allocation.
        return duckdb_snappy::GetUncompressedLength(input, input_length, &length) &&
               length == output_length &&
               duckdb_snappy::RawUncompress(input, input_length, output);
    } catch (...) {
        return false;
    }
}
