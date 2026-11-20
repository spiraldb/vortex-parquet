# Development

```sh
cargo test --locked --workspace --features vortex-parquet-duckdb/bundled
cargo clippy --locked --workspace --all-targets --features vortex-parquet-duckdb/bundled -- -D warnings
cargo +nightly-2026-11-13 fmt --all
```

Vortex is pinned to commit `a11813abbc2b16ce1bf01395f5eeb460555e9e06` (the scan I/O segment-request
branch, which adds the V2 scan's `ScanFile` and placed segment reads), using Rust 1.100.0.
The DataFusion adapter comes from the same upstream revision, using DataFusion 55 and Arrow 59.
Parquet's experimental page API stays on 58.4; its nested fallback transfers buffers to Vortex's
Arrow 59 through the Arrow C Data Interface. The DuckDB exporter consumes Vortex arrays directly.
The `bundled` feature builds DuckDB for the adapter tests. It can be omitted when linking against
a prebuilt DuckDB library; see the [adapter instructions](../duckdb/README.md#validation).
