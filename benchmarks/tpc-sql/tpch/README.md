# TPC-H query workload

The 22 query files in this directory are inputs for this repository's benchmark harness.
Their source is recorded in the [workload provenance](../README.md).

Supply a directory containing the eight `<table>.parquet` files, then run from the repository root:

```sh
cargo run --release --locked -p benchmarks -- \
    --suite tpch --file data/tpch-sf1 --tpch-sql-dir benchmarks/tpc-sql/tpch \
    --arms datafusion-parquet,vortex-parquet --digests
```

These adapted queries are an experimental workload, not audited TPC benchmark results.
