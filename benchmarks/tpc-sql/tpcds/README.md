# TPC-DS query workload

The 99 query files in this directory are inputs for this repository's benchmark harness.
Their source is recorded in the [workload provenance](../README.md).

Supply a directory containing the 24 `<table>.parquet` files, then run from the repository root:

```sh
cargo run --release --locked -p benchmarks -- \
    --suite tpcds --file data/tpcds-sf1 --tpcds-sql-dir benchmarks/tpc-sql/tpcds \
    --arms datafusion-parquet,vortex-parquet --digests
```

These adapted queries are an experimental workload, not audited TPC benchmark results.
