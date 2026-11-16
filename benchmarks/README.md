# Benchmark harness

`benchmarks` is a Rust benchmark executable. It runs the same SQL through DataFusion using the native
Parquet reader, the Parquet-through-Vortex reader, and Vortex-file reference configurations.
It interleaves executions, checks answer digests, and reports timings and reader metrics.

`sql/` contains ClickBench and focused projection, filtering, and string workloads. `tpc-sql/`
contains the shared TPC-H and TPC-DS query files. The separate DuckDB benchmark runner also uses
the TPC-H files here.

Run from the repository root, supplying your own input data:

```sh
cargo run --release --locked -p benchmarks -- \
    --suite tpch --file /path/to/tpch/parquet --tpch-sql-dir benchmarks/tpc-sql/tpch \
    --arms datafusion-parquet,vortex-parquet --digests
```

See [TPC workload provenance](tpc-sql/README.md) for the query sources.

## I/O

`--io auto` uses `FileReadAt` for local files and object storage for remote paths. Configure concurrency
and coalescing with `--io-concurrency` and `--io-coalesce-max-kb`. Compare I/O configurations within one
interleaved run:

```sh
benchmarks --file data/hits.parquet --io-arms file,file:192
```

`BENCHMARKS_COALESCE_MAX_KB`, `BENCHMARKS_COALESCE_DISTANCE_KB`, and `BENCHMARKS_IO_CONCURRENCY` also configure I/O;
explicit flags take precedence. `--io object-store` delegates to the source filesystem, which uses
the same local backend at the pinned Vortex revision.

## Preflight

Before a benchmark campaign, run `benchmarks --preflight` on small inputs. It checks query completion, matching
answer digests, per-query time limits, scan parallelism, and timing stability across repeated executions.
Suites without input data are reported as `NOT CHECKED`; a pass covers only the tested inputs and machine.

```sh
benchmarks --preflight --file data/hits.parquet \
    --tpch-dir data/tpch-sf1 --tpch-sql-dir benchmarks/tpc-sql/tpch \
    --tpcds-dir data/tpcds-sf1 --tpcds-sql-dir benchmarks/tpc-sql/tpcds
```

Use `--preflight-suites tpch` to check one suite.

## Results

Checked-in performance summaries must identify the tested commit, inputs, hardware, and reproduction
commands. Keep only results representative of the current or a very recent revision; refresh or remove
stale summaries. Keep datasets, raw timing samples, and profiles outside the tracked source tree, for
example in the ignored `target/benchmarks/` directory.
