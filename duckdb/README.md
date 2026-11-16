# DuckDB adapter

`read_parquet_vortex(path)` reads a local Parquet file through this repository's Vortex reader.
It supports both embedded Rust applications and a loadable DuckDB extension. No DataFusion
dependency is needed.

## Embedded Rust

```toml
[dependencies]
duckdb = "=1.20000.2"
vortex-parquet-duckdb = { path = "path/to/vortex-parquet/duckdb", features = ["bundled"] }
```

The adapter uses the unmodified upstream Rust crate. Registration takes a borrowed
`duckdb_database` handle, so applications that own a database through DuckDB's C API can
register the function before opening SQL connections. The database must outlive those
connections; registration never closes it.

```rust
// `database` is a live DuckDB database handle owned by the application.
unsafe { vortex_parquet_duckdb::register(database)? };
let connection = unsafe { duckdb::Connection::open_from_raw(database)? };
let count: i64 = connection.query_row(
    "SELECT count(*) FROM read_parquet_vortex(?) WHERE price > 100",
    ["data/items.parquet"],
    |row| row.get(0),
)?;
```

Use `register_with_format` to supply reader options, a cache policy, or shared scan metrics.
The adapter owns its Vortex runtime; the application does not need to start Tokio.

## Native build

Following `vortex-duckdb`'s setup, `build.rs` downloads checksum-pinned upstream DuckDB
**v2.0.0** headers and Snappy sources into Cargo's build directory, then compiles our small
C++ bridge and a private copy of the native Snappy codec.
The bridge owns table-function registration and the additional vector/filter APIs. No DuckDB
engine sources or modified Rust crate are vendored. Stock DuckDB **2.0.0** works; the C++ ABI
requires this exact version.

For an existing DuckDB checkout or an offline build, set both `DUCKDB_SOURCE_DIR` to its root
and `DUCKDB_VERSION=v2.0.0`. The source headers must match the linked runtime.
For embedded builds, use `DUCKDB_DOWNLOAD_LIB=1` to download the matching prebuilt upstream
library, `DUCKDB_LIB_DIR` to use an existing library, or the `bundled` feature to build the
engine through upstream duckdb-rs. Loadable extensions use the host's library.
The private Snappy build enables its NEON byte-shuffle path on AArch64 targets that advertise
NEON support.

## Loadable extension

Build from the repository root with Rust 1.100.0 and a C++20 compiler and a DuckDB **2.0.0** CLI on `PATH`:

```sh
cargo build --locked -p vortex-parquet-duckdb --example vortex_parquet --features loadable-extension
python3 duckdb/package_extension.py target/debug/examples/libvortex_parquet.dylib
duckdb -unsigned
```

On Linux the library suffix is `.so`. The packager accepts
`--duckdb /path/to/duckdb` and queries that executable for its platform. Build for the same
platform as that executable. Use `--release` and `target/release/examples/` for optimized builds.
The `bundled` and `loadable-extension` features are alternative linking modes; build them separately.

```sql
LOAD 'target/debug/examples/vortex_parquet.duckdb_extension';
CREATE VIEW events AS SELECT * FROM read_parquet_vortex('data/events.parquet');
SELECT category, count(*) FROM events WHERE price > 100 GROUP BY category;
```

The extension uses duckdb-rs's stable V1 C API (v1.5.6), which DuckDB 2.0 still provides.
The C++ bridge requires exactly DuckDB 2.0.0; the entrypoint checks the host version before
initializing Rust's API table or touching C++ objects. Its metadata follows
the [official Rust extension template](https://github.com/duckdb/extension-template-rs).
Local builds are unsigned. When upgrading DuckDB, update the source version and checksum in
`build.rs`, Cargo pin, runtime check, entrypoint, and packaging version together, then rerun
both embedded and loadable-extension tests.

Flat Snappy pages use DuckDB's upstream native Snappy decoder through a checked C++ shim.
DuckDB 2.0 does not export this codec from its shared library, so the build compiles the
downloaded sources in a private namespace without depending on the host's Parquet extension.
The packager copies the root `LICENSE` and `THIRD_PARTY.md` alongside the extension;
include the project license and third-party notices when redistributing binaries.
The shim checks the Snappy header's decoded length against the Parquet page's output allocation
before decoding. Vortex still reads, filters, and exports the pages. Other codecs and nested
Arrow fallback decoding use parquet-rs. The core reader defaults to parquet-rs and exposes
`ReaderOptions::with_snappy_decoder` for hosts supplying their own decoder. The additional
`with_snappy_append_decoder` callback lets a host append decoded bytes without first zeroing
the output allocation. DuckDB registration installs this append decoder while retaining the
supplied format's other options.

## Execution and scope

Binding reads the footer and reports the schema and exact cardinality. DuckDB workers claim row-group
partitions independently and each retains one Vortex batch. Projection is pushed into the reader.
Projected expressions and unlimited scan plans are prepared once per file and shared across its
row groups, so layout readers can reuse expression analysis throughout one execution.
Complete, exact integer bounds from the footer are exposed to DuckDB's optimizer, enabling
integer-key compression and related planning decisions. Missing or unsupported bounds remain unknown.
The registered function shares a Vortex session and I/O runtime across file binds; each execution
still creates its own scan cursors and filter state. On Unix, a registration-scoped cache retains
validated footers, converted schemas, and lazily computed exact integer bounds, bounded to 128
entries and a 64 MiB estimated memory charge including schema and bound storage. Each bind opens the
file and checks its device, inode, size, modification time, and change time before reuse. Other
platforms bypass this cache. Cached entries retain no file handles or decoded column data. Bounds
use the same file-version key as the footer, and multi-file scans merge every file's bounds;
any missing bound keeps the combined range unknown.
Each new file version validates all footer ranges; unchanged versions reuse that validation and
the converted Arrow/Vortex schema. Scan layouts are built only after projection and filter
columns are known. Filter-only fields and intermediate projection scopes remain available. Footer
counts need no scan layouts; `open_file` still exposes the complete physical hierarchy.
The exporter follows [`vortex-duckdb`](https://github.com/vortex-data/vortex/tree/ead3ef34728e467642e5b149db3e69767e1d5ac1/vortex-duckdb/src/exporter):
primitive buffers and long-string payloads are referenced through DuckDB-owned shared buffers;
constants, dictionaries, and chunk boundaries retain their physical structure. Decimal widths are
adapted to DuckDB's storage and checked against the declared precision. Lazy arrays execute only
until a supported export representation is available. Dictionaries carry DuckDB's reusable identity
and size so downstream operators can cache their hashes. Pages sharing dictionary values reuse
one converted DuckDB dictionary within an output batch; the cache retains the source array while
using its address as a key, and exported vectors retain their own buffer ownership. No Arrow array is constructed
on the export path. DuckDB's Rust wrapper still has an unconditional Arrow 60 dependency.

For PLAIN numeric pages without nulls on little-endian hosts, the reader borrows the physical
integer, floating-point, or decimal buffer directly when aligned. Unaligned values require one
alignment copy. Payload sizes are checked before construction; other pages keep their decoders.

PLAIN strings borrow Vortex varbinview's bulk UTF-8 validation approach. Contiguous runs are checked
with `simdutf8`: ASCII length prefixes can remain between strings, while non-ASCII prefixes split
the runs and are excluded. These separators enforce individual string boundaries without copying
payloads. Successful UTF-8 validation and decoded views remain with the immutable page for reuse;
binary data is not marked as validated UTF-8 until it has passed the UTF-8 check.

Dropping a scan after `LIMIT` releases pending work; buffers referenced by downstream DuckDB operators
remain alive until those operators release them. An empty virtual column lets unfiltered `count(*)`
use footer row counts without reading any column payload. Filtered counts read their predicate
columns and export only row cardinality.

SQL filters, joins, and aggregates work normally. The C++ shim forwards native comparison, null,
IN, conjunction, and nested-field filters into Vortex. Dynamic bounds retain a shared DuckDB handle;
Vortex reads their latest values so later row groups can use tighter Top-N bounds.
Uninitialized bounds do not exclude rows. Float predicates retain DuckDB's NaN semantics by staying
in DuckDB, along with unsupported expressions. Filter-only columns are removed from output.
DuckDB's optional Bloom and prefix-range join filters are probed on integer keys within the live
scan callback. Unsupported key types or casts and null keys are conservatively admitted; DuckDB
still evaluates the join. Each execution captures its own filter bindings, so prepared-statement
reuse and early termination cannot reuse a previous join's key set.
Dictionary probes are reused across pages within an input batch. Samples over exact footer bounds
estimate prefix-range selectivity: the scan keeps the most selective
known hint and omits broad hints, while retaining hints with unknown estimates. These estimates
only choose work to skip; the join still checks every key.
String `contains`, `prefix`, and `suffix` predicates on a plain column with a constant argument
are also pushed down, including simple `LIKE` patterns that DuckDB rewrites into these functions.
Literal pattern characters are escaped; collated columns and other string expressions stay in DuckDB.
Paths are individual local files or nonempty lists of local files with identical schemas. A file
list shares one scan and worker pool, preserves repeated paths, and combines exact cardinalities
and integer bounds across files. Globbing, remote URLs, and schema union are not implemented. The C++ shim
requires the matching DuckDB 2.0.0 C++ symbols; registration checks the runtime version. The macOS
embedded library, CLI, and Python package are tested here.

## Validation

```sh
cargo test --locked -p vortex-parquet-duckdb --features bundled
cargo clippy --locked -p vortex-parquet-duckdb --all-targets --features bundled -- -D warnings
python3 duckdb/test_extension.py target/debug/examples/vortex_parquet.duckdb_extension
```

For a prebuilt DuckDB shared library, omit `--features bundled` and set `DUCKDB_LIB_DIR` to its
directory. Set `DYLD_LIBRARY_PATH` on macOS or `LD_LIBRARY_PATH` on Linux to the same directory
when running tests. Embedded SQL tests cover values across vector and row-group boundaries,
projection, filters, joins, prepared-statement reuse, nulls and nested/logical types, empty files,
errors, and early termination. The separate CLI test verifies the packaged extension's loading
and ABI boundary.

## Benchmarking

**We must beat DuckDB's `read_parquet` on every TPC-H and ClickBench query.** Acceptance
requires **22/22 TPC-H wins and 43/43 ClickBench wins**, with correct answers and a median
`read_parquet_vortex / read_parquet` time ratio strictly below 1 for each query. Both readers
must scan the same complete Parquet inputs through their file-list interfaces. The baseline is
DuckDB's `read_parquet([...])`; DuckDB tables loaded into its internal storage are outside this
comparison. **Any incorrect answer, tie, or loss fails acceptance.** Geometric means and summed
medians are diagnostics; they cannot compensate for a losing query. Selected queries and reduced
input preflights do not establish acceptance.

`benchmark.py` rotates execution order, checks every returned answer, and records all samples,
input metadata, SQL, and the extension hash. Timings include planning, execution, and fetching
results. Both arms use file-list scans over the same Parquet inputs.

`suite_passed` requires complete input and query coverage, correct answers, and a win on every
query. The harness exits with status 1 for missing coverage, incorrect answers, execution errors,
invalid timings, ties, or losses. Selected queries and `--max-files` runs are diagnostics and
cannot pass the suite gate. Run `python duckdb/test_benchmark.py` to check the acceptance gate
with controlled timing and answer inputs.

LIMIT queries receive deterministic tie breakers, and ClickBench byte-length calls use DuckDB's
`STRLEN`; these are adapted suites, not official benchmark scores.

Install Python `duckdb==2.0.0` and build/package the release extension above, then run:

```sh
python duckdb/benchmark.py --extension target/release/examples/vortex_parquet.duckdb_extension \
    --suite tpch --data /path/to/tpch/sf1/parquet --warmup 2 --repeat 20 \
    --output target/benchmarks/tpch-results.jsonl
python duckdb/benchmark.py --extension target/release/examples/vortex_parquet.duckdb_extension \
    --suite clickbench --data /path/to/clickbench/parquet --warmup 1 --repeat 10 \
    --output target/benchmarks/clickbench-results.jsonl
```

Output files must not already exist. TPC-H requires all eight tables; ClickBench requires all
100 shards. Use `--max-files 1` only for a ClickBench preflight. The output directory above is
ignored by Git.
