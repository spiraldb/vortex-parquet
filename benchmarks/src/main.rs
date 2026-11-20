// SPDX-License-Identifier: Apache-2.0

//! Measure where the Parquet-to-Vortex boundary should be drawn.
//!
//! ```sh
//! benchmarks --file data/hits.parquet
//! benchmarks --file data/hits.parquet --arms datafusion-parquet,vortex-parquet --queries 0,1,6
//! ```
//!
//! See [`arm`] for the ladder of arms and what each one isolates.

#![deny(missing_docs)]

mod arm;
mod digest;
mod file_stats;
mod io;
mod preflight;
mod run;
mod source;
mod suite;

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use futures::TryStreamExt;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::arrow::ExecuteArrowExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::layout::scan::v2::ScanFile;
use vortex::scan::execution::driver::ScanDriver;
use vortex::session::VortexSession;
use vortex::utils::parallelism::get_available_parallelism;

use crate::arm::ArmRef;
use crate::arm::DataFusionParquet;
use crate::arm::VortexListing;
use crate::arm::VortexParquet;
use crate::arm::VortexScanApi;
use crate::io::Backend;
use crate::io::IoConfig;
use crate::run::CacheState;
use crate::run::Iterations;
use crate::source::Source;

/// What state the caches are in when a timed iteration begins.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum CacheArg {
    /// Register once, keep everything.
    Warm,
    /// Drop the arm's caches before each iteration. The honest default for comparing readers.
    #[default]
    Flushed,
    /// Re-register everything before each iteration.
    PureCold,
    /// Re-register everything **and** evict the data file from the OS page cache.
    ColdDisk,
}

/// Query workload to run.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum SuiteArg {
    /// ClickBench over one `hits` file.
    #[default]
    Clickbench,
    /// TPC-H over the eight `<table>.parquet` files in `--file`.
    Tpch,
    /// TPC-DS over the 24 `<table>.parquet` files in `--file`.
    Tpcds,
    /// Filter selectivity and row-group prunability over one `hits` file, varied independently.
    Selectivity,
    /// Projection width over one `hits` file, 1 to 105 columns.
    Projection,
    /// String and dictionary work over one `hits` file.
    Strings,
}

impl From<CacheArg> for CacheState {
    fn from(arg: CacheArg) -> Self {
        match arg {
            CacheArg::Warm => CacheState::Warm,
            CacheArg::Flushed => CacheState::Flushed,
            CacheArg::PureCold => CacheState::PureCold,
            CacheArg::ColdDisk => CacheState::ColdDisk,
        }
    }
}

impl std::fmt::Display for CacheArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Printed in every report header because cache state materially changes a reader comparison.
        let name = match self {
            CacheArg::Warm => "warm",
            CacheArg::Flushed => "flushed",
            CacheArg::PureCold => "pure-cold",
            CacheArg::ColdDisk => "cold-disk",
        };
        f.write_str(name)
    }
}

#[derive(Parser, Debug)]
#[command(about = "Measure where the Parquet-to-Vortex boundary should be drawn.")]
struct Args {
    /// ClickBench Parquet file, or a directory of benchmark `<table>.parquet` files.
    #[arg(long)]
    file: String,

    /// Query workload.
    #[arg(long, value_enum, default_value_t = SuiteArg::Clickbench)]
    suite: SuiteArg,

    /// Directory containing `q1.sql` through `q22.sql`, required by `--suite tpch`.
    #[arg(long)]
    tpch_sql_dir: Option<PathBuf>,

    /// Directory containing `01.sql` through `99.sql`, required by `--suite tpcds`.
    #[arg(long)]
    tpcds_sql_dir: Option<PathBuf>,

    /// Which arms to run, comma-separated. Defaults to all of them.
    ///
    /// Order here does *not* matter: the selection is filtered in the ladder's own order, so the
    /// baseline every geomean is taken against is the leftmost *known* arm among those named, not the
    /// one typed first. That keeps two runs naming the same set comparable however they were typed.
    #[arg(long, value_delimiter = ',')]
    arms: Option<Vec<String>>,

    /// Which queries to run, comma-separated by index. Defaults to all of them.
    #[arg(long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    /// Iterations whose timings are discarded.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Iterations whose timings are kept; the best is reported.
    #[arg(long, default_value_t = 3)]
    measured: usize,

    /// What state the caches are in when a timed iteration begins.
    ///
    /// Defaults to `flushed`, which is the state a *reader* comparison should use: the Vortex arms
    /// hold a decode cache and the baseline holds no equivalent, so `warm` lets them answer later
    /// queries from columns decoded during earlier ones. That is a real configuration but not a
    /// reader measurement.
    #[arg(long, value_enum, default_value_t = CacheArg::Flushed)]
    cache: CacheArg,

    /// Print every query's answer digest, for comparing runs by hand.
    #[arg(long)]
    digests: bool,

    /// Write every individual timing sample to this path as JSON Lines.
    ///
    /// The report reduces each cell to its best sample and two summary lines; everything else is
    /// computed and dropped. That is fine for reading a result and wrong for keeping one. A run whose
    /// effect sits inside the noise floor is not worthless -- it is a distribution, and pooling it with
    /// later runs, or recomputing a median or a quantile, needs the samples that produced it.
    ///
    /// One JSON object per line: a `run` header carrying provenance, then one `sample` per
    /// arm/query/iteration.
    #[arg(long, value_name = "PATH")]
    samples_out: Option<PathBuf>,

    /// Which reader backend to open data files through.
    ///
    /// `auto` opens local paths directly and remote paths through object storage. The legacy
    /// `object-store` option delegates to the source filesystem, which also uses a file reader for
    /// local paths at the pinned Vortex revision.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    io: Backend,

    /// Compare I/O configurations **as arms in one run**, comma-separated `backend[:concurrency]`.
    ///
    /// For example, `--io-arms file,file:192` compares two concurrency settings alongside the baseline.
    /// Arms are interleaved and rotated per query to reduce the effect of machine drift.
    #[arg(long, value_delimiter = ',', value_name = "SPECS")]
    io_arms: Option<Vec<String>>,

    /// Maximum in-flight read requests. Defaults to the backend's own (32 local, 192 object store).
    ///
    /// This bounds request concurrency for each opened reader.
    #[arg(long, value_name = "N")]
    io_concurrency: Option<usize>,

    /// Maximum coalesced span, in KiB. Defaults to the backend's own (4 MiB local, 16 MiB object).
    #[arg(long, value_name = "KIB")]
    io_coalesce_max_kb: Option<u64>,

    /// Maximum gap bridged when coalescing, in KiB. Defaults to the backend's own (1 MiB either way).
    #[arg(long, value_name = "KIB")]
    io_coalesce_distance_kb: Option<u64>,

    /// Rewrite the Parquet file with a different compression codec, then exit.
    ///
    /// Writes `<file>.<codec>.parquet` holding the same rows. Exists to test how much of the gap to
    /// the native Vortex format is the *codec* rather than the layout — the codec is a writer choice,
    /// not a property of Parquet.
    #[arg(long, value_name = "none|snappy|zstd")]
    recompress: Option<String>,

    /// Run the preflight gate over every available suite and exit non-zero unless it passes.
    ///
    /// The check a campaign must pass before it is allowed to cost anything. `--file` supplies the
    /// ClickBench file as usual; `--tpch-dir` and `--tpcds-dir` add the TPC suites if they are
    /// available. Whatever is not supplied is **named as skipped** rather than quietly omitted — a
    /// gate that covered half the suites and said "passed" is the failure it exists to prevent.
    #[arg(long)]
    preflight: bool,

    /// Restrict `--preflight` to these suites, comma-separated. Defaults to every one available.
    ///
    /// For iterating on one suite. A full gate is the thing that clears a campaign, so anything left
    /// out is reported as NOT CHECKED alongside the verdict rather than quietly dropped.
    #[arg(long, value_delimiter = ',', value_name = "NAMES")]
    preflight_suites: Option<Vec<String>>,

    /// TPC-H data directory, for `--preflight`. Skipped, loudly, if absent.
    #[arg(long, value_name = "DIR")]
    tpch_dir: Option<PathBuf>,

    /// TPC-DS data directory, for `--preflight`. Skipped, loudly, if absent.
    #[arg(long, value_name = "DIR")]
    tpcds_dir: Option<PathBuf>,

    /// Per-query wall-clock ceiling in seconds under `--preflight`.
    ///
    /// The check that turns a hang into a named failure instead of a lost run. Generous by default:
    /// at gate scale nothing healthy approaches it, and a tight value would fail slow machines rather
    /// than broken arms.
    #[arg(long, value_name = "SECONDS", default_value_t = 60)]
    preflight_budget: u64,

    /// How many times the baseline an arm may take on one query under `--preflight`.
    #[arg(long, value_name = "N", default_value_t = 20.0)]
    preflight_ratio: f64,

    /// Write `<file>.vortex` holding the same rows, then exit.
    ///
    /// The reference arms read that file. It is produced by reading the Parquet through this crate's
    /// own reader and writing the resulting stream out as Vortex, so the two files hold the same rows
    /// in the same order by construction.
    #[arg(long)]
    convert: bool,
}

/// The suites the gate can run, and the reason for each one it cannot.
///
/// Returns the cases alongside a list of skips **in the same call**, because the two must be reported
/// together. A gate that ran ClickBench, found nothing, and printed "PASSED" without mentioning that
/// TPC-DS never ran would be actively misleading — and TPC-DS is where the hang that cost two
/// campaigns lives.
fn preflight_cases(args: &Args, source: &Source) -> Result<(Vec<preflight::Case>, Vec<String>)> {
    const KNOWN: [&str; 6] = [
        "clickbench",
        "selectivity",
        "projection",
        "strings",
        "tpch",
        "tpcds",
    ];
    if let Some(requested) = &args.preflight_suites
        && let Some(bad) = requested
            .iter()
            .find(|name| !KNOWN.contains(&name.as_str()))
    {
        bail!(
            "unknown preflight suite {bad:?}. Known suites: {}",
            KNOWN.join(", ")
        );
    }
    let wanted = |name: &str| {
        args.preflight_suites
            .as_ref()
            .is_none_or(|requested| requested.iter().any(|r| r == name))
    };

    let mut cases = Vec::new();
    let mut skipped = Vec::new();
    for name in KNOWN {
        if !wanted(name) {
            skipped.push(format!("{name}: not in --preflight-suites; NOT CHECKED"));
        }
    }

    // The four single-file suites all read the ClickBench `hits` file the run was pointed at.
    let hits = vec![("hits".to_string(), source.clone())];
    for (name, built) in [
        ("clickbench", Ok(suite::clickbench())),
        ("selectivity", suite::selectivity()),
        ("projection", suite::projection()),
        ("strings", suite::strings()),
    ] {
        if !wanted(name) {
            continue;
        }
        match built {
            Ok(suite) => cases.push(preflight::Case {
                suite,
                sources: hits.clone(),
            }),
            Err(e) => skipped.push(format!("{name}: {e}")),
        }
    }

    // A TPC suite needs both its data directory and its SQL, and is skipped by name if either is
    // missing rather than silently dropped.
    let mut tpc = |name: &str,
                   data: Option<&PathBuf>,
                   sql: Option<&PathBuf>,
                   data_flag: &str,
                   sql_flag: &str,
                   build: fn(&Path) -> Result<suite::Suite>| {
        if !wanted(name) {
            return Ok(());
        }
        let (Some(data), Some(sql)) = (data, sql) else {
            skipped.push(format!(
                "{name}: needs both {data_flag} and {sql_flag}; NOT CHECKED"
            ));
            return Ok(());
        };
        if !data.is_dir() {
            skipped.push(format!(
                "{name}: {} is not a directory; NOT CHECKED",
                data.display()
            ));
            return Ok(());
        }
        let suite = build(sql)?;
        let sources = suite
            .tables
            .iter()
            .map(|table| {
                (
                    table.clone(),
                    Source::Local(data.join(format!("{table}.parquet"))),
                )
            })
            .collect();
        cases.push(preflight::Case { suite, sources });
        Ok::<_, anyhow::Error>(())
    };
    tpc(
        "tpch",
        args.tpch_dir.as_ref(),
        args.tpch_sql_dir.as_ref(),
        "--tpch-dir",
        "--tpch-sql-dir",
        suite::tpch,
    )?;
    tpc(
        "tpcds",
        args.tpcds_dir.as_ref(),
        args.tpcds_sql_dir.as_ref(),
        "--tpcds-dir",
        "--tpcds-sql-dir",
        suite::tpcds,
    )?;

    Ok((cases, skipped))
}

/// Every arm the benchmark harness knows about, in ladder order.
///
/// Order is load-bearing: the first one selected is the baseline every geomean is taken against.
fn all_arms(io: IoConfig) -> Vec<ArmRef> {
    vec![
        Arc::new(DataFusionParquet),
        Arc::new(VortexParquet::with_io(io)),
        Arc::new(VortexScanApi::with_io(io)),
        // Not given the configuration: DataFusion opens this one itself, through the object store it
        // was registered with, so the backend choice cannot reach it. Said here rather than left to
        // be inferred from a missing argument -- it means a `--io` sweep moves three arms and not
        // this one, which is exactly the kind of asymmetry that silently unbalances a comparison.
        Arc::new(VortexListing),
    ]
}

/// Parses one `--io-arms` spec, `backend[:concurrency]`, against `base`.
///
/// Everything not named in the spec is inherited from the run's own `--io*` flags, so a comparison
/// varies the one thing it names and nothing else.
fn parse_io_arm(spec: &str, base: IoConfig) -> Result<(String, IoConfig)> {
    let (backend_name, concurrency) = match spec.split_once(':') {
        Some((backend, concurrency)) => {
            let parsed: usize = concurrency.parse().with_context(|| {
                format!("{concurrency:?} in --io-arms {spec:?} is not a number")
            })?;
            (backend, Some(parsed))
        }
        None => (spec, base.concurrency),
    };
    let backend = Backend::from_str(backend_name, true)
        .map_err(|e| anyhow::anyhow!("--io-arms {spec:?}: {e}"))?;
    if concurrency == Some(0) {
        bail!("--io-arms {spec:?}: a concurrency of 0 would stall every read");
    }

    // The label carries the whole configuration, because it becomes the report's column heading and a
    // column headed `vortex-parquet` twice is unreadable.
    let label = match concurrency {
        Some(concurrency) => format!("vp-{backend}:{concurrency}"),
        None => format!("vp-{backend}"),
    };
    Ok((
        label,
        IoConfig {
            backend,
            concurrency,
            ..base
        },
    ))
}

/// The arms for an `--io-arms` comparison: the baseline, then one reader per configuration.
///
/// Only `vortex-parquet` is repeated. The baseline reads through DataFusion's own object store and
/// `vortex-listing` through DataFusion's listing table, so neither observes these settings — running
/// either one twice would produce two columns that differ only by noise and invite reading that noise
/// as an effect.
fn io_comparison_arms(specs: &[String], base: IoConfig) -> Result<Vec<ArmRef>> {
    let mut arms: Vec<ArmRef> = vec![Arc::new(DataFusionParquet)];
    let mut seen = Vec::new();
    for spec in specs {
        let (label, io) = parse_io_arm(spec, base)?;
        if seen.contains(&label) {
            bail!("--io-arms names {spec:?} twice; each configuration is one arm");
        }
        seen.push(label.clone());
        arms.push(Arc::new(VortexParquet::new(
            vortex_parquet::ParquetFormat::new(),
            label,
            io,
        )));
    }
    Ok(arms)
}

/// Resolves `--arms` against the known arms, preserving ladder order rather than argument order.
fn select_arms(requested: Option<&[String]>, io: IoConfig) -> Result<Vec<ArmRef>> {
    let all = all_arms(io);
    let Some(requested) = requested else {
        return Ok(all);
    };

    let known: Vec<&str> = all.iter().map(|a| a.name()).collect();
    for name in requested {
        if !known.contains(&name.as_str()) {
            bail!("unknown arm {name:?}. Known arms: {}", known.join(", "));
        }
    }
    // Filtered rather than mapped, so the ladder's order survives whatever order they were typed in
    // and the baseline is always the leftmost *known* arm rather than the first one named.
    Ok(all
        .into_iter()
        .filter(|a| requested.iter().any(|name| name == a.name()))
        .collect())
}

/// The tables a suite reads, for conversion.
///
/// Derived from the suite rather than from whatever `.parquet` files happen to be in the directory: a
/// stray file should not be converted, and a *missing* one should be an error here rather than a
/// registration failure several minutes into a run.
fn suite_tables(suite: SuiteArg, args: &Args) -> Result<Vec<String>> {
    let names = match suite {
        SuiteArg::Clickbench | SuiteArg::Selectivity | SuiteArg::Projection | SuiteArg::Strings => {
            vec!["hits".to_string()]
        }
        SuiteArg::Tpch => {
            suite::tpch(
                args.tpch_sql_dir
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("--suite tpch requires --tpch-sql-dir"))?,
            )?
            .tables
        }
        SuiteArg::Tpcds => {
            suite::tpcds(
                args.tpcds_sql_dir
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("--suite tpcds requires --tpcds-sql-dir"))?,
            )?
            .tables
        }
    };
    Ok(names)
}

/// Converts every `<table>.parquet` in `directory` to a `.vortex` sibling.
///
/// Sequential on purpose: each conversion streams a whole table through the reader, and running
/// several at once would compete for the same memory and I/O for no wall-clock gain in a setup step.
async fn convert_directory(directory: &Path, tables: &[String]) -> Result<()> {
    for table in tables {
        let parquet = directory.join(format!("{table}.parquet"));
        if !parquet.exists() {
            bail!("table {table} is missing: expected {}", parquet.display());
        }
        println!("converting {table} ...");
        convert(&parquet).await?;
    }
    println!("converted {} table(s)", tables.len());
    Ok(())
}

/// Writes every timing sample as JSON Lines.
///
/// Hand-rolled rather than via serde: the shape is four scalars and a label, and a schema this small
/// is easier to read in the file than in a derive. Strings are escaped for the two characters that can
/// appear in a SQL label or an arm name and break JSON.
fn write_samples(
    path: &Path,
    suite: &suite::Suite,
    results: &[run::ArmResults],
    args: &Args,
    source: &Source,
) -> Result<()> {
    use std::io::Write as _;

    fn escape(raw: &str) -> String {
        raw.replace('\\', "\\\\").replace('"', "\\\"")
    }

    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    // A header line, so a sample file is self-describing: pooling two runs later is only sound if you
    // can still tell which machine, file, cache state and iteration count produced each one.
    writeln!(
        out,
        r#"{{"kind":"run","suite":"{}","source":"{}","cache":"{}","warmup":{},"measured":{},"utc_epoch_s":{}}}"#,
        escape(suite.name),
        escape(&source.to_string()),
        args.cache,
        args.warmup,
        args.measured,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )?;

    for arm in results {
        for query in &arm.queries {
            let label = suite
                .labels
                .get(query.query)
                .cloned()
                .unwrap_or_else(|| format!("q{}", query.query));
            // One ANSWER record per arm/query, beside the timings.
            //
            // A timing without the answer that produced it is not evidence of anything: an arm can be
            // fast because it is wrong, and that is precisely the failure this benchmark harness exists to catch.
            // The digest is the reduction of the values returned, so two arms agreeing here is a real
            // cross-implementation check against arrow-rs rather than a row count that a scalar
            // aggregate would pass trivially. `null` means the result held a type the digest cannot
            // render, so that query is checked on its row count ALONE -- recorded rather than hidden,
            // because an unchecked query that looks checked is worse than one that admits it.
            writeln!(
                out,
                r#"{{"kind":"answer","arm":"{}","query":{},"label":"{}","rows":{},"digest":{},"unstable":{},"failed":{}}}"#,
                escape(&arm.arm),
                query.query,
                escape(&label),
                query
                    .rows
                    .map_or_else(|| "null".to_string(), |rows| rows.to_string()),
                query
                    .digest
                    .map_or_else(|| "null".to_string(), |d| format!("\"{d}\"")),
                query.unstable,
                query.error.is_some(),
            )?;

            // A failed query still gets a line, with no samples: an absent row and a zero-sample row
            // mean different things, and only one of them is a measurement.
            if query.samples.is_empty() {
                writeln!(
                    out,
                    r#"{{"kind":"sample","arm":"{}","query":{},"label":"{}","iteration":null,"ns":null,"failed":true}}"#,
                    escape(&arm.arm),
                    query.query,
                    escape(&label),
                )?;
                continue;
            }
            for (iteration, sample) in query.samples.iter().enumerate() {
                writeln!(
                    out,
                    r#"{{"kind":"sample","arm":"{}","query":{},"label":"{}","iteration":{},"ns":{},"rows":{},"unstable":{},"failed":false}}"#,
                    escape(&arm.arm),
                    query.query,
                    escape(&label),
                    iteration,
                    sample.as_nanos(),
                    query
                        .rows
                        .map_or_else(|| "null".to_string(), |rows| rows.to_string()),
                    query.unstable,
                )?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Where a conversion writes before it is complete.
///
/// A sibling of the destination rather than a temporary directory, so the rename that publishes it
/// cannot cross a filesystem boundary and stop being atomic.
fn partial_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    out.with_file_name(name)
}

/// A one-line description of the tuning knobs in effect, unset ones included.
///
/// Printed with every run because a sweep produces one report per grid cell, and a report that does
/// not say which cell it is cannot be filed against the run that produced it. Unset knobs are listed
/// as `default` rather than omitted, so a header carries the same fields whether or not anything was
/// swept — a reader can tell "the knob was at its default" from "this run predates the knob".
fn knobs() -> Result<String> {
    let mut parts: Vec<String> = [
        run::BATCH_SIZE_ENV,
        run::DF_CONFIG_ENV,
        io::COALESCE_DISTANCE_ENV,
        io::COALESCE_MAX_ENV,
        vortex_parquet::options::SELECTIVE_DECODE_THRESHOLD_ENV,
    ]
    .iter()
    .map(|name| {
        let value = std::env::var(name).unwrap_or_else(|_| "default".to_string());
        format!("{name}={value}")
    })
    .collect();

    // Resolved rather than echoed, unlike the others. This one's default is *not* DataFusion's, so
    // printing `default` would leave the reader unable to tell which baseline was measured — and it
    // is the setting most likely to make two runs incomparable.
    parts.insert(
        1,
        format!(
            "{}={}",
            run::PUSHDOWN_ENV,
            if run::pushdown_filters()? {
                "on"
            } else {
                "off"
            }
        ),
    );
    Ok(parts.join(" "))
}

/// Bytes read and read syscalls issued so far by this process, from `/proc/self/io`.
///
/// `rchar` and `syscr` — what the *process* asked the kernel for, which is the quantity the coalescing
/// configuration decides. Deliberately not `read_bytes`, which counts what reached the block device
/// and so reports zero for a file already in the page cache, hiding over-reading from a warm cache.
///
/// `None` off Linux, and `None` rather than an error if the file is unreadable — this is auditing, and
/// a run that cannot audit itself should say so rather than fail.
#[cfg(target_os = "linux")]
fn proc_io() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/self/io").ok()?;
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
    };
    Some((field("rchar:")?, field("syscr:")?))
}

/// Always `None`: `/proc/self/io` is Linux-only, so this accounting is unavailable elsewhere.
#[cfg(not(target_os = "linux"))]
fn proc_io() -> Option<(u64, u64)> {
    None
}

/// Renders the I/O a run performed, or `None` if either sample is missing.
///
/// Whole-process and whole-run: every arm, every query, and the warmup iterations too. That makes it
/// comparable **between runs of the same suite** — which is what a coalescing sweep needs — and not
/// comparable to a single arm's per-query figure. Said here because the number is otherwise easy to
/// quote as if it were per-query.
fn io_delta(before: Option<(u64, u64)>, after: Option<(u64, u64)>) -> Option<String> {
    let ((rchar_before, syscr_before), (rchar_after, syscr_after)) = (before?, after?);
    let bytes = rchar_after.saturating_sub(rchar_before);
    let reads = syscr_after.saturating_sub(syscr_before);
    let average = bytes.checked_div(reads).unwrap_or(0);
    Some(format!(
        "{:.2} GB in {reads} reads (~{} KiB each)",
        bytes as f64 / 1e9,
        average / (1 << 10)
    ))
}

/// A one-line description of the file's shape.
///
/// Printed with every run because **row-group count is the setting most likely to silently
/// invalidate a comparison**: with too few row groups the engines cannot parallelize, and what the
/// numbers then compare is their scheduling of a handful of large units rather than their readers. A
/// run whose header says `3 row groups` should not be read as a reader benchmark at all.
async fn describe(source: &Source) -> Result<String> {
    let session = VortexSession::default().with_tokio();
    let fs = source.filesystem(&session)?;
    let path = source.open_path()?;
    let handle = vortex_parquet::file::file_handle(&path, fs.as_ref(), &session).await?;
    let metadata = vortex_parquet::file_metadata(&handle).await?;

    let row_groups = metadata.row_groups().len();
    let rows: i64 = metadata.row_groups().iter().map(|rg| rg.num_rows()).sum();
    let columns = metadata.file_metadata().schema_descr().num_columns();
    let rows_per_group = if row_groups == 0 {
        0
    } else {
        rows / i64::try_from(row_groups)?
    };

    // Which encodings the file actually uses, counted over every column chunk. A decoder arm can only
    // be exercised by encodings that are present, so an arm that looks like a no-op may simply have
    // been handed a file with nothing for it to do — this is the line that tells them apart.
    let mut encodings: BTreeMap<String, usize> = BTreeMap::new();
    let mut codecs: BTreeMap<String, usize> = BTreeMap::new();
    for rg in metadata.row_groups() {
        for col in rg.columns() {
            for encoding in col.encodings() {
                *encodings.entry(format!("{encoding}")).or_default() += 1;
            }
            *codecs.entry(format!("{}", col.compression())).or_default() += 1;
        }
    }
    let render = |counts: BTreeMap<String, usize>| {
        counts
            .into_iter()
            .map(|(name, count)| format!("{name}x{count}"))
            .collect::<Vec<_>>()
            .join(" ")
    };

    Ok(format!(
        "{rows} rows, {columns} leaf columns, {row_groups} row groups (~{rows_per_group} rows each)\n\
         encodings: {}\n\
         codecs: {}",
        render(encodings),
        render(codecs)
    ))
}

/// Rewrites the Parquet file with a different codec, beside the original.
///
/// Read through this crate's own reader and written with arrow-rs, so the row order and values are
/// the originals; only the compression differs.
async fn recompress(path: &Path, codec: &str) -> Result<()> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::basic::ZstdLevel;
    use parquet::file::properties::WriterProperties;

    let compression = match codec {
        "none" => Compression::UNCOMPRESSED,
        "snappy" => Compression::SNAPPY,
        "zstd" => Compression::ZSTD(ZstdLevel::default()),
        other => bail!("unknown codec {other:?}; use none, snappy or zstd"),
    };
    let out = path.with_extension(format!("{codec}.parquet"));
    if out.exists() {
        println!("{} already exists; nothing to do", out.display());
        return Ok(());
    }
    // Written beside the destination and renamed once it is complete. Creating the final name first
    // means an interrupted rewrite leaves a truncated file that the check above then treats as a
    // finished artifact — a benchmark reading half a file and reporting it as a result.
    let partial = partial_path(&out);

    let session = VortexSession::default().with_tokio();
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8"))?;
    let file = vortex_parquet::open_path(path_str, &session).await?;
    // Row groups sized to match the source, so the only difference is the codec.
    let row_group_rows = 111_111usize;
    let mut stream = ordered_scan(&session, file);

    let started = Instant::now();
    let mut writer: Option<ArrowWriter<std::fs::File>> = None;
    let mut rows = 0usize;
    while let Some(array) = stream.try_next().await? {
        let mut ctx = session.create_execution_ctx();
        let batch = array.execute_arrow(None, &mut ctx)?;
        let batch = arrow_array::RecordBatch::from(
            batch
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| anyhow::anyhow!("a scan produced a non-struct array"))?
                .clone(),
        );
        // Created lazily: the writer needs a schema and the schema comes from the first batch.
        // `get_or_insert` rather than a match arm that re-reads the option, so there is no
        // "we just set it" unwrap to justify.
        let writer = match writer {
            Some(ref mut w) => w,
            None => {
                let properties = WriterProperties::builder()
                    .set_compression(compression)
                    .set_max_row_group_row_count(Some(row_group_rows))
                    .build();
                let file = std::fs::File::create(&partial)?;
                writer.get_or_insert(ArrowWriter::try_new(
                    file,
                    batch.schema(),
                    Some(properties),
                )?)
            }
        };
        rows += batch.num_rows();
        writer.write(&batch)?;
    }
    if let Some(w) = writer {
        w.close()?;
    }
    // Published only now that every write and the footer have succeeded.
    std::fs::rename(&partial, &out)?;
    println!(
        "wrote {} in {:.1}s ({rows} rows)",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Every row of `file`, in order, through Vortex's V2 scan.
fn ordered_scan(
    session: &VortexSession,
    file: ScanFile,
) -> futures::stream::BoxStream<'static, vortex::error::VortexResult<vortex::array::ArrayRef>> {
    let concurrency = get_available_parallelism().unwrap_or(1);
    ScanDriver::new(session.handle(), concurrency, true)
        .run(ScanBuilder::new(session.clone(), file))
}

/// Writes the Parquet file's rows out as a Vortex file beside it.
///
/// Read through this crate's own reader rather than arrow-rs's, so the Vortex file holds exactly what
/// the Parquet arms produce — if the reader were wrong, the reference arm would be wrong the same
/// way, which is the honest failure mode. Skips the work if the output already exists.
async fn convert(path: &Path) -> Result<()> {
    use vortex::file::WriteOptionsSessionExt;

    let out = path.with_extension("vortex");
    if out.exists() {
        println!("{} already exists; nothing to do", out.display());
        return Ok(());
    }
    let partial = partial_path(&out);

    let session = VortexSession::default().with_tokio();
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8"))?;
    let file = vortex_parquet::open_path(path_str, &session).await?;
    let dtype = file.layout.dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, ordered_scan(&session, file));

    let file = tokio::fs::File::create(&partial).await?;
    let started = Instant::now();
    let summary = session.write_options().write(file, stream).await?;
    tokio::fs::rename(&partial, &out).await?;
    println!(
        "wrote {} in {:.1}s ({} rows)",
        out.display(),
        started.elapsed().as_secs_f64(),
        summary.row_count()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let source = Source::parse(&args.file)?;
    // Rejected rather than silently clamped: `measured 0` used to run one iteration through a
    // `max(1)` and report zero, so the header described a methodology that was not the one used.
    if args.measured == 0 {
        bail!("--measured must be at least 1; it is the number of timed iterations");
    }

    // Resolved once, up front, and then carried by every arm. Reading the environment at each point
    // of use is how two arms in one run end up on different configurations.
    let io = IoConfig {
        backend: args.io,
        concurrency: args.io_concurrency,
        coalesce_distance: args.io_coalesce_distance_kb.map(|kb| kb << 10),
        coalesce_max: args.io_coalesce_max_kb.map(|kb| kb << 10),
    }
    .resolve_env()?;

    // The rewrite modes read and write with a plain `File`, so they are local-only. Said plainly
    // rather than by a type error, because "rewrite the copy in my bucket" is a reasonable thing to
    // expect and the answer is "download it first".
    if args.recompress.is_some() || args.convert {
        let path = source.local_path().ok_or_else(|| {
            anyhow::anyhow!("--recompress and --convert rewrite a local file; {source} is not one")
        })?;
        if let Some(codec) = &args.recompress {
            if !matches!(args.suite, SuiteArg::Clickbench) {
                bail!("--recompress operates on one ClickBench file");
            }
            return recompress(path, codec).await;
        }
        // A TPC suite is a DIRECTORY of `<table>.parquet`, and its reference arms read a `.vortex`
        // sibling per table. Converting one file at a time by hand is how a table gets silently
        // missed, which then fails at registration rather than here.
        if path.is_dir() {
            return convert_directory(path, &suite_tables(args.suite, &args)?).await;
        }
        return convert(path).await;
    }

    if args.preflight {
        let arms = select_arms(args.arms.as_deref(), io)?;
        let (cases, skipped) = preflight_cases(&args, &source)?;
        if cases.is_empty() {
            bail!("--preflight has nothing to check: no suite could be built");
        }
        // A gate clears a *configuration*, not a binary. Passing under one backend says nothing about
        // the other, so the one that ran is named at the top of the verdict.
        println!("io: {}", io.describe(&source));

        let report = preflight::gate(
            &cases,
            &arms,
            preflight::Thresholds {
                budget: std::time::Duration::from_secs(args.preflight_budget),
                ratio_limit: args.preflight_ratio,
                iterations: Iterations {
                    warmup: args.warmup,
                    measured: args.measured.max(2),
                },
            },
        )
        .await?;
        print!("{}", report.text);

        // Printed after the verdict, and counted into it below. A skipped suite is not a passing
        // suite, and the gate's whole purpose is that "it ran and looked fine" stops being enough.
        if !skipped.is_empty() {
            println!("\nSUITES NOT CHECKED ({}):", skipped.len());
            for skip in &skipped {
                println!("  {skip}");
            }
            println!(
                "The gate says nothing about these. TPC-DS in particular is where the q72 hang \
                 lives, so a pass without it is not a clearance to launch."
            );
        }

        if report.failures > 0 {
            bail!(
                "preflight found {} problem(s); a campaign launched now would produce numbers that \
                 have to be thrown away",
                report.failures
            );
        }
        return Ok(());
    }

    let mut suite = match args.suite {
        SuiteArg::Clickbench => suite::clickbench(),
        SuiteArg::Selectivity => suite::selectivity()?,
        SuiteArg::Projection => suite::projection()?,
        SuiteArg::Strings => suite::strings()?,
        SuiteArg::Tpch => suite::tpch(
            args.tpch_sql_dir
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--suite tpch requires --tpch-sql-dir"))?,
        )?,
        SuiteArg::Tpcds => suite::tpcds(
            args.tpcds_sql_dir
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--suite tpcds requires --tpcds-sql-dir"))?,
        )?,
    };
    let sources = match args.suite {
        // All four single-file suites read the same one `hits` file.
        SuiteArg::Clickbench | SuiteArg::Selectivity | SuiteArg::Projection | SuiteArg::Strings => {
            vec![("hits".to_string(), source.clone())]
        }
        SuiteArg::Tpch | SuiteArg::Tpcds => {
            let directory = source.local_path().ok_or_else(|| {
                anyhow::anyhow!(
                    "the TPC benchmark harness currently expects a local --file directory"
                )
            })?;
            if !directory.is_dir() {
                bail!(
                    "a TPC suite expects --file to be a directory, got {}",
                    directory.display()
                );
            }
            suite
                .tables
                .iter()
                .map(|table| {
                    (
                        table.clone(),
                        Source::Local(directory.join(format!("{table}.parquet"))),
                    )
                })
                .collect()
        }
    };
    if let Some(indices) = &args.queries {
        if let Some(bad) = indices.iter().find(|i| **i >= suite.queries.len()) {
            bail!(
                "query index {bad} is out of range; the suite has {} queries",
                suite.queries.len()
            );
        }
        suite.queries = indices.iter().map(|i| suite.queries[*i].clone()).collect();
        suite.labels = indices.iter().map(|i| suite.labels[*i].clone()).collect();
    }

    let arms = match args.io_arms.as_deref() {
        Some(specs) => io_comparison_arms(specs, io)?,
        None => select_arms(args.arms.as_deref(), io)?,
    };
    let iterations = Iterations {
        warmup: args.warmup,
        measured: args.measured,
    };

    eprintln!("running {} arms interleaved per query ...", arms.len());
    let io_before = proc_io();
    // No budget on a measurement run: a ceiling that fired here would silently turn a slow query into
    // an absent one, which is the very substitution the report exists to prevent. Hangs are the gate's
    // job, and the gate runs before this does.
    let results =
        run::run_suite(&suite, &arms, &sources, iterations, args.cache.into(), None).await?;
    let read_accounting = io_delta(io_before, proc_io());

    println!("data: {source}");
    for (table, table_source) in &sources {
        match describe(table_source).await {
            Ok(shape) => println!("shape[{table}]: {shape}"),
            Err(e) => println!("shape[{table}]: unavailable ({e})"),
        }
    }
    println!(
        "iterations: {} warmup + {} measured (best reported), cache: {}",
        args.warmup, args.measured, args.cache
    );
    println!("knobs: {}", knobs()?);
    // The I/O configuration is printed on its own line rather than folded into `knobs`, because it is
    // resolved rather than echoed: `auto` is two different backends depending on the path, and a run
    // that reports the flag instead of the effect cannot be compared against another one later.
    //
    // Under `--io-arms` there is no single configuration to report. Printing the run-level default
    // then would name a backend that no arm necessarily used — the header contradicting the table
    // directly beneath it.
    match args.io_arms.as_deref() {
        Some(specs) => println!(
            "io: per-arm ({}); each arm's column heading carries its own configuration",
            specs.join(", ")
        ),
        None => println!("io: {}", io.describe(&source)),
    }
    println!();
    if let Some(path) = args.samples_out.as_deref() {
        write_samples(path, &suite, &results, &args, &source)
            .map_err(|e| anyhow::anyhow!("failed to write samples to {}: {e}", path.display()))?;
        println!("samples: {} ({} arms)", path.display(), results.len());
    }

    let report = run::report(&suite, &results, args.digests);
    print!("{}", report.text);

    // Printed after the table because it answers a different question: not "how fast" but "did the
    // path this arm is named for actually run". An arm whose optional path never fired is measuring
    // its own fallback.
    let counters: Vec<String> = arms
        .iter()
        .filter_map(|arm| arm.counters().map(|c| format!("  {:<18} {c}", arm.name())))
        .collect();
    if !counters.is_empty() {
        println!("\ncounters (cumulative over the run):");
        for line in counters {
            println!("{line}");
        }
    }

    if let Some(io) = read_accounting {
        println!("\nread accounting (whole process): {io}");
    }

    // Exit non-zero on a result that must not be quoted. A disagreement means one arm is wrong, so
    // the timings describe incomparable work; a failure means the table has holes. Both used to print
    // and exit zero, which is how a bad run gets pasted into a document.
    if report.hard_disagreements > 0 || report.failures > 0 {
        bail!(
            "{} answer disagreement(s) and {} failure(s): these numbers are not comparable and \
             should not be quoted",
            report.hard_disagreements,
            report.failures
        );
    }

    Ok(())
}
