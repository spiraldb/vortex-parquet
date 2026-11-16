// SPDX-License-Identifier: Apache-2.0

//! Running a suite against a set of arms, and reporting the result.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::default_cache::DefaultCache;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::physical_plan::statistics::StatisticsContext;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use parking_lot::Mutex;
use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::arm::ArmRef;
use crate::digest::ResultDigest;
use crate::digest::digest;
use crate::source::Source;
use crate::suite::Suite;

/// How many times each query runs.
#[derive(Clone, Copy, Debug)]
pub struct Iterations {
    /// Runs whose timings are discarded, to let caches settle.
    pub warmup: usize,
    /// Runs whose timings are kept.
    pub measured: usize,
}

impl Default for Iterations {
    fn default() -> Self {
        // One warmup is enough to fault the file into the page cache, which is the dominant
        // first-run effect for a local file.
        Self {
            warmup: 1,
            measured: 3,
        }
    }
}

/// What state the caches are in when a timed iteration begins.
///
/// Spelled out because "cold" is ambiguous, and because an arm may cache decoded data across scans —
/// [`VortexParquet`](crate::arm::VortexParquet) does — so the states differ by far more than noise.
/// The first three do **not** touch the OS page cache: the data file stays resident throughout,
/// which is the separate sense in which those are warm measurements. [`CacheState::ColdDisk`] is the
/// one that does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheState {
    /// Register once, keep everything. What a repeated query against a live process sees.
    #[default]
    Warm,
    /// Register once, then drop the arm's caches before each iteration.
    ///
    /// Isolates *within-scan* reuse: the file is still open and its footer still parsed, but no
    /// decoded data survives from the previous query.
    Flushed,
    /// Re-register before each iteration: new reader tree, footer re-read, nothing retained.
    ///
    /// Applied to *every* arm — including the baseline, whose table provider caches statistics — so
    /// that flushing our own caches is not an advantage handed to us by the measurement.
    PureCold,
    /// As [`CacheState::PureCold`], and additionally evict the **data file from the OS page cache**
    /// before each iteration, so the bytes are read from the device.
    ///
    /// This distinguishes device reads from OS page-cache hits when evaluating I/O amplification.
    ColdDisk,
}

/// A wall-clock ceiling on a single query execution.
///
/// # Why a benchmark needs a stopwatch that gives up
///
/// A query that never returns prevents the final report. A budget turns that into a named query
/// failure so the report can identify the cause.
#[derive(Clone, Copy, Debug)]
pub struct QueryBudget {
    /// How long one execution may take before it is abandoned.
    pub per_query: Duration,
    /// Whether the first breach aborts the whole run.
    ///
    /// A breach is not a clean failure. `tokio::time::timeout` drops the query future, but the work
    /// DataFusion has already spawned keeps running on the runtime with nothing left to consume it —
    /// so every cell measured *after* a breach shares the machine with an abandoned query. Continuing
    /// would produce numbers, and they would be wrong in the direction that looks like a real
    /// regression.
    ///
    /// A gate therefore sets this; a measurement run that merely wants a backstop need not.
    pub fatal: bool,
}

/// The outcome of one query against one arm.
#[derive(Clone, Debug)]
pub struct QueryResult {
    /// Query index within the suite.
    pub query: usize,
    /// The best measured wall-clock time, or `None` if the query failed.
    pub best: Option<Duration>,
    /// Every measured sample, in the order it ran.
    ///
    /// Retained rather than reduced on the spot. Keeping only `best` makes two things unmeasurable:
    /// the **spread** between samples, which is the only honest resolution floor to quote an effect
    /// against, and **drift**, which is visible only as a trend from the first sample to the last.
    /// A stored minimum cannot reveal whether the machine became systematically slower during a run.
    pub samples: Vec<Duration>,
    /// The number of rows the query returned.
    pub rows: Option<usize>,
    /// A digest of the values returned, which is the *actual* agreement check.
    ///
    /// `None` means the result held a type [`digest`] cannot render, so this query is checked on its
    /// row count alone — reported rather than hidden, because an unchecked query that looks checked
    /// is worse than one that admits it.
    pub digest: Option<ResultDigest>,
    /// Whether the measured iterations disagreed with each other.
    ///
    /// Recorded rather than fatal. Some ClickBench queries slice a `LIMIT` out of a tie, so which
    /// rows come back depends on how the engine happened to schedule that run — the query is
    /// genuinely non-deterministic and no arm is wrong. Discarding it would remove a real timing;
    /// hiding it would let a *non*-ambiguous query drift unnoticed, which is what happened while this
    /// field did not exist and the digest was overwritten by whichever iteration ran last.
    pub unstable: bool,
    /// Whether the query failed by exceeding its [`QueryBudget`] rather than by erroring.
    ///
    /// Kept apart from `error` because the two mean different things to a gate: an error is an arm
    /// that cannot answer a query, a breach is an arm that may simply be slow — and only the latter
    /// leaves abandoned work running behind it.
    pub timed_out: bool,
    /// Why the query failed, if it did.
    pub error: Option<String>,
}

/// Every result for one arm.
#[derive(Clone, Debug)]
pub struct ArmResults {
    /// The arm's name.
    pub arm: String,
    /// One entry per query, in suite order.
    pub queries: Vec<QueryResult>,
}

/// The order arms run in for query `query_idx`, rotated one position per query.
///
/// Split out from [`run_suite`] so it can be tested at all: the loop it came from needs registered
/// arms and a live DataFusion context, so the rotation — the part that can silently misalign every
/// ratio in a report — was the one piece with no way to assert on it.
///
/// The contract is narrow and worth stating: this yields **every** arm index exactly once (it is a
/// permutation, not a sample), and consecutive queries start from consecutive arms.
fn arm_order(query_idx: usize, arms: usize) -> impl Iterator<Item = usize> {
    (0..arms).map(move |slot| (slot + query_idx) % arms)
}

/// Runs `suite` against every arm, reading from `path`.
///
/// # Why the arms are interleaved
///
/// The obvious shape is one arm at a time: register, run all 43 queries, move on. That was the shape,
/// and it silently confounded the result. A full suite takes minutes, a laptop's clock drifts over
/// minutes, and under a fixed arm order every arm is measured in a *different* thermal regime — so
/// part of every ratio is "when did this arm happen to run".
///
/// The effect was large enough for the baseline itself to vary by almost twofold between consecutive
/// runs. A ladder measured that way mostly describes the machine.
///
/// So the loop is inverted: every arm is registered up front, and each query is run against all arms
/// before moving to the next. Arms are then measured seconds apart rather than minutes, under
/// conditions as close to identical as this harness can make them. It does not remove drift — it
/// stops drift from aligning with the variable under test.
pub async fn run_suite(
    suite: &Suite,
    arms: &[ArmRef],
    sources: &[(String, Source)],
    iterations: Iterations,
    cache_state: CacheState,
    budget: Option<QueryBudget>,
) -> Result<Vec<ArmResults>> {
    let mut resetters = Vec::with_capacity(arms.len());
    for arm in arms {
        let reset = Resetter {
            state: cache_state,
            ctx: Mutex::new(session_context()?),
            arm,
            // Each arm chooses its own representation of every table. Reference arms map every
            // Parquet source to the `.vortex` file holding the same rows.
            tables: sources
                .iter()
                .map(|(table, source)| (table.clone(), arm.data_path(source)))
                .collect(),
        };
        let session = VortexSession::default().with_tokio();
        for (_, path) in &reset.tables {
            if !path.exists(&session).await? {
                anyhow::bail!(
                    "arm {} needs {}, which does not exist. Reference arms read the same rows \
                     stored as Vortex files.",
                    arm.name(),
                    path
                );
            }
        }
        reset.register().await?;
        resetters.push(reset);
    }

    let mut results: Vec<ArmResults> = arms
        .iter()
        .map(|arm| ArmResults {
            arm: arm.name().to_string(),
            queries: Vec::with_capacity(suite.queries.len()),
        })
        .collect();

    for (idx, sql) in suite.queries.iter().enumerate() {
        // Arms rotate one position per query, so no arm sits in the same slot twice in a row and each
        // one takes its turn paying the first-run cost within a query.
        for arm_idx in arm_order(idx, resetters.len()) {
            let result = run_query(idx, sql, iterations, &resetters[arm_idx], budget).await;
            // Abandoned work outlives the query that was dropped, so anything measured after a breach
            // is measured on a machine that is still busy. Stopping here is the only way the numbers
            // above the breach stay meaningful.
            if result.timed_out && budget.is_some_and(|b| b.fatal) {
                let label = suite
                    .labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("q{idx}"));
                bail!(
                    "{} exceeded its per-query budget on {label}; abandoning the run because the \
                     query's spawned work keeps running and would contaminate every later cell",
                    arms[arm_idx].name(),
                );
            }
            results[arm_idx].queries.push(result);
        }
    }

    Ok(results)
}

/// One registered [`SessionContext`] per arm, in the order the arms were given.
///
/// For inspecting *plans* rather than timings: the gate reads a query's partitioning shape out of the
/// planner, which needs a context with the tables in it and nothing else. Registration is the same
/// call [`run_suite`] makes, so a shape read here is the shape a measured run would get.
pub async fn registered_contexts(
    arms: &[ArmRef],
    sources: &[(String, Source)],
) -> Result<Vec<SessionContext>> {
    let mut contexts = Vec::with_capacity(arms.len());
    for arm in arms {
        let ctx = session_context()?;
        for (table, source) in sources {
            let path = arm.data_path(source);
            arm.register(&ctx, table, &path)
                .await
                .with_context(|| format!("arm {} failed to register {table}", arm.name()))?;
        }
        contexts.push(ctx);
    }
    Ok(contexts)
}

/// Puts the caches back into `state` between iterations.
struct Resetter<'a> {
    state: CacheState,
    /// Replaced wholesale by `PureCold`, so it is owned rather than borrowed.
    ///
    /// The `SessionContext` carries DataFusion's own file-metadata and list-files caches. Keeping it
    /// while flushing only the arm's cache would make "pure cold" mean "DataFusion warm, arm cold",
    /// which is the opposite of the uniformity the state exists to provide.
    ctx: Mutex<SessionContext>,
    arm: &'a ArmRef,
    tables: Vec<(String, Source)>,
}

impl Resetter<'_> {
    /// The context queries should run against right now.
    fn ctx(&self) -> SessionContext {
        self.ctx.lock().clone()
    }

    /// Registers every table in the suite.
    async fn register(&self) -> Result<()> {
        let ctx = self.ctx();
        for (table, _) in &self.tables {
            // Deregister first: re-registering the same name is how `PureCold` works, and a provider
            // left in place would be the thing measured.
            drop(ctx.deregister_table(table.as_str()));
        }
        for (table, path) in &self.tables {
            self.arm
                .register(&ctx, table, path)
                .await
                .with_context(|| format!("arm {} failed to register {table}", self.arm.name()))?;
        }
        Ok(())
    }

    /// Applies this state's reset, if any.
    async fn before_iteration(&self) -> Result<()> {
        match self.state {
            CacheState::Warm => Ok(()),
            CacheState::Flushed => {
                self.arm.flush();
                Ok(())
            }
            CacheState::PureCold => {
                // A fresh context, so DataFusion's caches go too.
                *self.ctx.lock() = session_context()?;
                // Flush *and* re-register. Re-registering alone is not enough: an arm may own its
                // cache across file opens — `ParquetFormat` does — so a fresh reader tree can be
                // handed the same warm decode cache, and "pure cold" would silently mean "cold
                // metadata, warm data".
                self.arm.flush();
                self.register().await
            }
            CacheState::ColdDisk => {
                *self.ctx.lock() = session_context()?;
                self.arm.flush();
                // Evicted *before* re-registering, because registration reads the footer and would
                // otherwise pull part of the file straight back in.
                // Object storage has no page cache to drop, and pretending otherwise would report a
                // warm-sounding number as a cold one. Every read against a bucket is already a real
                // request, so `flushed` and `pure-cold` measure true I/O there without this.
                for (_, source) in &self.tables {
                    let path = source.local_path().ok_or_else(|| {
                        anyhow::anyhow!(
                            "--cache cold-disk evicts local files, but {source} is remote. Reads \
                             against an object store are never page-cached, so --cache flushed \
                             already measures real I/O there."
                        )
                    })?;
                    drop_page_cache(path)?;
                }
                self.register().await
            }
        }
    }
}

/// Evicts `path` from the OS page cache.
///
/// `posix_fadvise(POSIX_FADV_DONTNEED)` rather than `/proc/sys/vm/drop_caches`: it needs no root, and
/// it drops **only this file** instead of everything else the box has cached.
///
/// The kernel treats `DONTNEED` as *advice* and may decline it — a dirty page, another process
/// holding the file mapped, or simply the kernel's discretion. So the eviction is **verified** rather
/// than assumed: `mincore` reports how many of the file's pages are still resident, and a sample that
/// is not actually cold fails the query instead of being reported as a cold measurement. An earlier
/// version of this comment claimed the read volume was reported and made the run auditable; it was
/// not, and nothing checked.
#[cfg(target_os = "linux")]
fn drop_page_cache(path: &Path) -> Result<()> {
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {} to evict it", path.display()))?;
    // SAFETY: `fd` is a valid descriptor owned by `file` for the duration of the call, and
    // `posix_fadvise` neither retains it nor writes through it.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        anyhow::bail!(
            "posix_fadvise(DONTNEED) on {} failed with {rc}",
            path.display()
        );
    }

    // Advice is not a guarantee, so check.
    let resident = resident_fraction(&file)?;
    if resident > MAX_RESIDENT_AFTER_EVICTION {
        anyhow::bail!(
            "after posix_fadvise(DONTNEED), {:.1}% of {} is still in the page cache, so this would \
             not have been a cold measurement",
            resident * 100.0,
            path.display()
        );
    }
    Ok(())
}

/// How much of a file may remain cached and still count as evicted.
///
/// Not zero: the footer was just read to register the table, and the kernel may hold a few pages for
/// reasons of its own. A percent of a 1.5 GB file is ~15 MB, far below anything that would flatter a
/// scan of the whole thing.
#[cfg(target_os = "linux")]
const MAX_RESIDENT_AFTER_EVICTION: f64 = 0.01;

/// The fraction of `file`'s pages currently resident in the page cache, via `mincore`.
#[cfg(target_os = "linux")]
fn resident_fraction(file: &std::fs::File) -> Result<f64> {
    use std::os::unix::io::AsRawFd;

    let len = usize::try_from(file.metadata()?.len())?;
    if len == 0 {
        return Ok(0.0);
    }
    // SAFETY: a read-only shared mapping of a valid descriptor; the result is checked against
    // MAP_FAILED before use and unmapped below.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        anyhow::bail!("mmap failed while checking page-cache residency");
    }

    // Everything from here to the unmap must not return early: `addr` is a live mapping and `?` would
    // leak it. The page size is read and validated first, and the two failure paths below unmap before
    // reporting, rather than relying on a later cleanup that an early return would skip.
    //
    // SAFETY: `sysconf` reads a process-global constant and touches no memory this owns.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        // SAFETY: unmapping exactly the mapping created above.
        unsafe { libc::munmap(addr, len) };
        anyhow::bail!("sysconf(_SC_PAGESIZE) reported {page_size}, which is not a page size");
    }
    #[expect(
        clippy::cast_sign_loss,
        reason = "checked positive immediately above, so the conversion cannot wrap"
    )]
    let page = page_size as usize;

    let pages = len.div_ceil(page);
    let mut vec = vec![0u8; pages];
    // SAFETY: `addr` maps `len` bytes and `vec` has one byte per page of that range.
    let rc = unsafe { libc::mincore(addr, len, vec.as_mut_ptr()) };
    // SAFETY: unmapping exactly the mapping created above.
    let unmapped = unsafe { libc::munmap(addr, len) };
    if rc != 0 {
        anyhow::bail!("mincore failed while checking page-cache residency");
    }
    // Reported rather than ignored: a failed unmap means this process is leaking a mapping of every
    // file it evicts, which over a long sweep is the difference between a cold measurement and an
    // out-of-memory kill.
    if unmapped != 0 {
        anyhow::bail!("munmap failed after checking page-cache residency");
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "a fraction of a page count; f64 is exact far past any real file"
    )]
    Ok(vec.iter().filter(|b| *b & 1 == 1).count() as f64 / pages as f64)
}

/// Only Linux offers `posix_fadvise(DONTNEED)`; macOS has no call that evicts another process's
/// cached pages for a file. The state is refused rather than silently downgraded to a warm run,
/// because a "cold" number that was actually warm is worse than no number.
#[cfg(not(target_os = "linux"))]
fn drop_page_cache(path: &Path) -> Result<()> {
    anyhow::bail!(
        "--cache cold-disk needs posix_fadvise(DONTNEED), which only Linux provides; {} cannot be \
         evicted here. Run cold measurements on Linux.",
        path.display()
    )
}

/// Builds the DataFusion session every arm runs in.
///
/// The part that matters is the cache manager: without a file-statistics and list-files cache, a
/// table provider re-resolves file metadata on *every query*, which reads as the format being slow
/// rather than the harness being slow. Applied to every arm equally, so it does not tilt the
/// comparison.
fn session_context() -> Result<SessionContext> {
    session_context_with(batch_size_override()?)
}

/// Environment variable setting the DataFusion batch size.
pub const BATCH_SIZE_ENV: &str = "BENCHMARKS_BATCH_SIZE";

/// The DataFusion batch size to run with, from [`BATCH_SIZE_ENV`], or DataFusion's default.
///
/// An environment variable rather than a flag because it is an experiment knob, not part of the
/// declared methodology: every arm gets the same value, and a run that sets it says so in its header.
fn batch_size_override() -> Result<Option<usize>> {
    let size: Option<usize> = parse_env(BATCH_SIZE_ENV)?;
    // Zero reaches a `debug_assert` inside DataFusion and produces an empty-batch scan in release,
    // which is not a batch size anyone means to test.
    if size == Some(0) {
        bail!("{BATCH_SIZE_ENV}=0 is not a batch size");
    }
    Ok(size)
}

/// Parses an environment knob, refusing a value it cannot read.
///
/// # Why this is not `.ok()`
///
/// It was. A typo in a sweep — `BENCHMARKS_BATCH_SIZE=64k` — silently fell back to the default while the
/// report header echoed the raw string, so the run claimed a configuration it had not used and the
/// cell looked like evidence that the knob did nothing. A knob that cannot be read is a broken
/// experiment, not a default.
pub fn parse_env<T: std::str::FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        // A value that is not valid Unicode is a *set* knob this cannot read, and treating it as
        // unset would run the default while the header claimed otherwise — the same failure the
        // strict parse exists to prevent.
        Err(std::env::VarError::NotUnicode(raw)) => {
            anyhow::bail!("{name}={raw:?} is not valid Unicode")
        }
        Ok(raw) => raw
            .parse()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{name}={raw:?} could not be read: {e}")),
    }
}

/// Environment variable controlling DataFusion's Parquet filter pushdown for the baseline arm.
pub const PUSHDOWN_ENV: &str = "BENCHMARKS_PUSHDOWN_FILTERS";

/// Environment variable applying arbitrary DataFusion settings, as comma-separated `key=value`.
pub const DF_CONFIG_ENV: &str = "BENCHMARKS_DF_CONFIG";

/// Whether to enable `datafusion.execution.parquet.pushdown_filters`. **On by default here.**
///
/// # Why this differs from DataFusion's own default
///
/// DataFusion ships it `false`. Leaving it there measures the configuration most users get, which is
/// a fair opponent but not the strongest one — and the difference is not cosmetic. arrow-rs only
/// engages its predicate cache (`CachedArrayReader`, the same idea as this crate's
/// [`crate::cache`]) when filters are pushed down, so with it off the baseline decodes every row and
/// then decodes the filter columns a second time for the output projection.
///
/// Enabling this gives the baseline its predicate cache and row filtering during decode.
///
/// Set `BENCHMARKS_PUSHDOWN_FILTERS=0` to restore DataFusion's own default.
pub fn pushdown_filters() -> Result<bool> {
    // Every value is either recognised or an error. This used to treat anything that was not exactly
    // `0` or `false` as "on", so a misspelling of `false` enabled pushdown and the report header then
    // presented it as a deliberate setting.
    match parse_env::<String>(PUSHDOWN_ENV)?.as_deref() {
        None => Ok(true),
        Some("1" | "true") => Ok(true),
        Some("0" | "false") => Ok(false),
        Some(other) => bail!("{PUSHDOWN_ENV}={other:?} is not a boolean; use 0/1 or true/false"),
    }
}

/// Builds the session, optionally overriding the batch size.
fn session_context_with(batch_size: Option<usize>) -> Result<SessionContext> {
    let cache_config = CacheManagerConfig::default()
        .with_file_statistics_cache(Some(Arc::new(DefaultCache::new(64 * 1024 * 1024))))
        .with_list_files_cache(Some(Arc::new(DefaultCache::new(64 * 1024 * 1024))))
        // DataFusion's default file-metadata cache holds 50 MB, which a wide file's footer can
        // exceed — it is then evicted between queries and re-parsed every time. 4 GB is far more than
        // any footer needs and removes the variable.
        .with_metadata_cache_limit(4 * 1024 * 1024 * 1024);

    let mut config = SessionConfig::new();
    if let Some(batch_size) = batch_size {
        config = config.with_batch_size(batch_size);
    }
    if pushdown_filters()? {
        // `reorder_filters` alongside it: pushdown without reordering evaluates predicates in written
        // order, which is not the configuration anyone tuning DataFusion would run.
        let options = config.options_mut();
        options.execution.parquet.pushdown_filters = true;
        options.execution.parquet.reorder_filters = true;
    }

    // An escape hatch for attributing an effect to a specific DataFusion setting, e.g. separating
    // late materialization from TopK dynamic filter pushdown. Comma-separated `key=value`; an
    // unknown key is fatal rather than ignored, because a silently-dropped setting would look like
    // "that knob does nothing".
    // Read through the same strict path as every other knob: an unreadable value used to be ignored
    // while the header echoed it as applied.
    let spec: Option<String> = parse_env(DF_CONFIG_ENV)?;
    if let Some(spec) = spec {
        for pair in spec.split(',').filter(|p| !p.trim().is_empty()) {
            let (key, value) = pair
                .split_once('=')
                .with_context(|| format!("{DF_CONFIG_ENV} entry {pair:?} is not key=value"))?;
            config
                .options_mut()
                .set(key.trim(), value.trim())
                .with_context(|| format!("{DF_CONFIG_ENV}: {key}={value} was rejected"))?;
        }
    }

    Ok(
        match RuntimeEnvBuilder::new()
            .with_cache_manager(cache_config)
            .build_arc()
        {
            Ok(rt) => SessionContext::new_with_state(
                SessionStateBuilder::new()
                    .with_runtime_env(rt)
                    .with_config(config)
                    .with_default_features()
                    .build(),
            ),
            // A runtime that will not build is not worth failing the whole run over; the uncached session
            // still produces valid, if slower, numbers.
            // Refused rather than substituted. The fallback built a context with no caches, no batch
            // size, no pushdown and none of `BENCHMARKS_DF_CONFIG` — and then the run reported the settings
            // it had asked for, so a broken runtime produced numbers labelled as a configuration that
            // never ran.
            Err(e) => {
                return Err(e).context(
                    "failed to build the DataFusion runtime; refusing to run with a different \
                     configuration from the one reported",
                );
            }
        },
    )
}

/// Runs one query, timing it.
///
/// A failing query is recorded rather than propagated: one unsupported query should not discard a
/// whole run, and an arm that fails where others succeed is itself a useful result.
async fn run_query(
    query: usize,
    sql: &str,
    iterations: Iterations,
    reset: &Resetter<'_>,
    budget: Option<QueryBudget>,
) -> QueryResult {
    let failed = |error: String| QueryResult {
        query,
        best: None,
        samples: Vec::new(),
        rows: None,
        digest: None,
        unstable: false,
        timed_out: false,
        error: Some(error),
    };
    let breached = |elapsed: Duration| QueryResult {
        timed_out: true,
        error: Some(format!(
            "exceeded the {:.1}s per-query budget (abandoned after {:.1}s)",
            budget.map_or(0.0, |b| b.per_query.as_secs_f64()),
            elapsed.as_secs_f64(),
        )),
        ..failed(String::new())
    };

    // One helper for both loops, so a warmup and a measured iteration are subject to the same ceiling.
    // A hang during warmup is the same hang.
    let bounded = async |ctx: &SessionContext| match budget {
        None => execute(ctx, sql).await.map_err(Bounded::Failed),
        Some(budget) => match tokio::time::timeout(budget.per_query, execute(ctx, sql)).await {
            Ok(Ok(answer)) => Ok(answer),
            Ok(Err(e)) => Err(Bounded::Failed(e)),
            Err(_) => Err(Bounded::Breached),
        },
    };

    for _ in 0..iterations.warmup {
        if let Err(e) = reset.before_iteration().await {
            return failed(format!("cache reset failed: {e}"));
        }
        let started = Instant::now();
        match bounded(&reset.ctx()).await {
            Ok(_) => {}
            Err(Bounded::Failed(e)) => return failed(e.to_string()),
            Err(Bounded::Breached) => return breached(started.elapsed()),
        }
    }

    // The answer is recorded from the iteration that produced `best`, not from whichever ran last.
    // Keeping them separate is how the reported digest came to describe a *different* execution from
    // the reported time — harmless while every iteration agrees, and undetectable when they do not.
    let mut best: Option<(Duration, usize, Option<ResultDigest>)> = None;
    // Every sample, in run order. `best` alone cannot express spread or drift, and both are needed to
    // say whether a difference is readable at all.
    let mut samples: Vec<Duration> = Vec::with_capacity(iterations.measured.max(1));
    let mut unstable = false;
    for _ in 0..iterations.measured.max(1) {
        if let Err(e) = reset.before_iteration().await {
            return failed(format!("cache reset failed: {e}"));
        }
        let ctx = reset.ctx();
        let started = Instant::now();
        match bounded(&ctx).await {
            Ok((row_count, iteration_digest)) => {
                let elapsed = started.elapsed();
                samples.push(elapsed);
                // Iterations of one query against one arm must agree with each other. A disagreement
                // is non-determinism inside a single arm, which no cross-arm comparison can detect
                // and which would make the timing meaningless — so it is an error, not a note.
                if let Some((_, first_rows, first_digest)) = &best
                    && (*first_rows != row_count || *first_digest != iteration_digest)
                {
                    unstable = true;
                }
                match &best {
                    Some((b, ..)) if *b <= elapsed => {}
                    _ => best = Some((elapsed, row_count, iteration_digest)),
                }
            }
            Err(Bounded::Failed(e)) => return failed(e.to_string()),
            Err(Bounded::Breached) => return breached(started.elapsed()),
        }
    }

    let (best, rows, digest) = match best {
        Some(best) => best,
        None => return failed("no measured iteration ran".to_string()),
    };
    let (best, rows, digest) = (Some(best), Some(rows), digest);

    QueryResult {
        query,
        best,
        samples,
        rows,
        digest,
        unstable,
        timed_out: false,
        error: None,
    }
}

/// Why a bounded execution did not produce an answer.
enum Bounded {
    /// The query itself failed.
    Failed(anyhow::Error),
    /// The query was still running when its [`QueryBudget`] ran out.
    Breached,
}

/// Executes `sql` to completion and returns the row count and answer digest.
///
/// Collecting rather than just planning is the point: a lazily-planned query measures nothing about
/// the reader.
async fn execute(ctx: &SessionContext, sql: &str) -> Result<(usize, Option<ResultDigest>)> {
    let frame = ctx.sql(sql).await?;
    if std::env::var_os("BENCHMARKS_EXPLAIN").is_some() {
        let plan = frame.clone().create_physical_plan().await?;
        eprintln!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
    }
    if std::env::var_os("BENCHMARKS_EXPLAIN_STATS").is_some() {
        let plan = frame.clone().create_physical_plan().await?;
        print_plan_statistics(&plan, 0);
    }
    let batches = frame.collect().await?;
    let rows = batches.iter().map(|b| b.num_rows()).sum();
    // Digested here because this is the only place holding the batches, and they are dropped
    // immediately afterwards — the alternative is retaining every arm's full result set for the
    // length of a run.
    Ok((rows, digest(&batches)))
}

/// The output partition count of every **leaf** in `sql`'s physical plan, ascending.
///
/// # Why the leaves, and why this is checked at all
///
/// A change in scan parallelism can preserve answers while changing performance. This reads scan
/// partition counts from the physical plan without executing the query; downstream redistribution
/// may provide additional parallelism.
pub async fn leaf_partitions(ctx: &SessionContext, sql: &str) -> Result<Vec<usize>> {
    fn walk(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<usize>) {
        if plan.children().is_empty() {
            out.push(plan.properties().output_partitioning().partition_count());
            return;
        }
        for child in plan.children() {
            walk(child, out);
        }
    }

    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let mut counts = Vec::new();
    walk(&plan, &mut counts);
    // Sorted rather than positional: two arms plan the same query into differently *shaped* trees, and
    // comparing leaf 3 to leaf 3 across them would compare unrelated scans. The multiset of widths is
    // the part that is comparable, and it is the part the defect moved.
    counts.sort_unstable();
    Ok(counts)
}

/// How many of a scan's advertised partitions actually carry rows.
///
/// # Why advertised partitions are not enough
///
/// [`leaf_partitions`] reads the advertised count. An unwrapped `LayoutReaderDataSource` defaults
/// `split_max_row_count` to `u64::MAX`, so it can supply one split to multiple output streams and
/// leave all but one empty. The current Vortex file arm uses `FooterStatistics` to configure bounded
/// splits. This probe checks productive outputs by executing the scan's leaves directly, without
/// executing the rest of the query.
///
/// Returns `(advertised, productive)` for the widest leaf in the plan.
pub async fn productive_partitions(ctx: &SessionContext, sql: &str) -> Result<(usize, usize)> {
    fn widest_leaf(plan: &Arc<dyn ExecutionPlan>, best: &mut Option<Arc<dyn ExecutionPlan>>) {
        if plan.children().is_empty() {
            let count = plan.properties().output_partitioning().partition_count();
            let better = best.as_ref().is_none_or(|current| {
                count > current.properties().output_partitioning().partition_count()
            });
            if better {
                *best = Some(Arc::clone(plan));
            }
            return;
        }
        for child in plan.children() {
            widest_leaf(child, best);
        }
    }

    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let mut leaf = None;
    widest_leaf(&plan, &mut leaf);
    let Some(leaf) = leaf else {
        return Ok((0, 0));
    };

    let advertised = leaf.properties().output_partitioning().partition_count();
    let task_ctx = ctx.task_ctx();
    let mut productive = 0usize;
    for partition in 0..advertised {
        let mut stream = leaf.execute(partition, Arc::clone(&task_ctx))?;
        // Only whether *any* row arrives; the stream is dropped as soon as that is known, so a wide
        // scan is not fully decoded just to answer this.
        let mut rows = 0usize;
        while let Some(batch) = futures::StreamExt::next(&mut stream).await {
            rows += batch?.num_rows();
            if rows > 0 {
                break;
            }
        }
        if rows > 0 {
            productive += 1;
        }
    }
    Ok((advertised, productive))
}

fn print_plan_statistics(plan: &Arc<dyn ExecutionPlan>, depth: usize) {
    let partitions = plan.properties().output_partitioning().partition_count();
    let indent = "  ".repeat(depth);
    match StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new()) {
        Ok(statistics) => {
            eprintln!(
                "{indent}{} partitions={partitions} statistics={statistics}",
                plan.name()
            );
        }
        Err(error) => {
            eprintln!(
                "{indent}{} partitions={partitions} statistics_error={error}",
                plan.name()
            );
        }
    }
    for child in plan.children() {
        print_plan_statistics(child, depth + 1);
    }
}

/// Renders results as a text table, one row per query and one column per arm.
///
/// Also flags cross-arm disagreements on the **answer**, by digesting the values rather than counting
/// the rows. An arm that is fast because it returns the wrong answer is the failure mode worth
/// catching automatically, and a row count cannot catch it at all for a scalar aggregate.
///
/// # Why a `LIMIT` demotes a disagreement to a warning
///
/// A query with no `LIMIT` has a uniquely determined result *set*, so two arms returning different
/// values is a bug in one of them. Add a `LIMIT` over a non-total `ORDER BY` and it is not: the
/// engine may return any rows from the tie at the boundary, and all of them are correct answers. That
/// is 7 of ClickBench's 32 `LIMIT` queries — q41 slices 10 rows out of 4,276 groups all tied at
/// `PageViews = 1`, and q17 has no `ORDER BY` at all.
///
/// The rule is derived from the SQL rather than from a list of known-flaky queries, because a
/// hand-maintained list silently rots.
pub fn report(suite: &Suite, results: &[ArmResults], show_digests: bool) -> Report {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(out, "suite: {}", suite.name);

    let width = 22;
    let label_width = 8;
    let _ = write!(out, "{:<label_width$}", "query");
    for r in results {
        let _ = write!(out, "{:>width$}", r.arm);
    }
    let _ = writeln!(out);

    let label_of = |idx: usize| {
        suite
            .labels
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("q{idx}"))
    };

    // A geomean over a per-arm query set is not a comparison: if one arm failed a query the others
    // answered, its ratio is taken over different work. So the set is fixed first — every query that
    // *every* arm timed — and both the geomean and the total are reported over exactly that set.
    let comparable: Vec<usize> = (0..suite.queries.len())
        .filter(|idx| {
            results
                .iter()
                .all(|r| r.queries.get(*idx).is_some_and(|q| q.best.is_some()))
        })
        .collect();

    let mut disagreements = Vec::new();
    let mut ambiguous = Vec::new();
    let mut unstable: Vec<usize> = Vec::new();
    let mut unchecked: Vec<usize> = Vec::new();
    let mut totals: Vec<Duration> = vec![Duration::ZERO; results.len()];
    // Geomean is accumulated in log space; a product of 43 ratios overflows nothing but reads worse.
    let mut log_ratio_sums: Vec<f64> = vec![0.0; results.len()];
    let mut ratio_counts: Vec<usize> = vec![0; results.len()];

    for idx in 0..suite.queries.len() {
        let _ = write!(out, "{:<label_width$}", label_of(idx));
        let mut answers = Vec::new();
        let baseline = results
            .first()
            .and_then(|r| r.queries.get(idx))
            .and_then(|q| q.best);

        for (arm_idx, r) in results.iter().enumerate() {
            match r.queries.get(idx) {
                Some(QueryResult {
                    best: Some(d),
                    rows,
                    digest,
                    ..
                }) => {
                    let _ = write!(
                        out,
                        "{:>width$}",
                        format!("{:.1} ms", d.as_secs_f64() * 1e3)
                    );
                    if comparable.contains(&idx) {
                        totals[arm_idx] += *d;
                        if let Some(baseline) = baseline
                            && baseline > Duration::ZERO
                        {
                            log_ratio_sums[arm_idx] +=
                                (d.as_secs_f64() / baseline.as_secs_f64()).ln();
                            ratio_counts[arm_idx] += 1;
                        }
                    }
                    if let Some(rows) = rows {
                        answers.push((r.arm.as_str(), *rows, *digest));
                    }
                }
                Some(QueryResult { error: Some(_), .. }) => {
                    let _ = write!(out, "{:>width$}", "failed");
                }
                _ => {
                    let _ = write!(out, "{:>width$}", "-");
                }
            }
        }
        let _ = writeln!(out);

        // Compared on (rows, digest) together: the row count still catches an arm whose result type
        // could not be digested, and the digest catches everything the row count cannot.
        if let Some((_, first_rows, first_digest)) = answers.first()
            && answers
                .iter()
                .any(|(_, rows, digest)| rows != first_rows || digest != first_digest)
        {
            if suite
                .queries
                .get(idx)
                .is_some_and(|sql| has_limit_clause(sql))
            {
                ambiguous.push((idx, answers.clone()));
            } else {
                disagreements.push((idx, answers.clone()));
            }
        }
        if answers.iter().any(|(_, _, digest)| digest.is_none()) {
            unchecked.push(idx);
        }
        if results
            .iter()
            .any(|r| r.queries.get(idx).is_some_and(|q| q.unstable))
        {
            unstable.push(idx);
        }
    }

    let _ = writeln!(out);
    let _ = write!(out, "{:<label_width$}", "total");
    for total in &totals {
        let _ = write!(out, "{:>width$}", format!("{:.2} s", total.as_secs_f64()));
    }
    let _ = writeln!(out);
    let _ = write!(out, "{:<label_width$}", "geomean");
    for (arm_idx, _) in results.iter().enumerate() {
        let cell = if ratio_counts[arm_idx] == 0 {
            "-".to_string()
        } else {
            let geomean = (log_ratio_sums[arm_idx]
                / f64::from(u32::try_from(ratio_counts[arm_idx]).unwrap_or(1)))
            .exp();
            format!("{geomean:.3}x")
        };
        let _ = write!(out, "{cell:>width$}");
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "(geomean is per-query ratio against {}, the leftmost arm; total and geomean cover the {} \
         of {} queries every arm timed)",
        results.first().map_or("-", |r| r.arm.as_str()),
        comparable.len(),
        suite.queries.len()
    );

    // The instrument reporting on itself. Printed on every report, not on request: an effect quoted
    // without its resolution floor is the failure mode this exists to prevent.
    write_stability(&mut out, results);

    if !unstable.is_empty() {
        let labels = unstable
            .iter()
            .map(|idx| label_of(*idx))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            out,
            "\nnon-deterministic across iterations of a single arm (reported time is the fastest \
             iteration; treat these queries' answers as unstable rather than as a disagreement \
             between arms): {labels}"
        );
    }

    let hard_disagreements = disagreements.len();
    if !disagreements.is_empty() {
        let _ = writeln!(
            out,
            "\nANSWER DISAGREEMENTS (an arm returning a different answer is wrong, not fast):"
        );
        for (idx, answers) in disagreements {
            let rendered = answers
                .iter()
                .map(|(arm, rows, digest)| match digest {
                    Some(digest) => format!("{arm}={rows}r/{digest}"),
                    None => format!("{arm}={rows}r/undigestible"),
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(out, "  {}: {rendered}", label_of(idx));
        }
    }

    if !ambiguous.is_empty() {
        let _ = writeln!(
            out,
            "\nambiguous under LIMIT (different rows from a tie at the boundary; not a wrong answer):"
        );
        for (idx, answers) in ambiguous {
            let rendered = answers
                .iter()
                .map(|(arm, rows, _)| format!("{arm}={rows}r"))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(out, "  {}: {rendered}", label_of(idx));
        }
    }

    if !unchecked.is_empty() {
        let labels = unchecked
            .iter()
            .map(|idx| label_of(*idx))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            out,
            "\nchecked on row count only, result type could not be digested: {labels}"
        );
    }

    if show_digests {
        let _ = writeln!(out, "\nanswer digests:");
        for idx in 0..suite.queries.len() {
            let rendered = results
                .iter()
                .map(|r| match r.queries.get(idx).and_then(|q| q.digest) {
                    Some(digest) => digest.to_string(),
                    None => "----------------".to_string(),
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(out, "  {:<8} {rendered}", label_of(idx));
        }
    }

    let failures: Vec<_> = results
        .iter()
        .flat_map(|r| {
            r.queries.iter().filter_map(|q| {
                q.error
                    .as_ref()
                    .map(|e| format!("  {} {}: {e}", r.arm, label_of(q.query)))
            })
        })
        .collect();
    let failure_count = failures.len();
    if !failures.is_empty() {
        let _ = writeln!(out, "\nfailures:");
        for f in failures {
            let _ = writeln!(out, "{f}");
        }
    }

    Report {
        text: out,
        hard_disagreements,
        failures: failure_count,
    }
}

/// Whether `sql` ends in a `LIMIT` clause, which is what makes a cross-arm disagreement excusable.
///
/// # Why not `contains("limit")`
///
/// Because that is what this was, and it excuses far more than it should. A comment mentioning a
/// limit, a column called `limit_reached`, a string literal — any of them turned a genuine wrong
/// answer into a note, in a check whose entire purpose is to fail the run when two arms disagree.
/// Since a disagreement is the one thing this benchmark harness exists to catch, the excuse has to be narrow.
///
/// So the clause is looked for where SQL puts it: as a trailing `LIMIT <n>` or `LIMIT <n> OFFSET
/// <n>`, after comments and trailing punctuation are stripped, matched on word boundaries. This is
/// not a parser and does not need to be — a query whose `LIMIT` is nested inside a subquery is not
/// one this suite contains, and erring towards *reporting* a disagreement is the safe direction.
fn has_limit_clause(sql: &str) -> bool {
    let without_comments: String = sql
        .lines()
        .map(|line| line.split_once("--").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join(" ");
    let trimmed = without_comments.trim().trim_end_matches(';');
    let mut tokens = trimmed.split_whitespace().rev();

    let Some(last_number) = tokens.next() else {
        return false;
    };
    if !last_number.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let Some(word) = tokens.next() else {
        return false;
    };
    if word.eq_ignore_ascii_case("limit") {
        return true;
    }
    if !word.eq_ignore_ascii_case("offset") {
        return false;
    }
    tokens.next().is_some_and(|count| {
        count.chars().all(|c| c.is_ascii_digit())
            && tokens
                .next()
                .is_some_and(|word| word.eq_ignore_ascii_case("limit"))
    })
}

/// A rendered report, plus the facts a caller must act on rather than read.
///
/// The text used to be the whole return value, which meant a run where two arms returned *different
/// answers* printed a normal-looking table and exited zero. A disagreement is not a note: one of the
/// arms is wrong, and every timing in the table is then describing incomparable work.
pub struct Report {
    /// The human-readable table.
    pub text: String,
    /// Queries where arms disagreed on the answer and the SQL carries no `LIMIT` to excuse it.
    pub hard_disagreements: usize,
    /// Query/arm pairs that failed outright.
    pub failures: usize,
}

/// Queries faster than this are excluded from the stability statistics.
///
/// A sub-millisecond query's spread is dominated by scheduling and timer resolution, so pooling it
/// with a 200 ms scan would report a noise floor that describes neither.
const STABILITY_FLOOR: Duration = Duration::from_millis(20);

/// How far a cell's last sample must sit from its first to count as having moved at all.
///
/// Cells inside this band are evidence of neither direction. Without it, a run so steady that the
/// first and last samples land on the same number reports "0% ended slower" and trips the
/// one-sidedness check from the wrong end — which is the opposite of the truth.
const DRIFT_DEADBAND: f64 = 0.01;

/// How many cells must have moved before a one-sidedness proportion means anything.
///
/// A verdict computed from two or three moving cells is noise wearing a percentage. Ten is the point
/// where a run of all-one-direction stops being a coin-flip artefact (p < 0.001 under a fair coin)
/// and starts being a trend.
const DRIFT_MIN_CELLS: usize = 10;

fn write_stability(out: &mut String, results: &[ArmResults]) {
    use std::fmt::Write as _;

    // (spread, first, last) per query/arm cell with enough samples and enough duration to mean
    // something.
    let mut spreads: Vec<f64> = Vec::new();
    let mut slower_at_end = 0usize;
    let mut faster_at_end = 0usize;
    let mut drift_ratios: Vec<f64> = Vec::new();
    for arm in results {
        for q in &arm.queries {
            if q.samples.len() < 2 {
                continue;
            }
            let Some(best) = q.best else { continue };
            if best < STABILITY_FLOOR {
                continue;
            }
            let min = q.samples.iter().min().copied().unwrap_or(best);
            let max = q.samples.iter().max().copied().unwrap_or(best);
            if min > Duration::ZERO {
                spreads.push(max.as_secs_f64() / min.as_secs_f64());
            }
            let (first, last) = (q.samples[0], q.samples[q.samples.len() - 1]);
            if first > Duration::ZERO {
                let ratio = last.as_secs_f64() / first.as_secs_f64();
                // A deadband, so cells that merely landed on the same number are not counted as
                // evidence in either direction. Without it a perfectly steady run reports "0% ended
                // slower" and trips the one-sidedness test from the wrong end.
                if ratio > 1.0 + DRIFT_DEADBAND {
                    slower_at_end += 1;
                    drift_ratios.push(ratio);
                } else if ratio < 1.0 - DRIFT_DEADBAND {
                    faster_at_end += 1;
                    drift_ratios.push(ratio);
                }
            }
        }
    }

    if spreads.is_empty() {
        let _ = writeln!(
            out,
            "\nstability: not measurable (no query over {} ms ran more than one measured iteration)",
            STABILITY_FLOOR.as_millis()
        );
        return;
    }

    spreads.sort_by(f64::total_cmp);
    let median = spreads[spreads.len() / 2];
    let worst = spreads[spreads.len() - 1];
    let cells = drift_ratios.len();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a mean over a cell count; f64 is exact well past any real suite"
    )]
    let mean_drift = if cells == 0 {
        1.0
    } else {
        drift_ratios.iter().sum::<f64>() / cells as f64
    };

    let _ = writeln!(
        out,
        "\nstability (over the {} query/arm cells above {} ms): per-cell spread p50 {:.3}x, worst \
         {:.3}x. **An effect smaller than the p50 spread is not readable at this sample count.**",
        spreads.len(),
        STABILITY_FLOOR.as_millis(),
        median,
        worst
    );

    // One-sidedness is the signal, not the magnitude: noise puts the last sample above the first
    // about half the time, so a count far from half means the run trended rather than wandered.
    // Cells inside the deadband are counted in neither direction and excluded from the base, so a
    // steady run reports "0 cells moved" rather than a spurious 0%.
    let decided = slower_at_end + faster_at_end;
    if decided == 0 {
        let _ = writeln!(
            out,
            "drift: no cell moved by more than {:.0}% between its first and last sample",
            DRIFT_DEADBAND * 100.0
        );
        return;
    }
    // A proportion computed from a handful of cells is not evidence of a trend. Without this, one
    // moving cell among hundreds of steady ones reads as "100% ONE-SIDED" and would condemn a run
    // that was in fact unusually quiet — the diagnostic crying wolf on its own best case.
    if decided < DRIFT_MIN_CELLS {
        let _ = writeln!(
            out,
            "drift: only {decided} of {} cells moved by more than {:.0}%, too few to call a trend \
             (needs {DRIFT_MIN_CELLS})",
            spreads.len(),
            DRIFT_DEADBAND * 100.0
        );
        return;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a fraction of a cell count; f64 is exact well past any real suite"
    )]
    let fraction = slower_at_end as f64 / decided as f64;
    let verdict = if fraction >= 0.8 || fraction <= 0.2 {
        " — ONE-SIDED: this run trended rather than wandered. Iterations are measured consecutively \
         within an arm, so a trend is not charged evenly across arms; treat these ratios as suspect."
    } else {
        ""
    };
    let _ = writeln!(
        out,
        "drift: {slower_at_end} of {decided} moving cells ended slower than they started ({:.0}%), \
         mean last/first {mean_drift:.3}x{verdict}",
        fraction * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `LIMIT` excuse must fire on a real trailing clause and nothing else.
    ///
    /// This gate decides whether a cross-arm disagreement fails the run or is filed as a note, so a
    /// false positive here converts a wrong answer into a passing benchmark. It was a substring
    /// match, which every case below except the first two would have excused.
    #[test]
    fn only_a_trailing_limit_clause_excuses_a_disagreement() {
        assert!(has_limit_clause("SELECT a FROM t ORDER BY b LIMIT 10"));
        assert!(has_limit_clause("SELECT a FROM t ORDER BY b limit 25;"));
        assert!(has_limit_clause(
            "SELECT a FROM t ORDER BY b LIMIT 10 OFFSET 1000;"
        ));

        assert!(!has_limit_clause("SELECT a FROM t"));
        // A comment mentioning the word.
        assert!(!has_limit_clause("-- no limit here\nSELECT a FROM t"));
        // A column whose name contains it.
        assert!(!has_limit_clause("SELECT limit_reached FROM t"));
        // A string literal.
        assert!(!has_limit_clause("SELECT 'limit' FROM t"));
        // The keyword without a count is not a clause this suite produces.
        assert!(!has_limit_clause("SELECT a FROM t LIMIT"));
    }

    /// The rotation must be a permutation: every arm exactly once, per query.
    ///
    /// This is the one piece of the interleaving that can silently corrupt every ratio in a report —
    /// drop an arm and its column is short, repeat one and it is timed twice — and it is the only part
    /// testable without registered arms and a live DataFusion context.
    #[test]
    fn the_rotation_is_always_a_permutation() {
        for arms in 1..8 {
            for query_idx in 0..24 {
                let mut seen: Vec<usize> = arm_order(query_idx, arms).collect();
                assert_eq!(seen.len(), arms, "rotation dropped or duplicated an arm");
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(
                    seen.len(),
                    arms,
                    "rotation yielded the same arm twice for query {query_idx} of {arms} arms"
                );
            }
        }
    }

    /// Consecutive queries start from consecutive arms, which is what stops any arm from always
    /// paying the first-run cost inside a query.
    #[test]
    fn consecutive_queries_start_from_consecutive_arms() {
        let arms = 4;
        let firsts: Vec<usize> = (0..8)
            .map(|q| arm_order(q, arms).next().expect("non-empty"))
            .collect();
        assert_eq!(firsts, [0, 1, 2, 3, 0, 1, 2, 3]);
    }
}
