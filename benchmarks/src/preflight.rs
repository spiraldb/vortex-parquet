// SPDX-License-Identifier: Apache-2.0

//! Check correctness, coverage, execution budgets, partitioning, and stability before larger runs.
//!
//! | check | the failure it catches |
//! |---|---|
//! | every query answers | an arm silently dropping out of the comparable set |
//! | digests agree across arms | an arm that is fast because it is wrong |
//! | per-query budget | an execution that exceeds the run budget |
//! | no arm collapses to one partition | work reshaped under a correct answer |
//! | the same cell twice, within its own floor | timing too unstable to compare |
//!
//! A pass establishes these checks only for the inputs, scale, and machine used. It does not
//! establish correctness or timing stability for other workloads.

use std::fmt::Write as _;
use std::time::Duration;

use anyhow::Result;
use anyhow::bail;

use crate::arm::ArmRef;
use crate::run;
use crate::run::ArmResults;
use crate::run::CacheState;
use crate::run::Iterations;
use crate::run::QueryBudget;
use crate::source::Source;
use crate::suite::Suite;

/// One suite, with the data it reads.
pub struct Case {
    /// The suite to run.
    pub suite: Suite,
    /// Table name to Parquet source, as [`run::run_suite`] expects.
    pub sources: Vec<(String, Source)>,
}

/// The thresholds the gate enforces.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Wall-clock ceiling on one query execution.
    ///
    /// Absolute rather than derived from the baseline, because the baseline has not run yet when the
    /// ceiling must apply — and because the defect it exists to catch is a query that never finishes,
    /// for which any generous constant works. Slow-but-finishing is caught by `ratio_limit` instead.
    pub budget: Duration,
    /// How many times the baseline an arm may take on a single query before the gate objects.
    ///
    /// Complements `budget` by catching an arm that finishes but is pathologically slow.
    pub ratio_limit: f64,
    /// Iterations per cell. Two measured is the minimum that yields a spread at all.
    pub iterations: Iterations,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            // Generous on purpose. At the scale a gate runs, nothing healthy comes near this, and a
            // tight ceiling would fail slow machines rather than broken arms.
            budget: Duration::from_secs(60),
            ratio_limit: 20.0,
            iterations: Iterations {
                warmup: 1,
                measured: 2,
            },
        }
    }
}

/// Cells faster than this are exempt from the reproducibility check.
///
/// The same floor [`run`] applies to its own stability statistics, and for the same reason: a
/// sub-20 ms query's run-to-run difference is scheduling noise, and holding it to a ratio would fail
/// the gate on the queries that carry the least information.
const REPRODUCIBILITY_FLOOR: Duration = Duration::from_millis(20);

/// The smallest run-to-run difference the gate will ever call a failure.
///
/// A cell whose two samples happen to land on nearly the same number reports a spread near 1.0, and
/// holding the *next* run to that would fail on noise the instrument cannot resolve. So a cell's own
/// spread is a floor on the tolerance, never a ceiling below this.
const MIN_REPRODUCIBILITY_TOLERANCE: f64 = 1.25;

/// What fraction of cells may fail to reproduce before the *machine* is the problem.
///
/// # Why one drifting cell is not a finding
///
/// Because it is the expected outcome. Across a hundred cells on a real machine, a couple will land
/// outside any tolerance you pick — a page fault, a scheduler decision, another process waking up.
/// Failing the gate on that means the gate fails almost every time it is run, and a check that always
/// fires is a check nobody reads. The first version of this did exactly that: it reported 2 drifting
/// cells out of 100 as a reason not to launch, on a machine that was in fact steady.
///
/// So the question the check actually asks is the one worth asking — *is this machine steady enough
/// to measure on* — and that is a property of the proportion, not of any single cell. This mirrors the
/// one-sidedness test in [`run`], which refuses to call a trend from too few moving cells for the same
/// reason.
const MAX_UNREPRODUCIBLE_FRACTION: f64 = 0.10;

/// How many cells must be checked before a proportion of them means anything.
///
/// Below this, "20% of cells drifted" is one cell out of five.
const MIN_REPRODUCIBILITY_CELLS: usize = 20;

/// One thing that went wrong, in the words the operator needs to act on it.
struct Finding {
    check: &'static str,
    detail: String,
}

/// Runs every case against every arm and reports whether a campaign may launch.
///
/// Returns the rendered report and the findings; an empty finding list is a pass.
pub async fn gate(cases: &[Case], arms: &[ArmRef], thresholds: Thresholds) -> Result<Report> {
    if arms.len() < 2 {
        bail!(
            "the gate compares arms against each other, so it needs at least two; got {}",
            arms.len()
        );
    }

    let mut out = String::new();
    let mut findings: Vec<Finding> = Vec::new();

    let _ = writeln!(
        out,
        "preflight: {} suite(s) x {} arms, budget {:.0}s/query, ratio limit {:.0}x, {} warmup + {} \
         measured, run twice",
        cases.len(),
        arms.len(),
        thresholds.budget.as_secs_f64(),
        thresholds.ratio_limit,
        thresholds.iterations.warmup,
        thresholds.iterations.measured,
    );
    let _ = writeln!(
        out,
        "arms: {}  (baseline: {})",
        arms.iter().map(|a| a.name()).collect::<Vec<_>>().join(", "),
        arms[0].name(),
    );

    for case in cases {
        let _ = writeln!(out, "\n--- {} ---", case.suite.name);
        check_partitioning(case, arms, &mut out, &mut findings).await?;
        check_answers_and_timings(case, arms, thresholds, &mut out, &mut findings).await?;
    }

    let _ = writeln!(out);
    if findings.is_empty() {
        let _ = writeln!(
            out,
            // Deliberately does NOT say "every cell reproduced": the reproducibility check passes on a
            // proportion, so a pass is consistent with a few cells having moved, and the per-suite
            // lines above give the count. Overstating a pass is the same species of error as
            // overstating a result.
            "PREFLIGHT PASSED. Every suite answered on every arm, digests agreed, nothing hung, every \
             arm that could demonstrate parallelism did, and the drifting-cell fraction stayed under \
             the limit (per-suite counts above). Any `note:` lines above are plan-shape differences \
             that measured as harmless, not clearances. This clears the five defects the gate knows \
             about at this scale — it does not certify the campaign's numbers."
        );
    } else {
        let _ = writeln!(out, "PREFLIGHT FAILED — {} finding(s):", findings.len());
        for finding in &findings {
            let _ = writeln!(out, "  [{}] {}", finding.check, finding.detail);
        }
        let _ = writeln!(
            out,
            "\nDo not launch a campaign. Every one of these is cheaper to fix here than to discover \
             in a result."
        );
    }

    Ok(Report {
        text: out,
        failures: findings.len(),
    })
}

/// A rendered gate result.
pub struct Report {
    /// The human-readable text.
    pub text: String,
    /// How many checks failed. Zero is a pass.
    pub failures: usize,
}

/// Fails any arm whose scan collapses to a single partition where the baseline parallelises.
///
/// Deliberately narrow. Two arms legitimately plan different partition counts — a different file
/// layout splits differently, and holding them to equality would fail the gate on a working change.
/// Actual counts are printed so a reader can inspect differences that do not fail the gate.
async fn check_partitioning(
    case: &Case,
    arms: &[ArmRef],
    out: &mut String,
    findings: &mut Vec<Finding>,
) -> Result<()> {
    let contexts = run::registered_contexts(arms, &case.sources).await?;
    let mut collapsed = 0usize;
    let mut planned = 0usize;
    let mut widest = (0usize, 0usize);
    let mut unplannable = 0usize;

    for (idx, sql) in case.suite.queries.iter().enumerate() {
        let label = case
            .suite
            .labels
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("q{idx}"));

        let baseline = match run::leaf_partitions(&contexts[0], sql).await {
            Ok(counts) => counts,
            // A query the *baseline* cannot plan is a broken suite, not a broken arm, and it will be
            // caught properly by the execution check below. Counted rather than failed here so the
            // two checks do not report the same defect twice in different words.
            Err(_) => {
                unplannable += 1;
                continue;
            }
        };
        let baseline_max = baseline.iter().copied().max().unwrap_or(0);
        planned += 1;
        // Remembered for the productive-partition probe below. A query whose scan is one partition
        // cannot demonstrate anything about parallelism, and the first query in a suite is often
        // exactly that -- ClickBench q0 is `SELECT COUNT(*)`.
        if baseline_max > widest.0 {
            widest = (baseline_max, idx);
        }
        if baseline_max <= 1 {
            // Nothing to be narrower than.
            continue;
        }

        for (arm, ctx) in arms.iter().zip(&contexts).skip(1) {
            let Ok(counts) = run::leaf_partitions(ctx, sql).await else {
                unplannable += 1;
                continue;
            };
            let arm_max = counts.iter().copied().max().unwrap_or(0);
            if arm_max <= 1 {
                collapsed += 1;
                // A single scan output is a note: DataFusion may redistribute its batches above the
                // scan. The productive-partition probe below determines whether this gate fails.
                let _ = writeln!(
                    out,
                    "  note: {} plans {label} with a single scan partition where {} plans {} \
                     (not a failure on its own -- see the productive-partition probe)",
                    arm.name(),
                    arms[0].name(),
                    baseline_max,
                );
            }
        }
    }

    // Advertised partitions are a claim; this executes one query's scan leaves and counts which ones
    // actually carry rows. An arm that advertises N and fills 1 is serialised while looking parallel,
    // which the check above cannot see — it is exactly how `vortex-scan-api` passed while doing all
    // its work in a single stream.
    if let Some(sql) = case.suite.queries.get(widest.1).filter(|_| widest.0 > 1) {
        let mut counts = Vec::new();
        for (arm, ctx) in arms.iter().zip(&contexts) {
            match run::productive_partitions(ctx, sql).await {
                Ok((advertised, productive)) => {
                    let _ = writeln!(
                        out,
                        "  productive partitions [{}]: {productive} of {advertised} advertised",
                        arm.name()
                    );
                    counts.push((arm.name().to_string(), advertised, productive));
                }
                Err(_) => continue,
            }
        }

        // Compare productive counts only when the baseline has enough work to fill multiple outputs.
        // Highly selective queries may leave most outputs empty because of pruning.
        match counts.split_first() {
            Some(((_, baseline_advertised, baseline_productive), arms_under_test))
                if *baseline_productive * 2 >= *baseline_advertised && *baseline_productive > 1 =>
            {
                for (name, advertised, productive) in arms_under_test {
                    if *advertised > 1 && *productive <= 1 {
                        findings.push(Finding {
                            check: "partitioning",
                            detail: format!(
                                "{}: {name} advertises {advertised} scan partitions but only \
                                 {productive} produced any rows, where the baseline filled \
                                 {baseline_productive} of {baseline_advertised} — this arm \
                                 executes serially while reporting parallelism, so its timings are \
                                 not comparable to an arm that really parallelises",
                                case.suite.name,
                            ),
                        });
                    }
                }
            }
            // Reported rather than skipped silently: "no finding" from a probe that could not
            // discriminate must not read like "no finding" from one that could.
            Some(((_, advertised, productive), _)) => {
                let _ = writeln!(
                    out,
                    "  (productive-partition probe INCONCLUSIVE: the baseline itself filled only \
                     {productive} of {advertised}, so this query cannot demonstrate parallelism)"
                );
            }
            None => {}
        }
    }

    let _ = writeln!(
        out,
        "partitioning: {planned} queries planned on every arm, {collapsed} collapsed to a single \
         partition{}",
        if unplannable > 0 {
            format!(", {unplannable} could not be planned (reported by the execution check)")
        } else {
            String::new()
        }
    );
    Ok(())
}

/// Runs the case twice and checks answers, budget, relative cost, and reproducibility.
async fn check_answers_and_timings(
    case: &Case,
    arms: &[ArmRef],
    thresholds: Thresholds,
    out: &mut String,
    findings: &mut Vec<Finding>,
) -> Result<()> {
    let budget = Some(QueryBudget {
        per_query: thresholds.budget,
        fatal: true,
    });

    // Two independent runs of the same cells. The second is not a repeat for confidence — it is the
    // reproducibility check, and it is the one check here that tests the *machine* rather than the
    // code. A box that cannot reproduce its own numbers cannot measure a 5% effect on rented time.
    let mut runs = Vec::with_capacity(2);
    for pass in 0..2 {
        let result = run::run_suite(
            &case.suite,
            arms,
            &case.sources,
            thresholds.iterations,
            // The state a reader comparison runs in, so the gate exercises the same cache path a
            // campaign will. `cold-disk` is Linux-only and a gate must run on the laptop too.
            CacheState::Flushed,
            budget,
        )
        .await;
        match result {
            Ok(results) => runs.push(results),
            // A fatal budget breach aborts the run, which is the point — but it must fail the *gate*
            // with the query named, not propagate as an opaque error.
            Err(e) => {
                findings.push(Finding {
                    check: "budget",
                    detail: format!("{} (pass {}): {e}", case.suite.name, pass + 1),
                });
                let _ = writeln!(out, "execution: ABORTED on pass {} — {e}", pass + 1);
                return Ok(());
            }
        }
    }

    let report = run::report(&case.suite, &runs[0], false);
    if report.failures > 0 {
        findings.push(Finding {
            check: "failures",
            detail: format!(
                "{}: {} query/arm pair(s) failed outright, which silently shrinks the set the \
                 geomeans cover",
                case.suite.name, report.failures
            ),
        });
    }
    if report.hard_disagreements > 0 {
        findings.push(Finding {
            check: "answers",
            detail: format!(
                "{}: {} query/queries where arms returned different answers with no LIMIT to excuse \
                 it — one of the arms is wrong, not fast",
                case.suite.name, report.hard_disagreements
            ),
        });
    }

    check_ratio(case, &runs[0], thresholds, findings);
    let repro = check_reproducibility(case, &runs[0], &runs[1]);
    judge_reproducibility(case, &repro, findings);

    // The drift count is printed whether or not it crossed the threshold. A gate that reports only
    // its verdict hides the trend that precedes a verdict — 9% one week and 11% the next is one
    // machine getting worse, and only the first of those is silent.
    let _ = writeln!(
        out,
        "execution: {} failure(s), {} disagreement(s); {} of {} cells did not reproduce{}",
        report.failures,
        report.hard_disagreements,
        repro.drifted.len(),
        repro.checked,
        if repro.checked < MIN_REPRODUCIBILITY_CELLS {
            format!(" (under {MIN_REPRODUCIBILITY_CELLS} cells, too few to judge a proportion)")
        } else {
            String::new()
        },
    );
    Ok(())
}

/// Fails any arm that is more than `ratio_limit` times the baseline on a single query.
fn check_ratio(
    case: &Case,
    results: &[ArmResults],
    thresholds: Thresholds,
    findings: &mut Vec<Finding>,
) {
    let Some(baseline_arm) = results.first() else {
        return;
    };
    for (idx, _) in case.suite.queries.iter().enumerate() {
        let Some(baseline) = baseline_arm.queries.get(idx).and_then(|q| q.best) else {
            continue;
        };
        // Below the floor a ratio is a comparison of two scheduling accidents.
        if baseline < REPRODUCIBILITY_FLOOR {
            continue;
        }
        let label = case
            .suite
            .labels
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("q{idx}"));
        for arm in results.iter().skip(1) {
            let Some(best) = arm.queries.get(idx).and_then(|q| q.best) else {
                continue;
            };
            let ratio = best.as_secs_f64() / baseline.as_secs_f64();
            if ratio > thresholds.ratio_limit {
                findings.push(Finding {
                    check: "ratio",
                    detail: format!(
                        "{}: {} takes {ratio:.1}x the baseline on {label} ({:.0} ms vs {:.0} ms), \
                         over the {:.0}x limit — slow enough to be a defect rather than a result",
                        case.suite.name,
                        arm.arm,
                        best.as_secs_f64() * 1e3,
                        baseline.as_secs_f64() * 1e3,
                        thresholds.ratio_limit,
                    ),
                });
            }
        }
    }
}

/// How many cells reproduced between two runs of the same suite.
struct Reproducibility {
    /// Cells eligible to be checked at all.
    checked: usize,
    /// Cells whose second run landed outside their own spread, with a description of each.
    drifted: Vec<String>,
}

/// Checks that the same cells, run twice, landed in the same place.
///
/// Two thresholds, doing two different jobs. The **per-cell** tolerance is that cell's own spread
/// across its samples, not a fixed percentage — a cell that wobbles 30% within a run cannot honestly
/// be held to 5% between runs, and that spread is the same quantity the report already publishes as
/// the floor an effect must exceed to be readable. The **verdict** is then taken over the proportion
/// of cells that drifted, not over any single one, because the question is whether the machine is
/// steady enough to measure on and one restless cell out of a hundred does not answer it.
fn check_reproducibility(
    case: &Case,
    first: &[ArmResults],
    second: &[ArmResults],
) -> Reproducibility {
    let mut checked = 0usize;
    let mut drifted = Vec::new();
    for (a, b) in first.iter().zip(second) {
        for (qa, qb) in a.queries.iter().zip(&b.queries) {
            let (Some(best_a), Some(best_b)) = (qa.best, qb.best) else {
                continue;
            };
            if best_a < REPRODUCIBILITY_FLOOR || best_b < REPRODUCIBILITY_FLOOR {
                continue;
            }
            // Widest spread either run observed within itself.
            let spread = |q: &run::QueryResult| {
                let min = q.samples.iter().min().copied();
                let max = q.samples.iter().max().copied();
                match (min, max) {
                    (Some(min), Some(max)) if min > Duration::ZERO => {
                        max.as_secs_f64() / min.as_secs_f64()
                    }
                    _ => 1.0,
                }
            };
            let tolerance = spread(qa)
                .max(spread(qb))
                .max(MIN_REPRODUCIBILITY_TOLERANCE);
            checked += 1;

            let ratio = best_b.as_secs_f64() / best_a.as_secs_f64();
            if ratio > tolerance || ratio < 1.0 / tolerance {
                let label = case
                    .suite
                    .labels
                    .get(qa.query)
                    .cloned()
                    .unwrap_or_else(|| format!("q{}", qa.query));
                drifted.push(format!(
                    "{} on {label} ran {:.0} ms then {:.0} ms ({ratio:.2}x, own tolerance \
                     {tolerance:.2}x)",
                    a.arm,
                    best_a.as_secs_f64() * 1e3,
                    best_b.as_secs_f64() * 1e3,
                ));
            }
        }
    }
    Reproducibility { checked, drifted }
}

/// Turns a drift count into a verdict on the machine.
///
/// Separate from the measuring so the rule is visible on its own: a finding is raised only when
/// *enough* cells were checked to make a proportion meaningful **and** the proportion is high enough
/// to mean the machine rather than the weather.
fn judge_reproducibility(case: &Case, repro: &Reproducibility, findings: &mut Vec<Finding>) {
    if repro.checked < MIN_REPRODUCIBILITY_CELLS || repro.drifted.is_empty() {
        return;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a fraction of a cell count; f64 is exact well past any real suite"
    )]
    let fraction = repro.drifted.len() as f64 / repro.checked as f64;
    if fraction <= MAX_UNREPRODUCIBLE_FRACTION {
        return;
    }
    findings.push(Finding {
        check: "reproducibility",
        detail: format!(
            "{}: {} of {} cells ({:.0}%) did not reproduce between two runs, over the {:.0}% limit — \
             this machine cannot currently resolve effects of the size a campaign is looking for, so \
             it would measure its own noise. Examples: {}",
            case.suite.name,
            repro.drifted.len(),
            repro.checked,
            fraction * 100.0,
            MAX_UNREPRODUCIBLE_FRACTION * 100.0,
            repro
                .drifted
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; "),
        ),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::QueryResult;

    /// A suite of `n` queries labelled `q0..`, with no sources: the checks under test read labels and
    /// nothing else off it.
    fn suite_of(n: usize) -> Case {
        Case {
            suite: Suite {
                name: "test",
                tables: vec!["hits".to_string()],
                queries: (0..n).map(|i| format!("SELECT {i}")).collect(),
                labels: (0..n).map(|i| format!("q{i}")).collect(),
            },
            sources: Vec::new(),
        }
    }

    /// One arm's results, one query, from an explicit list of sample milliseconds.
    fn arm(name: &str, samples_ms: &[u64]) -> ArmResults {
        let samples: Vec<Duration> = samples_ms
            .iter()
            .map(|ms| Duration::from_millis(*ms))
            .collect();
        ArmResults {
            arm: name.to_string(),
            queries: vec![QueryResult {
                query: 0,
                best: samples.iter().min().copied(),
                samples,
                rows: Some(1),
                digest: None,
                unstable: false,
                timed_out: false,
                error: None,
            }],
        }
    }

    /// The ratio check must fire on a pathologically slow arm and stay silent on a merely slower one.
    ///
    /// Ordinary timing differences should remain reportable without failing the preflight gate.
    #[test]
    fn the_ratio_check_fires_only_on_a_pathological_arm() {
        let case = suite_of(1);
        let thresholds = Thresholds {
            ratio_limit: 20.0,
            ..Thresholds::default()
        };

        let mut findings = Vec::new();
        check_ratio(
            &case,
            &[arm("baseline", &[100]), arm("slower", &[250])],
            thresholds,
            &mut findings,
        );
        assert!(findings.is_empty(), "2.5x is a result, not a defect");

        check_ratio(
            &case,
            &[arm("baseline", &[100]), arm("pathological", &[5_000])],
            thresholds,
            &mut findings,
        );
        assert_eq!(findings.len(), 1, "50x should fail the gate");
        assert!(findings[0].detail.contains("pathological"));
    }

    /// Cells under the floor are exempt, because a ratio between two sub-millisecond timings is a
    /// comparison of scheduling accidents and would fail the gate on the least informative queries.
    #[test]
    fn the_ratio_check_ignores_cells_below_the_floor() {
        let mut findings = Vec::new();
        check_ratio(
            &suite_of(1),
            // 1 ms vs 100 ms is a 100x ratio, and it means nothing at this magnitude.
            &[arm("baseline", &[1]), arm("noisy", &[100])],
            Thresholds::default(),
            &mut findings,
        );
        assert!(findings.is_empty());
    }

    /// Reproducibility is judged against the cell's *own* spread, not a fixed percentage.
    ///
    /// The point of the check is to refuse to measure on a machine that cannot hold still. A cell that
    /// already wobbles 3x within a single run cannot be held to a tighter bar between runs — doing so
    /// would report noise as a finding, which is how a diagnostic stops being believed.
    #[test]
    fn reproducibility_tolerates_exactly_the_cell_that_is_already_noisy() {
        let case = suite_of(1);

        // A steady cell that then moved 3x: over the 1.25x minimum tolerance, so it counts as drift.
        let repro = check_reproducibility(
            &case,
            &[arm("steady", &[100, 101])],
            &[arm("steady", &[300, 303])],
        );
        assert_eq!(repro.checked, 1);
        assert_eq!(
            repro.drifted.len(),
            1,
            "a 3x shift on a steady cell drifted"
        );

        // The same 3x shift on a cell that already spans 4x within each run is inside its own floor.
        let repro = check_reproducibility(
            &case,
            &[arm("noisy", &[100, 400])],
            &[arm("noisy", &[300, 1200])],
        );
        assert_eq!(repro.checked, 1);
        assert!(
            repro.drifted.is_empty(),
            "a cell cannot be held to a tighter bar between runs than it holds within one"
        );
    }

    /// A couple of restless cells among many is a normal machine, not a reason to refuse to measure.
    ///
    /// This is the case that made the check unusable in its first form: a real ClickBench gate run
    /// reported 2 drifting cells out of 100 — a steady machine by any reading — as grounds not to
    /// launch. A check that fires on its own best case teaches its operator to skip it.
    #[test]
    fn a_few_restless_cells_do_not_condemn_a_steady_machine() {
        let case = suite_of(1);
        let mut findings = Vec::new();
        judge_reproducibility(
            &case,
            &Reproducibility {
                checked: 100,
                drifted: (0..2).map(|i| format!("cell {i}")).collect(),
            },
            &mut findings,
        );
        assert!(findings.is_empty(), "2% is a machine, not a fault");

        // A quarter of the cells moving is a different statement entirely.
        judge_reproducibility(
            &case,
            &Reproducibility {
                checked: 100,
                drifted: (0..25).map(|i| format!("cell {i}")).collect(),
            },
            &mut findings,
        );
        assert_eq!(findings.len(), 1, "25% means the machine cannot hold still");
    }

    /// A proportion taken over a handful of cells is not a proportion.
    ///
    /// Without the minimum, a two-query suite where one cell moved reports "50% did not reproduce"
    /// and fails the gate on a sample size that cannot support the claim.
    #[test]
    fn a_proportion_needs_enough_cells_to_be_one() {
        let mut findings = Vec::new();
        judge_reproducibility(
            &suite_of(1),
            &Reproducibility {
                checked: 4,
                drifted: vec!["cell 0".to_string(), "cell 1".to_string()],
            },
            &mut findings,
        );
        assert!(
            findings.is_empty(),
            "50% of four cells is not evidence about a machine"
        );
    }

    /// A cell that reproduces perfectly must not be failed by its own tiny spread.
    ///
    /// Without a minimum tolerance, two samples landing on nearly the same number produce a ~1.00x
    /// spread, and then any run-to-run difference at all is "outside its own floor" — the check
    /// condemning its own best case, which is the bug the drift deadband exists to prevent elsewhere.
    #[test]
    fn a_steady_cell_is_not_failed_by_being_steady() {
        let repro = check_reproducibility(
            &suite_of(1),
            &[arm("steady", &[100, 100])],
            &[arm("steady", &[110, 110])],
        );
        assert!(
            repro.drifted.is_empty(),
            "10% on a steady cell is within the minimum tolerance"
        );
    }

    /// A cell that failed in either run is skipped rather than counted as reproduced.
    ///
    /// Counted, because "0 findings" from a gate that checked nothing reads identically to "0
    /// findings" from a gate that checked everything, and only one of them is a pass.
    #[test]
    fn cells_that_did_not_run_are_not_counted_as_reproduced() {
        let failed = ArmResults {
            arm: "broken".to_string(),
            queries: vec![QueryResult {
                query: 0,
                best: None,
                samples: Vec::new(),
                rows: None,
                digest: None,
                unstable: false,
                timed_out: true,
                error: Some("budget".to_string()),
            }],
        };
        let repro = check_reproducibility(&suite_of(1), &[arm("broken", &[100, 100])], &[failed]);
        assert_eq!(repro.checked, 0);
        assert!(
            repro.drifted.is_empty(),
            "an absent cell is the failure check's business, not this one"
        );
    }
}
