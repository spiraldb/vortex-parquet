// SPDX-License-Identifier: Apache-2.0

//! Query suites.
//!
//! A suite is a table name plus an ordered list of SQL queries. Nothing about it is arm-specific or
//! format-specific, which is the point: the same suite runs against every arm, so the only variable
//! is how bytes became Arrow batches.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;

/// A named set of queries over one table.
#[derive(Clone, Debug)]
pub struct Suite {
    /// The suite's name, used in reports.
    pub name: &'static str,
    /// The tables the queries reference, all of which must be registered before running.
    pub tables: Vec<String>,
    /// The queries, in order.
    pub queries: Vec<String>,
    /// What to call each query in reports, parallel to `queries`.
    pub labels: Vec<String>,
}

/// ClickBench: 43 queries over a single wide `hits` table.
///
/// The sharp instrument for benchmarking a *reader*: one table, no joins, dominated by scans,
/// projections, and selective filters — which is the surface the ladder actually controls. Numbers
/// here move when the reader changes.
///
/// The flip side, worth stating because it bounds what this benchmark harness can currently conclude: ClickBench is
/// flat and string-heavy, so it exercises nested projection not at all and numeric decode only
/// lightly. An arm that wins here has won on one workload.
/// Parses a suite file: one query per line, each optionally preceded by a `-- <label> <notes>` line.
///
/// The label is the first whitespace-delimited token after `--`; the rest of the comment is
/// documentation. A query with no preceding comment is labelled `q<index>`, which is what keeps
/// [`clickbench`] -- 43 bare lines -- working unchanged.
///
/// Keeping the label adjacent to its query is the point. A parallel list of labels in Rust drifts
/// silently the first time a query is inserted in the middle.
///
/// # Blank lines reset, and the first line of a comment block wins
///
/// Within an unbroken run of comment lines the *first* supplies the label, so a note can wrap without
/// the continuation becoming the label. A blank line clears the pending label, so a file header cannot
/// donate its first word to whatever query happens to follow. Both halves are load-bearing: each one
/// caught the other's bug while this was written.
fn parse_labelled(name: &'static str, tables: Vec<String>, sql: &str) -> Result<Suite> {
    let mut queries = Vec::new();
    let mut labels = Vec::new();
    let mut pending: Option<String> = None;

    for line in sql.lines().map(str::trim) {
        if line.is_empty() {
            pending = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("--") {
            if pending.is_none()
                && let Some(label) = rest.split_whitespace().next()
            {
                pending = Some(label.to_string());
            }
            continue;
        }
        labels.push(
            pending
                .take()
                .unwrap_or_else(|| format!("q{}", queries.len())),
        );
        queries.push(line.to_string());
    }

    anyhow::ensure!(!queries.is_empty(), "suite {name:?} parsed to zero queries");
    // Duplicate labels would make two report rows indistinguishable, and the report is the only place
    // these names are read.
    let mut seen = labels.clone();
    seen.sort_unstable();
    seen.dedup();
    anyhow::ensure!(
        seen.len() == labels.len(),
        "suite {name:?} has duplicate query labels"
    );

    Ok(Suite {
        name,
        tables,
        queries,
        labels,
    })
}

/// Filter selectivity and row-group prunability, varied independently.
///
/// Date, counter, and region filters project wide strings to exercise selection and pruning.
/// Observed selectivity and pruning depend on the supplied data and writer layout.
pub fn selectivity() -> Result<Suite> {
    parse_labelled(
        "selectivity",
        vec!["hits".to_string()],
        include_str!("../sql/selectivity.sql"),
    )
}

/// Projection width: how cost scales with the number of columns decoded, 1 to 105.
///
/// Uses `sum()` rather than `max()` throughout, deliberately: DataFusion's `AggregateStatistics` rule
/// can answer MIN/MAX from Parquet row-group statistics without decoding the column at all, which would
/// make a projection-width benchmark measure nothing.
pub fn projection() -> Result<Suite> {
    parse_labelled(
        "projection",
        vec!["hits".to_string()],
        include_str!("../sql/projection.sql"),
    )
}

/// String predicates, projected strings, and grouping workloads.
///
/// Equality without grouping exercises dictionary-aware filtering; grouping exercises aggregation
/// over the same fields. Dictionary cardinality depends on the supplied input.
pub fn strings() -> Result<Suite> {
    parse_labelled(
        "strings",
        vec!["hits".to_string()],
        include_str!("../sql/strings.sql"),
    )
}

pub fn clickbench() -> Suite {
    let queries: Vec<String> = include_str!("../sql/clickbench.sql")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();

    Suite {
        name: "clickbench",
        tables: vec!["hits".to_string()],
        labels: (0..queries.len()).map(|idx| format!("q{idx}")).collect(),
        queries,
    }
}

/// TPC-H: 22 queries over its eight-table schema.
///
/// SQL is loaded at runtime so the benchmark harness can consume the canonical query set from a Vortex benchmark
/// checkout without carrying a second copy that can drift. `sql_dir` contains `q1.sql` through
/// `q22.sql`.
pub fn tpch(sql_dir: &Path) -> Result<Suite> {
    let mut queries = Vec::with_capacity(22);
    for query in 1..=22 {
        let path = sql_dir.join(format!("q{query}.sql"));
        queries.push(
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read TPC-H query {}", path.display()))?,
        );
    }
    Ok(Suite {
        name: "tpch",
        tables: [
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        labels: (1..=22).map(|query| format!("q{query}")).collect(),
        queries,
    })
}

/// TPC-DS: 99 queries over its 24-table retail snowflake schema.
///
/// Vortex's benchmark checkout is the SQL authority, as it is for TPC-H. TPC-DS names its files
/// `01.sql` through `99.sql`; some files contain multiple statements and DataFusion executes the
/// complete string as one benchmark query.
pub fn tpcds(sql_dir: &Path) -> Result<Suite> {
    let mut queries = Vec::with_capacity(99);
    for query in 1..=99 {
        let path = sql_dir.join(format!("{query:02}.sql"));
        queries.push(
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read TPC-DS query {}", path.display()))?,
        );
    }
    Ok(Suite {
        name: "tpcds",
        tables: [
            "call_center",
            "catalog_page",
            "catalog_returns",
            "catalog_sales",
            "customer",
            "customer_address",
            "customer_demographics",
            "date_dim",
            "household_demographics",
            "income_band",
            "inventory",
            "item",
            "promotion",
            "reason",
            "ship_mode",
            "store",
            "store_returns",
            "store_sales",
            "time_dim",
            "warehouse",
            "web_page",
            "web_returns",
            "web_sales",
            "web_site",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        labels: (1..=99).map(|query| format!("q{query}")).collect(),
        queries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published suite is 43 queries; a short read of the SQL file would silently shrink the run.
    #[test]
    fn clickbench_has_every_query() {
        let suite = clickbench();
        assert_eq!(suite.queries.len(), 43);
        assert_eq!(suite.labels.len(), suite.queries.len());
    }
}
