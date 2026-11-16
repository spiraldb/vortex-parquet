// SPDX-License-Identifier: Apache-2.0

//! Counters that say which decode path a scan actually took.
//!
//! # Why this exists
//!
//! Whether page arrays can represent a field depends on its physical type and Dremel levels. Nested
//! fields fall back to arrow-rs at the field boundary and produce the same values, so result equality
//! alone cannot say how much of a workload reached page-native execution.
//!
//! These counters make the path assertable. They are not a performance-measurement tool — the benchmark harness
//! times queries, and an atomic increment per chunk would be a poor timer anyway. They answer one
//! question: "did the page path run?"

use parking_lot::Mutex;

/// Cumulative counts of which decode path each column chunk took.
///
/// Shared by every file one [`ParquetFormat`](crate::ParquetFormat) opens, and cumulative over its
/// lifetime.
/// One consistent reading of every counter.
///
/// Returned by [`ScanMetrics::snapshot`] so that quantities which are only meaningful *together* are
/// read together. `selective_rows` over `selective_rows_available` is a ratio; two independent atomic
/// loads can straddle a concurrent update and yield a ratio that never existed — including one above
/// 100%. Reading the set under a single lock makes every reported figure describe one moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanCounts {
    /// Column chunks decoded by the page path.
    pub page_decodes: u64,
    /// Column chunks decoded through arrow-rs.
    pub arrow_decodes: u64,
    /// Column chunks decoded through a pushed-down row selection.
    pub selective_decodes: u64,
    /// Rows kept by those selective decodes.
    pub selective_rows: u64,
    /// Rows those selective decodes would have decoded without a selection.
    pub selective_rows_available: u64,
    /// Offset indexes decoded lazily.
    pub offset_index_decodes: u64,
    /// Offset/column-index pairs decoded for statistics pruning.
    pub page_index_decodes: u64,
    /// Split-block Bloom filters decoded for pruning.
    pub bloom_filter_decodes: u64,
    /// Page filters executed via parquet-rs decoder skipping.
    pub decoder_filters: u64,
    /// LIKE predicates fused with Parquet decoding.
    pub fused_like_predicates: u64,
    /// Comparison predicates fused with Parquet decoding.
    pub fused_compare_predicates: u64,
}

/// Cumulative counts of which decode path each column chunk took.
///
/// Shared by every file one [`ParquetFormat`](crate::ParquetFormat) opens, and cumulative over its
/// lifetime.
///
/// # Why one lock rather than a counter each
///
/// Independent atomics make each *counter* correct and every *relationship between* counters
/// unreliable. Two hazards, both observed rather than theorised:
///
/// * A ratio read as two loads can mix a pre- and post-update value, so `selective_rows` could exceed
///   `selective_rows_available`.
/// * [`ScanMetrics::reset`] cleared counters one at a time, so a reader could see a half-cleared set —
///   a nonzero row count against zero decodes.
///
/// These counters are recorded per column chunk, not per row, so a lock here is not on any hot path.
#[derive(Debug, Default)]
pub struct ScanMetrics {
    counts: Mutex<ScanCounts>,
}

impl ScanMetrics {
    /// One consistent reading of every counter.
    pub fn snapshot(&self) -> ScanCounts {
        *self.counts.lock()
    }

    /// Records a chunk decoded by the page path, keeping its Parquet encoding.
    pub fn record_page_decode(&self) {
        self.counts.lock().page_decodes += 1;
    }

    /// Records a chunk decoded through arrow-rs, whether by configuration or by the page path
    /// declining it.
    pub fn record_arrow_decode(&self) {
        self.counts.lock().arrow_decodes += 1;
    }

    /// Records a chunk decoded through a pushed-down row selection rather than in full.
    ///
    /// `selected` and `available` are counted alongside the call because the interesting quantity is
    /// not how often the path ran but how much of the chunk it skipped: a selective decode of 99% of
    /// the rows is the path running and buying nothing.
    ///
    /// All three move under one lock, so the ratio is never observable mid-update.
    pub fn record_selective_decode(&self, selected: u64, available: u64) {
        let mut counts = self.counts.lock();
        counts.selective_decodes += 1;
        counts.selective_rows += selected;
        counts.selective_rows_available += available;
    }

    /// How many column chunks were decoded through a pushed-down row selection.
    pub fn selective_decodes(&self) -> u64 {
        self.snapshot().selective_decodes
    }

    /// Rows kept, and rows that would have been decoded without a selection.
    pub fn selective_rows(&self) -> (u64, u64) {
        let counts = self.snapshot();
        (counts.selective_rows, counts.selective_rows_available)
    }

    /// Records lazily decoding an offset index after a selective scan requests it.
    pub fn record_offset_index_decode(&self) {
        self.counts.lock().offset_index_decodes += 1;
    }

    /// Number of offset indexes decoded lazily.
    pub fn offset_index_decodes(&self) -> u64 {
        self.snapshot().offset_index_decodes
    }

    /// Records lazily decoding an offset/column-index pair for statistics pruning.
    pub fn record_page_index_decode(&self) {
        self.counts.lock().page_index_decodes += 1;
    }

    /// Number of page-index pairs decoded lazily for statistics pruning.
    pub fn page_index_decodes(&self) -> u64 {
        self.snapshot().page_index_decodes
    }

    /// Records lazily decoding a split-block Bloom filter for pruning.
    pub fn record_bloom_filter_decode(&self) {
        self.counts.lock().bloom_filter_decodes += 1;
    }

    /// Records filtering a page with parquet-rs decoder `skip`/`get`.
    pub fn record_decoder_filter(&self) {
        self.counts.lock().decoder_filters += 1;
    }

    /// Number of page filters executed without first materializing the full page.
    pub fn decoder_filters(&self) -> u64 {
        self.snapshot().decoder_filters
    }

    /// Records evaluating LIKE directly over decoded Parquet byte values.
    pub fn record_fused_like_predicate(&self) {
        self.counts.lock().fused_like_predicates += 1;
    }

    /// Number of LIKE page predicates fused with Parquet decoding.
    pub fn fused_like_predicates(&self) -> u64 {
        self.snapshot().fused_like_predicates
    }

    /// Records evaluating a comparison directly over decoded Parquet byte values.
    pub fn record_fused_compare_predicate(&self) {
        self.counts.lock().fused_compare_predicates += 1;
    }

    /// Number of comparison page predicates fused with Parquet decoding.
    pub fn fused_compare_predicates(&self) -> u64 {
        self.snapshot().fused_compare_predicates
    }

    /// Number of Bloom filters decoded lazily.
    pub fn bloom_filter_decodes(&self) -> u64 {
        self.snapshot().bloom_filter_decodes
    }

    /// How many column chunks the page path decoded.
    pub fn page_decodes(&self) -> u64 {
        self.snapshot().page_decodes
    }

    /// How many column chunks went through arrow-rs.
    pub fn arrow_decodes(&self) -> u64 {
        self.snapshot().arrow_decodes
    }

    /// Drops every count back to zero, all at once.
    ///
    /// One assignment under the lock rather than a store per counter, so a concurrent reader can never
    /// observe the reset half-applied.
    pub fn reset(&self) {
        *self.counts.lock() = ScanCounts::default();
    }
}

impl std::fmt::Display for ScanMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ONE snapshot for the whole line. Reading each counter separately here let the printed
        // percentages disagree with the counts printed beside them.
        let c = self.snapshot();
        let total = c.page_decodes + c.arrow_decodes;
        let pct = if total == 0 {
            0.0
        } else {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a percentage of a chunk count; f64 is exact well past any real file"
            )]
            {
                c.page_decodes as f64 * 100.0 / total as f64
            }
        };
        write!(
            f,
            "{}/{total} chunks decoded by the page path ({pct:.1}%), {} through arrow-rs",
            c.page_decodes, c.arrow_decodes
        )?;

        if c.selective_decodes > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a percentage of a row count; f64 is exact well past any real file"
            )]
            let kept_pct = if c.selective_rows_available == 0 {
                0.0
            } else {
                c.selective_rows as f64 * 100.0 / c.selective_rows_available as f64
            };
            write!(
                f,
                "; {} selective decodes keeping {}/{} rows ({kept_pct:.2}%)",
                c.selective_decodes, c.selective_rows, c.selective_rows_available
            )?;
        }
        if c.offset_index_decodes > 0 {
            write!(
                f,
                "; {} offset indexes decoded lazily",
                c.offset_index_decodes
            )?;
        }
        if c.page_index_decodes > 0 {
            write!(
                f,
                "; {} page-index pairs decoded for pruning",
                c.page_index_decodes
            )?;
        }
        if c.bloom_filter_decodes > 0 {
            write!(
                f,
                "; {} Bloom filters decoded for pruning",
                c.bloom_filter_decodes
            )?;
        }
        if c.decoder_filters > 0 {
            write!(
                f,
                "; {} page filters used decoder skipping",
                c.decoder_filters
            )?;
        }
        if c.fused_like_predicates + c.fused_compare_predicates > 0 {
            write!(
                f,
                "; {} LIKE and {} compare predicates fused with decoding",
                c.fused_like_predicates, c.fused_compare_predicates
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters must distinguish the two paths, or an arm that never fires looks like one that did.
    #[test]
    fn the_counters_separate_the_two_paths() {
        let metrics = ScanMetrics::default();
        metrics.record_page_decode();
        metrics.record_page_decode();
        metrics.record_arrow_decode();

        assert_eq!(metrics.page_decodes(), 2);
        assert_eq!(metrics.arrow_decodes(), 1);
        assert!(metrics.to_string().contains("2/3"));
        assert!(metrics.to_string().contains("66.7%"));

        metrics.reset();
        assert_eq!(metrics.page_decodes(), 0);
        assert_eq!(metrics.arrow_decodes(), 0);
    }

    /// A scan that decoded nothing must not divide by zero.
    #[test]
    fn an_empty_scan_reports_zero_rather_than_nan() {
        assert!(ScanMetrics::default().to_string().contains("0/0"));
    }

    /// A snapshot must never catch a record half-applied.
    ///
    /// The sequential version of this passes against independent atomics too, so it demonstrated
    /// nothing. Here a writer runs throughout and the invariant is checked on every snapshot: rows and
    /// decodes must stay in step, because `record_selective_decode` moves all three together.
    #[test]
    fn a_snapshot_never_catches_a_record_half_applied() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::Ordering;

        const KEPT: u64 = 7;
        const AVAILABLE: u64 = 11;
        /// How many concurrent snapshots the reader insists on taking.
        const READS: u64 = 20_000;

        let metrics = Arc::new(ScanMetrics::default());
        // **The reader decides when the writer stops.** A fixed write count with the reader looping
        // until the writer finishes cannot guarantee the two ever overlap: on a loaded machine the
        // writer completes before the reader is rescheduled, and the test then either exercises nothing
        // or fails its own coverage assertion. Inverting it makes the overlap deterministic.
        let reader_done = Arc::new(AtomicBool::new(false));
        let start = Arc::new(std::sync::Barrier::new(2));

        let writer = {
            let metrics = Arc::clone(&metrics);
            let reader_done = Arc::clone(&reader_done);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                let mut writes = 0u64;
                while !reader_done.load(Ordering::Acquire) {
                    metrics.record_selective_decode(KEPT, AVAILABLE);
                    writes += 1;
                }
                writes
            })
        };
        start.wait();

        for _ in 0..READS {
            let c = metrics.snapshot();
            assert_eq!(
                c.selective_rows,
                c.selective_decodes * KEPT,
                "a snapshot caught rows and decodes out of step"
            );
            assert_eq!(
                c.selective_rows_available,
                c.selective_decodes * AVAILABLE,
                "a snapshot caught available rows and decodes out of step"
            );
        }
        reader_done.store(true, Ordering::Release);
        let writes = writer.join().expect("the writing thread panicked");
        assert!(writes > 0, "the writer never recorded alongside the reader");

        let c = metrics.snapshot();
        assert_eq!(c.selective_decodes, writes);
        assert_eq!(c.selective_rows, writes * KEPT);
    }

    /// `reset` must clear everything or nothing, observed while writes are in flight.
    #[test]
    fn a_reset_is_not_observable_half_done() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::sync::atomic::Ordering;

        const KEPT: u64 = 3;
        const AVAILABLE: u64 = 5;
        /// How many concurrent resets the resetting thread insists on performing.
        const RESETS: u64 = 20_000;

        let metrics = Arc::new(ScanMetrics::default());
        let reader_done = Arc::new(AtomicBool::new(false));
        let start = Arc::new(std::sync::Barrier::new(2));

        let writer = {
            let metrics = Arc::clone(&metrics);
            let reader_done = Arc::clone(&reader_done);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                let mut writes = 0u64;
                while !reader_done.load(Ordering::Acquire) {
                    metrics.record_selective_decode(KEPT, AVAILABLE);
                    writes += 1;
                }
                writes
            })
        };
        start.wait();

        for _ in 0..RESETS {
            metrics.reset();
            let c = metrics.snapshot();
            assert_eq!(
                c.selective_rows,
                c.selective_decodes * KEPT,
                "a reset was observable half-applied"
            );
            assert_eq!(
                c.selective_rows_available,
                c.selective_decodes * AVAILABLE,
                "a reset was observable half-applied"
            );
        }
        reader_done.store(true, Ordering::Release);
        let writes = writer.join().expect("the writing thread panicked");
        assert!(
            writes > 0,
            "the writer never recorded alongside the resetting thread"
        );

        // A quiescent reset still clears every field, including ones the writer never touched.
        metrics.record_page_decode();
        metrics.record_bloom_filter_decode();
        metrics.reset();
        assert_eq!(metrics.snapshot(), ScanCounts::default());
    }
}
