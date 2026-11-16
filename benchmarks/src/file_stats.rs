// SPDX-License-Identifier: Apache-2.0

//! Expose a Vortex file's footer statistics and bounded row splits to the benchmark query planner.
//!
//! The wrapper takes field statistics from the opened file and delegates scan execution to a
//! `LayoutReaderDataSource` configured with a maximum split size. It uses metadata already held
//! by the opened file and requires no additional footer reads.

use std::sync::Arc;

use async_trait::async_trait;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision;
use vortex::file::VortexFile;
use vortex::layout::scan::layout::LayoutReaderDataSource;
use vortex::scan::DataSource;
use vortex::scan::DataSourceRef;
use vortex::scan::DataSourceScanRef;
use vortex::scan::PartitionRef;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;

/// Maximum rows per Vortex scan split, using Vortex's ideal split-size heuristic.
const SPLIT_MAX_ROWS: u64 = 100_000;

/// A [`DataSource`] that answers `field_statistics` from a Vortex file's footer.
///
/// Scan execution delegates to a layout reader source with bounded row splits.
pub struct FooterStatistics {
    inner: DataSourceRef,
    statistics: Option<vortex::file::FileStatistics>,
}

impl FooterStatistics {
    /// Wraps `file`'s data source, carrying its footer statistics.
    ///
    /// A file with no statistics in its footer is wrapped anyway and simply reports none — the same
    /// answer as the unwrapped source, so the arm stays runnable rather than failing on an older
    /// file.
    pub fn wrap(file: &VortexFile) -> VortexResult<DataSourceRef> {
        // The file helper leaves the split size at `u64::MAX`. Configure bounded work for the
        // output streams through the public layout source instead.
        let inner = LayoutReaderDataSource::new(file.layout_reader()?, file.session().clone())
            .with_split_max_row_count(SPLIT_MAX_ROWS);
        Ok(Arc::new(Self {
            inner: Arc::new(inner),
            statistics: file.file_stats().cloned(),
        }))
    }
}

#[async_trait]
impl DataSource for FooterStatistics {
    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.inner.row_count()
    }

    fn byte_size(&self) -> Precision<u64> {
        self.inner.byte_size()
    }

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.serialize()
    }

    fn deserialize_partition(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        self.inner.deserialize_partition(data, session)
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        self.inner.scan(scan_request).await
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(self
            .statistics
            .as_ref()
            .and_then(|statistics| statistics.get_by_path(field_path))
            .map(|(statistics, _)| statistics.to_stats_set())
            .unwrap_or_default())
    }
}
