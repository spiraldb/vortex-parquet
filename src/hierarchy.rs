// SPDX-License-Identifier: Apache-2.0

//! Layout nodes preserving Parquet's hierarchy without performing data I/O during open.
//!
//! Page boundaries are not part of the footer unless an offset index is referenced. Consequently a
//! column chunk has one of two page-set children: indexed pages, whose index bytes are fetched lazily,
//! or header-discovered pages, whose chained headers are walked only after the chunk is selected.
//! Actual data pages become encoding-specific arrays from [`crate::pages`] during execution.

use std::ops::Range;
use std::sync::Arc;

use parquet::basic::Compression;
use parquet::basic::Type as PhysicalType;
use vortex::array::EmptyMetadata;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::layout::Layout;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutDeserializeArgs;
use vortex::layout::LayoutId;
use vortex::layout::LayoutParts;
use vortex::layout::LayoutReaderContext;
use vortex::layout::LayoutReaderRef;
use vortex::layout::LayoutRef;
use vortex::layout::VTable;
use vortex::layout::layout_children;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;

use crate::layout::PageExecutionData;
use crate::reader::ParquetPageSetReader;

/// Parquet-file layout vtable.
#[derive(Clone, Debug)]
pub struct ParquetFile;

/// A Parquet file layout with one transparent chunked child.
pub type ParquetFileLayout = Layout<ParquetFile>;

/// Parquet-row-group layout vtable.
#[derive(Clone, Debug)]
pub struct ParquetRowGroup;

/// Per-row-group metadata.
#[derive(Clone, Debug)]
pub struct RowGroupData {
    /// Row-group ordinal in the file.
    pub index: usize,
}

/// A row-group layout with one transparent struct child.
pub type ParquetRowGroupLayout = Layout<ParquetRowGroup>;

/// Physical-column-chunk layout vtable.
#[derive(Clone, Debug)]
pub struct ParquetColumnChunk;

/// Per-column-chunk physical metadata sourced entirely from the footer.
#[derive(Clone, Debug)]
pub struct ColumnChunkData {
    /// Physical leaf ordinal in the Parquet schema.
    pub leaf: usize,
    /// Physical type.
    pub physical_type: PhysicalType,
    /// Compression codec applied outside the value encoding.
    pub compression: Compression,
    /// Exact byte range covering the dictionary and data pages.
    pub byte_range: Range<u64>,
    /// Offset-index range referenced by the footer, if present.
    pub offset_index: Option<Range<u64>>,
    /// Column-index range referenced by the footer, if present.
    pub column_index: Option<Range<u64>>,
    /// Split-block Bloom-filter range referenced by the footer, if present.
    pub bloom_filter: Option<Range<u64>>,
}

/// A physical column chunk with one lazily discovered page-set child.
pub type ParquetColumnChunkLayout = Layout<ParquetColumnChunk>;

/// Pages whose boundaries will be read from an offset index when needed.
#[derive(Clone, Debug)]
pub struct ParquetIndexedPages;

/// Indexed-page-set layout.
pub type ParquetIndexedPagesLayout = Layout<ParquetIndexedPages>;

/// Pages whose boundaries will be discovered from chained page headers when needed.
#[derive(Clone, Debug)]
pub struct ParquetHeaderPages;

/// Header-discovered-page-set layout.
pub type ParquetHeaderPagesLayout = Layout<ParquetHeaderPages>;

/// Footer-known metadata shared by both lazy page-set implementations.
#[derive(Clone, Debug)]
pub struct PageSetData {
    /// Exact column-chunk range containing the pages.
    pub chunk_range: Range<u64>,
    /// Offset-index range, present only for [`ParquetIndexedPages`].
    pub offset_index: Option<Range<u64>>,
    /// Column-index range when the writer supplied one.
    pub column_index: Option<Range<u64>>,
    /// Bloom-filter range when the writer supplied one.
    pub bloom_filter: Option<Range<u64>>,
    /// State required to turn this lazy page set into an executable reader.
    pub execution: PageExecutionData,
}

impl VTable for ParquetFile {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.parquet.file");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        _args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_bail!("Parquet file layouts are constructed in memory")
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        if idx == 0 {
            Ok(layout.dtype().clone())
        } else {
            vortex_bail!("file child {idx} out of bounds")
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        if idx == 0 {
            LayoutChildType::Transparent("row-groups".into())
        } else {
            vortex_panic!("file child {idx} out of bounds")
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout
            .slot(0)?
            .ok_or_else(|| vortex_err!("missing required layout child"))?
            .new_reader(format!("{name}.row-groups").into(), source, session, ctx)
    }
}

impl VTable for ParquetRowGroup {
    type LayoutData = RowGroupData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.parquet.row-group");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        _args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_bail!("Parquet row-group layouts are constructed in memory")
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        if idx == 0 {
            Ok(layout.dtype().clone())
        } else {
            vortex_bail!("row-group child {idx} out of bounds")
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        if idx == 0 {
            LayoutChildType::Transparent("columns".into())
        } else {
            vortex_panic!("row-group child {idx} out of bounds")
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout
            .slot(0)?
            .ok_or_else(|| vortex_err!("missing required layout child"))?
            .new_reader(format!("{name}.columns").into(), source, session, ctx)
    }
}

impl VTable for ParquetColumnChunk {
    type LayoutData = ColumnChunkData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.parquet.column-chunk");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        _args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_bail!("Parquet column-chunk layouts are constructed in memory")
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        if idx == 0 {
            Ok(layout.dtype().clone())
        } else {
            vortex_bail!("page-set child {idx} out of bounds")
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        if idx == 0 {
            LayoutChildType::Auxiliary("pages".into())
        } else {
            vortex_panic!("page-set child {idx} out of bounds")
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout
            .slot(0)?
            .ok_or_else(|| vortex_err!("missing required layout child"))?
            .new_reader(name, source, session, _ctx)
    }
}

macro_rules! page_set_vtable {
    ($vtable:ty, $id:literal, $description:literal) => {
        impl VTable for $vtable {
            type LayoutData = PageSetData;
            type Metadata = EmptyMetadata;

            fn id(&self) -> LayoutId {
                static ID: CachedId = CachedId::new($id);
                *ID
            }

            fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
                EmptyMetadata
            }

            fn deserialize(
                &self,
                _args: &LayoutDeserializeArgs<'_>,
                _metadata: &EmptyMetadata,
            ) -> VortexResult<Self::LayoutData> {
                vortex_bail!($description)
            }

            fn child_dtype(_layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
                vortex_bail!("lazy page set has no materialized child {idx}")
            }

            fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
                vortex_panic!("lazy page set has no materialized child {idx}")
            }

            fn new_reader(
                layout: &Layout<Self>,
                name: Arc<str>,
                source: Arc<dyn SegmentSource>,
                session: &VortexSession,
                _ctx: &LayoutReaderContext,
            ) -> VortexResult<LayoutReaderRef> {
                Ok(Arc::new(ParquetPageSetReader::new(
                    name,
                    layout.dtype().clone(),
                    layout.row_count(),
                    layout.segment_ids()[0],
                    layout.data().execution.clone(),
                    source,
                    session.clone(),
                )?))
            }
        }
    };
}

page_set_vtable!(
    ParquetIndexedPages,
    "vortex.parquet.pages.indexed",
    "indexed page layouts are constructed in memory"
);
page_set_vtable!(
    ParquetHeaderPages,
    "vortex.parquet.pages.headers",
    "header page layouts are constructed in memory"
);

/// Constructs the root Parquet-file wrapper.
pub fn file_layout(dtype: DType, row_count: u64, child: LayoutRef) -> ParquetFileLayout {
    LayoutParts::new(
        ParquetFile,
        dtype,
        row_count,
        Vec::new(),
        layout_children(vec![child]),
        (),
    )
    .into_typed()
}

/// Constructs one row-group wrapper.
pub fn row_group_layout(
    dtype: DType,
    row_count: u64,
    index: usize,
    child: LayoutRef,
) -> ParquetRowGroupLayout {
    LayoutParts::new(
        ParquetRowGroup,
        dtype,
        row_count,
        Vec::new(),
        layout_children(vec![child]),
        RowGroupData { index },
    )
    .into_typed()
}

/// Constructs a page set whose boundaries come from a lazily fetched offset index.
pub fn indexed_pages_layout(
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    data: PageSetData,
) -> ParquetIndexedPagesLayout {
    LayoutParts::new(
        ParquetIndexedPages,
        dtype,
        row_count,
        segment_ids,
        layout_children(Vec::new()),
        data,
    )
    .into_typed()
}

/// Constructs a page set whose boundaries are lazily discovered from page headers.
pub fn header_pages_layout(
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    data: PageSetData,
) -> ParquetHeaderPagesLayout {
    LayoutParts::new(
        ParquetHeaderPages,
        dtype,
        row_count,
        segment_ids,
        layout_children(Vec::new()),
        data,
    )
    .into_typed()
}

/// Constructs one physical column-chunk layout.
pub fn column_chunk_layout(
    dtype: DType,
    row_count: u64,
    segment_id: SegmentId,
    data: ColumnChunkData,
    pages: LayoutRef,
) -> ParquetColumnChunkLayout {
    LayoutParts::new(
        ParquetColumnChunk,
        dtype,
        row_count,
        vec![segment_id],
        layout_children(vec![pages]),
        data,
    )
    .into_typed()
}
