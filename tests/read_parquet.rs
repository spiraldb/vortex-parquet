// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests over real Parquet files.

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BinaryArray;
use arrow_array::Decimal64Array;
use arrow_array::Decimal128Array;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::Int8Array;
use arrow_array::Int16Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::TimestampMillisecondArray;
use arrow_array::UInt8Array;
use arrow_array::UInt16Array;
use arrow_array::UInt32Array;
use arrow_array::builder::Int32Builder;
use arrow_array::builder::ListBuilder;
use arrow_array_vortex::BooleanArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::SchemaRef;
use arrow_schema_vortex::Field as VortexArrowField;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::data_type::Int64Type;
use parquet::file::properties::EnabledStatistics;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterVersion;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use parquet::thrift::TSerializable;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::MaskFuture;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::Decimal;
use vortex::array::arrays::Dict;
use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::vtable::VTable;
use vortex::arrow::ArrowSessionExt;
use vortex::arrow::ExecuteArrowExt;
#[allow(dead_code)] // Reuse the production Arrow bridge; this test needs only array import.
#[path = "../src/arrow_bridge.rs"]
mod arrow_bridge;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::FieldMask;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::byte_length;
use vortex::expr::dynamic;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::gt_eq;
use vortex::expr::like;
use vortex::expr::list_contains_opts;
use vortex::expr::lit;
use vortex::expr::not_eq;
use vortex::expr::or;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::io::VortexReadAt;
use vortex::io::runtime::single::block_on;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::mask::Mask;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::fns::cast::Cast;
use vortex::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex::scalar_fn::fns::list_contains::ListContainsOptions;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_parquet::OpenedParquetFile;
use vortex_parquet::ParquetFormat;
use vortex_parquet::cache::CachePolicy;
use vortex_parquet::file::FileHandle;
use vortex_parquet::hierarchy::PageSetData;
use vortex_parquet::hierarchy::ParquetColumnChunk;
use vortex_parquet::hierarchy::ParquetFile;
use vortex_parquet::hierarchy::ParquetHeaderPages;
use vortex_parquet::hierarchy::ParquetIndexedPages;
use vortex_parquet::hierarchy::ParquetRowGroup;
use vortex_parquet::hierarchy::header_pages_layout;
use vortex_parquet::layout::FileContext;
use vortex_parquet::layout::PageExecutionData;
use vortex_parquet::layout::ParquetField;
use vortex_parquet::options::ReaderOptions;
use vortex_parquet::pages::ParquetBitPacked;
use vortex_parquet::pages::ParquetByteStreamSplit;
use vortex_parquet::pages::ParquetDeltaBinaryPacked;
use vortex_parquet::pages::ParquetDeltaByteArray;
use vortex_parquet::pages::ParquetDeltaLengthByteArray;
use vortex_parquet::pages::ParquetPlain;
use vortex_parquet::pages::ParquetPlainDictionary;
use vortex_parquet::pages::ParquetRle;
use vortex_parquet::pages::ParquetRleDictionary;
use vortex_parquet::segments::SegmentTable;

fn flat_table() -> RecordBatch {
    flat_table_n(40)
}

fn flat_table_n(rows: i64) -> RecordBatch {
    let ids = Int64Array::from_iter_values(0..rows);
    let scores = Float64Array::from_iter((0..rows).map(|i| {
        if i % 5 == 0 {
            None
        } else {
            Some(i as f64 * 1.5)
        }
    }));
    let names = StringArray::from_iter((0..rows).map(|i| {
        if i % 7 == 3 {
            None
        } else {
            Some(format!("label-{:02}", i % 6))
        }
    }));
    RecordBatch::try_from_iter_with_nullable([
        ("id", Arc::new(ids) as ArrowArrayRef, false),
        ("score", Arc::new(scores) as ArrowArrayRef, true),
        ("name", Arc::new(names) as ArrowArrayRef, true),
    ])
    .unwrap()
}

fn nested_struct_table() -> RecordBatch {
    let inner_fields = Fields::from(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, true),
    ]);
    let inner = ArrowStructArray::new(
        inner_fields,
        vec![
            Arc::new(Int32Array::from_iter_values(0..24)) as ArrowArrayRef,
            Arc::new(StringArray::from_iter(
                (0..24).map(|i| (i % 4 != 0).then(|| format!("inner-{i}"))),
            )) as ArrowArrayRef,
        ],
        None,
    );
    RecordBatch::try_from_iter_with_nullable([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..24)) as ArrowArrayRef,
            false,
        ),
        ("nested", Arc::new(inner) as ArrowArrayRef, false),
    ])
    .unwrap()
}

fn list_table() -> RecordBatch {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for i in 0..20 {
        for j in 0..(i % 4) {
            builder.values().append_value(i * 10 + j);
        }
        builder.append(true);
    }
    let tags: ListArray = builder.finish();
    RecordBatch::try_from_iter_with_nullable([
        (
            "id",
            Arc::new(Int64Array::from_iter_values(0..20)) as ArrowArrayRef,
            false,
        ),
        ("tags", Arc::new(tags) as ArrowArrayRef, true),
    ])
    .unwrap()
}

fn logical_integer_and_temporal_table() -> RecordBatch {
    RecordBatch::try_from_iter_with_nullable([
        (
            "small",
            Arc::new(Int16Array::from_iter_values(
                (0..200).map(|i| [i16::MIN, 0, i16::MAX][i % 3]),
            )) as ArrowArrayRef,
            false,
        ),
        (
            "i8",
            Arc::new(Int8Array::from_iter_values(
                (0..200).map(|i| [i8::MIN, 0, i8::MAX][i % 3]),
            )) as ArrowArrayRef,
            false,
        ),
        (
            "u8",
            Arc::new(UInt8Array::from_iter_values(
                (0..200).map(|i| [0, 1, u8::MAX][i % 3]),
            )) as ArrowArrayRef,
            false,
        ),
        (
            "u16",
            Arc::new(UInt16Array::from_iter_values(
                (0..200).map(|i| [0, 1, u16::MAX][i % 3]),
            )) as ArrowArrayRef,
            false,
        ),
        (
            "unsigned",
            Arc::new(UInt32Array::from_iter_values(
                (0..200).map(|value| u32::MAX - value),
            )) as ArrowArrayRef,
            false,
        ),
        (
            "time",
            Arc::new(TimestampMillisecondArray::from_iter_values(0..200)) as ArrowArrayRef,
            false,
        ),
    ])
    .unwrap()
}

fn write_parquet(batch: &RecordBatch, rows_per_group: usize) -> ByteBuffer {
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(rows_per_group))
        .set_data_page_row_count_limit(5)
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true)
        .build();
    write_parquet_with_properties(batch, properties)
}

fn write_parquet_with_properties(batch: &RecordBatch, properties: WriterProperties) -> ByteBuffer {
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(properties)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    ByteBuffer::copy_from(bytes)
}

/// Writes two data pages with INT64 decimal storage independently of the declared precision.
fn small_precision_decimal_file(values: &[i64], levels: &[i16], dictionary: bool) -> ByteBuffer {
    integer_decimal_file(2, values, levels, dictionary)
}

fn integer_decimal_file(
    precision: u8,
    values: &[i64],
    levels: &[i16],
    dictionary: bool,
) -> ByteBuffer {
    let schema = Arc::new(
        parse_message_type(&format!(
            "message test {{ REQUIRED INT64 id; OPTIONAL INT64 amount (DECIMAL({precision},0)); }}"
        ))
        .unwrap(),
    );
    let properties = Arc::new(
        WriterProperties::builder()
            .set_data_page_row_count_limit(2)
            .set_write_batch_size(2)
            .set_dictionary_enabled(dictionary)
            .set_statistics_enabled(EnabledStatistics::None)
            .build(),
    );
    let mut bytes = Vec::new();
    let mut writer = SerializedFileWriter::new(&mut bytes, schema, properties).unwrap();
    let mut group = writer.next_row_group().unwrap();
    let mut ids = group.next_column().unwrap().unwrap();
    ids.typed::<Int64Type>()
        .write_batch(&[0, 1, 2, 3], None, None)
        .unwrap();
    ids.close().unwrap();
    let mut amounts = group.next_column().unwrap().unwrap();
    amounts
        .typed::<Int64Type>()
        .write_batch(values, Some(levels), None)
        .unwrap();
    amounts.close().unwrap();
    group.close().unwrap();
    writer.close().unwrap();
    ByteBuffer::copy_from(bytes)
}

async fn open_materialized(
    session: &VortexSession,
    bytes: ByteBuffer,
) -> VortexResult<LayoutReaderRef> {
    let size = bytes.len() as u64;
    ParquetFormat::new()
        .with_options(ReaderOptions::default().with_materialized_projections())
        .open(FileHandle::new(Arc::new(bytes), size, session.clone()))
        .await
}

async fn read_materialized_amounts(
    session: &VortexSession,
    bytes: ByteBuffer,
    filter: Option<Expression>,
) -> VortexResult<ArrayRef> {
    let reader = open_materialized(session, bytes).await?;
    scan(session, reader, filter)
        .await?
        .apply(&get_item("amount", root()))
}

fn assert_amounts(actual: &ArrayRef, expected: &[Option<i64>], session: &VortexSession) {
    assert_amounts_with_precision(actual, expected, 2, session);
}

fn assert_amounts_with_precision(
    actual: &ArrayRef,
    expected: &[Option<i64>],
    precision: u8,
    session: &VortexSession,
) {
    let expected = Decimal64Array::from(expected.to_vec())
        .with_precision_and_scale(precision, 0)
        .unwrap();
    let expected = session
        .arrow()
        .from_arrow_array(arrow_bridge::import_array(&expected).unwrap(), true)
        .unwrap();
    assert_eq!(render(actual, session), render(&expected, session));
}

fn file_context(opened: &OpenedParquetFile) -> Arc<FileContext> {
    let groups = opened.layout.slot(0).unwrap().unwrap();
    let group = groups.slot(0).unwrap().unwrap();
    let fields = group.slot(0).unwrap().unwrap();
    let field = fields.slot(1).unwrap().unwrap();
    Arc::clone(&field.as_::<ParquetField>().data().ctx)
}

fn physical_reader(
    opened: &OpenedParquetFile,
    column: usize,
    session: &VortexSession,
) -> (LayoutReaderRef, SchemaRef) {
    let context = file_context(opened);
    let group = context.metadata.row_group(0);
    let row_count = u64::try_from(group.num_rows()).unwrap();
    let (byte_offset, byte_length) = group.column(column).byte_range();
    let dtype = opened
        .layout
        .dtype()
        .as_struct_fields()
        .field_by_index(column)
        .unwrap();
    let mut segments = SegmentTable::new();
    let id = segments
        .push(byte_offset..byte_offset + byte_length)
        .unwrap();
    let source = context.file.segment_source(segments).unwrap();
    let data = PageExecutionData {
        ctx: context,
        row_group: 0,
        root_column: column,
        byte_offset,
        byte_length,
        offset_indexes: vec![],
        column_indexes: vec![],
        bloom_filters: vec![],
    };
    let decoder_schema = Arc::clone(data.decode_inputs().arrow_metadata.schema());
    let reader = header_pages_layout(
        dtype,
        row_count,
        vec![id],
        PageSetData {
            chunk_range: byte_offset..byte_offset + byte_length,
            offset_index: None,
            column_index: None,
            bloom_filter: None,
            execution: data,
        },
    )
    .new_reader("physical".into(), source, session, &Default::default())
    .unwrap();
    (reader, decoder_schema)
}

#[derive(Clone)]
struct RecordingReadAt {
    data: ByteBuffer,
    reads: Arc<Mutex<Vec<std::ops::Range<u64>>>>,
}

impl VortexReadAt for RecordingReadAt {
    fn concurrency(&self) -> usize {
        16
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.data.len() as u64;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.reads.lock().push(offset..offset + length as u64);
        self.data.read_at(offset, length, alignment)
    }
}

fn reference(batch: &RecordBatch) -> ArrayRef {
    VortexSession::default()
        .arrow()
        .from_arrow_array(
            arrow_bridge::import_array(&arrow_array::StructArray::from(batch.clone())).unwrap(),
            false,
        )
        .unwrap()
}

fn render(array: &ArrayRef, session: &VortexSession) -> String {
    let mut ctx = session.create_execution_ctx();
    let arrow = array.clone().execute_arrow(None, &mut ctx).unwrap();
    format!("{arrow:?}")
        .replace("Utf8View", "Utf8")
        .replace("StringViewArray", "StringArray")
}

async fn open(session: &VortexSession, parquet: ByteBuffer) -> VortexResult<LayoutReaderRef> {
    let size = parquet.len() as u64;
    ParquetFormat::new()
        .open(FileHandle::new(Arc::new(parquet), size, session.clone()))
        .await
}

async fn scan(
    session: &VortexSession,
    reader: LayoutReaderRef,
    filter: Option<Expression>,
) -> VortexResult<ArrayRef> {
    {
        let filter = filter.map(|expr| expr.bind(reader.dtype())).transpose()?;
        ScanBuilder::new(session.clone(), reader)
            .with_some_filter(filter)
            .with_ordered(true)
            .into_array_stream()?
            .read_all()
            .await
    }
}

#[test]
fn flat_pages_decode_to_the_original_values() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = flat_table();
        for (dictionary, version) in [true, false].into_iter().flat_map(|dictionary| {
            [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0]
                .map(|version| (dictionary, version))
        }) {
            let bytes = write_parquet_with_properties(
                &batch,
                WriterProperties::builder()
                    .set_writer_version(version)
                    .set_max_row_group_row_count(Some(16))
                    .set_dictionary_enabled(dictionary)
                    .set_data_page_row_count_limit(5)
                    .set_write_batch_size(5)
                    .set_compression(Compression::SNAPPY)
                    .build(),
            );
            let reader = open(&session, bytes).await.unwrap();
            let result = scan(&session, reader, None).await.unwrap();
            let encodings: Vec<String> = result
                .depth_first_traversal()
                .map(|array| array.encoding_id().to_string())
                .collect();
            assert!(
                encodings
                    .iter()
                    .any(|id| id.starts_with("vortex.parquet.") || id == "vortex.dict"),
                "scan should retain Parquet pages or their native dictionary realization, got {encodings:?}"
            );
            assert_eq!(result.dtype(), reference(&batch).dtype());
            let expected =
                arrow_bridge::import_array(&ArrowStructArray::from(batch.clone())).unwrap();
            let target = VortexArrowField::new("row", expected.data_type().clone(), false);
            let actual = result
                .execute_arrow(Some(&target), &mut session.create_execution_ctx())
                .unwrap();
            assert_eq!(
                actual.to_data(),
                expected.to_data(),
                "dictionary {dictionary}, {version:?}"
            );
        }
    })
}

#[test]
fn physical_integers_decode_directly_to_narrow_unsigned_and_temporal_types() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = logical_integer_and_temporal_table();
        for nullable in [false, true] {
            let schema = Arc::new(arrow_schema::Schema::new(
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone().with_nullable(nullable))
                    .collect::<Vec<_>>(),
            ));
            let batch = RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap();
            for dictionary in [false, true] {
                for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
                    let format = ParquetFormat::new();
                    let bytes = write_parquet_with_properties(
                        &batch,
                        WriterProperties::builder()
                            .set_dictionary_enabled(dictionary)
                            .set_writer_version(version)
                            .set_compression(Compression::SNAPPY)
                            .build(),
                    );
                    let result = scan(
                        &session,
                        format
                            .open(FileHandle::new(
                                Arc::new(bytes.clone()),
                                bytes.len() as u64,
                                session.clone(),
                            ))
                            .await
                            .unwrap(),
                        None,
                    )
                    .await
                    .unwrap();
                    assert_eq!(
                        render(&result, &session),
                        render(&reference(&batch), &session)
                    );
                    assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
                    assert_eq!(
                        format.metrics().snapshot().page_decodes,
                        batch.num_columns() as u64
                    );
                }
            }
        }
    })
}

#[test]
fn int64_decimals_decode_filter_and_prune_in_the_unscaled_domain() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let decimal = DecimalDType::new(15, 2);
        let values = Decimal64Array::from_iter_values(0..400)
            .with_precision_and_scale(15, 2)
            .unwrap();
        let batch =
            RecordBatch::try_from_iter([("amount", Arc::new(values) as ArrowArrayRef)]).unwrap();
        let bytes = write_parquet(&batch, 400);
        let format = ParquetFormat::new();
        let reader = format
            .open(FileHandle::new(
                Arc::new(bytes.clone()),
                bytes.len() as u64,
                session.clone(),
            ))
            .await
            .unwrap();
        let threshold = Scalar::decimal(DecimalValue::I64(200), decimal, Nullability::NonNullable);
        let result = scan(
            &session,
            reader,
            Some(gt_eq(get_item("amount", root()), lit(threshold))),
        )
        .await
        .unwrap();
        assert_eq!(result.len(), 200);
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
        assert!(format.metrics().snapshot().page_decodes > 0);
    })
}

#[test]
fn every_parquet_value_encoding_has_its_own_vortex_array_id() {
    let ids = [
        ParquetPlain.id(),
        ParquetPlainDictionary.id(),
        ParquetRle.id(),
        ParquetBitPacked.id(),
        ParquetDeltaBinaryPacked.id(),
        ParquetDeltaLengthByteArray.id(),
        ParquetDeltaByteArray.id(),
        ParquetRleDictionary.id(),
        ParquetByteStreamSplit.id(),
    ];
    for (index, id) in ids.iter().enumerate() {
        assert!(!ids[..index].contains(id), "duplicate array id {id}");
    }
}

#[test]
fn nested_structs_reconstruct_at_the_field_boundary() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = nested_struct_table();
        let result = scan(
            &session,
            open(&session, write_parquet(&batch, 8)).await.unwrap(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            render(&result, &session),
            render(&reference(&batch), &session)
        );
    })
}

#[test]
fn lists_reconstruct_at_the_field_boundary() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = list_table();
        let result = scan(
            &session,
            open(&session, write_parquet(&batch, 7)).await.unwrap(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            render(&result, &session),
            render(&reference(&batch), &session)
        );
    })
}

#[test]
fn the_layout_models_file_row_group_column_chunk_and_lazy_pages() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 16);
        let size = bytes.len() as u64;
        let opened = ParquetFormat::new()
            .open_file(FileHandle::new(Arc::new(bytes), size, session.clone()))
            .await
            .unwrap();

        assert!(opened.layout.is::<ParquetFile>());
        let row_groups = opened.layout.slot(0).unwrap().unwrap();
        assert_eq!(row_groups.nchildren(), 3);
        let row_group = row_groups.slot(0).unwrap().unwrap();
        assert!(row_group.is::<ParquetRowGroup>());
        let fields = row_group.slot(0).unwrap().unwrap();
        // Struct slot zero is reserved for validity; field two is slot three.
        let field = fields.slot(3).unwrap().unwrap();
        assert!(field.is::<ParquetField>());
        assert!(matches!(field.dtype(), DType::Utf8(_)));
        let chunk = field.slot(0).unwrap().unwrap();
        assert!(chunk.is::<ParquetColumnChunk>());
        assert!(matches!(chunk.dtype(), DType::Utf8(_)));
        let chunk_data = &chunk.as_::<ParquetColumnChunk>().data();
        assert!(chunk_data.offset_index.is_some());
        assert_eq!(chunk.nchildren(), 1);
        let pages = chunk.slot(0).unwrap().unwrap();
        assert!(pages.is::<ParquetIndexedPages>());
        let page_data = pages.as_::<ParquetIndexedPages>().data();
        assert_eq!(page_data.chunk_range, chunk_data.byte_range);
        assert_eq!(page_data.offset_index, chunk_data.offset_index);
        assert!(pages.segment_ids().len() >= 2);
    })
}

#[test]
fn open_reads_only_the_footer() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 16);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let read = RecordingReadAt {
            data: bytes,
            reads: Arc::clone(&reads),
        };

        ParquetFormat::new()
            .open_file(FileHandle::new(Arc::new(read), size, session))
            .await
            .unwrap();

        let reads = reads.lock();
        assert_eq!(reads.len(), 2, "open performed non-footer I/O: {reads:?}");
        assert_eq!(reads[0], size - 8..size);
        assert_eq!(reads[1].end, size - 8);
    })
}

#[test]
fn footer_statistics_prune_a_row_group_without_reading_payloads() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 40);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let reader = ParquetFormat::new()
            .open(FileHandle::new(
                Arc::new(RecordingReadAt {
                    data: bytes,
                    reads: Arc::clone(&reads),
                }),
                size,
                session.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(reads.lock().len(), 2);

        let result = scan(
            &session,
            reader,
            Some(eq(get_item("id", root()), lit(-1i64))),
        )
        .await
        .unwrap();

        assert_eq!(result.len(), 0);
        assert_eq!(
            reads.lock().len(),
            2,
            "an impossible footer predicate read non-footer bytes: {:?}",
            reads.lock()
        );
    })
}

#[test]
fn data_source_reports_footer_statistics_to_query_planners() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 16);
        let size = bytes.len() as u64;
        let source = ParquetFormat::new()
            .open_data_source(FileHandle::new(Arc::new(bytes), size, session.clone()))
            .await
            .unwrap();

        assert_eq!(source.byte_size(), Precision::Absent);
        assert_eq!(source.row_count().as_exact(), Some(40));
        let stats = source
            .field_statistics(&FieldPath::from_name("id"))
            .await
            .unwrap();
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        assert_eq!(stats.get_as::<i64>(Stat::Min, &dtype).as_exact(), Some(0));
        assert_eq!(stats.get_as::<i64>(Stat::Max, &dtype).as_exact(), Some(39));
        assert_eq!(
            stats
                .get_as::<u64>(
                    Stat::UncompressedSizeInBytes,
                    &DType::Primitive(PType::U64, Nullability::NonNullable),
                )
                .as_exact(),
            Some(40 * 8)
        );
        assert_eq!(
            stats
                .get_as::<u64>(
                    Stat::NullCount,
                    &DType::Primitive(PType::U64, Nullability::NonNullable),
                )
                .as_exact(),
            Some(0)
        );
        let name_stats = source
            .field_statistics(&FieldPath::from_name("name"))
            .await
            .unwrap();
        assert!(matches!(
            name_stats.get_as::<u64>(
                Stat::UncompressedSizeInBytes,
                &DType::Primitive(PType::U64, Nullability::NonNullable),
            ),
            Precision::Inexact(_)
        ));
    })
}

#[test]
fn data_source_executes_exactly_one_task_per_row_group() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 16);
        let size = bytes.len() as u64;
        let source = ParquetFormat::new()
            .open_data_source(FileHandle::new(Arc::new(bytes), size, session.clone()))
            .await
            .unwrap();
        let scan = source
            .scan(ScanRequest {
                selection: Selection::IncludeByIndex(
                    StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter(
                        (0..40).step_by(2),
                    ))
                    .unwrap(),
                ),
                ..ScanRequest::default()
            })
            .await
            .unwrap();
        assert_eq!(scan.partition_count().as_exact(), Some(3));

        let partitions = scan.partitions().try_collect::<Vec<_>>().await.unwrap();
        let mut rows = 0usize;
        for partition in partitions {
            let chunks = partition
                .execute()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(chunks.len(), 1, "a row group was split into multiple tasks");
            rows += chunks[0].len();
        }
        assert_eq!(rows, 20);

        // Intermediate projections must retain their schema meaning when a scan builds only
        // referenced columns. Exercise both include/exclude forms with a filter-only column,
        // selected rows, a restricted row interval, and nonzero partition ordinals.
        for projection in [
            get_item("id", vortex::expr::select(["id", "name"], root())),
            get_item(
                "id",
                vortex::expr::select_exclude(["score", "name"], root()),
            ),
        ] {
            let scan = source
                .scan(ScanRequest {
                    projection,
                    filter: Some(gt_eq(get_item("score", root()), lit(20.0f64))),
                    row_range: Some(8..35),
                    partition_range: Some(1..3),
                    selection: Selection::IncludeByIndex(
                        StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter(
                            (0..40).step_by(2),
                        ))
                        .unwrap(),
                    ),
                    ordered: true,
                    ..ScanRequest::default()
                })
                .await
                .unwrap();
            assert_eq!(scan.partition_count().as_exact(), Some(2));
            let mut actual = Vec::new();
            let mut ctx = session.create_execution_ctx();
            for partition in scan.partitions().try_collect::<Vec<_>>().await.unwrap() {
                for chunk in partition
                    .execute()
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                {
                    for row in 0..chunk.len() {
                        actual.push(chunk.execute_scalar(row, &mut ctx).unwrap());
                    }
                }
            }
            let expected: Vec<Scalar> = [16i64, 18, 22, 24, 26, 28, 32, 34]
                .into_iter()
                .map(Into::into)
                .collect();
            assert_eq!(actual, expected);
        }
    })
}

#[test]
fn data_source_preserves_nulls_and_dynamic_bounds_when_eliminating_filters() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int64Array::from_iter_values(0..8)) as ArrowArrayRef,
            ),
            (
                "nullable",
                Arc::new(Int64Array::from(vec![
                    Some(0),
                    Some(1),
                    None,
                    Some(3),
                    Some(4),
                    Some(5),
                    Some(6),
                    Some(7),
                ])) as ArrowArrayRef,
            ),
            (
                "time",
                Arc::new(TimestampMillisecondArray::from_iter_values(1000..1008)) as ArrowArrayRef,
            ),
        ])
        .unwrap();
        for enabled in [EnabledStatistics::Chunk, EnabledStatistics::None] {
            let bytes = write_parquet_with_properties(
                &batch,
                WriterProperties::builder()
                    .set_max_row_group_row_count(Some(4))
                    .set_statistics_enabled(enabled)
                    .build(),
            );
            let reads = Arc::new(Mutex::new(Vec::new()));
            let source = ParquetFormat::new()
                .open_data_source(FileHandle::new(
                    Arc::new(RecordingReadAt {
                        data: bytes.clone(),
                        reads: Arc::clone(&reads),
                    }),
                    bytes.len() as u64,
                    session.clone(),
                ))
                .await
                .unwrap();
            let time_dtype = source
                .dtype()
                .as_struct_fields_opt()
                .unwrap()
                .field("time")
                .unwrap();
            let cases = [
                (gt_eq(get_item("id", root()), lit(0i64)), 8, true),
                (
                    vortex::expr::lt_eq(lit(0i64), get_item("id", root())),
                    8,
                    true,
                ),
                (
                    vortex::expr::lt_eq(get_item("id", root()), lit(7i64)),
                    8,
                    true,
                ),
                (
                    vortex::expr::lt(get_item("id", root()), lit(7i64)),
                    7,
                    false,
                ),
                (
                    vortex::expr::gt(get_item("id", root()), lit(0i64)),
                    7,
                    false,
                ),
                (eq(get_item("id", root()), lit(0i64)), 1, false),
                (not_eq(get_item("id", root()), lit(0i64)), 7, false),
                (not_eq(get_item("id", root()), lit(-1i64)), 8, true),
                (gt_eq(get_item("nullable", root()), lit(0i64)), 7, false),
                (
                    gt_eq(
                        get_item("time", root()),
                        lit(Scalar::from(1000i64).cast(&time_dtype).unwrap()),
                    ),
                    8,
                    true,
                ),
            ];
            for (filter, expected, redundant) in cases {
                reads.lock().clear();
                let scan = source
                    .scan(ScanRequest {
                        projection: vortex::expr::select(Vec::<&str>::new(), root()),
                        filter: Some(filter.clone()),
                        ..ScanRequest::default()
                    })
                    .await
                    .unwrap();
                let mut rows = 0;
                for partition in scan.partitions().try_collect::<Vec<_>>().await.unwrap() {
                    for chunk in partition
                        .execute()
                        .unwrap()
                        .try_collect::<Vec<_>>()
                        .await
                        .unwrap()
                    {
                        rows += chunk.len();
                    }
                }
                assert_eq!(rows, expected, "{filter}, statistics={enabled:?}");
                if redundant && enabled == EnabledStatistics::Chunk {
                    assert!(
                        reads.lock().is_empty(),
                        "redundant filter read data: {filter}"
                    );
                }
            }

            // Initially every row matches. A later refinement must still be observed by the scan.
            let threshold = Arc::new(AtomicI64::new(8));
            let dynamic_threshold = Arc::clone(&threshold);
            let scan = source
                .scan(ScanRequest {
                    projection: get_item("id", root()),
                    filter: Some(dynamic(
                        CompareOperator::Lt,
                        move || Some(dynamic_threshold.load(Ordering::SeqCst).into()),
                        DType::Primitive(PType::I64, Nullability::NonNullable),
                        true,
                        get_item("id", root()),
                    )),
                    ..ScanRequest::default()
                })
                .await
                .unwrap();
            threshold.store(3, Ordering::SeqCst);
            let mut rows = 0;
            for partition in scan.partitions().try_collect::<Vec<_>>().await.unwrap() {
                for chunk in partition
                    .execute()
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                {
                    rows += chunk.len();
                }
            }
            assert_eq!(rows, 3);
        }
    })
}

#[test]
fn bloom_filter_prunes_an_absent_value_inside_footer_bounds() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int64Array::from_iter_values(
                (0..200).map(|value| value * 2),
            )) as ArrowArrayRef,
        )])
        .unwrap();
        let properties = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .set_offset_index_disabled(true)
            .set_max_row_group_row_count(Some(batch.num_rows()))
            .build();
        let bytes = write_parquet_with_properties(&batch, properties);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let format = ParquetFormat::new();
        let reader = format
            .open(FileHandle::new(
                Arc::new(RecordingReadAt {
                    data: bytes,
                    reads: Arc::clone(&reads),
                }),
                size,
                session.clone(),
            ))
            .await
            .unwrap();
        let open_reads = reads.lock().len();

        let result = scan(
            &session,
            reader,
            Some(eq(get_item("id", root()), lit(21i64))),
        )
        .await
        .unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(format.metrics().bloom_filter_decodes(), 1);
        assert_eq!(format.metrics().snapshot().page_decodes, 0);
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
        assert!(
            reads.lock().len() > open_reads,
            "Bloom bytes were never read"
        );
    })
}

#[test]
fn materialized_projections_preserve_selected_nullable_rows() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = flat_table();
        let bytes = write_parquet(&batch, 40);
        let format = ParquetFormat::new()
            .with_options(ReaderOptions::default().with_materialized_projections());
        let reader = format
            .open(FileHandle::new(
                Arc::new(bytes.clone()),
                bytes.len() as u64,
                session.clone(),
            ))
            .await
            .unwrap();
        let result = scan(
            &session,
            reader,
            Some(not_eq(get_item("id", root()), lit(2_i64))),
        )
        .await
        .unwrap();
        let expected = reference(&batch)
            .filter((0..40).map(|id| id != 2).collect())
            .unwrap();
        assert_eq!(render(&result, &session), render(&expected, &session));
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
        assert!(format.metrics().snapshot().page_decodes > 0);
    })
}

#[test]
fn selected_decimal_over_precision_returns_an_error() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = small_precision_decimal_file(&[12, 34, 1_000, 56], &[1, 1, 1, 1], false);
        let error = read_materialized_amounts(&session, bytes, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("precision"), "{error}");
    })
}

#[test]
fn rejected_decimal_over_precision_is_not_evaluated() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = small_precision_decimal_file(&[12, 34, 1_000, 56], &[1, 1, 1, 1], false);
        let actual = read_materialized_amounts(
            &session,
            bytes,
            Some(not_eq(get_item("id", root()), lit(2_i64))),
        )
        .await
        .unwrap();
        assert_amounts(&actual, &[Some(12), Some(34), Some(56)], &session);
    })
}

#[test]
fn materialized_decimal_pages_preserve_nulls() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = small_precision_decimal_file(&[12, 34, 56], &[1, 1, 0, 1], false);
        let actual = read_materialized_amounts(&session, bytes, None)
            .await
            .unwrap();
        assert_amounts(&actual, &[Some(12), Some(34), None, Some(56)], &session);
    })
}

#[test]
fn dictionary_decimal_fallback_checks_selected_values() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = small_precision_decimal_file(&[12, 34, 1_000, 56], &[1, 1, 1, 1], true);
        let error = read_materialized_amounts(&session, bytes, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("precision"), "{error}");
    })
}

#[test]
fn dictionary_decimal_fallback_ignores_rejected_values_in_a_row_range() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = small_precision_decimal_file(&[12, 34, 1_000, 56], &[1, 1, 1, 1], true);
        let reader = open_materialized(&session, bytes).await.unwrap();
        let filter = eq(get_item("id", root()), lit(3_i64))
            .bind(reader.dtype())
            .unwrap();
        let result = ScanBuilder::new(session.clone(), reader)
            .with_row_range(1..4)
            .with_filter(filter)
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        let actual = result.apply(&get_item("amount", root())).unwrap();
        assert_amounts(&actual, &[Some(56)], &session);
    })
}

#[test]
fn dictionary_pages_reduce_lazily_to_native_dict_for_property_driven_pushdown() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        // Existing execution kernels must not suppress the independent dictionary reduction rules.
        session.kernels().register_execute_parent_kernel(
            Cast.id(),
            ParquetRleDictionary,
            CastExecuteAdaptor(ParquetRleDictionary),
        );
        let result = scan(
            &session,
            open(&session, write_parquet(&flat_table(), 40))
                .await
                .unwrap(),
            None,
        )
        .await
        .unwrap();
        let page = result
            .depth_first_traversal()
            .find(|array| {
                array.is::<ParquetRleDictionary>() && matches!(array.dtype(), DType::Utf8(_))
            })
            .expect("dictionary page should remain Parquet-encoded after layout read");

        let predicate = page
            .clone()
            .apply(&like(root(), lit("label-0_")))
            .unwrap()
            .optimize_recursive(&session)
            .unwrap();
        let mut ctx = session.create_execution_ctx();
        assert_eq!(
            predicate
                .fill_null(false)
                .unwrap()
                .execute::<Mask>(&mut ctx)
                .unwrap()
                .len(),
            page.len()
        );

        // A prepared set must keep dictionary pushdown and SQL/non-SQL null behavior.
        // The writer fixture has six labels, with a null at each index congruent to 3 mod 7.
        for selected in [false, true] {
            let rows = (0..page.len())
                .filter(|i| !selected || i % 3 != 1)
                .collect::<Vec<_>>();
            let needles = page
                .clone()
                .filter(Mask::from_indices(page.len(), rows.clone()))
                .unwrap();
            for sql_null_semantics in [false, true] {
                for candidates in [
                    vec![Some("label-00"), Some("label-02")],
                    vec![None, Some("label-02")],
                    vec![],
                ] {
                    let element_dtype = DType::Utf8(Nullability::Nullable);
                    let set = Scalar::list(
                        element_dtype.clone(),
                        candidates
                            .iter()
                            .map(|value| {
                                value.map_or_else(
                                    || Scalar::null(element_dtype.clone()),
                                    |value| Scalar::utf8(value, Nullability::Nullable),
                                )
                            })
                            .collect(),
                        Nullability::NonNullable,
                    );
                    let predicate = needles
                        .clone()
                        .apply_bound(
                            &list_contains_opts(
                                lit(set),
                                root(),
                                ListContainsOptions { sql_null_semantics },
                            )
                            .bind(needles.dtype())
                            .unwrap(),
                        )
                        .unwrap();
                    // Exercise direct scan execution: no recursive optimizer runs first.
                    let predicate = predicate.execute_until::<Dict>(&mut ctx).unwrap();
                    if !candidates.is_empty() {
                        assert!(
                            predicate.is::<Dict>(),
                            "prepared membership must retain dictionary values"
                        );
                    }
                    let expected = rows
                        .iter()
                        .copied()
                        .map(|i| {
                            if candidates.is_empty() && !sql_null_semantics {
                                return Some(false);
                            }
                            if i % 7 == 3 {
                                return None;
                            }
                            let value = format!("label-{:02}", i % 6);
                            if candidates.contains(&Some(value.as_str())) {
                                Some(true)
                            } else if sql_null_semantics && candidates.contains(&None) {
                                None
                            } else {
                                Some(false)
                            }
                        })
                        .collect::<Vec<_>>();
                    let actual = predicate.execute_arrow(None, &mut ctx).unwrap();
                    assert_eq!(
                        actual
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .iter()
                            .collect::<Vec<_>>(),
                        expected
                    );
                }
            }
        }

        let lengths = page
            .apply(&byte_length(root()))
            .unwrap()
            .optimize_recursive(&session)
            .unwrap();
        assert!(
            lengths.is::<Dict>(),
            "property-driven pushdown did not expose native dictionary codes: {}",
            lengths.encoding_id()
        );
    })
}

#[test]
fn mixed_dictionary_and_plain_pages_share_native_dictionary_values() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let values = StringArray::from_iter_values(
            (0..500).map(|value| format!("distinct-value-{value:04}")),
        );
        let batch =
            RecordBatch::try_from_iter([("value", Arc::new(values) as ArrowArrayRef)]).unwrap();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(500))
            .set_data_page_row_count_limit(50)
            .set_dictionary_page_size_limit(128)
            .set_dictionary_enabled(true)
            .build();
        let result = scan(
            &session,
            open(&session, write_parquet_with_properties(&batch, properties))
                .await
                .unwrap(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            render(&result, &session),
            render(&reference(&batch), &session)
        );
        let dictionary_page = result
            .depth_first_traversal()
            .find(|array| array.is::<ParquetRleDictionary>())
            .expect("mixed chunk should retain its dictionary page encoding");
        let reduced = dictionary_page
            .apply(&byte_length(root()))
            .unwrap()
            .optimize_recursive(&session)
            .unwrap();
        assert!(reduced.is::<Dict>());
        assert!(
            result
                .depth_first_traversal()
                .any(|array| array.encoding_id().as_ref().starts_with("vortex.parquet.")),
            "plain pages in a mixed chunk should remain Parquet page arrays"
        );
    })
}

#[test]
fn dictionary_values_cast_from_binary_to_utf8_only_when_executed() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let values = BinaryArray::from_iter_values(
            (0..40).map(|value| format!("label-{}", value % 4).into_bytes()),
        );
        let batch = RecordBatch::try_from_iter_with_nullable([(
            "value",
            Arc::new(values) as ArrowArrayRef,
            true,
        )])
        .unwrap();
        let result = scan(
            &session,
            open(&session, write_parquet(&batch, 40)).await.unwrap(),
            None,
        )
        .await
        .unwrap();
        let dictionary = result
            .depth_first_traversal()
            .find(|array| array.is::<ParquetRleDictionary>())
            .expect("dictionary-encoded binary page should remain Parquet-encoded");
        assert!(
            dictionary
                .encoding_id()
                .as_ref()
                .starts_with("vortex.parquet."),
            "dictionary page was eagerly materialized"
        );
        let cast = dictionary
            .cast(DType::Utf8(Nullability::Nullable))
            .unwrap()
            .optimize_recursive(&session)
            .unwrap();
        assert!(cast.encoding_id().as_ref().starts_with("vortex.parquet."));
        let mut ctx = session.create_execution_ctx();
        let matches = cast
            .clone()
            .apply(&like(root(), lit("label-1")))
            .unwrap()
            .fill_null(false)
            .unwrap()
            .execute::<Mask>(&mut ctx)
            .unwrap();
        assert_eq!(matches.true_count(), 10);
        let decoded = cast.execute::<Canonical>(&mut ctx).unwrap();
        assert!(matches!(decoded.dtype(), DType::Utf8(_)));
    })
}

#[test]
fn integer_predicates_preserve_boundaries_and_nulls() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        macro_rules! check {
            ($native:ty, $arrow:ty) => {{
                // Cover byte boundaries, full SIMD blocks, and both short and long tails.
                for (null_mode, row_count) in (0..4).flat_map(|mode| {
                    [7, 8, 15, 16, 17, 31, 193].map(|rows| (mode, rows))
                }) {
                    let values = (0..row_count)
                        .map(|index| {
                            if null_mode == 3 || (null_mode == 2 && index % 7 == 0) {
                                None
                            } else {
                                Some([
                                    <$native>::MIN,
                                    0,
                                    1,
                                    <$native>::MAX - 1,
                                    <$native>::MAX,
                                ][index % 5])
                            }
                        })
                        .collect::<Vec<Option<$native>>>();
                    let batch_for = |values: Vec<Option<$native>>| {
                        RecordBatch::try_from_iter_with_nullable([(
                            "value",
                            Arc::new(<$arrow>::from(values)) as ArrowArrayRef,
                            null_mode != 0,
                        )]).unwrap()
                    };
                    let batch = batch_for(values.clone());
                    for dictionary in [false, true] {
                        for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
                            let bytes = write_parquet_with_properties(
                                &batch,
                                WriterProperties::builder()
                                    .set_dictionary_enabled(dictionary)
                                    .set_writer_version(version)
                                    .set_statistics_enabled(EnabledStatistics::None)
                                    .build(),
                            );
                            for constant in [<$native>::MIN, 1, <$native>::MAX] {
                                for (make_expr, accepts) in [
                                    (vortex::expr::eq as fn(Expression, Expression) -> Expression,
                                     (|v: $native, c| v == c) as fn($native, $native) -> bool),
                                    (vortex::expr::not_eq, |v, c| v != c),
                                    (vortex::expr::gt, |v, c| v > c),
                                    (vortex::expr::gt_eq, |v, c| v >= c),
                                    (vortex::expr::lt, |v, c| v < c),
                                    (vortex::expr::lt_eq, |v, c| v <= c),
                                ] {
                                    let reader = open(&session, bytes.clone()).await.unwrap();
                                    let result = scan(
                                        &session,
                                        reader,
                                        Some(make_expr(get_item("value", root()), lit(constant))),
                                    ).await.unwrap();
                                    let expected = batch_for(values.iter().copied()
                                        .filter(|v| v.is_some_and(|v| accepts(v, constant)))
                                        .collect());
                                    let actual = result.execute_arrow(None, &mut session.create_execution_ctx(),
                                    ).unwrap();
                                    let expected = arrow_bridge::import_array(
                                        &ArrowStructArray::from(expected),
                                    ).unwrap();
                                    assert_eq!(actual.to_data(), expected.to_data(),
                                        "{}: {row_count} rows, null mode {null_mode}, dictionary {dictionary}, version {version:?}, constant {constant}",
                                        stringify!($native));
                                }
                            }
                        }
                    }
                }
            }};
        }
        check!(i8, Int8Array);
        check!(i16, Int16Array);
        check!(i64, Int64Array);
        check!(u8, UInt8Array);
        check!(u16, UInt16Array);
    })
}

#[test]
fn plain_byte_predicates_execute_while_the_parquet_decoder_is_live() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        // Exercise both ASCII and non-ASCII PLAIN length prefixes, with multibyte payloads.
        let long_128 = "é".repeat(64);
        let long_256 = "東京".repeat(42) + "abcd";
        let names = (0..60)
            .map(|index| match index % 12 {
                0 => Some(""),
                1 => None,
                2 => Some("label-01"),
                3 => Some("long-label-01-with-Unicode-東京"),
                4 => Some("another-long-value-without-a-match"),
                5 => Some("é"),
                6 => Some(long_128.as_str()),
                7 => Some(long_256.as_str()),
                8 => Some("ab"),
                9 => Some("cd"),
                10 => Some("\0\0a"),
                _ => Some("abc"),
            })
            .collect::<Vec<_>>();
        let batch_for = |values: Vec<Option<&str>>| {
            RecordBatch::try_from_iter([(
                "name",
                Arc::new(StringArray::from(values)) as ArrowArrayRef,
            )])
            .unwrap()
        };
        let batch = batch_for(names.clone());
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(batch.num_rows()))
            .set_data_page_row_count_limit(5)
            .set_write_batch_size(5)
            .set_dictionary_enabled(false)
            .build();
        let bytes = write_parquet_with_properties(&batch, properties);
        let size = bytes.len() as u64;

        // Prefix-only and cross-string candidates must not count; invalid candidates can
        // overlap a valid NUL-containing payload match, so advancing by needle length is wrong.
        for needle in [
            "label-01",
            "東京",
            "\0\0",
            "\u{3}\0\0\0abc",
            "ab\u{2}\0\0\0cd",
        ] {
            for negated in [false, true] {
                let like_format = ParquetFormat::new();
                let column = get_item("name", root());
                let pattern = lit(format!("%{needle}%"));
                let predicate = if negated {
                    vortex::expr::not_like(column, pattern)
                } else {
                    like(column, pattern)
                };
                let like_result = scan(
                    &session,
                    like_format
                        .open(FileHandle::new(
                            Arc::new(bytes.clone()),
                            size,
                            session.clone(),
                        ))
                        .await
                        .unwrap(),
                    Some(predicate),
                )
                .await
                .unwrap();
                let expected_like = batch_for(
                    names
                        .iter()
                        .copied()
                        .filter(|value| {
                            value.is_some_and(|value| value.contains(needle) != negated)
                        })
                        .collect(),
                );
                assert_eq!(
                    render(&like_result, &session),
                    render(&reference(&expected_like), &session),
                    "needle {needle:?}, negated {negated}"
                );
                assert!(like_format.metrics().fused_like_predicates() > 0);
            }
        }

        let compare_format = ParquetFormat::new();
        let compare_result = scan(
            &session,
            compare_format
                .open(FileHandle::new(Arc::new(bytes), size, session.clone()))
                .await
                .unwrap(),
            Some(not_eq(get_item("name", root()), lit(""))),
        )
        .await
        .unwrap();
        let expected_compare = batch_for(
            names
                .iter()
                .copied()
                .filter(|value| value.is_some_and(|value| !value.is_empty()))
                .collect(),
        );
        assert_eq!(
            render(&compare_result, &session),
            render(&reference(&expected_compare), &session)
        );
        assert!(compare_format.metrics().fused_compare_predicates() > 0);
    })
}

#[test]
fn a_file_without_an_offset_index_uses_lazy_header_pages() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int64Array::from_iter_values(0..400)) as ArrowArrayRef,
            ),
            (
                "name",
                Arc::new(StringArray::from_iter_values((0..400).map(|value| {
                    format!(
                        "row-{value:08x}-{a:016x}-{b:016x}-{c:016x}",
                        a = value * 1_000_003,
                        b = value * 1_000_033,
                        c = value * 1_000_037
                    )
                }))) as ArrowArrayRef,
            ),
        ])
        .unwrap();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(400))
            .set_data_page_row_count_limit(20)
            .set_data_page_size_limit(64)
            .set_write_batch_size(2)
            .set_offset_index_disabled(true)
            .set_dictionary_enabled(false)
            .build();
        let bytes = write_parquet_with_properties(&batch, properties);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let format = ParquetFormat::new();
        let opened = format
            .open_file(FileHandle::new(
                Arc::new(RecordingReadAt {
                    data: bytes,
                    reads: Arc::clone(&reads),
                }),
                size,
                session.clone(),
            ))
            .await
            .unwrap();
        let chunk = opened
            .layout
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(1)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap();
        assert!(chunk.slot(0).unwrap().unwrap().is::<ParquetHeaderPages>());
        let name_chunk = opened
            .layout
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(2)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .as_::<ParquetColumnChunk>()
            .data()
            .byte_range
            .clone();
        let open_reads = reads.lock().len();
        let dtype = opened.reader.dtype().clone();
        let result = ScanBuilder::new(session.clone(), opened.reader)
            .with_some_filter(Some(
                eq(get_item("id", root()), lit(1i64)).bind(&dtype).unwrap(),
            ))
            .with_projection(get_item("name", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
        assert!(format.metrics().snapshot().page_decodes > 0);
        assert_eq!(format.metrics().offset_index_decodes(), 0);
        let mut name_reads = reads.lock()[open_reads..]
            .iter()
            .filter_map(|read| {
                let start = read.start.max(name_chunk.start);
                let end = read.end.min(name_chunk.end);
                (start < end).then_some(start..end)
            })
            .collect::<Vec<_>>();
        name_reads.sort_by_key(|range| range.start);
        let mut merged: Vec<std::ops::Range<u64>> = Vec::new();
        for range in name_reads {
            if let Some(previous) = merged.last_mut()
                && range.start <= previous.end
            {
                previous.end = previous.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        let name_bytes = merged
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>();
        assert!(
            name_bytes < name_chunk.end - name_chunk.start,
            "header discovery fetched the complete projected column chunk"
        );
    })
}

#[test]
fn a_sparse_scan_loads_the_available_offset_index_lazily() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(400))
            .set_data_page_row_count_limit(20)
            .set_data_page_size_limit(64)
            .set_write_batch_size(10)
            .build();
        let bytes = write_parquet_with_properties(&flat_table_n(400), properties);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let format = ParquetFormat::new();
        let opened = format
            .open_file(FileHandle::new(
                Arc::new(RecordingReadAt {
                    data: bytes,
                    reads: Arc::clone(&reads),
                }),
                size,
                session.clone(),
            ))
            .await
            .unwrap();
        let fields = opened
            .layout
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap();
        let chunk_ranges = [2usize].map(|field| {
            fields
                .slot(field + 1)
                .unwrap()
                .unwrap()
                .slot(0)
                .unwrap()
                .unwrap()
                .as_::<ParquetColumnChunk>()
                .data()
                .byte_range
                .clone()
        });
        let reader = opened.reader;
        let open_reads = reads.lock().len();
        let dtype = reader.dtype().clone();
        let result = ScanBuilder::new(session.clone(), Arc::clone(&reader))
            .with_row_indices(
                StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter([399u64])).unwrap(),
            )
            .with_projection(get_item("name", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(reads.lock().len() > open_reads);
        assert!(format.metrics().offset_index_decodes() > 0);
        assert!(format.metrics().snapshot().page_decodes > 0);
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
        let data_reads = reads.lock()[open_reads..].to_vec();
        for chunk in chunk_ranges {
            let mut covered = data_reads
                .iter()
                .filter_map(|read| {
                    let start = read.start.max(chunk.start);
                    let end = read.end.min(chunk.end);
                    (start < end).then_some(start..end)
                })
                .collect::<Vec<_>>();
            covered.sort_by_key(|range| range.start);
            let mut merged: Vec<std::ops::Range<u64>> = Vec::new();
            for range in covered {
                if let Some(previous) = merged.last_mut()
                    && range.start <= previous.end
                {
                    previous.end = previous.end.max(range.end);
                } else {
                    merged.push(range);
                }
            }
            let bytes_read = merged
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>();
            assert!(
                bytes_read < chunk.end - chunk.start,
                "sparse scan fetched the complete column chunk {chunk:?}: {data_reads:?}"
            );
        }

        let decoded_indexes = format.metrics().offset_index_decodes();
        let repeated = ScanBuilder::new(session.clone(), Arc::clone(&reader))
            .with_row_indices(
                StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter([399u64])).unwrap(),
            )
            .with_projection(get_item("name", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(repeated.len(), 1);
        assert_eq!(
            format.metrics().offset_index_decodes(),
            decoded_indexes,
            "immutable offset indexes should be decoded once per opened column"
        );

        // Clip a sparse, non-byte-aligned selection across page boundaries. The null entry and
        // excluded indices on either side protect the placement of copied mask bits.
        let clipped = ScanBuilder::new(session.clone(), reader)
            .with_row_range(7..193)
            .with_row_indices(
                StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter([
                    0u64, 3, 19, 20, 41, 185, 399,
                ]))
                .unwrap(),
            )
            .with_projection(get_item("name", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        let expected = RecordBatch::try_from_iter([(
            "name",
            Arc::new(StringArray::from(vec![
                Some("label-01"),
                Some("label-02"),
                Some("label-05"),
                None,
            ])) as ArrowArrayRef,
        )])
        .unwrap();
        assert_eq!(
            render(&clipped, &session),
            render(
                &reference(&expected)
                    .apply(&get_item("name", root()))
                    .unwrap(),
                &session
            ),
        );
    })
}

#[test]
fn dynamic_filters_refine_footer_and_page_pruning_without_reloading_indexes() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(400))
            .set_data_page_row_count_limit(20)
            .set_data_page_size_limit(64)
            .set_write_batch_size(10)
            .set_dictionary_enabled(false)
            .build();
        let bytes = write_parquet_with_properties(&flat_table_n(400), properties);
        let format = ParquetFormat::new();
        let reader = format
            .open(FileHandle::new(
                Arc::new(bytes.clone()),
                bytes.len() as u64,
                session.clone(),
            ))
            .await
            .unwrap();

        let threshold = Arc::new(AtomicI64::new(400));
        let dynamic_threshold = Arc::clone(&threshold);
        let expr = dynamic(
            CompareOperator::Lt,
            move || Some(dynamic_threshold.load(Ordering::SeqCst).into()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
            true,
            get_item("id", root()),
        )
        .bind(reader.dtype())
        .unwrap();

        // Construct at 400, update before polling: page pruning must observe 31, not the stale RHS.
        let first = reader
            .pruning_evaluation(&(0..400), &expr, Mask::new_true(400))
            .unwrap();
        threshold.store(31, Ordering::SeqCst);
        let first = first.await.unwrap();
        assert!(
            first.true_count() > 0 && first.true_count() < 400,
            "dynamic page mask kept {} rows; metrics: {}",
            first.true_count(),
            format.metrics()
        );
        assert_eq!(format.metrics().page_index_decodes(), 1);

        // The same timing at row-group granularity: the footer now proves the whole group false,
        // without touching the already-loaded page index (or issuing another index decode).
        let second = reader
            .pruning_evaluation(&(0..400), &expr, Mask::new_true(400))
            .unwrap();
        threshold.store(-1, Ordering::SeqCst);
        assert!(second.await.unwrap().all_false());
        assert_eq!(format.metrics().page_index_decodes(), 1);

        // Dynamic filters normally tighten, but reevaluating a looser value is useful protection
        // against accidentally caching a predicate result instead of immutable index metadata.
        threshold.store(211, Ordering::SeqCst);
        let third = reader
            .pruning_evaluation(&(0..400), &expr, Mask::new_true(400))
            .unwrap()
            .await
            .unwrap();
        assert!(third.true_count() > first.true_count() && third.true_count() < 400);
        assert_eq!(format.metrics().page_index_decodes(), 1);

        // The scan engine observes the same dynamic value through its revision tracker. Once the
        // filter excludes this row group, its projected string column is never fetched or decoded.
        threshold.store(-1, Ordering::SeqCst);
        let dtype = reader.dtype().clone();
        let result = ScanBuilder::new(session.clone(), reader)
            .with_some_filter(Some(expr))
            .with_projection(get_item("name", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(format.metrics().snapshot().page_decodes, 0);
        assert_eq!(format.metrics().snapshot().arrow_decodes, 0);
    })
}

#[test]
fn row_group_boundaries_are_vortex_splits() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let reader = open(&session, write_parquet(&flat_table(), 16))
            .await
            .unwrap();
        let splits = SplitBy::Layout
            .splits(reader.as_ref(), &(0..reader.row_count()), &[FieldMask::All])
            .unwrap();
        assert!(splits.contains(&16) && splits.contains(&32), "{splits:?}");
    })
}

#[test]
fn filters_crossing_a_row_group_boundary_are_correct() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let base = flat_table();
        for nullable in [false, true] {
            // Cover every PLAIN numeric width, decimal storage, optional all-valid levels,
            // and the null-bearing numeric/string columns that must retain their nulls.
            let batch = RecordBatch::try_from_iter_with_nullable([
                ("id", Arc::clone(base.column(0)), false),
                ("score", Arc::clone(base.column(1)), true),
                ("name", Arc::clone(base.column(2)), true),
                (
                    "i32",
                    Arc::new(Int32Array::from_iter_values(
                        (0..40).map(|i| i32::MIN + i * 97),
                    )) as ArrowArrayRef,
                    nullable,
                ),
                (
                    "f32",
                    Arc::new(Float32Array::from_iter_values(
                        (0..40).map(|i| i as f32 * -1.25),
                    )) as ArrowArrayRef,
                    nullable,
                ),
                (
                    "f64",
                    Arc::new(Float64Array::from_iter_values(
                        (0..40).map(|i| f64::from(i) / 8.0 - 0.125),
                    )) as ArrowArrayRef,
                    nullable,
                ),
                (
                    "decimal",
                    Arc::new(
                        Decimal64Array::from_iter_values((0..40).map(|i| i64::from(i) * -12345))
                            .with_precision_and_scale(15, 2)
                            .unwrap(),
                    ) as ArrowArrayRef,
                    nullable,
                ),
            ])
            .unwrap();
            let selected = |slices: &[(usize, usize)]| {
                ChunkedArray::try_new(
                    slices
                        .iter()
                        .map(|&(offset, len)| reference(&batch.slice(offset, len))),
                    reference(&batch).dtype().clone(),
                )
                .unwrap()
                .into_array()
            };
            for dictionary in [false, true] {
                for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
                    let bytes = write_parquet_with_properties(
                        &batch,
                        WriterProperties::builder()
                            .set_max_row_group_row_count(Some(16))
                            .set_data_page_row_count_limit(4)
                            .set_write_batch_size(4)
                            .set_dictionary_enabled(dictionary)
                            .set_writer_version(version)
                            .build(),
                    );
                    let reader = ParquetFormat::new()
                        .with_options(
                            ReaderOptions::default()
                                .with_filter_kernel_threshold(1.0)
                                .unwrap(),
                        )
                        .open(FileHandle::new(
                            Arc::new(bytes.clone()),
                            bytes.len() as u64,
                            session.clone(),
                        ))
                        .await
                        .unwrap();
                    for (predicate, expected) in [
                        (
                            or(
                                or(
                                    eq(get_item("id", root()), lit(0i64)),
                                    eq(get_item("id", root()), lit(2i64)),
                                ),
                                or(
                                    eq(get_item("id", root()), lit(33i64)),
                                    eq(get_item("id", root()), lit(35i64)),
                                ),
                            ),
                            selected(&[(0, 1), (2, 1), (33, 1), (35, 1)]),
                        ),
                        (
                            gt_eq(get_item("id", root()), lit(25i64)),
                            reference(&batch.slice(25, 15)),
                        ),
                        // Dense and sparse masks must preserve positions across pages/row groups.
                        (
                            not_eq(get_item("id", root()), lit(25i64)),
                            selected(&[(0, 25), (26, 14)]),
                        ),
                    ] {
                        let result = scan(&session, Arc::clone(&reader), Some(predicate))
                            .await
                            .unwrap();
                        assert_eq!(render(&result, &session), render(&expected, &session));
                    }
                }
            }
        }
    })
}

#[test]
fn cache_policy_does_not_change_the_answer() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = write_parquet(&flat_table(), 16);
        let mut answers = Vec::new();
        for policy in [
            CachePolicy::Off,
            CachePolicy::Ephemeral,
            CachePolicy::Bounded(1 << 20),
            CachePolicy::JustSendIt,
        ] {
            let size = bytes.len() as u64;
            let reader = ParquetFormat::new()
                .with_cache_policy(policy)
                .open(FileHandle::new(
                    Arc::new(bytes.clone()),
                    size,
                    session.clone(),
                ))
                .await
                .unwrap();
            let result = scan(
                &session,
                reader,
                Some(gt_eq(get_item("id", root()), lit(25i64))),
            )
            .await
            .unwrap();
            answers.push(render(&result, &session));
        }
        assert!(answers.windows(2).all(|pair| pair[0] == pair[1]));
    })
}

#[test]
fn malformed_files_are_rejected() {
    fn first_data_page_payload(bytes: &[u8]) -> std::ops::Range<usize> {
        let metadata =
            ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
        let offset = usize::try_from(
            metadata
                .metadata()
                .row_group(0)
                .column(0)
                .data_page_offset(),
        )
        .unwrap();
        let mut cursor = std::io::Cursor::new(&bytes[offset..]);
        #[allow(deprecated)] // Match the pinned parquet-rs page-header format used by the reader.
        let compressed_size = parquet::format::PageHeader::read_from_in_protocol(
            &mut thrift::protocol::TCompactInputProtocol::new(&mut cursor),
        )
        .unwrap()
        .compressed_page_size;
        let start = offset + usize::try_from(cursor.position()).unwrap();
        let end = start + usize::try_from(compressed_size).unwrap();
        start..end
    }

    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        for bytes in [
            ByteBuffer::copy_from(b"this is not parquet"),
            ByteBuffer::copy_from(b"PAR1"),
        ] {
            assert!(open(&session, bytes).await.is_err());
        }

        // PLAIN physical INT32 values must fit their narrower logical annotation. Reject both
        // sides of each domain instead of silently wrapping during native integer conversion.
        for (array, invalid, constant) in [
            (
                Arc::new(Int8Array::from(vec![0; 193])) as ArrowArrayRef,
                [-129i32, 128],
                Scalar::from(1i8),
            ),
            (
                Arc::new(Int16Array::from(vec![0; 193])) as ArrowArrayRef,
                [-32769, 32768],
                Scalar::from(1i16),
            ),
            (
                Arc::new(UInt8Array::from(vec![0; 193])) as ArrowArrayRef,
                [-1, 256],
                Scalar::from(1u8),
            ),
            (
                Arc::new(UInt16Array::from(vec![0; 193])) as ArrowArrayRef,
                [-1, 65536],
                Scalar::from(1u16),
            ),
        ] {
            let batch =
                RecordBatch::try_from_iter_with_nullable([("value", array, false)]).unwrap();
            let bytes = write_parquet_with_properties(
                &batch,
                WriterProperties::builder()
                    .set_dictionary_enabled(false)
                    .set_compression(Compression::UNCOMPRESSED)
                    .set_statistics_enabled(EnabledStatistics::None)
                    .build(),
            );
            let payload = first_data_page_payload(&bytes);
            // Include vector-block interiors/boundaries, a scalar tail, and repeated errors:
            // two out-of-range values must never cancel one another during validation.
            for invalid in invalid {
                for indices in [&[0][..], &[63], &[128], &[192], &[0, 128]] {
                    let mut corrupted = bytes.to_vec();
                    for &index in indices {
                        let offset = payload.start + index * 4;
                        corrupted[offset..offset + 4].copy_from_slice(&invalid.to_le_bytes());
                    }
                    // The comparison rejects every row, including corrupt values, but must
                    // still validate the entire physical payload before producing its mask.
                    for predicate in [
                        None,
                        Some(eq(get_item("value", root()), lit(constant.clone()))),
                    ] {
                        let reader = open(&session, ByteBuffer::copy_from(&corrupted))
                            .await
                            .unwrap();
                        assert!(
                            scan(&session, reader, predicate)
                                .await
                                .and_then(|array| {
                                    array
                                        .execute_arrow(None, &mut session.create_execution_ctx())
                                        .map(|_| ())
                                })
                                .is_err(),
                            "out-of-range narrow integer {invalid} at indices {indices:?} was accepted"
                        );
                    }
                }
            }
        }

        // Corrupt a code while keeping dictionary values, run lengths, and file/page headers
        // valid. The unused two-bit code 3 must be rejected wherever it occurs in the stream.
        let batch = RecordBatch::try_from_iter_with_nullable([(
            "text",
            Arc::new(StringArray::from_iter_values(
                (0..192).map(|index| ["alpha", "beta", "gamma"][index / 64]),
            )) as ArrowArrayRef,
            false,
        )])
        .unwrap();
        let bytes = write_parquet_with_properties(
            &batch,
            WriterProperties::builder()
                .set_dictionary_enabled(true)
                .set_compression(Compression::UNCOMPRESSED)
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
        );
        let payload = first_data_page_payload(&bytes);
        let start = payload.start;
        let end = payload.end;
        assert_eq!(bytes[start], 2);
        let mut position = start + 1;
        let mut code_positions = Vec::new();
        while position < end {
            let mut run = 0usize;
            let mut shift = 0;
            loop {
                let byte = bytes[position];
                position += 1;
                run |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            code_positions.push(position);
            // A bit-packed group has eight two-bit codes; an RLE value occupies one byte.
            position += if run & 1 == 1 { (run >> 1) * 2 } else { 1 };
        }
        for position in [code_positions[0], *code_positions.last().unwrap()] {
            let mut malformed = bytes.to_vec();
            malformed[position] |= 3;
            let reader = open(&session, ByteBuffer::from(malformed)).await.unwrap();
            let error = scan(&session, reader, None)
                .await
                .and_then(|array| {
                    array
                        .execute_arrow(None, &mut session.create_execution_ctx())
                        .map(|_| ())
                })
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("dictionary code exceeds dictionary length")
            );
        }

        // A flat optional field has only levels 0 and 1. An out-of-range repeated level must
        // fail validation rather than silently replacing all the values with nulls.
        let batch = RecordBatch::try_from_iter_with_nullable([(
            "value",
            Arc::new(Int16Array::from_iter_values(0..192)) as ArrowArrayRef,
            true,
        )])
        .unwrap();
        let bytes = write_parquet_with_properties(
            &batch,
            WriterProperties::builder()
                .set_writer_version(WriterVersion::PARQUET_1_0)
                .set_dictionary_enabled(false)
                .set_compression(Compression::UNCOMPRESSED)
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
        );
        let start = first_data_page_payload(&bytes).start;
        let level_bytes = usize::try_from(u32::from_le_bytes(
            bytes[start..start + 4].try_into().unwrap(),
        ))
        .unwrap();
        assert_eq!(bytes[start + 4] & 1, 0, "fixture must use a repeated level");
        let level = start + 4 + level_bytes - 1;
        assert_eq!(bytes[level], 1);
        let mut malformed = bytes.to_vec();
        malformed[level] = u8::MAX;
        let reader = open(&session, ByteBuffer::from(malformed)).await.unwrap();
        let error = scan(&session, reader, None)
            .await
            .and_then(|array| {
                array
                    .execute_arrow(None, &mut session.create_execution_ctx())
                    .map(|_| ())
            })
            .unwrap_err();
        assert!(error.to_string().contains("definition levels"));

        // Keep the file and page headers valid, but make the PLAIN length prefixes exceed the
        // page. Lazy opening may succeed; executing the column must report an error, not panic.
        let first = b"first-value-12345";
        let second = b"second-value-6789";
        let batch = RecordBatch::try_from_iter_with_nullable([(
            "bytes",
            Arc::new(BinaryArray::from_vec(vec![
                first.as_slice(),
                second.as_slice(),
            ])) as ArrowArrayRef,
            false,
        )])
        .unwrap();
        let bytes = write_parquet_with_properties(
            &batch,
            WriterProperties::builder()
                .set_dictionary_enabled(false)
                .set_compression(Compression::UNCOMPRESSED)
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
        );
        let positions = bytes
            .windows(first.len())
            .enumerate()
            .filter_map(|(index, value)| (value == first).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 1);
        let prefix = positions[0] - 4;
        // An oversized payload, then a payload leaving only three bytes for the next length.
        for length in [
            u32::MAX,
            u32::try_from(first.len() + 4 + second.len() - 3).unwrap(),
        ] {
            let mut malformed = bytes.to_vec();
            malformed[prefix..prefix + 4].copy_from_slice(&length.to_le_bytes());
            for predicate in [
                None,
                Some(like(
                    vortex::expr::cast(
                        get_item("bytes", root()),
                        DType::Utf8(Nullability::NonNullable),
                    ),
                    lit("%not-present%"),
                )),
            ] {
                let reader = open(&session, ByteBuffer::from(malformed.clone()))
                    .await
                    .unwrap();
                let result = scan(&session, reader, predicate).await.and_then(|array| {
                    array
                        .execute_arrow(None, &mut session.create_execution_ctx())
                        .map(|_| ())
                });
                assert!(
                    result.is_err(),
                    "invalid PLAIN byte-array length {length} was accepted"
                );
            }
        }

        // Bulk UTF-8 checks must exclude length prefixes but still reject each invalid string.
        // In particular, a lead byte at one string's end cannot consume the next string's
        // continuation byte, even if concatenating just their payloads would be valid UTF-8.
        let values = [
            "first-string-1234".to_string(),
            "second-string-567".to_string(),
            "x".repeat(128),
        ];
        let batch = RecordBatch::try_from_iter([(
            "text",
            Arc::new(StringArray::from_iter_values(&values)) as ArrowArrayRef,
        )])
        .unwrap();
        let bytes = write_parquet_with_properties(
            &batch,
            WriterProperties::builder()
                .set_dictionary_enabled(false)
                .set_compression(Compression::UNCOMPRESSED)
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
        );
        let starts = values
            .iter()
            .map(|value| {
                let positions = bytes
                    .windows(value.len())
                    .enumerate()
                    .filter_map(|(index, candidate)| {
                        (candidate == value.as_bytes()).then_some(index)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(positions.len(), 1);
                positions[0]
            })
            .collect::<Vec<_>>();
        for mutations in [
            vec![(starts[0], 0xff)],
            vec![(starts[0] + values[0].len() - 1, 0xc3), (starts[1], 0xa9)],
            vec![(starts[2], 0x80)],
            vec![(starts[2] + values[2].len() - 1, 0xc3)],
        ] {
            let mut malformed = bytes.to_vec();
            for (offset, value) in mutations {
                malformed[offset] = value;
            }
            let reader = open(&session, ByteBuffer::from(malformed)).await.unwrap();
            let result = scan(&session, reader, None)
                .await
                .and_then(|array| array.execute_arrow(None, &mut session.create_execution_ctx()));
            assert!(result.is_err(), "invalid PLAIN UTF-8 string was accepted");
        }
    })
}

/// A sparse page-range read over a **nested** column must use the selected row group's indexes.
///
/// # Why this test exists
///
/// The nested/Dremel fallback is the one decode path that needs a `ParquetMetaData` carrying an
/// offset index, because arrow-rs resolves page ranges through it. That metadata is assembled for
/// the call and dropped. Assembling it wrongly — a misplaced row group, a dropped
/// leaf — would not fail loudly. It would return the wrong rows for nested fields only, on a path no
/// flat benchmark exercises and no other test drove with a *sparse* selection.
///
/// So this pins both the observable answer and the route that produced it: one nested Arrow decode,
/// one selective decode, and less than the whole nested column chunk fetched.
#[test]
fn a_sparse_page_range_over_a_nested_column_uses_its_row_group_indexes() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);

        const ROWS_PER_GROUP: usize = 256;
        const ROWS: usize = ROWS_PER_GROUP * 2;
        const SELECTED_ROWS: [usize; 2] = [ROWS_PER_GROUP, ROWS - 1];
        let inner_fields = Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]);
        let inner = ArrowStructArray::new(
            inner_fields.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    0..i32::try_from(ROWS).unwrap(),
                )) as ArrowArrayRef,
                Arc::new(StringArray::from_iter((0..ROWS).map(|i| {
                    (i % 7 != 0).then(|| format!("inner-{i:04}-{}", "x".repeat(128)))
                }))) as ArrowArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_from_iter_with_nullable([
            (
                "id",
                Arc::new(Int64Array::from_iter_values(
                    0..i64::try_from(ROWS).unwrap(),
                )) as ArrowArrayRef,
                false,
            ),
            ("nested", Arc::new(inner) as ArrowArrayRef, false),
        ])
        .unwrap();

        // No dictionary prefix and many small data pages make the byte-cost decision deterministic:
        // the last page of each nested leaf is well below 80% of the complete nested chunk.
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
            .set_data_page_row_count_limit(8)
            .set_data_page_size_limit(256)
            .set_write_batch_size(8)
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .build();
        let bytes = write_parquet_with_properties(&batch, properties);
        let size = bytes.len() as u64;
        let reads = Arc::new(Mutex::new(Vec::new()));
        let format = ParquetFormat::new();
        let opened = format
            .open_file(FileHandle::new(
                Arc::new(RecordingReadAt {
                    data: bytes,
                    reads: Arc::clone(&reads),
                }),
                size,
                session.clone(),
            ))
            .await
            .unwrap();
        let nested_chunk = opened
            .layout
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(1)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .slot(2)
            .unwrap()
            .unwrap()
            .slot(0)
            .unwrap()
            .unwrap()
            .as_::<ParquetColumnChunk>()
            .data()
            .byte_range
            .clone();
        let open_reads = reads.lock().len();

        // Selecting the first and last records of only row group 1 keeps its mask below the 10%
        // selective-decode threshold while forcing a range spanning the whole row group.
        // `get_mut(0)` therefore attaches these absolute page locations to the wrong row group.
        // Projecting only the nested field keeps another column's decode from obscuring the route.
        let dtype = opened.reader.dtype().clone();
        let result = ScanBuilder::new(session.clone(), opened.reader)
            .with_row_indices(
                StrictSortedBuffer::try_new(vortex::buffer::Buffer::from_iter(
                    SELECTED_ROWS.map(|row| u64::try_from(row).unwrap()),
                ))
                .unwrap(),
            )
            .with_projection(get_item("nested", root()).bind(&dtype).unwrap())
            .with_ordered(true)
            .into_array_stream()
            .unwrap()
            .read_all()
            .await
            .unwrap();

        let expected = ArrowStructArray::new(
            inner_fields,
            vec![
                Arc::new(Int32Array::from_iter_values(
                    SELECTED_ROWS.map(|row| i32::try_from(row).unwrap()),
                )) as ArrowArrayRef,
                Arc::new(StringArray::from_iter(SELECTED_ROWS.map(|row| {
                    (row % 7 != 0).then(|| format!("inner-{row:04}-{}", "x".repeat(128)))
                }))) as ArrowArrayRef,
            ],
            None,
        );
        assert_eq!(
            render(&result, &session),
            render(&reference(&RecordBatch::from(expected)), &session)
        );

        let counts = format.metrics().snapshot();
        assert_eq!(counts.offset_index_decodes, 2, "{counts:?}");
        assert_eq!(counts.arrow_decodes, 1, "{counts:?}");
        assert_eq!(counts.page_decodes, 0, "{counts:?}");
        assert_eq!(counts.selective_decodes, 1, "{counts:?}");
        assert_eq!(counts.selective_rows, 2, "{counts:?}");
        assert_eq!(
            counts.selective_rows_available,
            u64::try_from(ROWS_PER_GROUP).unwrap(),
            "{counts:?}"
        );

        let data_reads = reads.lock()[open_reads..].to_vec();
        let mut covered = data_reads
            .iter()
            .filter_map(|read| {
                let start = read.start.max(nested_chunk.start);
                let end = read.end.min(nested_chunk.end);
                (start < end).then_some(start..end)
            })
            .collect::<Vec<_>>();
        covered.sort_by_key(|range| range.start);
        let mut merged: Vec<std::ops::Range<u64>> = Vec::new();
        for range in covered {
            if let Some(previous) = merged.last_mut()
                && range.start <= previous.end
            {
                previous.end = previous.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        let bytes_read = merged
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>();
        assert!(
            bytes_read < nested_chunk.end - nested_chunk.start,
            "nested page-range decode fetched the complete chunk {nested_chunk:?}: {data_reads:?}"
        );
    })
}
#[test]
fn int64_decimal_hint_keeps_public_schema_and_nulls() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = integer_decimal_file(15, &[12, 34, 56], &[1, 1, 0, 1], true);
        let size = bytes.len() as u64;
        let format = ParquetFormat::new()
            .with_options(ReaderOptions::default().with_materialized_projections());
        let opened = format
            .open_file(FileHandle::new(Arc::new(bytes), size, session.clone()))
            .await
            .unwrap();
        let context = file_context(&opened);
        assert_eq!(
            context.arrow_metadata.schema().field(1).data_type(),
            &DataType::Decimal128(15, 0)
        );
        assert_eq!(
            context.decoder_metadata.schema().field(1).data_type(),
            &DataType::Decimal64(15, 0)
        );
        let source = format
            .open_data_source_with_metadata(context.file.clone(), Arc::clone(&context.metadata))
            .unwrap();
        let stats = source
            .field_statistics(&FieldPath::from_name("amount"))
            .await
            .unwrap();
        assert_eq!(
            stats
                .get_as::<u64>(
                    Stat::UncompressedSizeInBytes,
                    &DType::Primitive(PType::U64, Nullability::NonNullable)
                )
                .as_exact(),
            Some(4 * 16)
        );
        let (reader, decoder_schema) = physical_reader(&opened, 1, &session);
        assert_eq!(
            decoder_schema.field(1).data_type(),
            &DataType::Decimal64(15, 0)
        );
        let projection = root().bind(reader.dtype()).unwrap();
        let actual = reader
            .projection_evaluation(&(0..4), &projection, MaskFuture::ready(Mask::new_true(4)))
            .unwrap()
            .await
            .unwrap();
        let decoded = actual
            .execute_until::<Decimal>(&mut session.create_execution_ctx())
            .unwrap();
        assert_eq!(decoded.as_::<Decimal>().values_type(), DecimalType::I64);
        assert_amounts_with_precision(
            &decoded,
            &[Some(12), Some(34), None, Some(56)],
            15,
            &session,
        );
        assert!(format.metrics().snapshot().arrow_decodes > 0);
    })
}

#[test]
fn int32_decimal_hint_keeps_public_schema_and_scale() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let values = Decimal128Array::from(vec![Some(12_i128), None, Some(34)])
            .with_precision_and_scale(7, 2)
            .unwrap();
        let batch =
            RecordBatch::try_from_iter([("amount", Arc::new(values) as ArrowArrayRef)]).unwrap();
        let bytes = write_parquet(&batch, 3);
        let size = bytes.len() as u64;
        let format = ParquetFormat::new()
            .with_options(ReaderOptions::default().with_materialized_projections());
        let opened = format
            .open_file(FileHandle::new(Arc::new(bytes), size, session.clone()))
            .await
            .unwrap();
        let context = file_context(&opened);
        assert_eq!(
            context.arrow_metadata.schema().field(0).data_type(),
            &DataType::Decimal128(7, 2)
        );
        assert_eq!(
            context.decoder_metadata.schema().field(0).data_type(),
            &DataType::Decimal32(7, 2)
        );
        let actual = scan(&session, opened.reader, None).await.unwrap();
        assert_eq!(
            render(&actual, &session),
            render(&reference(&batch), &session)
        );
    })
}

#[test]
fn decimal_hint_checks_selected_precision() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = integer_decimal_file(
            15,
            &[12, 34, 1_000_000_000_000_000_i64, 56],
            &[1, 1, 1, 1],
            true,
        );
        let error = read_materialized_amounts(&session, bytes, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("precision"), "{error}");
    })
}

#[test]
fn decimal_hint_ignores_rejected_precision_overflow() {
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let bytes = integer_decimal_file(
            15,
            &[12, 34, 1_000_000_000_000_000_i64, 56],
            &[1, 1, 1, 1],
            true,
        );
        let actual = read_materialized_amounts(
            &session,
            bytes,
            Some(not_eq(get_item("id", root()), lit(2_i64))),
        )
        .await
        .unwrap();
        assert_amounts_with_precision(&actual, &[Some(12), Some(34), Some(56)], 15, &session);
    })
}
