// SPDX-License-Identifier: Apache-2.0

//! Parquet data pages as lazy Vortex arrays.
//!
//! Parquet's wire encodings are not Vortex encodings. In particular, Parquet's hybrid RLE/bit-pack
//! stream is not FastLanes bit-packing. This module therefore keeps the Parquet page and uses
//! parquet-rs's decoders, with a direct PLAIN byte-array path. The array boundary is useful: scalar
//! functions can be fused with the decoder later, without changing the layout or teaching the scan
//! engine about Parquet.

// Parquet keeps BIT_PACKED and PLAIN_DICTIONARY for reading legacy files. They still need distinct
// Vortex wrapper vtables even though new writers should not emit them.
#![allow(deprecated)]

mod buffer_pool;
mod dictionary_set;
mod hybrid_codes;

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
mod neon_compare;

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
mod neon_narrow;

use std::any::Any;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::io::Read;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::Once;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use parquet::basic::Compression;
use parquet::basic::Encoding;
use parquet::basic::Type as PhysicalType;
use parquet::column::page::Page;
use parquet::column::reader::ColumnReaderImpl;
use parquet::compression::CodecOptions;
use parquet::compression::create_codec;
use parquet::data_type::BoolType;
use parquet::data_type::ByteArray;
use parquet::data_type::ByteArrayType;
use parquet::data_type::DataType as ParquetDataType;
use parquet::data_type::DoubleType;
use parquet::data_type::FixedLenByteArray;
use parquet::data_type::FixedLenByteArrayType;
use parquet::data_type::FloatType;
use parquet::data_type::Int32Type;
use parquet::data_type::Int64Type;
use parquet::data_type::Int96;
use parquet::data_type::Int96Type;
use parquet::encodings::decoding::Decoder;
use parquet::encodings::decoding::DictDecoder;
use parquet::encodings::decoding::get_decoder;
use parquet::encodings::rle::RleDecoder;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::reader::ChunkReader;
use parquet::file::serialized_reader::SerializedPageReader;
use parquet::format::PageHeader;
#[allow(deprecated)]
use parquet::format::PageLocation;
use parquet::format::PageType;
use parquet::schema::types::ColumnDescPtr;
use parquet::schema::types::SchemaDescriptor;
use parquet::thrift::TSerializable;
use parquet::util::bit_util::BitReader;
use parquet::util::bit_util::num_required_bits;
use thrift::protocol::TCompactInputProtocol;
use vortex::array::Array;
use vortex::array::ArrayEq;
use vortex::array::ArrayHash;
use vortex::array::ArrayParts;
use vortex::array::ArrayRef;
use vortex::array::ArraySlots;
use vortex::array::ArrayView;
use vortex::array::EqMode;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::Filter;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::ScalarFnArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::decimal::copy_to_canonical_width;
use vortex::array::arrays::filter::FilterExecuteAdaptor;
use vortex::array::arrays::filter::FilterKernel;
use vortex::array::arrays::primitive::PrimitiveArrayExt;
use vortex::array::arrays::scalar_fn::AnyScalarFn;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::buffer::BufferHandle;
use vortex::array::builders::ArrayBuilder;
use vortex::array::builders::VarBinViewBuilder;
use vortex::array::builders::builder_with_capacity;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::optimizer::rules::ArrayParentReduceRule;
use vortex::array::validity::Validity;
use vortex::array::vtable::OperationsVTable;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityVTable;
use vortex::array::vtable::with_empty_buffers;
use vortex::buffer::Alignment;
use vortex::buffer::BitBuffer;
use vortex::buffer::BitBufferMut;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::TemporalMetadata;
use vortex::extension::datetime::TimeUnit;
use vortex::functions::byte_length::ByteLength;
use vortex::functions::byte_length::ByteLengthExecuteAdaptor;
use vortex::functions::byte_length::ByteLengthKernel;
use vortex::functions::like::Like;
use vortex::functions::like::LikeExecuteAdaptor;
use vortex::functions::like::LikeKernel;
use vortex::functions::like::LikeOptions;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::builtin::cast::Cast;
use vortex::scalar_fn::builtin::cast::CastExecuteAdaptor;
use vortex::scalar_fn::builtin::cast::CastKernel;
use vortex::scalar_fn::builtin::compare::Compare;
use vortex::scalar_fn::builtin::compare::CompareExecuteAdaptor;
use vortex::scalar_fn::builtin::compare::CompareKernel;
use vortex::scalar_fn::builtin::operators::CompareOperator;
use vortex::session::SessionExt;
use vortex::session::SessionVar;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;

use crate::chunk_reader::SegmentChunkReader;
use crate::metrics::ScanMetrics;
use crate::options::ReaderOptions;
use crate::options::SnappyDecoderKind;

/// Returns the physical leaf for a top-level field when that field is represented by one leaf.
pub fn single_leaf_index(schema: &SchemaDescriptor, root_column: usize) -> Option<usize> {
    let mut found = None;
    for leaf in 0..schema.num_columns() {
        if schema.get_column_root_idx(leaf) == root_column {
            if found.is_some() {
                return None;
            }
            found = Some(leaf);
        }
    }
    found
}

/// Byte values retained when a fused predicate decodes them before projection.
#[derive(Debug)]
struct DecodedByteValues {
    values: ByteValues,
    valid: Option<Arc<[bool]>>,
}

/// PLAIN values borrow one page buffer instead of allocating a reference-counted `Bytes` handle
/// for every row. Other encodings can reconstruct strings, so retain their decoder-owned values.
#[derive(Debug)]
enum ByteValues {
    Plain {
        buffer: ByteBuffer,
        views: Buffer<BinaryView>,
        utf8_validated: OnceLock<()>,
    },
    Decoded(Vec<ByteArray>),
}

impl ByteValues {
    fn len(&self) -> usize {
        match self {
            Self::Plain { views, .. } => views.len(),
            Self::Decoded(values) => values.len(),
        }
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone {
        (0..self.len()).map(|index| match self {
            Self::Plain { buffer, views, .. } => {
                let view = &views[index];
                if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let view = view.as_view();
                    let start = view.offset as usize;
                    &buffer[start..start + view.size as usize]
                }
            }
            Self::Decoded(values) => values[index].data(),
        })
    }
}

/// A Parquet page whose outer compression is deliberately left intact.
///
/// `Page` is used for the header-derived shape because parquet-rs's value decoders already consume
/// it, but its buffer is still the compressed wire payload until [`Self::page`] is called.  The
/// decompressed page is retained only by arrays which are still live; dropping the array releases
/// both buffers and prevents a scan-wide decompression cache.
#[derive(Clone, Debug)]
struct LazyParquetPage {
    compressed: Arc<Page>,
    uncompressed_size: usize,
    compression: Compression,
    crc: Option<u32>,
    decompressed: Arc<OnceLock<Arc<Page>>>,
    snappy_decoder: Option<SnappyDecoderKind>,
}

impl LazyParquetPage {
    fn new(
        compressed: Page,
        uncompressed_size: usize,
        compression: Compression,
        crc: Option<u32>,
    ) -> Self {
        Self {
            compressed: Arc::new(compressed),
            uncompressed_size,
            compression,
            crc,
            decompressed: Arc::new(OnceLock::new()),
            snappy_decoder: None,
        }
    }

    fn encoding(&self) -> Encoding {
        self.compressed.encoding()
    }

    fn num_values(&self) -> u32 {
        self.compressed.num_values()
    }

    fn is_data_page(&self) -> bool {
        self.compressed.is_data_page()
    }

    fn is_dictionary_page(&self) -> bool {
        self.compressed.is_dictionary_page()
    }

    fn page(&self) -> VortexResult<&Arc<Page>> {
        if let Some(expected) = self.crc {
            let actual = crc32fast::hash(self.compressed.buffer());
            vortex_ensure!(
                actual == expected,
                "Parquet page CRC mismatch: expected {expected:#010x}, found {actual:#010x}"
            );
        }
        if self.compression == Compression::UNCOMPRESSED
            || matches!(
                self.compressed.as_ref(),
                Page::DataPageV2 {
                    is_compressed: false,
                    ..
                }
            )
        {
            return Ok(&self.compressed);
        }
        if self.decompressed.get().is_none() {
            let page = decompress_page(
                self.compressed.as_ref(),
                self.uncompressed_size,
                self.compression,
                self.snappy_decoder,
            )?;
            drop(self.decompressed.set(Arc::new(page)));
        }
        self.decompressed
            .get()
            .ok_or_else(|| vortex_err!("Parquet page decompression cache was not initialized"))
    }
}

fn decompress_page(
    compressed: &Page,
    uncompressed_size: usize,
    compression: Compression,
    snappy_decoder: Option<SnappyDecoderKind>,
) -> VortexResult<Page> {
    let offset = match compressed {
        Page::DataPageV2 {
            def_levels_byte_len,
            rep_levels_byte_len,
            ..
        } => usize::try_from(*def_levels_byte_len)?
            .checked_add(usize::try_from(*rep_levels_byte_len)?)
            .ok_or_else(|| vortex_err!("Parquet V2 level byte lengths overflow usize"))?,
        _ => 0,
    };
    vortex_ensure!(
        offset <= compressed.buffer().len() && offset <= uncompressed_size,
        "Parquet page level prefix exceeds its payload"
    );
    let mut buffer = buffer_pool::PageBuffer::new(uncompressed_size);
    let output = &mut buffer.0;
    output.extend_from_slice(&compressed.buffer()[..offset]);
    let expected_values = uncompressed_size - offset;
    if compression == Compression::SNAPPY
        && let Some(decode) = snappy_decoder
    {
        match decode {
            SnappyDecoderKind::Slice(decode) => {
                output.resize(uncompressed_size, 0);
                if expected_values != 0 {
                    decode(&compressed.buffer()[offset..], &mut output[offset..])?;
                }
            }
            SnappyDecoderKind::Append(decode) => {
                if expected_values != 0 {
                    decode(&compressed.buffer()[offset..], output, expected_values)?;
                }
            }
        }
    } else {
        let mut codec = create_codec(compression, &CodecOptions::default())
            .map_err(|error| vortex_err!("failed to construct Parquet page codec: {error}"))?
            .ok_or_else(|| vortex_err!("compressed Parquet page has no decompressor"))?;
        if expected_values != 0 {
            codec
                .decompress(
                    &compressed.buffer()[offset..],
                    output,
                    Some(expected_values),
                )
                .map_err(|error| vortex_err!("failed to decompress Parquet page: {error}"))?;
        }
    }
    vortex_ensure!(
        output.len() == uncompressed_size,
        "Parquet page decompressed to {} bytes, expected {uncompressed_size}",
        output.len()
    );
    let buf = buffer.into_bytes();
    Ok(match compressed {
        Page::DataPage {
            num_values,
            encoding,
            def_level_encoding,
            rep_level_encoding,
            statistics,
            ..
        } => Page::DataPage {
            buf,
            num_values: *num_values,
            encoding: *encoding,
            def_level_encoding: *def_level_encoding,
            rep_level_encoding: *rep_level_encoding,
            statistics: statistics.clone(),
        },
        Page::DataPageV2 {
            num_values,
            encoding,
            num_nulls,
            num_rows,
            def_levels_byte_len,
            rep_levels_byte_len,
            is_compressed,
            statistics,
            ..
        } => Page::DataPageV2 {
            buf,
            num_values: *num_values,
            encoding: *encoding,
            num_nulls: *num_nulls,
            num_rows: *num_rows,
            def_levels_byte_len: *def_levels_byte_len,
            rep_levels_byte_len: *rep_levels_byte_len,
            is_compressed: *is_compressed,
            statistics: statistics.clone(),
        },
        Page::DictionaryPage {
            num_values,
            encoding,
            is_sorted,
            ..
        } => Page::DictionaryPage {
            buf,
            num_values: *num_values,
            encoding: *encoding,
            is_sorted: *is_sorted,
        },
    })
}

/// Per-page state.
#[derive(Clone, Debug)]
pub struct ParquetPageData {
    identity: u64,
    page: LazyParquetPage,
    dictionary: Option<LazyParquetPage>,
    dictionary_values: Option<ArrayRef>,
    descriptor: ColumnDescPtr,
    options: ReaderOptions,
    metrics: Arc<ScanMetrics>,
    decoded_bytes: Arc<OnceLock<DecodedByteValues>>,
    dictionary_codes: Arc<OnceLock<ArrayRef>>,
}

impl Display for ParquetPageData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.page.encoding())
    }
}

impl ArrayHash for ParquetPageData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.identity.hash(state);
    }
}

impl ArrayEq for ParquetPageData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.identity == other.identity
    }
}

impl ParquetPageData {
    /// The Parquet value encoding used by this page.
    pub fn encoding(&self) -> Encoding {
        self.page.encoding()
    }

    /// Decompresses this page on first use and returns parquet-rs's decoder input.
    pub fn page(&self) -> VortexResult<&Arc<Page>> {
        self.page.page()
    }

    /// Borrow a null-free PLAIN INT32 payload for decoding signed 16-bit output ranges.
    ///
    /// The requested logical type must be i16. Required pages and optional pages
    /// with one canonical all-present definition-level run are supported; other pages return
    /// `None`. Payload length is checked here, and each decoded range checks its values' bounds.
    pub fn plain_i16_view(&self, dtype: &DType) -> VortexResult<Option<PlainI16Page>> {
        if self.page.encoding() != Encoding::PLAIN
            || !self.page.is_data_page()
            || self.descriptor.physical_type() != PhysicalType::INT32
            || self.descriptor.max_rep_level() != 0
            || !matches!(dtype, DType::Primitive(PType::I16, _))
        {
            return Ok(None);
        }
        let page = self.page.page()?;
        let values = match self.descriptor.max_def_level() {
            0 => value_stream_and_validity(page, 0)?.0,
            1 => match single_present_run_values(page) {
                Some(values) => values,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let length = usize::try_from(self.page.num_values())?
            .checked_mul(4)
            .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
        vortex_ensure!(length <= values.len(), "truncated PLAIN numeric payload");
        Ok(Some(PlainI16Page {
            values: values.slice(..length),
        }))
    }
}

/// A shared, null-free PLAIN INT32 payload decoded into signed 16-bit output on demand.
#[derive(Clone)]
pub struct PlainI16Page {
    values: Bytes,
}

impl PlainI16Page {
    /// Number of values in the page.
    pub fn len(&self) -> usize {
        self.values.len() / 4
    }

    /// Whether the page contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Decode a range into caller-owned slots, validating the signed 16-bit domain.
    ///
    /// The destination length must equal the range length. On success every slot is initialized.
    /// On error its contents are unspecified; values outside the requested range are not decoded.
    pub fn decode_range_into(
        &self,
        range: std::ops::Range<usize>,
        output: &mut [MaybeUninit<i16>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            range.start <= range.end && range.end <= self.len(),
            "PLAIN narrow range is out of bounds"
        );
        vortex_ensure!(
            output.len() == range.len(),
            "PLAIN narrow output length mismatch"
        );
        let (values, _) = self.values[range.start * 4..range.end * 4].as_chunks::<4>();
        #[cfg(all(
            target_arch = "aarch64",
            target_endian = "little",
            target_feature = "neon"
        ))]
        // SAFETY: compiled only for little-endian aarch64 with NEON; lengths were checked above.
        let in_range = unsafe { neon_narrow::i16_values_into(values, output) };
        #[cfg(not(all(
            target_arch = "aarch64",
            target_endian = "little",
            target_feature = "neon"
        )))]
        let in_range = {
            let mut lost_bits = 0i32;
            for (bytes, slot) in values.iter().zip(output) {
                let low = i16::from_le_bytes([bytes[0], bytes[1]]);
                lost_bits |= i32::from_le_bytes(*bytes) ^ i32::from(low);
                slot.write(low);
            }
            lost_bits == 0
        };
        vortex_ensure!(in_range, "PLAIN integer value exceeds logical i16 range");
        Ok(())
    }
}

fn validate_page(
    data: &ParquetPageData,
    dtype: &DType,
    len: usize,
    slots: &[Option<ArrayRef>],
    encoding: Encoding,
) -> VortexResult<()> {
    vortex_ensure!(
        data.page.is_data_page() || data.page.is_dictionary_page(),
        "a Parquet page array must hold a data or dictionary page"
    );
    vortex_ensure!(
        data.page.encoding() == encoding,
        "page encoding does not match its vtable"
    );
    vortex_ensure!(
        usize::try_from(data.page.num_values())? == len,
        "Parquet page value count does not match array length"
    );
    vortex_ensure!(
        slots.len() == 1,
        "Parquet page array expects one validity slot"
    );
    if let Some(validity) = &slots[0] {
        vortex_ensure!(
            dtype.is_nullable(),
            "non-nullable page has a validity array"
        );
        vortex_ensure!(validity.len() == len, "page validity length mismatch");
    }
    Ok(())
}

macro_rules! parquet_page_encoding {
    ($vtable:ident, $array:ident, $id:literal, $encoding:path) => {
        #[doc = concat!("A lazy Vortex array holding a Parquet `", stringify!($encoding), "` page.")]
        pub type $array = Array<$vtable>;

        #[doc = concat!("VTable for Parquet `", stringify!($encoding), "` pages.")]
        #[derive(Clone, Debug)]
        pub struct $vtable;

        impl VTable for $vtable {
            type TypedArrayData = ParquetPageData;
            type OperationsVTable = Self;
            type ValidityVTable = Self;

            fn id(&self) -> vortex::array::ArrayId {
                static ID: CachedId = CachedId::new($id);
                *ID
            }

            fn validate(
                &self,
                data: &Self::TypedArrayData,
                dtype: &DType,
                len: usize,
                slots: &[Option<ArrayRef>],
            ) -> VortexResult<()> {
                validate_page(data, dtype, len, slots, $encoding)
            }

            fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
                0
            }

            fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
                vortex_panic!("Parquet page array buffer index {idx} out of bounds")
            }

            fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
                None
            }

            fn with_buffers(
                &self,
                array: ArrayView<'_, Self>,
                buffers: &[BufferHandle],
            ) -> VortexResult<ArrayParts<Self>> {
                with_empty_buffers(self, array, buffers)
            }

            fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
                match idx {
                    0 => "validity".to_owned(),
                    _ => vortex_panic!("Parquet page array slot {idx} out of bounds"),
                }
            }

            fn execute(
                array: Array<Self>,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<ExecutionResult> {
                if matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_))
                    && let Some(values) = binary_dictionary_values(array.data())?
                {
                    let codes = decode_dictionary_codes(array.data(), values.len())?;
                    let dictionary = DictArray::try_new(codes, values)?.into_array();
                    return Ok(ExecutionResult::done(materialize_dictionary_bytes(
                        dictionary,
                        array.dtype(),
                        ctx,
                    )?));
                }
                Ok(ExecutionResult::done(decode_page_parts(
                    array.data(),
                    array.dtype(),
                    ctx,
                )?))
            }
        }

        impl ArrayParentReduceRule<$vtable> for $vtable {
            type Parent = AnyScalarFn;

            fn reduce_parent(
                &self,
                array: ArrayView<'_, $vtable>,
                parent: ArrayView<'_, ScalarFn>,
                child_idx: usize,
            ) -> VortexResult<Option<ArrayRef>> {
                reduce_page_parent(array, $vtable, parent.as_ref(), child_idx)
            }
        }

        impl FilterKernel for $vtable {
            fn filter(
                array: ArrayView<'_, Self>,
                mask: &Mask,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<ArrayRef>> {
                let threshold = array
                    .data()
                    .options
                    .filter_kernel_threshold(array.data().encoding());
                // Preserve shared strings, decimals and i64 values, decoded codes and small dictionaries.
                // Sparse numeric masks also share the values instead of decoding every page.
                if (matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_))
                    || array.data().dictionary_codes.get().is_some()
                    || small_dictionary(array.data())
                    || (matches!(
                        $encoding,
                        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                    ) && (mask.density() <= threshold
                        || matches!(
                            array.dtype(),
                            DType::Decimal(..) | DType::Primitive(PType::I64, _)
                        ))))
                    && let Some(dictionary) = reduce_dictionary_page(array.data(), array.dtype())?
                {
                    return Ok(Some(dictionary.filter(mask.clone())?));
                }
                if mask.density() > threshold {
                    return Ok(None);
                }
                array.data().metrics.record_decoder_filter();
                Ok(Some(filter_page_parts(
                    array.data(),
                    array.dtype(),
                    mask,
                    ctx,
                )?))
            }
        }

        impl LikeKernel for $vtable {
            fn like(
                array: ArrayView<'_, Self>,
                pattern: &ArrayRef,
                options: LikeOptions,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<ArrayRef>> {
                like_page(array.data(), array.dtype(), pattern, options)
            }
        }

        impl CompareKernel for $vtable {
            fn compare(
                lhs: ArrayView<'_, Self>,
                rhs: &ArrayRef,
                operator: CompareOperator,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<ArrayRef>> {
                compare_page(lhs.data(), lhs.dtype(), rhs, operator)
            }
        }

        impl CastKernel for $vtable {
            fn cast(
                array: ArrayView<'_, Self>,
                dtype: &DType,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<ArrayRef>> {
                cast_page(array.data(), array.dtype(), dtype, _ctx)
            }
        }

        impl ByteLengthKernel for $vtable {
            fn byte_length(
                array: ArrayView<'_, Self>,
                _ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<ArrayRef>> {
                byte_length_page(array.data(), array.dtype())
            }
        }

        impl OperationsVTable<$vtable> for $vtable {
            type ProbeState = ();
            fn scalar_at(
                array: ArrayView<'_, $vtable>,
                index: usize,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Scalar> {
                decode_page_parts(array.data(), array.dtype(), ctx)?.execute_scalar(index, ctx)
            }
        }

        impl ValidityVTable<$vtable> for $vtable {
            fn validity(array: ArrayView<'_, $vtable>) -> VortexResult<Validity> {
                if let Some(validity) = array.slots()[0].as_ref() {
                    return Ok(Validity::Array(validity.clone()));
                }
                if !array.dtype().is_nullable() {
                    return Ok(Validity::NonNullable);
                }
                let page = array.data().page.page()?;
                let (_, valid) = value_stream_and_validity(
                    page.as_ref(),
                    array.data().descriptor.max_def_level(),
                )?;
                Ok(valid.map_or(Validity::AllValid, |valid| {
                    Validity::from(BitBuffer::from(valid))
                }))
            }
        }
    };
}

parquet_page_encoding!(
    ParquetPlain,
    ParquetPlainArray,
    "vortex.parquet.plain",
    Encoding::PLAIN
);
parquet_page_encoding!(
    ParquetPlainDictionary,
    ParquetPlainDictionaryArray,
    "vortex.parquet.plain-dictionary",
    Encoding::PLAIN_DICTIONARY
);
parquet_page_encoding!(
    ParquetRle,
    ParquetRleArray,
    "vortex.parquet.rle",
    Encoding::RLE
);
parquet_page_encoding!(
    ParquetBitPacked,
    ParquetBitPackedArray,
    "vortex.parquet.bit-packed",
    Encoding::BIT_PACKED
);
parquet_page_encoding!(
    ParquetDeltaBinaryPacked,
    ParquetDeltaBinaryPackedArray,
    "vortex.parquet.delta-binary-packed",
    Encoding::DELTA_BINARY_PACKED
);
parquet_page_encoding!(
    ParquetDeltaLengthByteArray,
    ParquetDeltaLengthByteArrayArray,
    "vortex.parquet.delta-length-byte-array",
    Encoding::DELTA_LENGTH_BYTE_ARRAY
);
parquet_page_encoding!(
    ParquetDeltaByteArray,
    ParquetDeltaByteArrayArray,
    "vortex.parquet.delta-byte-array",
    Encoding::DELTA_BYTE_ARRAY
);
parquet_page_encoding!(
    ParquetRleDictionary,
    ParquetRleDictionaryArray,
    "vortex.parquet.rle-dictionary",
    Encoding::RLE_DICTIONARY
);
parquet_page_encoding!(
    ParquetByteStreamSplit,
    ParquetByteStreamSplitArray,
    "vortex.parquet.byte-stream-split",
    Encoding::BYTE_STREAM_SPLIT
);

fn new_page_array(
    page: LazyParquetPage,
    dictionary: Option<LazyParquetPage>,
    dictionary_values: Option<ArrayRef>,
    descriptor: ColumnDescPtr,
    dtype: DType,
    options: ReaderOptions,
    metrics: Arc<ScanMetrics>,
) -> VortexResult<ArrayRef> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let len = usize::try_from(page.num_values())?;
    let encoding = page.encoding();
    let mut slots = ArraySlots::new();
    // Definition levels are part of the compressed page.  Decoding them here would force outer
    // decompression merely to construct the array, unlike Vortex FlatReader.  Kernels obtain
    // validity together with the value stream after they have established that the page is needed.
    slots.push(None);
    let data = ParquetPageData {
        identity: NEXT.fetch_add(1, Ordering::Relaxed),
        page,
        dictionary,
        dictionary_values,
        descriptor,
        options,
        metrics,
        decoded_bytes: Arc::new(OnceLock::new()),
        dictionary_codes: Arc::new(OnceLock::new()),
    };

    macro_rules! build {
        ($vtable:expr) => {
            Array::try_from_parts(ArrayParts::new($vtable, dtype, len, data, slots))?.into_array()
        };
    }
    Ok(match encoding {
        Encoding::PLAIN => build!(ParquetPlain),
        Encoding::PLAIN_DICTIONARY => build!(ParquetPlainDictionary),
        Encoding::RLE => build!(ParquetRle),
        Encoding::BIT_PACKED => build!(ParquetBitPacked),
        Encoding::DELTA_BINARY_PACKED => build!(ParquetDeltaBinaryPacked),
        Encoding::DELTA_LENGTH_BYTE_ARRAY => build!(ParquetDeltaLengthByteArray),
        Encoding::DELTA_BYTE_ARRAY => build!(ParquetDeltaByteArray),
        Encoding::RLE_DICTIONARY => build!(ParquetRleDictionary),
        Encoding::BYTE_STREAM_SPLIT => build!(ParquetByteStreamSplit),
    })
}

/// A `Read` adaptor which reports how many bytes the compact Thrift page header consumed.
struct TrackedRead<R> {
    inner: R,
    bytes_read: usize,
}

impl<R: Read> Read for TrackedRead<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.bytes_read += read;
        Ok(read)
    }
}

fn wire_encoding(encoding: parquet::format::Encoding) -> VortexResult<Encoding> {
    Ok(if encoding == parquet::format::Encoding::PLAIN {
        Encoding::PLAIN
    } else if encoding == parquet::format::Encoding::PLAIN_DICTIONARY {
        Encoding::PLAIN_DICTIONARY
    } else if encoding == parquet::format::Encoding::RLE {
        Encoding::RLE
    } else if encoding == parquet::format::Encoding::BIT_PACKED {
        Encoding::BIT_PACKED
    } else if encoding == parquet::format::Encoding::DELTA_BINARY_PACKED {
        Encoding::DELTA_BINARY_PACKED
    } else if encoding == parquet::format::Encoding::DELTA_LENGTH_BYTE_ARRAY {
        Encoding::DELTA_LENGTH_BYTE_ARRAY
    } else if encoding == parquet::format::Encoding::DELTA_BYTE_ARRAY {
        Encoding::DELTA_BYTE_ARRAY
    } else if encoding == parquet::format::Encoding::RLE_DICTIONARY {
        Encoding::RLE_DICTIONARY
    } else if encoding == parquet::format::Encoding::BYTE_STREAM_SPLIT {
        Encoding::BYTE_STREAM_SPLIT
    } else {
        vortex_bail!("unsupported Parquet wire encoding {encoding:?}")
    })
}

/// Parses one header and retains its payload without applying the column codec.
fn read_compressed_page<R: ChunkReader>(
    reader: &R,
    offset: u64,
    available: u64,
    compression: Compression,
) -> VortexResult<(Option<LazyParquetPage>, u64)> {
    let mut tracked = TrackedRead {
        inner: reader
            .get_read(offset)
            .map_err(|error| vortex_err!("failed to read Parquet page header: {error}"))?,
        bytes_read: 0,
    };
    let header = {
        let mut protocol = TCompactInputProtocol::new(&mut tracked);
        PageHeader::read_from_in_protocol(&mut protocol)
            .map_err(|error| vortex_err!("failed to decode Parquet page header: {error}"))?
    };
    let header_len = u64::try_from(tracked.bytes_read)?;
    let compressed_size = u64::try_from(header.compressed_page_size)
        .map_err(|_| vortex_err!("Parquet page has a negative compressed size"))?;
    let uncompressed_size = usize::try_from(header.uncompressed_page_size)
        .map_err(|_| vortex_err!("Parquet page has a negative uncompressed size"))?;
    let consumed = header_len
        .checked_add(compressed_size)
        .ok_or_else(|| vortex_err!("Parquet page byte length overflows u64"))?;
    vortex_ensure!(
        consumed <= available,
        "Parquet page consumes {consumed} bytes but only {available} remain"
    );
    let payload_offset = offset
        .checked_add(header_len)
        .ok_or_else(|| vortex_err!("Parquet page payload offset overflows u64"))?;
    let payload = reader
        .get_bytes(payload_offset, usize::try_from(compressed_size)?)
        .map_err(|error| vortex_err!("failed to retain compressed Parquet page: {error}"))?;
    let crc = header
        .crc
        .map(|value| u32::from_le_bytes(value.to_le_bytes()));

    let page = if header.type_ == PageType::INDEX_PAGE {
        None
    } else if header.type_ == PageType::DICTIONARY_PAGE {
        let dictionary = header
            .dictionary_page_header
            .ok_or_else(|| vortex_err!("Parquet dictionary page has no dictionary header"))?;
        Some(Page::DictionaryPage {
            buf: payload,
            num_values: u32::try_from(dictionary.num_values)?,
            encoding: wire_encoding(dictionary.encoding)?,
            is_sorted: dictionary.is_sorted.unwrap_or(false),
        })
    } else if header.type_ == PageType::DATA_PAGE {
        let data = header
            .data_page_header
            .ok_or_else(|| vortex_err!("Parquet V1 data page has no data header"))?;
        Some(Page::DataPage {
            buf: payload,
            num_values: u32::try_from(data.num_values)?,
            encoding: wire_encoding(data.encoding)?,
            def_level_encoding: wire_encoding(data.definition_level_encoding)?,
            rep_level_encoding: wire_encoding(data.repetition_level_encoding)?,
            statistics: None,
        })
    } else if header.type_ == PageType::DATA_PAGE_V2 {
        let data = header
            .data_page_header_v2
            .ok_or_else(|| vortex_err!("Parquet V2 data page has no data header"))?;
        Some(Page::DataPageV2 {
            buf: payload,
            num_values: u32::try_from(data.num_values)?,
            encoding: wire_encoding(data.encoding)?,
            num_nulls: u32::try_from(data.num_nulls)?,
            num_rows: u32::try_from(data.num_rows)?,
            def_levels_byte_len: u32::try_from(data.definition_levels_byte_length)?,
            rep_levels_byte_len: u32::try_from(data.repetition_levels_byte_length)?,
            is_compressed: data.is_compressed.unwrap_or(true),
            statistics: None,
        })
    } else {
        vortex_bail!("unsupported Parquet page type {:?}", header.type_)
    };
    Ok((
        page.map(|page| LazyParquetPage::new(page, uncompressed_size, compression, crc)),
        consumed,
    ))
}

fn read_compressed_chunk<R: ChunkReader>(
    reader: &R,
    column_metadata: &ColumnChunkMetaData,
) -> VortexResult<Vec<LazyParquetPage>> {
    let (mut offset, length) = column_metadata.byte_range();
    let mut remaining = length;
    let mut pages = Vec::new();
    while remaining != 0 {
        let (page, consumed) =
            read_compressed_page(reader, offset, remaining, column_metadata.compression())?;
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| vortex_err!("Parquet page offset overflows u64"))?;
        remaining -= consumed;
        if let Some(page) = page {
            pages.push(page);
        }
    }
    Ok(pages)
}

#[allow(deprecated)]
fn read_compressed_locations<R: ChunkReader>(
    reader: &R,
    column_metadata: &ColumnChunkMetaData,
    locations: &[PageLocation],
) -> VortexResult<Vec<LazyParquetPage>> {
    let mut pages = Vec::new();
    if let Some(dictionary_offset) = column_metadata.dictionary_page_offset() {
        let start = u64::try_from(dictionary_offset)?;
        let data_start = u64::try_from(column_metadata.data_page_offset())?;
        if start < data_start {
            let (dictionary, consumed) = read_compressed_page(
                reader,
                start,
                data_start - start,
                column_metadata.compression(),
            )?;
            vortex_ensure!(
                consumed == data_start - start,
                "Parquet dictionary range contains trailing bytes"
            );
            if let Some(dictionary) = dictionary {
                pages.push(dictionary);
            }
        }
    }
    for location in locations {
        let offset = u64::try_from(location.offset)?;
        let length = u64::try_from(location.compressed_page_size)?;
        let (page, consumed) =
            read_compressed_page(reader, offset, length, column_metadata.compression())?;
        vortex_ensure!(
            consumed == length,
            "Parquet offset-index page length is {length}, header describes {consumed}"
        );
        if let Some(page) = page {
            pages.push(page);
        }
    }
    Ok(pages)
}

/// Decodes a flat physical column chunk into one lazy Vortex array per Parquet data page.
///
/// `Ok(None)` means the logical type or Dremel levels require the field-level Arrow fallback.
#[expect(clippy::too_many_arguments)]
pub fn decode_chunk(
    chunk_reader: SegmentChunkReader,
    column_metadata: &ColumnChunkMetaData,
    descriptor: &ColumnDescPtr,
    dtype: &DType,
    row_count: u64,
    options: ReaderOptions,
    metrics: Arc<ScanMetrics>,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
) -> VortexResult<Option<ArrayRef>> {
    if descriptor.max_rep_level() != 0 || descriptor.max_def_level() > 1 {
        return Ok(None);
    }
    if !direct_dtype(dtype, descriptor.physical_type()) {
        return Ok(None);
    }

    let row_count = usize::try_from(row_count)?;
    let pages = read_compressed_chunk(&chunk_reader, column_metadata)?;
    decode_page_reader(
        pages,
        descriptor,
        dtype,
        Some(row_count),
        options,
        metrics,
        dictionary_values,
    )
}

/// Decodes the indexed pages needed by a sparse flat-column selection and filters within those
/// pages through the native Parquet page arrays.
#[allow(deprecated)]
#[expect(clippy::too_many_arguments)]
pub(crate) fn decode_selected_chunk(
    chunk_reader: SegmentChunkReader,
    column_metadata: &ColumnChunkMetaData,
    descriptor: &ColumnDescPtr,
    dtype: &DType,
    row_count: u64,
    all_locations: &[PageLocation],
    selected_locations: Vec<PageLocation>,
    row_range: &std::ops::Range<usize>,
    mask: &Mask,
    options: ReaderOptions,
    metrics: Arc<ScanMetrics>,
    dictionary_values: Arc<OnceLock<ArrayRef>>,
) -> VortexResult<Option<ArrayRef>> {
    if descriptor.max_rep_level() != 0
        || descriptor.max_def_level() > 1
        || !direct_dtype(dtype, descriptor.physical_type())
    {
        return Ok(None);
    }
    let total_rows = usize::try_from(row_count)?;
    if row_range.end < row_range.start
        || row_range.end - row_range.start != mask.len()
        || row_range.end > total_rows
    {
        return Ok(None);
    }
    let pages = read_compressed_locations(&chunk_reader, column_metadata, &selected_locations)?;
    let Some(array) = decode_page_reader(
        pages,
        descriptor,
        dtype,
        None,
        options,
        metrics,
        dictionary_values,
    )?
    else {
        return Ok(None);
    };

    let mut selected = BitBufferMut::with_capacity(array.len());
    for location in &selected_locations {
        let Some(page) = all_locations
            .iter()
            .position(|candidate| candidate.offset == location.offset)
        else {
            return Ok(None);
        };
        let start = usize::try_from(location.first_row_index)?;
        let end = all_locations
            .get(page + 1)
            .map_or(Ok(total_rows), |next| usize::try_from(next.first_row_index))?;
        if end < start || end > total_rows {
            return Ok(None);
        }
        let overlap_start = start.max(row_range.start);
        let overlap_end = end.min(row_range.end);
        if overlap_start >= overlap_end {
            selected.append_n(false, end - start);
            continue;
        }
        selected.append_n(false, overlap_start - start);
        match mask.slice(overlap_start - row_range.start..overlap_end - row_range.start) {
            Mask::AllTrue(len) => selected.append_n(true, len),
            Mask::AllFalse(len) => selected.append_n(false, len),
            Mask::Values(values) => selected.append_buffer(values.bit_buffer()),
        }
        selected.append_n(false, end - overlap_end);
    }
    if selected.len() != array.len() {
        return Ok(None);
    }
    Ok(Some(filter_page_chunks(
        array,
        Mask::from_buffer(selected.freeze()),
    )?))
}

/// Distributes a selection over decoded page chunks without resolving a chunk for every row.
///
/// The generic chunked filter turns a moderately dense mask into per-row take indices. These
/// chunks are Parquet pages, so retaining per-page masks also lets their decoder filter kernels
/// skip unselected values while the pages are still encoded.
pub(crate) fn filter_page_chunks(array: ArrayRef, mask: Mask) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        array.len() == mask.len(),
        "page filter mask length mismatch"
    );
    if mask.all_true() || mask.all_false() {
        return array.filter(mask);
    }
    let Some(chunked) = array.as_opt::<Chunked>() else {
        return array.filter(mask);
    };
    let mut offset = 0;
    let mut chunks = Vec::with_capacity(chunked.nchunks());
    for chunk in chunked.chunks() {
        let end = offset + chunk.len();
        let selected = mask.slice(offset..end);
        if !selected.all_false() {
            chunks.push(chunk.filter(selected)?);
        }
        offset = end;
    }
    Ok(ChunkedArray::try_new(chunks, array.dtype().clone())?.into_array())
}

/// Decodes sparse flat-column records directly into Vortex buffers.
///
/// This is the no-offset-index counterpart of [`decode_selected_chunk`]. It cannot avoid fetching
/// the column chunk, but `ColumnReaderImpl::skip_records` can skip complete pages from their headers
/// and skip encoded runs within the boundary pages. Unlike the Arrow fallback, the selected values
/// are constructed as Vortex arrays directly.
pub(crate) fn decode_selected_records(
    chunk_reader: SegmentChunkReader,
    column_metadata: &ColumnChunkMetaData,
    dtype: &DType,
    row_count: u64,
    row_range: &std::ops::Range<usize>,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let descriptor = column_metadata.column_descr_ptr();
    if descriptor.max_rep_level() != 0
        || descriptor.max_def_level() > 1
        || !direct_dtype(dtype, descriptor.physical_type())
    {
        return Ok(None);
    }
    let row_count = usize::try_from(row_count)?;
    vortex_ensure!(
        row_range.end >= row_range.start
            && row_range.end - row_range.start == mask.len()
            && row_range.end <= row_count,
        "selected row range does not fit the Parquet column chunk"
    );
    let reader =
        SerializedPageReader::new(Arc::new(chunk_reader), column_metadata, row_count, None)
            .map_err(|error| vortex_err!("failed to open selected-record page reader: {error}"))?;

    macro_rules! selected {
        ($parquet_type:ty) => {{
            read_selected_records::<$parquet_type>(
                ColumnReaderImpl::new(Arc::clone(&descriptor), Box::new(reader)),
                &descriptor,
                row_range,
                mask,
            )?
        }};
    }

    Ok(Some(match descriptor.physical_type() {
        PhysicalType::BOOLEAN => {
            let (values, validity) = selected!(BoolType);
            let mut values = values.into_iter();
            let array: BoolArray = match validity {
                Some(validity) => validity
                    .into_iter()
                    .map(|valid| valid.then(|| values.next()).flatten())
                    .collect(),
                None => values.collect(),
            };
            array.into_array()
        }
        PhysicalType::INT32 if dtype.is_decimal() => {
            let (values, validity) = selected!(Int32Type);
            build_decimal(values, validity, dtype, ctx)?
        }
        PhysicalType::INT32 => {
            let (values, validity) = selected!(Int32Type);
            let (values, validity) = scatter(values, validity);
            logicalize(
                PrimitiveArray::new(Buffer::from(values), validity).into_array(),
                dtype,
            )?
        }
        PhysicalType::INT64 if dtype.is_decimal() => {
            let (values, validity) = selected!(Int64Type);
            build_decimal(values, validity, dtype, ctx)?
        }
        PhysicalType::INT64 => {
            let (values, validity) = selected!(Int64Type);
            let (values, validity) = scatter(values, validity);
            logicalize(
                PrimitiveArray::new(Buffer::from(values), validity).into_array(),
                dtype,
            )?
        }
        PhysicalType::FLOAT => {
            let (values, validity) = selected!(FloatType);
            let (values, validity) = scatter(values, validity);
            logicalize(
                PrimitiveArray::new(Buffer::from(values), validity).into_array(),
                dtype,
            )?
        }
        PhysicalType::DOUBLE => {
            let (values, validity) = selected!(DoubleType);
            let (values, validity) = scatter(values, validity);
            logicalize(
                PrimitiveArray::new(Buffer::from(values), validity).into_array(),
                dtype,
            )?
        }
        PhysicalType::BYTE_ARRAY => {
            let (values, validity) = selected!(ByteArrayType);
            build_bytes(
                values.iter().map(ByteArray::data),
                validity.as_deref(),
                dtype,
            )?
        }
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            let (values, validity) = selected!(FixedLenByteArrayType);
            build_fixed_bytes(values, validity, dtype)?
        }
        PhysicalType::INT96 => {
            let (values, validity) = selected!(Int96Type);
            build_int96(values, validity, dtype)?
        }
    }))
}

type SelectedRecords<T> = (Vec<T>, Option<Vec<bool>>);

fn read_selected_records<T: ParquetDataType>(
    mut reader: ColumnReaderImpl<T>,
    descriptor: &ColumnDescPtr,
    row_range: &std::ops::Range<usize>,
    mask: &Mask,
) -> VortexResult<SelectedRecords<T::T>> {
    let selected = mask.true_count();
    let mut values = Vec::with_capacity(selected);
    let mut definition_levels =
        (descriptor.max_def_level() > 0).then(|| Vec::with_capacity(selected));

    let skipped = reader
        .skip_records(row_range.start)
        .map_err(|error| vortex_err!("failed to skip leading Parquet records: {error}"))?;
    vortex_ensure!(
        skipped == row_range.start,
        "skipped {skipped} of {} leading Parquet records",
        row_range.start
    );

    let mut cursor = 0usize;
    let all_slice = [(0, mask.len())];
    let slices: &[(usize, usize)] = match mask.slices() {
        vortex::mask::AllOr::All => &all_slice,
        vortex::mask::AllOr::None => &[],
        vortex::mask::AllOr::Some(slices) => slices,
    };
    if descriptor.max_def_level() == 0 && !slices.is_empty() {
        // Calling parquet-rs once for every isolated hit is expensive for highly fragmented masks.
        // Coalesce nearby runs, decode the small intervening gaps, and discard those gap values
        // before returning. The gap limit follows the mask density: sparse masks tolerate a wider
        // gap while denser masks avoid degenerating into a full-column decode.
        let max_gap = (mask.len() / selected / 2).clamp(1, 64);
        let mut cursor = 0usize;
        let mut slice_index = 0usize;
        while slice_index < slices.len() {
            let first_slice = slice_index;
            let start = slices[slice_index].0;
            let mut end = slices[slice_index].1;
            slice_index += 1;
            while let Some(&(next_start, next_end)) = slices.get(slice_index) {
                if next_start - end > max_gap {
                    break;
                }
                end = next_end;
                slice_index += 1;
            }

            let gap = start - cursor;
            if gap != 0 {
                let skipped = reader
                    .skip_records(gap)
                    .map_err(|error| vortex_err!("failed to skip Parquet mask gap: {error}"))?;
                vortex_ensure!(skipped == gap, "skipped {skipped} of {gap} Parquet records");
            }

            let span = end - start;
            let mut decoded = Vec::with_capacity(span);
            let mut span_levels =
                (descriptor.max_def_level() > 0).then(|| Vec::with_capacity(span));
            let (records, decoded_values, levels) = reader
                .read_records(span, span_levels.as_mut(), None, &mut decoded)
                .map_err(|error| vortex_err!("failed to read selected Parquet records: {error}"))?;
            vortex_ensure!(
                records == span && levels == span,
                "read {records} records, {decoded_values} values, and {levels} levels of {span} Parquet records"
            );

            let included = &slices[first_slice..slice_index];
            let mut included_index = 0usize;
            let mut decoded = decoded.into_iter();
            for offset in 0..span {
                let row = start + offset;
                while row >= included[included_index].1 {
                    included_index += 1;
                }
                let level = span_levels
                    .as_ref()
                    .map_or_else(|| descriptor.max_def_level(), |levels| levels[offset]);
                let value = (level == descriptor.max_def_level())
                    .then(|| decoded.next())
                    .flatten();
                if row >= included[included_index].0 {
                    if let Some(levels) = definition_levels.as_mut() {
                        levels.push(level);
                    }
                    if let Some(value) = value {
                        values.push(value);
                    }
                }
            }
            vortex_ensure!(
                decoded.next().is_none(),
                "Parquet decoder returned more than {decoded_values} values for a coalesced span"
            );
            cursor = end;
        }
        let validity = definition_levels.map(|levels| {
            levels
                .into_iter()
                .map(|level| level == descriptor.max_def_level())
                .collect::<Vec<_>>()
        });
        let expected_values = validity.as_ref().map_or(selected, |validity| {
            validity.iter().filter(|valid| **valid).count()
        });
        vortex_ensure!(
            validity.as_ref().map_or(selected, Vec::len) == selected
                && values.len() == expected_values,
            "decoded the wrong number of selected Parquet records or values"
        );
        return Ok((values, validity));
    }

    for &(start, end) in slices {
        let gap = start.saturating_sub(cursor);
        if gap != 0 {
            let skipped = reader
                .skip_records(gap)
                .map_err(|error| vortex_err!("failed to skip Parquet mask gap: {error}"))?;
            vortex_ensure!(skipped == gap, "skipped {skipped} of {gap} Parquet records");
        }
        let wanted = end - start;
        let (records, _values, levels) = reader
            .read_records(wanted, definition_levels.as_mut(), None, &mut values)
            .map_err(|error| vortex_err!("failed to read selected Parquet records: {error}"))?;
        vortex_ensure!(
            records == wanted && levels == wanted,
            "read {records} records and {levels} levels of {wanted} selected Parquet records"
        );
        cursor = end;
    }

    let validity = definition_levels.map(|levels| {
        levels
            .into_iter()
            .map(|level| level == descriptor.max_def_level())
            .collect::<Vec<_>>()
    });
    Ok((values, validity))
}

fn decode_page_reader(
    wire_pages: Vec<LazyParquetPage>,
    descriptor: &ColumnDescPtr,
    dtype: &DType,
    expected_rows: Option<usize>,
    options: ReaderOptions,
    metrics: Arc<ScanMetrics>,
    dictionary_values_cache: Arc<OnceLock<ArrayRef>>,
) -> VortexResult<Option<ArrayRef>> {
    let mut dictionary: Option<LazyParquetPage> = None;
    let mut data_pages = Vec::new();
    let mut decoded_rows = 0usize;
    for mut page in wire_pages {
        page.snappy_decoder = options.snappy_decoder();
        if page.is_dictionary_page() {
            if dictionary.replace(page).is_some() {
                return Ok(None);
            }
            continue;
        }
        if matches!(
            page.encoding(),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        ) && dictionary.is_none()
            && dictionary_values_cache.get().is_none()
        {
            return Ok(None);
        }
        decoded_rows = decoded_rows.saturating_add(usize::try_from(page.num_values())?);
        data_pages.push(page);
    }

    if expected_rows.is_some_and(|row_count| decoded_rows != row_count) {
        return Ok(None);
    }
    let uses_dictionary = data_pages.iter().any(|page| {
        matches!(
            page.encoding(),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        )
    });
    let dictionary_values = if uses_dictionary {
        if let Some(values) = dictionary_values_cache.get() {
            Some(values.clone())
        } else {
            dictionary
                .as_ref()
                .map(|dictionary| {
                    new_page_array(
                        dictionary.clone(),
                        None,
                        None,
                        Arc::clone(descriptor),
                        dtype.as_nonnullable(),
                        options,
                        Arc::clone(&metrics),
                    )
                })
                .transpose()?
                .map(|values| dictionary_values_cache.get_or_init(|| values).clone())
        }
    } else {
        None
    };

    let mut pages = data_pages
        .into_iter()
        .map(|page| {
            new_page_array(
                page,
                dictionary.clone(),
                dictionary_values.clone(),
                Arc::clone(descriptor),
                dtype.clone(),
                options,
                Arc::clone(&metrics),
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(Some(match pages.len() {
        0 => builder_with_capacity(dtype, 0).finish(),
        1 => pages.swap_remove(0),
        _ => ChunkedArray::try_new(pages, dtype.clone())?.into_array(),
    }))
}

pub(crate) fn direct_dtype(dtype: &DType, physical: PhysicalType) -> bool {
    if matches!(
        (dtype, physical),
        (DType::Bool(_), PhysicalType::BOOLEAN)
            | (DType::Primitive(PType::I32, _), PhysicalType::INT32)
            | (DType::Primitive(PType::I64, _), PhysicalType::INT64)
            | (DType::Primitive(PType::F32, _), PhysicalType::FLOAT)
            | (DType::Primitive(PType::F64, _), PhysicalType::DOUBLE)
            | (DType::Utf8(_), PhysicalType::BYTE_ARRAY)
            | (DType::Binary(_), PhysicalType::BYTE_ARRAY)
    ) {
        return true;
    }

    match (dtype, physical) {
        // These are value-preserving numeric adaptations from Parquet's wider physical integer.
        (
            DType::Primitive(PType::I8 | PType::I16 | PType::U8 | PType::U16, _),
            PhysicalType::INT32,
        ) => true,
        (DType::Primitive(PType::U32, _), PhysicalType::INT32)
        | (DType::Primitive(PType::U64, _), PhysicalType::INT64) => true,
        (DType::Decimal(..), PhysicalType::INT32 | PhysicalType::INT64) => true,
        (DType::Binary(_), PhysicalType::FIXED_LEN_BYTE_ARRAY) => true,
        (DType::Extension(ext), PhysicalType::INT96) => matches!(
            ext.metadata_opt::<AnyTemporal>(),
            Some(TemporalMetadata::Timestamp(..))
        ),
        // Temporal extension arrays store the same integer found on the Parquet wire. Constructing
        // the extension is metadata-only, not a numerical cast.
        (DType::Extension(ext), PhysicalType::INT32) => {
            matches!(ext.storage_dtype(), DType::Primitive(PType::I32, _))
        }
        (DType::Extension(ext), PhysicalType::INT64) => {
            matches!(ext.storage_dtype(), DType::Primitive(PType::I64, _))
        }
        _ => false,
    }
}

/// Small shared value tables are cheap to export and amortize decoding their u32 codes;
/// larger dictionaries use the separate dtype- and mask-specific preservation policies.
fn small_dictionary(data: &ParquetPageData) -> bool {
    data.dictionary_values.as_ref().is_some_and(|values| {
        !values.is_empty()
            && values.len() <= 256
            && usize::try_from(data.page.num_values()).is_ok_and(|rows| values.len() * 8 <= rows)
    })
}

fn decode_page_parts(
    data: &ParquetPageData,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Preserve shared dictionary values for strings, decimals and i64 during export. Plain pages
    // cannot reduce to a dictionary, including a plain fallback after a dictionary page.
    if matches!(
        data.page.encoding(),
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    ) && (matches!(
        dtype,
        DType::Utf8(_) | DType::Binary(_) | DType::Decimal(..) | DType::Primitive(PType::I64, _)
    ) || data.dictionary_codes.get().is_some()
        || small_dictionary(data))
        && data
            .dictionary_values
            .as_ref()
            .is_some_and(|values| values.dtype().eq_ignore_nullability(dtype))
        && let Some(dictionary) = reduce_dictionary_page(data, dtype)?
    {
        return Ok(dictionary);
    }
    if data.descriptor.physical_type() == PhysicalType::BYTE_ARRAY
        && let Some(decoded) = data.decoded_bytes.get()
    {
        return build_page_bytes(data, &decoded.values, decoded.valid.as_deref(), dtype, ctx);
    }
    let page = data.page.page()?;
    // Narrow PLAIN integers preserve the requested nullability themselves. A single all-present
    // definition-level run needs no expanded validity buffer or subsequent population count.
    let narrow_values = (data.page.encoding() == Encoding::PLAIN
        && matches!(
            dtype,
            DType::Primitive(PType::I8 | PType::I16 | PType::U8 | PType::U16, _)
        )
        && data.descriptor.max_def_level() == 1)
        .then(|| single_present_run_values(page))
        .flatten();
    let (values, valid) = match narrow_values {
        Some(values) => (values, None),
        None => value_stream_and_validity(page, data.descriptor.max_def_level())?,
    };
    let present = valid.as_ref().map_or_else(
        || usize::try_from(data.page.num_values()),
        |v| Ok(v.iter().filter(|x| **x).count()),
    )?;
    match data.descriptor.physical_type() {
        PhysicalType::BOOLEAN => decode_bool(data, values, valid, present),
        PhysicalType::INT32 if dtype.is_decimal() => {
            decode_decimal_primitive::<Int32Type, i32>(data, values, valid, present, dtype, ctx)
        }
        PhysicalType::INT32 => {
            if let Some(array) =
                decode_plain_narrow(data, &values, valid.as_deref(), present, dtype)?
            {
                return Ok(array);
            }
            decode_primitive::<Int32Type, i32>(data, values, valid, present, dtype)
        }
        PhysicalType::INT64 if dtype.is_decimal() => {
            decode_decimal_primitive::<Int64Type, i64>(data, values, valid, present, dtype, ctx)
        }
        PhysicalType::INT64 => {
            decode_primitive::<Int64Type, i64>(data, values, valid, present, dtype)
        }
        PhysicalType::FLOAT => {
            decode_primitive::<FloatType, f32>(data, values, valid, present, dtype)
        }
        PhysicalType::DOUBLE => {
            decode_primitive::<DoubleType, f64>(data, values, valid, present, dtype)
        }
        PhysicalType::BYTE_ARRAY => decode_bytes(data, values, valid, present, dtype, ctx),
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            decode_fixed_bytes(data, values, valid, present, dtype)
        }
        PhysicalType::INT96 => decode_int96(data, values, valid, present, dtype),
    }
}

/// Casts directly from the Parquet physical representation when the target is one this reader can
/// decode natively. In particular, Vortex deliberately has no generic Binary-to-Utf8 cast because
/// it cannot assume arbitrary bytes are UTF-8; here the cast is explicit and the UTF-8 builder
/// validates every decoded dictionary value.
fn cast_page(
    data: &ParquetPageData,
    source_dtype: &DType,
    target_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if source_dtype == target_dtype || !direct_dtype(target_dtype, data.descriptor.physical_type())
    {
        return Ok(None);
    }
    Ok(Some(decode_page_parts(data, target_dtype, ctx)?))
}

#[derive(Debug)]
enum SimpleLike {
    Equal(Vec<u8>),
    StartsWith(Vec<u8>),
    EndsWith(Vec<u8>),
    // `memmem::find` constructs a two-way/SIMD searcher on every call. A Parquet page invokes this
    // once per value, so compiling the finder with the SQL pattern is material: ClickBench's
    // `%google%` predicate otherwise spends a visible fraction of the scan rebuilding the identical
    // searcher hundreds of thousands of times.
    Contains(memchr::memmem::Finder<'static>),
    Always,
}

impl SimpleLike {
    fn compile(pattern: &str) -> Option<Self> {
        #[derive(Debug)]
        enum Token {
            Literal(Vec<u8>),
            Percent,
        }

        let mut tokens = Vec::new();
        let mut literal = Vec::new();
        let mut escaped = false;
        for byte in pattern.bytes() {
            if escaped {
                literal.push(byte);
                escaped = false;
                continue;
            }
            match byte {
                b'\\' => escaped = true,
                // `_` needs UTF-8 character semantics. Decline it, and all multi-component patterns,
                // to Vortex's complete LIKE implementation below this fast path.
                b'_' => return None,
                b'%' => {
                    if !literal.is_empty() {
                        tokens.push(Token::Literal(std::mem::take(&mut literal)));
                    }
                    if !matches!(tokens.last(), Some(Token::Percent)) {
                        tokens.push(Token::Percent);
                    }
                }
                _ => literal.push(byte),
            }
        }
        if escaped {
            return None;
        }
        if !literal.is_empty() {
            tokens.push(Token::Literal(literal));
        }

        match tokens.as_slice() {
            [] | [Token::Literal(_)] if pattern.is_empty() => Some(Self::Equal(Vec::new())),
            [Token::Literal(value)] => Some(Self::Equal(value.clone())),
            [Token::Literal(value), Token::Percent] => Some(Self::StartsWith(value.clone())),
            [Token::Percent, Token::Literal(value)] => Some(Self::EndsWith(value.clone())),
            [Token::Percent, Token::Literal(value), Token::Percent] => Some(Self::Contains(
                memchr::memmem::Finder::new(value).into_owned(),
            )),
            [Token::Percent] => Some(Self::Always),
            _ => None,
        }
    }

    #[inline]
    fn matches(&self, value: &[u8]) -> bool {
        match self {
            Self::Equal(pattern) => value == pattern,
            Self::StartsWith(pattern) => value.starts_with(pattern),
            Self::EndsWith(pattern) => value.ends_with(pattern),
            Self::Contains(finder) => finder.find(value).is_some(),
            Self::Always => true,
        }
    }
}

fn decoded_byte_predicate(
    data: &ParquetPageData,
    dtype: &DType,
    other_nullability: vortex::dtype::Nullability,
    mut predicate: impl FnMut(&[u8]) -> bool,
) -> VortexResult<ArrayRef> {
    let decoded = decoded_byte_values(data)?;
    let valid = decoded.valid.as_deref();

    let logical_len = valid.map_or(decoded.values.len(), <[bool]>::len);
    let mut bits = BitBufferMut::with_capacity(logical_len);
    let mut values = decoded.values.iter();
    match valid {
        Some(valid) if !valid.iter().all(|is_valid| *is_valid) => {
            for is_valid in valid {
                bits.append(if *is_valid {
                    predicate(
                        values.next().ok_or_else(|| {
                            vortex_err!("Parquet validity exceeds decoded values")
                        })?,
                    )
                } else {
                    false
                });
            }
        }
        _ => {
            for value in values.by_ref() {
                bits.append(predicate(value));
            }
        }
    }
    vortex_ensure!(
        values.next().is_none(),
        "Parquet predicate left decoded values unconsumed"
    );

    let validity = match valid {
        Some(valid) if valid.iter().all(|is_valid| *is_valid) => Validity::AllValid,
        Some(valid) => Validity::from(BitBuffer::from(valid)),
        None => Validity::from(dtype.nullability()),
    };
    Ok(BoolArray::new(bits.freeze(), validity.union_nullability(other_nullability)).into_array())
}

fn decoded_byte_values(data: &ParquetPageData) -> VortexResult<&DecodedByteValues> {
    if data.decoded_bytes.get().is_none() {
        let page = data.page.page()?;
        let (values, valid) = value_stream_and_validity(page, data.descriptor.max_def_level())?;
        let present = valid.as_ref().map_or_else(
            || usize::try_from(data.page.num_values()),
            |bits| Ok(bits.iter().filter(|bit| **bit).count()),
        )?;
        let decoded = decode_byte_values(data, values, present)?;
        // Another execution may win this race. Both results refer to the same immutable page, so
        // retaining either decode is correct and lets a later projection reuse it.
        drop(data.decoded_bytes.set(DecodedByteValues {
            values: decoded,
            valid: valid.map(Into::into),
        }));
    }
    let decoded = data
        .decoded_bytes
        .get()
        .ok_or_else(|| vortex_err!("Parquet byte decode cache was not initialized"))?;
    Ok(decoded)
}

fn byte_length_page(data: &ParquetPageData, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    if !matches!(dtype, DType::Utf8(_) | DType::Binary(_)) {
        return Ok(None);
    }
    let (lengths, validity) = match data.descriptor.physical_type() {
        PhysicalType::BYTE_ARRAY => {
            let decoded = decoded_byte_values(data)?;
            let lengths = decoded
                .values
                .iter()
                .map(|value| u64::try_from(value.len()))
                .collect::<Result<Vec<_>, _>>()?;
            match decoded.valid.as_deref() {
                Some(valid) if !valid.iter().all(|is_valid| *is_valid) => {
                    scatter(lengths, Some(valid.to_vec()))
                }
                Some(_) => (lengths, Validity::AllValid),
                None => (lengths, Validity::from(dtype.nullability())),
            }
        }
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            let page = data.page.page()?;
            let (_, valid) = value_stream_and_validity(page, data.descriptor.max_def_level())?;
            let present = valid.as_ref().map_or_else(
                || usize::try_from(data.page.num_values()),
                |bits| Ok(bits.iter().filter(|bit| **bit).count()),
            )?;
            let width = u64::try_from(data.descriptor.type_length())?;
            scatter(vec![width; present], valid)
        }
        _ => return Ok(None),
    };
    Ok(Some(
        PrimitiveArray::new(Buffer::from(lengths), validity).into_array(),
    ))
}

/// Search PLAIN storage in large runs without constructing a view for every value. Length
/// prefixes and adjacent strings may contain candidate bytes, so only matches wholly inside
/// one payload count. Even after the last match, parse every value to retain bounds validation.
fn plain_contains_page(
    data: &ParquetPageData,
    dtype: &DType,
    finder: &memchr::memmem::Finder<'_>,
    options: LikeOptions,
    pattern_nullability: vortex::dtype::Nullability,
) -> VortexResult<ArrayRef> {
    let page = data.page.page()?;
    let (values, valid) = value_stream_and_validity(page, data.descriptor.max_def_level())?;
    let count = usize::try_from(data.page.num_values())?;
    let needle_len = finder.needle().len();
    let mut next_match = finder.find(&values);
    let mut cursor = 0;
    let mut bits = BitBufferMut::with_capacity(count);
    for row in 0..count {
        if valid.as_ref().is_some_and(|valid| !valid[row]) {
            bits.append(false);
            continue;
        }
        let (length, rest) = values[cursor..]
            .split_first_chunk::<4>()
            .ok_or_else(|| vortex_err!("truncated PLAIN byte-array length"))?;
        let length = usize::try_from(u32::from_le_bytes(*length))?;
        vortex_ensure!(length <= rest.len(), "truncated PLAIN byte-array payload");
        let start = cursor + 4;
        let end = start + length;
        while let Some(position) = next_match
            && position < start
        {
            // An invalid candidate can overlap a valid payload match (notably with NULs).
            let resume = position + 1;
            next_match = finder.find(&values[resume..]).map(|offset| resume + offset);
        }
        let found =
            next_match.is_some_and(|position| position <= end && needle_len <= end - position);
        bits.append(found != options.negated);
        if found {
            // One hit is sufficient for this row. Resume at its boundary, including the next
            // length prefix so candidate offsets remain relative to the original page buffer.
            next_match = finder.find(&values[end..]).map(|offset| end + offset);
        }
        cursor = end;
    }
    let validity = match valid {
        Some(valid) if valid.iter().all(|valid| *valid) => Validity::AllValid,
        Some(valid) => Validity::from(BitBuffer::from(valid.as_slice())),
        None => Validity::from(dtype.nullability()),
    };
    Ok(BoolArray::new(
        bits.freeze(),
        validity.union_nullability(pattern_nullability),
    )
    .into_array())
}

fn like_page(
    data: &ParquetPageData,
    dtype: &DType,
    pattern: &ArrayRef,
    options: LikeOptions,
) -> VortexResult<Option<ArrayRef>> {
    if data.descriptor.physical_type() != PhysicalType::BYTE_ARRAY
        || !dtype.is_utf8()
        || options.case_insensitive
    {
        return Ok(None);
    }
    let Some(pattern_scalar) = pattern.as_constant() else {
        return Ok(None);
    };
    let pattern_nullability = pattern_scalar.dtype().nullability();
    let Some(pattern) = pattern_scalar.as_utf8_opt().and_then(|value| value.value()) else {
        return Ok(None);
    };
    let Some(compiled) = SimpleLike::compile(pattern.as_str()) else {
        return Ok(None);
    };
    let result = if data.page.encoding() == Encoding::PLAIN
        && let SimpleLike::Contains(finder) = &compiled
    {
        plain_contains_page(data, dtype, finder, options, pattern_nullability)?
    } else {
        decoded_byte_predicate(data, dtype, pattern_nullability, |value| {
            compiled.matches(value) != options.negated
        })?
    };
    data.metrics.record_fused_like_predicate();
    Ok(Some(result))
}

fn compare_page(
    data: &ParquetPageData,
    dtype: &DType,
    rhs: &ArrayRef,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    let Some(rhs) = rhs.as_constant() else {
        return Ok(None);
    };
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    if let Some(result) = compare_plain_narrow(data, dtype, rhs, operator)? {
        return Ok(Some(result));
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    if let Some(result) = compare_plain_i64(data, dtype, rhs, operator)? {
        return Ok(Some(result));
    }
    if data.descriptor.physical_type() != PhysicalType::BYTE_ARRAY
        || !matches!(dtype, DType::Utf8(_) | DType::Binary(_))
    {
        return Ok(None);
    }
    let bytes = match rhs.dtype() {
        DType::Utf8(_) => rhs
            .as_utf8_opt()
            .and_then(|value| value.value())
            .map(|value| value.as_bytes()),
        DType::Binary(_) => rhs
            .as_binary_opt()
            .and_then(|value| value.value())
            .map(|value| value.as_slice()),
        _ => None,
    };
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let predicate = match operator {
        CompareOperator::Eq => |ordering: std::cmp::Ordering| ordering.is_eq(),
        CompareOperator::NotEq => |ordering: std::cmp::Ordering| ordering.is_ne(),
        CompareOperator::Gt => |ordering: std::cmp::Ordering| ordering.is_gt(),
        CompareOperator::Gte => |ordering: std::cmp::Ordering| ordering.is_ge(),
        CompareOperator::Lt => |ordering: std::cmp::Ordering| ordering.is_lt(),
        CompareOperator::Lte => |ordering: std::cmp::Ordering| ordering.is_le(),
    };
    let result = decoded_byte_predicate(data, dtype, rhs.dtype().nullability(), |value| {
        predicate(value.cmp(bytes))
    })?;
    data.metrics.record_fused_compare_predicate();
    Ok(Some(result))
}

/// Compare null-free PLAIN i64 bytes directly, without an aligned logical-value copy.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
fn compare_plain_i64(
    data: &ParquetPageData,
    dtype: &DType,
    rhs: &Scalar,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    if data.page.encoding() != Encoding::PLAIN
        || data.descriptor.physical_type() != PhysicalType::INT64
        || data.descriptor.max_rep_level() != 0
        || !matches!(dtype, DType::Primitive(PType::I64, _))
        || !dtype.eq_ignore_nullability(rhs.dtype())
        || rhs.is_null()
    {
        return Ok(None);
    }
    let constant = i64::try_from(rhs)?;
    let page = data.page.page()?;
    let values = match data.descriptor.max_def_level() {
        0 => value_stream_and_validity(page, 0)?.0,
        1 => match single_present_run_values(page) {
            Some(values) => values,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    let len = usize::try_from(data.page.num_values())?;
    let byte_len = len
        .checked_mul(8)
        .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
    vortex_ensure!(byte_len <= values.len(), "truncated PLAIN numeric payload");
    let (values, _) = values[..byte_len].as_chunks::<8>();
    // SAFETY: compiled only for little-endian aarch64 with NEON; the full payload is checked.
    let bits = unsafe { neon_compare::compare_i64(values, constant, operator) };
    let validity = Validity::from(dtype.nullability()).union_nullability(rhs.dtype().nullability());
    data.metrics.record_fused_compare_predicate();
    Ok(Some(BoolArray::new(bits, validity).into_array()))
}

/// Compare a null-free PLAIN narrow integer page without allocating its logical value buffer.
/// Validate every physical value before exposing the selection mask.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
fn compare_plain_narrow(
    data: &ParquetPageData,
    dtype: &DType,
    rhs: &Scalar,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    if data.page.encoding() != Encoding::PLAIN
        || data.descriptor.physical_type() != PhysicalType::INT32
        || !dtype.eq_ignore_nullability(rhs.dtype())
        || rhs.is_null()
    {
        return Ok(None);
    }
    macro_rules! bounds {
        ($native:ty) => {
            (
                i32::from(<$native>::MIN),
                i32::from(<$native>::MAX),
                i32::from(<$native>::try_from(rhs)?),
            )
        };
    }
    let (low, high, constant) = match dtype {
        DType::Primitive(PType::I8, _) => bounds!(i8),
        DType::Primitive(PType::I16, _) => bounds!(i16),
        DType::Primitive(PType::U8, _) => bounds!(u8),
        DType::Primitive(PType::U16, _) => bounds!(u16),
        _ => return Ok(None),
    };
    let page = data.page.page()?;
    let present_run = (data.descriptor.max_def_level() == 1)
        .then(|| single_present_run_values(page))
        .flatten();
    let (values, valid) = match present_run {
        Some(values) => (values, None),
        None => value_stream_and_validity(page, data.descriptor.max_def_level())?,
    };
    if valid
        .as_ref()
        .is_some_and(|valid| valid.iter().any(|valid| !valid))
    {
        return Ok(None);
    }
    let len = usize::try_from(data.page.num_values())?;
    let byte_len = len
        .checked_mul(4)
        .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
    vortex_ensure!(byte_len <= values.len(), "truncated PLAIN numeric payload");
    let (values, _) = values[..byte_len].as_chunks::<4>();
    // SAFETY: this function is compiled only when little-endian aarch64 NEON is enabled.
    let (bits, min, max) = unsafe { neon_compare::compare(values, constant, operator) };
    vortex_ensure!(
        min >= low && max <= high,
        "PLAIN integer value exceeds logical {dtype} range"
    );
    let validity = Validity::from(dtype.nullability()).union_nullability(rhs.dtype().nullability());
    data.metrics.record_fused_compare_predicate();
    Ok(Some(BoolArray::new(bits, validity).into_array()))
}

/// Registers reduction and execution kernels for every Parquet page encoding in `session`.
///
/// Registration is idempotent for a session. A `ParquetFormat` may open thousands of files against
/// one session, and appending the same kernel on every open would turn kernel dispatch into a linear
/// walk through the file count.
pub(crate) fn register_kernels(session: &VortexSession) {
    session.get::<ParquetKernelRegistration>().0.call_once(|| {
        dictionary_set::register(session);
        register_filter_kernels(session);
        register_like_kernels(session);
        register_compare_kernels(session);
        register_cast_kernels(session);
        register_byte_length_kernels(session);
    });
}

#[derive(Clone, Debug, Default)]
struct ParquetKernelRegistration(Arc<Once>);

impl SessionVar for ParquetKernelRegistration {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn register_byte_length_kernels(session: &VortexSession) {
    macro_rules! register {
        ($vtable:expr) => {{
            let kernels = session.kernels();
            if !kernels.has_execute_parent(ByteLength.id(), $vtable.id()) {
                kernels.register_execute_parent_kernel(
                    ByteLength.id(),
                    $vtable,
                    ByteLengthExecuteAdaptor($vtable),
                );
            }
        }};
    }
    register!(ParquetPlain);
    register!(ParquetPlainDictionary);
    register!(ParquetRle);
    register!(ParquetBitPacked);
    register!(ParquetDeltaBinaryPacked);
    register!(ParquetDeltaLengthByteArray);
    register!(ParquetDeltaByteArray);
    register!(ParquetRleDictionary);
    register!(ParquetByteStreamSplit);
}

fn register_filter_kernels(session: &VortexSession) {
    macro_rules! register {
        ($vtable:expr) => {{
            let kernels = session.kernels();
            if !kernels.has_execute_parent(Filter.id(), $vtable.id()) {
                kernels.register_execute_parent_kernel(
                    Filter.id(),
                    $vtable,
                    FilterExecuteAdaptor($vtable),
                );
            }
        }};
    }
    register!(ParquetPlain);
    register!(ParquetPlainDictionary);
    register!(ParquetRle);
    register!(ParquetBitPacked);
    register!(ParquetDeltaBinaryPacked);
    register!(ParquetDeltaLengthByteArray);
    register!(ParquetDeltaByteArray);
    register!(ParquetRleDictionary);
    register!(ParquetByteStreamSplit);
}

fn register_like_kernels(session: &VortexSession) {
    macro_rules! register {
        ($vtable:expr) => {{
            let kernels = session.kernels();
            if !kernels.has_execute_parent(Like.id(), $vtable.id()) {
                kernels.register_execute_parent_kernel(
                    Like.id(),
                    $vtable,
                    LikeExecuteAdaptor($vtable),
                );
            }
        }};
    }
    register!(ParquetPlain);
    register!(ParquetPlainDictionary);
    register!(ParquetRle);
    register!(ParquetBitPacked);
    register!(ParquetDeltaBinaryPacked);
    register!(ParquetDeltaLengthByteArray);
    register!(ParquetDeltaByteArray);
    register!(ParquetRleDictionary);
    register!(ParquetByteStreamSplit);
}

fn register_compare_kernels(session: &VortexSession) {
    macro_rules! register {
        ($vtable:expr) => {{
            let kernels = session.kernels();
            if !kernels.has_execute_parent(Compare.id(), $vtable.id()) {
                kernels.register_execute_parent_kernel(
                    Compare.id(),
                    $vtable,
                    CompareExecuteAdaptor($vtable),
                );
            }
        }};
    }
    register!(ParquetPlain);
    register!(ParquetPlainDictionary);
    register!(ParquetRle);
    register!(ParquetBitPacked);
    register!(ParquetDeltaBinaryPacked);
    register!(ParquetDeltaLengthByteArray);
    register!(ParquetDeltaByteArray);
    register!(ParquetRleDictionary);
    register!(ParquetByteStreamSplit);
}

fn register_cast_kernels(session: &VortexSession) {
    macro_rules! register {
        ($vtable:expr) => {{
            let kernels = session.kernels();
            kernels.register_reduce_parent_rule(AnyScalarFn, $vtable, const { &$vtable });
            if !kernels.has_execute_parent(Cast.id(), $vtable.id()) {
                kernels.register_execute_parent_kernel(
                    Cast.id(),
                    $vtable,
                    CastExecuteAdaptor($vtable),
                );
            }
        }};
    }
    register!(ParquetPlain);
    register!(ParquetPlainDictionary);
    register!(ParquetRle);
    register!(ParquetBitPacked);
    register!(ParquetDeltaBinaryPacked);
    register!(ParquetDeltaLengthByteArray);
    register!(ParquetDeltaByteArray);
    register!(ParquetRleDictionary);
    register!(ParquetByteStreamSplit);
}

/// Filters a page while its Parquet decoder is live, skipping encoded values for false mask runs.
fn filter_page_parts(
    data: &ParquetPageData,
    dtype: &DType,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let page = data.page.page()?;
    let (values, valid) = value_stream_and_validity(page, data.descriptor.max_def_level())?;
    vortex_ensure!(
        mask.len() == usize::try_from(data.page.num_values())?,
        "filter mask length does not match Parquet page"
    );
    let present = valid.as_ref().map_or_else(
        || usize::try_from(data.page.num_values()),
        |bits| Ok(bits.iter().filter(|bit| **bit).count()),
    )?;

    match data.descriptor.physical_type() {
        PhysicalType::BOOLEAN => {
            let mut decoder = make_decoder::<BoolType>(data, values, present)?;
            let (decoded, selected_valid) =
                decode_masked::<BoolType, bool>(decoder.as_mut(), valid.as_deref(), mask)?;
            let mut next = decoded.into_iter();
            let result: BoolArray = match selected_valid {
                Some(valid) => valid
                    .into_iter()
                    .map(|is_valid| is_valid.then(|| next.next()).flatten())
                    .collect(),
                None => next.collect(),
            };
            Ok(result.into_array())
        }
        PhysicalType::INT32 if dtype.is_decimal() => {
            filter_decimal::<Int32Type, i32>(data, values, valid, present, mask, dtype, ctx)
        }
        PhysicalType::INT32 => {
            filter_primitive::<Int32Type, i32>(data, values, valid, present, mask, dtype)
        }
        PhysicalType::INT64 if dtype.is_decimal() => {
            filter_decimal::<Int64Type, i64>(data, values, valid, present, mask, dtype, ctx)
        }
        PhysicalType::INT64 => {
            filter_primitive::<Int64Type, i64>(data, values, valid, present, mask, dtype)
        }
        PhysicalType::FLOAT => {
            filter_primitive::<FloatType, f32>(data, values, valid, present, mask, dtype)
        }
        PhysicalType::DOUBLE => {
            filter_primitive::<DoubleType, f64>(data, values, valid, present, mask, dtype)
        }
        PhysicalType::BYTE_ARRAY => {
            let mut decoder = make_decoder::<ByteArrayType>(data, values, present)?;
            let (decoded, selected_valid) = decode_masked::<ByteArrayType, ByteArray>(
                decoder.as_mut(),
                valid.as_deref(),
                mask,
            )?;
            build_bytes(
                decoded.iter().map(ByteArray::data),
                selected_valid.as_deref(),
                dtype,
            )
        }
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            let mut decoder = make_decoder::<FixedLenByteArrayType>(data, values, present)?;
            let (decoded, selected_valid) = decode_masked::<
                FixedLenByteArrayType,
                FixedLenByteArray,
            >(decoder.as_mut(), valid.as_deref(), mask)?;
            build_fixed_bytes(decoded, selected_valid, dtype)
        }
        PhysicalType::INT96 => {
            let mut decoder = make_decoder::<Int96Type>(data, values, present)?;
            let (decoded, selected_valid) =
                decode_masked::<Int96Type, Int96>(decoder.as_mut(), valid.as_deref(), mask)?;
            build_int96(decoded, selected_valid, dtype)
        }
    }
}

fn filter_primitive<T, N>(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    mask: &Mask,
    dtype: &DType,
) -> VortexResult<ArrayRef>
where
    T: ParquetDataType<T = N>,
    N: NativePType + Default,
{
    if let Some(values) = filter_plain_numeric::<N>(data, &values, valid.as_deref(), present, mask)?
    {
        return logicalize(
            PrimitiveArray::new(values, Validity::from(dtype.nullability())).into_array(),
            dtype,
        );
    }
    let mut decoder = make_decoder::<T>(data, values, present)?;
    let (decoded, selected_valid) =
        decode_masked::<T, N>(decoder.as_mut(), valid.as_deref(), mask)?;
    let (values, validity) = scatter(decoded, selected_valid);
    logicalize(
        PrimitiveArray::new(Buffer::from(values), validity).into_array(),
        dtype,
    )
}

fn filter_decimal<T, N>(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    mask: &Mask,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: ParquetDataType<T = N>,
    N: NativePType + NativeDecimalType + Default,
{
    if let Some(values) = filter_plain_numeric::<N>(data, &values, valid.as_deref(), present, mask)?
    {
        let DType::Decimal(decimal_dtype, _) = dtype else {
            vortex_bail!("Parquet decimal decoder received non-decimal dtype {dtype}")
        };
        return decimal_array(
            values,
            *decimal_dtype,
            Validity::from(dtype.nullability()),
            ctx,
        );
    }
    let mut decoder = make_decoder::<T>(data, values, present)?;
    let (decoded, selected_valid) =
        decode_masked::<T, N>(decoder.as_mut(), valid.as_deref(), mask)?;
    build_decimal(decoded, selected_valid, dtype, ctx)
}

/// PLAIN numeric offsets are known without advancing a decoder. Read just the selected values,
/// including unaligned payloads, after validating the complete payload (also its unselected tail).
fn filter_plain_numeric<N: NativePType>(
    data: &ParquetPageData,
    values: &[u8],
    valid: Option<&[bool]>,
    present: usize,
    mask: &Mask,
) -> VortexResult<Option<Buffer<N>>> {
    if data.page.encoding() != Encoding::PLAIN || valid.is_some_and(|valid| valid.len() != present)
    {
        return Ok(None);
    }
    let width = size_of::<N>();
    let byte_len = present
        .checked_mul(width)
        .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
    vortex_ensure!(byte_len <= values.len(), "truncated PLAIN numeric payload");
    // Result collection loses the iterator's exact lower size bound. Reserve the known output
    // length explicitly so a sparse selection allocates once instead of growing while gathering.
    let mut selected = BufferMut::with_capacity(mask.true_count());
    let mut read = |index: usize| -> VortexResult<()> {
        selected.push(N::try_from_le_bytes(
            &values[index * width..(index + 1) * width],
        )?);
        Ok(())
    };
    match mask.indices() {
        vortex::mask::AllOr::All => (0..present).try_for_each(read)?,
        vortex::mask::AllOr::None => {}
        vortex::mask::AllOr::Some(indices) => indices.iter().copied().try_for_each(&mut read)?,
    }
    Ok(Some(selected.freeze()))
}

/// Uses `Decoder::skip` for false runs and `Decoder::get` for true runs.
fn decode_masked<T, N>(
    decoder: &mut dyn Decoder<T>,
    validity: Option<&[bool]>,
    mask: &Mask,
) -> VortexResult<(Vec<N>, Option<Vec<bool>>)>
where
    T: ParquetDataType<T = N>,
    N: Default,
{
    let all_slice = [(0, mask.len())];
    let slices = match mask.slices() {
        vortex::mask::AllOr::All => &all_slice[..],
        vortex::mask::AllOr::None => &[],
        vortex::mask::AllOr::Some(slices) => slices,
    };
    let selected_validity = validity.map(|validity| {
        let mut selected = Vec::with_capacity(mask.true_count());
        for &(start, end) in slices {
            selected.extend_from_slice(&validity[start..end]);
        }
        selected
    });
    let selected_present = selected_validity.as_ref().map_or_else(
        || mask.true_count(),
        |valid| valid.iter().filter(|bit| **bit).count(),
    );
    let mut decoded = Vec::with_capacity(selected_present);
    let present_in = |start, end| {
        validity.map_or(end - start, |validity| {
            validity[start..end].iter().filter(|bit| **bit).count()
        })
    };
    let mut previous_end = 0;
    // The final empty range retains the decoder's check of the unselected tail. Ranges are
    // already cached by Mask, so multiple columns can reuse them without expanding row booleans.
    for (start, end) in slices
        .iter()
        .copied()
        .chain(std::iter::once((mask.len(), mask.len())))
    {
        let skipped = present_in(previous_end, start);
        if skipped != 0 {
            let count = decoder
                .skip(skipped)
                .map_err(|error| vortex_err!("failed to skip Parquet values: {error}"))?;
            vortex_ensure!(count == skipped, "skipped {count} of {skipped} values");
        }
        let present = present_in(start, end);
        if present != 0 {
            let offset = decoded.len();
            decoded.resize_with(offset + present, N::default);
            let count = decoder
                .get(&mut decoded[offset..])
                .map_err(|error| vortex_err!("failed to read selected Parquet values: {error}"))?;
            vortex_ensure!(
                count == present,
                "decoded {count} of {present} selected values"
            );
        }
        previous_end = end;
    }
    Ok((decoded, selected_validity))
}

/// Returns a dictionary's binary payload without decoding codes or validating unreferenced text.
fn binary_dictionary_values(data: &ParquetPageData) -> VortexResult<Option<ArrayRef>> {
    if data.page.is_dictionary_page()
        || data.descriptor.physical_type() != PhysicalType::BYTE_ARRAY
        || !matches!(
            data.encoding(),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        )
    {
        return Ok(None);
    }
    let Some(source) = data.dictionary_values.as_ref() else {
        return Ok(None);
    };
    if matches!(source.dtype(), DType::Binary(_)) {
        return Ok(Some(source.clone()));
    }
    let Some(page) = source.as_opt::<ParquetPlain>() else {
        return Ok(None);
    };
    if !page.data().page.is_dictionary_page() {
        return Ok(None);
    }
    // A Parquet dictionary page contains bytes regardless of its field's logical string type.
    // Relabel only this known page representation, without executing a fallible UTF-8 cast.
    Ok(Some(
        Array::try_from_parts(ArrayParts::new(
            ParquetPlain,
            DType::Binary(page.dtype().nullability()),
            page.len(),
            page.data().clone(),
            page.slots().iter().cloned().collect::<ArraySlots>(),
        ))?
        .into_array(),
    ))
}

/// Gathers dictionary views before validating the selected logical string values.
fn materialize_dictionary_bytes(
    dictionary: ArrayRef,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let gathered = dictionary.execute::<VarBinViewArray>(ctx)?;
    if matches!(dtype, DType::Binary(_)) {
        return Ok(gathered.into_array());
    }
    // Only dictionary entries referenced by this page's nonnull codes are checked as UTF-8.
    let parts = gathered.into_data_parts();
    let views = Buffer::<BinaryView>::from_byte_buffer(parts.views.as_host().clone());
    let buffers = parts
        .buffers
        .iter()
        .map(|buffer| buffer.as_host().clone())
        .collect::<Vec<_>>()
        .into();
    Ok(VarBinViewArray::try_new(views, buffers, dtype.clone(), parts.validity, ctx)?.into_array())
}

/// Reduces a Parquet dictionary page to Vortex's native dictionary representation.
///
/// This is deliberately a structural reduction rather than a catalogue of predicate kernels.
/// Once the values and codes are separate, Vortex's generic `DictArray` rules can push any safe
/// scalar function through the dictionary. Those rules make the decision from scalar-function
/// properties such as fallibility and null sensitivity, and therefore automatically cover new
/// scalar functions without this crate learning their names.
fn reduce_dictionary_page(data: &ParquetPageData, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    if !matches!(
        data.page.encoding(),
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    ) {
        return Ok(None);
    }

    let values = data
        .dictionary_values
        .as_ref()
        .ok_or_else(|| vortex_err!("dictionary-encoded page has no dictionary values"))?
        .clone();
    // A metadata-only cast can leave the dictionary in the original physical dtype. Keep that
    // path on the page decoder so conversion and validation happen in the requested dtype.
    if !values.dtype().eq_ignore_nullability(dtype) {
        return Ok(None);
    }
    let codes = decode_dictionary_codes(data, values.len())?;
    Ok(Some(DictArray::try_new(codes, values)?.into_array()))
}

fn decode_dictionary_codes(
    data: &ParquetPageData,
    dictionary_len: usize,
) -> VortexResult<ArrayRef> {
    if let Some(codes) = data.dictionary_codes.get() {
        return Ok(codes.clone());
    }
    let page = data.page.page()?;
    // A complete all-present run needs neither a row-sized validity allocation nor a scatter.
    // Keep its nullable type even though every code in this page is valid.
    let present_run = (data.descriptor.max_def_level() == 1)
        .then(|| single_present_run_values(page))
        .flatten();
    let all_present = present_run.is_some();
    let (encoded_codes, valid) = match present_run {
        Some(values) => (values, None),
        None => value_stream_and_validity(page, data.descriptor.max_def_level())?,
    };
    vortex_ensure!(
        !encoded_codes.is_empty(),
        "dictionary code stream has no bit width"
    );
    let present = valid.as_ref().map_or_else(
        || usize::try_from(data.page.num_values()),
        |bits| Ok(bits.iter().filter(|bit| **bit).count()),
    )?;
    let bit_width = encoded_codes[0];
    let decoded = hybrid_codes::decode(encoded_codes.slice(1..), bit_width, present)?;
    // Reducing to the maximum lets the compiler validate a block of codes at a time.
    // Empty/all-null pages have no present codes and therefore need no dictionary entry.
    vortex_ensure!(
        decoded
            .iter()
            .copied()
            .max()
            .is_none_or(|code| { usize::try_from(code).is_ok_and(|code| code < dictionary_len) }),
        "Parquet dictionary code exceeds dictionary length {}",
        dictionary_len
    );
    let (codes, validity) = if all_present {
        (decoded, Validity::AllValid)
    } else {
        scatter(decoded, valid)
    };
    let codes = PrimitiveArray::new(Buffer::from(codes), validity).into_array();
    // Predicate reduction and projection share this immutable page and dictionary. Retain the
    // validated codes with the page, just as we retain decompressed bytes, so each consumer can
    // reuse the decode. Concurrent first executions may race; either result is equivalent.
    drop(data.dictionary_codes.set(codes.clone()));
    Ok(codes)
}

fn reduce_dictionary_parent(
    data: &ParquetPageData,
    dtype: &DType,
    parent: &ArrayRef,
    child_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Some(parent) = parent.as_opt::<ScalarFn>() else {
        return Ok(None);
    };
    let Some(dictionary) = reduce_dictionary_page(data, dtype)? else {
        return Ok(None);
    };
    let mut children = parent.iter_children().cloned().collect::<Vec<_>>();
    let Some(child) = children.get_mut(child_idx) else {
        return Ok(None);
    };
    *child = dictionary;
    Ok(Some(
        ScalarFnArray::try_new(parent.scalar_fn().clone(), children)?.into_array(),
    ))
}

fn reduce_page_parent<V>(
    array: ArrayView<'_, V>,
    vtable: V,
    parent: &ArrayRef,
    child_idx: usize,
) -> VortexResult<Option<ArrayRef>>
where
    V: VTable<TypedArrayData = ParquetPageData>,
{
    if child_idx == 0
        && let Some(parent) = parent.as_opt::<ScalarFn>()
        && parent.scalar_fn().id() == Cast.id()
        && direct_dtype(parent.dtype(), array.data().descriptor.physical_type())
        && (parent.dtype().is_nullable() || !array.dtype().is_nullable())
    {
        // A cast target supported by the Parquet decoder is a metadata rewrite until somebody asks
        // for values. This preserves the encoded page below LIKE/compare and performs conversion
        // (including UTF-8 validation) exactly once during decode.
        let slots: ArraySlots = array.slots().iter().cloned().collect();
        let cast = Array::try_from_parts(ArrayParts::new(
            vtable,
            parent.dtype().clone(),
            array.len(),
            array.data().clone(),
            slots,
        ))?
        .into_array();
        return Ok(Some(cast));
    }
    reduce_dictionary_parent(array.data(), array.dtype(), parent, child_idx)
}

fn make_decoder<T: ParquetDataType>(
    data: &ParquetPageData,
    values: Bytes,
    present: usize,
) -> VortexResult<Box<dyn Decoder<T>>> {
    // The legacy PLAIN_DICTIONARY enum is valid on a dictionary-page header, but the dictionary
    // payload itself is always PLAIN encoded. Passing that header enum to parquet-rs asks it to
    // construct a *code* decoder and either fails immediately or recursively asks for another
    // dictionary. Data-page headers retain their actual value encoding.
    let encoding = if data.page.is_dictionary_page() {
        Encoding::PLAIN
    } else {
        data.page.encoding()
    };
    let mut decoder: Box<dyn Decoder<T>> = if matches!(
        encoding,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    ) {
        let dictionary = data
            .dictionary
            .as_ref()
            .ok_or_else(|| vortex_err!("dictionary-encoded page has no dictionary"))?;
        let dictionary = dictionary.page()?;
        // Dictionary payloads are always PLAIN. Some legacy writers put PLAIN_DICTIONARY in the
        // dictionary-page header; that tag describes the dictionary scheme, not the payload bytes.
        let mut values_decoder = get_decoder::<T>(Arc::clone(&data.descriptor), Encoding::PLAIN)
            .map_err(|e| vortex_err!("failed to construct Parquet dictionary decoder: {e}"))?;
        values_decoder
            .set_data(
                dictionary.buffer().clone(),
                usize::try_from(dictionary.num_values())?,
            )
            .map_err(|e| vortex_err!("failed to initialize Parquet dictionary values: {e}"))?;
        let mut dict = DictDecoder::<T>::new();
        dict.set_dict(values_decoder)
            .map_err(|e| vortex_err!("failed to decode Parquet dictionary: {e}"))?;
        Box::new(dict)
    } else {
        get_decoder::<T>(Arc::clone(&data.descriptor), encoding)
            .map_err(|e| vortex_err!("failed to construct Parquet {encoding} decoder: {e}"))?
    };
    decoder
        .set_data(values, present)
        .map_err(|e| vortex_err!("failed to initialize Parquet {encoding} decoder: {e}"))?;
    Ok(decoder)
}

/// PLAIN numbers already have their native representation on little-endian hosts. Null-free
/// pages can borrow that storage; unaligned payloads need one copy, without zero initialization.
fn plain_numeric_buffer<N: NativePType>(
    data: &ParquetPageData,
    values: &Bytes,
    valid: Option<&[bool]>,
    present: usize,
) -> VortexResult<Option<Buffer<N>>> {
    // Callers already counted present values while decoding definition levels.
    // A matching logical row count proves there are no nulls without another full pass.
    if !cfg!(target_endian = "little")
        || data.page.encoding() != Encoding::PLAIN
        || valid.is_some_and(|valid| valid.len() != present)
    {
        return Ok(None);
    }
    let byte_len = present
        .checked_mul(size_of::<N>())
        .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
    vortex_ensure!(byte_len <= values.len(), "truncated PLAIN numeric payload");
    let buffer = ByteBuffer::from(values.slice(..byte_len)).aligned(Alignment::of::<N>());
    Ok(Some(Buffer::from_byte_buffer(buffer)))
}

/// Decode PLAIN INT32 directly into a smaller logical integer buffer. Validate every narrowed
/// value by round-tripping it before exposing the result, without a separate range scan.
fn decode_plain_narrow(
    data: &ParquetPageData,
    values: &[u8],
    valid: Option<&[bool]>,
    present: usize,
    dtype: &DType,
) -> VortexResult<Option<ArrayRef>> {
    if data.page.encoding() != Encoding::PLAIN
        || !matches!(
            dtype,
            DType::Primitive(PType::I8 | PType::I16 | PType::U8 | PType::U16, _)
        )
        || valid.is_some_and(|valid| valid.len() != present)
    {
        return Ok(None);
    }
    let byte_len = present
        .checked_mul(4)
        .ok_or_else(|| vortex_err!("PLAIN numeric payload length overflow"))?;
    vortex_ensure!(byte_len <= values.len(), "truncated PLAIN numeric payload");
    let (values, _) = values[..byte_len].as_chunks::<4>();
    // The Arrow schema may override physical field nullability. Every value is present here,
    // so preserve the requested logical nullability just as logicalize does for other decoders.
    let validity = Validity::from(dtype.nullability());
    macro_rules! narrow {
        ($native:ty) => {{
            let mut lost_bits = 0i32;
            let mut narrowed = BufferMut::<$native>::with_capacity(values.len());
            #[expect(
                clippy::cast_possible_truncation,
                reason = "round-trip validation rejects any truncation before returning the array"
            )]
            for (bytes, slot) in values
                .iter()
                .zip(&mut narrowed.spare_capacity_mut()[..values.len()])
            {
                let value = i32::from_le_bytes(*bytes);
                let narrow = value as $native;
                // OR preserves any mismatch, including signed-extension and unsigned-range
                // errors. Invalid values can never cancel one another in the reduction.
                lost_bits |= value ^ i32::from(narrow);
                slot.write(narrow);
            }
            vortex_ensure!(
                lost_bits == 0,
                "PLAIN integer value exceeds logical {dtype} range"
            );
            // SAFETY: the two equal-length slices above initialize every output slot exactly
            // once. No partially initialized buffer is exposed, including on validation failure.
            unsafe { narrowed.set_len(values.len()) };
            Some(PrimitiveArray::new(narrowed.freeze(), validity).into_array())
        }};
    }
    Ok(match dtype {
        DType::Primitive(PType::I8, _) => narrow!(i8),
        DType::Primitive(PType::I16, _) => {
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                target_feature = "neon"
            ))]
            {
                // SAFETY: this branch is compiled only for little-endian aarch64 with NEON.
                let (values, in_range) = unsafe { neon_narrow::i16_values(values) };
                vortex_ensure!(
                    in_range,
                    "PLAIN integer value exceeds logical {dtype} range"
                );
                Some(PrimitiveArray::new(values, validity).into_array())
            }
            #[cfg(not(all(
                target_arch = "aarch64",
                target_endian = "little",
                target_feature = "neon"
            )))]
            {
                narrow!(i16)
            }
        }
        DType::Primitive(PType::U8, _) => narrow!(u8),
        DType::Primitive(PType::U16, _) => narrow!(u16),
        _ => None,
    })
}

fn decode_primitive<T, N>(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    dtype: &DType,
) -> VortexResult<ArrayRef>
where
    T: ParquetDataType<T = N>,
    N: NativePType + Default,
{
    if let Some(values) = plain_numeric_buffer::<N>(data, &values, valid.as_deref(), present)? {
        let validity = if valid.is_some() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        return logicalize(PrimitiveArray::new(values, validity).into_array(), dtype);
    }
    let mut decoder = make_decoder::<T>(data, values, present)?;
    let mut decoded = vec![N::default(); present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet values: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} values"
    );
    let (values, validity) = scatter(decoded, valid);
    logicalize(
        PrimitiveArray::new(Buffer::from(values), validity).into_array(),
        dtype,
    )
}

fn decode_decimal_primitive<T, N>(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: ParquetDataType<T = N>,
    N: NativePType + NativeDecimalType + Default,
{
    if let Some(values) = plain_numeric_buffer::<N>(data, &values, valid.as_deref(), present)? {
        let DType::Decimal(decimal_dtype, _) = dtype else {
            vortex_bail!("Parquet decimal decoder received non-decimal dtype {dtype}")
        };
        let validity = if valid.is_some() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        return decimal_array(values, *decimal_dtype, validity, ctx);
    }
    let mut decoder = make_decoder::<T>(data, values, present)?;
    let mut decoded = vec![N::default(); present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet decimal values: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} decimal values"
    );
    build_decimal(decoded, valid, dtype, ctx)
}

fn build_decimal<N>(
    decoded: Vec<N>,
    valid: Option<Vec<bool>>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    N: NativePType + NativeDecimalType + Default,
{
    let DType::Decimal(decimal_dtype, _) = dtype else {
        vortex_bail!("Parquet decimal decoder received non-decimal dtype {dtype}")
    };
    let (values, validity) = scatter(decoded, valid);
    decimal_array(Buffer::from(values), *decimal_dtype, validity, ctx)
}

/// Applies Parquet's logical annotation without going through Arrow.
fn logicalize(array: ArrayRef, dtype: &DType) -> VortexResult<ArrayRef> {
    if array.dtype() == dtype {
        return Ok(array);
    }
    if let DType::Extension(ext) = dtype
        && ext.storage_dtype() == array.dtype()
    {
        return Ok(ExtensionArray::try_new(ext.clone(), array)?.into_array());
    }
    if let (Some(primitive), DType::Primitive(target, nullability)) =
        (array.as_opt::<Primitive>(), dtype)
        && primitive.ptype().byte_width() == target.byte_width()
        && primitive.ptype().is_signed_int()
        && target.is_unsigned_int()
    {
        let reinterpreted = primitive.reinterpret_cast(*target).into_array();
        return if reinterpreted.dtype().nullability() == *nullability {
            Ok(reinterpreted)
        } else {
            reinterpreted.cast(dtype.clone())
        };
    }
    array.cast(dtype.clone())
}

fn decode_bool(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
) -> VortexResult<ArrayRef> {
    let mut decoder = make_decoder::<BoolType>(data, values, present)?;
    let mut decoded = vec![false; present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet booleans: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} booleans"
    );
    let result: BoolArray = match valid {
        Some(valid) if valid.iter().all(|is_valid| *is_valid) => {
            let mut values = BitBufferMut::with_capacity(decoded.len());
            for value in decoded {
                values.append(value);
            }
            BoolArray::new(values.freeze(), Validity::AllValid)
        }
        Some(valid) => {
            let mut next = decoded.into_iter();
            valid
                .into_iter()
                .map(|v| v.then(|| next.next()).flatten())
                .collect()
        }
        None => decoded.into_iter().collect(),
    };
    Ok(result.into_array())
}

fn decode_bytes(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decoded = decode_byte_values(data, values, present)?;
    drop(data.decoded_bytes.set(DecodedByteValues {
        values: decoded,
        valid: valid.map(Into::into),
    }));
    let decoded = decoded_byte_values(data)?;
    build_page_bytes(data, &decoded.values, decoded.valid.as_deref(), dtype, ctx)
}

fn decode_byte_values(
    data: &ParquetPageData,
    values: Bytes,
    present: usize,
) -> VortexResult<ByteValues> {
    if data.page.encoding() == Encoding::PLAIN {
        let mut remaining = values.as_ref();
        let views = BufferMut::try_from_trusted_len_iter((0..present).map(|_| {
            let (length, rest) = remaining
                .split_first_chunk::<4>()
                .ok_or_else(|| vortex_err!("truncated PLAIN byte-array length"))?;
            let length = usize::try_from(u32::from_le_bytes(*length))?;
            vortex_ensure!(length <= rest.len(), "truncated PLAIN byte-array payload");
            let start = values.len() - rest.len();
            let view = BinaryView::make_view(&rest[..length], 0, u32::try_from(start)?);
            remaining = &rest[length..];
            Ok(view)
        }))?;
        return Ok(ByteValues::Plain {
            buffer: ByteBuffer::from(values),
            views: views.freeze(),
            utf8_validated: OnceLock::new(),
        });
    }
    let mut decoder = make_decoder::<ByteArrayType>(data, values, present)?;
    let mut decoded = vec![ByteArray::default(); present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet byte arrays: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} byte arrays"
    );
    Ok(ByteValues::Decoded(decoded))
}

fn decode_fixed_bytes(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let mut decoder = make_decoder::<FixedLenByteArrayType>(data, values, present)?;
    let mut decoded = vec![FixedLenByteArray::default(); present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet fixed byte arrays: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} fixed byte arrays"
    );
    build_fixed_bytes(decoded, valid, dtype)
}

fn build_fixed_bytes(
    decoded: Vec<FixedLenByteArray>,
    valid: Option<Vec<bool>>,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let logical_len = valid.as_ref().map_or(decoded.len(), Vec::len);
    let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), logical_len);
    let mut next = decoded.into_iter();
    match valid {
        Some(valid) if valid.iter().all(|is_valid| *is_valid) => {
            for value in next {
                builder.append_value(value.data());
            }
        }
        Some(valid) => {
            for is_valid in valid {
                if is_valid {
                    let value = next.next().ok_or_else(|| {
                        vortex_err!("Parquet validity exceeds decoded fixed byte arrays")
                    })?;
                    builder.append_value(value.data());
                } else {
                    builder.append_null();
                }
            }
        }
        None => {
            for value in next {
                builder.append_value(value.data());
            }
        }
    }
    Ok(builder.finish())
}

fn decode_int96(
    data: &ParquetPageData,
    values: Bytes,
    valid: Option<Vec<bool>>,
    present: usize,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let mut decoder = make_decoder::<Int96Type>(data, values, present)?;
    let mut decoded = vec![Int96::default(); present];
    let count = decoder
        .get(&mut decoded)
        .map_err(|e| vortex_err!("failed to decode Parquet INT96 values: {e}"))?;
    vortex_ensure!(
        count == present,
        "Parquet decoder returned {count} of {present} INT96 values"
    );
    build_int96(decoded, valid, dtype)
}

fn build_int96(
    decoded: Vec<Int96>,
    valid: Option<Vec<bool>>,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let DType::Extension(ext) = dtype else {
        vortex_bail!("Parquet INT96 requires a temporal extension dtype, found {dtype}")
    };
    let Some(TemporalMetadata::Timestamp(unit, _)) = ext.metadata_opt::<AnyTemporal>() else {
        vortex_bail!("Parquet INT96 requires a timestamp dtype, found {dtype}")
    };
    let values = decoded
        .into_iter()
        .map(|value| match unit {
            TimeUnit::Nanoseconds => value.to_nanos(),
            TimeUnit::Microseconds => value.to_micros(),
            TimeUnit::Milliseconds => value.to_millis(),
            TimeUnit::Seconds => value.to_seconds(),
            TimeUnit::Days => value.to_seconds().div_euclid(86_400),
        })
        .collect();
    let (values, validity) = scatter(values, valid);
    let storage = PrimitiveArray::new(Buffer::from(values), validity).into_array();
    Ok(ExtensionArray::try_new(ext.clone(), storage)?.into_array())
}

fn build_bytes<'a>(
    decoded: impl ExactSizeIterator<Item = &'a [u8]>,
    valid: Option<&[bool]>,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let logical_len = valid.map_or_else(|| decoded.len(), <[bool]>::len);
    let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), logical_len);
    let mut next = decoded;
    match valid {
        Some(valid) if valid.iter().all(|is_valid| *is_valid) => {
            for value in next {
                builder.append_value(value);
            }
        }
        Some(valid) => {
            for is_valid in valid {
                if *is_valid {
                    let value = next
                        .next()
                        .ok_or_else(|| vortex_err!("Parquet validity exceeds decoded values"))?;
                    builder.append_value(value);
                } else {
                    builder.append_null();
                }
            }
        }
        None => {
            for value in next {
                builder.append_value(value);
            }
        }
    }
    Ok(builder.finish())
}

/// Like Vortex's varbinview validation, check contiguous storage in bulk. PLAIN interleaves
/// four-byte lengths with payloads: ASCII lengths can remain inside a UTF-8 validation run,
/// while non-ASCII lengths delimit runs and are excluded. ASCII separators also prevent an
/// incomplete character in one string from being completed by bytes in the next string.
fn validate_plain_utf8(buffer: &[u8], count: usize) -> VortexResult<()> {
    let mut cursor = 0;
    let mut start = 0;
    for _ in 0..count {
        let (length, rest) = buffer[cursor..]
            .split_first_chunk::<4>()
            .ok_or_else(|| vortex_err!("truncated PLAIN byte-array length"))?;
        let size = usize::try_from(u32::from_le_bytes(*length))?;
        vortex_ensure!(size <= rest.len(), "truncated PLAIN byte-array payload");
        if !length.is_ascii() {
            vortex_ensure!(
                simdutf8::basic::from_utf8(&buffer[start..cursor]).is_ok(),
                "PLAIN byte-array payload is not valid UTF-8"
            );
            start = cursor + 4;
        }
        cursor += 4 + size;
    }
    vortex_ensure!(
        simdutf8::basic::from_utf8(&buffer[start..cursor]).is_ok(),
        "PLAIN byte-array payload is not valid UTF-8"
    );
    Ok(())
}

fn build_page_bytes(
    data: &ParquetPageData,
    decoded: &ByteValues,
    valid: Option<&[bool]>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let ByteValues::Plain {
        buffer,
        views,
        utf8_validated,
    } = decoded
    {
        vortex_ensure!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "byte-array page requires a UTF-8 or binary dtype"
        );
        if dtype.is_utf8() && utf8_validated.get().is_none() {
            validate_plain_utf8(buffer, views.len())?;
            utf8_validated.get_or_init(|| ());
        }
        let (views, validity) = match valid {
            None => (views.clone(), Validity::from(dtype.nullability())),
            Some(valid) if valid.iter().all(|valid| *valid) => (views.clone(), Validity::AllValid),
            Some(valid) => {
                let mut output = BufferMut::with_capacity(valid.len());
                let mut next = views.iter();
                for is_valid in valid {
                    output.push(if *is_valid {
                        *next
                            .next()
                            .ok_or_else(|| vortex_err!("Parquet validity exceeds decoded values"))?
                    } else {
                        BinaryView::default()
                    });
                }
                vortex_ensure!(
                    next.next().is_none(),
                    "Parquet byte decode returned more values than its validity"
                );
                (output.freeze(), Validity::from(BitBuffer::from(valid)))
            }
        };
        vortex_ensure!(
            dtype.nullability() == validity.nullability(),
            "validity nullability is incompatible with dtype nullability"
        );
        // SAFETY: decode_byte_values checked every length and offset and built the views with
        // BinaryView::make_view, including matching prefixes and buffer index zero. UTF-8 was
        // checked above for every present value. The loop above inserts empty views at nulls
        // and constructs equally sized validity; dtype and nullability are checked explicitly.
        return Ok(unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from([buffer.clone()]),
                dtype.clone(),
                validity,
            )
        }
        .into_array());
    }
    let mut candidates = vec![data.page.page()?.buffer().clone()];
    if let Some(dictionary) = &data.dictionary {
        candidates.push(dictionary.page()?.buffer().clone());
    }
    if let Some(array) = try_build_zero_copy_bytes(decoded.iter(), valid, dtype, &candidates, ctx)?
    {
        return Ok(array);
    }
    build_bytes(decoded.iter(), valid, dtype)
}

/// Builds views directly over decompressed Parquet storage when its decoder returned `Bytes`
/// slices into a page or dictionary buffer. Encodings such as delta byte array may reconstruct
/// values elsewhere; those safely decline this path and use the copying builder.
fn try_build_zero_copy_bytes<'a>(
    decoded: impl ExactSizeIterator<Item = &'a [u8]>,
    valid: Option<&[bool]>,
    dtype: &DType,
    candidates: &[Bytes],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut views =
        BufferMut::<BinaryView>::with_capacity(valid.map_or_else(|| decoded.len(), <[bool]>::len));
    let logical_valid = valid.unwrap_or(&[]);
    let logical_len = valid.map_or_else(|| decoded.len(), <[bool]>::len);
    let mut next = decoded;
    for index in 0..logical_len {
        if valid.is_some() && !logical_valid[index] {
            views.push(BinaryView::default());
            continue;
        }
        let value = next
            .next()
            .ok_or_else(|| vortex_err!("Parquet validity exceeds decoded values"))?;
        let bytes = value;
        if bytes.len() <= BinaryView::MAX_INLINED_SIZE {
            views.push(BinaryView::make_view(bytes, 0, 0));
            continue;
        }
        let value_start = bytes.as_ptr() as usize;
        let Some((buffer_index, offset)) =
            candidates.iter().enumerate().find_map(|(idx, buffer)| {
                let buffer_start = buffer.as_ptr() as usize;
                let offset = value_start.checked_sub(buffer_start)?;
                (offset.checked_add(bytes.len())? <= buffer.len()).then_some((idx, offset))
            })
        else {
            return Ok(None);
        };
        let Ok(buffer_index) = u32::try_from(buffer_index) else {
            return Ok(None);
        };
        let Ok(offset) = u32::try_from(offset) else {
            return Ok(None);
        };
        views.push(BinaryView::make_view(bytes, buffer_index, offset));
    }
    vortex_ensure!(
        next.next().is_none(),
        "Parquet byte decode returned more values than its validity"
    );
    let validity = match valid {
        Some(valid) if valid.iter().all(|is_valid| *is_valid) => Validity::AllValid,
        Some(valid) => Validity::from(BitBuffer::from(valid)),
        None => Validity::from(dtype.nullability()),
    };
    let buffers: Arc<[ByteBuffer]> = candidates
        .iter()
        .cloned()
        .map(ByteBuffer::from)
        .collect::<Vec<_>>()
        .into();
    Ok(Some(
        VarBinViewArray::try_new(views.freeze(), buffers, dtype.clone(), validity, ctx)?
            .into_array(),
    ))
}

fn scatter<T: Default>(values: Vec<T>, valid: Option<Vec<bool>>) -> (Vec<T>, Validity) {
    let Some(valid) = valid else {
        return (values, Validity::NonNullable);
    };
    if valid.iter().all(|is_valid| *is_valid) {
        return (values, Validity::AllValid);
    }
    let mut next = values.into_iter();
    let mut out = Vec::with_capacity(valid.len());
    let mut bits = BitBufferMut::with_capacity(valid.len());
    for is_valid in valid {
        bits.append(is_valid);
        out.push(if is_valid {
            next.next().unwrap_or_default()
        } else {
            T::default()
        });
    }
    (out, Validity::from(bits.freeze()))
}

/// Recognize exactly one canonical RLE definition-level run marking every row present.
/// Other encodings, multiple runs, nulls, and malformed streams retain the general decoder.
fn single_present_run_values(page: &Page) -> Option<Bytes> {
    let (buf, count, start, end) = match page {
        Page::DataPage {
            buf,
            num_values,
            def_level_encoding: Encoding::RLE,
            ..
        } => {
            let length = u32::from_le_bytes(*buf.first_chunk::<4>()?);
            let end = 4usize.checked_add(usize::try_from(length).ok()?)?;
            (buf, *num_values, 4, end)
        }
        Page::DataPageV2 {
            buf,
            num_values,
            def_levels_byte_len,
            rep_levels_byte_len: 0,
            ..
        } => (
            buf,
            *num_values,
            0,
            usize::try_from(*def_levels_byte_len).ok()?,
        ),
        _ => return None,
    };
    if count == 0 {
        return None;
    }
    let mut levels = buf.get(start..end)?;
    let mut header = u64::from(count) << 1;
    loop {
        let mut expected = u8::try_from(header & 0x7f).ok()?;
        header >>= 7;
        if header != 0 {
            expected |= 0x80;
        }
        let (&byte, rest) = levels.split_first()?;
        if byte != expected {
            return None;
        }
        levels = rest;
        if header == 0 {
            break;
        }
    }
    (levels == [1]).then(|| buf.slice(end..))
}

fn value_stream_and_validity(
    page: &Page,
    max_def_level: i16,
) -> VortexResult<(Bytes, Option<Vec<bool>>)> {
    match page {
        Page::DataPage {
            buf,
            num_values,
            def_level_encoding,
            ..
        } => {
            if max_def_level == 0 {
                return Ok((buf.clone(), None));
            }
            let count = usize::try_from(*num_values)?;
            let (end, levels) = match def_level_encoding {
                Encoding::RLE => {
                    vortex_ensure!(buf.len() >= 4, "truncated V1 definition-level length");
                    let level_bytes =
                        u32::from_le_bytes(buf[..4].try_into().map_err(|_| {
                            vortex_err!("definition-level length is not four bytes")
                        })?) as usize;
                    let end = 4usize
                        .checked_add(level_bytes)
                        .ok_or_else(|| vortex_err!("definition-level length overflows"))?;
                    vortex_ensure!(end <= buf.len(), "definition levels extend past the page");
                    (end, buf.slice(4..end))
                }
                Encoding::BIT_PACKED => {
                    let bit_width = usize::from(num_required_bits(u64::try_from(max_def_level)?));
                    let level_bytes = count
                        .checked_mul(bit_width)
                        .and_then(|bits| bits.checked_add(7))
                        .map(|bits| bits / 8)
                        .ok_or_else(|| vortex_err!("definition-level length overflows"))?;
                    vortex_ensure!(
                        level_bytes <= buf.len(),
                        "BIT_PACKED definition levels extend past the page"
                    );
                    (level_bytes, buf.slice(..level_bytes))
                }
                encoding => vortex_bail!("invalid V1 definition-level encoding {encoding}"),
            };
            let valid = decode_levels(levels, count, max_def_level, *def_level_encoding)?;
            Ok((buf.slice(end..), Some(valid)))
        }
        Page::DataPageV2 {
            buf,
            num_values,
            def_levels_byte_len,
            rep_levels_byte_len,
            ..
        } => {
            vortex_ensure!(
                *rep_levels_byte_len == 0,
                "nested V2 page reached flat decoder"
            );
            if max_def_level == 0 {
                return Ok((buf.clone(), None));
            }
            let end = usize::try_from(*def_levels_byte_len)?;
            vortex_ensure!(
                end <= buf.len(),
                "V2 definition levels extend past the page"
            );
            let valid = decode_levels(
                buf.slice(0..end),
                usize::try_from(*num_values)?,
                max_def_level,
                Encoding::RLE,
            )?;
            Ok((buf.slice(end..), Some(valid)))
        }
        // A dictionary page is not a row stream, but its payload is a PLAIN value stream. Modeling
        // that stream as a lazy page array is what permits cast-on-decode and dictionary-value
        // predicates without eagerly materializing the dictionary.
        Page::DictionaryPage { buf, .. } => Ok((buf.clone(), None)),
    }
}

fn decode_levels(
    data: Bytes,
    count: usize,
    max_level: i16,
    encoding: Encoding,
) -> VortexResult<Vec<bool>> {
    // Flat optional columns need only a Boolean result. Decode runs directly into that
    // buffer instead of materializing i16 levels and then mapping them to validity bytes.
    // The lookup also rejects malformed repeated levels outside the field's 0..=1 domain.
    if encoding == Encoding::RLE && max_level == 1 {
        let mut decoder = RleDecoder::new(1);
        decoder
            .set_data(data)
            .map_err(|e| vortex_err!("failed to initialize definition-level decoder: {e}"))?;
        let mut valid = vec![false; count];
        let decoded = decoder
            .get_batch_with_dict(&[false, true], &mut valid, count)
            .map_err(|e| vortex_err!("failed to decode definition levels: {e}"))?;
        vortex_ensure!(
            decoded == count,
            "decoded {decoded} of {count} definition levels"
        );
        return Ok(valid);
    }
    let bit_width = num_required_bits(u64::try_from(max_level)?);
    let mut levels = vec![0i16; count];
    let decoded = match encoding {
        Encoding::RLE => {
            let mut decoder = RleDecoder::new(bit_width);
            decoder
                .set_data(data)
                .map_err(|e| vortex_err!("failed to initialize definition-level decoder: {e}"))?;
            decoder
                .get_batch(&mut levels)
                .map_err(|e| vortex_err!("failed to decode definition levels: {e}"))?
        }
        Encoding::BIT_PACKED => BitReader::new(data).get_batch(&mut levels, usize::from(bit_width)),
        encoding => vortex_bail!("invalid definition-level encoding {encoding}"),
    };
    vortex_ensure!(
        decoded == count,
        "decoded {decoded} of {count} definition levels"
    );
    Ok(levels.into_iter().map(|level| level == max_level).collect())
}

fn decimal_array<N: NativeDecimalType>(
    values: Buffer<N>,
    dtype: DecimalDType,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let width = dtype.canonical_values_type();
    let values = if N::DECIMAL_TYPE == width {
        BufferHandle::new_host(values.into_byte_buffer())
    } else {
        let mask = validity.execute_mask(values.len(), ctx)?;
        copy_to_canonical_width(values.as_slice(), &mask, dtype)?
    };
    Ok(DecimalArray::try_new_handle_from_trusted_values(
        values,
        dtype.canonical_values_type(),
        dtype,
        validity,
    )?
    .into_array())
}
