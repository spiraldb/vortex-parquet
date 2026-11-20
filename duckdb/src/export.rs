// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct vector export, following vortex-duckdb's exporter at the pinned Vortex revision.
//! Constants, dictionaries and chunks retain their physical structure. Primitive and string
//! buffers stay alive through DuckDB-owned reference counts, including after a scan is dropped.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use duckdb::core::DataChunkHandle;
use duckdb::core::LogicalTypeHandle;
use duckdb::core::LogicalTypeId;
use duckdb::ffi;
use num_traits::ToPrimitive;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::Constant;
use vortex::array::arrays::Dict;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::listview::ListViewArraySlotsExt;
use vortex::array::arrays::map::MapArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::array::match_each_decimal_value_type;
use vortex::array::matcher::Matcher;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::PType;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::TemporalMetadata;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_parquet::pages::ParquetPlain;
use vortex_parquet::pages::PlainI16Page;

use crate::Result;

unsafe extern "C" {
    fn vp_buffer_new(owner: *mut c_void, release: unsafe extern "C" fn(*mut c_void))
    -> *mut c_void;
    fn vp_buffer_free(buffer: *mut c_void);
    fn vp_vector_buffer(
        vector: ffi::duckdb_vector,
        buffer: *mut c_void,
        data: *mut c_void,
        count: u64,
        width: u64,
    );
    fn vp_string_buffer(vector: ffi::duckdb_vector, buffer: *mut c_void);
    fn vp_vector_constant(vector: ffi::duckdb_vector);
    fn vp_vector_reference(vector: ffi::duckdb_vector, source: ffi::duckdb_vector) -> bool;
    fn vp_dictionary_new(source: ffi::duckdb_vector, size: u64) -> *mut c_void;
    fn vp_dictionary_free(dictionary: *mut c_void);
    fn vp_dictionary_clone(dictionary: *mut c_void) -> *mut c_void;
    fn vp_vector_dictionary(
        vector: ffi::duckdb_vector,
        dictionary: *mut c_void,
        selection: ffi::duckdb_selection_vector,
        count: u64,
    ) -> bool;
}

/// DuckDB's reusable dictionary carries a stable identity and size, enabling cached hashes and
/// dictionary-aware scalar execution across output chunks (as in vortex-duckdb).
struct Dictionary(*mut c_void);
// SAFETY: owns a reference-counted DuckDB buffer; the scan cursor serializes access.
unsafe impl Send for Dictionary {}
impl Dictionary {
    fn new(values: &Vector, size: usize) -> Result<Self> {
        // SAFETY: the live vector is referenced, not consumed; DuckDB retains all buffer owners.
        let pointer = unsafe { vp_dictionary_new(values.0, size as u64) };
        if pointer.is_null() {
            return Err("DuckDB could not create a reusable dictionary".into());
        }
        Ok(Self(pointer))
    }
    fn try_clone(&self) -> Result<Self> {
        // SAFETY: copies the live shared buffer handle; each wrapper releases one reference.
        let pointer = unsafe { vp_dictionary_clone(self.0) };
        if pointer.is_null() {
            return Err("DuckDB could not retain a reusable dictionary".into());
        }
        Ok(Self(pointer))
    }
}
impl Drop for Dictionary {
    fn drop(&mut self) {
        // SAFETY: releases this wrapper's shared reference once; exported vectors keep theirs.
        unsafe { vp_dictionary_free(self.0) };
    }
}

struct Owner(*mut c_void);
// SAFETY: the holder wraps an immutable, Send + Sync allocation and DuckDB's atomic shared_ptr.
unsafe impl Send for Owner {}
impl Owner {
    fn new<T: Send + Sync + 'static>(value: T) -> Self {
        unsafe extern "C" fn release<T>(value: *mut c_void) {
            // SAFETY: created with Box::into_raw below; the last C++ shared owner calls this once.
            drop(unsafe { Box::from_raw(value.cast::<T>()) });
        }
        // SAFETY: the C++ buffer takes ownership of this allocation and its matching destructor.
        Self(unsafe { vp_buffer_new(Box::into_raw(Box::new(value)).cast(), release::<T>) })
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        // SAFETY: drops our shared reference; vectors retain their own references as needed.
        unsafe { vp_buffer_free(self.0) };
    }
}

struct Vector(ffi::duckdb_vector);
// SAFETY: owned vectors only move between workers under the scan's exclusive cursor lock.
unsafe impl Send for Vector {}
impl Vector {
    fn new(dtype: &DType, capacity: usize) -> Result<Self> {
        let chunk = DataChunkHandle::new(&[logical_type(dtype)?]);
        // SAFETY: obtain an owned logical type from a live vector, create a separate vector, then
        // release the temporary type. The new vector owns its type and buffers.
        let vector = unsafe {
            let mut ty = ffi::duckdb_vector_get_column_type(ffi::duckdb_data_chunk_get_vector(
                chunk.get_ptr(),
                0,
            ));
            let vector = ffi::duckdb_create_vector(ty, capacity as u64);
            ffi::duckdb_destroy_logical_type(&raw mut ty);
            vector
        };
        if vector.is_null() {
            return Err("DuckDB could not allocate a vector".into());
        }
        Ok(Self(vector))
    }
}
impl Drop for Vector {
    fn drop(&mut self) {
        // SAFETY: this is our owning vector handle, released exactly once.
        unsafe { ffi::duckdb_destroy_vector(&raw mut self.0) };
    }
}

pub(crate) fn logical_type(dtype: &DType) -> Result<LogicalTypeHandle> {
    use LogicalTypeId as L;
    let id = match dtype {
        DType::Null => L::SqlNull,
        DType::Bool(_) => L::Boolean,
        DType::Primitive(p, _) => match p {
            PType::I8 => L::Tinyint,
            PType::I16 => L::Smallint,
            PType::I32 => L::Integer,
            PType::I64 => L::Bigint,
            PType::U8 => L::UTinyint,
            PType::U16 => L::USmallint,
            PType::U32 => L::UInteger,
            PType::U64 => L::UBigint,
            PType::F32 => L::Float,
            PType::F64 => L::Double,
            _ => return Err(format!("unsupported DuckDB primitive {p}").into()),
        },
        DType::Utf8(_) => L::Varchar,
        DType::Binary(_) => L::Blob,
        DType::Decimal(d, _) => {
            if !(1..=38).contains(&d.precision())
                || d.scale() < 0
                || d.scale() as u8 > d.precision()
            {
                return Err(format!("unsupported DuckDB decimal {d}").into());
            }
            return Ok(LogicalTypeHandle::decimal(
                d.precision(),
                u8::try_from(d.scale())?,
            ));
        }
        DType::Struct(fields, _) => {
            let fields = fields
                .names()
                .iter()
                .zip(fields.fields())
                .map(|(name, dtype)| Ok((name.as_ref(), logical_type(&dtype)?)))
                .collect::<Result<Vec<_>>>()?;
            return Ok(LogicalTypeHandle::struct_type(&fields));
        }
        DType::Map(map, _) => {
            return Ok(LogicalTypeHandle::map(
                &logical_type(&map.key_dtype())?,
                &logical_type(&map.value_dtype())?,
            ));
        }
        DType::List(element, _) => return Ok(LogicalTypeHandle::list(&logical_type(element)?)),
        DType::FixedSizeList(element, size, _) => {
            return Ok(LogicalTypeHandle::array(
                &logical_type(element)?,
                u64::from(*size),
            ));
        }
        DType::Extension(ext) => match ext.metadata_opt::<AnyTemporal>() {
            Some(TemporalMetadata::Date(TimeUnit::Days)) => L::Date,
            Some(TemporalMetadata::Time(TimeUnit::Microseconds)) => L::Time,
            Some(TemporalMetadata::Time(TimeUnit::Nanoseconds)) => L::TimeNs,
            Some(TemporalMetadata::Timestamp(unit, tz)) => {
                if tz.is_some() {
                    if unit != &TimeUnit::Microseconds {
                        return Err("timezone timestamps must use microseconds".into());
                    }
                    L::TimestampTZ
                } else {
                    match unit {
                        TimeUnit::Seconds => L::TimestampS,
                        TimeUnit::Milliseconds => L::TimestampMs,
                        TimeUnit::Microseconds => L::Timestamp,
                        TimeUnit::Nanoseconds => L::TimestampNs,
                        _ => return Err("unsupported timestamp unit".into()),
                    }
                }
            }
            _ => return Err(format!("unsupported DuckDB extension {}", ext.id()).into()),
        },
        _ => return Err(format!("unsupported DuckDB dtype {dtype}").into()),
    };
    Ok(id.into())
}

pub(crate) struct ArrayExporter {
    columns: Vec<Column>,
    offset: usize,
    length: usize,
}
impl ArrayExporter {
    pub(crate) fn new(array: ArrayRef, ctx: &mut ExecutionCtx) -> Result<Self> {
        let array = array.execute::<StructArray>(ctx)?;
        let mut dictionaries = HashMap::default();
        let columns = array
            .iter_unmasked_fields()
            .map(|a| Column::new(a.clone(), ctx, false, &mut dictionaries))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            columns,
            offset: 0,
            length: array.len(),
        })
    }
    // SAFETY: output must be the live, exclusively borrowed DuckDB callback chunk.
    pub(crate) unsafe fn export(
        &mut self,
        output: ffi::duckdb_data_chunk,
        vector_size: usize,
    ) -> Result<bool> {
        if self.offset == self.length {
            return Ok(false);
        }
        let mut len = vector_size.min(self.length - self.offset);
        for col in &self.columns {
            len = col.preferred_len(self.offset, len);
        }
        if len == 0 {
            return Err("empty exporter batch before end of array".into());
        }
        // SAFETY: exclusively borrowed live callback output; reset releases the prior references.
        unsafe {
            ffi::duckdb_data_chunk_reset(output);
        }
        // SAFETY: callback output is a live chunk; never index beyond its bound columns.
        let count = unsafe { ffi::duckdb_data_chunk_get_column_count(output) };
        // DuckDB supplies an unused virtual BOOLEAN vector for a zero-column scan. Only its
        // cardinality is consumed, as in vortex-duckdb's zero-projection exporter.
        if !self.columns.is_empty() && usize::try_from(count)? != self.columns.len() {
            return Err(format!(
                "export has {} columns but DuckDB expects {count}",
                self.columns.len()
            )
            .into());
        }
        for (index, col) in self.columns.iter().enumerate() {
            // SAFETY: binding and projection establish the corresponding output column and type.
            let vector = unsafe { ffi::duckdb_data_chunk_get_vector(output, index as u64) };
            col.export(vector, self.offset, len)?;
        }
        // SAFETY: the caller lends a live output chunk with capacity for vector_size rows.
        unsafe { ffi::duckdb_data_chunk_set_size(output, len as u64) };
        self.offset += len;
        Ok(true)
    }
}

struct Column {
    mask: Mask,
    kind: Kind,
}

/// The most values a dictionary may have and still be exported as a DuckDB dictionary vector.
///
/// DuckDB sizes a dictionary child's validity mask to its standard vector size wherever it
/// rebuilds one, as TopN's boundary check does, so a code past that size indexes outside the mask
/// and fails an internal assertion. Larger dictionaries are exported flat.
fn max_dictionary_len() -> Result<usize> {
    // SAFETY: a pure query of a compile-time DuckDB constant.
    Ok(usize::try_from(unsafe { ffi::duckdb_vector_size() })?)
}

/// Stop lazy execution as soon as an encoding supported by this exporter becomes visible.
struct Exportable;
impl Matcher for Exportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Constant>()
            || array.is::<Chunked>()
            || array.is::<Dict>()
            || (array.is::<ParquetPlain>()
                && matches!(array.dtype(), DType::Primitive(PType::I16, _))))
        .then_some(array)
    }
}
enum Kind {
    Null,
    Primitive {
        bytes: ByteBuffer,
        width: usize,
        owner: Owner,
    },
    I16Page(PlainI16Page),
    Bool(BitBuffer),
    Bytes {
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        owner: Owner,
    },
    Struct(Vec<Column>),
    List {
        offsets: Vec<usize>,
        sizes: Vec<usize>,
        elements: Vector,
        elements_len: usize,
    },
    FixedList {
        size: usize,
        elements: Box<Column>,
    },
    Constant(Box<Column>),
    Dict {
        codes: Buffer<u32>,
        values: Dictionary,
    },
    Chunked {
        offsets: Vec<usize>,
        chunks: Vec<Column>,
    },
}
// Retain source arrays with address keys so an allocator cannot recycle an address during export.
// This cache lives only while constructing one batch; column owners retain the DuckDB buffers.
type DictionaryCache = HashMap<usize, (ArrayRef, Dictionary)>;

impl Column {
    fn new(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
        flatten: bool,
        dictionaries: &mut DictionaryCache,
    ) -> Result<Self> {
        let array = if flatten {
            array
        } else {
            array.execute_until::<Exportable>(ctx)?
        };
        let len = array.len();
        if !flatten {
            if let Some(page) = array.as_opt::<ParquetPlain>()
                && let Some(view) = page.data().plain_i16_view(array.dtype())?
            {
                return Ok(Self {
                    mask: Mask::new_true(len),
                    kind: Kind::I16Page(view),
                });
            }
            if let Some(a) = array.as_opt::<Constant>() {
                let one = a.array().clone().slice(0..len.min(1))?;
                return Ok(Self {
                    mask: Mask::new_true(len),
                    kind: Kind::Constant(Box::new(Self::new(one, ctx, true, dictionaries)?)),
                });
            }
            if let Some(a) = array.as_opt::<Chunked>() {
                let offsets = a.chunk_offset_values().to_vec();
                let chunks = a
                    .chunks()
                    .iter()
                    .map(|a| Self::new(a.clone(), ctx, false, dictionaries))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Self {
                    mask: Mask::new_true(len),
                    kind: Kind::Chunked { offsets, chunks },
                });
            }
            if let Some(a) = array.as_opt::<Dict>()
                && a.values().len() <= max_dictionary_len()?
                && a.codes().validity()?.execute_mask(len, ctx)?.all_true()
            {
                let key = a.values().addr();
                let values = if let Some((_, dictionary)) = dictionaries.get(&key) {
                    dictionary.try_clone()?
                } else {
                    let values = Vector::new(a.values().dtype(), a.values().len())?;
                    let exporter = Self::new(a.values().clone(), ctx, true, dictionaries)?;
                    if !a.values().is_empty() {
                        exporter.export(values.0, 0, a.values().len())?;
                    }
                    let dictionary = Dictionary::new(&values, a.values().len())?;
                    dictionaries.insert(key, (a.values().clone(), dictionary.try_clone()?));
                    dictionary
                };
                let codes = dictionary_codes(a.codes().clone(), a.values().len(), ctx)?;
                return Ok(Self {
                    mask: Mask::new_true(len),
                    kind: Kind::Dict { codes, values },
                });
            }
        }
        let canonical = array.execute::<Canonical>(ctx)?;
        let mask = canonical
            .clone()
            .into_array()
            .validity()?
            .execute_mask(len, ctx)?;
        let kind = match canonical {
            Canonical::Null(_) => Kind::Null,
            Canonical::Primitive(a) => {
                let width = a.ptype().byte_width();
                let bytes = a.buffer_handle().clone().into_host_sync();
                Kind::Primitive {
                    owner: Owner::new(bytes.clone()),
                    bytes,
                    width,
                }
            }
            Canonical::Bool(a) => Kind::Bool(a.to_bit_buffer()),
            Canonical::VarBinView(a) => {
                let VarBinViewDataParts { views, buffers, .. } = a.into_data_parts();
                let buffers: Arc<[ByteBuffer]> = buffers
                    .iter()
                    .cloned()
                    .map(|b| b.into_host_sync())
                    .collect::<Vec<_>>()
                    .into();
                Kind::Bytes {
                    views: Buffer::from_byte_buffer(views.into_host_sync()),
                    owner: Owner::new(Arc::clone(&buffers)),
                    buffers,
                }
            }
            Canonical::Decimal(a) => {
                let precision = a.decimal_dtype().precision();
                let bound = 10i128
                    .checked_pow(u32::from(precision))
                    .ok_or("unsupported decimal precision")?;
                let width = match precision {
                    1..=4 => 2,
                    5..=9 => 4,
                    10..=18 => 8,
                    19..=38 => 16,
                    _ => return Err("unsupported decimal precision".into()),
                };
                let bytes = match_each_decimal_value_type!(a.values_type(), |D| {
                    let values = a.buffer::<D>();
                    let all_valid = mask.all_true();
                    if size_of::<D>() == width {
                        // Matching storage is exported directly. Its precision fits D because the
                        // DuckDB storage width above is selected from the same precision ranges.
                        // Compare in that native width instead of widening every value to i128.
                        validate_decimal_precision(&values, precision, &mask)?;
                        values.into_byte_buffer()
                    } else {
                        let mut converted = Vec::with_capacity(len);
                        for (i, value) in values.iter().enumerate() {
                            let value = if all_valid || mask.value(i) {
                                value.to_i128().ok_or("decimal exceeds precision")?
                            } else {
                                0
                            };
                            if value <= -bound || value >= bound {
                                return Err(
                                    "Parquet decimal value exceeds its declared precision".into()
                                );
                            }
                            converted.push(value);
                        }
                        decimal_buffer(&converted, width)?
                    }
                });
                Kind::Primitive {
                    owner: Owner::new(bytes.clone()),
                    bytes,
                    width,
                }
            }
            Canonical::Struct(a) => Kind::Struct(
                a.iter_unmasked_fields()
                    .map(|a| Self::new(a.clone(), ctx, flatten, dictionaries))
                    .collect::<Result<Vec<_>>>()?,
            ),
            Canonical::List(a) => {
                let offsets = integer_values(a.offsets().clone(), ctx)?;
                let sizes = integer_values(a.sizes().clone(), ctx)?;
                let elements_len = a.elements().len();
                for (&offset, &size) in offsets.iter().zip(&sizes) {
                    if offset
                        .checked_add(size)
                        .is_none_or(|end| end > elements_len)
                    {
                        return Err("list child range out of bounds".into());
                    }
                }
                let elements = Vector::new(a.elements().dtype(), elements_len)?;
                if elements_len > 0 {
                    Self::new(a.elements().clone(), ctx, true, dictionaries)?.export(
                        elements.0,
                        0,
                        elements_len,
                    )?;
                }
                Kind::List {
                    offsets,
                    sizes,
                    elements,
                    elements_len,
                }
            }
            Canonical::FixedSizeList(a) => Kind::FixedList {
                size: a.list_size() as usize,
                elements: Box::new(Self::new(a.elements().clone(), ctx, true, dictionaries)?),
            },
            Canonical::Map(a) => return Self::new(a.entries().clone(), ctx, true, dictionaries),
            Canonical::Extension(a) => {
                return Self::new(a.storage_array().clone(), ctx, true, dictionaries);
            }
            _ => return Err("unsupported canonical DuckDB export".into()),
        };
        Ok(Self { mask, kind })
    }

    fn preferred_len(&self, offset: usize, max: usize) -> usize {
        match &self.kind {
            Kind::Chunked { offsets, chunks } => {
                let i = offsets.partition_point(|&v| v <= offset).saturating_sub(1);
                chunks[i].preferred_len(offset - offsets[i], max.min(offsets[i + 1] - offset))
            }
            Kind::Struct(fields) => fields
                .iter()
                .fold(max, |len, c| c.preferred_len(offset, len)),
            _ => max,
        }
    }

    // The caller supplies an exclusive vector of the bound type with capacity >= len.
    #[expect(
        clippy::host_endian_bytes,
        reason = "DuckDB string_t stores a native u32 length and native pointer, not serialized data"
    )]
    fn export(&self, vector: ffi::duckdb_vector, offset: usize, len: usize) -> Result<()> {
        if vector.is_null() {
            return Err("DuckDB export received a missing vector".into());
        }
        if offset
            .checked_add(len)
            .is_none_or(|end| end > self.mask.len())
        {
            return Err("export range out of bounds".into());
        }
        // SAFETY: each branch writes the physical representation corresponding to logical_type;
        // nested capacities are established before access, and external buffers retain ownership.
        unsafe {
            match &self.kind {
                Kind::Null => {}
                Kind::I16Page(page) => {
                    // ArrayExporter resets the chunk before exporting. DuckDB 2.0.0's
                    // VectorCache::ResetFromCache restores the owned, aligned output buffer,
                    // including struct children. Dictionary and constant children are flattened
                    // at construction and cannot reach this branch.
                    let output = std::slice::from_raw_parts_mut(
                        ffi::duckdb_vector_get_data(vector).cast::<MaybeUninit<i16>>(),
                        len,
                    );
                    page.decode_range_into(offset..offset + len, output)?;
                }
                Kind::Primitive {
                    bytes,
                    width,
                    owner,
                } => vp_vector_buffer(
                    vector,
                    owner.0,
                    bytes.as_ptr().add(offset * width).cast_mut().cast(),
                    len as u64,
                    *width as u64,
                ),
                Kind::Bool(bits) => {
                    let dst = std::slice::from_raw_parts_mut(
                        ffi::duckdb_vector_get_data(vector).cast::<bool>(),
                        len,
                    );
                    for (dst, value) in dst.iter_mut().zip(bits.slice(offset..offset + len).iter())
                    {
                        *dst = value;
                    }
                }
                Kind::Bytes {
                    views,
                    buffers,
                    owner,
                } => {
                    let dst = std::slice::from_raw_parts_mut(
                        ffi::duckdb_vector_get_data(vector).cast::<[u8; 16]>(),
                        len,
                    );
                    for (dst, view) in dst.iter_mut().zip(&views[offset..offset + len]) {
                        // DuckDB and Vortex inline strings have the same size/prefix/12-byte layout.
                        if view.is_inlined() {
                            *dst = std::mem::transmute::<BinaryView, [u8; 16]>(*view);
                        } else {
                            let reference = view.as_view();
                            dst[..4].copy_from_slice(&view.len().to_ne_bytes());
                            dst[4..8].copy_from_slice(&reference.prefix);
                            let ptr = buffers[reference.buffer_index as usize]
                                .as_ptr()
                                .add(reference.offset as usize);
                            dst[8..16].copy_from_slice(&(ptr as usize).to_ne_bytes());
                        }
                    }
                    vp_string_buffer(vector, owner.0);
                }
                Kind::Struct(fields) => {
                    for (i, field) in fields.iter().enumerate() {
                        field.export(
                            ffi::duckdb_struct_vector_get_child(vector, i as u64),
                            offset,
                            len,
                        )?;
                    }
                }
                Kind::List {
                    offsets,
                    sizes,
                    elements,
                    elements_len,
                } => {
                    if !vp_vector_reference(ffi::duckdb_list_vector_get_child(vector), elements.0) {
                        return Err("DuckDB rejected the list element vector type".into());
                    }
                    if ffi::duckdb_list_vector_set_size(vector, *elements_len as u64)
                        != ffi::DuckDBSuccess
                    {
                        return Err("could not set list size".into());
                    }
                    let dst = std::slice::from_raw_parts_mut(
                        ffi::duckdb_vector_get_data(vector).cast::<ffi::duckdb_list_entry>(),
                        len,
                    );
                    for (i, entry) in dst.iter_mut().enumerate() {
                        *entry = ffi::duckdb_list_entry {
                            offset: offsets[offset + i] as u64,
                            length: sizes[offset + i] as u64,
                        };
                    }
                }
                Kind::FixedList { size, elements } => elements.export(
                    ffi::duckdb_array_vector_get_child(vector),
                    offset * size,
                    len * size,
                )?,
                Kind::Constant(value) => {
                    value.export(vector, 0, 1)?;
                    vp_vector_constant(vector);
                    return Ok(());
                }
                Kind::Dict { codes, values } => {
                    let selection = ffi::duckdb_create_selection_vector(len as u64);
                    if selection.is_null() {
                        return Err("could not allocate dictionary selection".into());
                    }
                    std::ptr::copy_nonoverlapping(
                        codes[offset..offset + len].as_ptr(),
                        ffi::duckdb_selection_vector_get_data_ptr(selection),
                        len,
                    );
                    if !vp_vector_dictionary(vector, values.0, selection, len as u64) {
                        ffi::duckdb_destroy_selection_vector(selection);
                        return Err("DuckDB rejected the dictionary value vector type".into());
                    }
                    ffi::duckdb_destroy_selection_vector(selection);
                    return Ok(());
                }
                Kind::Chunked { offsets, chunks } => {
                    let i = offsets.partition_point(|&v| v <= offset).saturating_sub(1);
                    return chunks[i].export(vector, offset - offsets[i], len);
                }
            }
            if !self.mask.all_true() {
                ffi::duckdb_vector_ensure_validity_writable(vector);
                let validity = ffi::duckdb_vector_get_validity(vector);
                for i in 0..len {
                    ffi::duckdb_validity_set_row_validity(
                        validity,
                        i as u64,
                        self.mask.value(offset + i),
                    );
                }
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::unnecessary_fallible_conversions,
    reason = "the integer dispatch also includes signed and wider types"
)]
fn integer_values(array: ArrayRef, ctx: &mut ExecutionCtx) -> Result<Vec<usize>> {
    let array = array.execute::<PrimitiveArray>(ctx)?;
    vortex::array::match_each_integer_ptype!(array.ptype(), |T| {
        array
            .as_slice::<T>()
            .iter()
            .map(|v| usize::try_from(*v).map_err(Into::into))
            .collect()
    })
}

#[expect(
    clippy::unnecessary_fallible_conversions,
    clippy::useless_conversion,
    reason = "the integer dispatch includes u32, signed, and wider types"
)]
fn dictionary_codes(
    array: ArrayRef,
    dictionary_len: usize,
    ctx: &mut ExecutionCtx,
) -> Result<Buffer<u32>> {
    let array = array.execute::<PrimitiveArray>(ctx)?;
    if array.ptype() == PType::U32 {
        let codes = array.to_buffer::<u32>();
        if codes
            .iter()
            .copied()
            .max()
            .is_some_and(|code| code as usize >= dictionary_len)
        {
            return Err("dictionary code out of bounds".into());
        }
        return Ok(codes);
    }
    // Other code widths need only one conversion buffer. Check bounds in the same pass.
    vortex::array::match_each_integer_ptype!(array.ptype(), |T| {
        array
            .as_slice::<T>()
            .iter()
            .map(|&code| -> Result<u32> {
                let code = u32::try_from(code)?;
                if code as usize >= dictionary_len {
                    return Err("dictionary code out of bounds".into());
                }
                Ok(code)
            })
            .collect::<Result<Vec<_>>>()
            .map(Buffer::from)
    })
}

fn validate_decimal_precision<D: NativeDecimalType>(
    values: &[D],
    precision: u8,
    mask: &Mask,
) -> Result<()> {
    let minimum = D::MIN_BY_PRECISION
        .get(usize::from(precision))
        .ok_or("decimal precision exceeds storage precision")?;
    let maximum = D::MAX_BY_PRECISION
        .get(usize::from(precision))
        .ok_or("decimal precision exceeds storage precision")?;
    let outside = |value: &D| value < minimum || value > maximum;
    let invalid = if mask.all_true() {
        values.iter().any(outside)
    } else {
        values
            .iter()
            .enumerate()
            .any(|(i, value)| mask.value(i) && outside(value))
    };
    if invalid {
        return Err("Parquet decimal value exceeds its declared precision".into());
    }
    Ok(())
}

fn decimal_buffer(values: &[i128], width: usize) -> Result<ByteBuffer> {
    macro_rules! convert {
        ($t:ty) => {
            Buffer::<$t>::from(
                values
                    .iter()
                    .map(|v| <$t>::try_from(*v))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )
            .into_byte_buffer()
        };
    }
    Ok(match width {
        2 => convert!(i16),
        4 => convert!(i32),
        8 => convert!(i64),
        16 => Buffer::from(values.to_vec()).into_byte_buffer(),
        _ => return Err("unsupported decimal width".into()),
    })
}
