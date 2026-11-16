// SPDX-License-Identifier: Apache-2.0

//! Zero-copy import of the Parquet decoder's Arrow 58 buffers into Vortex's Arrow 59.

use arrow_array::Array;
use arrow_array::ffi::to_ffi;
use arrow_array_vortex::ffi::FFI_ArrowArray;
use arrow_array_vortex::ffi::FFI_ArrowSchema;
use arrow_array_vortex::ffi::from_ffi;
use arrow_array_vortex::make_array;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

pub(crate) fn import_array(array: &dyn Array) -> VortexResult<arrow_array_vortex::ArrayRef> {
    let (mut array, mut schema) =
        to_ffi(&array.to_data()).map_err(|error| vortex_err!("Arrow export failed: {error}"))?;
    // SAFETY: both versions implement the stable Arrow C Data Interface. from_raw moves ownership
    // and clears the source release callback. Import retains the buffers until the last consumer
    // drops them; the producer's release callback remains responsible for its own allocations.
    let (array, schema) = unsafe {
        (
            FFI_ArrowArray::from_raw((&raw mut array).cast()),
            FFI_ArrowSchema::from_raw((&raw mut schema).cast()),
        )
    };
    // SAFETY: the array and matching schema were exported from a valid Arrow array above.
    let data = unsafe { from_ffi(array, &schema) }?;
    Ok(make_array(data))
}

pub(crate) fn import_schema(
    schema: &arrow_schema::Schema,
) -> VortexResult<arrow_schema_vortex::Schema> {
    let mut schema = arrow_schema::ffi::FFI_ArrowSchema::try_from(schema)
        .map_err(|error| vortex_err!("Arrow schema export failed: {error}"))?;
    // SAFETY: this is an owning move between implementations of the stable ArrowSchema ABI.
    let schema = unsafe { FFI_ArrowSchema::from_raw((&raw mut schema).cast()) };
    Ok(arrow_schema_vortex::Schema::try_from(&schema)?)
}
