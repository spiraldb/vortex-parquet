// SPDX-License-Identifier: Apache-2.0

//! Decode PLAIN INT32 into i16 while checking each discarded half is a sign extension.

use std::arch::aarch64::vdupq_n_u16;
use std::arch::aarch64::veorq_u16;
use std::arch::aarch64::vld2q_u16;
use std::arch::aarch64::vmaxvq_u16;
use std::arch::aarch64::vorrq_u16;
use std::arch::aarch64::vreinterpretq_s16_u16;
use std::arch::aarch64::vreinterpretq_u16_s16;
use std::arch::aarch64::vshrq_n_s16;
use std::arch::aarch64::vst1q_s16;
use std::mem::MaybeUninit;

use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;

#[target_feature(enable = "neon")]
pub(super) fn i16_values(values: &[[u8; 4]]) -> (Buffer<i16>, bool) {
    let mut output = BufferMut::<i16>::with_capacity(values.len());
    let in_range = i16_values_into(values, &mut output.spare_capacity_mut()[..values.len()]);
    // SAFETY: the decoder initialized exactly values.len() slots, including invalid values.
    unsafe { output.set_len(values.len()) };
    (output.freeze(), in_range)
}

#[target_feature(enable = "neon")]
pub(super) fn i16_values_into(values: &[[u8; 4]], output: &mut [MaybeUninit<i16>]) -> bool {
    assert_eq!(values.len(), output.len());
    let (blocks, tail) = values.as_chunks::<8>();
    let mut high = vdupq_n_u16(0);
    for (block, slots) in blocks.iter().zip(output.as_chunks_mut::<8>().0) {
        // SAFETY: the block contains exactly 32 initialized bytes. LD2 separates the two
        // little-endian halves without requiring alignment. The store initializes exactly
        // the eight output slots paired with this block.
        unsafe {
            let halves = vld2q_u16(block.as_ptr().cast());
            let low = vreinterpretq_s16_u16(halves.0);
            let sign = vreinterpretq_u16_s16(vshrq_n_s16::<15>(low));
            high = vorrq_u16(high, veorq_u16(halves.1, sign));
            vst1q_s16(slots.as_mut_ptr().cast(), low);
        }
    }
    let mut invalid = i32::from(vmaxvq_u16(high));
    for (bytes, slot) in tail.iter().zip(&mut output[blocks.len() * 8..]) {
        let low = i16::from_le_bytes([bytes[0], bytes[1]]);
        invalid |= i32::from_le_bytes(*bytes) ^ i32::from(low);
        slot.write(low);
    }
    invalid == 0
}
