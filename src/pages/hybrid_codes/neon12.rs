// SPDX-License-Identifier: Apache-2.0

//! Unpack a complete 48-byte block of 32 twelve-bit dictionary codes.

use std::arch::aarch64::vaddq_u8;
use std::arch::aarch64::vandq_u16;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vdupq_n_u16;
use std::arch::aarch64::vget_low_u16;
use std::arch::aarch64::vld1q_s16;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vmovl_high_u16;
use std::arch::aarch64::vmovl_u16;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_u16_u8;
use std::arch::aarch64::vshlq_u16;
use std::arch::aarch64::vst1q_u32;
use std::mem::MaybeUninit;

#[target_feature(enable = "neon")]
pub(super) fn unpack(input: &[u8], output: &mut [MaybeUninit<u32>; 32]) {
    assert!(input.len() >= 48, "truncated twelve-bit block");
    // Each pair of codes occupies three bytes. Gather overlapping two-byte windows,
    // shift the odd windows by four bits, and discard the following code's high bits.
    let gather = [0u8, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];
    let shifts = [0i16, -4, 0, -4, 0, -4, 0, -4];
    // SAFETY: both arrays contain exactly 16 initialized bytes; loads need no alignment.
    let (gather, shifts) = unsafe { (vld1q_u8(gather.as_ptr()), vld1q_s16(shifts.as_ptr())) };
    let mask = vdupq_n_u16(0x0fff);
    macro_rules! block {
        ($input:literal, $output:literal, $gather:expr) => {{
            // SAFETY: input offsets 0, 12, 24 and 32 each leave at least 16 bytes in the
            // checked 48-byte block. Each output offset leaves eight writable u32 slots;
            // the two stores initialize exactly those slots, without requiring alignment.
            unsafe {
                let bytes = vld1q_u8(input.as_ptr().add($input));
                let windows = vreinterpretq_u16_u8(vqtbl1q_u8(bytes, $gather));
                let codes = vandq_u16(vshlq_u16(windows, shifts), mask);
                vst1q_u32(
                    output.as_mut_ptr().cast::<u32>().add($output),
                    vmovl_u16(vget_low_u16(codes)),
                );
                vst1q_u32(
                    output.as_mut_ptr().cast::<u32>().add($output + 4),
                    vmovl_high_u16(codes),
                );
            }
        }};
    }
    block!(0, 0, gather);
    block!(12, 8, gather);
    block!(24, 16, gather);
    // The final group starts at byte 36. Load from byte 32 and offset the gather by four
    // rather than reading four bytes beyond the complete block.
    block!(32, 24, vaddq_u8(gather, vdupq_n_u8(4)));
}
