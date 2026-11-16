// SPDX-License-Identifier: Apache-2.0

//! Unpack a complete 56-byte block of 32 fourteen-bit dictionary codes.

use std::arch::aarch64::vaddq_u8;
use std::arch::aarch64::vandq_u32;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vdupq_n_u32;
use std::arch::aarch64::vld1q_s32;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_u32_u8;
use std::arch::aarch64::vshlq_u32;
use std::arch::aarch64::vst1q_u32;
use std::mem::MaybeUninit;

#[target_feature(enable = "neon")]
pub(super) fn unpack(input: &[u8], output: &mut [MaybeUninit<u32>; 32]) {
    assert!(input.len() >= 56, "truncated fourteen-bit block");
    // Four codes occupy seven bytes, starting at bit offsets 0, 6, 4 and 2 within
    // their first byte. Gather into u32 lanes; duplicated bytes above each code
    // are discarded by the shift and mask.
    let gather = [0u8, 1, 0, 0, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6, 0, 0];
    let shifts = [0i32, -6, -4, -2];
    // SAFETY: both arrays contain exactly 16 initialized bytes; loads need no alignment.
    let (gather, shifts) = unsafe { (vld1q_u8(gather.as_ptr()), vld1q_s32(shifts.as_ptr())) };
    let mask = vdupq_n_u32(0x3fff);
    let next = vaddq_u8(gather, vdupq_n_u8(7));
    macro_rules! block {
        ($input:literal, $output:literal, $first:expr, $second:expr) => {{
            // SAFETY: offsets 0, 14, 28 and 40 each leave at least 16 bytes in the
            // checked input. Each pair of stores initializes eight writable u32 slots.
            unsafe {
                let bytes = vld1q_u8(input.as_ptr().add($input));
                let first = vreinterpretq_u32_u8(vqtbl1q_u8(bytes, $first));
                let second = vreinterpretq_u32_u8(vqtbl1q_u8(bytes, $second));
                vst1q_u32(
                    output.as_mut_ptr().cast::<u32>().add($output),
                    vandq_u32(vshlq_u32(first, shifts), mask),
                );
                vst1q_u32(
                    output.as_mut_ptr().cast::<u32>().add($output + 4),
                    vandq_u32(vshlq_u32(second, shifts), mask),
                );
            }
        }};
    }
    block!(0, 0, gather, next);
    block!(14, 8, gather, next);
    block!(28, 16, gather, next);
    // The last eight codes start at byte 42. Load at byte 40 so the complete
    // 16-byte load stays within the block, then shift both gathers by two bytes.
    block!(
        40,
        24,
        vaddq_u8(gather, vdupq_n_u8(2)),
        vaddq_u8(next, vdupq_n_u8(2))
    );
}
