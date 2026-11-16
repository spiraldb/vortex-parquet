// SPDX-License-Identifier: Apache-2.0

//! Compare little-endian PLAIN integers, collecting physical bounds for narrow logical types.

use std::arch::aarch64::int32x4_t;
use std::arch::aarch64::int64x2_t;
use std::arch::aarch64::uint32x4_t;
use std::arch::aarch64::uint64x2_t;
use std::arch::aarch64::vaddv_u8;
use std::arch::aarch64::vand_u8;
use std::arch::aarch64::vandq_u8;
use std::arch::aarch64::vceqq_s32;
use std::arch::aarch64::vceqq_s64;
use std::arch::aarch64::vcgeq_s32;
use std::arch::aarch64::vcgeq_s64;
use std::arch::aarch64::vcgtq_s32;
use std::arch::aarch64::vcgtq_s64;
use std::arch::aarch64::vcleq_s32;
use std::arch::aarch64::vcleq_s64;
use std::arch::aarch64::vcltq_s32;
use std::arch::aarch64::vcltq_s64;
use std::arch::aarch64::vcombine_u8;
use std::arch::aarch64::vcombine_u16;
use std::arch::aarch64::vcombine_u32;
use std::arch::aarch64::vdupq_n_s32;
use std::arch::aarch64::vdupq_n_s64;
use std::arch::aarch64::vdupq_n_u64;
use std::arch::aarch64::veorq_u64;
use std::arch::aarch64::vget_high_u8;
use std::arch::aarch64::vget_low_u8;
use std::arch::aarch64::vld1_u8;
use std::arch::aarch64::vld1q_s32;
use std::arch::aarch64::vld1q_s64;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vmaxq_s32;
use std::arch::aarch64::vmaxvq_s32;
use std::arch::aarch64::vminq_s32;
use std::arch::aarch64::vminvq_s32;
use std::arch::aarch64::vmovn_u16;
use std::arch::aarch64::vmovn_u32;
use std::arch::aarch64::vmovn_u64;
use std::arch::aarch64::vmvnq_u32;

use vortex::buffer::BitBuffer;
use vortex::buffer::BufferMut;
use vortex::scalar_fn::fns::operators::CompareOperator;

#[target_feature(enable = "neon")]
pub(super) fn compare(
    values: &[[u8; 4]],
    constant: i32,
    operator: CompareOperator,
) -> (BitBuffer, i32, i32) {
    macro_rules! apply {
        ($vector:expr, $op:tt) => {
            compare_values(values, constant, $vector, |value| value $op constant)
        };
    }
    match operator {
        CompareOperator::Eq => apply!(|a, b| vceqq_s32(a, b), ==),
        CompareOperator::NotEq => apply!(|a, b| vmvnq_u32(vceqq_s32(a, b)), !=),
        CompareOperator::Gt => apply!(|a, b| vcgtq_s32(a, b), >),
        CompareOperator::Gte => apply!(|a, b| vcgeq_s32(a, b), >=),
        CompareOperator::Lt => apply!(|a, b| vcltq_s32(a, b), <),
        CompareOperator::Lte => apply!(|a, b| vcleq_s32(a, b), <=),
    }
}

#[target_feature(enable = "neon")]
fn compare_values(
    values: &[[u8; 4]],
    constant: i32,
    vector_compare: impl Fn(int32x4_t, int32x4_t) -> uint32x4_t,
    scalar_compare: impl Fn(i32) -> bool,
) -> (BitBuffer, i32, i32) {
    let mut output = BufferMut::<u8>::with_capacity(values.len().div_ceil(8));
    let (blocks, tail) = values.as_chunks::<16>();
    let rhs = vdupq_n_s32(constant);
    let mut min = vdupq_n_s32(0);
    let mut max = vdupq_n_s32(0);
    let weights = [1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: this loads exactly the 16 initialized bytes of weights; no alignment is required.
    let weights = unsafe { vld1q_u8(weights.as_ptr()) };
    for (block, selected_bytes) in blocks.iter().zip(
        output
            .spare_capacity_mut()
            .as_chunks_mut::<2>()
            .0
            .iter_mut(),
    ) {
        // SAFETY: each block contains 16 initialized four-byte values. Each unaligned load reads
        // four values wholly within that block. This module is only built for little-endian NEON.
        let (first, second, third, fourth) = unsafe {
            (
                vld1q_s32(block.as_ptr().cast()),
                vld1q_s32(block.as_ptr().add(4).cast()),
                vld1q_s32(block.as_ptr().add(8).cast()),
                vld1q_s32(block.as_ptr().add(12).cast()),
            )
        };
        min = vminq_s32(
            min,
            vminq_s32(vminq_s32(first, second), vminq_s32(third, fourth)),
        );
        max = vmaxq_s32(
            max,
            vmaxq_s32(vmaxq_s32(first, second), vmaxq_s32(third, fourth)),
        );
        let ab = vcombine_u16(
            vmovn_u32(vector_compare(first, rhs)),
            vmovn_u32(vector_compare(second, rhs)),
        );
        let cd = vcombine_u16(
            vmovn_u32(vector_compare(third, rhs)),
            vmovn_u32(vector_compare(fourth, rhs)),
        );
        let bytes = vcombine_u8(vmovn_u16(ab), vmovn_u16(cd));
        let selected = vandq_u8(bytes, weights);
        selected_bytes[0].write(vaddv_u8(vget_low_u8(selected)));
        selected_bytes[1].write(vaddv_u8(vget_high_u8(selected)));
    }
    let mut min = vminvq_s32(min);
    let mut max = vmaxvq_s32(max);
    if !tail.is_empty() {
        let mut bits = 0u16;
        for (index, bytes) in tail.iter().enumerate() {
            let value = i32::from_le_bytes(*bytes);
            min = min.min(value);
            max = max.max(value);
            bits |= u16::from(scalar_compare(value)) << index;
        }
        for (slot, byte) in output.spare_capacity_mut()[blocks.len() * 2..]
            .iter_mut()
            .zip(&bits.to_le_bytes()[..tail.len().div_ceil(8)])
        {
            slot.write(*byte);
        }
    }
    // SAFETY: capacity is at least ceil(values.len()/8). The complete blocks initialized
    // two bytes each, and the tail initialized its ceil(tail.len()/8) bytes immediately after
    // them. Together these cover exactly the length being exposed, including empty input.
    unsafe { output.set_len(values.len().div_ceil(8)) };
    (BitBuffer::new(output.freeze(), values.len()), min, max)
}

#[target_feature(enable = "neon")]
pub(super) fn compare_i64(
    values: &[[u8; 8]],
    constant: i64,
    operator: CompareOperator,
) -> BitBuffer {
    macro_rules! apply {
        ($vector:expr, $op:tt) => {
            compare_i64_values(values, constant, $vector, |value| value $op constant)
        };
    }
    match operator {
        CompareOperator::Eq => apply!(|a, b| vceqq_s64(a, b), ==),
        CompareOperator::NotEq => {
            apply!(|a, b| veorq_u64(vceqq_s64(a, b), vdupq_n_u64(u64::MAX)), !=)
        }
        CompareOperator::Gt => apply!(|a, b| vcgtq_s64(a, b), >),
        CompareOperator::Gte => apply!(|a, b| vcgeq_s64(a, b), >=),
        CompareOperator::Lt => apply!(|a, b| vcltq_s64(a, b), <),
        CompareOperator::Lte => apply!(|a, b| vcleq_s64(a, b), <=),
    }
}

#[target_feature(enable = "neon")]
fn compare_i64_values(
    values: &[[u8; 8]],
    constant: i64,
    vector_compare: impl Fn(int64x2_t, int64x2_t) -> uint64x2_t,
    scalar_compare: impl Fn(i64) -> bool,
) -> BitBuffer {
    let mut output = BufferMut::<u8>::with_capacity(values.len().div_ceil(8));
    let (blocks, tail) = values.as_chunks::<8>();
    let rhs = vdupq_n_s64(constant);
    let weights = [1u8, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: load the eight initialized weight bytes, with no alignment requirement.
    let weights = unsafe { vld1_u8(weights.as_ptr()) };
    for (block, slot) in blocks.iter().zip(output.spare_capacity_mut()) {
        // SAFETY: each unaligned load reads two complete i64 values inside this eight-value
        // block. The module is compiled only for little-endian NEON targets.
        let (first, second, third, fourth) = unsafe {
            (
                vld1q_s64(block.as_ptr().cast()),
                vld1q_s64(block.as_ptr().add(2).cast()),
                vld1q_s64(block.as_ptr().add(4).cast()),
                vld1q_s64(block.as_ptr().add(6).cast()),
            )
        };
        let ab = vcombine_u32(
            vmovn_u64(vector_compare(first, rhs)),
            vmovn_u64(vector_compare(second, rhs)),
        );
        let cd = vcombine_u32(
            vmovn_u64(vector_compare(third, rhs)),
            vmovn_u64(vector_compare(fourth, rhs)),
        );
        let bytes = vmovn_u16(vcombine_u16(vmovn_u32(ab), vmovn_u32(cd)));
        slot.write(vaddv_u8(vand_u8(bytes, weights)));
    }
    if !tail.is_empty() {
        let mut bits = 0u8;
        for (index, bytes) in tail.iter().enumerate() {
            bits |= u8::from(scalar_compare(i64::from_le_bytes(*bytes))) << index;
        }
        output.spare_capacity_mut()[blocks.len()].write(bits);
    }
    // SAFETY: every complete block initializes one byte, and a nonempty tail initializes
    // exactly one further byte. This covers precisely ceil(values.len()/8) output bytes.
    unsafe { output.set_len(values.len().div_ceil(8)) };
    BitBuffer::new(output.freeze(), values.len())
}
