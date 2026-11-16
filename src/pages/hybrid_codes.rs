// SPDX-License-Identifier: Apache-2.0

//! Decode Parquet dictionary indices, including short bit-packed runs, in blocks.

use std::mem::MaybeUninit;

use bytes::Bytes;
use parquet::encodings::rle::RleDecoder;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
mod neon12;

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "neon"
))]
mod neon14;

pub(super) fn decode(input: Bytes, bit_width: u8, count: usize) -> VortexResult<Vec<u32>> {
    vortex_ensure!(bit_width <= 32, "invalid dictionary bit width {bit_width}");
    let mut output = Vec::with_capacity(count);
    if decode_runs(
        &input,
        usize::from(bit_width),
        &mut output.spare_capacity_mut()[..count],
    )
    .is_some()
    {
        // SAFETY: decode_runs returns Some only after writing every requested slot.
        unsafe { output.set_len(count) };
    } else {
        // The fast path may have written a prefix. Initialize the entire allocation before
        // lending an ordinary slice to parquet-rs; no uninitialized values are exposed.
        output.resize(count, 0);
        // Some writers omit the final bit-packed group's unused padding. Keep parquet-rs's
        // compatibility and malformed-stream handling whenever the complete run is unavailable.
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(input).map_err(|error| {
            vortex_err!("failed to initialize dictionary code decoder: {error}")
        })?;
        let decoded = decoder
            .get_batch(&mut output)
            .map_err(|error| vortex_err!("failed to decode dictionary codes: {error}"))?;
        vortex_ensure!(
            decoded == count,
            "decoded {decoded} of {count} dictionary codes"
        );
    }
    Ok(output)
}

// Success guarantees that every output slot has been initialized. Failure may leave a prefix.
fn decode_runs(mut input: &[u8], width: usize, mut output: &mut [MaybeUninit<u32>]) -> Option<()> {
    if output.is_empty() {
        return None;
    }
    while !output.is_empty() {
        let header = read_header(&mut input)?;
        let run = usize::try_from(header >> 1).ok()?;
        if run == 0 {
            return None;
        }
        if header & 1 == 0 {
            let bytes = width.div_ceil(8);
            let value = input.get(..bytes)?;
            let value = match bytes {
                0 => 0,
                1 => u32::from(value[0]),
                2 => u32::from(u16::from_le_bytes(value.try_into().ok()?)),
                3 => u32::from(value[0]) | (u32::from(value[1]) << 8) | (u32::from(value[2]) << 16),
                4 => u32::from_le_bytes(value.try_into().ok()?),
                _ => return None,
            };
            input = &input[bytes..];
            let count = run.min(output.len());
            let (values, rest) = output.split_at_mut(count);
            values.fill(MaybeUninit::new(value));
            output = rest;
        } else {
            let bytes = run.checked_mul(width)?;
            let mut packed = input.get(..bytes)?;
            input = &input[bytes..];
            let count = run.checked_mul(8)?;
            if count > i32::MAX as usize {
                return None;
            }
            let count = count.min(output.len());
            let (values, rest) = output.split_at_mut(count);
            let (blocks, tail) = values.as_chunks_mut::<32>();
            for block in blocks {
                unpack_block(packed, block, width);
                packed = &packed[width * 4..];
            }
            if !tail.is_empty() {
                let bytes = (tail.len() * width).div_ceil(8);
                let mut padded = [0u8; 128];
                padded[..bytes].copy_from_slice(&packed[..bytes]);
                let mut decoded = [MaybeUninit::uninit(); 32];
                unpack_block(&padded, &mut decoded, width);
                tail.copy_from_slice(&decoded[..tail.len()]);
            }
            output = rest;
        }
    }
    Some(())
}

fn read_header(input: &mut &[u8]) -> Option<u32> {
    let first = *input.first()?;
    *input = &input[1..];
    if first < 128 {
        return Some(u32::from(first));
    }
    let mut value = u32::from(first & 127);
    for shift in [7, 14, 21, 28] {
        let byte = *input.first()?;
        *input = &input[1..];
        if shift == 28 && byte > 15 {
            return None;
        }
        value |= u32::from(byte & 127) << shift;
        if byte < 128 {
            return Some(value);
        }
    }
    None
}

// Every successful call initializes all 32 output slots.
// Compile each bit width into fixed shifts and word loads. A complete 32-code block occupies
// exactly width * 4 bytes; callers validate that range or supply a zero-padded tail buffer.
fn unpack_block(input: &[u8], output: &mut [MaybeUninit<u32>; 32], width: usize) {
    macro_rules! dispatch {
        ($($width:literal),*) => {
            match width {
                $($width => unpack::<$width>(input, output),)*
                _ => unreachable!("dictionary bit width was validated"),
            }
        };
    }
    dispatch!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    );
}

fn unpack<const WIDTH: usize>(input: &[u8], output: &mut [MaybeUninit<u32>; 32]) {
    assert!(input.len() >= WIDTH * 4, "truncated bit-packed block");
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    if WIDTH == 12 {
        // SAFETY: this branch is compiled only for little-endian aarch64 with NEON.
        unsafe { neon12::unpack(input, output) };
        return;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "neon"
    ))]
    if WIDTH == 14 {
        // SAFETY: this branch is compiled only for little-endian aarch64 with NEON.
        unsafe { neon14::unpack(input, output) };
        return;
    }
    if WIDTH == 0 {
        output.fill(MaybeUninit::new(0));
        return;
    }
    macro_rules! code {
        ($($index:literal),*) => { $(
            output[$index].write(unpack_one::<WIDTH, $index>(input));
        )* };
    }
    code!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31
    );
}

#[inline(always)]
fn unpack_one<const WIDTH: usize, const INDEX: usize>(input: &[u8]) -> u32 {
    let word = |index: usize| {
        u32::from_le_bytes([
            input[index * 4],
            input[index * 4 + 1],
            input[index * 4 + 2],
            input[index * 4 + 3],
        ])
    };
    let start = INDEX * WIDTH;
    let shift = start % 32;
    let mut value = word(start / 32) >> shift;
    if shift + WIDTH > 32 {
        value |= word(start / 32 + 1) << (32 - shift);
    }
    value & (u32::MAX >> (32 - WIDTH))
}

#[cfg(test)]
mod tests {
    use parquet::encodings::rle::RleEncoder;

    use super::decode;

    #[test]
    fn dictionary_indices_follow_the_hybrid_wire_format() {
        // The Parquet specification's three-bit example: 0 through 7, LSB-packed.
        assert_eq!(
            decode(vec![3, 0x88, 0xc6, 0xfa].into(), 3, 8).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
        // A five-byte run header, with only the requested prefix materialized.
        assert_eq!(
            decode(
                vec![0xfe, 0xff, 0xff, 0xff, 0x0f, 0xff, 0xff, 0xff, 0xff].into(),
                32,
                1
            )
            .unwrap(),
            vec![u32::MAX]
        );
        // Valid prefix of an unpadded final group; requesting a missing value must fail.
        assert_eq!(
            decode(vec![3, 0x88, 0xc6].into(), 3, 5).unwrap(),
            vec![0, 1, 2, 3, 4]
        );
        assert!(decode(vec![3, 0x88, 0xc6].into(), 3, 6).is_err());
        // Fall back after a complete RLE prefix has already filled part of the output.
        assert_eq!(
            decode(vec![4, 7, 3, 0x88, 0xc6].into(), 3, 7).unwrap(),
            vec![7, 7, 0, 1, 2, 3, 4]
        );
        assert!(decode(vec![4, 7, 3, 0x88, 0xc6].into(), 3, 8).is_err());
        assert!(decode(vec![0x80].into(), 3, 1).is_err());
        assert!(decode(vec![2].into(), 8, 1).is_err());
        assert!(decode(vec![2, 0].into(), 33, 1).is_err());

        // An independent writer covers all widths, RLE/packed transitions, multi-byte run
        // headers, block boundaries, and page counts ending before the final padding.
        for width in 0..=32u8 {
            let mask = u32::MAX.checked_shr(32 - u32::from(width)).unwrap_or(0);
            let expected = (0..513u32)
                .map(|i| {
                    if i < 73 {
                        mask
                    } else if i < 207 {
                        0
                    } else {
                        i.wrapping_mul(0x9e37_79b9) & mask
                    }
                })
                .collect::<Vec<_>>();
            let mut encoder = RleEncoder::new(width, 4096);
            for &value in &expected {
                encoder.put(u64::from(value));
            }
            let bytes = bytes::Bytes::from(encoder.consume());
            for count in [0, 1, 31, 32, 33, 72, 73, 74, 206, 207, 208, 511, 512, 513] {
                assert_eq!(
                    decode(bytes.clone(), width, count).unwrap(),
                    expected[..count],
                    "width={width}, count={count}"
                );
            }
        }
    }
}
