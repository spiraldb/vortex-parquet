// SPDX-License-Identifier: Apache-2.0
//! Use the privately built upstream Snappy decoder for flat Parquet pages.

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

unsafe extern "C" {
    fn vp_snappy_decode(
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: usize,
    ) -> bool;
}

pub(crate) fn decode(
    input: &[u8],
    output: &mut Vec<u8>,
    expected_length: usize,
) -> VortexResult<()> {
    let start = output.len();
    output.reserve(expected_length);
    // SAFETY: reserve guarantees enough space after the initialized prefix. The compressed
    // input cannot alias this mutable allocation. The shim checks the decoded length before
    // writing and Snappy bounds-checks malformed input. Failure leaves the Vec length unchanged.
    vortex_ensure!(
        unsafe {
            vp_snappy_decode(
                input.as_ptr(),
                input.len(),
                output.spare_capacity_mut().as_mut_ptr().cast(),
                expected_length,
            )
        },
        "invalid Snappy page or decoded-length mismatch"
    );
    // SAFETY: a successful raw decode initialized exactly expected_length bytes after start.
    // The prefix remains initialized, and reserve ensured the resulting length fits capacity.
    unsafe { output.set_len(start + expected_length) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn native_decoder_checks_lengths_and_rejects_corrupt_streams() {
        // Independent Snappy wire fixtures: length=3, followed by a three-byte literal,
        // or a one-byte literal and a two-byte backreference at distance one.
        for (wire, expected) in [
            (&[3, 8, b'a', b'b', b'c'][..], b"abc"),
            (&[3, 0, b'a', 6, 1, 0][..], b"aaa"),
        ] {
            for prefix in [&[][..], &[0x7f, 0x23, 0x91][..]] {
                let mut output = prefix.to_vec();
                assert!(decode(wire, &mut output, expected.len()).is_ok());
                assert_eq!(&output[..prefix.len()], prefix);
                assert_eq!(&output[prefix.len()..], expected);
                for length in [2, 4] {
                    let mut wrong_size = prefix.to_vec();
                    assert!(decode(wire, &mut wrong_size, length).is_err());
                    assert_eq!(wrong_size, prefix);
                }
            }
        }
        for wire in [&[][..], &[3, 8, b'a'][..], &[3, 2, 0, 0][..]] {
            let mut output = vec![0x7f, 0x23, 0x91];
            assert!(decode(wire, &mut output, 3).is_err());
            assert_eq!(output, [0x7f, 0x23, 0x91]);
        }
        let mut empty = Vec::new();
        assert!(decode(&[0], &mut empty, 0).is_ok());
        assert!(empty.is_empty());
    }
}
