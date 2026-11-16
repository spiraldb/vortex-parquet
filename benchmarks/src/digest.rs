// SPDX-License-Identifier: Apache-2.0

//! A digest over a query's result set, so arms are compared on answers rather than row counts.
//!
//! # Why a row count is not enough
//!
//! A scalar aggregate returns exactly one row whether or not the number in it is right. ClickBench's
//! q0 is `SELECT COUNT(*)`, and several more queries return a handful of rows — so a row-count
//! comparison cannot tell right from wrong in exactly the queries where a wrong answer is most likely
//! to be *silent*.
//!
//! This matters most for changes to *which rows are read* — pruning, caching, pushdown, a new decode
//! path. Such a bug does not make a query slow, it makes rows vanish: skip a row group you should have
//! read and the sum changes while the row count does not.
//!
//! # What is and is not detected
//!
//! **Order-insensitive.** Per-row hashes are combined commutatively, so a permutation of the same rows
//! digests the same. That is deliberate: not every query in the suite has a total order, and two
//! arms returning equal-sorting rows in different orders are both correct. The cost is that an arm
//! which returns the right rows in the wrong order for an `ORDER BY` query is not caught here. Detecting
//! that would mean false disagreements on every query with ties, which would make the check worthless
//! by making it noisy.
//!
//! **Floats are rounded to [`FLOAT_SIGNIFICANT_DIGITS`].** Summing ~10^8 `f64`s in a different order
//! gives a different answer in the low bits, and the arms deliberately parallelize differently, so a
//! bitwise digest would report disagreements that are not errors. Seven significant digits sits above
//! the reassociation noise — worst-case relative error accumulating over 10^8 values is on the order of
//! 10^-9 — and far below any real defect: dropping a single row group of a hundred changes an aggregate
//! by about 1%.
//!
//! **Integers, decimals, strings and dates are exact**, which is most of what ClickBench returns.

use std::fmt;

use arrow_array::RecordBatch;
use arrow_schema::DataType;
use datafusion::arrow::util::display::ArrayFormatter;
use datafusion::arrow::util::display::FormatOptions;

/// Significant digits retained when hashing a floating-point value. See the module docs.
pub const FLOAT_SIGNIFICANT_DIGITS: usize = 7;

/// What a null formats to.
///
/// A sentinel *string* cannot be collision-free: a real value containing the same bytes digests
/// identically. So the sentinel is only what the Arrow formatter is told to print, and the actual
/// null/non-null distinction is carried by a type tag in [`hash_field`], which no value can forge.
const NULL: &str = "\u{0}null";

/// A digest of one query's result set.
///
/// Stable across runs and machines — [`fnv1a`] rather than a `DefaultHasher`, whose seed and algorithm
/// are explicitly unstable — so a digest printed in one run can be compared against another by hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultDigest(u64);

impl fmt::Display for ResultDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Short enough to sit in a report line, wide enough that a collision is not the explanation for
        // an unexpected match.
        write!(f, "{:016x}", self.0)
    }
}

/// Digests `batches`, or returns `None` if any column has a type that cannot be formatted.
///
/// `None` rather than an error, and reported as such: a type this cannot render is a gap in the check,
/// not a failure of the query, and silently digesting the columns it *does* understand would be worse
/// than admitting the gap.
pub fn digest(batches: &[RecordBatch]) -> Option<ResultDigest> {
    // Commutative, hence order-insensitive. `wrapping_add` rather than `xor` because xor cancels
    // duplicate rows in pairs, which would make a result set containing a row twice digest the same as
    // one containing it zero times.
    let mut total: u64 = 0;

    for batch in batches {
        let options = FormatOptions::default().with_null(NULL);
        let formatters: Vec<Option<ArrayFormatter<'_>>> = batch
            .columns()
            .iter()
            .map(|column| match column.data_type() {
                // Handled separately below, with rounding.
                DataType::Float16 | DataType::Float32 | DataType::Float64 => None,
                _ => Some(ArrayFormatter::try_new(column.as_ref(), &options)),
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|formatter| match formatter {
                Some(Ok(formatter)) => Some(Some(formatter)),
                Some(Err(_)) => None,
                None => Some(None),
            })
            .collect::<Option<Vec<_>>>()?;

        for row in 0..batch.num_rows() {
            let mut hash = FNV_OFFSET;
            for (idx, formatter) in formatters.iter().enumerate() {
                let column = batch.column(idx);
                let rendered = match formatter {
                    Some(formatter) => formatter.value(row).to_string(),
                    None => rounded_float(column.as_ref(), row),
                };
                hash = hash_field(hash, column.is_null(row), rendered.as_bytes());
            }
            total = total.wrapping_add(hash);
        }
    }

    Some(ResultDigest(total))
}

/// Formats a floating-point value at [`FLOAT_SIGNIFICANT_DIGITS`], via a downcast to `f64`.
fn rounded_float(column: &dyn arrow_array::Array, row: usize) -> String {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float16Type;
    use arrow_array::types::Float32Type;
    use arrow_array::types::Float64Type;

    if column.is_null(row) {
        return NULL.to_string();
    }

    let value: f64 = match column.data_type() {
        DataType::Float16 => column.as_primitive::<Float16Type>().value(row).to_f64(),
        DataType::Float32 => f64::from(column.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => column.as_primitive::<Float64Type>().value(row),
        // Unreachable: the caller only routes float columns here.
        _ => return NULL.to_string(),
    };

    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf"
        } else {
            "-inf"
        }
        .to_string();
    }
    // Collapses `-0.0` onto `0.0`, which are equal but format differently.
    if value == 0.0 {
        return "0".to_string();
    }
    // Scientific notation, so the digits retained are *significant* digits rather than decimal places —
    // otherwise the tolerance would depend on the magnitude of the value.
    format!("{:.*e}", FLOAT_SIGNIFICANT_DIGITS - 1, value)
}

/// Mixes one field into a row's hash, unambiguously.
///
/// # Why a separator byte is not enough
///
/// The first version appended each field's rendering followed by `0x1f`. That is ambiguous: a value
/// may itself contain `0x1f`, so `("a\x1fb", "c")` and `("a", "b\x1fc")` produce the same byte
/// stream and the same digest. A benchmark that compares answers by digest would call those equal.
///
/// So the length is hashed before the bytes, which makes the encoding prefix-free: no arrangement of
/// field boundaries can produce the same stream as a different arrangement. Null-ness is a separate
/// tag rather than a reserved string, so no value can impersonate a null either.
fn hash_field(mut hash: u64, is_null: bool, bytes: &[u8]) -> u64 {
    hash = fnv1a(hash, &[u8::from(is_null)]);
    hash = fnv1a(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a(hash, bytes)
}

/// FNV-1a offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a, chosen for being short, deterministic, and specified — not for collision resistance.
///
/// This compares results within and across runs of a benchmark; it is not a security boundary.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Float64Array;
    use arrow_array::Int64Array;
    use arrow_array::StringArray;

    use super::*;

    /// One batch from parallel columns, for brevity below.
    fn batch(ids: Vec<Option<i64>>, values: Vec<Option<f64>>) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int64Array::from(ids)) as _),
            ("value", Arc::new(Float64Array::from(values)) as _),
        ])
        .expect("the fixture batch is well-formed")
    }

    /// A different value must give a different digest. Without this the check is decorative.
    #[test]
    fn a_changed_value_changes_the_digest() {
        let a = digest(&[batch(vec![Some(1), Some(2)], vec![Some(1.5), Some(2.5)])]);
        let b = digest(&[batch(vec![Some(1), Some(2)], vec![Some(1.5), Some(2.6)])]);
        assert!(a.is_some() && b.is_some());
        assert_ne!(a, b, "a changed aggregate must change the digest");
    }

    /// The row count is exactly what this is meant to stop being the check.
    #[test]
    fn a_scalar_aggregate_with_a_wrong_value_is_caught() {
        // Both are one row, so a row-count comparison cannot tell them apart at all.
        let right = digest(&[batch(vec![None], vec![Some(1_234_567.0)])]);
        let wrong = digest(&[batch(vec![None], vec![Some(1_234_000.0)])]);
        assert_ne!(right, wrong);
    }

    /// Row order must not matter: arms may return equal-sorting rows in either order.
    #[test]
    fn row_order_does_not_change_the_digest() {
        let forward = digest(&[batch(vec![Some(1), Some(2)], vec![Some(1.5), Some(2.5)])]);
        let reverse = digest(&[batch(vec![Some(2), Some(1)], vec![Some(2.5), Some(1.5)])]);
        assert_eq!(forward, reverse);
    }

    /// Batching must not matter either, since arms choose their own batch sizes.
    #[test]
    fn batch_boundaries_do_not_change_the_digest() {
        let one = digest(&[batch(vec![Some(1), Some(2)], vec![Some(1.5), Some(2.5)])]);
        let two = digest(&[
            batch(vec![Some(1)], vec![Some(1.5)]),
            batch(vec![Some(2)], vec![Some(2.5)]),
        ]);
        assert_eq!(one, two);
    }

    /// A duplicated row must not cancel, which is why the combination is addition and not xor.
    #[test]
    fn a_duplicated_row_does_not_cancel() {
        let twice = digest(&[batch(vec![Some(1), Some(1)], vec![Some(1.5), Some(1.5)])]);
        let none = digest(&[batch(vec![], vec![])]);
        assert_ne!(twice, none);
    }

    /// Float reassociation noise must not read as a disagreement, or the check is too noisy to use.
    #[test]
    fn float_reassociation_noise_is_tolerated() {
        let value = 1.234_567_890_123e12;
        // A relative perturbation of ~1e-12, the order of magnitude two different summation orders
        // over a large column produce.
        let jittered = value * (1.0 + 1e-12);
        assert_ne!(value, jittered, "the fixture must actually differ in bits");
        assert_eq!(
            digest(&[batch(vec![None], vec![Some(value)])]),
            digest(&[batch(vec![None], vec![Some(jittered)])]),
            "last-bit float differences must not be reported as disagreements"
        );
    }

    /// But a difference above the tolerance must still be caught, or the tolerance is a blindfold.
    #[test]
    fn a_float_difference_above_the_tolerance_is_caught() {
        let value = 1.234_567_890_123e12;
        // 1e-5 relative: far below a dropped row group, far above reassociation noise.
        let wrong = value * (1.0 + 1e-5);
        assert_ne!(
            digest(&[batch(vec![None], vec![Some(value)])]),
            digest(&[batch(vec![None], vec![Some(wrong)])])
        );
    }

    /// Nulls must be distinguishable from values, including from a string that looks like one.
    #[test]
    fn a_null_is_not_the_string_null() {
        let with_null = RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vec![None::<&str>])) as _,
        )])
        .unwrap();
        let with_text = RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vec![Some("null")])) as _,
        )])
        .unwrap();
        assert_ne!(digest(&[with_null]), digest(&[with_text]));
    }

    /// A value containing the separator byte must not be able to forge a field boundary.
    ///
    /// The separator-only encoding this replaced digested these two identically.
    #[test]
    fn a_value_containing_the_separator_cannot_forge_a_boundary() {
        let left = RecordBatch::try_from_iter(vec![
            (
                "a",
                Arc::new(StringArray::from(vec![Some("a\u{1f}b")])) as _,
            ),
            ("b", Arc::new(StringArray::from(vec![Some("c")])) as _),
        ])
        .unwrap();
        let right = RecordBatch::try_from_iter(vec![
            ("a", Arc::new(StringArray::from(vec![Some("a")])) as _),
            (
                "b",
                Arc::new(StringArray::from(vec![Some("b\u{1f}c")])) as _,
            ),
        ])
        .unwrap();
        assert_ne!(digest(&[left]), digest(&[right]));
    }

    /// A real string equal to the null sentinel must not digest as a null.
    #[test]
    fn a_value_cannot_impersonate_a_null() {
        let with_null = RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vec![None::<&str>])) as _,
        )])
        .unwrap();
        let forged = RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vec![Some("\u{0}null")])) as _,
        )])
        .unwrap();
        assert_ne!(digest(&[with_null]), digest(&[forged]));
    }

    /// Column boundaries must be respected, or ("a","bc") and ("ab","c") collide.
    #[test]
    fn column_boundaries_are_respected() {
        let left = RecordBatch::try_from_iter(vec![
            ("a", Arc::new(StringArray::from(vec![Some("a")])) as _),
            ("b", Arc::new(StringArray::from(vec![Some("bc")])) as _),
        ])
        .unwrap();
        let right = RecordBatch::try_from_iter(vec![
            ("a", Arc::new(StringArray::from(vec![Some("ab")])) as _),
            ("b", Arc::new(StringArray::from(vec![Some("c")])) as _),
        ])
        .unwrap();
        assert_ne!(digest(&[left]), digest(&[right]));
    }
}
