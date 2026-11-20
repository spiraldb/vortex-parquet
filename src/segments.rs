// SPDX-License-Identifier: Apache-2.0

//! Mapping Parquet byte ranges onto Vortex's segment machinery.
//!
//! A Vortex file's footer lists its segments — `(offset, length, alignment)` triples — and the
//! reader addresses data by [`SegmentId`], an index into that list. Nothing about that indirection
//! is Vortex-specific: it is just "a table of byte ranges in a file", and Parquet's column chunks
//! are exactly that. So the table can be built from a Parquet footer and handed to the same
//! coalescing I/O driver Vortex uses for its own files.
//!
//! That is what [`SegmentTable`] does. Register the byte ranges the reader will want, keep the
//! returned [`SegmentId`]s in the layout, and hand [`SegmentTable::locations`] to the V2 scan's
//! [`ScanFile`](vortex::layout::scan::v2::ScanFile). The session's I/O scheduler then handles
//! request coalescing, deduplication, retention and alignment for free.

use std::ops::Range;

use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::segments::SegmentId;
use vortex::io::segments::SegmentLocation;

/// A builder for the byte ranges this reader intends to read from a Parquet file.
///
/// Byte ranges are registered up front — while parsing the footer — and each registration returns
/// the [`SegmentId`] the layout should store. Ranges may overlap and need not be sorted; the I/O
/// driver sorts and coalesces at request time.
#[derive(Clone, Debug, Default)]
pub struct SegmentTable {
    locations: Vec<SegmentLocation>,
}

impl SegmentTable {
    /// Creates an empty segment table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty segment table with space for `capacity` segments.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            locations: Vec::with_capacity(capacity),
        }
    }

    /// Registers an unaligned byte range, returning its [`SegmentId`].
    ///
    /// Unaligned is right for compressed Parquet pages, which are parsed byte-wise rather than
    /// reinterpreted as a typed slice.
    pub fn push(&mut self, byte_range: Range<u64>) -> VortexResult<SegmentId> {
        self.push_aligned(byte_range, Alignment::none())
    }

    /// Registers a byte range that must be delivered at `alignment`, returning its [`SegmentId`].
    ///
    /// Alignment matters when the bytes will be reinterpreted as a typed slice without a copy — a
    /// zero-copy path from an uncompressed, plain-encoded Parquet buffer into a Vortex primitive
    /// array, say. Requesting alignment that is not needed costs a memcpy in the driver, so prefer
    /// [`push`](Self::push) unless the zero-copy path is real.
    pub fn push_aligned(
        &mut self,
        byte_range: Range<u64>,
        alignment: Alignment,
    ) -> VortexResult<SegmentId> {
        if byte_range.end < byte_range.start {
            vortex_bail!(
                "SegmentTable: inverted byte range {}..{}",
                byte_range.start,
                byte_range.end
            );
        }

        // `SegmentLocation` stores lengths as `u32`, which caps a single segment at 4GiB. That is not a
        // limit worth working around: a segment is the unit of a single coalesced read, and wanting
        // more than 4GiB in one read should mean splitting it into several segments so the driver
        // can pipeline them.
        let length = u32::try_from(byte_range.end - byte_range.start).map_err(|_| {
            vortex_err!(
                "SegmentTable: segment {}..{} exceeds the {}-byte maximum segment length; \
                 split it into multiple segments",
                byte_range.start,
                byte_range.end,
                u32::MAX
            )
        })?;

        let id = SegmentId::try_from(self.locations.len())?;
        self.locations.push(SegmentLocation {
            offset: byte_range.start,
            length,
            alignment,
        });
        Ok(id)
    }

    /// Returns the number of registered segments.
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Returns `true` if no segments have been registered.
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Returns the byte range registered for `id`, or `None` if `id` is not in this table.
    pub fn byte_range(&self, id: SegmentId) -> Option<Range<u64>> {
        self.locations
            .get(usize::try_from(*id).ok()?)
            .map(|location| location.offset..location.offset + u64::from(location.length))
    }

    /// Returns the total number of bytes spanned by the registered segments, counting overlaps
    /// once per segment.
    pub fn total_bytes(&self) -> u64 {
        self.locations
            .iter()
            .map(|location| u64::from(location.length))
            .sum::<u64>()
    }

    /// Returns the registered locations, in [`SegmentId`] order.
    pub fn locations(&self) -> &[SegmentLocation] {
        &self.locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_assigned_in_registration_order() {
        let mut table = SegmentTable::new();
        let a = table.push(0..10).unwrap();
        let b = table.push(10..30).unwrap();

        assert_eq!(table.byte_range(a), Some(0..10));
        assert_eq!(table.byte_range(b), Some(10..30));
        assert_eq!(table.len(), 2);
        assert_eq!(table.total_bytes(), 30);
    }

    #[test]
    fn overlapping_and_unsorted_ranges_are_allowed() {
        let mut table = SegmentTable::new();
        let late = table.push(100..200).unwrap();
        let early = table.push(0..150).unwrap();

        assert_eq!(table.byte_range(late), Some(100..200));
        assert_eq!(table.byte_range(early), Some(0..150));
    }

    #[test]
    fn empty_range_is_a_valid_zero_length_segment() {
        let mut table = SegmentTable::new();
        let id = table.push(42..42).unwrap();
        assert_eq!(table.byte_range(id), Some(42..42));
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // the point of the test
    fn inverted_range_is_rejected() {
        let mut table = SegmentTable::new();
        assert!(table.push(30..10).is_err());
    }

    #[test]
    fn oversized_segment_is_rejected_with_an_actionable_error() {
        let mut table = SegmentTable::new();
        let err = table
            .push(0..(u64::from(u32::MAX) + 1))
            .expect_err("segment longer than u32::MAX must be rejected");
        assert!(
            err.to_string().contains("split it into multiple segments"),
            "error should say what to do about it, got: {err}"
        );
    }

    #[test]
    fn unknown_id_has_no_byte_range() {
        let mut table = SegmentTable::new();
        table.push(0..10).unwrap();
        assert_eq!(table.byte_range(SegmentId::from(7)), None);
    }
}
