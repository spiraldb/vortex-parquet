// SPDX-License-Identifier: Apache-2.0

//! A [`ChunkReader`] that serves bytes this crate has already fetched.
//!
//! arrow-rs's Parquet decoder reads through [`ChunkReader`], addressing data by absolute file offset.
//! This crate, by contrast, wants to own I/O: byte ranges are declared up front in a
//! [`SegmentTable`](crate::segments::SegmentTable) so Vortex's driver can coalesce and pipeline them.
//!
//! This type reconciles the two. It holds already-fetched byte ranges and their offsets in the file,
//! and reports the file's real length so that offsets in the Parquet metadata line up. arrow-rs then
//! decodes from memory without issuing any I/O of its own, and without the bytes being copied: a
//! `ByteBuffer` is an owned immutable `'static` byte range, which is exactly what `Bytes::from_owner`
//! needs to take ownership of.
//!
//! # Why several spans rather than one
//!
//! A leaf covering a single column chunk holds one contiguous range. A leaf covering several columns
//! of a row group holds one range *per column* — and they are disjoint, because the columns it did
//! not need sit in between. Registering their enclosing span instead would fetch those unwanted
//! columns, which would make a projection-selective read cost the whole row group; the point of
//! declaring byte ranges up front is lost if the ranges are widened to be contiguous.
//!
//! So the reader holds a list of spans and dispatches each request to the one covering it. Spans are
//! kept sorted, and lookup is a binary search: a wide file gives a hundred-odd of them per row group
//! and the decoder asks per page.
//!
//! A request covered by no span is a bug in this crate — it means the segments registered for the
//! leaf did not cover everything the decoder wanted — so it fails loudly rather than silently
//! reading elsewhere.

use std::io::Read;
use std::ops::Range;

use bytes::Bytes;
use parquet::errors::ParquetError;
use parquet::errors::Result as ParquetResult;
use parquet::file::reader::ChunkReader;
use parquet::file::reader::Length;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

/// One already-fetched byte range, and where it sits in the file.
#[derive(Clone)]
struct Span {
    range: Range<u64>,
    bytes: Bytes,
}

/// In-memory windows onto a file, presented to arrow-rs as if they were the whole file.
#[derive(Clone)]
pub struct SegmentChunkReader {
    /// Sorted by start offset, and non-overlapping, which is what the binary search in
    /// [`SegmentChunkReader::span_for`] relies on.
    spans: Vec<Span>,
    /// The total length of the file the ranges came from.
    file_len: u64,
}

impl SegmentChunkReader {
    /// Wraps `buffer`, which holds the file's bytes over `range`.
    pub fn new(buffer: ByteBuffer, range: Range<u64>, file_len: u64) -> VortexResult<Self> {
        Self::new_multi(vec![(range, buffer)], file_len)
    }

    /// Wraps several already-fetched ranges, in any order.
    ///
    /// # Why this validates rather than trusts
    ///
    /// [`Self::span_for`] finds the last span starting at or before an offset and then stops looking,
    /// which is only correct if the spans are **disjoint**. Nothing structurally guaranteed that: the
    /// page-fetch path builds this from ranges derived from a Parquet offset index, which is *data*,
    /// and two overlapping page ranges would make a read of the second span's bytes resolve against
    /// the first — a wrong answer with no error. A buffer whose length disagrees with its range is the
    /// same class of fault: every offset into it would be shifted.
    ///
    /// Both are rejected here, once, rather than defended against at each of the read sites.
    pub fn new_multi(spans: Vec<(Range<u64>, ByteBuffer)>, file_len: u64) -> VortexResult<Self> {
        let mut spans: Vec<Span> = spans
            .into_iter()
            .map(|(range, buffer)| {
                if range.end < range.start {
                    vortex_bail!(
                        "a segment was given the inverted range {}..{}",
                        range.start,
                        range.end
                    );
                }
                if range.end > file_len {
                    vortex_bail!(
                        "a segment covering {}..{} extends past the end of a {file_len}-byte file",
                        range.start,
                        range.end
                    );
                }
                let declared = range.end - range.start;
                let held = u64::try_from(buffer.len())?;
                if declared != held {
                    vortex_bail!(
                        "a segment covering {}..{} was handed {held} bytes, not {declared}",
                        range.start,
                        range.end
                    );
                }
                Ok(Span {
                    range,
                    // `Bytes::from_owner` hands ownership of the buffer over without copying it. The
                    // obvious `copy_from_slice` would memcpy every compressed column chunk the scan
                    // touches, for nothing: `ByteBuffer` is already an owned, immutable, `'static`
                    // byte range, which is exactly what `Bytes` needs to borrow from.
                    bytes: Bytes::from_owner(buffer),
                })
            })
            .collect::<VortexResult<_>>()?;
        spans.sort_by_key(|span| span.range.start);

        for pair in spans.windows(2) {
            if pair[0].range.end > pair[1].range.start {
                vortex_bail!(
                    "segments {}..{} and {}..{} overlap; reads resolve against the first span \
                     covering an offset, so overlapping spans return the wrong bytes",
                    pair[0].range.start,
                    pair[0].range.end,
                    pair[1].range.start,
                    pair[1].range.end
                );
            }
        }

        Ok(Self { spans, file_len })
    }

    /// The span covering `start`, if any.
    fn span_for(&self, start: u64) -> Option<&Span> {
        // The last span beginning at or before `start`; spans are disjoint, so no earlier one can
        // cover it either.
        let idx = self
            .spans
            .partition_point(|span| span.range.start <= start)
            .checked_sub(1)?;
        let span = self.spans.get(idx)?;
        (start < span.range.end || span.range.is_empty()).then_some(span)
    }

    /// Returns the slice covering `start..start + length` in file coordinates.
    fn slice(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let end = start
            .checked_add(length.try_into().unwrap_or(u64::MAX))
            .ok_or_else(|| ParquetError::General("chunk read length overflows".into()))?;

        let span = self.span_for(start).filter(|span| end <= span.range.end);
        let Some(span) = span else {
            let held = self
                .spans
                .iter()
                .map(|span| format!("{}..{}", span.range.start, span.range.end))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ParquetError::General(format!(
                "vortex-parquet: the decoder asked for bytes {start}..{end}, which no segment \
                 registered for this leaf covers. Held: [{held}]. The byte ranges computed for the \
                 leaf are too narrow."
            )));
        };

        let local_start = usize::try_from(start - span.range.start)
            .map_err(|_| ParquetError::General("chunk offset too large for usize".into()))?;
        Ok(span.bytes.slice(local_start..local_start + length))
    }
}

impl Length for SegmentChunkReader {
    fn len(&self) -> u64 {
        // The *file's* length, not the held range's: Parquet metadata offsets are absolute, and a
        // short length makes the decoder reject them as out of bounds.
        self.file_len
    }
}

impl ChunkReader for SegmentChunkReader {
    type T = ChunkCursor;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        // Streams to the end of the span holding `start`, which is as far as a contiguous read can
        // go: the next span is a different column chunk with a gap in between.
        let span = self.span_for(start).map(|span| span.range.clone());
        let range = span.unwrap_or(start..start);
        let begin = start.max(range.start);
        let length = usize::try_from(range.end.saturating_sub(begin))
            .map_err(|_| ParquetError::General("chunk range too large for usize".into()))?;
        Ok(ChunkCursor {
            bytes: self.slice(begin, length)?,
            position: 0,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        self.slice(start, length)
    }
}

/// A `Read` cursor over a slice of held bytes.
pub struct ChunkCursor {
    bytes: Bytes,
    position: usize,
}

impl Read for ChunkCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.bytes.len() - self.position;
        let n = remaining.min(buf.len());
        buf[..n].copy_from_slice(&self.bytes[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    /// Overlapping spans are rejected, because reads resolve against the first covering span.
    ///
    /// `span_for` finds the last span starting at or before an offset and stops. With overlapping
    /// spans that silently returns bytes from the wrong one — the right length, the wrong data — which
    /// is the failure this constructor exists to make impossible.
    #[test]
    fn overlapping_spans_are_rejected() {
        let a = ByteBuffer::copy_from(vec![0u8; 20]);
        let b = ByteBuffer::copy_from(vec![1u8; 20]);
        let Err(err) = SegmentChunkReader::new_multi(vec![(100..120, a), (110..130, b)], 1000)
        else {
            panic!("overlapping spans must not build a reader");
        };
        assert!(format!("{err}").contains("overlap"), "{err}");
    }

    /// A range that starts after it ends, or runs past the file, is rejected.
    ///
    /// Both were accepted: the length check used `saturating_sub`, so an inverted range paired with an
    /// empty buffer looked consistent, and nothing tied a span to the file it claimed to be part of —
    /// so the reader could serve bytes from outside its own advertised file.
    #[test]
    fn a_range_outside_the_file_is_rejected() {
        let empty = ByteBuffer::copy_from(Vec::<u8>::new());
        // Built rather than written literally: an inverted literal is a clippy error, and the point
        // is that such a range can arrive from data, not from source.
        let (start, end) = (120u64, 100u64);
        let inverted_range = start..end;
        let Err(inverted) = SegmentChunkReader::new_multi(vec![(inverted_range, empty)], 1000)
        else {
            panic!("an inverted range must not build a reader");
        };
        assert!(format!("{inverted}").contains("inverted"), "{inverted}");

        let past = ByteBuffer::copy_from(vec![0u8; 20]);
        let Err(err) = SegmentChunkReader::new_multi(vec![(990..1010, past)], 1000) else {
            panic!("a range past the end of the file must not build a reader");
        };
        assert!(format!("{err}").contains("past the end"), "{err}");
    }

    /// A buffer whose length disagrees with its declared range is rejected.
    ///
    /// Every offset into such a span is shifted, so reads succeed and return the wrong bytes.
    #[test]
    fn a_buffer_that_does_not_match_its_range_is_rejected() {
        let short = ByteBuffer::copy_from(vec![0u8; 5]);
        let Err(err) = SegmentChunkReader::new_multi(vec![(100..120, short)], 1000) else {
            panic!("a mismatched buffer must not build a reader");
        };
        assert!(format!("{err}").contains("not 20"), "{err}");
    }

    use super::*;

    fn reader() -> SegmentChunkReader {
        // A window covering file bytes 100..110 of a 1000-byte file.
        let buffer = ByteBuffer::copy_from((0u8..10).collect::<Vec<_>>());
        SegmentChunkReader::new(buffer, 100..110, 1000).unwrap()
    }

    #[test]
    fn reports_the_files_length_not_the_windows() {
        assert_eq!(reader().len(), 1000);
    }

    #[test]
    fn serves_bytes_at_absolute_file_offsets() {
        let bytes = reader().get_bytes(102, 3).unwrap();
        assert_eq!(bytes.as_ref(), &[2, 3, 4]);
    }

    #[test]
    fn a_read_before_the_window_fails_with_an_actionable_message() {
        let err = reader()
            .get_bytes(90, 5)
            .expect_err("a read before the window must fail");
        assert!(
            err.to_string().contains("too narrow"),
            "error should say what to fix, got: {err}"
        );
    }

    /// Several disjoint spans, as a leaf covering several columns of a row group holds.
    fn multi_reader() -> SegmentChunkReader {
        SegmentChunkReader::new_multi(
            vec![
                (
                    100..110,
                    ByteBuffer::copy_from((0u8..10).collect::<Vec<_>>()),
                ),
                (
                    300..305,
                    ByteBuffer::copy_from((50u8..55).collect::<Vec<_>>()),
                ),
            ],
            1000,
        )
        .unwrap()
    }

    #[test]
    fn each_span_serves_its_own_range() {
        let reader = multi_reader();
        assert_eq!(reader.get_bytes(102, 3).unwrap().as_ref(), &[2, 3, 4]);
        assert_eq!(reader.get_bytes(301, 2).unwrap().as_ref(), &[51, 52]);
    }

    /// The gap between spans is the whole reason they are not merged: reading it would mean having
    /// fetched the columns in between, which is what holding disjoint spans avoids.
    #[test]
    fn a_read_in_the_gap_between_spans_fails() {
        let err = multi_reader()
            .get_bytes(200, 4)
            .expect_err("a read between spans must fail");
        assert!(
            err.to_string().contains("no segment"),
            "error should name the cause, got: {err}"
        );
    }

    /// A read must not be served across a gap even when both ends are held.
    #[test]
    fn a_read_spanning_two_spans_fails() {
        assert!(multi_reader().get_bytes(105, 200).is_err());
    }

    /// Spans may be handed over in any order; lookup sorts them.
    #[test]
    fn spans_are_ordered_on_construction() {
        let reader = SegmentChunkReader::new_multi(
            vec![
                (
                    300..305,
                    ByteBuffer::copy_from((50u8..55).collect::<Vec<_>>()),
                ),
                (
                    100..110,
                    ByteBuffer::copy_from((0u8..10).collect::<Vec<_>>()),
                ),
            ],
            1000,
        )
        .unwrap();
        assert_eq!(reader.get_bytes(100, 2).unwrap().as_ref(), &[0, 1]);
        assert_eq!(reader.get_bytes(300, 2).unwrap().as_ref(), &[50, 51]);
    }

    #[test]
    fn a_read_past_the_window_fails() {
        assert!(reader().get_bytes(108, 5).is_err());
    }

    #[test]
    fn get_read_streams_to_the_end_of_the_window() {
        let mut out = Vec::new();
        reader()
            .get_read(105)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, vec![5, 6, 7, 8, 9]);
    }
}
