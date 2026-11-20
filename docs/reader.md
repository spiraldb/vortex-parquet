# Reader reference

## Architecture

Opening a file builds an in-memory layout that keeps Parquet's physical boundaries and uses Vortex's
generic layouts for logical composition:

```text
ParquetFile
  └─ ChunkedLayout                       row groups
     └─ ParquetRowGroup
        └─ StructLayout                  logical top-level fields
           └─ ParquetField               Dremel reconstruction boundary
              └─ ParquetColumnChunk      one physical leaf
                 └─ ParquetIndexedPages  lazy offset-index discovery
                    or ParquetHeaderPages lazy chained-header discovery
```

Opening reads exactly the eight-byte tail and serialized footer metadata. It never reads page-index,
page-header, or payload bytes. The footer's index locations select the indexed-pages layout; without
an offset index the header-pages layout defers the sequential header walk until the column chunk is
actually read. `ParquetFormat::open` returns a Vortex V2 `ScanFile`: the inspectable root layout,
the byte location of every segment the layouts name, and a placed segment reader over the file.

Scans run on Vortex's planner-driven (V2) executor. The file and row-group layouts lower to their
children's plans, so a file becomes a `Concat` of row groups, each a `Pack` of fields. Each
`ParquetField` lowers to `vortex.parquet.plan.field_scan`, a leaf that reads the field's
column-chunk segment and decodes it, under a `LegacyStats` zone map built from the row group's
footer statistics (`vortex.parquet.plan.footer_zones`). The column-chunk and page-set layouts
describe the file for inspection only. The session's I/O scheduler coalesces, deduplicates and
retains segment reads. A field's decode is shared, through the format's decode cache, by every plan
over it: each filter conjunct, the projection, and every scan chunk of the row group. The leaf
reads whole column chunks; page-index pruning, Bloom-filter pruning and page-granular reads are not
yet expressed as plans (spiraldb/vortex-parquet#6).

For a flat field, each compressed data page becomes a lazy `vortex.parquet.page` array. Constructing
the array parses only its header and retains its compressed payload, dictionary page,
`ColumnDescPtr`, physical encoding, and logical Vortex dtype. Outer decompression, definition-level
decoding, and value decoding occur only when a consuming kernel or Arrow conversion forces that
page. Canonicalization calls parquet-rs's value decoders, covering the encodings accepted for the
page's physical type:

- `PLAIN`
- `RLE` / `BIT_PACKED` levels
- `RLE_DICTIONARY`
- `DELTA_BINARY_PACKED`
- `DELTA_LENGTH_BYTE_ARRAY`
- `DELTA_BYTE_ARRAY`
- `BYTE_STREAM_SPLIT`

This is intentionally a wrapper array rather than a conversion to Vortex's superficially similar
bit-packed arrays: their byte formats are different. Dictionary encoding is the important exception.
Once deferred page discovery runs, the dictionary values remain one shared compressed Parquet array
and dictionary data pages remain Parquet-encoded too. A property-driven parent reduction exposes a
native Vortex `DictArray` over `u32` codes when an operation benefits from dictionary structure.
Executing or filtering dictionary string pages also retains this representation, avoiding per-row
byte objects and repeated string validation; merely loading the chunk expands neither values nor
codes. Small numeric dictionaries (at most 256 values and at least eight page rows per value)
also retain their codes during sparse filtering, avoiding repeated decoder skips and value lookups.
Larger numeric dictionaries use the same path when mask density is at or below the configured
filter-kernel threshold (10% by default), sharing their values table across data pages.
Nested structs, lists, and maps currently
decode through arrow-rs at `ParquetField`, which is the boundary that owns repetition and
definition-level reconstruction.

Dictionary indices use a checked hybrid RLE/bit-packed decoder with 32-code blocks and
fixed-width unpacking. Short final groups are padded locally. Streams that cannot take this
path fall back to parquet-rs, including writers that omit unused final padding; decoded indices
still undergo dictionary-bound validation before use.
On little-endian ARM NEON, twelve-bit indices use vector byte gathers and shifts to decode
32 codes per block, with every load contained within the block's 48 bytes.
Optional dictionary pages with one complete all-present definition run retain compact all-valid
validity, avoiding a row-sized validity allocation and scatter while preserving nullable types.

Prepared membership sets also push into dictionary values during direct execution, so `IN`
predicates probe the set once per dictionary value. SQL null semantics and empty-set behavior
are preserved; dictionaries with more values than selected rows keep the ordinary execution path.
Selected-page filters resolve to their dictionary representation before membership falls back to
canonical values, so prior row selection does not prevent this pushdown.

## Pushdown

Each flat field's footer min, max and null count form a one-zone statistics table under the field's
leaf. Vortex's zone pruning falsifies the filter against it, so a row group whose statistics
disprove the filter is neither read nor decoded. Bounds are only used when trustworthy: deprecated
min/max fields are dropped unless their ordering is signed, and decimal bounds outside the declared
precision are dropped. Page-index and Bloom-filter pruning are not yet expressed on the V2 path
(spiraldb/vortex-parquet#6).

Dictionary pages reduce to Vortex `DictArray` on demand. Vortex's generic dictionary rule then decides
whether to push a scalar function into the distinct values using scalar-function fallibility and
null-sensitivity properties. This covers comparisons and string predicates without enumerating them
here. Plain byte-array pages additionally have direct `LIKE` and comparison kernels over the live
Parquet decoder, including the plain portions of mixed dictionary/plain chunks. Null predicates
execute from the page's definition-level validity and do not decode values.

On little-endian ARM NEON, PLAIN INT32 pages with logical signed 16-bit values decode through
deinterleaving loads. Every discarded high half must match the low half's sign extension.

The decoder casts physical values directly into their logical Vortex dtype. Density policies can be
configured either through `ReaderOptions` or environment variables:

- `VORTEX_PARQUET_FILTER_KERNEL_THRESHOLD` (default `1.0`) controls when a page filter uses decoder
  skipping. Append an encoding name for an override, such as
  `VORTEX_PARQUET_FILTER_KERNEL_THRESHOLD_DELTA_BYTE_ARRAY`.
- `VORTEX_PARQUET_FILTER_EVAL_THRESHOLD` and `VORTEX_PARQUET_SELECTIVE_DECODE_THRESHOLD` are still
  accepted but have no effect on the V2 path: Vortex's filter planner orders conjunct evaluation,
  and selective record decoding returns with spiraldb/vortex-parquet#6.

Footer row counts and flat-column min, max, null count, and Arrow memory-size estimates are exposed
through the Vortex `DataSource`, preserving exact/inexact precision for DataFusion planning.
Before choosing scan columns, the data source also removes static comparison conjuncts when exact,
trusted file bounds and a zero null count prove that every row matches. This avoids reading columns
used only by those filters. Dynamic comparisons, unknown statistics, and unsupported domains retain
their filters.

The data source schedules exactly one partition per Parquet row group, each a V2 scan of that row
group's rows. Vortex's executor runs it in chunks of at most 64K rows; the shared decode means a
row group is still decompressed once. The default decode cache is ephemeral: it shares live work
but holds no completed decoded array after the last plan releases it.
