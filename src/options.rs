// SPDX-License-Identifier: Apache-2.0

//! Reader tuning knobs.

// Parquet retains these two encodings for legacy files.
#![allow(deprecated)]

use parquet::basic::Encoding;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

/// Environment variable controlling when a filter conjunct is evaluated only on surviving rows.
pub const FILTER_EVAL_THRESHOLD_ENV: &str = "VORTEX_PARQUET_FILTER_EVAL_THRESHOLD";

/// Environment variable controlling when projection decodes only filter-selected records.
pub const SELECTIVE_DECODE_THRESHOLD_ENV: &str = "VORTEX_PARQUET_SELECTIVE_DECODE_THRESHOLD";

/// Environment variable controlling when page filter kernels use decoder `skip`/`get`.
///
/// Encoding-specific variables append the upper-case Parquet encoding name, for example
/// `VORTEX_PARQUET_FILTER_KERNEL_THRESHOLD_DELTA_BYTE_ARRAY`. An encoding-specific value wins over
/// this common value.
pub const FILTER_KERNEL_THRESHOLD_ENV: &str = "VORTEX_PARQUET_FILTER_KERNEL_THRESHOLD";

const ENCODING_COUNT: usize = 9;

/// A host-provided Snappy decoder. Success must initialize the entire output with the exact
/// decoded bytes; malformed streams and decoded-length mismatches must return an error.
pub type SnappyDecoder = fn(&[u8], &mut [u8]) -> VortexResult<()>;

/// A host-provided decoder that appends `expected_length` decoded bytes to the supplied buffer.
/// The buffer has enough spare capacity for that output. Existing bytes must remain unchanged;
/// malformed streams and decoded-length mismatches must return an error.
pub type SnappyAppendDecoder = fn(&[u8], &mut Vec<u8>, usize) -> VortexResult<()>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum SnappyDecoderKind {
    Slice(SnappyDecoder),
    Append(SnappyAppendDecoder),
}

/// Performance policy captured by [`crate::ParquetFormat`] and copied into its lazy page arrays.
///
/// A density is a number from zero through one. Environment values are read once when the format is
/// constructed, not during page execution, so a benchmark process observes one stable policy.
#[derive(Clone, Copy, Debug)]
pub struct ReaderOptions {
    materialize_projections: bool,
    filter_eval_threshold: f64,
    selective_decode_threshold: f64,
    filter_kernel_thresholds: [f64; ENCODING_COUNT],
    snappy_decoder: Option<SnappyDecoderKind>,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            materialize_projections: false,
            filter_eval_threshold: 0.2,
            selective_decode_threshold: 0.1,
            // Dense masks are cheaper to decode in bulk and filter afterwards. Decoder skipping
            // is reserved for sparse masks, where it avoids enough value work to repay its runs.
            filter_kernel_thresholds: [0.1; ENCODING_COUNT],
            snappy_decoder: None,
        }
    }
}

impl ReaderOptions {
    /// Uses the host's Snappy implementation for lazy flat pages. Other codecs and the nested
    /// Arrow fallback continue to use parquet-rs. The callback receives a correctly sized output.
    pub fn with_snappy_decoder(mut self, decoder: SnappyDecoder) -> Self {
        self.snappy_decoder = Some(SnappyDecoderKind::Slice(decoder));
        self
    }

    /// Uses a host decoder that appends directly to the page buffer. This lets the host avoid
    /// initializing output bytes before decompression while owning its buffer safety guarantees.
    pub fn with_snappy_append_decoder(mut self, decoder: SnappyAppendDecoder) -> Self {
        self.snappy_decoder = Some(SnappyDecoderKind::Append(decoder));
        self
    }

    pub(crate) fn snappy_decoder(&self) -> Option<SnappyDecoderKind> {
        self.snappy_decoder
    }

    /// Materializes selected projected columns inside their independently scheduled tasks.
    /// Computed decimal projections retain deferred execution. The default keeps all native page
    /// arrays deferred for consumers that can use them.
    pub fn with_materialized_projections(mut self) -> Self {
        self.materialize_projections = true;
        self
    }

    pub(crate) fn materialize_projections(&self) -> bool {
        self.materialize_projections
    }

    /// Reads valid density overrides from the process environment on top of the defaults.
    ///
    /// Invalid values are ignored here so [`crate::ParquetFormat::new`] remains infallible. Use
    /// [`Self::try_from_env`] in benchmark harnesses and services that want configuration mistakes
    /// to fail startup.
    pub fn from_env() -> Self {
        let mut options = Self::default();
        if let Ok(Some(value)) = env_density(FILTER_EVAL_THRESHOLD_ENV) {
            options.filter_eval_threshold = value;
        }
        if let Ok(Some(value)) = env_density(SELECTIVE_DECODE_THRESHOLD_ENV) {
            options.selective_decode_threshold = value;
        }
        if let Ok(Some(value)) = env_density(FILTER_KERNEL_THRESHOLD_ENV) {
            options.filter_kernel_thresholds.fill(value);
        }
        for encoding in ENCODINGS {
            let name = format!("{FILTER_KERNEL_THRESHOLD_ENV}_{}", encoding_name(encoding));
            if let Ok(Some(value)) = env_density(&name) {
                options.filter_kernel_thresholds[encoding_index(encoding)] = value;
            }
        }
        options
    }

    /// Reads density overrides from the process environment, rejecting malformed configuration.
    pub fn try_from_env() -> VortexResult<Self> {
        let mut options = Self::default();
        if let Some(value) = env_density(FILTER_EVAL_THRESHOLD_ENV)? {
            options.filter_eval_threshold = value;
        }
        if let Some(value) = env_density(SELECTIVE_DECODE_THRESHOLD_ENV)? {
            options.selective_decode_threshold = value;
        }
        if let Some(value) = env_density(FILTER_KERNEL_THRESHOLD_ENV)? {
            options.filter_kernel_thresholds.fill(value);
        }
        for encoding in ENCODINGS {
            let name = format!("{FILTER_KERNEL_THRESHOLD_ENV}_{}", encoding_name(encoding));
            if let Some(value) = env_density(&name)? {
                options.filter_kernel_thresholds[encoding_index(encoding)] = value;
            }
        }
        Ok(options)
    }

    /// Sets the density below which later filter conjuncts run only on rows already selected.
    pub fn with_filter_eval_threshold(mut self, threshold: f64) -> VortexResult<Self> {
        validate_density("filter evaluation threshold", threshold)?;
        self.filter_eval_threshold = threshold;
        Ok(self)
    }

    /// Sets the density below which projection decodes only selected Parquet records.
    pub fn with_selective_decode_threshold(mut self, threshold: f64) -> VortexResult<Self> {
        validate_density("selective decode threshold", threshold)?;
        self.selective_decode_threshold = threshold;
        Ok(self)
    }

    /// Sets the decoder-skipping density for every Parquet value encoding.
    pub fn with_filter_kernel_threshold(mut self, threshold: f64) -> VortexResult<Self> {
        validate_density("filter kernel threshold", threshold)?;
        self.filter_kernel_thresholds.fill(threshold);
        Ok(self)
    }

    /// Overrides the decoder-skipping density for one Parquet value encoding.
    pub fn with_encoding_filter_kernel_threshold(
        mut self,
        encoding: Encoding,
        threshold: f64,
    ) -> VortexResult<Self> {
        validate_density("encoding filter kernel threshold", threshold)?;
        self.filter_kernel_thresholds[encoding_index(encoding)] = threshold;
        Ok(self)
    }

    /// Density below which filter conjunct evaluation is narrowed by the existing mask.
    pub fn filter_eval_threshold(&self) -> f64 {
        self.filter_eval_threshold
    }

    /// Largest mask density at which projection decodes only selected Parquet records.
    pub fn selective_decode_threshold(&self) -> f64 {
        self.selective_decode_threshold
    }

    /// Largest mask density at which `encoding` uses its decoder-skipping filter kernel.
    pub fn filter_kernel_threshold(&self, encoding: Encoding) -> f64 {
        self.filter_kernel_thresholds[encoding_index(encoding)]
    }
}

fn env_density(name: &str) -> VortexResult<Option<f64>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .ok_or_else(|| vortex::error::vortex_err!("{name} is not valid UTF-8"))?
        .parse::<f64>()
        .map_err(|error| vortex::error::vortex_err!("invalid {name}: {error}"))?;
    validate_density(name, value)?;
    Ok(Some(value))
}

fn validate_density(name: &str, value: f64) -> VortexResult<()> {
    vortex_ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{name} must be a finite density between 0 and 1, got {value}"
    );
    Ok(())
}

const ENCODINGS: [Encoding; ENCODING_COUNT] = [
    Encoding::PLAIN,
    Encoding::PLAIN_DICTIONARY,
    Encoding::RLE,
    Encoding::BIT_PACKED,
    Encoding::DELTA_BINARY_PACKED,
    Encoding::DELTA_LENGTH_BYTE_ARRAY,
    Encoding::DELTA_BYTE_ARRAY,
    Encoding::RLE_DICTIONARY,
    Encoding::BYTE_STREAM_SPLIT,
];

fn encoding_index(encoding: Encoding) -> usize {
    match encoding {
        Encoding::PLAIN => 0,
        Encoding::PLAIN_DICTIONARY => 1,
        Encoding::RLE => 2,
        Encoding::BIT_PACKED => 3,
        Encoding::DELTA_BINARY_PACKED => 4,
        Encoding::DELTA_LENGTH_BYTE_ARRAY => 5,
        Encoding::DELTA_BYTE_ARRAY => 6,
        Encoding::RLE_DICTIONARY => 7,
        Encoding::BYTE_STREAM_SPLIT => 8,
    }
}

fn encoding_name(encoding: Encoding) -> &'static str {
    match encoding {
        Encoding::PLAIN => "PLAIN",
        Encoding::PLAIN_DICTIONARY => "PLAIN_DICTIONARY",
        Encoding::RLE => "RLE",
        Encoding::BIT_PACKED => "BIT_PACKED",
        Encoding::DELTA_BINARY_PACKED => "DELTA_BINARY_PACKED",
        Encoding::DELTA_LENGTH_BYTE_ARRAY => "DELTA_LENGTH_BYTE_ARRAY",
        Encoding::DELTA_BYTE_ARRAY => "DELTA_BYTE_ARRAY",
        Encoding::RLE_DICTIONARY => "RLE_DICTIONARY",
        Encoding::BYTE_STREAM_SPLIT => "BYTE_STREAM_SPLIT",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn densities_are_validated() {
        assert!(
            ReaderOptions::default()
                .with_filter_eval_threshold(0.0)
                .is_ok()
        );
        assert!(
            ReaderOptions::default()
                .with_filter_eval_threshold(1.0)
                .is_ok()
        );
        assert!(
            ReaderOptions::default()
                .with_filter_eval_threshold(f64::NAN)
                .is_err()
        );
        assert!(
            ReaderOptions::default()
                .with_filter_kernel_threshold(1.01)
                .is_err()
        );
    }

    #[test]
    fn one_encoding_can_override_the_common_threshold() -> VortexResult<()> {
        let options = ReaderOptions::default()
            .with_filter_kernel_threshold(0.3)?
            .with_encoding_filter_kernel_threshold(Encoding::DELTA_BYTE_ARRAY, 0.8)?;
        assert_eq!(options.filter_kernel_threshold(Encoding::PLAIN), 0.3);
        assert_eq!(
            options.filter_kernel_threshold(Encoding::DELTA_BYTE_ARRAY),
            0.8
        );
        Ok(())
    }
}
