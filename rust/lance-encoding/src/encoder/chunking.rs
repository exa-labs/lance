// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A [`FieldEncodingStrategy`] that encodes large top-level columns as several
//! concurrently encoded pages.
//!
//! Structural (2.1+) encoders accumulate each top-level array handed to
//! `maybe_encode` and flush it as one page through one encode task, so a
//! single large `write_batch` serializes all compression of a column on one
//! thread. Feeding the array in row chunks gives each chunk its own page and
//! task; the writer awaits those tasks together, so their compression overlaps.

use arrow_array::{Array, ArrayRef};
use futures::future::BoxFuture;
use lance_core::datatypes::Field;
use lance_core::{Error, Result};

use super::{
    ColumnIndexSequence, EncodeTask, EncodedColumn, EncodingOptions, FieldEncoder,
    FieldEncodingStrategy, OutOfLineBuffers,
};
use crate::repdef::RepDefBuilder;

/// Wraps another strategy so each top-level column is split into row chunks
/// of roughly `chunk_bytes` of in-memory data before reaching its encoder.
///
/// The chunk count is computed per column and per `write_batch`, so a column
/// smaller than `chunk_bytes` is passed through as a single chunk. Nested
/// fields are built by the inner strategy and are never chunked themselves.
/// Each chunk becomes at least one page, so the setting trades page count for
/// encode parallelism.
///
/// `S` is typically [`super::StructuralEncodingStrategy`]: 2.0 files already
/// split large pages through `max_page_bytes`.
#[derive(Debug)]
pub struct ColumnChunkingStrategy<S> {
    inner: S,
    chunk_bytes: u64,
}

impl<S: FieldEncodingStrategy> ColumnChunkingStrategy<S> {
    /// `chunk_bytes` is the target in-memory size (arrow buffer bytes) of one
    /// chunk of a top-level column; it must be greater than zero.
    pub fn try_new(inner: S, chunk_bytes: u64) -> Result<Self> {
        if chunk_bytes == 0 {
            return Err(Error::invalid_input(
                "ColumnChunkingStrategy chunk_bytes must be greater than zero, got 0",
            ));
        }
        Ok(Self { inner, chunk_bytes })
    }
}

impl<S: FieldEncodingStrategy> FieldEncodingStrategy for ColumnChunkingStrategy<S> {
    fn create_field_encoder(
        &self,
        _encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        options: &EncodingOptions,
    ) -> Result<Box<dyn FieldEncoder>> {
        // The inner strategy is passed as the root so that only top-level
        // fields are wrapped: a nested child's rep/def levels belong to its
        // parent's rows and cannot be re-sliced here.
        let inner = self
            .inner
            .create_field_encoder(&self.inner, field, column_index, options)?;
        Ok(Box::new(ChunkingFieldEncoder {
            inner,
            chunk_bytes: self.chunk_bytes,
        }))
    }
}

struct ChunkingFieldEncoder {
    inner: Box<dyn FieldEncoder>,
    chunk_bytes: u64,
}

impl FieldEncoder for ChunkingFieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        if !repdef.is_empty() {
            return Err(Error::internal(format!(
                "ColumnChunkingStrategy wraps only top-level fields, but received rep/def \
                 levels for {num_rows} rows starting at row {row_number}"
            )));
        }
        let array_bytes = array.to_data().get_slice_memory_size()? as u64;
        let mut tasks = Vec::new();
        for (offset, len) in chunk_ranges(array.len(), array_bytes, self.chunk_bytes) {
            tasks.extend(self.inner.maybe_encode(
                array.slice(offset, len),
                external_buffers,
                RepDefBuilder::default(),
                row_number + offset as u64,
                len as u64,
            )?);
        }
        Ok(tasks)
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.inner.flush(external_buffers)
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<EncodedColumn>>> {
        self.inner.finish(external_buffers)
    }

    fn num_columns(&self) -> u32 {
        self.inner.num_columns()
    }
}

/// Splits `num_rows` into `ceil(num_bytes / chunk_bytes)` contiguous
/// `(offset, len)` ranges of near-equal row count, never more ranges than rows.
fn chunk_ranges(num_rows: usize, num_bytes: u64, chunk_bytes: u64) -> Vec<(usize, usize)> {
    if num_rows == 0 {
        return Vec::new();
    }
    let num_chunks = num_bytes.div_ceil(chunk_bytes).clamp(1, num_rows as u64) as usize;
    let rows_per_chunk = num_rows.div_ceil(num_chunks);
    (0..num_rows)
        .step_by(rows_per_chunk)
        .map(|offset| (offset, rows_per_chunk.min(num_rows - offset)))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::encoder::StructuralEncodingStrategy;
    use crate::version::LanceFileVersion;

    #[rstest]
    #[case::empty(0, 0, 64, vec![])]
    #[case::below_target(10, 63, 64, vec![(0, 10)])]
    #[case::just_over_target(10, 65, 64, vec![(0, 5), (5, 5)])]
    #[case::uneven_rows(10, 300, 64, vec![(0, 2), (2, 2), (4, 2), (6, 2), (8, 2)])]
    #[case::more_chunks_than_rows(3, 1 << 30, 1, vec![(0, 1), (1, 1), (2, 1)])]
    fn test_chunk_ranges(
        #[case] num_rows: usize,
        #[case] num_bytes: u64,
        #[case] chunk_bytes: u64,
        #[case] expected: Vec<(usize, usize)>,
    ) {
        assert_eq!(chunk_ranges(num_rows, num_bytes, chunk_bytes), expected);
    }

    #[test]
    fn test_zero_chunk_bytes_rejected() {
        let error = ColumnChunkingStrategy::try_new(
            StructuralEncodingStrategy::with_version(LanceFileVersion::V2_2),
            0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("chunk_bytes must be greater than zero")
        );
    }
}
