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

use std::ops::Range;

use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
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
/// fields are built by the inner strategy and are never chunked themselves,
/// and a column containing a dictionary at any depth is never split. When a column is split, each chunk
/// becomes its own page, so the setting trades page count for encode
/// parallelism.
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
        let ranges = chunk_ranges(array.as_ref(), self.chunk_bytes)?;
        let is_split = ranges.len() > 1;
        let mut tasks = Vec::with_capacity(ranges.len());
        for rows in ranges {
            tasks.extend(self.inner.maybe_encode(
                array.slice(rows.start, rows.len()),
                external_buffers,
                RepDefBuilder::default(),
                row_number + rows.start as u64,
                rows.len() as u64,
            )?);
            // Leaf encoders buffer input until their per-column cache budget
            // is exceeded; flushing at every boundary keeps a chunk smaller
            // than that budget from merging into its neighbours' page.
            if is_split {
                tasks.extend(self.inner.flush(external_buffers)?);
            }
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

/// Contiguous row ranges covering `array`, each holding about `chunk_bytes`
/// of in-memory data and at least one row.
///
/// Boundaries follow the data's actual bytes when a slice's size can be
/// measured exactly, so clustered large values don't pile into one chunk;
/// other types split into ranges of equal row count. A column containing a
/// dictionary at any depth is returned whole, because every slice would carry
/// the entire shared dictionary into its own page.
fn chunk_ranges(array: &dyn Array, chunk_bytes: u64) -> Result<Vec<Range<usize>>> {
    let num_rows = array.len();
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    if contains_dictionary(array.data_type()) {
        return Ok(std::iter::once(0..num_rows).collect());
    }
    let total_bytes = slice_bytes(array, 0..num_rows)?;
    let num_chunks = total_bytes.div_ceil(chunk_bytes).clamp(1, num_rows as u64);
    if num_chunks == 1 {
        return Ok(std::iter::once(0..num_rows).collect());
    }
    if !has_exact_slice_size(array.data_type()) {
        let rows_per_chunk = num_rows.div_ceil(num_chunks as usize);
        return Ok((0..num_rows)
            .step_by(rows_per_chunk)
            .map(|start| start..(start + rows_per_chunk).min(num_rows))
            .collect());
    }

    let mut ranges = Vec::with_capacity(num_chunks as usize);
    let mut start = 0;
    let mut start_bytes = 0;
    for chunk in 1..num_chunks {
        let target_bytes = total_bytes * chunk / num_chunks;
        // A single row larger than a chunk can carry the prefix past several
        // targets; those boundaries are skipped rather than cut as tiny chunks.
        if start_bytes >= target_bytes {
            continue;
        }
        // Smallest end whose prefix reaches the target; prefix size only grows
        // with the row count.
        let (mut low, mut high) = (start + 1, num_rows);
        while low < high {
            let mid = low + (high - low) / 2;
            if slice_bytes(array, 0..mid)? >= target_bytes {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        if low == num_rows {
            break;
        }
        ranges.push(start..low);
        start = low;
        start_bytes = slice_bytes(array, 0..start)?;
    }
    ranges.push(start..num_rows);
    Ok(ranges)
}

fn slice_bytes(array: &dyn Array, rows: Range<usize>) -> Result<u64> {
    Ok(array
        .slice(rows.start, rows.len())
        .to_data()
        .get_slice_memory_size()? as u64)
}

fn contains_dictionary(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(..) => true,
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_dictionary(field.data_type())),
        DataType::List(item)
        | DataType::LargeList(item)
        | DataType::ListView(item)
        | DataType::LargeListView(item)
        | DataType::FixedSizeList(item, _)
        | DataType::Map(item, _) => contains_dictionary(item.data_type()),
        DataType::RunEndEncoded(_, values) => contains_dictionary(values.data_type()),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_dictionary(field.data_type())),
        _ => false,
    }
}

/// Whether [`arrow_data::ArrayData::get_slice_memory_size`] of a slice counts
/// only the rows in that slice. Offset-indexed children (lists, maps), shared
/// dictionaries and view buffers are reported whole, or not at all, for every
/// slice.
fn has_exact_slice_size(data_type: &DataType) -> bool {
    match data_type {
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| has_exact_slice_size(field.data_type())),
        DataType::FixedSizeList(item, _) => has_exact_slice_size(item.data_type()),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::Map(..)
        | DataType::Dictionary(..)
        | DataType::RunEndEncoded(..)
        | DataType::Union(..)
        | DataType::Utf8View
        | DataType::BinaryView => false,
        _ => true,
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int64Type;
    use arrow_array::{
        BinaryArray, DictionaryArray, Int32Array, Int64Array, ListArray, StructArray,
    };
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
    use rstest::rstest;

    use super::*;
    use crate::encoder::StructuralEncodingStrategy;
    use crate::version::LanceFileVersion;

    fn binary(value_sizes: &[usize]) -> ArrayRef {
        Arc::new(BinaryArray::from_iter_values(
            value_sizes.iter().map(|&size| vec![7u8; size]),
        ))
    }

    fn repeat(size: usize, count: usize) -> Vec<usize> {
        vec![size; count]
    }

    #[rstest]
    #[case::empty(binary(&[]), 64, vec![])]
    #[case::below_target(Arc::new(Int64Array::from(vec![1_i64; 10])) as ArrayRef, 1024, vec![0..10])]
    #[case::fixed_width_even(Arc::new(Int64Array::from(vec![1_i64; 100])) as ArrayRef, 200, vec![0..25, 25..50, 50..75, 75..100])]
    // 8 large values then 120 small ones: boundaries follow the bytes, not
    // the row count, so the large values spread across chunks.
    #[case::clustered_large_values(
        binary(&[repeat(1_000, 8), repeat(10, 120)].concat()),
        2_000,
        vec![0..2, 2..4, 4..6, 6..8, 8..128],
    )]
    // One value larger than several chunks takes a chunk of its own; the
    // targets it overshoots are skipped instead of cut as one-row chunks.
    #[case::single_oversized_value(
        binary(&[repeat(10_000, 1), repeat(10, 100)].concat()),
        2_000,
        vec![0..1, 1..101],
    )]
    #[case::struct_of_binary(
        Arc::new(StructArray::from(vec![(
            Arc::new(ArrowField::new("payload", ArrowDataType::Binary, false)),
            binary(&[repeat(1_000, 8), repeat(10, 120)].concat()),
        )])) as ArrayRef,
        2_000,
        vec![0..2, 2..4, 4..6, 6..8, 8..128],
    )]
    fn test_chunk_ranges(
        #[case] array: ArrayRef,
        #[case] chunk_bytes: u64,
        #[case] expected: Vec<Range<usize>>,
    ) {
        assert_eq!(chunk_ranges(array.as_ref(), chunk_bytes).unwrap(), expected);
    }

    #[test]
    fn test_chunk_ranges_keeps_dictionary_whole() {
        let keys = Int32Array::from_iter_values((0..10_000).map(|row| row % 4));
        let values = binary(&repeat(1 << 20, 4));
        let array = DictionaryArray::new(keys, values);
        assert_eq!(chunk_ranges(&array, 1 << 16).unwrap(), vec![0..10_000]);
    }

    #[test]
    fn test_chunk_ranges_keeps_nested_dictionary_whole() {
        let keys = Int32Array::from_iter_values((0..10_000).map(|row| row % 4));
        let dictionary: ArrayRef =
            Arc::new(DictionaryArray::new(keys, binary(&repeat(1 << 20, 4))));
        let array = StructArray::from(vec![(
            Arc::new(ArrowField::new(
                "tag",
                dictionary.data_type().clone(),
                false,
            )),
            dictionary,
        )]);
        assert_eq!(chunk_ranges(&array, 1 << 16).unwrap(), vec![0..10_000]);
    }

    #[test]
    fn test_chunk_ranges_splits_lists_by_row_count() {
        let array = ListArray::from_iter_primitive::<Int64Type, _, _>(
            (0..100).map(|row| Some(vec![Some(row); 10])),
        );
        let ranges = chunk_ranges(&array, 2_000).unwrap();
        assert!(ranges.len() > 1);
        let rows_per_chunk = ranges[0].len();
        assert!(
            ranges[..ranges.len() - 1]
                .iter()
                .all(|r| r.len() == rows_per_chunk)
        );
        assert_eq!(ranges.last().unwrap().end, 100);
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
