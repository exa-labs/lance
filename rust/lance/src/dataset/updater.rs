// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::types::{self, ArrowPrimitiveType};
use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray,
    LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, MapArray, PrimitiveArray,
    RecordBatch, StringArray, StructArray, new_empty_array, new_null_array,
};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};
use arrow_schema::DataType;
use futures::StreamExt;
use lance_core::datatypes::{OnMissing, OnTypeMismatch};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result, datatypes::Schema};
use lance_table::format::{DataFile, Fragment};
use lance_table::utils::stream::ReadBatchFutStream;

use super::Dataset;
use super::fragment::FragmentReader;
use super::scanner::get_default_batch_size;
use super::write::{GenericWriter, cleanup_data_fragments, open_update_writer};
use crate::dataset::FileFragment;
use crate::dataset::utils::SchemaAdapter;

/// Update or insert a new column.
///
/// To use, call [`Updater::next`] to get the next [`RecordBatch`] as input,
/// then call [`Updater::update`] to update the batch. Repeat until
/// [`Updater::next`] returns `None`.
///
/// `write_schema` dictates the schema of the new file, while `final_schema` is
/// the schema of the full fragment after the update. These are optional and if
/// not specified, the updater will infer the write schema from the first batch
/// of results and will append them to the current schema to get the final schema.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    input_stream: ReadBatchFutStream,

    /// The last batch read from the file, with deleted rows removed
    last_input: Option<RecordBatch>,

    writer: Option<Box<dyn GenericWriter>>,

    /// The final schema of the fragment after the update.
    final_schema: Option<Schema>,

    /// The schema the new files will be written in. This only contains new columns.
    write_schema: Option<Schema>,

    /// The adapter to convert the logical data to physical data.
    schema_adapter: Option<SchemaAdapter>,

    finished: bool,

    deletion_restorer: DeletionRestorer,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the final schema (all the fields).
    ///
    /// If the schemas are not known, they can be None and will be inferred from
    /// the first batch of results.
    pub(super) async fn try_new(
        fragment: FileFragment,
        reader: FragmentReader,
        deletion_vector: DeletionVector,
        schemas: Option<(Schema, Schema)>,
        batch_size: Option<u32>,
    ) -> Result<Self> {
        let (write_schema, final_schema) = if let Some((write_schema, final_schema)) = schemas {
            (Some(write_schema), Some(final_schema))
        } else {
            (None, None)
        };

        let legacy_batch_size = reader.legacy_num_rows_in_batch(0);

        let batch_size = match (&legacy_batch_size, batch_size) {
            // If this is a v1 dataset we must use the row group size of the file
            (Some(legacy_batch_size), _) => *legacy_batch_size,
            // If this is a v2 dataset, let the user pick the batch size
            (None, Some(user_specified_batch_size)) => user_specified_batch_size,
            // Otherwise, default to 1024 if the user didn't specify anything
            (None, None) => get_default_batch_size().unwrap_or(1024) as u32,
        };

        let input_stream = reader.read_all(batch_size).await?;

        Ok(Self {
            fragment,
            input_stream,
            last_input: None,
            writer: None,
            write_schema,
            final_schema,
            // The schema adapter needs the data schema, not the logical schema, so it can't be
            // created until after the first batch is read.
            schema_adapter: None,
            finished: false,
            deletion_restorer: DeletionRestorer::new(deletion_vector, legacy_batch_size),
        })
    }

    pub fn fragment(&self) -> &FileFragment {
        &self.fragment
    }

    pub fn dataset(&self) -> &Dataset {
        self.fragment.dataset()
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    pub async fn next(&mut self) -> Result<Option<&RecordBatch>> {
        if self.finished {
            return Ok(None);
        }
        let batch = self.input_stream.next().await;
        match batch {
            None => {
                if !self.deletion_restorer.is_exhausted() {
                    // This can happen only if there is a batch size (e.g. v1 file) and the
                    // last batch(es) are entirely deleted.
                    return Err(Error::not_supported_source("Missing too many rows in merge, run compaction to materialize deletions first".into()));
                }
                self.finished = true;
                Ok(None)
            }
            Some(batch) => {
                self.last_input = Some(batch.await?);
                Ok(self.last_input.as_ref())
            }
        }
    }

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// It is the caller's responsibility to close the [`FileWriter`].
    ///
    /// Internal use only.
    async fn new_writer(&mut self, schema: Schema) -> Result<Box<dyn GenericWriter>> {
        let data_storage_version = self
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_version()?;

        open_update_writer(self.dataset(), &schema, data_storage_version).await
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::invalid_input(
                "Fragment Updater: no input data is available before update".to_string(),
            ));
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::invalid_input(format!(
                "Fragment Updater: new batch has different size with the source batch: {} != {}",
                last.num_rows(),
                batch.num_rows()
            )));
        };

        // Add back in deleted rows
        let batch = self.deletion_restorer.restore(batch)?;
        self.write_batches(vec![batch]).await
    }

    /// Update multiple batches without concatenating them.
    ///
    /// V2 writers can preserve the input batch boundaries all the way through
    /// the data writer, avoiding Arrow's 32-bit offset limit when a logical
    /// updater read batch contains more than 2 GiB of variable-width values.
    pub async fn update_batches(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::invalid_input(
                "Fragment Updater: no input data is available before update".to_string(),
            ));
        };

        if batches.is_empty() {
            return Err(Error::invalid_input(
                "Fragment Updater: cannot update with an empty batch list".to_string(),
            ));
        }

        let total_rows = batches
            .iter()
            .try_fold(0usize, |total, batch| total.checked_add(batch.num_rows()))
            .ok_or_else(|| {
                Error::invalid_input(
                    "Fragment Updater: input batch row count overflowed usize".to_string(),
                )
            })?;
        if last.num_rows() != total_rows {
            return Err(Error::invalid_input(format!(
                "Fragment Updater: new batches have different size with the source batch: {} != {}",
                last.num_rows(),
                total_rows
            )));
        }

        if self.deletion_restorer.legacy_batch_size.is_some() {
            // Legacy row groups must be restored as whole batches so that
            // DeletionRestorer can validate each output against its row-group size.
            let batch = arrow_select::concat::concat_batches(&batches[0].schema(), batches.iter())?;
            return self.update(batch).await;
        }

        let mut restored_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            restored_batches.push(self.deletion_restorer.restore(batch)?);
        }
        self.write_batches(restored_batches).await
    }

    async fn write_batches(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        debug_assert!(!batches.is_empty());
        self.ensure_writer(&batches[0]).await?;

        let schema_adapter = if let Some(schema_adapter) = self.schema_adapter.as_ref() {
            schema_adapter
        } else {
            self.schema_adapter = Some(SchemaAdapter::new(batches[0].schema()));
            self.schema_adapter.as_ref().unwrap()
        };

        let batches = batches
            .into_iter()
            .map(|batch| schema_adapter.to_physical_batch(batch))
            .collect::<Result<Vec<_>>>()?;

        let writer = self.writer.as_mut().unwrap();
        writer.write(&batches).await?;

        Ok(())
    }

    async fn ensure_writer(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }

        if self.write_schema.is_none() {
            // Need to infer the schema.
            let output_schema = batch.schema();
            let mut final_schema = self.fragment.schema().merge(output_schema.as_ref())?;
            final_schema.set_field_id(Some(self.fragment.dataset().manifest.max_field_id()));
            self.final_schema = Some(final_schema);
            self.final_schema.as_ref().unwrap().validate()?;
            self.write_schema = Some(self.final_schema.as_ref().unwrap().project_by_schema(
                output_schema.as_ref(),
                OnMissing::Error,
                OnTypeMismatch::Error,
            )?);
        }

        self.writer = Some(
            self.new_writer(self.write_schema.as_ref().unwrap().clone())
                .await?,
        );
        Ok(())
    }

    /// Finish updating this fragment, and returns the updated [`Fragment`].
    pub async fn finish(&mut self) -> Result<Fragment> {
        if let Some(writer) = self.writer.as_mut() {
            let (_, data_file) = writer.finish().await?;
            self.fragment.metadata.files.push(data_file);
        }

        Ok(self.fragment.metadata().clone())
    }

    /// Clean up any data file and blob sidecars created by the current unfinished writer.
    pub(super) async fn cleanup_unfinished_writer(&mut self) {
        let Some(writer) = self.writer.take() else {
            return;
        };
        let (path, base_id) = writer.data_file_path();
        let path = path.to_string();
        drop(writer);

        if path.is_empty() {
            return;
        }

        let mut fragment = Fragment::new(self.fragment.id() as u64);
        // cleanup_data_fragments only needs path/base_id to remove the unfinished
        // data file and any blob sidecars. Build a minimal synthetic fragment so
        // we can reuse the shared cleanup path without fabricating full metadata.
        fragment
            .files
            .push(DataFile::new(path, vec![], vec![], 0, 0, None, base_id));
        cleanup_data_fragments(
            &self.dataset().object_store,
            &self.dataset().base,
            None,
            &[fragment],
        )
        .await;
    }

    /// Get the final schema of the fragment after the update.
    ///
    /// This may be None if the schema is not known. This can happen if it was
    /// not specified up front and the first batch of results has not yet been
    /// processed.
    pub fn schema(&self) -> Option<&Schema> {
        self.final_schema.as_ref()
    }
}

/// Restores deleted rows.
///
/// All data files in a fragment must have the same # of rows (including deleted rows)
/// When we run the update process the next/update methods don't actually calculate on
/// deleted rows.  This means the updated batches will have fewer rows than the original
/// data files.  This struct restores the deleted rows, inserting arbitrary values into the
/// batches where the deleted rows should be.
///
/// To do this we scan through the deletion vector in sorted order, merging deleted rows
/// in as appropriate.
struct DeletionRestorer {
    current_row_id: u32,

    /// Number of rows in each batch, only used in legacy files for validation
    legacy_batch_size: Option<u32>,

    deletion_vector_iter: Option<Box<dyn Iterator<Item = u32> + Send>>,

    last_deleted_row_id: Option<u32>,
}

impl DeletionRestorer {
    fn new(deletion_vector: DeletionVector, legacy_batch_size: Option<u32>) -> Self {
        Self {
            current_row_id: 0,
            legacy_batch_size,
            deletion_vector_iter: Some(deletion_vector.into_sorted_iter()),
            last_deleted_row_id: None,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.deletion_vector_iter.is_none()
    }

    fn is_full(batch_size: Option<u32>, num_rows: u32) -> bool {
        if let Some(legacy_batch_size) = batch_size {
            // We should never encounter the case that `batch_size < num_rows` because
            // that would mean we have a v1 writer and it generated a batch with more rows
            // than expected
            debug_assert!(legacy_batch_size >= num_rows);
            legacy_batch_size == num_rows
        } else {
            false
        }
    }

    /// Given a batch of `num_rows`, walk through the deletion vector, and figure out where blanks
    /// should be inserted.
    ///
    /// For example, if self.current_row_id is 10 and the deletion vector is [11, 12, 19, 25] and
    /// num_rows is 7 then this function will at least return [1, 2] and the batch will at least
    /// span row ids 10..18.
    ///
    /// Then, in the example we need to choose whether the returned batch should include
    /// row 19 (and have 10 rows) or not (and have 9 rows).  This is only a concern in v1 files
    /// where we want to match the original row group size (which is the batch size).  If the
    /// batch size is 9 then we do not include 19 and return as above.
    ///
    /// If the batch size is 10 (or unset) then we do include 19 and the return will be [1, 2, 9]
    ///
    /// In v2 files, since the batch size will be unset, we will always include as many deleted
    /// rows at the end as we can.
    fn deleted_batch_offsets_in_range(&mut self, mut num_rows: u32) -> Vec<u32> {
        let mut deleted = Vec::new();
        let first_row_id = self.current_row_id;
        // The last row id (exclusive) in the batch
        let mut last_row_id = first_row_id + num_rows;
        // If there are zero deleted rows then the range covered will be first_row_id..last_row_id
        if self.deletion_vector_iter.is_none() {
            return deleted;
        }
        let deletion_vector_iter = self.deletion_vector_iter.as_mut().unwrap();

        // Now we need to walk through our deletion vector and figure out where to insert blanks
        let mut next_deleted_id = if self.last_deleted_row_id.is_some() {
            self.last_deleted_row_id
        } else {
            deletion_vector_iter.next()
        };
        loop {
            if let Some(next_deleted_id) = next_deleted_id {
                if next_deleted_id > last_row_id
                    || (next_deleted_id == last_row_id
                        && Self::is_full(self.legacy_batch_size, num_rows))
                {
                    // Either the next deleted id is out of range or it is the next row but
                    // we are full.  Either way, stash it and return
                    self.last_deleted_row_id = Some(next_deleted_id);
                    return deleted;
                }
                // Otherwise, the deleted row is in range, and we have space in our batch
                // and so we include it
                deleted.push(next_deleted_id - first_row_id);
                last_row_id += 1;
                num_rows += 1;
            } else {
                // Deleted row ids iterator is exhausted
                self.deletion_vector_iter = None;
                return deleted;
            }
            next_deleted_id = deletion_vector_iter.next();
        }
    }

    fn restore(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let deleted_batch_offsets = self.deleted_batch_offsets_in_range(batch.num_rows() as u32);
        let batch = add_blanks(batch, &deleted_batch_offsets)?;

        if let Some(batch_size) = self.legacy_batch_size {
            // validation just in case, when the input has a fixed batch size then the
            // output should have the same fixed batch size (except the last batch)
            let is_last = self.is_exhausted();
            if batch.num_rows() != batch_size as usize && !is_last {
                return Err(Error::internal(format!(
                    "Fragment Updater: batch size mismatch: {} != {}",
                    batch.num_rows(),
                    batch_size
                )));
            }
        }

        self.current_row_id += batch.num_rows() as u32;
        Ok(batch)
    }
}

/// Add blank rows where there are deleted rows
pub(crate) fn add_blanks(batch: RecordBatch, batch_offsets: &[u32]) -> Result<RecordBatch> {
    // Fast early return
    if batch_offsets.is_empty() {
        return Ok(batch);
    }

    if batch.num_rows() == 0 {
        if batch
            .schema()
            .fields()
            .iter()
            .any(|field| !field.is_nullable())
        {
            return Err(Error::not_supported_source(
                "Missing too many rows in merge, run compaction to materialize deletions first"
                    .into(),
            ));
        }
        // Deleted rows need placeholder values to preserve row alignment. These
        // blanks are never surfaced by scans because the rows are deleted.
        let columns = batch
            .schema()
            .fields()
            .iter()
            .map(|field| new_null_array(field.data_type(), batch_offsets.len()))
            .collect();
        return RecordBatch::try_new(batch.schema(), columns).map_err(Into::into);
    }

    let blank_columns = batch
        .schema()
        .fields()
        .iter()
        .map(|field| blank_array(field.data_type(), field.is_nullable(), 1))
        .collect::<Result<Vec<_>>>()?;

    let mut indices = Vec::with_capacity(batch.num_rows() + batch_offsets.len());
    let mut batch_pos = 0;
    let mut next_id = 0;
    for batch_offset in batch_offsets {
        let num_rows = *batch_offset - next_id;
        indices.extend((batch_pos..batch_pos + num_rows).map(|row| (0, row as usize)));
        indices.push((1, 0));
        next_id = *batch_offset + 1;
        batch_pos += num_rows;
    }
    indices.extend((batch_pos..batch.num_rows() as u32).map(|row| (0, row as usize)));

    let arrays = batch
        .columns()
        .iter()
        .zip(blank_columns)
        .map(|(array, blank)| {
            arrow_select::interleave::interleave(
                &[array.as_ref(), blank.as_ref()],
                indices.as_ref(),
            )
            .map_err(|e| Error::arrow(format!("Failed to add blanks: {}", e)))
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(batch.schema(), arrays)?;

    Ok(batch)
}

fn blank_array(data_type: &DataType, nullable: bool, len: usize) -> Result<ArrayRef> {
    if nullable {
        return Ok(new_null_array(data_type, len));
    }

    macro_rules! primitive {
        ($type:ty) => {
            Ok(Arc::new(
                PrimitiveArray::<$type>::from_value(
                    <$type as ArrowPrimitiveType>::default_value(),
                    len,
                )
                .with_data_type(data_type.clone()),
            ))
        };
    }

    match data_type {
        DataType::Null => Ok(new_null_array(data_type, len)),
        DataType::Boolean => Ok(Arc::new(BooleanArray::from(vec![false; len]))),
        DataType::Int8 => primitive!(types::Int8Type),
        DataType::Int16 => primitive!(types::Int16Type),
        DataType::Int32 => primitive!(types::Int32Type),
        DataType::Int64 => primitive!(types::Int64Type),
        DataType::UInt8 => primitive!(types::UInt8Type),
        DataType::UInt16 => primitive!(types::UInt16Type),
        DataType::UInt32 => primitive!(types::UInt32Type),
        DataType::UInt64 => primitive!(types::UInt64Type),
        DataType::Float16 => primitive!(types::Float16Type),
        DataType::Float32 => primitive!(types::Float32Type),
        DataType::Float64 => primitive!(types::Float64Type),
        DataType::Decimal32(_, _) => primitive!(types::Decimal32Type),
        DataType::Decimal64(_, _) => primitive!(types::Decimal64Type),
        DataType::Decimal128(_, _) => primitive!(types::Decimal128Type),
        DataType::Decimal256(_, _) => primitive!(types::Decimal256Type),
        DataType::Date32 => primitive!(types::Date32Type),
        DataType::Date64 => primitive!(types::Date64Type),
        DataType::Time32(arrow_schema::TimeUnit::Second) => primitive!(types::Time32SecondType),
        DataType::Time32(arrow_schema::TimeUnit::Millisecond) => {
            primitive!(types::Time32MillisecondType)
        }
        DataType::Time64(arrow_schema::TimeUnit::Microsecond) => {
            primitive!(types::Time64MicrosecondType)
        }
        DataType::Time64(arrow_schema::TimeUnit::Nanosecond) => {
            primitive!(types::Time64NanosecondType)
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
            primitive!(types::TimestampSecondType)
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            primitive!(types::TimestampMillisecondType)
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            primitive!(types::TimestampMicrosecondType)
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            primitive!(types::TimestampNanosecondType)
        }
        DataType::Interval(arrow_schema::IntervalUnit::YearMonth) => {
            primitive!(types::IntervalYearMonthType)
        }
        DataType::Interval(arrow_schema::IntervalUnit::DayTime) => {
            primitive!(types::IntervalDayTimeType)
        }
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano) => {
            primitive!(types::IntervalMonthDayNanoType)
        }
        DataType::Duration(arrow_schema::TimeUnit::Second) => primitive!(types::DurationSecondType),
        DataType::Duration(arrow_schema::TimeUnit::Millisecond) => {
            primitive!(types::DurationMillisecondType)
        }
        DataType::Duration(arrow_schema::TimeUnit::Microsecond) => {
            primitive!(types::DurationMicrosecondType)
        }
        DataType::Duration(arrow_schema::TimeUnit::Nanosecond) => {
            primitive!(types::DurationNanosecondType)
        }
        DataType::Utf8 => Ok(Arc::new(StringArray::from(
            std::iter::repeat_n("", len).collect::<Vec<_>>(),
        ))),
        DataType::LargeUtf8 => Ok(Arc::new(LargeStringArray::from(
            std::iter::repeat_n("", len).collect::<Vec<_>>(),
        ))),
        DataType::Utf8View => Ok(Arc::new(arrow_array::StringViewArray::from(
            std::iter::repeat_n("", len).collect::<Vec<_>>(),
        ))),
        DataType::Binary => Ok(Arc::new(BinaryArray::from(
            std::iter::repeat_n(b"".as_ref(), len).collect::<Vec<_>>(),
        ))),
        DataType::LargeBinary => Ok(Arc::new(LargeBinaryArray::from(
            std::iter::repeat_n(b"".as_ref(), len).collect::<Vec<_>>(),
        ))),
        DataType::BinaryView => Ok(Arc::new(arrow_array::BinaryViewArray::from(
            std::iter::repeat_n(b"".as_ref(), len).collect::<Vec<_>>(),
        ))),
        DataType::FixedSizeBinary(size) => {
            let size = usize::try_from(*size).map_err(|_| {
                Error::invalid_input(format!("Invalid fixed-size binary width: {size}"))
            })?;
            let values = vec![
                0;
                size.checked_mul(len).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Fixed-size binary buffer size overflow: {size} * {len}"
                    ))
                })?
            ];
            Ok(Arc::new(FixedSizeBinaryArray::new(
                size as i32,
                Buffer::from(values),
                Some(NullBuffer::new_valid(len)),
            )))
        }
        DataType::List(field) => Ok(Arc::new(ListArray::new(
            field.clone(),
            OffsetBuffer::new_zeroed(len + 1),
            new_empty_array(field.data_type()),
            None,
        ))),
        DataType::LargeList(field) => Ok(Arc::new(LargeListArray::new(
            field.clone(),
            OffsetBuffer::new_zeroed(len + 1),
            new_empty_array(field.data_type()),
            None,
        ))),
        DataType::FixedSizeList(field, size) => {
            let size = usize::try_from(*size).map_err(|_| {
                Error::invalid_input(format!("Invalid fixed-size list width: {size}"))
            })?;
            Ok(Arc::new(FixedSizeListArray::new(
                field.clone(),
                size as i32,
                blank_array(
                    field.data_type(),
                    field.is_nullable(),
                    size.checked_mul(len).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Fixed-size list value count overflow: {size} * {len}"
                        ))
                    })?,
                )?,
                None,
            )))
        }
        DataType::Struct(fields) => Ok(Arc::new(StructArray::new(
            fields.clone(),
            fields
                .iter()
                .map(|field| blank_array(field.data_type(), field.is_nullable(), len))
                .collect::<Result<Vec<_>>>()?,
            None,
        ))),
        DataType::Map(field, ordered) => {
            let entries = blank_array(field.data_type(), false, 0)?;
            let entries = entries
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| Error::invalid_input("Map entries must be a struct"))?
                .clone();
            Ok(Arc::new(MapArray::new(
                field.clone(),
                OffsetBuffer::new_zeroed(len + 1),
                entries,
                None,
                *ordered,
            )))
        }
        _ => Err(Error::not_supported_source(
            format!("Cannot construct blank value for Arrow data type {data_type:?}").into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use arrow::{array::AsArray, datatypes::Int32Type};
    use arrow_array::{
        Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, Int32Array, RecordBatch,
        RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_datagen::RowCount;
    use lance_encoding::version::LanceFileVersion;
    use std::sync::Arc;

    use super::add_blanks;
    use crate::Error;
    use crate::dataset::{Dataset, WriteParams};

    #[test]
    fn test_restore_deletes() {
        for batch_size in &[None, Some(10)] {
            let mut restorer = super::DeletionRestorer::new(
                vec![11, 12, 19, 20, 25].into_iter().collect(),
                *batch_size,
            );

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(10))
                .unwrap();
            // First batch is rows ids 0..9 so nothing is restored
            let restored = restorer.restore(batch.clone()).unwrap();
            assert_eq!(restored, batch);

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(7))
                .unwrap();
            // Next batch is rows ids 10..16 so we need to restore 11, 12
            // 19, and maybe 20 (depends on batch size)
            let restored = restorer.restore(batch).unwrap();
            let values = restored.column(0).as_primitive::<Int32Type>();
            assert_eq!(values.value(0), 0);
            assert_eq!(values.value(1), 0);
            assert_eq!(values.value(2), 0);
            assert_eq!(values.value(3), 1);
            assert_eq!(values.value(4), 2);
            assert_eq!(values.value(5), 3);
            assert_eq!(values.value(6), 4);
            assert_eq!(values.value(7), 5);
            assert_eq!(values.value(8), 6);
            assert_eq!(values.value(9), 0);
            if *batch_size == Some(10) {
                assert_eq!(values.len(), 10);
            } else {
                assert_eq!(values.value(10), 0);
                assert_eq!(values.len(), 11);
            }
        }
    }

    #[test]
    fn test_add_blanks() {
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();

        let with_blanks = add_blanks(batch.clone(), &[5, 7]).unwrap();

        assert_eq!(with_blanks.num_rows(), 12);
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), i as i32);
        }
        assert_eq!(values.value(5), 0);
        assert_eq!(values.value(6), 5);
        assert_eq!(values.value(7), 0);
        for i in 8..12 {
            assert_eq!(values.value(i), (i - 2) as i32);
        }

        let with_blanks = add_blanks(batch, &[0, 11]).unwrap();
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        for i in 1..11 {
            assert_eq!(values.value(i), (i - 1) as i32);
        }
        assert_eq!(values.value(11), 0);
    }

    #[test]
    fn test_add_blanks_uses_type_appropriate_values() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("nullable_int", DataType::Int32, true),
            Field::new("required_int", DataType::Int32, false),
            Field::new("nullable_binary", DataType::Binary, true),
            Field::new("required_binary", DataType::Binary, false),
            Field::new("nullable_string", DataType::Utf8, true),
            Field::new("required_string", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(BinaryArray::from(vec![
                    Some(b"one".as_ref()),
                    Some(b"two".as_ref()),
                ])),
                Arc::new(BinaryArray::from(vec![b"one".as_ref(), b"two".as_ref()])),
                Arc::new(StringArray::from(vec![Some("one"), Some("two")])),
                Arc::new(StringArray::from(vec!["one", "two"])),
            ],
        )
        .unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();
        assert!(with_blanks.column(0).is_null(1));
        assert_eq!(
            with_blanks.column(1).as_primitive::<Int32Type>().value(1),
            0
        );
        assert!(with_blanks.column(2).is_null(1));
        assert_eq!(with_blanks.column(3).as_binary::<i32>().value(1), b"");
        assert!(with_blanks.column(4).is_null(1));
        assert_eq!(with_blanks.column(5).as_string::<i32>().value(1), "");
    }

    #[test]
    fn test_add_blanks_does_not_duplicate_large_variable_width_values() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Binary,
            false,
        )]));
        let large_value = vec![42; 1024 * 1024];
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from(vec![large_value.as_slice()]))],
        )
        .unwrap();
        let deleted_rows = 100_000;
        let offsets = (0..deleted_rows as u32).collect::<Vec<_>>();

        let with_blanks = add_blanks(batch, &offsets).unwrap();
        let values = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(values.len(), deleted_rows + 1);
        assert_eq!(values.value(deleted_rows), large_value.as_slice());
        assert_eq!(values.value_data().len(), large_value.len());
    }

    #[test]
    fn test_add_blanks_recursively_constructs_nested_values() {
        let struct_fields = vec![
            Arc::new(Field::new("number", DataType::Int32, false)),
            Arc::new(Field::new("text", DataType::Utf8, false)),
        ];
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "nested",
            DataType::Struct(struct_fields.clone().into()),
            false,
        )]));
        let nested = StructArray::new(
            struct_fields.into(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as arrow_array::ArrayRef,
                Arc::new(StringArray::from(vec!["one", "two"])) as arrow_array::ArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_new(schema, vec![Arc::new(nested)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();
        let nested = with_blanks
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(nested.column(0).as_primitive::<Int32Type>().value(1), 0);
        assert_eq!(nested.column(1).as_string::<i32>().value(1), "");
    }

    #[test]
    fn test_add_blanks_constructs_fixed_size_list_values() {
        let item_field = Arc::new(Field::new("item", DataType::Int32, false));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "nested",
            DataType::FixedSizeList(item_field.clone(), 2),
            false,
        )]));
        let nested = FixedSizeListArray::new(
            item_field,
            2,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            None,
        );
        let batch = RecordBatch::try_new(schema, vec![Arc::new(nested)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();
        let nested = with_blanks
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let nested_value = nested.value(1);
        let values = nested_value.as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        assert_eq!(values.value(1), 0);
    }

    #[test]
    fn test_add_blanks_constructs_zero_width_fixed_size_binary_values() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::FixedSizeBinary(0),
            false,
        )]));
        let values = FixedSizeBinaryArray::new(
            0,
            arrow_buffer::Buffer::from(Vec::<u8>::new()),
            Some(arrow_buffer::NullBuffer::new_valid(2)),
        );
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap();

        let with_blanks = add_blanks(batch, &[1]).unwrap();
        assert_eq!(with_blanks.column(0).len(), 3);
    }

    #[test]
    fn test_add_blanks_rejects_non_nullable_empty_batch() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::new_empty(schema);

        let error = add_blanks(batch, &[0]).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("run compaction to materialize deletions first")
        );
    }

    #[tokio::test]
    async fn test_update_batches_rejects_row_count_mismatch() -> crate::Result<()> {
        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "source",
            DataType::Int32,
            false,
        )]));
        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3]))],
        )?;
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(source_batch)], source_schema),
            "memory://updater-row-count-mismatch",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await?;

        let fragment = dataset
            .get_fragments()
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("test dataset has no fragments"))?;
        let mut updater = fragment.updater::<String>(None, None, None).await?;
        let _ = updater.next().await?;

        let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "output",
            DataType::Int32,
            false,
        )]));
        let output = RecordBatch::try_new(
            output_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 11]))],
        )?;
        let error = updater
            .update_batches(vec![])
            .await
            .expect_err("empty batch lists must be rejected");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("cannot update with an empty batch list")
        );

        let error = updater
            .update_batches(vec![output])
            .await
            .expect_err("row-count mismatch must be rejected");

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("new batches have different size with the source batch: 4 != 2")
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_preserves_visible_rows_with_deletions() -> crate::Result<()> {
        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::Binary, false),
        ]));
        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2])),
                Arc::new(BinaryArray::from(vec![
                    b"zero".as_ref(),
                    b"deleted".as_ref(),
                    b"two".as_ref(),
                ])),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(source_batch)], source_schema),
            "memory://updater-merge-deletions",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await?;
        dataset.delete("id = 1").await?;
        let before_merge = dataset.scan().try_into_batch().await?;

        let right_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("metadata", DataType::Utf8, true),
        ]));
        let right_batch = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 2])),
                Arc::new(StringArray::from(vec!["first", "last"])),
            ],
        )?;
        dataset
            .merge(
                RecordBatchIterator::new(vec![Ok(right_batch)], right_schema),
                "id",
                "id",
            )
            .await?;
        let mut scan = dataset.scan();
        let after_merge = scan.project(&["id", "payload"]).unwrap();
        let after_merge = after_merge.try_into_batch().await?;

        assert_eq!(before_merge, after_merge);
        Ok(())
    }

    #[tokio::test]
    async fn test_update_batches_legacy_delegates_to_single_batch_update() -> crate::Result<()> {
        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "source",
            DataType::Int32,
            false,
        )]));
        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3]))],
        )?;
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(source_batch)], source_schema),
            "memory://updater-legacy-batches",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Legacy),
                ..Default::default()
            }),
        )
        .await?;

        let fragment = dataset
            .get_fragments()
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("test dataset has no fragments"))?;
        let mut updater = fragment.updater::<String>(None, None, None).await?;
        let input = updater
            .next()
            .await?
            .ok_or_else(|| Error::invalid_input("legacy updater did not yield its input batch"))?;
        assert_eq!(input.num_rows(), 4);

        let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "output",
            DataType::Int32,
            false,
        )]));
        let output = RecordBatch::try_new(
            output_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 11, 12, 13]))],
        )?;
        updater
            .update_batches(vec![output.slice(0, 2), output.slice(2, 2)])
            .await?;
        let updated_fragment = updater.finish().await?;
        assert_eq!(updated_fragment.files.len(), 2);
        Ok(())
    }
}
