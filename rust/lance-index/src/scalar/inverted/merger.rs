// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::compute::concat_batches;
use arrow_array::builder::{ListBuilder, UInt32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{ArrayRef, ListArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use fst::Streamer;
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempDir;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::Result;
use lance_io::object_store::ObjectStore;
use std::sync::Arc;

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexStore, IndexWriter};

use super::{
    builder::{
        doc_file_path, inverted_list_schema, posting_file_path, token_file_path, InnerBuilder,
        PositionRecorder,
    },
    DocSet, InvertedPartition, PostingList, PostingListBuilder, TokenMap, TokenSetFormat,
};

pub trait Merger {
    // Merge the partitions and write new partitions,
    // the new partitions are returned.
    // This method would read all the input partitions at the same time,
    // so it's not recommended to pass too many partitions.
    async fn merge(&mut self) -> Result<Vec<u64>>;
}

const SPILL_TOKEN_ID_COL: &str = "_merge_token_id";
const SPILL_DOC_IDS_COL: &str = "_merge_doc_ids";
const SPILL_FREQS_COL: &str = "_merge_freqs";
const SPILL_POSITIONS_COL: &str = "_merge_positions";

const DEFAULT_STREAM_READ_TOKENS: usize = 1024;
const DEFAULT_SPILL_FLUSH_TOKENS: usize = 64;
const DEFAULT_BUCKET_SIZE_MB: u64 = 256;
const DEFAULT_SPILL_READ_BATCH_ROWS: u64 = 1024;

fn parse_env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// A merger that merges partitions based on their size,
// it would read the posting lists for each token from
// the partitions and write them to a new partition,
// until the size of the new partition reaches the target size.
pub struct SizeBasedMerger<'a> {
    dest_store: &'a dyn IndexStore,
    input: Vec<InvertedPartition>,
    with_position: bool,
    target_size: u64,
    token_set_format: TokenSetFormat,
    builder: InnerBuilder,
    partitions: Vec<u64>,
    streaming: bool,
    stream_read_tokens: usize,
    spill_flush_tokens: usize,
    bucket_postings_limit: u64,
    spill_read_batch_rows: u64,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is compressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        dest_store: &'a dyn IndexStore,
        input: Vec<InvertedPartition>,
        target_size: u64,
        token_set_format: TokenSetFormat,
    ) -> Self {
        let max_id = input.iter().map(|p| p.id()).max().unwrap_or(0);
        let with_position = input
            .first()
            .map(|p| p.inverted_list.has_positions())
            .unwrap_or(false);
        let streaming = parse_env_bool("LANCE_FTS_STREAMING_MERGE");
        let stream_read_tokens =
            parse_env_usize("LANCE_FTS_MERGE_READ_TOKENS", DEFAULT_STREAM_READ_TOKENS).max(1);
        let spill_flush_tokens =
            parse_env_usize("LANCE_FTS_MERGE_SPILL_TOKENS", DEFAULT_SPILL_FLUSH_TOKENS).max(1);
        let bucket_size_mb =
            parse_env_u64("LANCE_FTS_MERGE_BUCKET_SIZE_MB", DEFAULT_BUCKET_SIZE_MB);
        let bytes_per_posting = if with_position { 12 } else { 8 };
        let bucket_postings_limit = ((bucket_size_mb << 20) / bytes_per_posting).max(1);
        let spill_read_batch_rows = parse_env_u64(
            "LANCE_FTS_MERGE_SPILL_READ_ROWS",
            DEFAULT_SPILL_READ_BATCH_ROWS,
        )
        .max(1);

        Self {
            dest_store,
            input,
            with_position,
            target_size,
            token_set_format,
            builder: InnerBuilder::new(max_id + 1, with_position, token_set_format),
            partitions: Vec::new(),
            streaming,
            stream_read_tokens,
            spill_flush_tokens,
            bucket_postings_limit,
            spill_read_batch_rows,
        }
    }

    #[cfg(test)]
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    async fn flush(&mut self) -> Result<()> {
        if !self.builder.tokens.is_empty() {
            log::info!("flushing partition {}", self.builder.id());
            let start = std::time::Instant::now();
            self.builder.write(self.dest_store).await?;
            log::info!(
                "flushed partition {} in {:?}",
                self.builder.id(),
                start.elapsed()
            );
            self.partitions.push(self.builder.id());
            self.builder = InnerBuilder::new(
                self.builder.id() + 1,
                self.with_position,
                self.token_set_format,
            );
        }
        Ok(())
    }

    async fn merge_in_memory(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            for part in self.input.iter() {
                part.store()
                    .copy_index_file(&token_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&posting_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&doc_file_path(part.id()), self.dest_store)
                    .await?;
            }

            return Ok(self.input.iter().map(|p| p.id()).collect());
        }

        log::info!(
            "merging {} partitions with target size {} MiB",
            self.input.len(),
            self.target_size / 1024 / 1024
        );
        let mut estimated_size = 0;
        let start = std::time::Instant::now();
        let parts = std::mem::take(&mut self.input);
        let num_parts = parts.len();
        for (idx, part) in parts.into_iter().enumerate() {
            if self.builder.docs.len() + part.docs.len() > u32::MAX as usize
                || estimated_size >= self.target_size
            {
                self.flush().await?;
                estimated_size = 0;
            }

            let mut token_id_map = vec![u32::MAX; part.tokens.len()];
            match &part.tokens.tokens {
                TokenMap::HashMap(map) => {
                    for (token, token_id) in map.iter() {
                        let new_token_id = self.builder.tokens.get_or_add(token.as_str());
                        let index = *token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
                TokenMap::Fst(map) => {
                    let mut stream = map.stream();
                    while let Some((token, token_id)) = stream.next() {
                        let token_id = token_id as u32;
                        let token = String::from_utf8_lossy(token);
                        let new_token_id = self.builder.tokens.get_or_add(token.as_ref());
                        let index = token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
            }
            let doc_id_offset = self.builder.docs.len() as u32;
            for (row_id, num_tokens) in part.docs.iter() {
                self.builder.docs.append(*row_id, *num_tokens);
            }
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::new(part.inverted_list.has_positions())
                });

            let postings = part
                .inverted_list
                .read_batch(part.inverted_list.has_positions())
                .await?;
            for token_id in 0..part.tokens.len() as u32 {
                let posting_list = part
                    .inverted_list
                    .posting_list_from_batch(&postings.slice(token_id as usize, 1), token_id)?;
                let new_token_id = token_id_map[token_id as usize];
                debug_assert_ne!(new_token_id, u32::MAX);
                let builder = &mut self.builder.posting_lists[new_token_id as usize];
                let old_size = builder.size();
                for (doc_id, freq, positions) in posting_list.iter() {
                    let new_doc_id = doc_id_offset + doc_id as u32;
                    let positions = match positions {
                        Some(positions) => PositionRecorder::Position(positions.collect()),
                        None => PositionRecorder::Count(freq),
                    };
                    builder.add(new_doc_id, positions);
                }
                let new_size = builder.size();
                estimated_size += new_size - old_size;
            }
            log::info!(
                "merged {}/{} partitions in {:?}",
                idx + 1,
                num_parts,
                start.elapsed()
            );
        }

        self.flush().await?;
        Ok(self.partitions.clone())
    }

    async fn merge_streaming(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            for part in self.input.iter() {
                part.store()
                    .copy_index_file(&token_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&posting_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&doc_file_path(part.id()), self.dest_store)
                    .await?;
            }

            return Ok(self.input.iter().map(|p| p.id()).collect());
        }

        log::info!(
            "streaming merge {} partitions with target size {} MiB",
            self.input.len(),
            self.target_size / 1024 / 1024
        );
        let start = std::time::Instant::now();
        let parts = std::mem::take(&mut self.input);
        let num_parts = parts.len();
        let mut group_parts = Vec::new();
        let mut group_docs_len = 0usize;
        let mut group_estimated_size = 0u64;

        for (idx, part) in parts.into_iter().enumerate() {
            let part_estimated_size = estimate_partition_size(&part, self.with_position);
            if !group_parts.is_empty()
                && (group_docs_len + part.docs.len() > u32::MAX as usize
                    || group_estimated_size + part_estimated_size >= self.target_size)
            {
                self.merge_group_streaming(std::mem::take(&mut group_parts))
                    .await?;
                group_docs_len = 0;
                group_estimated_size = 0;
            }
            group_docs_len += part.docs.len();
            group_estimated_size += part_estimated_size;
            group_parts.push(part);
            log::info!(
                "queued {}/{} partitions in {:?}",
                idx + 1,
                num_parts,
                start.elapsed()
            );
        }

        if !group_parts.is_empty() {
            self.merge_group_streaming(group_parts).await?;
        }

        Ok(self.partitions.clone())
    }

    async fn merge_group_streaming(&mut self, parts: Vec<InvertedPartition>) -> Result<()> {
        if parts.is_empty() {
            return Ok(());
        }

        let mut builder =
            InnerBuilder::new(self.builder.id(), self.with_position, self.token_set_format);
        let mut token_id_maps = Vec::with_capacity(parts.len());
        let mut doc_id_offsets = Vec::with_capacity(parts.len());

        for part in parts.iter() {
            let mut token_id_map = vec![u32::MAX; part.tokens.len()];
            match &part.tokens.tokens {
                TokenMap::HashMap(map) => {
                    for (token, token_id) in map.iter() {
                        let new_token_id = builder.tokens.get_or_add(token.as_str());
                        let index = *token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
                TokenMap::Fst(map) => {
                    let mut stream = map.stream();
                    while let Some((token, token_id)) = stream.next() {
                        let token_id = token_id as u32;
                        let token = String::from_utf8_lossy(token);
                        let new_token_id = builder.tokens.get_or_add(token.as_ref());
                        let index = token_id as usize;
                        debug_assert!(index < token_id_map.len());
                        token_id_map[index] = new_token_id;
                    }
                }
            }
            let doc_id_offset = builder.docs.len() as u32;
            doc_id_offsets.push(doc_id_offset);
            for (row_id, num_tokens) in part.docs.iter() {
                builder.docs.append(*row_id, *num_tokens);
            }
            token_id_maps.push(token_id_map);
        }

        let total_tokens = builder.tokens.len();
        let mut token_lengths = vec![0u64; total_tokens];
        for (part_idx, part) in parts.iter().enumerate() {
            let lengths = part.inverted_list.posting_lengths();
            let token_id_map = &token_id_maps[part_idx];
            for (old_token_id, len) in lengths.iter().enumerate() {
                let new_token_id = token_id_map[old_token_id];
                debug_assert_ne!(new_token_id, u32::MAX);
                token_lengths[new_token_id as usize] += *len as u64;
            }
        }

        let (buckets, bucket_for_token) =
            build_buckets(&token_lengths, self.bucket_postings_limit, builder.id());

        let spill_dir = TempDir::default();
        let spill_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            spill_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut spill_writer = SpillWriter::new(
            spill_store.clone(),
            &buckets,
            self.with_position,
            self.spill_flush_tokens,
        );

        for (part_idx, part) in parts.iter().enumerate() {
            let doc_id_offset = doc_id_offsets[part_idx];
            let token_id_map = &token_id_maps[part_idx];
            let mut start_token = 0usize;
            let total_part_tokens = part.tokens.len();
            while start_token < total_part_tokens {
                let end_token = (start_token + self.stream_read_tokens).min(total_part_tokens);
                let (batch, base_row) = part
                    .inverted_list
                    .read_token_range(start_token, end_token, self.with_position)
                    .await?;
                for token_id in start_token..end_token {
                    let posting_list = part.inverted_list.posting_list_from_range_batch(
                        &batch,
                        token_id,
                        start_token,
                        base_row,
                    )?;
                    let new_token_id = token_id_map[token_id];
                    debug_assert_ne!(new_token_id, u32::MAX);
                    let bucket_idx = bucket_for_token[new_token_id as usize];
                    let (doc_ids, freqs, positions) =
                        collect_posting_list(&posting_list, doc_id_offset, self.with_position);
                    spill_writer
                        .append(bucket_idx, new_token_id, doc_ids, freqs, positions)
                        .await?;
                }
                start_token = end_token;
            }
        }

        spill_writer.finish().await?;

        let docs = Arc::new(std::mem::take(&mut builder.docs));
        let schema = inverted_list_schema(self.with_position);
        let mut writer = self
            .dest_store
            .new_index_file(&posting_file_path(builder.id()), schema.clone())
            .await?;
        let mut buffer = Vec::new();
        let mut size_sum = 0usize;

        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            if bucket.start == bucket.end {
                continue;
            }
            let mut posting_lists = (bucket.start..bucket.end)
                .map(|_| PostingListBuilder::new(self.with_position))
                .collect::<Vec<_>>();
            let reader = spill_store.open_index_file(&bucket.file_name).await?;
            let num_batches = reader.num_batches(self.spill_read_batch_rows).await;
            for batch_idx in 0..num_batches {
                let batch = reader
                    .read_record_batch(batch_idx as u64, self.spill_read_batch_rows)
                    .await?;
                append_spill_batch(&batch, bucket.start, self.with_position, &mut posting_lists)?;
            }

            write_posting_lists_to_writer(
                &mut writer,
                docs.clone(),
                posting_lists,
                &schema,
                &mut buffer,
                &mut size_sum,
            )
            .await?;
            spill_store.delete_index_file(&bucket.file_name).await?;
            log::info!(
                "merged spill bucket {}/{} for partition {}",
                bucket_idx + 1,
                buckets.len(),
                builder.id()
            );
        }

        if !buffer.is_empty() {
            let batch = concat_batches(&schema, buffer.iter())?;
            buffer.clear();
            writer.write_record_batch(batch).await?;
        }
        writer.finish().await?;

        builder.write_tokens(self.dest_store).await?;
        builder.write_docs(self.dest_store, docs).await?;

        self.partitions.push(builder.id());
        self.builder =
            InnerBuilder::new(builder.id() + 1, self.with_position, self.token_set_format);

        Ok(())
    }
}

impl Merger for SizeBasedMerger<'_> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        if self.streaming {
            self.merge_streaming().await
        } else {
            self.merge_in_memory().await
        }
    }
}

struct BucketRange {
    start: u32,
    end: u32,
    file_name: String,
}

struct SpillBuffer {
    token_ids: Vec<u32>,
    doc_ids: Vec<Vec<u32>>,
    freqs: Vec<Vec<u32>>,
    positions: Option<Vec<Vec<Vec<u32>>>>,
}

impl SpillBuffer {
    fn new(with_position: bool) -> Self {
        Self {
            token_ids: Vec::new(),
            doc_ids: Vec::new(),
            freqs: Vec::new(),
            positions: with_position.then(Vec::new),
        }
    }

    fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    fn clear(&mut self) {
        self.token_ids.clear();
        self.doc_ids.clear();
        self.freqs.clear();
        if let Some(positions) = &mut self.positions {
            positions.clear();
        }
    }
}

struct SpillWriter {
    store: Arc<dyn IndexStore>,
    schema: Arc<ArrowSchema>,
    with_position: bool,
    flush_tokens: usize,
    file_names: Vec<String>,
    writers: Vec<Option<Box<dyn IndexWriter>>>,
    buffers: Vec<SpillBuffer>,
}

impl SpillWriter {
    fn new(
        store: Arc<dyn IndexStore>,
        buckets: &[BucketRange],
        with_position: bool,
        flush_tokens: usize,
    ) -> Self {
        let file_names = buckets
            .iter()
            .map(|bucket| bucket.file_name.clone())
            .collect();
        let schema = spill_schema(with_position);
        let mut writers = Vec::with_capacity(buckets.len());
        writers.resize_with(buckets.len(), || None);
        let buffers = (0..buckets.len())
            .map(|_| SpillBuffer::new(with_position))
            .collect();
        Self {
            store,
            schema,
            with_position,
            flush_tokens,
            file_names,
            writers,
            buffers,
        }
    }

    async fn append(
        &mut self,
        bucket_idx: usize,
        token_id: u32,
        doc_ids: Vec<u32>,
        freqs: Vec<u32>,
        positions: Option<Vec<Vec<u32>>>,
    ) -> Result<()> {
        let buffer = &mut self.buffers[bucket_idx];
        buffer.token_ids.push(token_id);
        buffer.doc_ids.push(doc_ids);
        buffer.freqs.push(freqs);
        if self.with_position {
            let positions = positions.expect("positions are required");
            buffer
                .positions
                .as_mut()
                .expect("positions buffer is missing")
                .push(positions);
        }
        if buffer.token_ids.len() >= self.flush_tokens {
            self.flush_bucket(bucket_idx).await?;
        }
        Ok(())
    }

    async fn flush_bucket(&mut self, bucket_idx: usize) -> Result<()> {
        if self.buffers[bucket_idx].is_empty() {
            return Ok(());
        }
        if self.writers[bucket_idx].is_none() {
            let writer = self
                .store
                .new_index_file(&self.file_names[bucket_idx], self.schema.clone())
                .await?;
            self.writers[bucket_idx] = Some(writer);
        }
        let batch = spill_batch_from_buffer(&mut self.buffers[bucket_idx], self.schema.clone())?;
        if let Some(writer) = &mut self.writers[bucket_idx] {
            writer.write_record_batch(batch).await?;
        }
        Ok(())
    }

    async fn finish(&mut self) -> Result<()> {
        for bucket_idx in 0..self.buffers.len() {
            self.flush_bucket(bucket_idx).await?;
        }
        for writer in self.writers.iter_mut().flatten() {
            writer.finish().await?;
        }
        Ok(())
    }
}

fn spill_schema(with_position: bool) -> Arc<ArrowSchema> {
    let mut fields = vec![
        Field::new(SPILL_TOKEN_ID_COL, DataType::UInt32, false),
        Field::new(
            SPILL_DOC_IDS_COL,
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            false,
        ),
        Field::new(
            SPILL_FREQS_COL,
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            false,
        ),
    ];
    if with_position {
        let position_item = Field::new("item", DataType::UInt32, true);
        let position_list = Field::new("item", DataType::List(Arc::new(position_item)), true);
        fields.push(Field::new(
            SPILL_POSITIONS_COL,
            DataType::List(Arc::new(position_list)),
            true,
        ));
    }
    Arc::new(ArrowSchema::new(fields))
}

fn spill_batch_from_buffer(
    buffer: &mut SpillBuffer,
    schema: Arc<ArrowSchema>,
) -> Result<RecordBatch> {
    let token_ids = UInt32Array::from(std::mem::take(&mut buffer.token_ids));
    let doc_ids = build_list_array(std::mem::take(&mut buffer.doc_ids));
    let freqs = build_list_array(std::mem::take(&mut buffer.freqs));
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(token_ids) as ArrayRef,
        Arc::new(doc_ids) as ArrayRef,
        Arc::new(freqs) as ArrayRef,
    ];
    if let Some(positions) = buffer.positions.as_mut() {
        let positions = build_positions_array(std::mem::take(positions));
        columns.push(Arc::new(positions) as ArrayRef);
    }
    buffer.clear();
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn build_list_array(values: Vec<Vec<u32>>) -> ListArray {
    let mut builder = ListBuilder::new(UInt32Builder::new());
    for value in values {
        builder.values().append_slice(&value);
        builder.append(true);
    }
    builder.finish()
}

fn build_positions_array(values: Vec<Vec<Vec<u32>>>) -> ListArray {
    let mut builder = ListBuilder::new(ListBuilder::new(UInt32Builder::new()));
    for positions_per_doc in values {
        for positions in positions_per_doc {
            builder.values().values().append_slice(&positions);
            builder.values().append(true);
        }
        builder.append(true);
    }
    builder.finish()
}

fn estimate_partition_size(part: &InvertedPartition, with_position: bool) -> u64 {
    let lengths = part.inverted_list.posting_lengths();
    let total_len: u64 = lengths.iter().map(|len| *len as u64).sum();
    let bytes_per_posting = if with_position { 12 } else { 8 };
    total_len * bytes_per_posting
}

fn build_buckets(
    token_lengths: &[u64],
    limit: u64,
    partition_id: u64,
) -> (Vec<BucketRange>, Vec<usize>) {
    if token_lengths.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut buckets = Vec::new();
    let mut start = 0usize;
    let mut acc = 0u64;
    for (idx, len) in token_lengths.iter().enumerate() {
        if idx > start && acc + *len > limit {
            buckets.push(BucketRange {
                start: start as u32,
                end: idx as u32,
                file_name: format!("merge_spill_{}_{}", partition_id, buckets.len()),
            });
            start = idx;
            acc = 0;
        }
        acc += *len;
    }
    buckets.push(BucketRange {
        start: start as u32,
        end: token_lengths.len() as u32,
        file_name: format!("merge_spill_{}_{}", partition_id, buckets.len()),
    });
    let mut bucket_for_token = vec![0usize; token_lengths.len()];
    for (bucket_idx, bucket) in buckets.iter().enumerate() {
        for token_id in bucket.start..bucket.end {
            bucket_for_token[token_id as usize] = bucket_idx;
        }
    }
    (buckets, bucket_for_token)
}

fn collect_posting_list(
    posting_list: &PostingList,
    doc_id_offset: u32,
    with_position: bool,
) -> (Vec<u32>, Vec<u32>, Option<Vec<Vec<u32>>>) {
    let mut doc_ids = Vec::with_capacity(posting_list.len());
    let mut freqs = Vec::with_capacity(posting_list.len());
    let mut positions = with_position.then(Vec::new);
    for (doc_id, freq, pos_iter) in posting_list.iter() {
        let new_doc_id = doc_id_offset + doc_id as u32;
        doc_ids.push(new_doc_id);
        freqs.push(freq);
        if with_position {
            let positions_vec = pos_iter
                .expect("positions are required")
                .collect::<Vec<_>>();
            positions
                .as_mut()
                .expect("positions buffer is missing")
                .push(positions_vec);
        }
    }
    (doc_ids, freqs, positions)
}

fn append_spill_batch(
    batch: &RecordBatch,
    bucket_start: u32,
    with_position: bool,
    posting_lists: &mut [PostingListBuilder],
) -> Result<()> {
    let token_ids = batch
        .column_by_name(SPILL_TOKEN_ID_COL)
        .expect("spill token id column missing")
        .as_primitive::<UInt32Type>();
    let doc_ids = batch
        .column_by_name(SPILL_DOC_IDS_COL)
        .expect("spill doc ids column missing")
        .as_list::<i32>();
    let freqs = batch
        .column_by_name(SPILL_FREQS_COL)
        .expect("spill freqs column missing")
        .as_list::<i32>();
    let positions = if with_position {
        Some(
            batch
                .column_by_name(SPILL_POSITIONS_COL)
                .expect("spill positions column missing")
                .as_list::<i32>()
                .clone(),
        )
    } else {
        None
    };

    for row in 0..batch.num_rows() {
        let token_id = token_ids.value(row);
        let builder_idx = (token_id - bucket_start) as usize;
        let builder = &mut posting_lists[builder_idx];

        let doc_list_value = doc_ids.value(row);
        let doc_list = doc_list_value.as_primitive::<UInt32Type>();
        let freq_list_value = freqs.value(row);
        let freq_list = freq_list_value.as_primitive::<UInt32Type>();
        debug_assert_eq!(doc_list.len(), freq_list.len());
        let positions_list = if with_position {
            Some(positions.as_ref().expect("positions missing").value(row))
        } else {
            None
        };
        for idx in 0..doc_list.len() {
            let doc_id = doc_list.value(idx);
            let freq = freq_list.value(idx);
            let position_recorder = if with_position {
                let positions_value = positions_list
                    .as_ref()
                    .expect("positions missing")
                    .as_list::<i32>()
                    .value(idx);
                let positions_array = positions_value.as_primitive::<UInt32Type>();
                let positions_vec = positions_array.values().to_vec();
                PositionRecorder::Position(positions_vec.into())
            } else {
                PositionRecorder::Count(freq)
            };
            builder.add(doc_id, position_recorder);
        }
    }
    Ok(())
}

async fn write_posting_lists_to_writer(
    writer: &mut Box<dyn IndexWriter>,
    docs: Arc<DocSet>,
    posting_lists: Vec<PostingListBuilder>,
    schema: &Arc<ArrowSchema>,
    buffer: &mut Vec<RecordBatch>,
    size_sum: &mut usize,
) -> Result<()> {
    let mut batches = stream::iter(posting_lists)
        .map(|posting_list| {
            let block_max_scores = docs.calculate_block_max_scores(
                posting_list.doc_ids.iter(),
                posting_list.frequencies.iter(),
            );
            spawn_cpu(move || posting_list.to_batch(block_max_scores))
        })
        .buffered(get_num_compute_intensive_cpus());

    while let Some(batch) = batches.try_next().await? {
        *size_sum += batch.get_array_memory_size();
        buffer.push(batch);
        if *size_sum >= *super::builder::LANCE_FTS_FLUSH_SIZE << 20 {
            let batch = concat_batches(schema, buffer.iter())?;
            buffer.clear();
            *size_sum = 0;
            writer.write_record_batch(batch).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use std::sync::Arc;

    async fn run_merge(streaming: bool) -> Result<InvertedPartition> {
        let src_dir = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let token_set_format = TokenSetFormat::default();

        let mut builder0 = InnerBuilder::new(0, false, token_set_format);
        let apple_id = builder0.tokens.add("apple".to_owned());
        let banana_id = builder0.tokens.add("banana".to_owned());
        builder0
            .posting_lists
            .resize_with(builder0.tokens.len(), || PostingListBuilder::new(false));
        let doc_id = builder0.docs.append(10, 2);
        builder0.posting_lists[apple_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder0.posting_lists[banana_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder0.write(src_store.as_ref()).await?;

        let mut builder1 = InnerBuilder::new(1, false, token_set_format);
        let banana_id = builder1.tokens.add("banana".to_owned());
        let carrot_id = builder1.tokens.add("carrot".to_owned());
        builder1
            .posting_lists
            .resize_with(builder1.tokens.len(), || PostingListBuilder::new(false));
        let doc_id = builder1.docs.append(20, 2);
        builder1.posting_lists[banana_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder1.posting_lists[carrot_id as usize].add(doc_id, PositionRecorder::Count(1));
        builder1.write(src_store.as_ref()).await?;

        let partition0 = InvertedPartition::load(
            src_store.clone(),
            0,
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;
        let partition1 = InvertedPartition::load(
            src_store.clone(),
            1,
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;

        let mut merger = SizeBasedMerger::new(
            dest_store.as_ref(),
            vec![partition0, partition1],
            u64::MAX,
            token_set_format,
        )
        .with_streaming(streaming);
        let merged_partitions = merger.merge().await?;
        assert_eq!(merged_partitions, vec![2]);

        let merged = InvertedPartition::load(
            dest_store.clone(),
            merged_partitions[0],
            None,
            &LanceCache::no_cache(),
            token_set_format,
        )
        .await?;

        Ok(merged)
    }

    #[tokio::test]
    async fn test_merge_reuses_token_ids_for_shared_tokens() -> Result<()> {
        let merged = run_merge(false).await?;
        assert_eq!(merged.tokens.len(), 3);
        assert_eq!(merged.docs.len(), 2);
        assert_eq!(merged.docs.row_id(0), 10);
        assert_eq!(merged.docs.row_id(1), 20);

        let banana_token_id = merged.tokens.get("banana").unwrap();
        let posting = merged
            .inverted_list
            .posting_list(banana_token_id, false, &NoOpMetricsCollector)
            .await?;
        let doc_ids: Vec<u64> = posting.iter().map(|(doc_id, _, _)| doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_streaming_reuses_token_ids_for_shared_tokens() -> Result<()> {
        let merged = run_merge(true).await?;
        assert_eq!(merged.tokens.len(), 3);
        assert_eq!(merged.docs.len(), 2);
        assert_eq!(merged.docs.row_id(0), 10);
        assert_eq!(merged.docs.row_id(1), 20);

        let banana_token_id = merged.tokens.get("banana").unwrap();
        let posting = merged
            .inverted_list
            .posting_list(banana_token_id, false, &NoOpMetricsCollector)
            .await?;
        let doc_ids: Vec<u64> = posting.iter().map(|(doc_id, _, _)| doc_id).collect();
        assert_eq!(doc_ids, vec![0, 1]);
        Ok(())
    }
}
