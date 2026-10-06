// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Bytes, BytesMut};
use lance_arrow::DataTypeExt;
use lance_file::format::{MAJOR_VERSION, MINOR_VERSION};
use lance_file::{
    previous::writer::ManifestProvider as PreviousManifestProvider, version::LanceFileVersion,
};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use rayon::prelude::*;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::{ops::Range, sync::Arc};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tracing::instrument;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::{
    encodings::{Encoder, binary::BinaryEncoder, plain::PlainEncoder},
    object_store::ObjectStore,
    object_writer::WriteResult,
    traits::{WriteExt, Writer},
    utils::read_message,
};

use crate::format::{
    DataStorageFormat, Fragment, IndexMetadata, MAGIC, Manifest, Transaction, pb,
    pb_manifest_without_fragments,
};

use super::commit::ManifestLocation;

/// Read Manifest on URI.
///
/// This only reads manifest files. It does not read data files.
#[instrument(level = "debug", skip(object_store))]
pub async fn read_manifest(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<Manifest> {
    let file_size = if let Some(known_size) = known_size {
        known_size
    } else {
        object_store.inner.head(path).await?.size
    };
    const PREFETCH_SIZE: u64 = 64 * 1024;
    let initial_start = file_size.saturating_sub(PREFETCH_SIZE);
    let range = Range {
        start: initial_start,
        end: file_size,
    };
    let buf = object_store.inner.get_range(path, range).await?;

    // In case of corruption, the known_size might be wrong. We can retry without
    // the size to be more robust.
    if (buf.len() < 16 || !buf.ends_with(MAGIC)) && known_size.is_some() {
        return Box::pin(read_manifest(object_store, path, None)).await;
    }

    if buf.len() < 16 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: file size is smaller than 16 bytes".to_string(),
        ));
    }
    if !buf.ends_with(MAGIC) {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: magic number does not match".to_string(),
        ));
    }
    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    let manifest_len = file_size as usize - manifest_pos;

    let buf: Bytes = if manifest_len <= buf.len() {
        // The prefetch captured the entire manifest. We just need to trim the buffer.
        buf.slice(buf.len() - manifest_len..buf.len())
    } else {
        // The prefetch only captured part of the manifest. We need to make an
        // additional range request to read the remainder.
        let mut buf2: BytesMut = object_store
            .inner
            .get_range(
                path,
                Range {
                    start: manifest_pos as u64,
                    end: file_size - PREFETCH_SIZE,
                },
            )
            .await?
            .into_iter()
            .collect();
        buf2.extend_from_slice(&buf);
        buf2.freeze()
    };

    let recorded_length = LittleEndian::read_u32(&buf[0..4]) as usize;
    // Need to trim the magic number at end and message length at beginning
    let buf = buf.slice(4..buf.len() - 16);

    if buf.len() != recorded_length {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {}, got {}",
            recorded_length,
            buf.len()
        )));
    }

    let proto = pb::Manifest::decode(buf)?;
    Manifest::try_from(proto)
}

#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_indexes(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Vec<IndexMetadata>> {
    if let Some(pos) = manifest.index_section.as_ref() {
        let result = read_index_section(object_store, &location.path, location.size, *pos).await;
        // A stale cached size makes the index offset fall outside the sized view,
        // so the read fails as "file size is too small". Retry once with the true
        // size; surface any other error unchanged.
        let section = match result {
            Err(e)
                if location.size.is_some() && e.to_string().contains("file size is too small") =>
            {
                read_index_section(object_store, &location.path, None, *pos).await?
            }
            other => other?,
        };

        let indices = section
            .indices
            .into_iter()
            .map(IndexMetadata::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(indices)
    } else {
        Ok(vec![])
    }
}

/// Read the index section message at `pos`, opening the manifest with a known
/// size when one is provided.
async fn read_index_section(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
    pos: usize,
) -> Result<pb::IndexSection> {
    let reader = if let Some(size) = size {
        object_store.open_with_size(path, size as usize).await?
    } else {
        object_store.open(path).await?
    };
    read_message(reader.as_ref(), pos).await
}

/// Write the index section and inline transaction, if present, and record
/// their positions in `manifest`.
async fn write_index_and_transaction(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    mut transaction: Option<Transaction>,
) -> Result<()> {
    // Write indices if presented.
    if let Some(indices) = indices.as_ref() {
        let section = pb::IndexSection {
            indices: indices.iter().map(|i| i.into()).collect(),
        };
        let pos = writer.write_protobuf(&section).await?;
        manifest.index_section = Some(pos);
    }

    // Write inline transaction if presented.
    if let Some(tx) = transaction.take() {
        // Convert to protobuf at the write boundary to persist inline
        let pb_tx: pb::Transaction = tx.into();
        let pos = writer.write_protobuf(&pb_tx).await?;
        manifest.transaction_section = Some(pos);
    }
    Ok(())
}

/// Field key of `Manifest.fragments` in table.proto: field 2, wire type 2
/// (length-delimited).
const MANIFEST_FRAGMENTS_KEY: u8 = (2 << 3) | 2;

/// The u32 length prefix of a manifest message of `len` bytes.
///
/// Readers take the prefix as the message length, so a message that does not
/// fit in a u32 cannot be written.
fn manifest_message_len_prefix(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        Error::invalid_input(format!(
            "Manifest message is {len} bytes, which exceeds the {} byte limit of its \
             u32 length prefix",
            u32::MAX
        ))
    })
}

/// Whether any data file of `fragment` has an unknown size.
///
/// File sizes are cached in place once learned, so the encoding of such a
/// fragment can change without the fragment list changing. It is encoded on
/// every write instead of cached.
fn has_unknown_file_size(fragment: &Fragment) -> bool {
    fragment
        .files
        .iter()
        .chain(fragment.overlays.iter().map(|overlay| &overlay.data_file))
        .any(|file| file.file_size_bytes.get().is_none())
}

/// Append `manifest`'s protobuf message to `buf`, preceded by its u32
/// little-endian length, the layout [`WriteExt::write_protobuf`] produces.
///
/// Fragments are appended one `fragments` field at a time: bytes cached in
/// [`Manifest::encoded_fragments`] are copied as they are, and only the other
/// fragments are encoded. Fields go in prost's order (`fields`, `fragments`,
/// then the rest), so the message decodes to the same `pb::Manifest` as
/// `pb::Manifest::from(manifest)` and its bytes differ at most in the order of
/// map entries.
///
/// Returns, for each fragment, the range of `buf` holding its encoded bytes,
/// or `None` when those bytes must not be reused.
///
/// When many fragments have no cached bytes (the first write after a manifest
/// is loaded), they are encoded in parallel in chunks that are then appended
/// in order.
fn encode_manifest_message(
    manifest: &Manifest,
    buf: &mut Vec<u8>,
) -> Result<Vec<Option<Range<usize>>>> {
    let fragments = manifest.fragments.as_slice();
    let cached = manifest.encoded_fragments();
    let cached_len: usize = cached.map_or(0, |cached| {
        cached.iter().flatten().map(|bytes| bytes.len() + 6).sum()
    });
    buf.reserve(cached_len + 64 * 1024);

    let len_pos = buf.len();
    buf.extend_from_slice(&[0; 4]);
    let msg_start = buf.len();

    let mut rest = pb_manifest_without_fragments(manifest);
    let head = pb::Manifest {
        fields: std::mem::take(&mut rest.fields),
        ..Default::default()
    };
    head.encode(buf)?;

    let num_uncached = cached.map_or(fragments.len(), |cached| {
        cached.iter().filter(|bytes| bytes.is_none()).count()
    });
    let mut ranges = Vec::with_capacity(fragments.len());
    if num_uncached < PARALLEL_ENCODE_MIN_FRAGMENTS {
        append_fragments(fragments, cached, buf, &mut ranges)?;
    } else {
        let chunks = fragments
            .par_chunks(PARALLEL_ENCODE_CHUNK_FRAGMENTS)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let start = chunk_index * PARALLEL_ENCODE_CHUNK_FRAGMENTS;
                let cached = cached.map(|cached| &cached[start..start + chunk.len()]);
                let mut chunk_buf = Vec::new();
                let mut chunk_ranges = Vec::with_capacity(chunk.len());
                append_fragments(chunk, cached, &mut chunk_buf, &mut chunk_ranges)?;
                Ok((chunk_buf, chunk_ranges))
            })
            .collect::<Result<Vec<_>>>()?;
        buf.reserve(chunks.iter().map(|(chunk_buf, _)| chunk_buf.len()).sum());
        for (chunk_buf, chunk_ranges) in chunks {
            let offset = buf.len();
            buf.extend_from_slice(&chunk_buf);
            ranges.extend(
                chunk_ranges
                    .into_iter()
                    .map(|range| range.map(|range| range.start + offset..range.end + offset)),
            );
        }
    }

    rest.encode(buf)?;

    let msg_len = manifest_message_len_prefix(buf.len() - msg_start)?;
    buf[len_pos..msg_start].copy_from_slice(&msg_len.to_le_bytes());
    Ok(ranges)
}

/// Fewest fragments without cached bytes for which encoding runs in parallel.
const PARALLEL_ENCODE_MIN_FRAGMENTS: usize = 64 * 1024;

/// Fragments per chunk when encoding in parallel.
const PARALLEL_ENCODE_CHUNK_FRAGMENTS: usize = 16 * 1024;

/// Append each fragment to `buf` as a `Manifest.fragments` field, copying
/// `cached[i]` when present and encoding `fragments[i]` otherwise, and push the
/// range of its encoded bytes in `buf` (or `None` if they must not be reused)
/// to `ranges`.
fn append_fragments(
    fragments: &[Fragment],
    cached: Option<&[Option<Bytes>]>,
    buf: &mut Vec<u8>,
    ranges: &mut Vec<Option<Range<usize>>>,
) -> Result<()> {
    for (i, fragment) in fragments.iter().enumerate() {
        buf.push(MANIFEST_FRAGMENTS_KEY);
        let range = match cached.and_then(|cached| cached[i].as_ref()) {
            Some(bytes) => {
                prost::encode_length_delimiter(bytes.len(), buf)?;
                let start = buf.len();
                buf.extend_from_slice(bytes);
                start..buf.len()
            }
            None => {
                let delimiter_pos = buf.len();
                pb::DataFragment::from(fragment).encode_length_delimited(buf)?;
                let len = prost::decode_length_delimiter(&buf[delimiter_pos..])?;
                buf.len() - len..buf.len()
            }
        };
        ranges.push((!has_unknown_file_size(fragment)).then_some(range));
    }
    Ok(())
}

/// Keep the encoded fragments written in `buf` on `manifest` for the next write.
fn cache_encoded_fragments(
    manifest: &mut Manifest,
    buf: &Bytes,
    ranges: Vec<Option<Range<usize>>>,
) {
    let encoded = ranges
        .into_iter()
        .map(|range| range.map(|range| buf.slice(range)))
        .collect();
    manifest.set_encoded_fragments(encoded);
}

/// Write the manifest message and return its position.
async fn write_manifest_message(writer: &mut dyn Writer, manifest: &mut Manifest) -> Result<usize> {
    let pos = writer.tell().await?;
    let mut buf = Vec::new();
    let ranges = encode_manifest_message(manifest, &mut buf)?;
    writer.write_all(&buf).await?;
    cache_encoded_fragments(manifest, &Bytes::from(buf), ranges);
    Ok(pos)
}

async fn do_write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<usize> {
    write_index_and_transaction(writer, manifest, indices, transaction).await?;
    write_manifest_message(writer, manifest).await
}

/// Write manifest to an open file.
pub async fn write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<usize> {
    write_dictionaries(writer, manifest).await?;
    do_write_manifest(writer, manifest, indices, transaction).await
}

/// Serialize a complete manifest file into one buffer: the bytes
/// [`write_manifest`] followed by `write_magics` produce, for commit handlers
/// that upload the finished file in one request.
pub async fn write_manifest_file_to_buffer(
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<Bytes> {
    let mut writer = BufferWriter::default();
    write_dictionaries(&mut writer, manifest).await?;
    write_index_and_transaction(&mut writer, manifest, indices, transaction).await?;
    let pos = writer.buf.len();
    let ranges = encode_manifest_message(manifest, &mut writer.buf)?;
    writer
        .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
        .await?;
    let buf = Bytes::from(writer.buf);
    cache_encoded_fragments(manifest, &buf, ranges);
    Ok(buf)
}

/// A [`Writer`] that appends to an in-memory buffer.
#[derive(Default)]
struct BufferWriter {
    buf: Vec<u8>,
}

impl AsyncWrite for BufferWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.buf).poll_write(cx, data)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.buf).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.buf).poll_shutdown(cx)
    }
}

#[async_trait]
impl Writer for BufferWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.buf.len())
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        Ok(WriteResult {
            size: self.buf.len(),
            e_tag: None,
        })
    }
}

/// Write the dictionary values of a legacy-format schema and record their
/// positions in the schema.
async fn write_dictionaries(writer: &mut dyn Writer, manifest: &mut Manifest) -> Result<()> {
    // Write dictionary values.
    let max_field_id = manifest.schema.max_field_id().unwrap_or(-1);
    let is_legacy_storage = manifest.should_use_legacy_format();
    for field_id in 0..max_field_id + 1 {
        if let Some(field) = manifest.schema.mut_field_by_id(field_id)
            && field.data_type().is_dictionary()
            && is_legacy_storage
        {
            let dict_info = field.dictionary.as_mut().ok_or_else(|| {
                Error::io(format!("Lance field {} misses dictionary info", field.name))
            })?;

            let value_arr = dict_info.values.as_ref().ok_or_else(|| {
                Error::io(format!(
                    "Lance field {} is dictionary type, but misses the dictionary value array",
                    field.name
                ))
            })?;

            let data_type = value_arr.data_type();
            let pos = match data_type {
                dt if dt.is_numeric() => {
                    let mut encoder = PlainEncoder::new(writer, dt);
                    encoder.encode(&[value_arr]).await?
                }
                dt if dt.is_binary_like() => {
                    let mut encoder = BinaryEncoder::new(writer);
                    encoder.encode(&[value_arr]).await?
                }
                _ => {
                    return Err(Error::schema(format!(
                        "Does not support {} as dictionary value type",
                        value_arr.data_type()
                    )));
                }
            };
            dict_info.offset = pos;
            dict_info.length = value_arr.len();
        }
    }
    Ok(())
}

/// Implementation of ManifestProvider that describes a Lance file by writing
/// a manifest that contains nothing but default fields and the schema
pub struct ManifestDescribing {}

#[async_trait]
impl PreviousManifestProvider for ManifestDescribing {
    async fn store_schema(
        object_writer: &mut dyn Writer,
        schema: &Schema,
    ) -> Result<Option<usize>> {
        let mut manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::new(LanceFileVersion::Legacy),
            HashMap::new(),
        );
        let pos = do_write_manifest(object_writer, &mut manifest, None, None).await?;
        Ok(Some(pos))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatch};
    use std::collections::HashMap;

    use crate::format::SelfDescribingFileReader;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
    use lance_file::previous::{
        reader::FileReader as PreviousFileReader, writer::FileWriter as PreviousFileWriter,
    };
    use rand::{Rng, distr::Alphanumeric};
    use tokio::io::AsyncWriteExt;

    use super::*;

    async fn test_roundtrip_manifest(prefix_size: usize, manifest_min_size: usize) {
        let store = ObjectStore::memory();
        let path = Path::from("/read_large_manifest");

        let mut writer = store.create(&path).await.unwrap();

        // Write prefix we should ignore
        let prefix: Vec<u8> = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(prefix_size)
            .collect();
        writer.write_all(&prefix).await.unwrap();

        let long_name: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(manifest_min_size)
            .map(char::from)
            .collect();

        let arrow_schema =
            ArrowSchema::new(vec![ArrowField::new(long_name, DataType::Int64, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let mut config = HashMap::new();
        config.insert("key".to_string(), "value".to_string());

        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        let pos = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let roundtripped_manifest = read_manifest(&store, &path, None).await.unwrap();

        assert_eq!(manifest, roundtripped_manifest);

        store.inner.delete(&path).await.unwrap();
    }

    #[tokio::test]
    async fn test_read_large_manifest() {
        test_roundtrip_manifest(0, 100_000).await;
        test_roundtrip_manifest(1000, 100_000).await;
        test_roundtrip_manifest(1000, 1000).await;
    }

    #[tokio::test]
    async fn test_update_schema_metadata() {
        let store = ObjectStore::memory();
        let path = Path::from("/update_schema_metadata");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = PreviousFileWriter::<ManifestDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = Int32Array::from_iter_values(0..10);
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array)]).unwrap();
        file_writer
            .write(std::slice::from_ref(&batch))
            .await
            .unwrap();
        let mut metadata = HashMap::new();
        metadata.insert(String::from("lance:extra"), String::from("for_test"));
        file_writer.finish_with_metadata(&metadata).await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let reader = PreviousFileReader::try_new_self_described_from_reader(reader.into(), None)
            .await
            .unwrap();
        let schema = ArrowSchema::from(reader.schema());
        assert_eq!(schema.metadata().get("lance:extra").unwrap(), "for_test");
    }

    /// A manifest whose fragments exercise most `DataFragment` fields.
    fn manifest_with_fragments(num_fragments: u64) -> Manifest {
        use crate::format::{DeletionFile, DeletionFileType, Fragment, RowIdMeta};

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, false),
            ArrowField::new("b", DataType::Utf8, true),
        ]);
        let fragments = (0..num_fragments)
            .map(|id| {
                let mut fragment = Fragment::new(id).with_physical_rows(1000 + id as usize);
                for field_id in 0..2 {
                    fragment.add_file(
                        format!("data/{id}-{field_id}.lance"),
                        vec![field_id],
                        vec![field_id],
                        &LanceFileVersion::V2_0,
                        std::num::NonZero::new(4096 + id),
                    );
                }
                if id % 3 == 0 {
                    fragment.deletion_file = Some(DeletionFile {
                        read_version: id,
                        id,
                        file_type: DeletionFileType::Bitmap,
                        num_deleted_rows: Some(7),
                        base_id: None,
                    });
                }
                fragment.row_id_meta = Some(RowIdMeta::Inline(vec![id as u8; 5]));
                fragment
            })
            .collect();
        let mut manifest = Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(fragments),
            DataStorageFormat::new(LanceFileVersion::V2_0),
            HashMap::new(),
        );
        // One entry per map keeps prost's map encoding order deterministic,
        // so messages can be compared byte for byte.
        manifest
            .config_mut()
            .insert("key".to_string(), "value".to_string());
        manifest.version = 42;
        manifest.update_max_fragment_id();
        manifest
    }

    /// The length-prefixed message `write_protobuf` writes for `manifest`.
    fn prost_manifest_message(manifest: &Manifest) -> Vec<u8> {
        let message = pb::Manifest::from(manifest).encode_to_vec();
        let mut buf = (message.len() as u32).to_le_bytes().to_vec();
        buf.extend_from_slice(&message);
        buf
    }

    fn encode_message(manifest: &mut Manifest) -> Vec<u8> {
        let mut buf = Vec::new();
        let ranges = encode_manifest_message(manifest, &mut buf).unwrap();
        let bytes = Bytes::from(buf.clone());
        cache_encoded_fragments(manifest, &bytes, ranges);
        buf
    }

    #[test]
    fn test_encode_manifest_message_matches_prost() {
        let mut manifest = manifest_with_fragments(20);
        let expected = prost_manifest_message(&manifest);

        // Cold: every fragment is encoded.
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(encode_message(&mut manifest), expected);
        let cached = manifest.encoded_fragments().unwrap();
        assert!(cached.iter().all(Option::is_some));

        // Warm: every fragment is copied from the cache.
        assert_eq!(encode_message(&mut manifest), expected);

        // Partly warm: fragments without cached bytes are encoded.
        let partial = manifest
            .encoded_fragments()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, bytes)| bytes.clone().filter(|_| i % 4 != 0))
            .collect();
        manifest.set_encoded_fragments(partial);
        assert_eq!(encode_message(&mut manifest), expected);

        // No fragments.
        let mut empty = manifest_with_fragments(0);
        assert_eq!(encode_message(&mut empty), prost_manifest_message(&empty));
    }

    #[test]
    fn test_encode_manifest_message_parallel_matches_prost() {
        let num_fragments = (PARALLEL_ENCODE_MIN_FRAGMENTS + 1234) as u64;
        let mut manifest = manifest_with_fragments(num_fragments);
        let expected = prost_manifest_message(&manifest);

        // Cold: all fragments uncached, encoded in parallel chunks.
        assert_eq!(encode_message(&mut manifest), expected);
        let cached = manifest.encoded_fragments().unwrap();
        assert_eq!(cached.len(), num_fragments as usize);
        assert!(cached.iter().all(Option::is_some));

        // Warm: copied sequentially from the parallel-encoded cache.
        assert_eq!(encode_message(&mut manifest), expected);
    }

    #[test]
    fn test_encode_manifest_message_ignores_cache_of_replaced_fragments() {
        let mut manifest = manifest_with_fragments(10);
        encode_message(&mut manifest);
        assert!(manifest.encoded_fragments().is_some());

        // Replacing the fragment list, or mutating it, makes the cache stale.
        let mut fragments = manifest.fragments.as_ref().clone();
        fragments[3].physical_rows = Some(1);
        manifest.fragments = Arc::new(fragments);
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(
            encode_message(&mut manifest),
            prost_manifest_message(&manifest)
        );

        Arc::make_mut(&mut manifest.fragments)[4].physical_rows = Some(2);
        assert!(manifest.encoded_fragments().is_none());
        assert_eq!(
            encode_message(&mut manifest),
            prost_manifest_message(&manifest)
        );
    }

    #[test]
    fn test_manifest_message_len_prefix_rejects_oversized_messages() {
        assert_eq!(manifest_message_len_prefix(0).unwrap(), 0);
        assert_eq!(
            manifest_message_len_prefix(u32::MAX as usize).unwrap(),
            u32::MAX
        );
        let err = manifest_message_len_prefix(u32::MAX as usize + 1).unwrap_err();
        assert!(
            err.to_string().contains("exceeds"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_write_manifest_file_to_buffer_matches_object_writer() {
        let store = ObjectStore::memory();
        for indices in [None, Some(vec![])] {
            let path = Path::from("manifest");
            let mut written = manifest_with_fragments(50);
            let mut writer = store.create(&path).await.unwrap();
            let pos = write_manifest(writer.as_mut(), &mut written, indices.clone(), None)
                .await
                .unwrap();
            writer
                .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
                .await
                .unwrap();
            Writer::shutdown(writer.as_mut()).await.unwrap();
            let expected = store.read_one_all(&path).await.unwrap();

            let mut buffered = manifest_with_fragments(50);
            let file = write_manifest_file_to_buffer(&mut buffered, indices, None)
                .await
                .unwrap();
            assert_eq!(file, expected);
            assert_eq!(buffered, written);
            assert!(buffered.encoded_fragments().is_some());

            let read_back = read_manifest(&store, &path, None).await.unwrap();
            assert_eq!(read_back, buffered);
        }
    }
}
