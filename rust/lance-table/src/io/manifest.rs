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
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::{ops::Range, sync::Arc};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tracing::instrument;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::{
    encodings::{Encoder, binary::BinaryEncoder, plain::PlainEncoder},
    object_store::ObjectStore,
    object_writer::WriteResult,
    traits::{Reader, WriteExt, Writer},
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

    let proto = pb::Manifest::decode(manifest_message_body(buf)?)?;
    Manifest::try_from(proto)
}

/// Bytes of the little-endian u32 length prefix that precedes every section
/// message of a manifest file (index section, inline transaction, manifest).
const LEN_PREFIX_LEN: usize = 4;

/// Bytes of the footer that follows the manifest message: its i64 position,
/// the u16 major and minor versions and the magic.
const FOOTER_LEN: usize = 16;

/// Whether `recorded`, a section message's u32 length prefix, is valid for a
/// message of `actual` bytes: the prefix holds the length modulo 2^32.
///
/// The prefix of a message over [`u32::MAX`] bytes keeps only the low 32 bits
/// of its length (see [`section_len_prefix`]), so readers take the length from
/// the file layout and check the prefix against it. Any other mismatch means
/// the file is corrupt or the message is not where the layout says.
pub fn section_len_matches(recorded: u32, actual: usize) -> bool {
    // Truncation intended: compare the low 32 bits.
    actual as u32 == recorded
}

/// The body of the manifest message, given the bytes of a manifest file from
/// the message's position (its length prefix) to the end of the file.
///
/// The message runs up to the footer, so its length is known from the layout;
/// the prefix is checked against it with [`section_len_matches`].
pub fn manifest_message_body(tail: Bytes) -> Result<Bytes> {
    if tail.len() < LEN_PREFIX_LEN + FOOTER_LEN {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest message and footer take {} bytes, fewer than the {} \
             of a length prefix and footer",
            tail.len(),
            LEN_PREFIX_LEN + FOOTER_LEN
        )));
    }
    let recorded = LittleEndian::read_u32(&tail[..LEN_PREFIX_LEN]);
    let body = tail.slice(LEN_PREFIX_LEN..tail.len() - FOOTER_LEN);
    if !section_len_matches(recorded, body.len()) {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {} (mod 2^32), got {}",
            recorded,
            body.len()
        )));
    }
    Ok(body)
}

/// Read the section message (index section or inline transaction) at `pos`
/// of a manifest file.
///
/// `next_section` is the position of the section written right after it, if
/// any; otherwise the manifest message follows it. Its position is confirmed
/// from the first block when that block holds the u32 just past the message
/// and that u32 is a valid manifest prefix for the reader's size (see
/// [`confirmed_manifest_position`]), the common case of a small section. Only
/// otherwise is it read from the footer, which may take one more request.
///
/// See [`section_message_len`] for how the length is chosen. Fails with
/// "file size is too small" when `reader`'s size ends before the message or
/// does not end at a footer, as a stale cached size would, so callers can retry
/// with the true size.
pub async fn read_section_message<M: Message + Default>(
    reader: &dyn Reader,
    pos: usize,
    next_section: Option<usize>,
) -> Result<M> {
    let file_size = reader.size().await?;
    if pos + LEN_PREFIX_LEN > file_size {
        return Err(Error::io("file size is too small".to_string()));
    }
    let first = pos..file_size.min(pos + reader.block_size());
    let buf = reader.get_range(first.clone()).await?;
    let recorded = LittleEndian::read_u32(&buf);
    // A manifest message (at least its prefix) and the footer follow every section.
    let room = file_size.saturating_sub(pos + LEN_PREFIX_LEN + LEN_PREFIX_LEN + FOOTER_LEN);
    let layout_end = match next_section.filter(|&next| next > pos) {
        Some(next) => next,
        None => match confirmed_manifest_position(&buf, pos, recorded, file_size, room) {
            Some(manifest_pos) => manifest_pos,
            None if first.end == file_size => manifest_position_from_footer(&buf)?,
            None => {
                let footer = file_size - FOOTER_LEN.min(file_size)..file_size;
                manifest_position_from_footer(&reader.get_range(footer).await?)?
            }
        },
    };
    let len = section_message_len(pos, recorded, layout_end, room)?;

    let end = pos + LEN_PREFIX_LEN + len;
    if end > file_size {
        return Err(Error::io("file size is too small".to_string()));
    }
    let buf = if end <= first.end {
        buf
    } else {
        let rest = reader.get_range(first.end..end).await?;
        let mut joined = BytesMut::with_capacity(end - pos);
        joined.extend_from_slice(&buf);
        joined.extend_from_slice(&rest);
        joined.freeze()
    };
    Ok(M::decode(&buf[LEN_PREFIX_LEN..LEN_PREFIX_LEN + len])?)
}

/// The length of the section message at `pos` whose u32 prefix is
/// `recorded`, given `layout_end`, where the file layout says the section
/// ends (the next section's position), and `room`, the most bytes the
/// message can take before the end of the reader.
///
/// - The distance to `layout_end` when the prefix matches it modulo 2^32
///   ([`section_len_matches`]). It is used even when it exceeds `room`, so a
///   stale reader size fails as too small instead of yielding a message cut at
///   the wrapped length.
/// - Otherwise the prefix, when `room` is too small for the true length to be
///   2^32 or more bytes longer: files whose sections are not laid out back to
///   back are read as before.
/// - Otherwise an error: the length is ambiguous and the layout disagrees.
fn section_message_len(pos: usize, recorded: u32, layout_end: usize, room: usize) -> Result<usize> {
    let layout_len = layout_end.checked_sub(pos + LEN_PREFIX_LEN);
    match layout_len {
        Some(len) if section_len_matches(recorded, len) => Ok(len),
        _ if !may_exceed_u32_prefix(recorded, room) => Ok(recorded as usize),
        _ => Err(Error::invalid_input(format!(
            "Invalid format: section length does not match. The section at {pos} ends at \
             {layout_end}, but its length prefix is {recorded} (mod 2^32)"
        ))),
    }
}

/// Whether a message with u32 prefix `recorded` and at most `room` bytes could
/// be 2^32 or more bytes longer than the prefix says.
fn may_exceed_u32_prefix(recorded: u32, room: usize) -> bool {
    (recorded as u64) + (1u64 << 32) <= room as u64
}

/// The position of the manifest message after the section at `pos`, when
/// `first`, the bytes read from `pos`, confirms it without the footer.
///
/// The prefix `recorded` must be unambiguous for the reader's `room`, and the
/// u32 just past the message it describes must be a valid prefix for a
/// manifest message running from there to a footer at `file_size`. A stale
/// reader size, or a wrapped prefix read against one, fails that check except
/// by a 2^-32 coincidence; the caller then reads the footer.
fn confirmed_manifest_position(
    first: &[u8],
    pos: usize,
    recorded: u32,
    file_size: usize,
    room: usize,
) -> Option<usize> {
    if may_exceed_u32_prefix(recorded, room) {
        return None;
    }
    let offset = LEN_PREFIX_LEN + recorded as usize;
    let manifest_prefix = first.get(offset..offset + LEN_PREFIX_LEN)?;
    let manifest_pos = pos + offset;
    let manifest_len = file_size.checked_sub(manifest_pos + LEN_PREFIX_LEN + FOOTER_LEN)?;
    section_len_matches(LittleEndian::read_u32(manifest_prefix), manifest_len)
        .then_some(manifest_pos)
}

/// The manifest message's position, read from `tail`, bytes that end where
/// the reader ends.
///
/// Missing magic means the reader does not end at the footer, as with a stale
/// cached size, and is reported as a too-small file size.
fn manifest_position_from_footer(tail: &[u8]) -> Result<usize> {
    if tail.len() < FOOTER_LEN || !tail.ends_with(MAGIC) {
        return Err(Error::io(
            "manifest footer not found at the end of the reader: file size is too small or the \
             file is corrupt"
                .to_string(),
        ));
    }
    let footer = &tail[tail.len() - FOOTER_LEN..];
    Ok(LittleEndian::read_i64(&footer[..8]) as usize)
}

#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_indexes(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Vec<IndexMetadata>> {
    if let Some(pos) = manifest.index_section.as_ref() {
        let next = manifest.transaction_section;
        let result =
            read_index_section(object_store, &location.path, location.size, *pos, next).await;
        // A stale cached size makes the index offset fall outside the sized view,
        // so the read fails as "file size is too small". Retry once with the true
        // size; surface any other error unchanged.
        let section = match result {
            Err(e)
                if location.size.is_some() && e.to_string().contains("file size is too small") =>
            {
                read_index_section(object_store, &location.path, None, *pos, next).await?
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

/// Read the index section message at `pos`, followed by the section at
/// `next_section` (if any), opening the manifest with a known size when one is
/// provided.
async fn read_index_section(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
    pos: usize,
    next_section: Option<usize>,
) -> Result<pb::IndexSection> {
    let reader = if let Some(size) = size {
        object_store.open_with_size(path, size as usize).await?
    } else {
        object_store.open(path).await?
    };
    read_section_message(reader.as_ref(), pos, next_section).await
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
        let pos = write_section_message(writer, "Index section", &section).await?;
        manifest.index_section = Some(pos);
    }

    // Write inline transaction if presented.
    if let Some(tx) = transaction.take() {
        // Convert to protobuf at the write boundary to persist inline
        let pb_tx: pb::Transaction = tx.into();
        let pos = write_section_message(writer, "Inline transaction", &pb_tx).await?;
        manifest.transaction_section = Some(pos);
    }
    Ok(())
}

/// Write `message` preceded by its u32 little-endian length prefix (see
/// [`section_len_prefix`]) and return its position.
async fn write_section_message(
    writer: &mut dyn Writer,
    what: &str,
    message: &impl Message,
) -> Result<usize> {
    let pos = writer.tell().await?;
    let body = message.encode_to_vec();
    let prefix = section_len_prefix(what, body.len())?;
    writer.write_all(&prefix.to_le_bytes()).await?;
    writer.write_all(&body).await?;
    Ok(pos)
}

/// Field key of `Manifest.fragments` in table.proto: field 2, wire type 2
/// (length-delimited).
const MANIFEST_FRAGMENTS_KEY: u8 = (2 << 3) | 2;

/// Environment variable that lets writers emit section messages (the manifest
/// message, index section and inline transaction) larger than [`u32::MAX`]
/// bytes. See [`section_len_prefix`].
pub const ALLOW_LARGE_MANIFEST_ENV: &str = "LANCE_ALLOW_LARGE_MANIFEST";

/// Whether [`ALLOW_LARGE_MANIFEST_ENV`] is set to `1` or `true`. Read once
/// per process.
fn large_manifest_allowed() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        std::env::var(ALLOW_LARGE_MANIFEST_ENV).is_ok_and(|value| {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true")
        })
    })
}

/// The u32 length prefix of a section message (`what`, for errors) of `len`
/// bytes.
///
/// A message over [`u32::MAX`] bytes is an error unless the environment
/// variable `LANCE_ALLOW_LARGE_MANIFEST` is `1` (read once per process); then
/// the prefix is the low 32 bits of the length. Readers that check the prefix
/// with [`section_len_matches`] take the true length from the file layout,
/// but older readers reject such a file or misread it, so set the variable
/// only once every reader of the table accepts it.
fn section_len_prefix(what: &str, len: usize) -> Result<u32> {
    section_len_prefix_with(what, len, large_manifest_allowed())
}

/// [`section_len_prefix`], with `allow_large` in place of the environment.
fn section_len_prefix_with(what: &str, len: usize, allow_large: bool) -> Result<u32> {
    match u32::try_from(len) {
        Ok(prefix) => Ok(prefix),
        // Truncation intended: the prefix holds the length modulo 2^32.
        Err(_) if allow_large => Ok(len as u32),
        Err(_) => Err(Error::invalid_input(format!(
            "{what} is {len} bytes, which exceeds the {} byte limit of its u32 length \
             prefix. Set {ALLOW_LARGE_MANIFEST_ENV}=1 to write it with the low 32 bits of its \
             length once every reader of the table derives the length from the file layout",
            u32::MAX
        ))),
    }
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

    let msg_len = section_len_prefix("Manifest message", buf.len() - msg_start)?;
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
        // Checked before encoding: sizes only ever go from unknown to known,
        // so a fragment whose sizes are all known here is encoded with them.
        // Checking afterwards could cache bytes that missed a size filled in
        // by a concurrent reader.
        let reusable = !has_unknown_file_size(fragment);
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
        ranges.push(reusable.then_some(range));
    }
    Ok(())
}

/// Keep the encoded fragments written in `buf` on `manifest` for the next write.
///
/// The cached slices keep all of `buf` alive, so `buf` should hold the
/// manifest message and nothing else.
fn cache_encoded_fragments(
    manifest: &mut Manifest,
    buf: &Bytes,
    ranges: Vec<Option<Range<usize>>>,
) {
    let encoded = ranges
        .into_iter()
        .map(|range| range.map(|range| buf.slice(range)))
        .collect();
    manifest.set_encoded_fragments(encoded, buf.len());
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

/// Serialize a complete manifest file in memory, for commit handlers that
/// upload the finished file in one request. The returned chunks, concatenated,
/// are the bytes [`write_manifest`] followed by `write_magics` produce.
///
/// The manifest message is a chunk of its own: the encoded fragments cached on
/// `manifest` are slices of it, and must not keep the index and transaction
/// sections alive.
pub async fn write_manifest_file_to_chunks(
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<Vec<Bytes>> {
    let mut sections = BufferWriter::default();
    write_dictionaries(&mut sections, manifest).await?;
    write_index_and_transaction(&mut sections, manifest, indices, transaction).await?;
    let pos = sections.buf.len();

    let mut message = Vec::new();
    let ranges = encode_manifest_message(manifest, &mut message)?;
    let message = Bytes::from(message);
    cache_encoded_fragments(manifest, &message, ranges);

    let mut trailer = BufferWriter::default();
    trailer
        .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
        .await?;
    Ok(vec![
        Bytes::from(sections.buf),
        message,
        Bytes::from(trailer.buf),
    ])
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
        manifest.set_encoded_fragments(partial, expected.len());
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
    fn test_section_len_prefix_requires_opt_in_above_u32_max() {
        let what = "Manifest message";
        for allow_large in [false, true] {
            assert_eq!(section_len_prefix_with(what, 0, allow_large).unwrap(), 0);
            assert_eq!(
                section_len_prefix_with(what, u32::MAX as usize, allow_large).unwrap(),
                u32::MAX
            );
        }

        let len = (1usize << 32) + 7;
        let err = section_len_prefix_with(what, len, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("exceeds") && err.contains(ALLOW_LARGE_MANIFEST_ENV),
            "unexpected error: {err}"
        );
        // The low 32 bits: the same bytes as a wrapping `len as u32` cast.
        assert_eq!(section_len_prefix_with(what, len, true).unwrap(), 7);
        assert_eq!(
            section_len_prefix_with(what, (2usize << 32) + 7, true).unwrap(),
            7
        );
    }

    #[test]
    fn test_section_len_matches_modulo_2_pow_32() {
        assert!(section_len_matches(0, 0));
        assert!(section_len_matches(1234, 1234));
        assert!(!section_len_matches(1234, 1235));
        assert!(!section_len_matches(1235, 1234));
        assert!(section_len_matches(u32::MAX, u32::MAX as usize));

        assert!(section_len_matches(5, (1usize << 32) + 5));
        assert!(section_len_matches(5, (2usize << 32) + 5));
        assert!(section_len_matches(0, 1usize << 32));
        assert!(!section_len_matches(6, (1usize << 32) + 5));
        assert!(!section_len_matches(u32::MAX, 1usize << 32));
    }

    #[test]
    fn test_may_exceed_u32_prefix() {
        assert!(!may_exceed_u32_prefix(0, 0));
        assert!(!may_exceed_u32_prefix(0, u32::MAX as usize));
        assert!(may_exceed_u32_prefix(0, 1usize << 32));
        assert!(!may_exceed_u32_prefix(10, (1usize << 32) + 9));
        assert!(may_exceed_u32_prefix(10, (1usize << 32) + 10));
        assert!(!may_exceed_u32_prefix(u32::MAX, (2usize << 32) - 2));
    }

    #[test]
    fn test_confirmed_manifest_position() {
        // A 10 byte section at 100 followed by a 30 byte manifest message.
        let mut first = vec![0u8; LEN_PREFIX_LEN + 10 + LEN_PREFIX_LEN];
        first[..4].copy_from_slice(&10u32.to_le_bytes());
        first[14..18].copy_from_slice(&30u32.to_le_bytes());
        let file_size = 100 + 14 + LEN_PREFIX_LEN + 30 + FOOTER_LEN;
        let room = file_size - 100 - 24;
        assert_eq!(
            confirmed_manifest_position(&first, 100, 10, file_size, room),
            Some(114)
        );
        // A reader size that does not end at the footer: not confirmed.
        assert_eq!(
            confirmed_manifest_position(&first, 100, 10, file_size - 1, room - 1),
            None
        );
        assert_eq!(confirmed_manifest_position(&first, 100, 10, 120, 0), None);
        // The u32 past the message was not read.
        assert_eq!(
            confirmed_manifest_position(&first[..17], 100, 10, file_size, room),
            None
        );
        // A prefix that may be wrapped is never confirmed from the first block.
        assert_eq!(
            confirmed_manifest_position(&first, 100, 10, file_size, 1usize << 33),
            None
        );
    }

    #[test]
    fn test_section_message_len() {
        // Back to back with the next section: the layout length.
        assert_eq!(section_message_len(100, 50, 154, 1000).unwrap(), 50);
        // Sections not laid out back to back are read by their prefix while the
        // length cannot be ambiguous.
        assert_eq!(section_message_len(100, 50, 400, 1000).unwrap(), 50);
        assert_eq!(section_message_len(100, 50, 50, 1000).unwrap(), 50);

        // A wrapped prefix with the layout agreeing modulo 2^32.
        let wrapped_end = 104 + (1usize << 32) + 50;
        assert_eq!(
            section_message_len(100, 50, wrapped_end, 1usize << 33).unwrap(),
            (1usize << 32) + 50
        );
        // The same with a stale reader size: the layout length still wins, so
        // the read fails as too small instead of decoding a cut message.
        assert_eq!(
            section_message_len(0, 0, LEN_PREFIX_LEN + (1usize << 32), 8192).unwrap(),
            1usize << 32
        );

        // Ambiguous and the layout disagrees.
        let err = section_message_len(100, 50, wrapped_end - 1, 1usize << 33)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("section length does not match"),
            "unexpected error: {err}"
        );
    }

    /// A manifest file tail: `prefix`, `body_len` zero bytes, then a footer.
    fn manifest_tail(prefix: u32, body_len: usize) -> Bytes {
        let mut tail = vec![0u8; LEN_PREFIX_LEN + body_len + FOOTER_LEN];
        tail[..LEN_PREFIX_LEN].copy_from_slice(&prefix.to_le_bytes());
        let footer_start = tail.len() - MAGIC.len();
        tail[footer_start..].copy_from_slice(MAGIC);
        Bytes::from(tail)
    }

    #[test]
    fn test_manifest_message_body_checks_prefix() {
        let body = manifest_message_body(manifest_tail(100, 100)).unwrap();
        assert_eq!(body.len(), 100);
        assert_eq!(manifest_message_body(manifest_tail(0, 0)).unwrap().len(), 0);

        for (prefix, body_len) in [(99, 100), (101, 100), (100 + (1 << 31), 100)] {
            let err = manifest_message_body(manifest_tail(prefix, body_len))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("manifest length does not match"),
                "unexpected error: {err}"
            );
        }

        let err = manifest_message_body(Bytes::from(vec![0u8; LEN_PREFIX_LEN + FOOTER_LEN - 1]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Invalid format"), "unexpected error: {err}");
    }

    /// The body of a message over `u32::MAX` bytes is taken from the layout and
    /// accepted when the prefix holds its low 32 bits. The zeroed buffer is
    /// allocated lazily and only its first and last pages are touched.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_manifest_message_body_accepts_wrapped_prefix() {
        let body_len = (1usize << 32) + 8;
        let body = manifest_message_body(manifest_tail(8, body_len)).unwrap();
        assert_eq!(body.len(), body_len);

        let err = manifest_message_body(manifest_tail(9, body_len))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("manifest length does not match"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_section_messages() {
        let store = ObjectStore::memory();
        let path = Path::from("sections");
        let transaction = pb::Transaction {
            read_version: 7,
            uuid: "transaction".to_string(),
            ..Default::default()
        };

        let mut manifest = manifest_with_fragments(10);
        let mut writer = store.create(&path).await.unwrap();
        let pos = write_manifest(
            writer.as_mut(),
            &mut manifest,
            Some(vec![]),
            Some(Transaction::from(transaction.clone())),
        )
        .await
        .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let read_back = read_manifest(&store, &path, None).await.unwrap();
        assert_eq!(read_back, manifest);
        let index_pos = read_back.index_section.unwrap();
        let transaction_pos = read_back.transaction_section.unwrap();

        let reader = store.open(&path).await.unwrap();
        let section: pb::IndexSection =
            read_section_message(reader.as_ref(), index_pos, Some(transaction_pos))
                .await
                .unwrap();
        assert_eq!(section, pb::IndexSection { indices: vec![] });
        let read_transaction: pb::Transaction =
            read_section_message(reader.as_ref(), transaction_pos, None)
                .await
                .unwrap();
        assert_eq!(read_transaction, transaction);

        // A reader whose size ends inside the message reports a too-small size.
        // So does one that ends past the message but before the footer.
        let manifest_pos = transaction_pos + LEN_PREFIX_LEN + transaction.encoded_len();
        for size in [transaction_pos + LEN_PREFIX_LEN + 1, manifest_pos + 8] {
            let short = store.open_with_size(&path, size).await.unwrap();
            let err =
                read_section_message::<pb::Transaction>(short.as_ref(), transaction_pos, None)
                    .await
                    .unwrap_err();
            assert!(
                err.to_string().contains("file size is too small"),
                "unexpected error: {err}"
            );
        }
    }

    /// Append `message` followed by an unknown length-delimited field of
    /// `padding` bytes, which decoders skip, and return the message length.
    #[cfg(unix)]
    fn write_padded_message(
        file: &mut std::fs::File,
        message: &impl Message,
        padding: usize,
    ) -> usize {
        use std::io::{Seek, SeekFrom, Write};

        let mut head = message.encode_to_vec();
        prost::encoding::encode_key(
            (1 << 29) - 1,
            prost::encoding::WireType::LengthDelimited,
            &mut head,
        );
        prost::encoding::encode_varint(padding as u64, &mut head);
        let len = head.len() + padding;
        file.write_all(&(len as u32).to_le_bytes()).unwrap();
        file.write_all(&head).unwrap();
        // Leave a hole: the file system reads it back as zeros.
        file.seek(SeekFrom::Current(padding as i64)).unwrap();
        len
    }

    /// Reads a manifest file whose inline transaction and manifest message are
    /// both larger than `u32::MAX` bytes, with prefixes holding the low 32 bits
    /// of their lengths. The file is sparse, but each read holds up to two
    /// copies of a 4 GiB message in memory, so the test is not run by default.
    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "reads two messages over 4 GiB; needs about 10 GiB of memory"]
    async fn test_read_manifest_file_with_messages_over_4_gib() {
        use std::io::Write;

        let padding = (1usize << 32) + 1000;
        let dir = lance_core::utils::tempfile::TempDir::try_new().unwrap();
        let file_path = dir.std_path().join("large.manifest");
        let mut file = std::fs::File::create(&file_path).unwrap();

        let transaction = pb::Transaction {
            read_version: 7,
            uuid: "large transaction".to_string(),
            ..Default::default()
        };
        let transaction_len = write_padded_message(&mut file, &transaction, padding);
        assert!(transaction_len > u32::MAX as usize);

        let mut manifest = manifest_with_fragments(10);
        manifest.transaction_section = Some(0);
        let manifest_pos = LEN_PREFIX_LEN + transaction_len;
        let manifest_len = write_padded_message(&mut file, &pb::Manifest::from(&manifest), padding);
        assert!(manifest_len > u32::MAX as usize);

        file.write_all(&(manifest_pos as i64).to_le_bytes())
            .unwrap();
        file.write_all(&MAJOR_VERSION.to_le_bytes()).unwrap();
        file.write_all(&MINOR_VERSION.to_le_bytes()).unwrap();
        file.write_all(MAGIC).unwrap();
        drop(file);

        let (store, path) = ObjectStore::from_uri(file_path.to_str().unwrap())
            .await
            .unwrap();
        let read_back = read_manifest(&store, &path, None).await.unwrap();
        assert_eq!(read_back, manifest);

        let reader = store.open(&path).await.unwrap();
        let read_transaction: pb::Transaction = read_section_message(reader.as_ref(), 0, None)
            .await
            .unwrap();
        assert_eq!(read_transaction, transaction);

        // A next section that disagrees with the prefix is rejected.
        let err =
            read_section_message::<pb::Transaction>(reader.as_ref(), 0, Some(manifest_pos - 1))
                .await
                .unwrap_err();
        assert!(
            err.to_string().contains("section length does not match"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_write_manifest_file_to_chunks_matches_object_writer() {
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
            let chunks = write_manifest_file_to_chunks(&mut buffered, indices, None)
                .await
                .unwrap();
            assert_eq!(chunks.concat(), expected);
            assert_eq!(buffered, written);

            // The cached fragments are slices of the message chunk alone, so
            // they do not keep the index and transaction sections alive.
            let message = chunks[1].as_ptr_range();
            let cached = buffered.encoded_fragments().unwrap();
            for bytes in cached.iter().flatten() {
                let slice = bytes.as_ptr_range();
                assert!(message.start <= slice.start && slice.end <= message.end);
            }

            let read_back = read_manifest(&store, &path, None).await.unwrap();
            assert_eq!(read_back, buffered);
        }
    }
}
