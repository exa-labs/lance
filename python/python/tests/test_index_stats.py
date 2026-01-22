# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
from pathlib import Path

import pyarrow as pa

from lance.file import LanceFileWriter
from lance.tools.index_stats import (
    _parse_args_from_list,
    analyze_fts,
    iter_fst_map_items,
    parse_ivf_proto,
)


def _encode_varint(value: int) -> bytes:
    buf = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            buf.append(to_write | 0x80)
        else:
            buf.append(to_write)
            break
    return bytes(buf)


def _write_lance(path: Path, table: pa.Table) -> None:
    with LanceFileWriter(str(path), table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)


def test_parse_ivf_proto_packed() -> None:
    offsets = [0, 3, 7]
    lengths = [3, 4, 5]
    offsets_buf = b"".join(_encode_varint(v) for v in offsets)
    lengths_buf = b"".join(_encode_varint(v) for v in lengths)
    buffer = (
        _encode_varint((2 << 3) | 2)
        + _encode_varint(len(offsets_buf))
        + offsets_buf
        + _encode_varint((3 << 3) | 2)
        + _encode_varint(len(lengths_buf))
        + lengths_buf
    )
    parsed = parse_ivf_proto(buffer)
    assert parsed["offsets"] == offsets
    assert parsed["lengths"] == lengths


def test_iter_fst_map_items() -> None:
    fst_bytes = bytes(
        [
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            16,
            130,
            207,
            201,
            201,
            0,
            16,
            133,
            203,
            197,
            203,
            197,
            0,
            16,
            129,
            196,
            199,
            199,
            197,
            2,
            1,
            0,
            1,
            8,
            15,
            99,
            98,
            97,
            17,
            3,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            46,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            237,
            117,
            252,
            236,
        ]
    )
    items = {
        key.decode("utf-8"): value for key, value in iter_fst_map_items(fst_bytes)
    }
    assert items == {"apple": 0, "banana": 1, "carrot": 2}


def test_analyze_fts_arrow(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.lance"
    metadata_writer = LanceFileWriter(str(metadata_path), pa.schema([]))
    metadata_writer.add_schema_metadata("partitions", json.dumps([0, 1]))
    metadata_writer.add_schema_metadata("token_set_format", "arrow")
    metadata_writer.close()

    tokens0 = pa.table(
        {
            "_token": ["apple", "banana"],
            "_token_id": pa.array([0, 1], type=pa.uint32()),
        }
    )
    invert0 = pa.table({"_length": pa.array([3, 5], type=pa.uint32())})
    docs0 = pa.table({"_row_id": pa.array([1, 2, 3], type=pa.uint64())})
    _write_lance(tmp_path / "part_0_tokens.lance", tokens0)
    _write_lance(tmp_path / "part_0_invert.lance", invert0)
    _write_lance(tmp_path / "part_0_docs.lance", docs0)

    tokens1 = pa.table(
        {
            "_token": ["banana", "date"],
            "_token_id": pa.array([0, 1], type=pa.uint32()),
        }
    )
    invert1 = pa.table({"_length": pa.array([2, 7], type=pa.uint32())})
    docs1 = pa.table({"_row_id": pa.array([4, 5, 6, 7], type=pa.uint64())})
    _write_lance(tmp_path / "part_1_tokens.lance", tokens1)
    _write_lance(tmp_path / "part_1_invert.lance", invert1)
    _write_lance(tmp_path / "part_1_docs.lance", docs1)

    stats = analyze_fts(
        str(tmp_path),
        compare=(0, 1),
        include_term_lengths=True,
        compute_union=True,
    )

    assert stats["partition_count"] == 2
    assert stats["unique_terms"] == 3
    assert stats["bytes_summary"]["count"] == 2
    assert stats["short_posting_lists"]["count"] == 4
    assert stats["short_posting_lists"]["total_length"] == 17
    assert stats["uncompressed_length"] == 17

    compare = stats["compare"]
    assert compare["common_terms"] == 1
    assert compare["only_left"] == 1
    assert compare["only_right"] == 1

    part0 = next(part for part in stats["partitions"] if part["partition_id"] == 0)
    part1 = next(part for part in stats["partitions"] if part["partition_id"] == 1)

    assert part0["term_count"] == 2
    assert part1["term_count"] == 2
    assert part0["term_lengths"]["apple"] == 3
    assert part0["term_lengths"]["banana"] == 5
    assert part1["term_lengths"]["banana"] == 2
    assert part1["term_lengths"]["date"] == 7
    assert part0["short_posting_lists"]["count"] == 2
    assert part0["short_posting_lists"]["total_length"] == 8
    assert part1["short_posting_lists"]["count"] == 2
    assert part1["short_posting_lists"]["total_length"] == 9
    assert part0["uncompressed_length"] == 8
    assert part1["uncompressed_length"] == 9


def test_parse_args_preserves_object_uri() -> None:
    uri = "gs://bucket/path/index_dir"
    args = _parse_args_from_list(["fts", uri])
    assert args.path == uri


def test_parse_args_progress_flag() -> None:
    uri = "gs://bucket/path/index_dir"
    args = _parse_args_from_list(["fts", uri])
    assert args.progress is True
    args = _parse_args_from_list(["fts", uri, "--no-progress"])
    assert args.progress is False
