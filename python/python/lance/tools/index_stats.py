# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Index inspection utilities for Lance index files."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from lance.file import LanceFileReader, LanceFileSession

TRANS_INDEX_THRESHOLD = 32
EMPTY_ADDRESS = 0

COMMON_INPUTS_INV = [
    116,
    101,
    47,
    111,
    97,
    115,
    114,
    105,
    112,
    99,
    110,
    119,
    46,
    104,
    108,
    109,
    45,
    100,
    117,
    48,
    49,
    50,
    103,
    61,
    58,
    98,
    102,
    51,
    121,
    53,
    38,
    95,
    52,
    118,
    57,
    54,
    55,
    56,
    107,
    37,
    63,
    120,
    67,
    68,
    65,
    83,
    70,
    73,
    66,
    69,
    106,
    80,
    84,
    122,
    82,
    78,
    77,
    43,
    76,
    79,
    113,
    72,
    71,
    87,
    85,
    86,
    44,
    89,
    75,
    74,
    90,
    88,
    81,
    59,
    41,
    40,
    126,
    91,
    93,
    36,
    33,
    39,
    42,
    64,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    35,
    60,
    62,
    92,
    94,
    96,
    123,
    124,
    125,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
]


def _normalize_metadata(metadata: Dict[object, object]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(key, (bytes, bytearray)):
            key_str = key.decode("utf-8")
        else:
            key_str = str(key)
        if isinstance(value, (bytes, bytearray)):
            value_str = value.decode("utf-8")
        else:
            value_str = str(value)
        normalized[key_str] = value_str
    return normalized


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


def _percentile(sorted_values: Sequence[int], percentile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(math.floor(k))
    upper = int(math.ceil(k))
    if lower == upper:
        return float(sorted_values[lower])
    weight = k - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _summarize(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "sum": 0,
        }
    sorted_vals = sorted(values)
    total = sum(values)
    count = len(values)
    return {
        "count": count,
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(total) / count,
        "p50": _percentile(sorted_vals, 50),
        "p90": _percentile(sorted_vals, 90),
        "p99": _percentile(sorted_vals, 99),
        "sum": float(total),
    }


def _read_varint(buf: bytes, idx: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if idx >= len(buf):
            raise ValueError("Unexpected end of buffer while reading varint")
        byte = buf[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, idx
        shift += 7
        if shift >= 64:
            raise ValueError("Varint is too long")


def parse_ivf_proto(buffer: bytes) -> Dict[str, object]:
    offsets: List[int] = []
    lengths: List[int] = []
    loss: Optional[float] = None
    idx = 0
    while idx < len(buffer):
        key, idx = _read_varint(buffer, idx)
        field_number = key >> 3
        wire_type = key & 0x7
        if field_number in (2, 3):
            target = offsets if field_number == 2 else lengths
            if wire_type == 0:
                value, idx = _read_varint(buffer, idx)
                target.append(int(value))
            elif wire_type == 2:
                length, idx = _read_varint(buffer, idx)
                end = idx + length
                while idx < end:
                    value, idx = _read_varint(buffer, idx)
                    target.append(int(value))
            else:
                raise ValueError(
                    f"Unsupported wire type {wire_type} for field {field_number}"
                )
        elif field_number == 5 and wire_type == 1:
            if idx + 8 > len(buffer):
                raise ValueError("Unexpected end of buffer while reading double")
            loss = float(struct.unpack("<d", buffer[idx : idx + 8])[0])
            idx += 8
        else:
            if wire_type == 0:
                _, idx = _read_varint(buffer, idx)
            elif wire_type == 1:
                idx += 8
            elif wire_type == 2:
                length, idx = _read_varint(buffer, idx)
                idx += length
            elif wire_type == 5:
                idx += 4
            else:
                raise ValueError(f"Unsupported wire type {wire_type}")
    return {"offsets": offsets, "lengths": lengths, "loss": loss}


@dataclass(frozen=True)
class _FstMeta:
    version: int
    root_addr: int
    length: int


class _PackSizes:
    def __init__(self, value: int) -> None:
        self._value = value & 0xFF

    def transition_pack_size(self) -> int:
        return (self._value >> 4) & 0x0F

    def output_pack_size(self) -> int:
        return self._value & 0x0F


class _FstNode:
    def __init__(self, data: bytes, version: int, addr: int) -> None:
        self.data = data[: addr + 1]
        self.version = version
        self.start = addr
        self.state = "empty_final"
        self.end = addr
        self.is_final = True
        self.ntrans = 0
        self.sizes = _PackSizes(0)
        self.final_output = 0

        if addr == EMPTY_ADDRESS:
            return

        state_byte = self.data[addr]
        tag = (state_byte & 0b1100_0000) >> 6
        if tag == 0b11:
            self.state = "one_trans_next"
            self.is_final = False
            self.ntrans = 1
            input_len = self._input_len(state_byte)
            self.end = len(self.data) - 1 - input_len
            return

        if tag == 0b10:
            self.state = "one_trans"
            self.is_final = False
            self.ntrans = 1
            input_len = self._input_len(state_byte)
            pack_index = len(self.data) - 1 - input_len - 1
            self.sizes = _PackSizes(self.data[pack_index])
            osize = self.sizes.output_pack_size()
            tsize = self.sizes.transition_pack_size()
            self.end = len(self.data) - 1 - input_len - 1 - tsize - osize
            return

        self.state = "any_trans"
        self.is_final = (state_byte & 0b0100_0000) != 0
        self.ntrans, ntrans_len = self._read_ntrans(state_byte)
        pack_index = len(self.data) - 1 - ntrans_len - 1
        self.sizes = _PackSizes(self.data[pack_index])
        osize = self.sizes.output_pack_size()
        tsize = self.sizes.transition_pack_size()
        index_size = self._trans_index_size(self.ntrans)
        total_trans_size = self.ntrans + (self.ntrans * tsize) + index_size
        final_osize = osize if self.is_final else 0
        self.end = (
            len(self.data)
            - 1
            - ntrans_len
            - 1
            - total_trans_size
            - (self.ntrans * osize)
            - final_osize
        )
        if osize != 0 and self.is_final:
            at = (
                len(self.data)
                - 1
                - ntrans_len
                - 1
                - total_trans_size
                - (self.ntrans * osize)
                - osize
            )
            self.final_output = _unpack_uint(self.data[at:], osize)

    @staticmethod
    def _common_input(state_byte: int) -> Optional[int]:
        idx = state_byte & 0b0011_1111
        if idx == 0:
            return None
        return COMMON_INPUTS_INV[idx - 1]

    @classmethod
    def _input_len(cls, state_byte: int) -> int:
        return 0 if cls._common_input(state_byte) is not None else 1

    def _read_ntrans(self, state_byte: int) -> Tuple[int, int]:
        n = state_byte & 0b0011_1111
        if n != 0:
            return int(n), 0
        n_raw = self.data[-2]
        if n_raw == 1:
            return 256, 1
        return int(n_raw), 1

    def _trans_index_size(self, ntrans: int) -> int:
        if self.version >= 2 and ntrans > TRANS_INDEX_THRESHOLD:
            return 256
        return 0

    def _one_trans_input(self) -> int:
        common = self._common_input(self.data[self.start])
        if common is not None:
            return common
        return self.data[self.start - 1]

    def _one_trans_addr(self) -> int:
        if self.state == "one_trans_next":
            return self.end - 1
        tsize = self.sizes.transition_pack_size()
        if tsize == 0:
            return EMPTY_ADDRESS
        input_len = self._input_len(self.data[self.start])
        pos = self.start - input_len - 1 - tsize
        return _unpack_delta(self.data[pos:], tsize, self.end)

    def _one_trans_output(self) -> int:
        if self.state != "one_trans":
            return 0
        osize = self.sizes.output_pack_size()
        if osize == 0:
            return 0
        input_len = self._input_len(self.data[self.start])
        tsize = self.sizes.transition_pack_size()
        pos = self.start - input_len - 1 - tsize - osize
        return _unpack_uint(self.data[pos:], osize)

    def transitions(self) -> List[Tuple[int, int, int]]:
        if self.state == "empty_final":
            return []
        if self.state in ("one_trans_next", "one_trans"):
            return [
                (
                    self._one_trans_input(),
                    self._one_trans_addr(),
                    self._one_trans_output(),
                )
            ]
        transitions: List[Tuple[int, int, int]] = []
        ntrans_len = 0 if (self.data[self.start] & 0b0011_1111) != 0 else 1
        tsize = self.sizes.transition_pack_size()
        osize = self.sizes.output_pack_size()
        index_size = self._trans_index_size(self.ntrans)
        total_trans_size = self.ntrans + (self.ntrans * tsize) + index_size
        for i in range(self.ntrans):
            inp_at = self.start - ntrans_len - 1 - index_size - i - 1
            inp = self.data[inp_at]
            if osize == 0:
                out = 0
            else:
                out_at = (
                    self.start
                    - ntrans_len
                    - 1
                    - total_trans_size
                    - (i * osize)
                    - osize
                )
                out = _unpack_uint(self.data[out_at:], osize)
            if tsize == 0:
                addr = EMPTY_ADDRESS
            else:
                addr_at = (
                    self.start
                    - ntrans_len
                    - 1
                    - index_size
                    - self.ntrans
                    - (i * tsize)
                    - tsize
                )
                addr = _unpack_delta(self.data[addr_at:], tsize, self.end)
            transitions.append((inp, addr, out))
        return transitions


def _read_fst_meta(data: bytes) -> _FstMeta:
    if len(data) < 36:
        raise ValueError("FST data too short")
    version = int.from_bytes(data[0:8], "little")
    if version == 0:
        raise ValueError("Unsupported FST version 0")
    if version <= 2:
        end = len(data)
    else:
        end = len(data) - 4
    root_addr = int.from_bytes(data[end - 8 : end], "little")
    length = int.from_bytes(data[end - 16 : end - 8], "little")
    return _FstMeta(version=version, root_addr=root_addr, length=length)


def _unpack_uint(slice_bytes: bytes, nbytes: int) -> int:
    if nbytes <= 0 or nbytes > 8:
        raise ValueError("Invalid packed integer size")
    value = 0
    for i in range(nbytes):
        value |= slice_bytes[i] << (8 * i)
    return value


def _unpack_delta(slice_bytes: bytes, nbytes: int, node_addr: int) -> int:
    delta = _unpack_uint(slice_bytes, nbytes)
    if delta == EMPTY_ADDRESS:
        return EMPTY_ADDRESS
    return node_addr - delta


def iter_fst_map_items(data: bytes) -> Iterator[Tuple[bytes, int]]:
    meta = _read_fst_meta(data)
    if meta.length == 0:
        return
    stack: List[Tuple[int, bytes, int]] = [(meta.root_addr, b"", 0)]
    while stack:
        addr, prefix, acc = stack.pop()
        node = _FstNode(data, meta.version, addr)
        if node.is_final:
            yield prefix, acc + node.final_output
        transitions = node.transitions()
        for inp, next_addr, out in reversed(transitions):
            stack.append((next_addr, prefix + bytes([inp]), acc + out))


def _load_lance_metadata(reader: LanceFileReader) -> Dict[str, str]:
    metadata = reader.metadata().schema.metadata
    return _normalize_metadata(metadata)


def _load_lengths(reader: LanceFileReader) -> List[int]:
    num_rows = reader.metadata().num_rows
    table = reader.read_all(batch_size=num_rows).to_table()
    if "_length" not in table.column_names:
        raise ValueError("Missing _length column in invert file")
    return [int(x) for x in table.column("_length").to_pylist()]

def _detect_token_format(reader: LanceFileReader, fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    schema = reader.metadata().schema
    if any(field.name == "_token" for field in schema):
        return "arrow"
    if any(field.name == "_token_fst_bytes" for field in schema):
        return "fst"
    raise ValueError("Unable to detect token format for tokens file")


@dataclass
class _TokenData:
    format: str
    count: int
    id_to_token: Optional[Dict[int, str]] = None
    tokens: Optional[set] = None


def _load_tokens(
    reader: LanceFileReader,
    token_format: str,
    need_id_map: bool,
    need_set: bool,
) -> _TokenData:
    num_rows = reader.metadata().num_rows
    if token_format == "arrow":
        if not need_id_map and not need_set:
            return _TokenData(format="arrow", count=num_rows)
        table = reader.read_all(batch_size=num_rows).to_table()
        tokens_col = table.column("_token")
        token_ids_col = table.column("_token_id")
        tokens_list = tokens_col.to_pylist()
        token_ids_list = token_ids_col.to_pylist()
        id_to_token: Dict[int, str] = {}
        token_set: Optional[set] = set() if need_set else None
        for token, token_id_val in zip(tokens_list, token_ids_list):
            token_id = int(token_id_val)
            id_to_token[token_id] = token
            if token_set is not None:
                token_set.add(token)
        return _TokenData(
            format="arrow",
            count=len(id_to_token),
            id_to_token=id_to_token if need_id_map else None,
            tokens=token_set,
        )

    if token_format != "fst":
        raise ValueError(f"Unsupported token format: {token_format}")

    if not need_id_map and not need_set:
        table = reader.read_all(batch_size=num_rows).to_table()
        next_id = int(table.column("_token_next_id").chunk(0)[0].as_py())
        return _TokenData(format="fst", count=next_id)

    table = reader.read_all(batch_size=num_rows).to_table()
    if num_rows == 0:
        return _TokenData(format="fst", count=0)
    fst_bytes = table.column("_token_fst_bytes").chunk(0)[0].as_py()
    id_to_token = {} if need_id_map else None
    token_set: Optional[set] = set() if need_set else None
    count = 0
    for key_bytes, token_id in iter_fst_map_items(fst_bytes):
        token = key_bytes.decode("utf-8", errors="replace")
        count += 1
        if id_to_token is not None:
            id_to_token[int(token_id)] = token
        if token_set is not None:
            token_set.add(token)
    return _TokenData(
        format="fst",
        count=count,
        id_to_token=id_to_token,
        tokens=token_set,
    )


@dataclass
class PartitionStats:
    partition_id: int
    docs_count: int
    term_count: int
    lengths_summary: Dict[str, Optional[float]]
    short_posting_count: int
    short_posting_total_length: int
    uncompressed_length: int
    bytes_total: int
    bytes_breakdown: Dict[str, int]
    tokens: Optional[set] = None
    term_lengths: Optional[Dict[str, int]] = None


@dataclass
class VectorStats:
    partitions: int
    lengths: List[int]
    lengths_summary: Dict[str, Optional[float]]


class ProgressReporter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def info(self, message: str) -> None:
        if self.enabled:
            print(message, file=sys.stderr, flush=True)

    def partition_start(self, index: int, total: int, partition_id: int) -> None:
        self.info(f"[{index}/{total}] partition {partition_id}: loading")

    def partition_done(
        self, index: int, total: int, partition_id: int, docs: int, terms: int
    ) -> None:
        self.info(
            f"[{index}/{total}] partition {partition_id}: done "
            f"(docs={docs}, terms={terms})"
        )


def _is_object_uri(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.scheme != "file")


def _file_size(reader: LanceFileReader) -> int:
    stats = reader.file_statistics()
    return int(sum(column.size_bytes for column in stats.columns))


def _make_session(base_path: Union[str, Path]) -> LanceFileSession:
    return LanceFileSession(str(base_path))


def _partition_file_paths(partition_id: Optional[int], partitioned: bool) -> Dict[str, str]:
    if partitioned:
        if partition_id is None:
            raise ValueError("Partition id is required for partitioned index")
        prefix = f"part_{partition_id}_"
    else:
        prefix = ""
    return {
        "tokens": f"{prefix}tokens.lance",
        "invert": f"{prefix}invert.lance",
        "docs": f"{prefix}docs.lance",
    }


def _discover_partitions(session: LanceFileSession) -> Tuple[List[int], Optional[str], bool]:
    token_format = None
    partitions: Optional[List[int]] = None

    if session.contains("metadata.lance"):
        metadata_reader = session.open_reader("metadata.lance")
        metadata = _load_lance_metadata(metadata_reader)
        if "token_set_format" in metadata:
            token_format = metadata["token_set_format"].lower()
        if "partitions" in metadata:
            partitions = json.loads(metadata["partitions"])

    files = session.list()
    part_tokens = [
        path
        for path in files
        if path.endswith("_tokens.lance") and Path(path).name.startswith("part_")
    ]
    partitioned = bool(part_tokens)

    if partitions is None:
        if part_tokens:
            ids: List[int] = []
            for path in part_tokens:
                name = Path(path).name
                match = re.match(r"part_(\\d+)_tokens\\.lance$", name)
                if not match:
                    raise ValueError(f"Unexpected partition file name: {name}")
                ids.append(int(match.group(1)))
            partitions = ids
        elif "tokens.lance" in files:
            partitions = [0]
        else:
            raise FileNotFoundError("No tokens.lance or partitioned token files found")

    return partitions, token_format, partitioned


def analyze_vector(index_path: Union[str, Path], progress: bool = False) -> VectorStats:
    index_path_str = str(index_path)
    if not index_path_str.endswith("index.idx"):
        if _is_object_uri(index_path_str):
            index_path_str = index_path_str.rstrip("/") + "/index.idx"
        else:
            local_path = Path(index_path_str)
            if local_path.is_dir():
                index_path_str = str(local_path / "index.idx")
    reporter = ProgressReporter(progress)
    reporter.info(f"Reading vector index metadata: {index_path_str}")
    reader = LanceFileReader(index_path_str)
    metadata = _normalize_metadata(reader.metadata().schema.metadata)
    if "lance:ivf" not in metadata:
        raise ValueError("Missing lance:ivf metadata in index file")
    ivf_buffer_index = int(metadata["lance:ivf"])
    ivf_bytes = reader.read_global_buffer(ivf_buffer_index)
    parsed = parse_ivf_proto(ivf_bytes)
    lengths = [int(v) for v in parsed["lengths"]]
    reporter.info(f"Vector partitions: {len(lengths)}")
    return VectorStats(
        partitions=len(lengths),
        lengths=lengths,
        lengths_summary=_summarize(lengths),
    )


def analyze_fts(
    index_dir: Union[str, Path],
    compare: Optional[Tuple[int, int]] = None,
    include_term_lengths: bool = False,
    compute_union: bool = True,
    progress: bool = False,
) -> Dict[str, object]:
    session = _make_session(index_dir)
    partitions, token_format, partitioned = _discover_partitions(session)
    reporter = ProgressReporter(progress)
    reporter.info(f"Scanning FTS index with {len(partitions)} partitions")
    stats: List[PartitionStats] = []
    sizes: List[int] = []
    bytes_totals: List[int] = []
    all_terms: Optional[set] = set() if compute_union else None
    global_short_count = 0
    global_short_total = 0
    global_uncompressed = 0

    compare_ids = set(compare) if compare else set()

    for idx, part_id in enumerate(partitions, start=1):
        reporter.partition_start(idx, len(partitions), part_id)
        file_paths = _partition_file_paths(part_id if partitioned else None, partitioned)
        docs_reader = session.open_reader(file_paths["docs"])
        docs_count = int(docs_reader.metadata().num_rows)
        invert_reader = session.open_reader(file_paths["invert"])
        lengths = _load_lengths(invert_reader)
        sizes.append(docs_count)
        need_id_map = include_term_lengths
        need_set = compute_union or (compare is not None and part_id in compare_ids)
        tokens_reader = session.open_reader(file_paths["tokens"])
        detected_format = _detect_token_format(tokens_reader, token_format)
        tokens = _load_tokens(tokens_reader, detected_format, need_id_map, need_set)
        if all_terms is not None and tokens.tokens is not None:
            all_terms.update(tokens.tokens)

        short_count = 0
        short_total = 0
        uncompressed = 0
        for length in lengths:
            if length < 128:
                short_count += 1
                short_total += length
            uncompressed += length % 128
        global_short_count += short_count
        global_short_total += short_total
        global_uncompressed += uncompressed

        bytes_breakdown = {
            "tokens": _file_size(tokens_reader),
            "invert": _file_size(invert_reader),
            "docs": _file_size(docs_reader),
        }
        bytes_total = sum(bytes_breakdown.values())
        bytes_totals.append(bytes_total)

        term_lengths: Optional[Dict[str, int]] = None
        if include_term_lengths:
            if tokens.id_to_token is None:
                raise ValueError("Token ID mapping requested but unavailable")
            term_lengths = {}
            for token_id, token in tokens.id_to_token.items():
                if token_id < len(lengths):
                    term_lengths[token] = int(lengths[token_id])

        stats.append(
            PartitionStats(
                partition_id=part_id,
                docs_count=docs_count,
                term_count=tokens.count,
                lengths_summary=_summarize(lengths),
                short_posting_count=short_count,
                short_posting_total_length=short_total,
                uncompressed_length=uncompressed,
                bytes_total=bytes_total,
                bytes_breakdown=bytes_breakdown,
                tokens=tokens.tokens,
                term_lengths=term_lengths,
            )
        )
        reporter.partition_done(idx, len(partitions), part_id, docs_count, tokens.count)

    compare_stats = None
    if compare:
        left_id, right_id = compare
        left = next((p for p in stats if p.partition_id == left_id), None)
        right = next((p for p in stats if p.partition_id == right_id), None)
        if left is None or right is None:
            raise ValueError("Compare partition IDs not found")
        if left.tokens is None or right.tokens is None:
            raise ValueError("Tokens not loaded for comparison")
        common = left.tokens & right.tokens
        only_left = left.tokens - right.tokens
        only_right = right.tokens - left.tokens
        compare_stats = {
            "left": left_id,
            "right": right_id,
            "common_terms": len(common),
            "only_left": len(only_left),
            "only_right": len(only_right),
            "union": len(common) + len(only_left) + len(only_right),
        }

    return {
        "partitions": [
            {
                "partition_id": p.partition_id,
                "docs_count": p.docs_count,
                "term_count": p.term_count,
                "posting_list_length_summary": p.lengths_summary,
                "short_posting_lists": {
                    "count": p.short_posting_count,
                    "total_length": p.short_posting_total_length,
                },
                "uncompressed_length": p.uncompressed_length,
                "bytes_total": p.bytes_total,
                "bytes_breakdown": p.bytes_breakdown,
                "term_lengths": p.term_lengths,
            }
            for p in stats
        ],
        "partition_count": len(partitions),
        "docs_count_summary": _summarize(sizes),
        "bytes_summary": _summarize(bytes_totals),
        "short_posting_lists": {
            "count": global_short_count,
            "total_length": global_short_total,
        },
        "uncompressed_length": global_uncompressed,
        "unique_terms": len(all_terms) if all_terms is not None else None,
        "compare": compare_stats,
    }


def _vector_text_report(stats: VectorStats, verbose: bool) -> str:
    lines = [
        f"Partitions: {stats.partitions}",
        f"Partition sizes (rows): {stats.lengths_summary}",
    ]
    if verbose:
        lines.append("Partition lengths (rows):")
        lines.append(
            ", ".join(str(value) for value in stats.lengths)
            if stats.lengths
            else "(empty)"
        )
    return "\n".join(lines)


def _fts_text_report(stats: Dict[str, object], verbose: bool) -> str:
    lines = [
        f"Partitions: {stats['partition_count']}",
        f"Docs count summary: {stats['docs_count_summary']}",
        f"Bytes summary: {stats['bytes_summary']}",
        f"Short posting lists (global): {stats['short_posting_lists']}",
        f"Uncompressed length (global): {stats['uncompressed_length']}",
        f"Unique terms (union): {stats['unique_terms']}",
    ]
    lines.append("Short posting lists per partition:")
    for part in stats["partitions"]:
        short_stats = part["short_posting_lists"]
        lines.append(
            "  "
            f"partition {part['partition_id']}: "
            f"short_count={short_stats['count']}, "
            f"short_total_length={short_stats['total_length']}, "
            f"uncompressed_length={part['uncompressed_length']}"
        )
    if verbose:
        lines.append("Partition details:")
        for part in stats["partitions"]:
            breakdown = part["bytes_breakdown"]
            lines.append(
                "  "
                f"partition {part['partition_id']}: docs={part['docs_count']}, "
                f"terms={part['term_count']}, "
                f"bytes={_format_bytes(part['bytes_total'])} "
                f"(tokens={_format_bytes(breakdown['tokens'])}, "
                f"invert={_format_bytes(breakdown['invert'])}, "
                f"docs={_format_bytes(breakdown['docs'])})"
            )
            lines.append(
                f"    posting list lengths: {part['posting_list_length_summary']}"
            )
    if stats.get("compare"):
        lines.append(f"Compare: {stats['compare']}")
    return "\n".join(lines)


def _parse_args_from_list(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Lance index files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vector_parser = subparsers.add_parser("vector", help="Analyze vector index")
    vector_parser.add_argument(
        "path",
        type=str,
        help="Path to index.idx or its directory (supports object storage URIs)",
    )
    vector_parser.add_argument("--format", choices=["text", "json"], default="text")
    vector_parser.add_argument("--verbose", action="store_true")
    vector_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress updates",
    )

    fts_parser = subparsers.add_parser("fts", help="Analyze FTS index")
    fts_parser.add_argument(
        "path",
        type=str,
        help="Path to FTS index directory (supports object storage URIs)",
    )
    fts_parser.add_argument("--format", choices=["text", "json"], default="text")
    fts_parser.add_argument("--verbose", action="store_true")
    fts_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress updates",
    )
    fts_parser.add_argument(
        "--compare",
        nargs=2,
        type=int,
        metavar=("LEFT", "RIGHT"),
        help="Compare term overlap between two partitions",
    )
    fts_parser.add_argument(
        "--terms-out",
        type=Path,
        help="Write term -> posting length mappings to a JSON file",
    )
    fts_parser.add_argument(
        "--skip-union",
        action="store_true",
        help="Skip computing union of terms across partitions",
    )
    return parser.parse_args(argv)


def _parse_args() -> argparse.Namespace:
    return _parse_args_from_list()


def main() -> None:
    args = _parse_args()
    if args.command == "vector":
        stats = analyze_vector(args.path, progress=args.progress)
        if args.format == "json":
            print(json.dumps(stats.__dict__, indent=2))
        else:
            print(_vector_text_report(stats, args.verbose))
        return

    compare = tuple(args.compare) if args.compare else None
    include_term_lengths = args.terms_out is not None
    stats = analyze_fts(
        args.path,
        compare=compare,
        include_term_lengths=include_term_lengths,
        compute_union=not args.skip_union,
        progress=args.progress,
    )

    if include_term_lengths and args.terms_out is not None:
        term_rows = []
        for part in stats["partitions"]:
            term_lengths = part.get("term_lengths")
            if not term_lengths:
                continue
            for term, length in term_lengths.items():
                term_rows.append(
                    {
                        "partition_id": part["partition_id"],
                        "term": term,
                        "posting_length": length,
                    }
                )
        args.terms_out.write_text(json.dumps(term_rows, indent=2))

    if args.format == "json":
        print(json.dumps(stats, indent=2))
    else:
        print(_fts_text_report(stats, args.verbose))


if __name__ == "__main__":
    main()
