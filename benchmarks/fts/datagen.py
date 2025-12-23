#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import math
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

SYSTEM_WORD_LISTS = (
    "/usr/share/dict/words",
    "/usr/dict/words",
)

DEFAULT_WORDS_TEXT = (
    "the of and to in a is that be it by are for was as he with on his at from "
    "this have or one had not but what all were when we there can an your which "
    "their said if do will each about how up out them then she many some so these "
    "would other into has more her two like him see time could no make than first "
    "been its who now people my made over did down only way find use may water "
    "long little very after words called just where most know get through back "
    "much go good new write our used me man too any day same right look think also "
    "around another came come work three word must because does part even place well "
    "such here take why things help put years different away again off went old "
    "number great tell men say small every found still between name should home big "
    "give air line set own under read last never us left end along while might "
    "next sound below saw something thought both few those always show large often "
    "together asked house don world going want school important until form food "
    "keep children feet land side without boy once animals life enough took "
    "sometimes four head above kind began almost live page got earth need far hand "
    "high year mother light country father let night being really small but or "
    "enough always more under around even every small large great find first last"
)

DEFAULT_WORDS = [w for w in DEFAULT_WORDS_TEXT.split() if w]


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def load_word_list(path: Path, min_len: int, max_len: int) -> List[str]:
    words: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = line.strip().lower()
            if not word or not word.isalpha():
                continue
            if min_len and len(word) < min_len:
                continue
            if max_len and len(word) > max_len:
                continue
            words.append(word)
    if not words:
        raise ValueError(f"No usable words found in {path}")
    return words


def resolve_words(
    word_list: Optional[str],
    min_len: int,
    max_len: int,
    vocab_size: Optional[int],
    rng: random.Random,
) -> Sequence[str]:
    words: List[str] = []
    if word_list:
        words = load_word_list(Path(word_list), min_len, max_len)
    else:
        for candidate in SYSTEM_WORD_LISTS:
            path = Path(candidate)
            if path.exists():
                words = load_word_list(path, min_len, max_len)
                break
    if not words:
        words = [w for w in DEFAULT_WORDS if (min_len <= len(w) <= max_len)]

    if not words:
        raise ValueError("No words available after applying filters")

    if vocab_size and vocab_size < len(words):
        words = rng.sample(words, vocab_size)
    return tuple(words)


def build_cum_weights(
    distribution: str,
    vocab_size: int,
    normal_mean: float,
    normal_stddev: float,
) -> Optional[List[float]]:
    if distribution == "uniform":
        return None
    if distribution != "normal":
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not (0.0 <= normal_mean <= 1.0):
        raise ValueError("normal_mean must be between 0 and 1")
    if not (0.0 < normal_stddev <= 1.0):
        raise ValueError("normal_stddev must be in (0, 1]")

    mu = normal_mean * (vocab_size - 1)
    sigma = max(normal_stddev * vocab_size, 1e-6)
    weights = [
        math.exp(-0.5 * ((idx - mu) / sigma) ** 2) for idx in range(vocab_size)
    ]
    if not any(weight > 0 for weight in weights):
        raise ValueError("Normal distribution produced zero weights")
    return list(itertools.accumulate(weights))


def sample_words(
    rng: random.Random,
    vocab: Sequence[str],
    cum_weights: Optional[Sequence[float]],
    words_per_doc: int,
) -> List[str]:
    if cum_weights is None:
        return rng.choices(vocab, k=words_per_doc)
    return rng.choices(vocab, cum_weights=cum_weights, k=words_per_doc)


def iter_docs(
    rows: int,
    words_per_doc: int,
    vocab: Sequence[str],
    cum_weights: Optional[Sequence[float]],
    rng: random.Random,
) -> Iterable[str]:
    for _ in range(rows):
        yield " ".join(sample_words(rng, vocab, cum_weights, words_per_doc))


def write_batches(
    db_uri: str,
    table_name: str,
    total_rows: int,
    words_per_doc: int,
    vocab: Sequence[str],
    cum_weights: Optional[Sequence[float]],
    batch_rows: int,
    seed: int,
    mode: str,
    with_id: bool,
    start_id: int,
    log_every: int,
) -> None:
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover - only used when lancedb missing
        raise SystemExit(
            "lancedb is required for this script. Install with 'pip install lancedb'."
        ) from exc
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - only used when pyarrow missing
        raise SystemExit(
            "pyarrow is required for this script. Install with 'pip install pyarrow'."
        ) from exc

    rng = random.Random(seed)
    db = lancedb.connect(db_uri)

    table = None
    table_exists = False
    rows_written = 0
    next_id = start_id
    start_time = time.perf_counter()

    while rows_written < total_rows:
        batch_size = min(batch_rows, total_rows - rows_written)
        docs = list(iter_docs(batch_size, words_per_doc, vocab, cum_weights, rng))
        arrays = {"doc": pa.array(docs, type=pa.large_string())}
        if with_id:
            ids = range(next_id, next_id + batch_size)
            arrays["id"] = pa.array(ids, type=pa.uint64())
            next_id += batch_size

        batch = pa.table(arrays)

        if table is None:
            if mode == "append":
                try:
                    table = db.open_table(table_name)
                    table_exists = True
                    table.add(batch)
                except Exception:
                    table = db.create_table(table_name, data=batch)
            else:
                table = db.create_table(table_name, data=batch, mode="overwrite")
        else:
            table.add(batch)

        rows_written += batch_size
        if log_every and rows_written % log_every == 0:
            elapsed = time.perf_counter() - start_time
            rate = rows_written / max(elapsed, 1e-6)
            print(
                f"wrote {rows_written:,} rows in {elapsed:.1f}s "
                f"({rate:,.0f} rows/s)"
            )

    elapsed = time.perf_counter() - start_time
    rate = rows_written / max(elapsed, 1e-6)
    print(
        f"done: wrote {rows_written:,} rows in {elapsed:.1f}s "
        f"({rate:,.0f} rows/s)"
    )
    if mode == "append" and with_id and table_exists and start_id == 0:
        print(
            "warning: appended with id column starting at 0; "
            "consider setting --start-id to avoid collisions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("db_uri", help="lancedb database URI")
    parser.add_argument("table", help="table name")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--words-per-doc", type=int, default=8_192)
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--word-list", type=str, default=None)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument("--max-word-len", type=int, default=24)
    parser.add_argument(
        "--distribution",
        choices=["normal", "uniform"],
        default="normal",
        help="Distribution used to sample words",
    )
    parser.add_argument(
        "--normal-mean",
        type=float,
        default=0.5,
        help="Mean for normal distribution as fraction of vocab (0-1)",
    )
    parser.add_argument(
        "--normal-stddev",
        type=float,
        default=0.15,
        help="Stddev for normal distribution as fraction of vocab (0-1]",
    )
    parser.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    parser.add_argument("--with-id", action="store_true", default=False)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vocab = resolve_words(
        args.word_list,
        args.min_word_len,
        args.max_word_len,
        args.vocab_size,
        rng,
    )

    avg_len = sum(len(word) for word in vocab) / len(vocab)
    approx_doc_bytes = args.words_per_doc * (avg_len + 1) - 1
    approx_total = approx_doc_bytes * args.rows
    print(
        "estimated raw text size: "
        f"{human_bytes(approx_total)} "
        f"({args.rows:,} rows x {args.words_per_doc:,} words)"
    )
    print(
        f"vocab size: {len(vocab):,}, batch size: {args.batch_rows:,}, "
        f"mode: {args.mode}, distribution: {args.distribution}"
    )
    cum_weights = build_cum_weights(
        args.distribution, len(vocab), args.normal_mean, args.normal_stddev
    )
    if cum_weights is not None:
        print(
            f"normal mean: {args.normal_mean:.3f}, "
            f"normal stddev: {args.normal_stddev:.3f}"
        )

    write_batches(
        db_uri=args.db_uri,
        table_name=args.table,
        total_rows=args.rows,
        words_per_doc=args.words_per_doc,
        vocab=vocab,
        cum_weights=cum_weights,
        batch_rows=args.batch_rows,
        seed=args.seed,
        mode=args.mode,
        with_id=args.with_id,
        start_id=args.start_id,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
