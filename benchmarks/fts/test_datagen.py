#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import importlib.util
import random
from pathlib import Path

DATAGEN_PATH = Path(__file__).with_name("datagen.py")

spec = importlib.util.spec_from_file_location("fts_datagen", DATAGEN_PATH)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError("Failed to load datagen module")

fts_datagen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fts_datagen)


def test_uniform_weights_none():
    weights = fts_datagen.build_cum_weights("uniform", 10, 0.5, 0.2)
    assert weights is None


def test_normal_distribution_skewed():
    rng = random.Random(0)
    vocab = ["a", "b", "c", "d", "e"]
    cum_weights = fts_datagen.build_cum_weights("normal", len(vocab), 0.5, 0.12)
    words = fts_datagen.sample_words(rng, vocab, cum_weights, 10_000)
    counts = Counter(words)

    assert counts["c"] > counts["a"]
    assert counts["c"] > counts["e"]
