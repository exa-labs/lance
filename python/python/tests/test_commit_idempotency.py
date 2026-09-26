# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json

import lance
import pyarrow as pa
import pytest
from lance.commit import DuplicateTransactionError
from lance.dataset import LanceOperation, Transaction

PROPERTY = "idempotency_keys"


def _staged(uri, read_version, values, keys):
    fragments = lance.fragment.write_fragments(pa.table({"x": values}), uri)
    properties = {PROPERTY: json.dumps(keys)} if keys is not None else {}
    return Transaction(
        read_version=read_version,
        operation=LanceOperation.Append(fragments),
        transaction_properties=properties,
    )


def _values(uri):
    return sorted(lance.dataset(uri).to_table()["x"].to_pylist())


def test_overlapping_keys_raise_duplicate_transaction(tmp_path):
    uri = str(tmp_path / "ds")
    lance.write_dataset(pa.table({"x": [0]}), uri)
    first = _staged(uri, 1, [1, 2], ["a", "b"])
    second = _staged(uri, 1, [2, 3], ["b", "c"])

    lance.LanceDataset.commit(uri, first, idempotency_property=PROPERTY)
    with pytest.raises(DuplicateTransactionError) as raised:
        lance.LanceDataset.commit(
            uri, second, max_retries=0, idempotency_property=PROPERTY
        )

    assert isinstance(raised.value, OSError)
    assert raised.value.version == 2
    assert raised.value.keys == ["b"]
    assert _values(uri) == [0, 1, 2]


def test_disjoint_keys_rebase(tmp_path):
    uri = str(tmp_path / "ds")
    lance.write_dataset(pa.table({"x": [0]}), uri)
    first = _staged(uri, 1, [1], ["a"])
    second = _staged(uri, 1, [2], ["b"])

    lance.LanceDataset.commit(uri, first, idempotency_property=PROPERTY)
    landed = lance.LanceDataset.commit(uri, second, idempotency_property=PROPERTY)

    assert landed.version == 3
    assert _values(uri) == [0, 1, 2]


def test_without_the_option_overlapping_commits_both_land(tmp_path):
    uri = str(tmp_path / "ds")
    lance.write_dataset(pa.table({"x": [0]}), uri)
    first = _staged(uri, 1, [1, 2], ["a", "b"])
    second = _staged(uri, 1, [2, 3], ["b", "c"])

    lance.LanceDataset.commit(uri, first)
    lance.LanceDataset.commit(uri, second)

    assert _values(uri) == [0, 1, 2, 2, 3]


def test_missing_property_on_our_transaction_is_rejected(tmp_path):
    uri = str(tmp_path / "ds")
    lance.write_dataset(pa.table({"x": [0]}), uri)
    txn = _staged(uri, 1, [1], None)

    with pytest.raises(OSError, match="idempotency property"):
        lance.LanceDataset.commit(uri, txn, idempotency_property=PROPERTY)
