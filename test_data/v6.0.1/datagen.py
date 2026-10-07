# /// script
# requires-python = ">=3.10"
# dependencies = ["pylance==6.0.1", "pyarrow==21.0.0", "numpy"]
# ///
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate a legacy list page whose structural count exceeds a u16."""

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.file import LanceFileWriter

assert lance.__version__ == "6.0.1"

vector = np.arange(2048, dtype=np.float16).tolist()
values = [vector] * 64 + [None] * 70_000 + [vector]
table = pa.table(
    {
        "id": pa.array(range(len(values)), type=pa.int64()),
        "vector": pa.array(values, type=pa.large_list(pa.float16())),
    },
    schema=pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "vector",
                pa.large_list(pa.float16()),
                metadata={
                    b"lance-encoding:structural-encoding": b"miniblock",
                },
            ),
        ]
    ),
)
with LanceFileWriter(
    str(Path(__file__).with_name("wrapped_bitpacked_levels.lance")),
    table.schema,
    version="2.1",
) as writer:
    writer.write_batch(table)
