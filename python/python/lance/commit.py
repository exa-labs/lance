# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from contextlib import AbstractContextManager
from typing import Callable

CommitLock = Callable[[int], AbstractContextManager]


class CommitConflictError(Exception):
    pass


class DuplicateTransactionError(OSError):
    """A transaction committed after the read version already carries some of
    this commit's idempotency keys (see ``idempotency_property`` on
    :meth:`lance.LanceDataset.commit`).

    ``version`` is the version that carries them and ``keys`` the overlapping
    keys. The version may be this writer's own earlier attempt, when a put
    landed but its response was lost; callers tell the two cases apart by
    their own commit identity.
    """

    def __init__(self, message: str, version: int, keys: list[str]):
        super().__init__(message)
        self.version = version
        self.keys = keys
