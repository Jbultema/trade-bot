from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

DEFAULT_DB_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_DB_CONNECT_RETRY_SECONDS = 5.0


class DatabaseWriteLockTimeout(TimeoutError):
    """Raised when another trade-bot writer does not release the store promptly."""


@contextmanager
def database_write_lock(
    db_path: str | Path,
    *,
    timeout_seconds: float = DEFAULT_DB_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = 0.05,
) -> Iterator[None]:
    """Serialize trade-bot writers across dashboard and worker processes."""

    path = Path(db_path)
    lock_path = path.with_suffix(f"{path.suffix}.write.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise DatabaseWriteLockTimeout(
                        f"Timed out waiting for database writer lock: {lock_path}"
                    ) from error
                time.sleep(poll_seconds)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def connect_with_retry(
    db_path: str | Path,
    *,
    read_only: bool,
    timeout_seconds: float = DEFAULT_DB_CONNECT_RETRY_SECONDS,
    poll_seconds: float = 0.1,
) -> duckdb.DuckDBPyConnection:
    """Retry only transient DuckDB file-lock conflicts, preserving other failures."""

    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            return duckdb.connect(str(db_path), read_only=read_only)
        except duckdb.IOException as error:
            message = str(error).lower()
            if "lock" not in message or time.monotonic() >= deadline:
                raise
            time.sleep(poll_seconds)
