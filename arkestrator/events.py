"""Session event log (jsonl + in-memory tail for the UI) and the shared UI state board."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_list"):
        return value.as_list()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


class EventLog:
    """Append-only events. ``folder`` None keeps them in memory only (tests)."""

    def __init__(self, folder: Optional[Path], clock, capacity: int = 400):
        self.clock = clock
        self._lock = threading.Lock()
        self._tail: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._sequence = 0
        self._file = None
        self.folder = folder
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self._file = (folder / "events.jsonl").open("a", encoding="utf-8")

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            row = {"seq": self._sequence, "t": round(self.clock.now(), 3), "wall": time.time(), "kind": kind, **fields}
            self._tail.append(row)
            if self._file is not None:
                self._file.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
                self._file.flush()
            return row

    def tail(self, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = [row for row in self._tail if row["seq"] > after]
        return rows[-limit:]

    def kinds(self) -> list[str]:
        with self._lock:
            return [row["kind"] for row in self._tail]

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class StateBoard:
    """Thread-safe snapshot of what the UI shows."""

    def __init__(self, **initial: Any):
        self._lock = threading.Lock()
        self._state: dict[str, Any] = dict(initial)

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, ensure_ascii=False, default=_jsonable))
