"""Keeps at most ``resident`` π0.5 servers loaded (LRU) and loads the next one in background.

The orchestrator calls ``prefetch`` right after parsing a command, so the model loads while the
robot searches and walks; ``acquire`` then usually returns immediately. Older models are closed
*before* the next one loads (VRAM), and a model in use is never evicted: the loader waits until
it is released. A broken client (failed exchange) is dropped and loaded again on next use.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Optional

from ..errors import RecoverableError
from .client import PolicyClient


class _Slot:
    def __init__(self):
        self.client: Optional[PolicyClient] = None
        self.error: Optional[str] = None
        self.done = threading.Event()
        self.in_use = False

    @property
    def usable(self) -> bool:
        if not self.done.is_set():
            return True  # still loading
        return self.client is not None and self.client.healthy


class PolicyPool:
    def __init__(self, factory: Callable[[object], PolicyClient], resident: int = 1, log=None):
        if resident < 1:
            raise ValueError("resident must be at least 1")
        self.factory = factory
        self.resident = resident
        self.log = log
        self._changed = threading.Condition()
        self._slots: "OrderedDict[str, _Slot]" = OrderedDict()

    def _emit(self, kind: str, **fields) -> None:
        if self.log is not None:
            self.log.emit(kind, **fields)

    def prefetch(self, profile) -> None:
        with self._changed:
            slot = self._slots.get(profile.name)
            if slot is not None and slot.usable:
                self._slots.move_to_end(profile.name)
                return
            if slot is not None:
                self._slots.pop(profile.name)
                threading.Thread(target=self._close_slot, args=(profile.name, slot), daemon=True).start()
            slot = _Slot()
            self._slots[profile.name] = slot
        self._emit("vla_loading", profile=profile.name)
        threading.Thread(target=self._load, args=(profile, slot), daemon=True, name=f"vla-load-{profile.name}").start()

    def _make_room(self, keep: str) -> None:
        """Close least-recently-used idle models until ``keep`` fits; wait while others are in use."""
        while True:
            with self._changed:
                others = [(name, s) for name, s in self._slots.items() if name != keep]
                if len(others) < self.resident:
                    return
                victim = next(((name, s) for name, s in others if not s.in_use), None)
                if victim is None:
                    self._changed.wait(0.5)
                    continue
                self._slots.pop(victim[0])
            self._close_slot(*victim)

    def _load(self, profile, slot: _Slot) -> None:
        try:
            self._make_room(keep=profile.name)
            slot.client = self.factory(profile)
            self._emit("vla_ready", profile=profile.name)
        except Exception as exc:  # reported to the task through acquire()
            slot.error = f"{type(exc).__name__}: {exc}"
            self._emit("vla_load_failed", profile=profile.name, error=slot.error)
        finally:
            slot.done.set()
            with self._changed:
                self._changed.notify_all()

    def _close_slot(self, name: str, slot: _Slot) -> None:
        slot.done.wait()
        if slot.client is not None:
            slot.client.close()
            self._emit("vla_unloaded", profile=name)

    def acquire(self, profile, wait: Callable[[], None]) -> PolicyClient:
        """Block (calling ``wait`` between checks, so STOP still works) until loaded."""
        self.prefetch(profile)
        with self._changed:
            slot = self._slots[profile.name]
        while not slot.done.is_set():
            wait()
        if slot.client is None or not slot.client.healthy:
            with self._changed:
                if self._slots.get(profile.name) is slot:
                    self._slots.pop(profile.name)
            raise RecoverableError("vla_unavailable", say="Модель захвата не загрузилась", error=slot.error)
        with self._changed:
            slot.in_use = True
        return slot.client

    def release(self, profile) -> None:
        with self._changed:
            slot = self._slots.get(profile.name)
            if slot is not None:
                slot.in_use = False
            self._changed.notify_all()

    def status(self) -> dict[str, str]:
        with self._changed:
            result = {}
            for name, slot in self._slots.items():
                if not slot.done.is_set():
                    result[name] = "loading"
                elif slot.client is None:
                    result[name] = "error"
                elif not slot.client.healthy:
                    result[name] = "broken"
                else:
                    result[name] = "in_use" if slot.in_use else "ready"
            return result

    def close_all(self) -> None:
        with self._changed:
            slots = list(self._slots.items())
            self._slots.clear()
        for name, slot in slots:
            self._close_slot(name, slot)
