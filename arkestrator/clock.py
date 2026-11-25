"""Time source shared by every control loop.

The real daemon uses the monotonic clock. Offline runs use ``SimClock``: ``sleep``
advances simulated time and steps the mock world, so a 30 s walk takes milliseconds in
tests, or real time in the interactive mock (``realtime_factor=1``).
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class Clock:
    def now(self) -> float:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class MonotonicClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


class SimClock(Clock):
    """Simulated time; ``sleep`` advances it in ``step_s`` increments."""

    def __init__(self, step_s: float = 0.02, realtime_factor: float = 0.0):
        if step_s <= 0:
            raise ValueError("step_s must be positive")
        self.step_s = float(step_s)
        self.realtime_factor = float(realtime_factor)
        self._now = 0.0
        self._lock = threading.RLock()
        self._listeners: list[Callable[[float], None]] = []

    def subscribe(self, listener: Callable[[float], None]) -> None:
        """``listener(dt)`` runs after every simulated step (under the clock lock)."""
        with self._lock:
            self._listeners.append(listener)

    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        remaining = max(0.0, float(seconds))
        while remaining > 1e-12:
            dt = min(self.step_s, remaining)
            with self._lock:
                self._now += dt
                listeners = list(self._listeners)
                for listener in listeners:
                    listener(dt)
            remaining -= dt
            if self.realtime_factor > 0:
                time.sleep(dt * self.realtime_factor)
