"""Perception service interface (ASR, command parsing, detection, verification, re-ID).

The real service (``perception_service/``) runs in its own GPU environment on g1-gpu and is
reached through ``perception.client.PerceptionClient``. It never opens cameras: the daemon
sends it JPEG frames.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class RawDetection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


class Perception(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, wav: bytes) -> str: ...

    @abc.abstractmethod
    def parse_command(self, text: str, catalog: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """``{"profile": name, "object": str, "attributes": [..]}``; None if not a fetch command."""

    @abc.abstractmethod
    def detect(self, jpeg: bytes, labels: Sequence[str]) -> list[RawDetection]: ...

    @abc.abstractmethod
    def verify(self, jpeg: bytes, bbox: Sequence[float], question: str) -> Optional[bool]: ...

    @abc.abstractmethod
    def embed_person(self, jpeg: bytes, bbox: Sequence[float]) -> Optional[tuple[float, ...]]: ...

    @abc.abstractmethod
    def health(self) -> dict[str, Any]: ...
