"""Vision: turns camera frames into detections with robot-frame positions.

``Vision`` is what skills use. ``PerceptionVision`` is the real implementation: it sends each
new frame to the perception service and lifts boxes to 3D with the aligned depth (median of
the central half of the box, as in GR00T-pickup-bottle ``disposal/vision.py``) and the camera
extrinsics. The mock backend provides its own ``Vision``.
"""

from __future__ import annotations

import abc
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .errors import RecoverableError
from .perception.base import Perception
from .types import Detection, Sighting


class Vision(abc.ABC):
    @abc.abstractmethod
    def poll(self, labels: Sequence[str], with_embeddings: bool = False) -> Sighting[list[Detection]]:
        """Detections on frames newer than the previous poll; ``fresh=False`` if none arrived."""

    @abc.abstractmethod
    def verify(self, detection: Detection, question: str, attributes: Sequence[str] = ()) -> Optional[bool]:
        """Ask the VLM about the crop of ``detection``; None when unavailable.

        ``attributes`` repeats the requested properties for backends without a VLM (mock)."""

    @abc.abstractmethod
    def ask(self, question: str) -> Optional[bool]:
        """Yes/no question about the newest head frame (e.g. «робот держит бутылку?»)."""


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera optical frame (x right, y down, z forward) → torso frame (x fwd, y left, z up)."""

    rotation: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]

    def to_torso(self, point: Sequence[float]) -> tuple[float, float, float]:
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation_m, dtype=float)
        p = r @ np.asarray(point, dtype=float) + t
        return (float(p[0]), float(p[1]), float(p[2]))


def bbox_depth_point(
    depth_m: np.ndarray,
    bbox: Sequence[float],
    intrinsics: dict[str, float],
    min_valid: int = 20,
) -> Optional[tuple[float, float, float]]:
    """3D point (camera optical frame) of a box: median depth of its central half.

    Returns None when fewer than ``min_valid`` pixels carry a finite positive depth.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array in meters")
    h, w = depth.shape
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 4, (y1 - y0) / 4
    c0, c1 = max(0, int(round(cx - hw))), min(w, int(round(cx + hw)) + 1)
    r0, r1 = max(0, int(round(cy - hh))), min(h, int(round(cy + hh)) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    patch = depth[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < min_valid:
        return None
    z = float(np.median(valid))
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    ppx, ppy = float(intrinsics["ppx"]), float(intrinsics["ppy"])
    return ((cx - ppx) * z / fx, (cy - ppy) * z / fy, z)


class PerceptionVision(Vision):
    """Real vision over the robot's frames and the perception service.

    Detection runs in a background thread on the newest frames (like GR00T-pickup-bottle
    ``BinPerception``), so a slow model never stretches the 20 ms control tick while the base
    turns. ``poll`` only picks up finished results. ``verify``/``ask`` are synchronous and are
    called while the base stands still.
    """

    def __init__(
        self,
        robot,
        perception: Perception,
        extrinsics: dict[str, CameraExtrinsics],
        cameras: Sequence[str],
        person_labels: Sequence[str] = ("person",),
        cache_frames: int = 16,
        idle_sleep_s: float = 0.01,
    ):
        self.robot = robot
        self.perception = perception
        self.extrinsics = dict(extrinsics)
        self.cameras = tuple(cameras)
        self.person_labels = tuple(person_labels)
        self._cache: "OrderedDict[tuple[str, int], object]" = OrderedDict()
        self._cache_frames = cache_frames
        self._idle_sleep_s = idle_sleep_s
        self._lock = threading.Condition()
        self._request: tuple[tuple[str, ...], bool] = ((), False)
        self._latest: Optional[tuple[int, list[Detection]]] = None
        self._sequence = 0
        self._returned = 0
        self._error: Optional[str] = None
        self._last_number: dict[str, int] = {}
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self._lock.notify_all()
        self._thread.join(2.0)

    def poll(self, labels: Sequence[str], with_embeddings: bool = False) -> Sighting[list[Detection]]:
        request = (tuple(labels), bool(with_embeddings))
        with self._lock:
            if request != self._request:
                # New question: results computed for the previous labels are discarded.
                self._request = request
                self._latest = None
                self._returned = self._sequence
                self._lock.notify_all()
                return Sighting(False)
            if self._error is not None:
                raise RecoverableError("perception_unavailable", say="Зрение недоступно", error=self._error)
            if self._latest is None or self._latest[0] == self._returned:
                return Sighting(False)
            sequence, detections = self._latest
            self._returned = sequence
            return Sighting(True, detections)

    def _run(self) -> None:
        while not self._closed.is_set():
            with self._lock:
                request = self._request
                if not request[0]:
                    self._lock.wait(0.05)
                    continue
            try:
                detections, fresh = self._detect(*request)
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                self._closed.wait(0.1)
                continue
            with self._lock:
                self._error = None
                if fresh and request == self._request:
                    self._sequence += 1
                    self._latest = (self._sequence, detections)
            if not fresh:
                self._closed.wait(self._idle_sleep_s)

    def _detect(self, labels: tuple[str, ...], with_embeddings: bool) -> tuple[list[Detection], bool]:
        frames = self.robot.frames()
        detections: list[Detection] = []
        fresh = False
        for name in self.cameras:
            frame = frames.get(name)
            if frame is None or frame.jpeg is None or self._last_number.get(name) == frame.number:
                continue
            self._last_number[name] = frame.number
            fresh = True
            self._remember(frame)
            for raw in self.perception.detect(frame.jpeg, labels):
                position = None
                if frame.depth_m is not None and frame.intrinsics and name in self.extrinsics:
                    point = bbox_depth_point(frame.depth_m, raw.bbox, frame.intrinsics)
                    if point is not None:
                        position = self.extrinsics[name].to_torso(point)
                embedding = None
                if with_embeddings and raw.label in self.person_labels:
                    embedding = self.perception.embed_person(frame.jpeg, raw.bbox)
                detections.append(
                    Detection(
                        label=raw.label,
                        score=raw.score,
                        camera=name,
                        frame_number=frame.number,
                        stamp=frame.stamp,
                        capture_pose=frame.pose,
                        position=position,
                        bbox=tuple(raw.bbox),
                        embedding=embedding,
                    )
                )
        return detections, fresh

    def verify(self, detection: Detection, question: str, attributes: Sequence[str] = ()) -> Optional[bool]:
        with self._lock:
            frame = self._cache.get((detection.camera, detection.frame_number))
        if frame is None or detection.bbox is None:
            return None
        return self.perception.verify(frame.jpeg, detection.bbox, question)

    def ask(self, question: str) -> Optional[bool]:
        frame = self.robot.frames().get(self.cameras[0]) if self.cameras else None
        if frame is None or frame.jpeg is None:
            return None
        return self.perception.verify(frame.jpeg, (), question)

    def _remember(self, frame) -> None:
        with self._lock:
            self._cache[(frame.camera, frame.number)] = frame
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
