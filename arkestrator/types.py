"""Shared value types. Units: meters, radians, seconds.

Frames: odometry/world frame from ``rt/odommodestate``; robot (torso) frame has x forward,
y left, z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def to_local(self, wx: float, wy: float) -> tuple[float, float]:
        """World point → robot frame (x forward, y left)."""
        dx, dy = wx - self.x, wy - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def to_world(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + c * lx - s * ly, self.y + s * lx + c * ly)

    def distance_to(self, wx: float, wy: float) -> float:
        return math.hypot(wx - self.x, wy - self.y)

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.yaw]


@dataclass(frozen=True)
class Odometry:
    pose: Pose2D
    speed_m_s: float  # horizontal speed magnitude
    yaw_speed_rad_s: float
    stamp: float


@dataclass(frozen=True)
class Detection:
    """One detected object or person on one camera frame.

    ``position`` is in the robot frame *at capture time* (``capture_pose``); use
    ``world_xy`` to carry it through later base motion.
    """

    label: str
    score: float
    camera: str
    frame_number: int
    stamp: float
    capture_pose: Pose2D
    position: Optional[tuple[float, float, float]] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    embedding: Optional[tuple[float, ...]] = None
    attributes: tuple[str, ...] = ()

    @property
    def world_xy(self) -> Optional[tuple[float, float]]:
        if self.position is None:
            return None
        return self.capture_pose.to_world(self.position[0], self.position[1])

    @property
    def bearing(self) -> Optional[float]:
        if self.position is None:
            return None
        return math.atan2(self.position[1], self.position[0])

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "camera": self.camera,
            "frame": self.frame_number,
            "position": None if self.position is None else [round(v, 3) for v in self.position],
            "world_xy": None if self.world_xy is None else [round(v, 3) for v in self.world_xy],
        }


@dataclass(frozen=True)
class Task:
    """A parsed fetch command."""

    profile: str
    object_label: str
    text: str
    source: str  # "vlm" | "keywords"
    attributes: tuple[str, ...] = ()


T = TypeVar("T")


@dataclass(frozen=True)
class Sighting(Generic[T]):
    """Result of polling vision: ``fresh`` is False when no new frame arrived since last poll."""

    fresh: bool
    value: Optional[T] = None


@dataclass
class TaskRecord:
    """What happened during one task (for the UI and outputs/<session>/tasks.jsonl)."""

    task: Task
    started: float
    finished: Optional[float] = None
    outcome: str = "running"  # success | recovered:<reason> | safe_hold:<reason>
    grasp_attempts: int = 0
    stages: list[str] = field(default_factory=list)
