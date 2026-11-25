"""Obstacle source interface. v1 has no lidar, so the corridor is always UNKNOWN.

A lidar (G1 MID-360) or SLAM implementation plugs in here later; ``Navigator`` refuses a walk
when the source reports BLOCKED, and also on UNKNOWN when ``require_clear_corridor`` is set.
"""

from __future__ import annotations

import enum

from ..types import Pose2D


class Corridor(str, enum.Enum):
    UNKNOWN = "unknown"
    CLEAR = "clear"
    BLOCKED = "blocked"


class ObstacleSource:
    def check_corridor(self, pose: Pose2D, distance_m: float, lateral: bool) -> Corridor:
        raise NotImplementedError


class NullObstacles(ObstacleSource):
    def check_corridor(self, pose: Pose2D, distance_m: float, lateral: bool) -> Corridor:
        return Corridor.UNKNOWN
