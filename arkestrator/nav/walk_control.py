# Source pin: GR00T-pickup-bottle bottle/walk_control.py, working tree on top of 9133ac6
# (uncommitted lateral-walk change), sha256 4b6aea72aa66d7bb87ecea2c147c5369003377f0ef03fade5c75b56a10d97e61.
# Copied verbatim below this header; tests/test_navigator.py::test_vendored_walk_control_matches_pin
# checks the hash. Change it only together with the pin.

"""Forward-walk target from measured odometry.

The factory controller owns the legs. This module only turns a requested
distance and speed into velocity commands; it has no mode, freshness, thermal
or arm-hold gates.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WalkFeedback:
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    yaw_rad: float
    yaw_speed_rad_s: float


@dataclass(frozen=True)
class WalkDecision:
    velocity: tuple[float, float, float]
    done: bool
    forward_m: float
    lateral_m: float
    heading_error_rad: float


class ForwardWalk:
    """Walk the requested odometry distance at the requested speed."""

    def __init__(self, distance_m, *, speed_m_s=0.2, heading_gain=1.2, lateral=False):
        if not math.isfinite(distance_m) or distance_m == 0:
            raise ValueError("Walking distance must be finite and nonzero")
        if not math.isfinite(speed_m_s) or speed_m_s <= 0:
            raise ValueError("Walking speed must be finite and positive")
        self.distance_m = float(distance_m)
        self.speed_m_s = float(speed_m_s)
        self.heading_gain = float(heading_gain)
        self.direction = math.copysign(1., distance_m)
        # lateral=True steps sideways (+ left) along the starting heading instead of forward.
        self.lateral = bool(lateral)
        self.origin = None
        self.origin_yaw = None

    def step(self, feedback):
        if not isinstance(feedback, WalkFeedback):
            raise ValueError("Walking feedback is required")
        position = tuple(float(value) for value in feedback.position_m)
        yaw = float(feedback.yaw_rad)
        if len(position) != 3 or not all(math.isfinite(value) for value in (*position, yaw)):
            raise ValueError("Invalid walking odometry")
        if self.origin is None:
            self.origin = position
            self.origin_yaw = yaw

        axis = (math.cos(self.origin_yaw), math.sin(self.origin_yaw))
        dx = position[0] - self.origin[0]
        dy = position[1] - self.origin[1]
        forward = dx * axis[0] + dy * axis[1]
        lateral = -dx * axis[1] + dy * axis[0]
        heading = math.atan2(math.sin(yaw - self.origin_yaw), math.cos(yaw - self.origin_yaw))
        progress = lateral if self.lateral else forward
        done = self.direction * progress >= abs(self.distance_m)
        speed = self.direction * self.speed_m_s
        command = (0.0, 0.0, 0.0) if done else (
            (0.0, speed) if self.lateral else (speed, 0.0)) + (-self.heading_gain * heading,)
        return WalkDecision(command, done, forward, lateral, heading)
