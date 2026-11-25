"""Base motion primitives on top of odometry, ported from GR00T-pickup-bottle
``bottle/disposal/pipeline.py`` (walk / turn_until / turn_by / wait_still / Progress) so the
same field-tested behaviour runs against the mock and the G1 backend.

Invariants enforced here:
* Base velocity is sent only inside the robot's mobile segment (arms held, no pose changes).
* No turn while ``near_table`` is set: the robot stands close to the table to grasp, so it
  must BACKSTEP first. Violations raise ``NavigationRuleError`` (a programming error).
* Walking at the table (BACKSTEP, final approach, side steps) keeps the guards of
  ``post_grasp_walk.perform_motion``: straight corridor, heading, wrong direction → SAFE_HOLD,
  heading correction limited to ``careful_max_yaw_rate_rad_s``.
* Every base command is finite and within ``max_linear_m_s`` / ``max_yaw_rate_rad_s``.
* Only forward-walk waypoints recorded away from the table are retraced on the way back.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Callable, Optional, Tuple, TypeVar

from ..config import NavConfig
from ..errors import NavigationRuleError, RecoverableError, SafeHoldError
from ..types import Pose2D, Sighting, wrap_angle
from .obstacles import Corridor, NullObstacles, ObstacleSource
from .walk_control import ForwardWalk, WalkFeedback

T = TypeVar("T")


class _Progress:
    """Fail when the base accepts commands but does not move (19.11: 80 s at 0.1 m/s)."""

    def __init__(self, nav: "Navigator", minimum: float, what: str):
        self.nav, self.minimum, self.what = nav, minimum, what
        self.best = 0.0
        self.since = nav.clock.now()

    def check(self, amount: float) -> None:
        now = self.nav.clock.now()
        if abs(amount) >= self.best + self.minimum:
            self.best, self.since = abs(amount), now
        elif now - self.since > self.nav.cfg.stall_timeout_s:
            raise RecoverableError("base_stalled", say="Не могу сдвинуться с места", what=self.what)


class Navigator:
    def __init__(
        self,
        robot,
        clock,
        cfg: NavConfig,
        checkpoint: Callable[[], None],
        obstacles: Optional[ObstacleSource] = None,
        log=None,
    ):
        self.robot = robot
        self.clock = clock
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.obstacles = obstacles or NullObstacles()
        self.log = log
        self.near_table = False
        self.waypoints: list[Pose2D] = []
        self.guard: Optional[Callable[[], None]] = None  # e.g. carry check while holding
        self._segment_depth = 0
        self._segment_stack: Optional[ExitStack] = None

    # --- helpers ---------------------------------------------------------------------------
    def _emit(self, kind: str, **fields) -> None:
        if self.log is not None:
            self.log.emit(kind, **fields)

    def pose(self) -> Pose2D:
        return self.robot.odometry().pose

    def tick(self, velocity: Optional[Tuple[float, float, float]] = None) -> None:
        """One control period: checks (STOP, faults, guard), then the command, then sleep."""
        self.checkpoint()
        if self.guard is not None:
            self.guard()
        if velocity is not None:
            if self._segment_depth == 0:
                raise NavigationRuleError("base velocity outside a mobile segment")
            vx, vy, wz = velocity
            if not all(math.isfinite(v) for v in velocity):
                raise NavigationRuleError(f"non-finite base command {velocity}")
            if math.hypot(vx, vy) > self.cfg.max_linear_m_s + 1e-9 or abs(wz) > self.cfg.max_yaw_rate_rad_s + 1e-9:
                raise NavigationRuleError(f"base command {velocity} exceeds the configured caps")
            self.robot.set_base_velocity(vx, vy, wz)
        self.clock.sleep(self.cfg.tick_s)

    def segment(self):
        """Re-entrant mobile segment: the robot's hold is entered once for nested motions."""
        nav = self

        class _Segment:
            def __enter__(self_inner):
                if nav._segment_depth == 0:
                    stack = ExitStack()
                    stack.enter_context(nav.robot.mobile_segment())
                    nav._segment_stack = stack
                nav._segment_depth += 1

            def __exit__(self_inner, *exc):
                nav._segment_depth -= 1
                if nav._segment_depth == 0:
                    stack, nav._segment_stack = nav._segment_stack, None
                    try:
                        nav.robot.stop_base()
                    finally:
                        stack.__exit__(*exc)
                return False

        return _Segment()

    def wait_still(self) -> None:
        self.robot.stop_base()
        settled = None
        while True:
            odom = self.robot.odometry()
            still = odom.speed_m_s < self.cfg.still_speed_m_s and abs(odom.yaw_speed_rad_s) < self.cfg.still_yaw_rad_s
            now = self.clock.now()
            if still:
                if settled is None:
                    settled = now
                if now - settled >= self.cfg.settle_s:
                    return
            else:
                settled = None
            self.tick()

    def _require_turn_allowed(self) -> None:
        if self.near_table:
            raise NavigationRuleError("turn requested at the table before BACKSTEP")

    def _check_corridor(self, distance: float, lateral: bool) -> None:
        status = self.obstacles.check_corridor(self.pose(), distance, lateral)
        if status == Corridor.BLOCKED:
            raise RecoverableError("path_blocked", say="Путь перекрыт", distance_m=distance)
        if status != Corridor.CLEAR and self.cfg.require_clear_corridor:
            raise RecoverableError("corridor_unknown", say="Не вижу, свободен ли путь", distance_m=distance)

    def _record_waypoint(self) -> None:
        if not self.near_table:
            self.waypoints.append(self.pose())

    # --- primitives --------------------------------------------------------------------------
    def begin_task(self) -> None:
        self.waypoints = [self.pose()]

    def walk(self, distance_m: float, lateral: bool = False, speed_m_s: Optional[float] = None) -> None:
        if abs(distance_m) < self.cfg.min_walk_m:
            return
        self._check_corridor(distance_m, lateral)
        walker = ForwardWalk(
            distance_m,
            speed_m_s=speed_m_s or self.cfg.walk_speed_m_s,
            heading_gain=self.cfg.heading_gain,
            lateral=lateral,
        )
        careful = self.near_table
        yaw_cap = self.cfg.careful_max_yaw_rate_rad_s if careful else self.cfg.max_yaw_rate_rad_s
        self._emit("walk", distance_m=round(distance_m, 3), lateral=lateral, careful=careful)
        with self.segment():
            progress = _Progress(self, 0.02, "walk")
            while True:
                odom = self.robot.odometry()
                pose = odom.pose
                decision = walker.step(
                    WalkFeedback((pose.x, pose.y, 0.0), (odom.speed_m_s, 0.0, 0.0), pose.yaw, odom.yaw_speed_rad_s)
                )
                if careful:
                    self._check_table_walk(decision, lateral, walker.direction)
                if decision.done:
                    break
                progress.check(decision.lateral_m if lateral else decision.forward_m)
                vx, vy, wz = decision.velocity
                self.tick((vx, vy, max(-yaw_cap, min(yaw_cap, wz))))
            self.wait_still()
        if not lateral:  # side steps are local corrections, not route points
            self._record_waypoint()

    def _check_table_walk(self, decision, lateral: bool, direction: float) -> None:
        off_axis = decision.forward_m if lateral else decision.lateral_m
        along = decision.lateral_m if lateral else decision.forward_m
        if abs(off_axis) > self.cfg.careful_corridor_m or abs(decision.heading_error_rad) > math.radians(self.cfg.careful_heading_deg):
            self.robot.stop_base()
            raise SafeHoldError("table_walk_left_corridor", off_axis_m=round(off_axis, 3),
                                heading_deg=round(math.degrees(decision.heading_error_rad), 1))
        if direction * along < -self.cfg.careful_wrong_direction_m:
            self.robot.stop_base()
            raise SafeHoldError("table_walk_wrong_direction", along_m=round(along, 3))

    def turn_by(self, angle_rad: float) -> None:
        """Turn by a measured angle on odometry yaw; stop a braking lead early (gait overshoot)."""
        self._require_turn_allowed()
        lead = math.radians(self.cfg.center_brake_lead_deg)
        if abs(angle_rad) <= lead:
            return
        rate = math.copysign(self.cfg.center_yaw_rate_rad_s, angle_rad)
        self._emit("turn_by", angle_deg=round(math.degrees(angle_rad), 1))
        with self.segment():
            previous = self.pose().yaw
            turned = 0.0
            progress = _Progress(self, math.radians(3), "turn")
            while abs(turned) < abs(angle_rad) - lead:
                current = self.pose().yaw
                turned += wrap_angle(current - previous)
                previous = current
                progress.check(turned)
                self.tick((0.0, 0.0, rate))
            self.wait_still()

    def turn_until(
        self,
        rate_rad_s: float,
        max_angle_rad: float,
        poll: Callable[[], Sighting[T]],
    ) -> Tuple[Optional[T], float]:
        """Turn in place until ``poll`` sees the target on ``detect_frames`` fresh frames in a row.

        Returns (target or None, swept angle).
        """
        self._require_turn_allowed()
        self._emit("turn_until", rate=rate_rad_s, max_deg=round(math.degrees(max_angle_rad), 1))
        with self.segment():
            previous = self.pose().yaw
            swept, hits = 0.0, 0
            progress = _Progress(self, math.radians(3), "turn_until")
            while abs(swept) < max_angle_rad:
                progress.check(swept)
                sighting = poll()
                if sighting.fresh:
                    hits = hits + 1 if sighting.value is not None else 0
                    if hits >= self.cfg.detect_frames:
                        self.wait_still()
                        return sighting.value, swept
                current = self.pose().yaw
                swept += wrap_angle(current - previous)
                previous = current
                self.tick((0.0, 0.0, rate_rad_s))
            self.wait_still()
            return None, swept

    def backstep(self) -> None:
        """Step back from the table; required before any turn after a table approach."""
        self._emit("backstep", distance_m=self.cfg.backstep_m)
        self.walk(-self.cfg.backstep_m, speed_m_s=self.cfg.backstep_speed_m_s)
        self.near_table = False

    def face_point(self, wx: float, wy: float) -> None:
        lx, ly = self.pose().to_local(wx, wy)
        bearing = math.atan2(ly, lx)
        if abs(bearing) > math.radians(self.cfg.center_box_deg):
            self.turn_by(bearing)

    def face_yaw(self, yaw: float) -> None:
        delta = wrap_angle(yaw - self.pose().yaw)
        if abs(delta) > math.radians(self.cfg.center_box_deg):
            self.turn_by(delta)

    def go_to_point(self, wx: float, wy: float, stop_short_m: float = 0.0) -> None:
        """Face the point, walk the forward part, then side-step the lateral remainder."""
        self.face_point(wx, wy)
        lx, ly = self.pose().to_local(wx, wy)
        forward = lx - stop_short_m
        if forward > self.cfg.min_walk_m:
            self.walk(forward)
        lx, ly = self.pose().to_local(wx, wy)
        if abs(ly) > self.cfg.retrace_min_segment_m / 2:
            self.walk(ly, lateral=True)

    def retrace(self) -> None:
        """Walk back through the recorded away-from-table waypoints to the task start."""
        if self.near_table:
            raise NavigationRuleError("retrace requested at the table before BACKSTEP")
        targets = list(reversed(self.waypoints))
        self._emit("retrace", waypoints=[p.as_list() for p in targets])
        for waypoint in targets:
            if self.pose().distance_to(waypoint.x, waypoint.y) < self.cfg.retrace_min_segment_m:
                continue
            self.go_to_point(waypoint.x, waypoint.y)
