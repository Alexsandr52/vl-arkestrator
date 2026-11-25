"""Offline simulation of the robot, its camera and the room.

Not a physics model: a kinematic base with a gait dead band (the factory gait ignores
< ~0.2 m/s and < ~0.3 rad/s) and a first-order lag, a head camera with a field of view, objects
on tables and people. Scenario knobs inject grasp misses, drops while carrying, a base stall
and a STOP-class fault, so every FSM path can be exercised without hardware.

The mock enforces the backend contract: base velocity only inside ``mobile_segment``, arm pose
changes only outside it, nothing before ``start_session``.
"""

from __future__ import annotations

import math
import random
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import yaml

from ..clock import SimClock
from ..errors import RecoverableError, SafeHoldError, StopRequested
from ..perception.base import Perception
from ..types import Detection, Odometry, Pose2D, Sighting, wrap_angle
from ..vision import Vision
from ..vla.client import PolicyClient, Prediction
from .base import CameraFrame, PolicyOutcome, ReleaseConfirmation, RobotBackend, TailOutcome

HAND_IN_CARRY = (0.30, -0.15, 0.10)  # robot frame, where a held object appears to the camera


@dataclass
class MockObject:
    name: str
    label: str
    x: float
    y: float
    z: float = 0.0
    attributes: tuple[str, ...] = ()
    state: str = "table"  # table | held | floor | given


@dataclass
class MockPerson:
    name: str
    x: float
    y: float
    embedding: tuple[float, ...] = ()


@dataclass
class MockScenario:
    grasp_results: dict[str, list[bool]] = field(default_factory=dict)
    drop_after_m: Optional[float] = None
    fault_after_s: Optional[float] = None
    stall: bool = False
    reach_tolerance_m: float = 0.10
    yaw_drift_rad_s: float = 0.0  # unexpected yaw while walking (e.g. a slip at the table)
    slip_after_tail: bool = False  # tail reports pressure but the object stays on the table
    vlm_denies: int = 0  # the VLM wrongly answers «not holding» this many times

    def next_grasp(self, profile: str) -> bool:
        queue = self.grasp_results.get(profile)
        return queue.pop(0) if queue else True


class MockWorld:
    FOV_RAD = math.radians(35)
    MIN_RANGE_M = 0.2
    MAX_RANGE_M = 5.0
    LAG_S = 0.25

    def __init__(self, objects: Sequence[MockObject], persons: Sequence[MockPerson],
                 scenario: Optional[MockScenario] = None, robot_pose: Optional[Pose2D] = None,
                 seed: int = 0, noise_m: float = 0.005):
        self.objects = list(objects)
        self.persons = list(persons)
        self.scenario = scenario or MockScenario()
        self.pose = robot_pose or Pose2D(0.0, 0.0, 0.0)
        self.command = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.held: Optional[MockObject] = None
        self.walked_holding_m = 0.0
        self.rng = random.Random(seed)
        self.noise_m = noise_m
        self.lock = threading.RLock()

    def step(self, dt: float) -> None:
        with self.lock:
            vx, vy, wz = self.command
            linear = math.hypot(vx, vy)
            if linear < 0.15:
                vx = vy = 0.0
                if abs(wz) < 0.25:
                    wz = 0.0
            if self.scenario.stall:
                vx = vy = wz = 0.0
            a = min(1.0, dt / self.LAG_S)
            cvx, cvy, cwz = self.velocity
            cvx += (vx - cvx) * a
            cvy += (vy - cvy) * a
            cwz += (wz - cwz) * a
            if abs(cvx) < 1e-4 and vx == 0.0:
                cvx = 0.0
            if abs(cvy) < 1e-4 and vy == 0.0:
                cvy = 0.0
            if abs(cwz) < 1e-4 and wz == 0.0:
                cwz = 0.0
            self.velocity = (cvx, cvy, cwz)
            c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
            self.pose = Pose2D(
                self.pose.x + (c * cvx - s * cvy) * dt,
                self.pose.y + (s * cvx + c * cvy) * dt,
                wrap_angle(self.pose.yaw + (cwz + (self.scenario.yaw_drift_rad_s if (cvx or cvy) else 0.0)) * dt),
            )
            if self.held is not None:
                self.walked_holding_m += math.hypot(cvx, cvy) * dt

    def held_world(self) -> tuple[float, float]:
        return self.pose.to_world(HAND_IN_CARRY[0], HAND_IN_CARRY[1])

    def visible(self, labels: Sequence[str]):
        """(kind, item, robot-frame position) for everything in the camera view."""
        with self.lock:
            pose = self.pose
            seen = []
            for obj in self.objects:
                if obj.label not in labels or obj.state in ("given",):
                    continue
                if obj.state == "held":
                    seen.append(("object", obj, HAND_IN_CARRY))
                    continue
                z = obj.z if obj.state == "table" else obj.z - 0.75
                seen.append(("object", obj, (*pose.to_local(obj.x, obj.y), z)))
            if "person" in labels:
                for person in self.persons:
                    seen.append(("person", person, (*pose.to_local(person.x, person.y), 0.0)))
            visible = []
            for kind, item, (lx, ly, lz) in seen:
                distance = math.hypot(lx, ly)
                if kind == "object" and item.state == "held":
                    visible.append((kind, item, (lx, ly, lz)))
                elif self.MIN_RANGE_M <= distance <= self.MAX_RANGE_M and abs(math.atan2(ly, lx)) <= self.FOV_RAD:
                    n = self.noise_m
                    visible.append((kind, item, (lx + self.rng.gauss(0, n), ly + self.rng.gauss(0, n), lz)))
            return visible

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "objects": [
                    {"name": o.name, "label": o.label, "state": o.state,
                     "xy": list(self.held_world()) if o.state == "held" else [o.x, o.y]}
                    for o in self.objects
                ],
                "persons": [{"name": p.name, "xy": [p.x, p.y]} for p in self.persons],
            }


class MockRobot(RobotBackend):
    CAMERA_FPS = 15.0

    def __init__(self, world: MockWorld, clock: SimClock):
        self.world = world
        self.clock = clock
        self._arms_owned = False
        self.arm_pose = "factory"
        self.in_segment = False
        self.frozen = False
        self.frame_number = 0
        self._frame_accumulator = 0.0
        self._session_started: Optional[float] = None
        self._last_policy: Optional[tuple[str, bool, Optional[MockObject]]] = None
        self._stop_requested = threading.Event()
        self._release_token: Optional[str] = None
        self._grasped_from: Optional[tuple[float, float]] = None
        self.calls: list[str] = []
        clock.subscribe(self._step)

    @property
    def arms_owned(self) -> bool:
        return self._arms_owned

    def _step(self, dt: float) -> None:
        if self.frozen:
            self.world.command = (0.0, 0.0, 0.0)
        self.world.step(dt)
        self._frame_accumulator += dt
        period = 1.0 / self.CAMERA_FPS
        while self._frame_accumulator >= period:
            self._frame_accumulator -= period
            self.frame_number += 1

    def _require_motion_allowed(self) -> None:
        if not self._arms_owned:
            raise RuntimeError("mock: no session (arms not taken over)")
        if self._stop_requested.is_set():
            raise StopRequested()
        if self.frozen:
            raise SafeHoldError("safe_hold_latched")

    def _sleep_observing_stop(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 1e-9:
            if self._stop_requested.is_set():
                raise StopRequested()
            step = min(0.05, remaining)
            self.clock.sleep(step)
            remaining -= step

    def _arm_motion(self, pose: str, seconds: float) -> None:
        self._require_motion_allowed()
        if self.in_segment:
            raise RuntimeError("mock: arm pose change inside a mobile segment")
        self.calls.append(f"arms:{pose}")
        self.arm_pose = pose
        self._sleep_observing_stop(seconds)

    # --- session
    def start_session(self) -> None:
        self.calls.append("start_session")
        self._stop_requested.clear()
        self._arms_owned = True
        self.frozen = False
        self._release_token = None
        self._session_started = self.clock.now()
        self.arm_pose = "ready"
        self._sleep_observing_stop(2.0)

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.stop_base()

    def enter_safe_hold(self, reason: str) -> None:
        self.calls.append(f"safe_hold:{reason}")
        self.stop_base()
        self.frozen = True

    def prepare_release(self) -> str:
        self.calls.append("prepare_release")
        self.stop_base()
        self.frozen = True
        self._release_token = uuid.uuid4().hex
        return self._release_token

    def end_session(self, confirmation: ReleaseConfirmation) -> None:
        if not confirmation.valid():
            raise ValueError("release needs both confirmations")
        if self._release_token is None or confirmation.token != self._release_token:
            raise SafeHoldError("release_token_invalid")
        self._release_token = None  # single use
        self.calls.append("end_session")
        self.stop_base()
        self.arm_pose = "factory"
        self._arms_owned = False
        self.frozen = False
        self._stop_requested.clear()
        self._session_started = None

    def check(self) -> None:
        fault = self.world.scenario.fault_after_s
        if (
            fault is not None
            and self._session_started is not None
            and not self.frozen
            and self.clock.now() - self._session_started > fault
        ):
            raise SafeHoldError("overheated_joints", joints=["right_shoulder_pitch_joint"])

    def health(self) -> dict[str, Any]:
        return {"backend": "mock", "arm_pose": self.arm_pose, "frozen": self.frozen, "world": self.world.snapshot()}

    # --- base
    @contextmanager
    def mobile_segment(self):
        self._require_motion_allowed()
        if self.in_segment:
            raise RuntimeError("mock: nested mobile segment")
        self.in_segment = True
        try:
            yield
        finally:
            self.stop_base()
            try:
                self._wait_base_still()
            finally:
                self.in_segment = False

    def _wait_base_still(self) -> None:
        while True:
            with self.world.lock:
                vx, vy, wz = self.world.velocity
            if math.hypot(vx, vy) < 0.01 and abs(wz) < 0.01:
                return
            self.clock.sleep(0.02)

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        if not self.in_segment:
            raise RuntimeError("mock: base velocity outside a mobile segment")
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            raise SafeHoldError("non_finite_base_command")
        self._require_motion_allowed()
        self.world.command = (vx, vy, wz)

    def stop_base(self) -> None:
        self.world.command = (0.0, 0.0, 0.0)

    def odometry(self) -> Odometry:
        with self.world.lock:
            vx, vy, wz = self.world.velocity
            return Odometry(self.world.pose, math.hypot(vx, vy), wz, self.clock.now())

    def frames(self) -> dict[str, CameraFrame]:
        return {"head": CameraFrame("head", self.frame_number, self.clock.now(), self.world.pose)}

    # --- arms
    def move_arms(self, pose: str) -> None:
        self._arm_motion(pose, 1.0)

    def open_hands(self, hand: str) -> None:
        self._arm_motion(f"{self.arm_pose}+open", 0.5)
        held = self.world.held
        if held is not None:
            with self.world.lock:
                if "handover" in self.arm_pose:
                    held.state = "given"
                else:
                    held.x, held.y = self.world.held_world()
                    held.state = "floor"  # opened away from the table: it falls
                self.world.held = None

    def release_at_table(self, profile) -> None:
        self._arm_motion("lowered_to_table", 2.0)
        held = self.world.held
        with self.world.lock:
            if held is not None:
                held.x, held.y = self._grasped_from or self.world.held_world()
                held.state = "table"
                self.world.held = None
        self._arm_motion("lowered_to_table+open", 0.5)

    def prepare_grasp(self, profile) -> None:
        self._arm_motion(profile.grasp.start_pose, 1.5)

    def retract_after_miss(self, profile) -> None:
        self._arm_motion("retracted", 0.5)

    def _object_at_reach(self, profile) -> Optional[MockObject]:
        tx, ty, _ = profile.taught.object_in_robot_m
        best, best_d = None, None
        for obj in self.world.objects:
            if obj.state != "table" or obj.label not in profile.detector_classes:
                continue
            lx, ly = self.world.pose.to_local(obj.x, obj.y)
            d = math.hypot(lx - tx, ly - ty)
            if best_d is None or d < best_d:
                best, best_d = obj, d
        if best is None or best_d > self.world.scenario.reach_tolerance_m:
            return None
        return best

    def run_policy(self, profile, policy, checkpoint) -> PolicyOutcome:
        self._require_motion_allowed()
        if self.in_segment:
            raise RuntimeError("mock: VLA inside a mobile segment")
        self.calls.append(f"policy:{profile.name}")
        target = self._object_at_reach(profile)
        success = target is not None and self.world.scenario.next_grasp(profile.name)
        duration = 4.0 if success else profile.grasp.max_policy_seconds
        started, steps = self.clock.now(), 0
        while self.clock.now() - started < duration:
            checkpoint()
            self._sleep_observing_stop(1.0 / 15.0)
            steps += 1
        self._last_policy = (profile.name, success, target if success else None)
        contact = bool(success)
        return PolicyOutcome(
            end_reason="contact" if contact else "time",
            contact=contact,
            seconds=round(self.clock.now() - started, 2),
            steps=steps,
            details={"object_at_reach": target is not None},
        )

    def run_tail(self, profile) -> TailOutcome:
        self._arm_motion("carry", 3.0)
        name, success, target = self._last_policy or (None, False, None)
        bottle_like = profile.grasp.tail == "bottle_squeeze_lift_carry"
        slips = self.world.scenario.slip_after_tail
        if success and target is not None:
            with self.world.lock:
                self._grasped_from = (target.x, target.y)
                if slips:
                    self.world.scenario.slip_after_tail = False
                    return TailOutcome(True if bottle_like else None, "pressure_rise_target_reached" if bottle_like else "done")
                target.state = "held"
                self.world.held = target
                self.world.walked_holding_m = 0.0
            return TailOutcome(True if bottle_like else None, "pressure_rise_target_reached" if bottle_like else "done")
        return TailOutcome(False if bottle_like else None, "fist_reached_without_pressure" if bottle_like else "done")

    def carry_check(self, profile) -> None:
        limit = self.world.scenario.drop_after_m
        held = self.world.held
        if held is not None and limit is not None and self.world.walked_holding_m > limit:
            with self.world.lock:
                held.x, held.y = self.world.held_world()
                held.state = "floor"
                self.world.held = None
            raise RecoverableError("object_dropped", say=f"Уронил {profile.accusative_ru}")


class MockVision(Vision):
    def __init__(self, robot: MockRobot, camera: str = "head"):
        self.robot = robot
        self.camera = camera
        self._last = -1

    def poll(self, labels: Sequence[str], with_embeddings: bool = False) -> Sighting[list[Detection]]:
        frame = self.robot.frames()[self.camera]
        if frame.number == self._last:
            return Sighting(False)
        self._last = frame.number
        detections = []
        for kind, item, position in self.robot.world.visible(labels):
            detections.append(
                Detection(
                    label=item.label if kind == "object" else "person",
                    score=0.9,
                    camera=self.camera,
                    frame_number=frame.number,
                    stamp=frame.stamp,
                    capture_pose=frame.pose,
                    position=position,
                    embedding=(item.embedding or None) if (kind == "person" and with_embeddings) else None,
                    attributes=item.attributes if kind == "object" else (),
                )
            )
        return Sighting(True, detections)

    def verify(self, detection: Detection, question: str, attributes: Sequence[str] = ()) -> Optional[bool]:
        return all(a in detection.attributes for a in attributes)

    def ask(self, question: str) -> Optional[bool]:
        scenario = self.robot.world.scenario
        if scenario.vlm_denies > 0:
            scenario.vlm_denies -= 1
            return False
        return self.robot.world.held is not None


class MockPolicy(PolicyClient):
    def __init__(self, profile):
        self.profile_name = profile.name
        self.execute_steps = profile.vla.execute_steps
        self.horizon = profile.vla.horizon

    def predict(self, jpeg: bytes, state, frame_stamp=None) -> Prediction:
        import numpy as np

        chunk = np.zeros((self.horizon, 28), dtype=np.float32)
        return Prediction(chunk[: self.execute_steps], chunk, self.execute_steps, 0.0, frame_stamp, {"mock": True})

    def close(self) -> None:
        pass


class FakePerception(Perception):
    """Deterministic stand-in for tests and the mock daemon (no models)."""

    def __init__(self, transcripts: Optional[list[str]] = None, parse=None):
        self.transcripts = list(transcripts or [])
        self._parse = parse

    def transcribe(self, wav: bytes) -> str:
        if not self.transcripts:
            raise RuntimeError("no ASR in mock mode")
        return self.transcripts.pop(0)

    def parse_command(self, text, catalog):
        if self._parse is None:
            raise RuntimeError("no VLM in mock mode")
        return self._parse(text, catalog)

    def detect(self, jpeg, labels):
        return []

    def verify(self, jpeg, bbox, question):
        return None

    def embed_person(self, jpeg, bbox):
        return None

    def health(self):
        return {"backend": "fake"}


def load_mock_world(path, seed: int = 0) -> tuple[MockWorld, dict[str, tuple[float, float, float]]]:
    """World + mock taught poses from ``config/mock_world.example.yaml``-style files."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    objects = [
        MockObject(o["name"], o["label"], *map(float, o["position"]), attributes=tuple(o.get("attributes", ())))
        for o in data.get("objects", [])
    ]
    persons = [
        MockPerson(p["name"], *map(float, p["position"]), embedding=tuple(float(v) for v in p.get("embedding", ())))
        for p in data.get("persons", [])
    ]
    scenario = MockScenario(**(data.get("scenario") or {}))
    world = MockWorld(objects, persons, scenario, Pose2D(*map(float, data.get("robot", [0, 0, 0]))), seed=seed)
    taught = {name: tuple(float(v) for v in point) for name, point in (data.get("taught") or {}).items()}
    return world, taught
