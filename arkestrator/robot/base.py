"""The robot as the orchestrator sees it.

Two implementations: ``robot.mock.MockRobot`` (offline simulation) and the G1 backend built
on GR00T-pickup-bottle (``SessionPump`` owner, arm_sdk, ``BaseMotion``, ``Frames``), added in
phase P4. The orchestrator never talks to DDS directly.

Contract for implementations:
* ``check()`` raises ``SafeHoldError`` on STOP-class faults (worker/camera failure, overheating,
  mode change, non-finite state, stale odometry, remote stick not neutral). Called every tick.
* ``request_stop()`` is thread-safe (UI thread). The ongoing arm motion, VLA or base command stops
  at the next control tick, the call in progress raises ``StopRequested``, and the backend holds
  the measured pose (owner STOP path → ``hold_before_release``).
* Latch: after ``request_stop()`` or ``enter_safe_hold()`` every arm or base command raises
  ``SafeHoldError`` until the session ends through ``prepare_release`` + ``end_session``.
* Base velocity is accepted only inside ``mobile_segment()``; the arms keep their pose there.
  ``set_base_velocity`` rejects non-finite values. Leaving the segment means the velocity lease
  is closed *and* the base is measured still. Arm poses change only outside segments.
* Arm motions plan their own clearance and keep the existing speed/URDF limits; long motions
  observe ``request_stop()`` while they run.
* ``prepare_release()`` freezes the arms in the release hold and returns a fresh single-use
  token; ``end_session()`` accepts only that token with both operator confirmations.
"""

from __future__ import annotations

import abc
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..clock import Clock
from ..types import Odometry


@dataclass(frozen=True)
class CameraFrame:
    camera: str
    number: int
    stamp: float
    pose: Any  # Pose2D of the base at capture time
    jpeg: Optional[bytes] = None
    depth_m: Any = None  # HxW float32 meters aligned to color, or None
    intrinsics: Optional[dict[str, float]] = None  # fx, fy, ppx, ppy


@dataclass(frozen=True)
class PolicyOutcome:
    """How the VLA phase ended. ``contact`` is measured (tactile/finger stall), not commanded."""

    end_reason: str  # contact | finger_stall | time
    contact: bool
    seconds: float
    steps: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TailOutcome:
    """Scripted tail result. ``holding``: True = pressure/stall on the object, False = fist
    closed without pressure (empty), None = this tail cannot tell (box, napkin)."""

    holding: Optional[bool]
    stop_reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseConfirmation:
    """Operator confirmation for one release hold (``token`` from ``prepare_release``)."""

    token: str
    arm_path_clear: bool
    object_supported_or_absent: bool

    def valid(self) -> bool:
        return bool(self.token) and self.arm_path_clear is True and self.object_supported_or_absent is True


class RobotBackend(abc.ABC):
    clock: Clock

    # --- session -------------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def arms_owned(self) -> bool:
        """True while this process owns the arms (takeover done, not yet released)."""

    @abc.abstractmethod
    def start_session(self) -> None:
        """Preflight, arm takeover, preparation stances, then the ``ready`` pose."""

    @abc.abstractmethod
    def request_stop(self) -> None:
        """Thread-safe STOP for arms and base (see module docstring)."""

    @abc.abstractmethod
    def enter_safe_hold(self, reason: str) -> None:
        """Zero the base and freeze the arms where they are; latches until release."""

    @abc.abstractmethod
    def prepare_release(self) -> str:
        """Enter the release hold (base still, arms frozen) and return a fresh one-time token."""

    @abc.abstractmethod
    def end_session(self, confirmation: ReleaseConfirmation) -> None:
        """Return arms to the factory stand and release them (existing release protocol)."""

    @abc.abstractmethod
    def check(self) -> None:
        """Raise ``SafeHoldError`` on STOP-class faults."""

    @abc.abstractmethod
    def health(self) -> dict[str, Any]:
        """Temperatures, mode, camera ages… for the UI (measured values only)."""

    # --- base ----------------------------------------------------------------------------
    @abc.abstractmethod
    def mobile_segment(self) -> AbstractContextManager:
        """Allow base motion with arms held; on exit the base is stopped and measured still."""

    @abc.abstractmethod
    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None: ...

    @abc.abstractmethod
    def stop_base(self) -> None:
        """Zero base velocity. Thread-safe (the UI STOP button calls it)."""

    @abc.abstractmethod
    def odometry(self) -> Odometry: ...

    # --- cameras -------------------------------------------------------------------------
    @abc.abstractmethod
    def frames(self) -> dict[str, CameraFrame]:
        """Newest frame per camera."""

    # --- arms ----------------------------------------------------------------------------
    @abc.abstractmethod
    def move_arms(self, pose: str) -> None:
        """Move to a named pose (``ready``, a profile start pose, a handover pose)."""

    @abc.abstractmethod
    def open_hands(self, hand: str) -> None: ...

    @abc.abstractmethod
    def prepare_grasp(self, profile) -> None:
        """Open hand, go to the profile start pose, verify it, re-baseline tactile sensors."""

    @abc.abstractmethod
    def retract_after_miss(self, profile) -> None:
        """Small move up and back so the open hand clears the object before the next try."""

    @abc.abstractmethod
    def release_at_table(self, profile) -> None:
        """After the tail but before a positive verdict: bring the hand back down to where the
        object was grasped and open it there (never drop it from carry height)."""

    @abc.abstractmethod
    def run_policy(self, profile, policy, checkpoint: Callable[[], None]) -> PolicyOutcome:
        """Re-arm the model phase and run the VLA until the profile end condition."""

    @abc.abstractmethod
    def run_tail(self, profile) -> TailOutcome: ...

    @abc.abstractmethod
    def carry_check(self, profile) -> None:
        """Raise ``RecoverableError('object_dropped')`` when the carry check fails."""
