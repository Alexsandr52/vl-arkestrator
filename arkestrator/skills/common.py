"""What every skill needs, plus stationary measurement and target selection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

from ..config import SessionConfig
from ..types import Detection, Sighting


@dataclass
class SkillContext:
    robot: Any
    vision: Any
    nav: Any
    vla: Any
    clock: Any
    cfg: SessionConfig
    log: Any
    speaker: Any
    checkpoint: Callable[[], None]
    stage: Callable[..., None]
    # ready | grasp | carry | handover — commanded arm pose family, for recovery decisions.
    arms: str = "ready"
    notes: dict[str, Any] = field(default_factory=dict)

    def say(self, text: str) -> None:
        self.speaker.say(text)
        self.log.emit("say", text=text)

    def wait(self, seconds: float = 0.05) -> None:
        self.checkpoint()
        self.clock.sleep(seconds)


Selector = Callable[[list[Detection]], Optional[Detection]]


def poll_select(ctx: SkillContext, labels: Sequence[str], select: Selector, with_embeddings: bool = False):
    """Adapter for ``Navigator.turn_until``: newest frame → selected detection or None."""

    def poll() -> Sighting[Detection]:
        sighting = ctx.vision.poll(labels, with_embeddings)
        if not sighting.fresh:
            return Sighting(False)
        return Sighting(True, select(sighting.value or []))

    return poll


def fuse(found: list[Detection]) -> Detection:
    """Median world position over a few stationary frames, expressed at the latest capture pose."""
    latest = found[-1]
    points = [d.world_xy for d in found if d.world_xy is not None]
    if not points:
        return latest
    wx = statistics.median(p[0] for p in points)
    wy = statistics.median(p[1] for p in points)
    z = statistics.median(d.position[2] for d in found if d.position is not None)
    lx, ly = latest.capture_pose.to_local(wx, wy)
    embedding = next((d.embedding for d in reversed(found) if d.embedding is not None), None)
    return replace(latest, position=(lx, ly, z), embedding=embedding)


def measure(
    ctx: SkillContext,
    labels: Sequence[str],
    select: Selector,
    frames_needed: Optional[int] = None,
    timeout_s: Optional[float] = None,
    with_embeddings: bool = False,
) -> Optional[Detection]:
    """Stationary measurement: fuse ``frames_needed`` fresh hits, or None by the deadline."""
    frames_needed = frames_needed or ctx.cfg.nav.measure_frames
    deadline = ctx.clock.now() + (timeout_s if timeout_s is not None else ctx.cfg.nav.measure_timeout_s)
    found: list[Detection] = []
    while ctx.clock.now() < deadline and len(found) < frames_needed:
        sighting = ctx.vision.poll(labels, with_embeddings)
        if sighting.fresh:
            chosen = select(sighting.value or [])
            if chosen is not None:
                found.append(chosen)
        ctx.nav.tick()
    return fuse(found) if found else None


def local_xy(ctx: SkillContext, detection: Detection) -> tuple[float, float]:
    """Where a detection is now, in the current robot frame (carried by odometry)."""
    return ctx.nav.pose().to_local(*detection.world_xy)


def select_object(
    labels: Sequence[str],
    near: Optional[tuple[float, float]] = None,
    radius: float = 0.25,
    exclude: Sequence[tuple[float, float]] = (),
    exclude_radius: float = 0.3,
) -> Selector:
    labels = tuple(labels)

    def select(detections: list[Detection]) -> Optional[Detection]:
        candidates = []
        for d in detections:
            if d.label not in labels or d.position is None:
                continue
            wx, wy = d.world_xy
            if any(math.hypot(wx - ex, wy - ey) < exclude_radius for ex, ey in exclude):
                continue
            if near is not None and math.hypot(wx - near[0], wy - near[1]) > radius:
                continue
            candidates.append(d)
        if not candidates:
            return None
        if near is not None:
            return min(candidates, key=lambda d: math.hypot(d.world_xy[0] - near[0], d.world_xy[1] - near[1]))
        return max(candidates, key=lambda d: d.score)

    return select


def center_on(ctx: SkillContext, labels: Sequence[str], select: Selector, last_direction: float) -> Optional[Detection]:
    """Dead-band centering (port of disposal ``center_tag``): turn by the measured bearing until
    the target is within ±center_box_deg; recover by turning back if it drops out of view."""
    cfg = ctx.cfg.nav
    for _ in range(cfg.center_attempts):
        observation = measure(ctx, labels, select)
        if observation is None:
            ctx.log.emit("center_lost")
            seen, _ = ctx.nav.turn_until(
                last_direction * cfg.center_yaw_rate_rad_s,
                math.radians(cfg.recover_turn_deg),
                poll_select(ctx, labels, select),
            )
            if seen is None:
                return None
            continue
        bearing = observation.bearing
        if abs(bearing) <= math.radians(cfg.center_box_deg):
            return observation
        last_direction = math.copysign(1.0, bearing)
        ctx.nav.turn_by(bearing)
    # Attempts used up but the target may be in view: the approach side-steps fix the rest.
    return measure(ctx, labels, select)
