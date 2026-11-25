"""GRASP: prepare → VLA → scripted tail → verify; on a miss open, retract, re-locate, retry.

Verification per profile (``grasp.verify``):
* tactile: the VLA ended on measured contact and the tail reports pressure/stall on the object
  (a closed fist without pressure means empty);
* visual: on most of ``visual_votes`` fresh frames the object is no longer where it stood on the
  table (same xy and height), and the VLM does not deny that the robot holds it.
After ``max_attempts`` misses the task ends as a recoverable failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..errors import RecoverableError
from ..types import Detection
from .common import SkillContext, local_xy, measure, select_object


@dataclass(frozen=True)
class GraspVerdict:
    success: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspResult:
    attempts: int
    verdict: GraspVerdict
    fix: Detection


def object_left_table(ctx: SkillContext, profile, fix: Detection) -> tuple[bool, dict[str, Any]]:
    spec = profile.grasp.verify
    radius = ctx.cfg.grasp.visual_check_radius_m
    labels = profile.detector_classes
    fx, fy = fix.world_xy
    fz = fix.position[2]
    votes: list[bool] = []
    deadline = ctx.clock.now() + ctx.cfg.nav.measure_timeout_s * spec.visual_votes
    while len(votes) < spec.visual_votes and ctx.clock.now() < deadline:
        sighting = ctx.vision.poll(labels)
        if sighting.fresh:
            still_there = any(
                d.label in labels
                and d.position is not None
                and math.hypot(d.world_xy[0] - fx, d.world_xy[1] - fy) <= radius
                and abs(d.position[2] - fz) <= radius / 3
                for d in sighting.value or []
            )
            votes.append(not still_there)
        ctx.nav.tick()
    left = sum(votes)
    holding_answer = ctx.vision.ask(f"Робот держит {profile.accusative_ru} в руке? Ответь да или нет.")
    ok = left >= spec.visual_needed and holding_answer is not False
    return ok, {"votes_left_table": left, "votes": len(votes), "vlm_holding": holding_answer}


def verify_grasp(ctx: SkillContext, profile, outcome, tail, fix: Detection) -> GraspVerdict:
    spec = profile.grasp.verify
    evidence: dict[str, Any] = {
        "end_reason": outcome.end_reason,
        "contact": outcome.contact,
        "tail": None if tail is None else tail.stop_reason,
        "tail_holding": None if tail is None else tail.holding,
    }
    if spec.tactile == "required":
        tactile_ok = bool(outcome.contact) and tail is not None and tail.holding is True
        evidence["tactile_ok"] = tactile_ok
        if not tactile_ok:
            return GraspVerdict(False, evidence)
    if spec.visual == "required":
        visual_ok, details = object_left_table(ctx, profile, fix)
        evidence.update(details, visual_ok=visual_ok)
        if not visual_ok:
            return GraspVerdict(False, evidence)
    return GraspVerdict(True, evidence)


def relocate(ctx: SkillContext, profile, fix: Detection) -> Detection:
    """After a miss: re-measure the object and side/forward-step if it moved (arms at ready)."""
    labels = profile.detector_classes
    fresh = measure(ctx, labels, select_object(labels, near=fix.world_xy, radius=0.3))
    if fresh is None:
        ctx.log.emit("relocate", seen=False)
        return fix
    target_x, target_y, _ = profile.taught.object_in_robot_m
    x, y = local_xy(ctx, fresh)
    dx, dy = x - target_x, y - target_y
    tol = max(ctx.cfg.grasp.relocate_tolerance_m, ctx.cfg.nav.final_min_m)
    ctx.log.emit("relocate", seen=True, error_m=[round(dx, 3), round(dy, 3)])
    if abs(dx) <= tol and abs(dy) <= tol:
        return fresh
    ctx.robot.move_arms(profile.ready_pose)
    ctx.arms = "ready"
    if abs(dy) > tol:
        ctx.nav.walk(dy, lateral=True)
    if abs(dx) > tol:
        ctx.nav.walk(dx)
    again = measure(ctx, labels, select_object(labels, near=fresh.world_xy, radius=0.3))
    return again or fresh


def grasp_object(ctx: SkillContext, profile, fix: Detection) -> GraspResult:
    max_attempts = profile.grasp.max_attempts
    policy = ctx.vla.acquire(profile, wait=ctx.wait)
    try:
        for attempt in range(1, max_attempts + 1):
            ctx.stage("GRASP", attempt=attempt, max_attempts=max_attempts)
            ctx.arms = "grasp"  # set before moving: recovery must treat the hand as at the object
            ctx.robot.prepare_grasp(profile)
            ctx.checkpoint()
            outcome = ctx.robot.run_policy(profile, policy, ctx.checkpoint)
            tail = None
            if outcome.contact or profile.grasp.end == "contact_or_time":
                ctx.arms = "carry_unverified"
                tail = ctx.robot.run_tail(profile)
            verdict = verify_grasp(ctx, profile, outcome, tail, fix)
            ctx.log.emit("grasp_attempt", attempt=attempt, success=verdict.success, **verdict.evidence)
            if verdict.success:
                ctx.arms = "carry"
                return GraspResult(attempt, verdict, fix)
            if ctx.arms == "carry_unverified":
                # Never open at carry height: lower back to the grasp spot first.
                ctx.robot.release_at_table(profile)
                ctx.arms = "grasp"
            else:
                ctx.robot.open_hands(profile.grasp.hand)
            ctx.robot.retract_after_miss(profile)
            ctx.arms = "grasp_retracted"
            if attempt < max_attempts:
                ctx.say("Не получилось, пробую ещё раз")
                fix = relocate(ctx, profile, fix)
        ctx.robot.move_arms(profile.ready_pose)
        ctx.arms = "ready"
        raise RecoverableError("grasp_failed", say=f"Не смог взять {profile.accusative_ru}", attempts=max_attempts)
    finally:
        ctx.vla.release(profile)
