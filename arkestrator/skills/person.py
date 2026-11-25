"""The commander: captured at command time, found again on return, approached to a standoff.

Matching order: appearance embedding (cosine, top-1 with margin) → the remembered place
(single person within ``match_radius_m``) → when nobody was seen at command time, the nearest
person in front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from ..types import Detection, Pose2D
from .common import SkillContext, center_on, local_xy, measure, poll_select


@dataclass(frozen=True)
class Commander:
    robot_pose: Pose2D
    world_xy: Optional[tuple[float, float]]
    embedding: Optional[tuple[float, ...]]

    @property
    def seen(self) -> bool:
        return self.world_xy is not None


def _cosine(a, b) -> float:
    if len(a) != len(b):
        return -1.0  # embeddings from different models never match
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else -1.0


def match_person(detections: list[Detection], commander: Commander, cfg) -> Optional[Detection]:
    people = [d for d in detections if d.label in cfg.labels and d.position is not None]
    if not people:
        return None
    if commander.embedding is not None:
        scored = sorted(
            ((_cosine(commander.embedding, d.embedding), d) for d in people if d.embedding is not None),
            key=lambda item: item[0],
            reverse=True,
        )
        if scored:
            best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else -1.0)
            if best >= cfg.reid_threshold and best - second >= cfg.reid_margin:
                return scored[0][1]
    if commander.world_xy is not None:
        near = [
            d for d in people
            if math.hypot(d.world_xy[0] - commander.world_xy[0], d.world_xy[1] - commander.world_xy[1]) <= cfg.match_radius_m
        ]
        return near[0] if len(near) == 1 else None
    in_front = [d for d in people if math.hypot(*d.position[:2]) <= cfg.commander_max_range_m]
    return min(in_front, key=lambda d: math.hypot(*d.position[:2])) if in_front else None


def capture_commander(ctx: SkillContext) -> Commander:
    cfg = ctx.cfg.person
    pose = ctx.nav.pose()

    def central(detections):
        candidates = [
            d for d in detections
            if d.label in cfg.labels
            and d.position is not None
            and math.hypot(*d.position[:2]) <= cfg.commander_max_range_m
            and abs(d.bearing) <= math.radians(cfg.commander_max_bearing_deg)
        ]
        return min(candidates, key=lambda d: abs(d.bearing)) if candidates else None

    seen = measure(ctx, cfg.labels, central, with_embeddings=True)
    if seen is None:
        ctx.log.emit("commander", seen=False, pose=pose.as_list())
        ctx.say("Не вижу вас. Принесу на это место")
        return Commander(pose, None, None)
    ctx.log.emit("commander", seen=True, embedding=seen.embedding is not None, **seen.summary())
    return Commander(pose, seen.world_xy, seen.embedding)


def find_commander(ctx: SkillContext, commander: Commander) -> Optional[Detection]:
    cfg = ctx.cfg.person
    nav = ctx.nav
    if commander.world_xy is not None:
        nav.face_point(*commander.world_xy)
    else:
        nav.face_yaw(commander.robot_pose.yaw)

    def select(detections):
        return match_person(detections, commander, cfg)

    found = measure(ctx, cfg.labels, select, with_embeddings=True)
    if found is not None:
        return found
    seen, _ = nav.turn_until(
        cfg.search_yaw_rate_rad_s,
        math.radians(cfg.search_sweep_deg),
        poll_select(ctx, cfg.labels, select, with_embeddings=True),
    )
    if seen is None:
        return None
    return center_on(ctx, cfg.labels, select, math.copysign(1.0, cfg.search_yaw_rate_rad_s)) or seen


def approach_person(ctx: SkillContext, commander: Commander, person: Detection) -> None:
    cfg = ctx.cfg.person
    nav = ctx.nav
    known = person
    for step in range(cfg.max_approach_steps):
        tracking = replace(commander, world_xy=known.world_xy)
        fresh = measure(ctx, cfg.labels, lambda ds, t=tracking: match_person(ds, t, cfg), with_embeddings=True)
        if fresh is not None:
            known = fresh
        x, y = local_xy(ctx, known)
        distance, bearing = math.hypot(x, y), math.atan2(y, x)
        ctx.log.emit("approach_person", step=step, distance_m=round(distance, 3),
                     bearing_deg=round(math.degrees(bearing), 1), seen=fresh is not None)
        if abs(bearing) > math.radians(ctx.cfg.nav.center_box_deg):
            nav.turn_by(bearing)
            continue
        if distance <= cfg.standoff_m + cfg.standoff_tolerance_m:
            return
        nav.walk(min(cfg.max_step_m, distance - cfg.standoff_m))
