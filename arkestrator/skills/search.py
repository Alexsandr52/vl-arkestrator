"""SEARCH: look ahead, then turn in place until the object is seen, center on it, and (when
asked for a property like «красную») confirm it with the VLM; rejected candidates are skipped."""

from __future__ import annotations

import math

from ..errors import RecoverableError
from ..types import Detection, Task
from .common import SkillContext, center_on, measure, poll_select, select_object


def needs_verification(task: Task, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "attributes":
        return bool(task.attributes)
    return False


def verification_question(task: Task, profile) -> str:
    described = " ".join([*task.attributes, profile.display_ru]).strip()
    return f"В рамке {described}? Ответь да или нет."


def search_object(ctx: SkillContext, task: Task, profile) -> Detection:
    cfg = ctx.cfg.nav
    labels = profile.detector_classes
    rejected: list[tuple[float, float]] = []
    budget = math.radians(cfg.search_sweep_deg)
    direction = math.copysign(1.0, cfg.search_yaw_rate_rad_s)
    ctx.say(f"Ищу {profile.accusative_ru}")

    def select(detections):
        return select_object(labels, exclude=rejected)(detections)

    fix = measure(ctx, labels, select)  # maybe it is already in front
    while True:
        if fix is None:
            if budget < math.radians(5):
                raise RecoverableError("object_not_found", say=f"Не нашёл {profile.accusative_ru}")
            seen, swept = ctx.nav.turn_until(cfg.search_yaw_rate_rad_s, budget, poll_select(ctx, labels, select))
            budget -= abs(swept)
            if seen is None:
                raise RecoverableError("object_not_found", say=f"Не нашёл {profile.accusative_ru}")
            ctx.log.emit("object_seen", **seen.summary())
        fix = center_on(ctx, labels, select, direction)
        if fix is None:
            ctx.log.emit("object_lost_while_centering")
            continue
        if needs_verification(task, ctx.cfg.grasp.verify_target):
            answer = ctx.vision.verify(fix, verification_question(task, profile), task.attributes)
            ctx.log.emit("object_verify", answer=answer, **fix.summary())
            if answer is False:
                rejected.append(fix.world_xy)
                ctx.say("Это не то, ищу дальше")
                fix = None
                continue
        ctx.log.emit("object_fix", **fix.summary())
        return fix
