"""HANDOVER: extend the arm, say «возьмите», open only on «отпусти» (voice or UI button).

The release request is cleared on entry, so an earlier request is never reused. There is no
timeout: the robot holds the object until the person asks for it (or STOP).
"""

from __future__ import annotations

import threading

from .common import SkillContext


def handover(ctx: SkillContext, profile, release_event: threading.Event, set_waiting) -> None:
    ctx.stage("HANDOVER")
    ctx.robot.move_arms(profile.handover_pose)
    ctx.arms = "handover"
    release_event.clear()
    set_waiting(True)
    try:
        ctx.say("Возьмите, пожалуйста. Скажите «отпусти», когда держите")
        while not release_event.is_set():
            ctx.checkpoint()
            ctx.clock.sleep(0.05)
    finally:
        set_waiting(False)
    release_event.clear()
    ctx.log.emit("handover_release")
    ctx.robot.open_hands(profile.grasp.hand)
    ctx.clock.sleep(ctx.cfg.handover.open_settle_s)
    ctx.robot.move_arms(profile.ready_pose)
    ctx.arms = "ready"
