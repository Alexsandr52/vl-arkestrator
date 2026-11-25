"""APPROACH_TABLE: bring the object to the taught grasp position in the robot frame.

Port of the disposal approach (standoff → one coarse near turn → side step → final approach →
final side step). Up close the gait shuffles the feet on every turn, so only one coarse turn is
made at the standoff; the rest is corrected by straight and sideways steps. The object position
is re-measured at each stop and otherwise carried by odometry.
"""

from __future__ import annotations

import math

from ..types import Detection
from .common import SkillContext, local_xy, measure, select_object


def approach_table(ctx: SkillContext, profile, fix: Detection) -> tuple[Detection, tuple[float, float]]:
    cfg = ctx.cfg.nav
    nav = ctx.nav
    labels = profile.detector_classes
    target_x, target_y, _ = profile.taught.object_in_robot_m
    known = fix

    def locate() -> tuple[float, float, bool]:
        nonlocal known
        fresh = measure(ctx, labels, select_object(labels, near=known.world_xy, radius=0.25))
        if fresh is not None:
            known = fresh
        x, y = local_xy(ctx, known)
        return x, y, fresh is not None

    ctx.stage("APPROACH_TABLE", step="standoff")
    x, y, seen = locate()
    nav.walk(x - cfg.standoff_m)

    x, y, seen = locate()
    bearing = math.atan2(y, x)
    ctx.log.emit("at_standoff", x=round(x, 3), y=round(y, 3), seen=seen, bearing_deg=round(math.degrees(bearing), 1))
    if abs(bearing) > math.radians(cfg.near_max_bearing_deg):
        nav.turn_by(bearing)
        x, y, seen = locate()

    if abs(y - target_y) > cfg.sidestep_min_m:
        ctx.stage("APPROACH_TABLE", step="sidestep")
        nav.walk(y - target_y, lateral=True)

    # From here the robot is close to the table: no turns until BACKSTEP.
    nav.near_table = True
    ctx.stage("APPROACH_TABLE", step="final")
    x, y, seen = locate()
    if abs(x - target_x) > cfg.final_min_m:
        nav.walk(x - target_x)
    x, y, seen = locate()
    if abs(y - target_y) > cfg.sidestep_min_m:
        nav.walk(y - target_y, lateral=True)
        x, y, seen = locate()

    error = (x - target_x, y - target_y)
    ctx.log.emit("at_table", x=round(x, 3), y=round(y, 3), seen=seen,
                 error_m=[round(error[0], 3), round(error[1], 3)])
    return known, error
