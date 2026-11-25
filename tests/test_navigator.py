import hashlib
import math
from pathlib import Path

import pytest

from arkestrator.clock import SimClock
from arkestrator.config import NavConfig
from arkestrator.errors import NavigationRuleError, RecoverableError
from arkestrator.nav.navigator import Navigator
from arkestrator.nav.obstacles import Corridor, ObstacleSource
from arkestrator.robot.mock import MockRobot, MockScenario, MockWorld
from arkestrator.types import Pose2D

PIN = "4b6aea72aa66d7bb87ecea2c147c5369003377f0ef03fade5c75b56a10d97e61"


def make_nav(scenario=None, obstacles=None, cfg=None):
    clock = SimClock(step_s=0.02)
    robot = MockRobot(MockWorld([], [], scenario or MockScenario(), Pose2D(0, 0, 0)), clock)
    robot.start_session()
    nav = Navigator(robot, clock, cfg or NavConfig(), checkpoint=lambda: None, obstacles=obstacles)
    nav.begin_task()
    return nav, robot


def test_vendored_walk_control_matches_pin():
    path = Path(__file__).resolve().parents[1] / "arkestrator" / "nav" / "walk_control.py"
    lines = path.read_text().splitlines(keepends=True)
    body = "".join(lines[5:])  # 4-line pin header + blank line
    assert hashlib.sha256(body.encode()).hexdigest() == PIN


def test_walk_reaches_distance_and_stops():
    nav, robot = make_nav()
    nav.walk(1.0)
    pose = nav.pose()
    assert pose.x == pytest.approx(1.0, abs=0.06)
    assert abs(pose.y) < 0.05
    assert robot.odometry().speed_m_s < 0.02


def test_lateral_walk():
    nav, _ = make_nav()
    nav.walk(-0.3, lateral=True)
    assert nav.pose().y == pytest.approx(-0.3, abs=0.05)


def test_turn_by_with_brake_lead():
    nav, _ = make_nav()
    nav.turn_by(math.radians(90))
    assert math.degrees(nav.pose().yaw) == pytest.approx(90, abs=6)


def test_no_turn_at_table_until_backstep():
    nav, _ = make_nav()
    nav.walk(0.5)
    nav.near_table = True
    with pytest.raises(NavigationRuleError):
        nav.turn_by(math.radians(45))
    with pytest.raises(NavigationRuleError):
        nav.retrace()
    nav.backstep()
    assert not nav.near_table
    assert nav.pose().x == pytest.approx(0.2, abs=0.06)
    nav.turn_by(math.radians(45))


def test_velocity_outside_segment_is_a_bug():
    nav, _ = make_nav()
    with pytest.raises(NavigationRuleError):
        nav.tick((0.2, 0.0, 0.0))


def test_stalled_base_is_recoverable():
    nav, _ = make_nav(MockScenario(stall=True))
    with pytest.raises(RecoverableError) as err:
        nav.walk(1.0)
    assert err.value.reason == "base_stalled"


def test_retrace_skips_table_waypoints_and_returns_home():
    nav, _ = make_nav()
    nav.walk(1.2)  # to the standoff, recorded
    nav.near_table = True
    nav.walk(0.4)  # final approach, not recorded
    nav.backstep()
    nav.turn_by(math.radians(180))
    nav.retrace()
    home = nav.pose()
    assert math.hypot(home.x, home.y) < 0.2
    assert all(w.x < 1.3 for w in nav.waypoints)


class Blocked(ObstacleSource):
    def check_corridor(self, pose, distance_m, lateral):
        return Corridor.BLOCKED


def test_blocked_corridor_refuses_walk():
    nav, _ = make_nav(obstacles=Blocked())
    with pytest.raises(RecoverableError) as err:
        nav.walk(1.0)
    assert err.value.reason == "path_blocked"


def test_unknown_corridor_refused_only_when_required():
    from dataclasses import replace

    nav, _ = make_nav(cfg=replace(NavConfig(), require_clear_corridor=True))
    with pytest.raises(RecoverableError) as err:
        nav.walk(1.0)
    assert err.value.reason == "corridor_unknown"
