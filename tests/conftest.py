"""Offline test bench: simulated room + orchestrator, no hardware, no models, no network."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arkestrator.clock import SimClock  # noqa: E402
from arkestrator.config import SessionConfig  # noqa: E402
from arkestrator.events import EventLog, StateBoard  # noqa: E402
from arkestrator.orchestrator import Orchestrator  # noqa: E402
from arkestrator.profiles import TaughtPose, load_profiles  # noqa: E402
from arkestrator.robot.mock import (  # noqa: E402
    MockObject,
    MockPerson,
    MockPolicy,
    MockRobot,
    MockScenario,
    MockVision,
    MockWorld,
)
from arkestrator.vla.pool import PolicyPool  # noqa: E402
from arkestrator.voice import LogSpeaker  # noqa: E402

TAUGHT = {"bottle": (0.42, -0.18, 0.0), "box": (0.45, 0.0, 0.0)}


def default_objects():
    return [MockObject("bottle_red", "bottle", -1.8, 1.2, 0.0, attributes=("красный",))]


def default_persons():
    return [MockPerson("operator", 1.3, 0.1, embedding=(1, 0, 0, 0, 0, 0, 0, 0))]


def make_system(objects=None, persons=None, scenario=None, taught=TAUGHT, perception=None,
                robot_pose=None, cfg=None, tmp_taught=None):
    clock = SimClock(step_s=0.02)
    world = MockWorld(
        default_objects() if objects is None else objects,
        default_persons() if persons is None else persons,
        scenario or MockScenario(),
        robot_pose,
    )
    robot = MockRobot(world, clock)
    vision = MockVision(robot)
    profiles = load_profiles(ROOT / "profiles")
    for name, profile in list(profiles.items()):
        point = taught.get(name)
        profiles[name] = replace(
            profile,
            vla=replace(profile.vla, checkpoint=f"mock://{name}"),
            taught=TaughtPose(point, "test", "test") if point else None,
        )
    log = EventLog(None, clock)
    board = StateBoard()
    speaker = LogSpeaker()
    pool = PolicyPool(MockPolicy, resident=1, log=log)
    orch = Orchestrator(robot, vision, perception, pool, speaker, profiles, cfg or SessionConfig(),
                        clock, log, board, taught_dir=tmp_taught)
    system = SimpleNamespace(clock=clock, world=world, robot=robot, vision=vision, profiles=profiles,
                             log=log, board=board, speaker=speaker, pool=pool, orch=orch)

    def watchdog(dt):
        if clock.now() > SIM_BUDGET_S:
            raise RuntimeError("simulated time budget exceeded (a wait never ended)")

    clock.subscribe(watchdog)
    return system


SIM_BUDGET_S = 3000.0


def start(system):
    assert system.orch.request_session_start()["ok"]
    system.orch.process_pending()
    assert system.orch.phase.value == "IDLE"


def auto_release(system, after_s: float = 2.0):
    """A person who says «отпусти» ``after_s`` seconds after the robot starts waiting."""
    state = {"since": None}

    def listener(dt):
        orch = system.orch
        if orch.awaiting_release:
            if state["since"] is None:
                state["since"] = system.clock.now()
            elif system.clock.now() - state["since"] >= after_s and not orch.release_flag.is_set():
                orch.release_object()
        else:
            state["since"] = None

    system.clock.subscribe(listener)


def run_command(system, text: str):
    reply = system.orch.submit_text(text)
    system.orch.process_pending()
    return reply


def events(system, kind):
    return [e for e in system.log.tail(limit=10_000) if e["kind"] == kind]


@pytest.fixture
def system():
    s = make_system()
    auto_release(s)
    return s
