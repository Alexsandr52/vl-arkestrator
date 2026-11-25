"""Wiring: backend + vision + perception + VLA pool + orchestrator + web panel.

``--mock`` runs the whole system offline (simulated room, no models). The G1 backend is added in
phase P4: it needs the pinned GR00T-pickup-bottle checkout with the opt-in model re-arm.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .clock import SimClock
from .config import ROOT, load_local, load_session_config
from .events import EventLog, StateBoard
from .orchestrator import Orchestrator
from .profiles import TaughtPose, load_profiles
from .ui.server import UiServer
from .vla.pool import PolicyPool
from .voice import LogSpeaker


def build_mock(args, cfg, session_dir: Path):
    from .robot.mock import MockPolicy, MockRobot, MockVision, load_mock_world

    clock = SimClock(step_s=cfg.nav.tick_s, realtime_factor=args.realtime)
    world, taught = load_mock_world(args.world, seed=args.seed)
    robot = MockRobot(world, clock)
    vision = MockVision(robot)
    profiles = load_profiles(cfg.path(cfg.profiles_dir), taught_dir=None, local=load_local())
    for name, profile in list(profiles.items()):
        point = taught.get(name)
        profiles[name] = replace(
            profile,
            vla=replace(profile.vla, checkpoint=profile.vla.checkpoint or f"mock://{name}"),
            taught=TaughtPose(point, "mock", "mock_world") if point else None,
        )
    log = EventLog(session_dir, clock)
    pool = PolicyPool(MockPolicy, resident=cfg.vla_resident, log=log)
    return clock, robot, vision, None, profiles, log, pool, session_dir / "taught"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arkestrator", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="offline simulation, no hardware, no models")
    mode.add_argument("--g1", action="store_true", help="real robot on g1-gpu (phase P4)")
    parser.add_argument("--world", default=str(ROOT / "config" / "mock_world.example.yaml"))
    parser.add_argument("--realtime", type=float, default=1.0, help="mock: 1 = real time, 0 = as fast as possible")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=None, help="session yaml (default config/session.yaml)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_session_config(Path(args.config) if args.config else None)
    if args.g1:
        print(
            "G1 backend is not available yet (phase P4). It needs the pinned GR00T-pickup-bottle checkout\n"
            "with the opt-in model re-arm and a fresh user authorization for the physical session.",
            file=sys.stderr,
        )
        return 2

    session_dir = cfg.path(cfg.outputs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    clock, robot, vision, perception, profiles, log, pool, taught_dir = build_mock(args, cfg, session_dir)
    board = StateBoard()
    speaker = LogSpeaker()
    orchestrator = Orchestrator(
        robot, vision, perception, pool, speaker, profiles, cfg, clock, log, board, taught_dir=taught_dir
    )
    ui = UiServer(orchestrator, board, log, robot, host=cfg.ui.host, port=args.port or cfg.ui.port)
    orchestrator.start()
    ui.start()
    log.emit("daemon_started", mode="mock", url=ui.url, outputs=str(session_dir))
    print(f"Панель: {ui.url}  (события: {session_dir / 'events.jsonl'})", flush=True)
    if not args.no_browser:
        webbrowser.open(ui.url)

    # Exiting never drops the arms: the first Ctrl+C/SIGTERM with a session is a STOP (SAFE_HOLD),
    # the panel stays up for the confirmed release; a second signal exits with the arms held.
    signals = {"count": 0}
    stopped = threading.Event()

    def on_signal(*_):
        signals["count"] += 1
        if signals["count"] == 1 and robot.arms_owned:
            orchestrator.stop()
            print("STOP: руки в удержании. Отдайте руки в панели (подготовить → подтвердить); "
                  "повторный сигнал — выход без отдачи рук.", flush=True)
        else:
            stopped.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    while not stopped.wait(0.5):
        if signals["count"] and not robot.arms_owned:
            break
    log.emit("daemon_exit", arms_owned=robot.arms_owned)
    ui.shutdown()
    orchestrator.shutdown()
    pool.close_all()
    log.close()
    return 0
