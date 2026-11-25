"""Regression tests for the robot-safety audit of 24.11.2026 (B = blocker, S = should-fix)."""

import math
from dataclasses import replace

import pytest

from arkestrator.config import ConfigError, NavConfig, load_session_config
from arkestrator.errors import NavigationRuleError, SafeHoldError, StopRequested
from arkestrator.nav.navigator import Navigator
from arkestrator.robot.base import ReleaseConfirmation
from arkestrator.robot.mock import MockScenario
from arkestrator.voice import classify_utterance
from conftest import auto_release, events, make_system, run_command, start
from test_navigator import make_nav
from test_orchestrator import last_outcome, release_session


def test_b1_fault_while_idle_drops_the_queued_task():
    system = make_system(scenario=MockScenario(fault_after_s=1.0))
    start(system)
    assert system.orch.submit_text("принеси бутылку")["ok"]  # accepted while still IDLE
    system.orch._idle_watch()  # the worker notices the fault before taking the task
    assert system.orch.phase.value == "SAFE_HOLD"
    calls_before = list(system.robot.calls)
    system.orch.process_pending()
    assert events(system, "task_dropped")
    assert system.robot.calls == calls_before  # nothing moved
    assert system.orch.phase.value == "SAFE_HOLD"


def test_b1_safe_hold_cannot_be_left_except_by_release():
    system = make_system()
    start(system)
    system.orch.stop()
    system.orch._idle_watch()
    assert system.orch.phase.value == "SAFE_HOLD"
    system.orch._set_phase(type(system.orch.phase).IDLE)
    assert system.orch.phase.value == "SAFE_HOLD"
    with pytest.raises((SafeHoldError, StopRequested)):
        system.robot.move_arms("ready")
    release_session(system)


def test_b2_stop_interrupts_an_arm_motion():
    system = make_system()
    start(system)
    state = {"stopped_during": None}

    def stop_while_extending(dt):
        if system.robot.arm_pose == "handover_right" and state["stopped_during"] is None:
            state["stopped_during"] = system.robot.arm_pose
            system.orch.stop()

    system.clock.subscribe(stop_while_extending)
    run_command(system, "принеси бутылку")
    assert state["stopped_during"] == "handover_right"
    assert system.orch.phase.value == "SAFE_HOLD"
    assert last_outcome(system) == "safe_hold:stop_requested"
    assert not system.orch.awaiting_release
    release_session(system)


def test_b4_release_needs_a_fresh_token_each_time():
    system = make_system()
    start(system)
    orch = system.orch
    assert orch.request_release_prepare()["ok"]
    orch.process_pending()
    first = orch.release_token
    orch.request_session_end(ReleaseConfirmation(first, True, True))
    orch.process_pending()
    assert orch.phase.value == "OFF"
    start(system)
    orch.request_release_prepare()
    orch.process_pending()
    assert orch.release_token != first
    assert not orch.request_session_end(ReleaseConfirmation(first, True, True))["ok"]


def test_b5_table_walk_leaving_the_corridor_is_a_safe_hold():
    nav, _ = make_nav(MockScenario(yaw_drift_rad_s=0.3))
    nav.walk(0.6)  # away from the table the heading controller absorbs the drift
    nav.near_table = True
    with pytest.raises(SafeHoldError) as err:
        nav.backstep()
    assert err.value.reason == "table_walk_left_corridor"


def test_b5_heading_correction_is_limited_at_the_table():
    nav, robot = make_nav()
    sent = []
    original = robot.set_base_velocity
    robot.set_base_velocity = lambda vx, vy, wz: (sent.append(wz), original(vx, vy, wz))
    robot.world.pose = replace(robot.world.pose, yaw=0.0)
    nav.near_table = True
    nav.walk(0.2)
    assert sent and max(abs(w) for w in sent) <= NavConfig().careful_max_yaw_rate_rad_s + 1e-9


def test_s11_commands_and_config_are_bounded(tmp_path):
    nav, _ = make_nav()
    with nav.segment():
        with pytest.raises(NavigationRuleError):
            nav.tick((1.0, 0.0, 0.0))
        with pytest.raises(NavigationRuleError):
            nav.tick((0.0, 0.0, float("nan")))
    bad = tmp_path / "s.yaml"
    bad.write_text("nav:\n  search_yaw_rate_rad_s: -3\n")
    with pytest.raises(ConfigError, match="search_yaw_rate"):
        load_session_config(bad)


def test_s1_near_table_survives_a_session_restart():
    system = make_system()
    start(system)

    def stop_at_table(dt):
        if system.orch.nav.near_table and system.orch.phase.value == "GRASP" and not system.orch.stop_flag.is_set():
            system.orch.stop()

    system.clock.subscribe(stop_at_table)
    run_command(system, "принеси бутылку")
    assert system.orch.phase.value == "SAFE_HOLD"
    release_session(system)
    start(system)
    assert system.orch.nav.near_table
    assert "шага назад" in system.orch.attention
    auto_release(system)
    system.clock._listeners.remove(stop_at_table)
    run_command(system, "принеси бутылку")
    assert system.orch.history[-1]["stages"][0] == "BACKSTEP"


def test_s3_unverified_lift_is_put_back_on_the_table_not_dropped():
    system = make_system(scenario=MockScenario(vlm_denies=1))
    auto_release(system)
    start(system)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "success"
    assert system.orch.history[-1]["grasp_attempts"] == 2
    assert "arms:lowered_to_table" in system.robot.calls
    assert system.world.objects[0].state == "given"
    assert all(e.get("state") != "floor" for e in events(system, "object_state"))


def test_s5_failed_recovery_does_not_accept_new_tasks():
    system = make_system(scenario=MockScenario(grasp_results={"bottle": [False, False, False]}))
    start(system)

    def stall_in_recovery(dt):
        if system.orch.phase.value == "RECOVERING":
            system.world.scenario.stall = True

    system.clock.subscribe(stall_in_recovery)
    run_command(system, "принеси бутылку")
    assert system.orch.phase.value == "SAFE_HOLD"
    assert "Восстановление не завершено" in system.orch.attention
    assert not system.orch.submit_text("принеси бутылку")["ok"]


@pytest.mark.parametrize("text", ["не отпускай", "я не держу", "нет, не отдавай"])
def test_s8_negated_release_is_not_a_release(text):
    assert classify_utterance(text) != "release"


def test_navigator_rejects_velocity_without_segment():
    nav = Navigator(make_nav()[1], make_nav()[1].clock, NavConfig(), checkpoint=lambda: None)
    with pytest.raises(NavigationRuleError):
        nav.tick((0.2, 0.0, 0.0))
    assert math.isfinite(nav.pose().x)
