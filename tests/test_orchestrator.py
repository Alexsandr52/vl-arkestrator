import math

from arkestrator.robot.base import ReleaseConfirmation
from arkestrator.robot.mock import FakePerception, MockObject, MockPerson, MockScenario
from conftest import auto_release, events, make_system, run_command, start


def release_session(system):
    orch = system.orch
    assert orch.request_release_prepare()["ok"]
    orch.process_pending()
    assert orch.phase.value == "RELEASE_READY"
    token = orch.release_token
    assert not orch.request_session_end(ReleaseConfirmation(token, True, False))["ok"]
    assert not orch.request_session_end(ReleaseConfirmation("stale", True, True))["ok"]
    assert orch.request_session_end(ReleaseConfirmation(token, True, True))["ok"]
    orch.process_pending()
    assert orch.phase.value == "OFF" and not system.robot.arms_owned
    assert orch.release_token is None


def operator_distance(system):
    pose = system.world.pose
    return math.hypot(pose.x - 1.3, pose.y - 0.1)


def last_outcome(system):
    return system.orch.history[-1]["outcome"]


def test_fetch_bottle_with_one_retry(system):
    system.world.scenario.grasp_results = {"bottle": [False, True]}
    start(system)
    reply = run_command(system, "Найди бутылку и принеси")
    assert reply["ok"], reply
    assert last_outcome(system) == "success"
    assert system.orch.history[-1]["grasp_attempts"] == 2
    assert system.world.objects[0].state == "given"
    assert system.orch.phase.value == "IDLE"
    assert system.orch.holding is None
    assert 0.6 < operator_distance(system) < 1.2
    stages = system.orch.history[-1]["stages"]
    assert stages.index("GRASP") < stages.index("BACKSTEP") < stages.index("RETURN")
    assert "Не получилось, пробую ещё раз" in system.speaker.recent()


def test_waits_for_next_command_after_success(system):
    system.world.objects.append(MockObject("bottle_2", "bottle", 2.0, -1.5, 0.0))
    start(system)
    assert run_command(system, "принеси бутылку")["ok"]
    assert last_outcome(system) == "success"
    assert run_command(system, "принеси ещё бутылку")["ok"]
    assert last_outcome(system) == "success"
    assert system.orch.phase.value == "IDLE"
    assert [o.state for o in system.world.objects] == ["given", "given"]


def test_object_not_found_returns_to_idle():
    system = make_system(objects=[])
    start(system)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "recovered:object_not_found"
    assert system.orch.phase.value == "IDLE"
    assert "Не нашёл бутылку" in system.speaker.recent()


def test_grasp_failure_steps_back_and_idles():
    system = make_system(scenario=MockScenario(grasp_results={"bottle": [False, False, False]}))
    start(system)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "recovered:grasp_failed"
    assert system.orch.history[-1]["grasp_attempts"] == 3
    assert not system.orch.nav.near_table
    assert system.robot.arm_pose == "ready"
    assert system.orch.phase.value == "IDLE"
    assert "Не смог взять бутылку" in system.speaker.recent()
    kinds = [e["kind"] for e in system.log.tail(limit=10_000)]
    assert "backstep" in kinds[kinds.index("task_failed"):]


def test_drop_while_carrying_recovers():
    system = make_system(scenario=MockScenario(drop_after_m=1.0))
    start(system)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "recovered:object_dropped"
    assert system.orch.holding is None
    assert system.world.objects[0].state == "floor"
    assert system.orch.phase.value == "IDLE"


def test_stop_goes_to_safe_hold_and_needs_confirmed_release():
    system = make_system()
    start(system)

    def stop_when_searching(dt):
        if system.orch.phase.value == "SEARCH" and not system.orch.stop_flag.is_set():
            system.orch.stop()

    system.clock.subscribe(stop_when_searching)
    run_command(system, "принеси бутылку")
    assert system.orch.phase.value == "SAFE_HOLD"
    assert system.robot.frozen
    assert not system.orch.submit_text("принеси бутылку")["ok"]
    assert not system.orch.request_session_end(ReleaseConfirmation("x", True, True))["ok"]  # not prepared
    release_session(system)


def test_cancel_during_approach_recovers_to_idle():
    system = make_system()
    start(system)

    def cancel_at_table(dt):
        if system.orch.nav.near_table and system.orch.phase.value == "APPROACH_TABLE":
            system.orch.cancel()

    system.clock.subscribe(cancel_at_table)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "recovered:cancelled"
    assert not system.orch.nav.near_table
    assert system.orch.phase.value == "IDLE"


def test_stale_release_is_ignored_and_cancel_refused_during_handover():
    system = make_system()
    start(system)
    seen = {"cancel": None, "stale": False}

    def person(dt):
        orch = system.orch
        if orch.phase.value == "RETURN" and not seen["stale"]:
            seen["stale"] = True
            orch.release_flag.set()  # pressed too early: must not open the hand later
        if orch.awaiting_release and seen["cancel"] is None:
            seen["cancel"] = orch.cancel()["ok"]
            seen["waiting_since"] = system.clock.now()
        if orch.awaiting_release and system.clock.now() - seen.get("waiting_since", 1e9) > 3.0:
            orch.release_object()

    system.clock.subscribe(person)
    run_command(system, "принеси бутылку")
    assert seen["stale"] and seen["cancel"] is False
    assert last_outcome(system) == "success"
    release = events(system, "handover_release")[0]
    waiting = [e for e in events(system, "phase") if e["phase"] == "HANDOVER"][0]
    assert release["t"] - waiting["t"] > 3.0


def test_command_while_busy_is_rejected(system):
    start(system)
    replies = []

    def second_command(dt):
        if system.orch.phase.value == "APPROACH_TABLE" and not replies:
            replies.append(system.orch.submit_text("принеси коробку"))

    system.clock.subscribe(second_command)
    run_command(system, "принеси бутылку")
    assert replies and not replies[0]["ok"]
    assert len(system.orch.history) == 1


def test_disabled_and_untaught_profiles_are_refused():
    system = make_system(taught={"bottle": (0.42, -0.18, 0.0)})
    start(system)
    napkin = system.orch.submit_text("принеси салфетку")
    assert not napkin["ok"] and "салфетку" in napkin["message"]
    box = system.orch.submit_text("принеси коробку")
    assert not box["ok"] and "Teach" in box["message"]
    assert system.orch.phase.value == "IDLE"


def test_attribute_selects_the_right_bottle():
    objects = [
        MockObject("bottle_red", "bottle", -1.8, 1.2, 0.0, attributes=("красный",)),
        MockObject("bottle_blue", "bottle", -1.8, 2.0, 0.0, attributes=("синий",)),
    ]
    system = make_system(objects=objects)
    auto_release(system)
    start(system)
    run_command(system, "принеси синюю бутылку")
    assert last_outcome(system) == "success"
    states = {o.name: o.state for o in system.world.objects}
    assert states == {"bottle_red": "table", "bottle_blue": "given"}


def test_commander_not_seen_brings_object_to_start_place():
    system = make_system(persons=[MockPerson("far", -3.0, -3.0)])
    auto_release(system)
    start(system)
    run_command(system, "принеси бутылку")
    assert last_outcome(system) == "success"
    assert "Не вижу вас. Принесу на это место" in system.speaker.recent()
    pose = system.world.pose
    assert math.hypot(pose.x, pose.y) < 0.4


def test_fault_goes_to_safe_hold():
    system = make_system(scenario=MockScenario(fault_after_s=20.0))
    start(system)
    run_command(system, "принеси бутылку")
    assert system.orch.phase.value == "SAFE_HOLD"
    assert last_outcome(system) == "safe_hold:overheated_joints"


def test_vlm_parser_used_when_available_and_falls_back_to_keywords():
    calls = []

    def parse(text, catalog):
        calls.append(text)
        raise RuntimeError("model not loaded")

    system = make_system(perception=FakePerception(parse=parse))
    auto_release(system)
    start(system)
    assert run_command(system, "принеси бутылку")["ok"]
    assert calls == ["принеси бутылку"]
    assert events(system, "parse")[-1]["parser"] == "keywords"


def test_voice_release_during_handover():
    system = make_system(perception=FakePerception(transcripts=["отпусти"], parse=lambda t, c: {"profile": "bottle", "object": "бутылка", "attributes": []}))
    start(system)
    replies = []

    def say_release(dt):
        if system.orch.awaiting_release and not replies:
            replies.append(system.orch.submit_audio(b"RIFF...."))

    system.clock.subscribe(say_release)
    run_command(system, "принеси бутылку")
    assert replies[0]["ok"] and replies[0]["heard"] == "отпусти"
    assert last_outcome(system) == "success"


def test_teach_records_object_position(tmp_path):
    system = make_system(objects=[MockObject("bottle", "bottle", 0.45, -0.2, 0.0)], taught={}, tmp_taught=tmp_path)
    reply = system.orch.request_teach("bottle")
    assert reply["ok"]
    system.orch.process_pending()
    taught = system.orch.profiles["bottle"].taught
    assert taught is not None
    assert taught.object_in_robot_m[0] == pytest_approx(0.45)
    assert (tmp_path / "bottle.json").exists()
    start(system)
    assert not system.orch.request_teach("bottle")["ok"]


def pytest_approx(value, tol=0.03):
    import pytest

    return pytest.approx(value, abs=tol)
