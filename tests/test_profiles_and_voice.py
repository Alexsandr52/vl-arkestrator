import copy
import json
from pathlib import Path

import pytest
import yaml

from arkestrator.config import ConfigError, load_session_config
from arkestrator.profiles import ProfileError, TaughtPose, load_profiles, load_taught, parse_profile, save_taught
from arkestrator.voice import classify_utterance, parse_fetch_keywords

ROOT = Path(__file__).resolve().parents[1]


def bottle_data():
    return yaml.safe_load((ROOT / "profiles" / "bottle.yaml").read_text())


def test_repo_profiles_load():
    profiles = load_profiles(ROOT / "profiles", local={"vla": {"bottle": {"checkpoint": "/x"}}})
    assert set(profiles) == {"bottle", "box", "napkin"}
    assert profiles["bottle"].vla.checkpoint == "/x"
    assert profiles["box"].vla.execute_steps == 10
    assert profiles["box"].vla.image_key == "observation.images.head_camera"
    assert not profiles["napkin"].enabled
    ready, why = profiles["bottle"].ready_for_tasks
    assert not ready and "Teach" in why
    ready, why = profiles["box"].ready_for_tasks
    assert not ready and "чекпойнт" in why


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(typo=1), "unknown keys"),
        (lambda d: d["vla"].update(execute_steps=60), "execute_steps"),
        (lambda d: d["grasp"]["verify"].update(tactile="ignore", visual="ignore"), "at least one"),
        (lambda d: d["grasp"].update(max_attempts=0), "max_attempts"),
        (lambda d: d["vla"].update(contract="grpc"), "contract"),
    ],
)
def test_profile_validation(mutate, message):
    data = copy.deepcopy(bottle_data())
    mutate(data)
    with pytest.raises(ProfileError, match=message):
        parse_profile(data, "bottle.yaml")


def test_taught_roundtrip_and_range(tmp_path):
    path = tmp_path / "bottle.json"
    save_taught(path, TaughtPose((0.42, -0.18, 0.0), "2026-11-24T10:00:00", "teach"))
    assert load_taught(path).object_in_robot_m == (0.42, -0.18, 0.0)
    path.write_text(json.dumps({"object_in_robot_m": [3.0, 0, 0]}))
    with pytest.raises(ProfileError, match="workspace"):
        load_taught(path)


def test_session_config_example_matches_defaults_and_rejects_typos(tmp_path):
    cfg = load_session_config(ROOT / "config" / "session.example.yaml")
    assert cfg.nav.backstep_m == 0.30 and cfg.nav.search_yaw_rate_rad_s == -0.30
    bad = tmp_path / "s.yaml"
    bad.write_text("nav:\n  backstp_m: 0.3\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_session_config(bad)


@pytest.mark.parametrize(
    "text, kind",
    [("Отпусти", "release"), ("держу, отпускай", "release"), ("стоп", "cancel"), ("отмена!", "cancel"),
     ("найди бутылку и принеси", "command"), ("   ", "empty")],
)
def test_classify_utterance(text, kind):
    assert classify_utterance(text) == kind


def test_keyword_parser():
    catalog = [p.catalog_entry() for p in load_profiles(ROOT / "profiles").values()]
    assert parse_fetch_keywords("Найди красную бутылку и принеси", catalog) == {
        "profile": "bottle", "object": "бутылка", "attributes": ["красный"]}
    assert parse_fetch_keywords("принеси коробку", catalog)["profile"] == "box"
    assert parse_fetch_keywords("салфетку дай", catalog)["profile"] == "napkin"
    assert parse_fetch_keywords("привет робот", catalog) is None
