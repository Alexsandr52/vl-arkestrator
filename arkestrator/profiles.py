"""Task profiles: one per VLA skill (bottle, box, napkin…).

A profile says how to recognise the object in speech and on camera, which π0.5 server and
checkpoint to run, how the grasp ends, which scripted tail follows, how success is verified
and where the robot must stand (taught pose). Unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

CONTRACTS = ("arkestrator_stdio_v1", "lerobot_stdio_v1", "box_stdio_v1")
TAILS = ("bottle_squeeze_lift_carry", "box_squeeze_tuck", "napkin_pinch")
HANDS = ("right", "left", "both")
# contact: VLA ends on measured contact; a timeout without contact is a miss (no tail).
# contact_or_time: VLA ends on contact or timeout and the tail always runs (box pipeline).
ENDS = ("contact", "contact_or_time")
CHECK_MODES = ("required", "ignore")
CARRY_CHECKS = ("carry_pressure", "none")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class VlaSpec:
    contract: str
    policy: str
    server: str
    image_key: str
    prompt: str
    horizon: int
    execute_steps: int
    device: str
    python: Optional[str] = None
    checkpoint: Optional[str] = None
    noise_seed: Optional[int] = None  # fixed per request when the checkpoint was validated so (box: 123)


@dataclass(frozen=True)
class VerifySpec:
    tactile: str
    visual: str
    visual_votes: int
    visual_needed: int


@dataclass(frozen=True)
class GraspSpec:
    hand: str
    start_pose: str
    end: str
    max_policy_seconds: float
    tail: str
    max_attempts: int
    verify: VerifySpec
    carry_check: str


@dataclass(frozen=True)
class TaughtPose:
    """Object position in the robot frame when the robot stands as in the training data."""

    object_in_robot_m: tuple[float, float, float]
    recorded: str
    source: str


@dataclass(frozen=True)
class Profile:
    name: str
    enabled: bool
    display_ru: str
    accusative_ru: str
    synonyms_ru: tuple[str, ...]
    detector_classes: tuple[str, ...]
    vla: VlaSpec
    grasp: GraspSpec
    ready_pose: str
    handover_pose: str
    taught: Optional[TaughtPose] = None

    @property
    def ready_for_tasks(self) -> tuple[bool, str]:
        """Whether a fetch task may start, and why not (Russian, for TTS/UI)."""
        if not self.enabled:
            return False, f"Пока не умею брать {self.accusative_ru}"
        if not self.vla.checkpoint:
            return False, f"Для {self.display_ru} не задан чекпойнт"
        if self.taught is None:
            return False, f"Сначала покажите, как стоять у стола для {self.display_ru} (Teach)"
        return True, ""

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_ru": self.display_ru,
            "synonyms_ru": list(self.synonyms_ru),
            "detector_classes": list(self.detector_classes),
            "enabled": self.enabled,
        }


def _take(data: dict[str, Any], key: str, kind, where: str, default=...):
    if key not in data:
        if default is ...:
            raise ProfileError(f"{where}: missing '{key}'")
        return default
    value = data[key]
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{where}.{key}: expected a number")
        return float(value)
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError(f"{where}.{key}: expected an integer")
        return value
    if kind is bool:
        if not isinstance(value, bool):
            raise ProfileError(f"{where}.{key}: expected true/false")
        return value
    if kind is str:
        if value is None and default is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"{where}.{key}: expected a non-empty string")
        return value
    if kind is list:
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            raise ProfileError(f"{where}.{key}: expected a non-empty list of strings")
        return tuple(value)
    if kind is dict:
        if not isinstance(value, dict):
            raise ProfileError(f"{where}.{key}: expected a mapping")
        return value
    raise TypeError(kind)


def _reject_unknown(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ProfileError(f"{where}: unknown keys {unknown}")


def _choice(value: str, choices: tuple[str, ...], where: str) -> str:
    if value not in choices:
        raise ProfileError(f"{where}: '{value}' not in {list(choices)}")
    return value


def parse_profile(data: dict[str, Any], where: str) -> Profile:
    if not isinstance(data, dict):
        raise ProfileError(f"{where}: expected a mapping")
    _reject_unknown(
        data,
        {"name", "enabled", "display_ru", "accusative_ru", "synonyms_ru", "detector_classes",
         "vla", "grasp", "ready_pose", "handover_pose"},
        where,
    )
    vla_data = _take(data, "vla", dict, where)
    _reject_unknown(
        vla_data,
        {"contract", "policy", "server", "image_key", "prompt", "horizon", "execute_steps",
         "device", "python", "checkpoint", "noise_seed"},
        f"{where}.vla",
    )
    vla = VlaSpec(
        contract=_choice(_take(vla_data, "contract", str, f"{where}.vla"), CONTRACTS, f"{where}.vla.contract"),
        policy=_choice(_take(vla_data, "policy", str, f"{where}.vla"), ("pi05",), f"{where}.vla.policy"),
        server=_take(vla_data, "server", str, f"{where}.vla"),
        image_key=_take(vla_data, "image_key", str, f"{where}.vla"),
        prompt=_take(vla_data, "prompt", str, f"{where}.vla"),
        horizon=_take(vla_data, "horizon", int, f"{where}.vla"),
        execute_steps=_take(vla_data, "execute_steps", int, f"{where}.vla"),
        device=_choice(_take(vla_data, "device", str, f"{where}.vla"), ("cuda", "cpu"), f"{where}.vla.device"),
        python=_take(vla_data, "python", str, f"{where}.vla", None),
        checkpoint=_take(vla_data, "checkpoint", str, f"{where}.vla", None),
        noise_seed=vla_data.get("noise_seed"),
    )
    if vla.noise_seed is not None and (isinstance(vla.noise_seed, bool) or not isinstance(vla.noise_seed, int)):
        raise ProfileError(f"{where}.vla.noise_seed: expected an integer or null")
    if not 0 < vla.execute_steps <= vla.horizon:
        raise ProfileError(f"{where}.vla: execute_steps must be within 1..horizon")
    if vla.contract == "box_stdio_v1" and (vla.prompt, vla.horizon, vla.execute_steps) != ("Pick up box", 50, 10):
        raise ProfileError(f"{where}.vla: box_stdio_v1 runs only 'Pick up box' with horizon 50, execute_steps 10")

    grasp_data = _take(data, "grasp", dict, where)
    _reject_unknown(
        grasp_data,
        {"hand", "start_pose", "end", "max_policy_seconds", "tail", "max_attempts", "verify", "carry_check"},
        f"{where}.grasp",
    )
    verify_data = _take(grasp_data, "verify", dict, f"{where}.grasp")
    _reject_unknown(verify_data, {"tactile", "visual", "visual_votes", "visual_needed"}, f"{where}.grasp.verify")
    verify = VerifySpec(
        tactile=_choice(_take(verify_data, "tactile", str, f"{where}.grasp.verify"), CHECK_MODES, f"{where}.grasp.verify.tactile"),
        visual=_choice(_take(verify_data, "visual", str, f"{where}.grasp.verify"), CHECK_MODES, f"{where}.grasp.verify.visual"),
        visual_votes=_take(verify_data, "visual_votes", int, f"{where}.grasp.verify"),
        visual_needed=_take(verify_data, "visual_needed", int, f"{where}.grasp.verify"),
    )
    if not 0 < verify.visual_needed <= verify.visual_votes:
        raise ProfileError(f"{where}.grasp.verify: visual_needed must be within 1..visual_votes")
    if verify.tactile == "ignore" and verify.visual == "ignore":
        raise ProfileError(f"{where}.grasp.verify: at least one check must be required")
    grasp = GraspSpec(
        hand=_choice(_take(grasp_data, "hand", str, f"{where}.grasp"), HANDS, f"{where}.grasp.hand"),
        start_pose=_take(grasp_data, "start_pose", str, f"{where}.grasp"),
        end=_choice(_take(grasp_data, "end", str, f"{where}.grasp"), ENDS, f"{where}.grasp.end"),
        max_policy_seconds=_take(grasp_data, "max_policy_seconds", float, f"{where}.grasp"),
        tail=_choice(_take(grasp_data, "tail", str, f"{where}.grasp"), TAILS, f"{where}.grasp.tail"),
        max_attempts=_take(grasp_data, "max_attempts", int, f"{where}.grasp"),
        verify=verify,
        carry_check=_choice(_take(grasp_data, "carry_check", str, f"{where}.grasp"), CARRY_CHECKS, f"{where}.grasp.carry_check"),
    )
    if grasp.max_attempts < 1:
        raise ProfileError(f"{where}.grasp.max_attempts must be at least 1")
    if grasp.max_policy_seconds <= 0:
        raise ProfileError(f"{where}.grasp.max_policy_seconds must be positive")

    synonyms = tuple(s.lower().replace("ё", "е") for s in _take(data, "synonyms_ru", list, where))
    return Profile(
        name=_take(data, "name", str, where),
        enabled=_take(data, "enabled", bool, where),
        display_ru=_take(data, "display_ru", str, where),
        accusative_ru=_take(data, "accusative_ru", str, where),
        synonyms_ru=synonyms,
        detector_classes=_take(data, "detector_classes", list, where),
        vla=vla,
        grasp=grasp,
        ready_pose=_take(data, "ready_pose", str, where),
        handover_pose=_take(data, "handover_pose", str, where),
    )


def load_taught(path: Path) -> Optional[TaughtPose]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    point = data.get("object_in_robot_m")
    if (
        not isinstance(point, list)
        or len(point) != 3
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point)
    ):
        raise ProfileError(f"{path}: object_in_robot_m must be [x, y, z] in meters")
    x, y, z = (float(v) for v in point)
    if not (0.15 <= x <= 1.2 and abs(y) <= 0.6 and -0.5 <= z <= 1.5):
        raise ProfileError(f"{path}: taught object position {point} is outside the plausible workspace")
    return TaughtPose((x, y, z), str(data.get("recorded", "")), str(data.get("source", "teach")))


def save_taught(path: Path, pose: TaughtPose) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"object_in_robot_m": list(pose.object_in_robot_m), "recorded": pose.recorded, "source": pose.source},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_profiles(
    profiles_dir: Path,
    taught_dir: Optional[Path] = None,
    local: Optional[dict[str, Any]] = None,
) -> dict[str, Profile]:
    """Load ``profiles/*.yaml``; fill python/checkpoint from ``local.json`` → ``vla.<name>``."""
    from dataclasses import replace

    local_vla = (local or {}).get("vla", {})
    profiles: dict[str, Profile] = {}
    for path in sorted(Path(profiles_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = parse_profile(data, str(path.name))
        if profile.name != path.stem:
            raise ProfileError(f"{path.name}: name '{profile.name}' must match the file name")
        overrides = local_vla.get(profile.name, {}) if isinstance(local_vla, dict) else {}
        for key in ("python", "checkpoint"):
            value = overrides.get(key)
            if value not in (None, "") and not isinstance(value, str):
                raise ProfileError(f"local.json vla.{profile.name}.{key}: expected a path string")
        vla = replace(
            profile.vla,
            python=overrides.get("python") or profile.vla.python,
            checkpoint=overrides.get("checkpoint") or profile.vla.checkpoint,
        )
        taught = load_taught(Path(taught_dir) / f"{profile.name}.json") if taught_dir else None
        profiles[profile.name] = replace(profile, vla=vla, taught=taught)
    if not profiles:
        raise ProfileError(f"No profiles in {profiles_dir}")
    return profiles
