"""Session configuration: defaults + ``config/session.yaml`` + ``config/local.json``.

Navigation numbers are taken from GR00T-pickup-bottle ``disposal.yaml`` (field-tested 19.11);
see ``config/session.example.yaml`` for the reasons behind each value.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class NavConfig:
    tick_s: float = 0.02
    walk_speed_m_s: float = 0.20
    heading_gain: float = 1.2
    backstep_m: float = 0.30
    backstep_speed_m_s: float = 0.20
    search_yaw_rate_rad_s: float = -0.30
    search_sweep_deg: float = 360.0
    detect_frames: int = 2
    center_box_deg: float = 10.0
    center_brake_lead_deg: float = 2.0
    center_yaw_rate_rad_s: float = 0.30
    center_attempts: int = 4
    recover_turn_deg: float = 40.0
    near_max_bearing_deg: float = 20.0
    settle_s: float = 0.4
    still_speed_m_s: float = 0.02
    still_yaw_rad_s: float = 0.02
    stall_timeout_s: float = 4.0
    min_walk_m: float = 0.02
    standoff_m: float = 0.80
    sidestep_min_m: float = 0.04
    final_min_m: float = 0.03
    measure_frames: int = 3
    measure_timeout_s: float = 2.0
    retrace_min_segment_m: float = 0.25
    require_clear_corridor: bool = False
    # Walking at the table (BACKSTEP, final approach, side steps): guards of
    # GR00T-pickup-bottle post_grasp_walk.perform_motion; a violation ends in SAFE_HOLD.
    careful_corridor_m: float = 0.06
    careful_heading_deg: float = 10.0
    careful_wrong_direction_m: float = 0.03
    careful_max_yaw_rate_rad_s: float = 0.12
    # Hard caps on any base command (a typo in the config must not spin the robot).
    max_linear_m_s: float = 0.35
    max_yaw_rate_rad_s: float = 0.6


@dataclass(frozen=True)
class PersonConfig:
    labels: tuple[str, ...] = ("person",)
    commander_max_range_m: float = 3.5
    commander_max_bearing_deg: float = 35.0
    standoff_m: float = 0.9
    standoff_tolerance_m: float = 0.10
    max_step_m: float = 0.5
    max_approach_steps: int = 8
    search_yaw_rate_rad_s: float = 0.30
    search_sweep_deg: float = 360.0
    reid_threshold: float = 0.6
    reid_margin: float = 0.05
    match_radius_m: float = 1.5


@dataclass(frozen=True)
class GraspConfig:
    relocate_tolerance_m: float = 0.04
    visual_check_radius_m: float = 0.12
    verify_target: str = "attributes"  # never | attributes | always


@dataclass(frozen=True)
class HandoverConfig:
    open_settle_s: float = 1.0


@dataclass(frozen=True)
class UiConfig:
    host: str = "127.0.0.1"
    port: int = 8770


@dataclass(frozen=True)
class SessionConfig:
    nav: NavConfig = field(default_factory=NavConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    outputs_dir: str = "outputs"
    profiles_dir: str = "profiles"
    taught_dir: str = "config/taught"
    vla_resident: int = 1

    def path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate


_SECTIONS = {
    "nav": NavConfig,
    "person": PersonConfig,
    "grasp": GraspConfig,
    "handover": HandoverConfig,
    "ui": UiConfig,
}


def _build(cls, values: dict[str, Any], where: str):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    converted = {}
    for name, value in values.items():
        default = known[name].default
        if isinstance(default, tuple):
            value = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}.{name}: expected true/false")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where}.{name}: expected a number")
            value = float(value)
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where}.{name}: expected an integer")
        converted[name] = value
    return cls(**converted)


def load_session_config(path: Optional[Path] = None) -> SessionConfig:
    """Load ``config/session.yaml`` if present, else the defaults above."""
    if path is None:
        path = ROOT / "config" / "session.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    sections = {}
    top = {}
    for key, value in data.items():
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise ConfigError(f"{path}:{key}: expected a mapping")
            sections[key] = _build(_SECTIONS[key], value, key)
        else:
            top[key] = value
    allowed_top = {"outputs_dir", "profiles_dir", "taught_dir", "vla_resident"}
    unknown = sorted(set(top) - allowed_top)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")
    config = SessionConfig(**sections, **top)
    _check_bounds(config)
    if config.grasp.verify_target not in ("never", "attributes", "always"):
        raise ConfigError("grasp.verify_target must be never, attributes or always")
    if config.vla_resident < 1:
        raise ConfigError("vla_resident must be at least 1")
    return config


_BOUNDS = {
    # (section, key): (min, max) — plausible ranges, not tuning advice.
    ("nav", "walk_speed_m_s"): (0.15, 0.35),
    ("nav", "backstep_m"): (0.15, 0.5),
    ("nav", "backstep_speed_m_s"): (0.1, 0.3),
    ("nav", "search_yaw_rate_rad_s"): (-0.6, 0.6),
    ("nav", "center_yaw_rate_rad_s"): (0.2, 0.6),
    ("nav", "search_sweep_deg"): (0.0, 360.0),
    ("nav", "standoff_m"): (0.4, 1.5),
    ("nav", "careful_corridor_m"): (0.02, 0.15),
    ("nav", "careful_heading_deg"): (3.0, 20.0),
    ("nav", "careful_max_yaw_rate_rad_s"): (0.0, 0.3),
    ("nav", "max_linear_m_s"): (0.15, 0.4),
    ("nav", "max_yaw_rate_rad_s"): (0.2, 0.8),
    ("nav", "stall_timeout_s"): (1.0, 20.0),
    ("nav", "tick_s"): (0.01, 0.1),
    ("person", "standoff_m"): (0.6, 2.0),
    ("person", "max_step_m"): (0.1, 1.0),
    ("person", "search_yaw_rate_rad_s"): (-0.6, 0.6),
}


def _check_bounds(config: SessionConfig) -> None:
    for (section, key), (low, high) in _BOUNDS.items():
        value = getattr(getattr(config, section), key)
        if not low <= value <= high:
            raise ConfigError(f"{section}.{key}={value} outside [{low}, {high}]")
    nav = config.nav
    if abs(nav.search_yaw_rate_rad_s) > nav.max_yaw_rate_rad_s or nav.walk_speed_m_s > nav.max_linear_m_s:
        raise ConfigError("nav speeds exceed max_linear_m_s / max_yaw_rate_rad_s")


def load_local(path: Optional[Path] = None) -> dict[str, Any]:
    """Machine-specific settings (paths, hosts). Missing file → empty settings."""
    if path is None:
        path = ROOT / "config" / "local.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object")
    return data
