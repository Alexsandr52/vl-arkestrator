"""Client side of the bottle release protocol: return to the factory stand, then release.

Source pin: GR00T-pickup-bottle @ 1471a14 (branch ``arkestrator``)
  - sdk/g1_control/_compat/model_factory500.py: ``hold_before_release`` writes
    ``safe_abort_hold.json``; ``_return_confirmed`` / ``release_confirmed`` accept requests
    whose ``issued_monotonic`` is 0..2 s old on the same host.
  - tools/bottle_control_ui.py ``_write_attempt_request``: the operator panel pre-checks the
    hold status (active, ready, session, token, status age 0..1.5 s) before writing.
This module mirrors those pre-checks and payloads; it adds no condition of its own.

The worker publishes two tokens per hold: ``return_token`` (move the arms to the factory
stand) and ``release_token`` (hand the arms to the factory controller). Release is accepted
only after the return completed. Nothing here sends a motor command; the worker does.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

STATUS_FILE = "safe_abort_hold.json"
RETURN_FILE = "return_to_factory_stand.json"
RELEASE_FILE = "release_supported.json"
# bottle_control_ui: the hold status must be at most 1.5 s old when a request is written.
STATUS_MAX_AGE_S = 1.5


class ReleaseRequestError(RuntimeError):
    """The worker is not waiting for this request; nothing was written."""


@dataclass(frozen=True)
class HoldStatus:
    session: str
    active: bool
    release_ready: bool
    release_token: str
    return_ready: bool
    return_token: str
    return_completed: bool
    return_failed: bool
    return_in_progress: bool
    return_stage: str
    wait_reason: str
    time_monotonic: float
    raw: dict[str, Any]

    def age(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.time_monotonic

    def fresh(self, now: Optional[float] = None) -> bool:
        age = self.age(now)
        return math.isfinite(age) and 0 <= age <= STATUS_MAX_AGE_S


def read_hold_status(folder: Path) -> Optional[HoldStatus]:
    """The worker's hold status, or None while it is absent or unreadable."""
    try:
        raw = json.loads((Path(folder) / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    stamp = raw.get("time_monotonic")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return None
    return HoldStatus(
        session=str(raw.get("session", "")),
        active=raw.get("active") is True,
        release_ready=raw.get("release_ready") is True,
        release_token=str(raw.get("release_token") or ""),
        return_ready=raw.get("return_ready") is True,
        return_token=str(raw.get("return_token") or ""),
        return_completed=raw.get("return_completed") is True,
        return_failed=raw.get("return_failed") is True,
        return_in_progress=raw.get("return_in_progress") is True,
        return_stage=str(raw.get("return_stage") or ""),
        wait_reason=str(raw.get("return_wait_reason") or ""),
        time_monotonic=float(stamp),
        raw=raw,
    )


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(path)


def _checked_status(folder: Path, ready_field: str, token_field: str, token: str) -> HoldStatus:
    folder = Path(folder)
    status = read_hold_status(folder)
    if status is None:
        raise ReleaseRequestError("Контроллер не публикует статус удержания")
    if not status.active or status.session != folder.name:
        raise ReleaseRequestError("Контроллер не в удержании этой сессии")
    if getattr(status, ready_field) is not True:
        raise ReleaseRequestError("Контроллер ещё не готов: " + (status.wait_reason or "нет готовности"))
    if not token or getattr(status, token_field) != token:
        raise ReleaseRequestError("Токен не совпадает с текущим удержанием")
    if not status.fresh():
        raise ReleaseRequestError("Статус удержания устарел")
    return status


def write_return_request(folder: Path, return_token: str, *, operator_confirmed_clearance: bool) -> None:
    """Ask the worker to move the arms to the factory stand (``return_to_factory_stand.json``)."""
    if operator_confirmed_clearance is not True:
        raise ReleaseRequestError("Нет подтверждения оператора, что путь рук свободен")
    folder = Path(folder)
    _checked_status(folder, "return_ready", "return_token", return_token)
    _write_atomic(
        folder / RETURN_FILE,
        dict(
            session=folder.name,
            **{"return": True},
            operator_confirmed_clearance=True,
            return_token=return_token,
            issued_monotonic=time.monotonic(),
        ),
    )


def write_release_request(folder: Path, release_token: str, *, supported_and_sweep_clear: bool) -> None:
    """Hand the arms to the factory controller (``release_supported.json``)."""
    if supported_and_sweep_clear is not True:
        raise ReleaseRequestError("Нет подтверждения оператора: предмет поддержан и путь рук свободен")
    folder = Path(folder)
    status = _checked_status(folder, "release_ready", "release_token", release_token)
    if not status.return_completed:
        raise ReleaseRequestError("Отдача рук возможна только после возврата в заводскую стойку")
    _write_atomic(
        folder / RELEASE_FILE,
        dict(
            session=folder.name,
            release=True,
            box_supported_and_arm_sweep_clear=True,
            release_token=release_token,
            issued_monotonic=time.monotonic(),
        ),
    )
