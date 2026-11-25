"""Release-protocol client against the bottle worker's own acceptance functions (offline)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from arkestrator.robot.g1 import release

BOTTLE_SDK = Path(__file__).resolve().parents[1] / "third_party" / "GR00T-pickup-bottle" / "sdk"


@pytest.fixture()
def bottle_accepts():
    """The worker-side checks from the pinned checkout (pure file/clock logic)."""
    if not (BOTTLE_SDK / "g1_control").is_dir():
        pytest.skip("pinned bottle checkout missing (git submodule update --init)")
    sys.path.insert(0, str(BOTTLE_SDK))
    try:
        from g1_control._compat.model_factory500 import _return_confirmed, release_confirmed
    finally:
        sys.path.remove(str(BOTTLE_SDK))
    return _return_confirmed, release_confirmed


def write_status(folder: Path, **override):
    status = dict(
        active=True,
        pid=1234,
        session=folder.name,
        release_ready=True,
        release_token="REL",
        return_ready=True,
        return_token="RET",
        return_completed=False,
        return_failed=False,
        return_in_progress=False,
        return_stage="HOLDING",
        return_wait_reason="",
        time_monotonic=time.monotonic(),
    )
    status.update(override)
    (folder / release.STATUS_FILE).write_text(json.dumps(status))


@pytest.fixture()
def folder(tmp_path):
    session = tmp_path / "20261125_120000"
    session.mkdir()
    return session


def test_return_request_is_accepted_by_the_worker(folder, bottle_accepts):
    return_confirmed, _ = bottle_accepts
    write_status(folder)
    release.write_return_request(folder, "RET", operator_confirmed_clearance=True)
    assert return_confirmed(folder / release.RETURN_FILE, folder.name, "RET")
    assert not return_confirmed(folder / release.RETURN_FILE, folder.name, "OTHER")


def test_release_request_is_accepted_by_the_worker_after_return(folder, bottle_accepts):
    _, release_confirmed = bottle_accepts
    write_status(folder, return_completed=True, return_ready=False)
    release.write_release_request(folder, "REL", supported_and_sweep_clear=True)
    assert release_confirmed(folder / release.RELEASE_FILE, folder.name, token="REL")


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(active=False), "не в удержании"),
        (dict(session="other"), "не в удержании"),
        (dict(return_ready=False, return_wait_reason="база движется"), "база движется"),
        (dict(return_token="NEW"), "Токен"),
        (dict(time_monotonic=time.monotonic() - 5.0), "устарел"),
        (dict(time_monotonic=time.monotonic() + 5.0), "устарел"),
    ],
)
def test_return_request_pre_checks_write_nothing(folder, override, message):
    write_status(folder, **override)
    with pytest.raises(release.ReleaseRequestError, match=message):
        release.write_return_request(folder, "RET", operator_confirmed_clearance=True)
    assert not (folder / release.RETURN_FILE).exists()


def test_operator_confirmations_are_required(folder):
    write_status(folder, return_completed=True)
    with pytest.raises(release.ReleaseRequestError):
        release.write_return_request(folder, "RET", operator_confirmed_clearance=False)
    with pytest.raises(release.ReleaseRequestError):
        release.write_release_request(folder, "REL", supported_and_sweep_clear=False)
    assert not (folder / release.RETURN_FILE).exists()
    assert not (folder / release.RELEASE_FILE).exists()


def test_release_before_return_is_refused(folder):
    write_status(folder, return_completed=False)
    with pytest.raises(release.ReleaseRequestError, match="возврата"):
        release.write_release_request(folder, "REL", supported_and_sweep_clear=True)
    assert not (folder / release.RELEASE_FILE).exists()


def test_missing_or_broken_status(folder):
    with pytest.raises(release.ReleaseRequestError, match="не публикует"):
        release.write_return_request(folder, "RET", operator_confirmed_clearance=True)
    (folder / release.STATUS_FILE).write_text("{broken")
    assert release.read_hold_status(folder) is None
    (folder / release.STATUS_FILE).write_text(json.dumps(dict(active=True, time_monotonic=True)))
    assert release.read_hold_status(folder) is None


def test_empty_token_never_matches(folder):
    write_status(folder, return_token="")
    with pytest.raises(release.ReleaseRequestError, match="Токен"):
        release.write_return_request(folder, "", operator_confirmed_clearance=True)
