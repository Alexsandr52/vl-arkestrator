import hashlib
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arkestrator.errors import RecoverableError
from arkestrator.vla.client import (
    BoxStdioContract,
    ContractError,
    LerobotStdioContract,
    StdioPolicyClient,
    encode_request,
    jpeg_size,
    make_contract,
)
from arkestrator.vla.contract import BOTTLE_DATASET_ORDER, BOTTLE_TO_CANONICAL, JOINT_NAMES
from arkestrator.vla.pool import PolicyPool

FAKES = Path(__file__).with_name("fakes")
STATE = np.linspace(-0.5, 0.5, 28)


def fake_jpeg(width=1280, height=720) -> bytes:
    """SOI + SOF0 header + filler + EOI: enough for the size check, never decoded."""
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + sof + bytes(range(256)) * 4 + b"\xff\xd9"


JPEG = fake_jpeg()


def spec(**overrides):
    values = dict(contract="lerobot_stdio_v1", checkpoint="/ckpt/bottle", horizon=50, execute_steps=50,
                  prompt="Pick up bottle", image_key="observation.images.head", noise_seed=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def lerobot_client(vla=None, server_checkpoint=None, mode="ok", ark=False):
    vla = vla or spec()
    command = [sys.executable, str(FAKES / "fake_lerobot_server.py"), "--checkpoint", server_checkpoint or vla.checkpoint,
               "--horizon", str(vla.horizon), "--mode", mode, "--task", "Pick up bottle"]
    if ark:
        command += ["--image-key", "observation.images.head"]
    return StdioPolicyClient("bottle", command, make_contract(vla), execute_steps=vla.execute_steps)


def test_joint_contract_is_named_and_identity_mapped():
    assert len(JOINT_NAMES) == 28
    assert JOINT_NAMES[7] == "right_shoulder_pitch_joint"
    assert JOINT_NAMES[21] == "right_hand_thumb_0_joint"
    assert BOTTLE_TO_CANONICAL["right_ee_0"] == "right_hand_thumb_0_joint"
    assert list(BOTTLE_TO_CANONICAL) == list(BOTTLE_DATASET_ORDER)


def test_jpeg_size_parser():
    assert jpeg_size(JPEG) == (1280, 720)
    assert jpeg_size(b"not a jpeg") is None
    with pytest.raises(ValueError, match="1280x720"):
        encode_request(fake_jpeg(640, 480), STATE, 0)


def test_lerobot_contract_roundtrip():
    client = lerobot_client()
    try:
        first = client.predict(JPEG, STATE, frame_stamp=12.5)
        second = client.predict(JPEG, STATE)
        assert first.actions.shape == (50, 28) and first.chunk.shape == (50, 28)
        assert first.frame_stamp == 12.5
        assert np.allclose(first.actions[0], STATE.astype(np.float32))
        assert second.meta["request_id"] == 1
    finally:
        client.close()


def test_lerobot_rejects_wrong_checkpoint():
    with pytest.raises(ContractError, match="checkpoint"):
        lerobot_client(server_checkpoint="/ckpt/other")


def test_failed_exchange_breaks_the_client():
    client = lerobot_client(mode="bad_motor")
    with pytest.raises(ContractError, match="motor"):
        client.predict(JPEG, STATE)
    assert not client.healthy
    with pytest.raises(ContractError, match="broken"):
        client.predict(JPEG, STATE)
    client = lerobot_client(mode="garbage_second")
    client.predict(JPEG, STATE)
    with pytest.raises(ContractError, match="JSON"):
        client.predict(JPEG, STATE)
    assert not client.healthy


def test_bad_input_keeps_the_client_usable():
    client = lerobot_client()
    try:
        with pytest.raises(ValueError):
            client.predict(JPEG, np.zeros(27))
        assert client.healthy
        assert client.predict(JPEG, STATE).meta["request_id"] == 0
    finally:
        client.close()


def test_wrong_task_is_rejected():
    client = lerobot_client(spec(prompt="Pick up box"))
    with pytest.raises(ContractError, match="task"):
        client.predict(JPEG, STATE)


def test_arkestrator_contract_requires_profile_fields():
    vla = spec(contract="arkestrator_stdio_v1")
    with pytest.raises(ContractError, match="task differs|image_key differs"):
        lerobot_client(vla)  # plain bottle server: no profile fields in READY
    client = lerobot_client(vla, ark=True)
    try:
        assert client.predict(JPEG, STATE).actions.shape == (50, 28)
    finally:
        client.close()
    with pytest.raises(ContractError, match="image_key"):
        lerobot_client(spec(contract="arkestrator_stdio_v1", image_key="observation.images.head_camera"), ark=True)


def sha_identity(jpeg):
    return hashlib.sha256(jpeg).hexdigest()


def box_client(mode="ok"):
    vla = spec(contract="box_stdio_v1", checkpoint="/ckpt/box", prompt="Pick up box", execute_steps=10)
    command = [sys.executable, str(FAKES / "fake_box_server.py"), "--checkpoint", "/ckpt/box", "--mode", mode]
    return StdioPolicyClient("box", command, make_contract(vla, rgb_sha256=sha_identity), execute_steps=10)


def test_box_contract_executes_only_ten_steps():
    client = box_client()
    try:
        prediction = client.predict(JPEG, STATE)
        assert prediction.chunk.shape == (50, 28)
        assert prediction.actions.shape == (10, 28)
        assert np.array_equal(prediction.actions, prediction.chunk[:10])
        assert client.predict(JPEG, STATE).meta["request_id"] == 1
    finally:
        client.close()


def test_box_contract_rejects_inconsistent_head():
    client = box_client("bad_head")
    with pytest.raises(ContractError, match="head"):
        client.predict(JPEG, STATE)
    assert not client.healthy


def test_box_contract_guards_its_settings():
    with pytest.raises(ValueError):
        make_contract(spec(contract="box_stdio_v1", prompt="Pick up bottle", execute_steps=10))
    with pytest.raises(ValueError):
        make_contract(spec(contract="box_stdio_v1", prompt="Pick up box", execute_steps=50))
    with pytest.raises(ValueError):
        BoxStdioContract("/ckpt/box", 10)
    assert isinstance(make_contract(spec()), LerobotStdioContract)


def test_encode_request_validates_state():
    with pytest.raises(ValueError):
        encode_request(JPEG, np.zeros(27), 0)
    bad = STATE.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        encode_request(JPEG, bad, 0)


class Recorder:
    def __init__(self, profile, events):
        self.profile_name = profile.name
        self.execute_steps = 1
        self.closed = False
        self.healthy = True
        self.events = events
        events.append(("load", profile.name))

    def predict(self, jpeg, state, frame_stamp=None):
        raise NotImplementedError

    def close(self):
        self.closed = True
        self.events.append(("close", self.profile_name))


def wait_for(predicate, limit=3.0):
    deadline = time.time() + limit
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_pool_closes_old_model_before_loading_next():
    events = []

    def factory(profile):
        if profile.name == "broken":
            raise RuntimeError("CUDA out of memory")
        return Recorder(profile, events)

    pool = PolicyPool(factory, resident=1)
    bottle, box, broken = (SimpleNamespace(name=n) for n in ("bottle", "box", "broken"))
    pool.acquire(bottle, wait=lambda: None)
    pool.prefetch(box)
    assert not wait_for(lambda: ("load", "box") in events, limit=0.3), "must not load while bottle is in use"
    pool.release(bottle)
    assert wait_for(lambda: ("load", "box") in events)
    assert events.index(("close", "bottle")) < events.index(("load", "box"))
    assert wait_for(lambda: pool.status() == {"box": "ready"})
    with pytest.raises(RecoverableError) as err:
        pool.acquire(broken, wait=lambda: time.sleep(0.01))
    assert err.value.reason == "vla_unavailable"


def test_pool_reloads_a_broken_client():
    events = []
    pool = PolicyPool(lambda p: Recorder(p, events), resident=1)
    bottle = SimpleNamespace(name="bottle")
    client = pool.acquire(bottle, wait=lambda: None)
    pool.release(bottle)
    client.healthy = False
    assert pool.status() == {"bottle": "broken"}
    fresh = pool.acquire(bottle, wait=lambda: time.sleep(0.01))
    assert fresh is not client and fresh.healthy
    assert wait_for(lambda: client.closed)
    assert threading.active_count() >= 1
