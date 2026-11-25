"""π0.5 stdio client for three server contracts.

Wire format (all servers): request = 4-byte big-endian length + ``np.savez`` with
``image_rgb_jpeg`` (uint8 JPEG bytes, 1280×720), ``state`` (float32[28], canonical order,
radians) and ``request_id`` (int64, sequential from 0). Replies are text lines ``READY {json}``
once, then ``RESULT {json}`` per request. Nothing here sends motor commands; every reply is
checked against the profile and rejected (never repaired) on any mismatch. After any failure
in the middle of an exchange the client is marked broken and closed: the server's request
sequence can no longer be trusted, so the pool reloads it.

* ``lerobot_stdio_v1`` — GR00T-pickup-bottle ``bottle/inference/server.py`` as is. Checks:
  policy/model type, checkpoint, fps 15, horizon, joint order, ``motor_commands_sent is False``,
  JPEG sha256, task (from RESULT).
* ``arkestrator_stdio_v1`` — our ``vla_servers/pi05_stdio_server.py``: the same protocol, plus
  READY must state the profile's ``task``, ``image_key``, ``noise_seed`` and a chunk at least
  as long as the horizon, so a server started with another profile is refused at load.
* ``box_stdio_v1`` — vzyat-korobku-peremestit ``pi05/pi05_fixed_1000_stdio_20261029.py``:
  ``robot_commands_sent == 0``, pi05, seed 123 fixed per request, n_action_steps 10 of a
  50-step chunk, joint names, units, representation, fps, schema, checkpoint, sha256 of the
  *decoded* RGB.

``Prediction.actions`` holds only the steps the profile may execute (``execute_steps``); the
whole chunk is in ``Prediction.chunk``. The backend must treat a chunk as expired at step index
``len(actions)`` (box: 10, as in ``model_box_guarded_run``), never play the rest open-loop.
"""

from __future__ import annotations

import abc
import hashlib
import io
import json
import os
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .contract import BOTTLE_DATASET_ORDER, FPS, IMAGE_SIZE, JOINT_NAMES, REPRESENTATION, UNITS

MAX_REQUEST_BYTES = 4_000_000  # box server accepts < 4 MB; bottle server 8 MiB
MAX_LINE_BYTES = 1 << 20


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    actions: np.ndarray  # (execute_steps, 28) canonical order, radians — the only executable steps
    chunk: np.ndarray  # (horizon, 28) whole predicted chunk, for logging
    execute_steps: int
    roundtrip_s: float
    frame_stamp: Optional[float] = None  # camera acquisition time of the input frame
    meta: dict[str, Any] = field(default_factory=dict)


class PolicyClient(abc.ABC):
    profile_name: str
    execute_steps: int

    @property
    def healthy(self) -> bool:
        return True

    @abc.abstractmethod
    def predict(self, jpeg: bytes, state: Sequence[float], frame_stamp: Optional[float] = None) -> Prediction: ...

    @abc.abstractmethod
    def close(self) -> None: ...


def jpeg_size(jpeg: bytes) -> Optional[tuple[int, int]]:
    """(width, height) from the first SOF marker, without decoding; None if not found."""
    data = bytes(jpeg)
    if not data.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    return None


def decode_rgb_sha256(jpeg: bytes) -> str:
    """sha256 of the RGB pixels the box server sees (cv2.imdecode → BGR2RGB)."""
    import cv2  # available in the bottle-controller env

    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def encode_request(jpeg: bytes, state: Sequence[float], request_id: int) -> bytes:
    if not isinstance(jpeg, (bytes, bytearray)) or not jpeg:
        raise ValueError("Expected JPEG bytes")
    if jpeg_size(jpeg) != IMAGE_SIZE:
        raise ValueError(f"Expected a {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} JPEG, got {jpeg_size(jpeg)}")
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (28,) or not np.isfinite(state_array).all():
        raise ValueError("State must be 28 finite joint positions")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        image_rgb_jpeg=np.frombuffer(bytes(jpeg), dtype=np.uint8),
        state=state_array,
        request_id=np.int64(request_id),
    )
    data = buffer.getvalue()
    if not 100 < len(data) < MAX_REQUEST_BYTES:
        raise ValueError("Request size outside the server limits")
    return struct.pack("!I", len(data)) + data


def read_line(stream, kind: str) -> dict[str, Any]:
    line = stream.readline(MAX_LINE_BYTES)
    prefix = (kind + " ").encode()
    if not line.endswith(b"\n") or not line.startswith(prefix):
        raise ContractError(f"Expected {kind} line from the policy server")
    try:
        value = json.loads(line[len(prefix):])
    except ValueError as exc:
        raise ContractError(f"{kind} payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{kind} payload is not an object")
    return value


def _actions(value, horizon: int, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{what}: actions are not numeric") from exc
    if array.shape != (horizon, 28) or not np.isfinite(array).all():
        raise ContractError(f"{what}: expected finite ({horizon}, 28) actions, got {array.shape}")
    return array


def _reject(kind: str, problems: list[str]) -> None:
    if problems:
        raise ContractError(f"{kind} rejected: " + ", ".join(problems))


class LerobotStdioContract:
    """``lerobot_stdio_v1``; with ``ready_fields`` it is ``arkestrator_stdio_v1``."""

    def __init__(self, checkpoint: str, horizon: int, prompt: str, ready_fields: Optional[dict[str, Any]] = None):
        self.checkpoint = checkpoint
        self.horizon = horizon
        self.prompt = prompt
        self.ready_fields = dict(ready_fields or {})

    def check_ready(self, ready: dict[str, Any]) -> None:
        problems = []
        if ready.get("policy_type") != "pi05" or ready.get("model_type") != "pi05":
            problems.append("policy is not pi05")
        if ready.get("checkpoint") != self.checkpoint:
            problems.append("checkpoint differs")
        if ready.get("fps") != FPS:
            problems.append("fps differs")
        if ready.get("horizon") != self.horizon:
            problems.append("horizon differs")
        if ready.get("joint_order") != list(BOTTLE_DATASET_ORDER):
            problems.append("joint order differs")
        if ready.get("motor_commands_sent") is not False:
            problems.append("server claims motor commands")
        for key, expected in self.ready_fields.items():
            if key not in ready or ready[key] != expected:
                problems.append(f"{key} differs")
        if self.ready_fields:
            chunk = ready.get("chunk_size")
            if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < self.horizon:
                problems.append("chunk_size shorter than horizon")
        _reject("READY", problems)

    def check_result(self, reply: dict[str, Any], jpeg: bytes, request_id: int) -> np.ndarray:
        problems = []
        if reply.get("request_id") != request_id:
            problems.append("request_id differs")
        if reply.get("image_jpeg_sha256") != hashlib.sha256(jpeg).hexdigest():
            problems.append("image hash differs")
        if reply.get("checkpoint") != self.checkpoint:
            problems.append("checkpoint differs")
        if reply.get("joint_order") != list(BOTTLE_DATASET_ORDER):
            problems.append("joint order differs")
        if reply.get("units") != UNITS or reply.get("representation") != REPRESENTATION:
            problems.append("units/representation differ")
        if reply.get("fps") != FPS or reply.get("horizon") != self.horizon:
            problems.append("fps/horizon differ")
        if reply.get("policy_type") != "pi05" or reply.get("model_type") != "pi05":
            problems.append("policy is not pi05")
        if reply.get("motor_commands_sent") is not False:
            problems.append("server claims motor commands")
        if reply.get("task") != self.prompt:
            problems.append("task differs")
        _reject("RESULT", problems)
        # BOTTLE_DATASET_ORDER[i] is JOINT_NAMES[i]: no reordering needed.
        return _actions(reply.get("actions"), self.horizon, "RESULT")


class BoxStdioContract:
    """``box_stdio_v1`` (original box server, checks as in box ``model_box_policy.Predictor``)."""

    N_ACTION_STEPS = 10
    CHUNK = 50
    NOISE_SEED = 123

    def __init__(self, checkpoint: str, horizon: int, rgb_sha256: Callable[[bytes], str] = decode_rgb_sha256):
        if horizon != self.CHUNK:
            raise ValueError("box_stdio_v1 returns full 50-step chunks; set horizon: 50")
        self.checkpoint = checkpoint
        self.horizon = horizon
        self.rgb_sha256 = rgb_sha256

    def check_ready(self, ready: dict[str, Any]) -> None:
        problems = []
        if ready.get("robot_commands_sent") != 0:
            problems.append("server claims robot commands")
        if ready.get("policy_type") != "pi05":
            problems.append("policy is not pi05")
        if ready.get("horizon") != self.N_ACTION_STEPS:
            problems.append("n_action_steps differs")
        if ready.get("checkpoint") != self.checkpoint:
            problems.append("checkpoint differs")
        if ready.get("noise_seed") != self.NOISE_SEED or ready.get("randomness_mode") != "fixed_per_request":
            problems.append("noise seed/mode differ")
        if not isinstance(ready.get("username"), str) or not ready.get("username"):
            problems.append("username missing")
        _reject("READY", problems)

    def check_result(self, reply: dict[str, Any], jpeg: bytes, request_id: int) -> np.ndarray:
        problems = []
        if reply.get("request_id") != request_id:
            problems.append("request_id differs")
        if reply.get("robot_commands_sent") != 0:
            problems.append("server claims robot commands")
        if reply.get("image_rgb_sha256") != self.rgb_sha256(jpeg):
            problems.append("image hash differs")
        if reply.get("checkpoint") != self.checkpoint:
            problems.append("checkpoint differs")
        if reply.get("schema") != "g1_dex3_prediction_v1" or reply.get("mode") != "prediction_only":
            problems.append("schema/mode differ")
        if reply.get("joint_names") != list(JOINT_NAMES):
            problems.append("joint names differ")
        if reply.get("units") != UNITS or reply.get("representation") != REPRESENTATION:
            problems.append("units/representation differ")
        if reply.get("fps") != FPS or reply.get("policy_type") != "pi05":
            problems.append("fps/policy differ")
        if reply.get("n_action_steps") != self.N_ACTION_STEPS or reply.get("training_chunk_size") != self.CHUNK:
            problems.append("chunk sizes differ")
        if reply.get("noise_seed") != self.NOISE_SEED:
            problems.append("noise seed differs")
        _reject("RESULT", problems)
        head = _actions(reply.get("actions"), self.N_ACTION_STEPS, "RESULT.actions")
        full = _actions(reply.get("full_actions"), self.CHUNK, "RESULT.full_actions")
        if not np.array_equal(full[: self.N_ACTION_STEPS], head):
            raise ContractError("RESULT rejected: actions are not the head of full_actions")
        return full


def make_contract(vla, **kwargs):
    """Contract for a profile's ``VlaSpec``."""
    if not vla.checkpoint:
        raise ValueError("VLA checkpoint is not set")
    if vla.contract == "lerobot_stdio_v1":
        return LerobotStdioContract(vla.checkpoint, vla.horizon, vla.prompt)
    if vla.contract == "arkestrator_stdio_v1":
        fields = {"task": vla.prompt, "image_key": vla.image_key, "noise_seed": vla.noise_seed}
        return LerobotStdioContract(vla.checkpoint, vla.horizon, vla.prompt, ready_fields=fields)
    if vla.contract == "box_stdio_v1":
        if vla.prompt != "Pick up box" or vla.execute_steps != BoxStdioContract.N_ACTION_STEPS:
            raise ValueError("The original box server runs only 'Pick up box' with execute_steps 10")
        return BoxStdioContract(vla.checkpoint, vla.horizon, **kwargs)
    raise ValueError(f"Unknown VLA contract {vla.contract!r}")


class StdioPolicyClient(PolicyClient):
    """Runs one policy server as a child process (own process group) and speaks its contract."""

    def __init__(
        self,
        profile_name: str,
        command: Sequence[str],
        contract,
        execute_steps: int,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        stderr_path: Optional[str] = None,
    ):
        self.profile_name = profile_name
        self.contract = contract
        self.execute_steps = execute_steps
        self.broken: Optional[str] = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._stderr = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        try:
            self.ready = read_line(self._process.stdout, "READY")
            self.contract.check_ready(self.ready)
        except Exception:
            self.close()
            raise

    @property
    def healthy(self) -> bool:
        return self.broken is None and self._process.poll() is None

    def predict(self, jpeg: bytes, state: Sequence[float], frame_stamp: Optional[float] = None) -> Prediction:
        with self._lock:
            if self.broken is not None:
                raise ContractError(f"Policy client is broken: {self.broken}")
            if self._process.poll() is not None:
                raise ContractError(f"Policy server exited with code {self._process.returncode}")
            request_id = self._sequence
            request = encode_request(jpeg, state, request_id)  # input errors leave the client usable
            started = time.monotonic()
            try:
                self._process.stdin.write(request)
                self._process.stdin.flush()
                reply = read_line(self._process.stdout, "RESULT")
                chunk = self.contract.check_result(reply, jpeg, request_id)
            except Exception as exc:
                self.broken = f"{type(exc).__name__}: {exc}"
                self._close_locked()
                raise ContractError(f"Policy exchange failed, client closed: {self.broken}") from exc
            self._sequence += 1
            if len(chunk) < self.execute_steps:
                raise ContractError("chunk shorter than execute_steps")
            return Prediction(
                actions=chunk[: self.execute_steps],
                chunk=chunk,
                execute_steps=self.execute_steps,
                roundtrip_s=time.monotonic() - started,
                frame_stamp=frame_stamp,
                meta={k: v for k, v in reply.items() if k not in ("actions", "full_actions")},
            )

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    process.wait()
        if self._stderr not in (None, subprocess.DEVNULL):
            self._stderr.close()
            self._stderr = subprocess.DEVNULL

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(self._process.pid, sig)  # the server and any workers it spawned
        except ProcessLookupError:
            pass
