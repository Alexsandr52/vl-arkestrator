# vl-arkestrator instructions

Read README.md and docs/architecture.md before changing the project.

## Project boundaries
- This repo owns the session daemon, task FSM, navigation skills, grasp retry/verification,
  VLA client and profiles, perception protocol, voice handling and the web UI.
- Robot control primitives (DDS owner `SessionPump`, arm_sdk takeover/release, preparation
  stances, tactile guards, IK, camera streams) belong to `GR00T-pickup-bottle`. They are used
  from a pinned checkout, never edited here. Changes there go to that repo with their own ADR.
- The box task (`vzyat-korobku-peremestit`) is read-only for this repo: its checkpoint and
  venv may be used, its code, installation and evidence are not changed.
- Everything that touches the robot, cameras or the GPU runs on `g1-gpu`. The laptop runs only
  the mock backend, tests and the browser.

## Hardware
- Do not send hardware commands while testing. Fresh user authorization is required for each
  physical session and for GPU work (model loading, VRAM measurements).
- Keep command lifecycle, STOP, arm ownership, temperature/mode/NaN checks, speed and URDF
  limits and the release protocol (`return_to_factory_stand.json`, `release_supported.json`)
  unchanged without a requested change. Never reuse stale release or request files.
- One motor owner at a time: share the `.xr/xr_launcher.lock` and `.xr/robot_motor.lock` locks
  with the bottle and box projects. Never kill other users' GPU processes.
- Do not add new stop conditions, timeouts or stage gates unless the user requests them. The
  grasp retry limit (`max_attempts`) is part of the requested retry feature and stays configurable.
- STOP is the UI button only. Voice "стоп"/"отмена" cancels the task through the normal recovery.

## Data and contracts
- Joints by name (`arkestrator/vla/contract.py::JOINT_NAMES`), radians, meters, seconds.
  Check limits and timestamps; reject stale chunks instead of clipping.
- Keep measured state separate from sent commands. A sim/mock success or an SDK release is not
  a physical success: `physical_grasp_success` is set by the operator only. Never rewrite evidence.

## Never commit
Datasets, checkpoints, weights, outputs, raw frames, audio, logs, `config/local.json`,
`config/taught/`, keys, passwords.

## Verification
- Safe offline checks: `.venv/bin/python -m pytest -q` (mock backend, fake VLA servers; no
  hardware, no GPU, no network).
- "Ready" for a phase means: offline tests pass, `vla-contract-reviewer` / `robot-safety-auditor`
  reviewed the diff where relevant, and the user authorized the next physical step.
- Python 3.10 compatible code (the `bottle-controller` env on g1-gpu is 3.10).
