"""Fake ``lerobot_stdio_v1`` server (same framing as GR00T-pickup-bottle bottle/inference/server.py)."""

import argparse
import hashlib
import io
import json
import struct
import sys

import numpy as np

ORDER = [f"{p}_{i}" for p in ("left_arm", "right_arm", "left_ee", "right_ee") for i in range(7)]


def emit(kind, value):
    print(kind + " " + json.dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--task", default="Pick up bottle")
    p.add_argument("--mode", default="ok")
    p.add_argument("--image-key", default=None, help="emit arkestrator_stdio_v1 READY fields")
    p.add_argument("--noise-seed", type=int, default=None)
    args = p.parse_args()
    ready = {"policy_type": "pi05", "model_type": "pi05", "checkpoint": args.checkpoint, "fps": 15,
             "horizon": args.horizon, "joint_order": ORDER, "motor_commands_sent": False}
    if args.image_key:
        ready.update(task=args.task, image_key=args.image_key, noise_seed=args.noise_seed, chunk_size=50)
    emit("READY", ready)
    served = 0
    while True:
        header = sys.stdin.buffer.read(4)
        if not header:
            return
        (size,) = struct.unpack("!I", header)
        data = sys.stdin.buffer.read(size)
        with np.load(io.BytesIO(data), allow_pickle=False) as request:
            jpeg = request["image_rgb_jpeg"].tobytes()
            state = request["state"]
            request_id = int(request["request_id"])
        actions = np.tile(state, (args.horizon, 1)).tolist()
        reply = {
            "checkpoint": args.checkpoint, "model_type": "pi05", "task": args.task,
            "joint_order": ORDER, "units": "radian", "representation": "absolute_joint_position",
            "motor_commands_sent": args.mode == "bad_motor", "policy_type": "pi05",
            "request_id": request_id, "image_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
            "fps": 15, "horizon": args.horizon, "actions": actions,
        }
        served += 1
        if args.mode == "garbage_second" and served == 2:
            print("RESULT {not json", flush=True)
            continue
        emit("RESULT", reply)


if __name__ == "__main__":
    main()
