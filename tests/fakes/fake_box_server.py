"""Fake ``box_stdio_v1`` server (reply fields of vzyat-korobku-peremestit pi05_fixed_1000_stdio).

The real server hashes the decoded RGB; here "decoding" is the identity, and the test client is
given the same hash function.
"""

import argparse
import hashlib
import io
import json
import struct
import sys

import numpy as np

JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")] + [
    f"{s}_hand_{j}_joint" for s in ("left", "right") for j in (
        "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", default="ok")
    args = p.parse_args()
    print("READY " + json.dumps(dict(username="ildar", robot_commands_sent=0, policy_type="pi05", horizon=10,
                                     checkpoint=args.checkpoint, noise_seed=123,
                                     randomness_mode="fixed_per_request")), flush=True)
    sequence = 0
    while True:
        header = sys.stdin.buffer.read(4)
        if not header:
            return
        (size,) = struct.unpack("!I", header)
        with np.load(io.BytesIO(sys.stdin.buffer.read(size)), allow_pickle=False) as data:
            jpeg = data["image_rgb_jpeg"].tobytes()
            state = data["state"]
            request_id = int(data["request_id"])
        if request_id != sequence:
            raise ValueError("Unexpected request sequence")
        full = np.tile(state, (50, 1)) + np.arange(50)[:, None] * 0.001
        head = full[:10].copy()
        if args.mode == "bad_head":
            head[0, 0] += 1.0
        result = dict(schema="g1_dex3_prediction_v1", mode="prediction_only", policy_type="pi05",
                      checkpoint=args.checkpoint, joint_names=JOINTS, units="radian",
                      representation="absolute_joint_position", fps=15, inference_seconds=0.1,
                      training_chunk_size=50, n_action_steps=10, num_inference_steps=10,
                      actions=head.tolist(), full_actions=full.tolist(),
                      noise_seed=123, randomness_mode="fixed_per_request")
        print("RESULT " + json.dumps(dict(request_id=request_id, image_rgb_sha256=hashlib.sha256(jpeg).hexdigest(),
                                          robot_commands_sent=0, **result)), flush=True)
        sequence += 1


if __name__ == "__main__":
    main()
