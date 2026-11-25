"""The 28-joint G1 + Dex3 action/state contract shared by every π0.5 profile.

Canonical order = named joints below (box ``g1_contract.JOINT_NAMES``). The bottle dataset
uses ``left_arm_0..right_ee_6``; GR00T-pickup-bottle ``bottle/inference/policy.py`` asserts that
index i of that order is ``JOINT_NAMES[i]``, so the mapping is the identity. Units: radians,
absolute joint positions, 15 Hz.
"""

from __future__ import annotations

ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")

JOINT_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_JOINTS) + tuple(
    f"{side}_hand_{joint}_joint" for side in ("left", "right") for joint in HAND_JOINTS
)

BOTTLE_DATASET_ORDER = tuple(
    f"{part}_{i}" for part in ("left_arm", "right_arm", "left_ee", "right_ee") for i in range(7)
)

# Identity by construction; kept explicit so a reviewer can see the mapping.
BOTTLE_TO_CANONICAL = dict(zip(BOTTLE_DATASET_ORDER, JOINT_NAMES, strict=True))

FPS = 15
IMAGE_SIZE = (1280, 720)  # width, height of the head camera RGB both checkpoints were trained on
UNITS = "radian"
REPRESENTATION = "absolute_joint_position"

assert len(JOINT_NAMES) == 28 and len(set(JOINT_NAMES)) == 28
assert len(BOTTLE_DATASET_ORDER) == 28
