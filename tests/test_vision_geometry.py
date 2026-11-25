import time

import numpy as np
import pytest

from arkestrator.errors import RecoverableError
from arkestrator.perception.base import Perception, RawDetection
from arkestrator.robot.base import CameraFrame
from arkestrator.types import Pose2D
from arkestrator.vision import CameraExtrinsics, PerceptionVision, bbox_depth_point

INTRINSICS = {"fx": 600.0, "fy": 600.0, "ppx": 640.0, "ppy": 360.0}
# Optical (x right, y down, z forward) → torso (x forward, y left, z up), camera at 1 m height.
LEVEL_CAMERA = CameraExtrinsics(((0, 0, 1), (-1, 0, 0), (0, -1, 0)), (0.0, 0.0, 1.0))


def test_bbox_depth_point_uses_median_of_central_half():
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    depth[300:310, 700:710] = 9.0  # outlier inside the box
    point = bbox_depth_point(depth, (640, 300, 760, 420), INTRINSICS)
    assert point == pytest.approx(((700 - 640) * 2.0 / 600, 0.0, 2.0), abs=1e-6)


def test_bbox_depth_point_rejects_missing_depth():
    depth = np.zeros((720, 1280), dtype=np.float32)
    assert bbox_depth_point(depth, (600, 300, 700, 400), INTRINSICS) is None
    assert bbox_depth_point(np.ones((720, 1280)), (700, 300, 600, 400), INTRINSICS) is None


def test_extrinsics_to_torso():
    assert LEVEL_CAMERA.to_torso((0.3, 0.0, 2.0)) == pytest.approx((2.0, -0.3, 1.0))


class OneBox(Perception):
    def transcribe(self, wav): return ""
    def parse_command(self, text, catalog): return None
    def detect(self, jpeg, labels): return [RawDetection("bottle", 0.8, (640, 300, 760, 420))]
    def verify(self, jpeg, bbox, question): return True
    def embed_person(self, jpeg, bbox): return None
    def health(self): return {}


class Frames:
    def __init__(self):
        self.number = 1

    def frames(self):
        depth = np.full((720, 1280), 2.0, dtype=np.float32)
        return {"head": CameraFrame("head", self.number, 0.0, Pose2D(1.0, 0.0, 0.0), b"jpeg", depth, INTRINSICS)}


def wait_fresh(vision, labels, limit=2.0):
    deadline = time.time() + limit
    while time.time() < deadline:
        sighting = vision.poll(labels)
        if sighting.fresh:
            return sighting
        time.sleep(0.01)
    raise AssertionError("no fresh detections")


def test_perception_vision_lifts_boxes_in_background():
    robot = Frames()
    vision = PerceptionVision(robot, OneBox(), {"head": LEVEL_CAMERA}, ["head"])
    try:
        assert not vision.poll(["bottle"]).fresh  # first poll only states the question
        sighting = wait_fresh(vision, ["bottle"])
        assert len(sighting.value) == 1
        detection = sighting.value[0]
        assert detection.position == pytest.approx((2.0, -0.2, 1.0))
        assert detection.world_xy == pytest.approx((3.0, -0.2))
        assert not vision.poll(["bottle"]).fresh  # same frame is not reported twice
        robot.number += 1
        assert wait_fresh(vision, ["bottle"]).value[0].frame_number == 2
        assert vision.verify(detection, "?") is True
    finally:
        vision.close()


class SlowBox(OneBox):
    def __init__(self):
        self.calls = 0

    def detect(self, jpeg, labels):
        self.calls += 1
        time.sleep(0.2)
        return super().detect(jpeg, labels)


class Failing(OneBox):
    def detect(self, jpeg, labels):
        raise ConnectionError("perceptiond is down")


def test_slow_model_never_blocks_poll_and_errors_surface():
    vision = PerceptionVision(Frames(), SlowBox(), {"head": LEVEL_CAMERA}, ["head"])
    try:
        vision.poll(["bottle"])
        started = time.time()
        for _ in range(5):
            vision.poll(["bottle"])
        assert time.time() - started < 0.05
    finally:
        vision.close()
    vision = PerceptionVision(Frames(), Failing(), {"head": LEVEL_CAMERA}, ["head"])
    try:
        vision.poll(["bottle"])
        deadline = time.time() + 2
        with pytest.raises(RecoverableError, match="perception_unavailable"):
            while time.time() < deadline:
                vision.poll(["bottle"])
                time.sleep(0.01)
    finally:
        vision.close()
