"""Pose model wrapper.

Contract:
    PoseModel(model_name).detect(frame: np.ndarray HxWx3) -> list[Detection]

Each Detection carries named keypoints (including ankles, feet, hips),
a bounding box, and the body center. Implementation lands at P3 against
ultralytics yolo11n-pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    keypoints: dict  # name -> (x, y, confidence)
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)


class PoseModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # loaded at P3

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []
