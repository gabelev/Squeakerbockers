"""Pose model wrapper.

Contract:
    PoseModel(model_name).detect(frame)              -> list[Detection]
    PoseModel(model_name).detect_with_overlay(frame) -> (list[Detection], np.ndarray)

Backed by ultralytics yolo11n-pose. Load weights once at construction —
the model is heavy; construct from a module-level singleton, not per
connection. See SPEC §5 (pose.py) and §3 (handler copy()).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO-17 keypoint names in the order YOLO returns them.
KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass
class Detection:
    keypoints: dict  # name -> (x, y, confidence)
    bbox: tuple      # (x1, y1, x2, y2)
    center: tuple    # (cx, cy)


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype == np.float32:
        scale = 255.0 if float(frame.max()) <= 1.0 else 1.0
        return np.clip(frame * scale, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


class PoseModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = YOLO(model_name)

    def warmup(self, h: int = 480, w: int = 640) -> None:
        """Run one inference on a blank frame to pay JIT/compile cost up front."""
        self._model.predict(np.zeros((h, w, 3), dtype=np.uint8), verbose=False)

    def _result_to_detections(self, result) -> list[Detection]:
        if result.keypoints is None or result.keypoints.data is None:
            return []
        kpts = result.keypoints.data.cpu().numpy()  # [N, 17, 3] -> (x, y, conf)
        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else None
        )
        dets: list[Detection] = []
        for i, person in enumerate(kpts):
            named = {
                KEYPOINT_NAMES[j]: (float(person[j, 0]), float(person[j, 1]), float(person[j, 2]))
                for j in range(len(KEYPOINT_NAMES))
            }
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
            else:
                x1 = y1 = x2 = y2 = 0.0
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dets.append(Detection(keypoints=named, bbox=(x1, y1, x2, y2), center=(cx, cy)))
        return dets

    def detect(self, frame: np.ndarray, conf: float = 0.15) -> list[Detection]:
        results = self._model(_as_uint8(frame), conf=conf, verbose=False)
        return self._result_to_detections(results[0])

    def detect_with_overlay(
        self, frame: np.ndarray, conf: float = 0.15
    ) -> tuple[list[Detection], np.ndarray]:
        img = _as_uint8(frame)
        results = self._model(img, conf=conf, verbose=False)
        detections = self._result_to_detections(results[0])
        annotated = results[0].plot()  # uint8, same colorspace as input
        return detections, annotated
