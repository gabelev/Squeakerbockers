"""Movement features from pose detections.

Contract:
    FeatureExtractor()(detections) -> Features

Per person from ankle/foot keypoints over a rolling window: motion_energy,
foot_velocity, pivot_sharpness (the screech signal — matters most), pan,
proximity. Then aggregate across people into one Features. Implementation
lands at P4.
"""
from __future__ import annotations

from collections import deque

from state import Features


class FeatureExtractor:
    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._history: deque = deque(maxlen=window)

    def __call__(self, detections: list) -> Features:
        return Features()
