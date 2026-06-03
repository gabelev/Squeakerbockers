"""Movement features from pose detections.

Per tracked person, maintain a short rolling history of ankle positions
and body center. From that derive:

- motion_energy: speed of the body center
- foot_velocity: max recent ankle speed
- pivot_sharpness: max ankle acceleration magnitude (the screech signal)
- pan: weighted body-center x mapped to [0, 1]
- proximity: bbox height
- person_count: number of currently tracked detections

Aggregated across people into one Features. SPEC §5 (features.py).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from state import Features


class _TrackHistory:
    """Rolling history of one tracked person's salient keypoints."""

    def __init__(self, window: int) -> None:
        self.left_ankle: deque[tuple[float, float]] = deque(maxlen=window)
        self.right_ankle: deque[tuple[float, float]] = deque(maxlen=window)
        self.center: deque[tuple[float, float]] = deque(maxlen=window)


def _kpt_xy(kpt: tuple[float, float, float]) -> tuple[float, float] | None:
    x, y, conf = kpt
    return (x, y) if conf > 0.2 else None


def _hop(points: deque, n: int = 1) -> tuple[float, float] | None:
    """Difference between points[-1] and points[-1-n], or None if not enough."""
    if len(points) < n + 1:
        return None
    x1, y1 = points[-1]
    x0, y0 = points[-1 - n]
    return (x1 - x0, y1 - y0)


class FeatureExtractor:
    def __init__(self, window: int = 5, default_frame_width: int = 1280) -> None:
        self.window = window
        self.default_frame_width = default_frame_width
        self._histories: dict[int, _TrackHistory] = {}

    def __call__(self, detections, frame_width: int | None = None) -> Features:
        width = frame_width or self.default_frame_width
        if not detections:
            self._histories.clear()
            return Features()

        seen_ids = {d.track_id for d in detections if d.track_id is not None}
        # Prune histories for tracks that vanished.
        self._histories = {
            tid: h for tid, h in self._histories.items() if tid in seen_ids
        }

        motions: list[float] = []
        foot_vs: list[float] = []
        pivots: list[float] = []
        xs: list[float] = []
        prox: list[float] = []

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            hist = self._histories.setdefault(tid, _TrackHistory(self.window))
            la = _kpt_xy(det.keypoints["left_ankle"])
            ra = _kpt_xy(det.keypoints["right_ankle"])
            if la is not None:
                hist.left_ankle.append(la)
            if ra is not None:
                hist.right_ankle.append(ra)
            hist.center.append(det.center)

            # Body center speed.
            d_center = _hop(hist.center)
            motion = math.hypot(*d_center) if d_center else 0.0

            # Per-ankle velocity and acceleration.
            la_v = _hop(hist.left_ankle)
            ra_v = _hop(hist.right_ankle)
            la_v_prev = _hop(hist.left_ankle, n=2)
            ra_v_prev = _hop(hist.right_ankle, n=2)

            foot_v_left = math.hypot(*la_v) if la_v else 0.0
            foot_v_right = math.hypot(*ra_v) if ra_v else 0.0

            la_accel = (
                math.hypot(la_v[0] - la_v_prev[0], la_v[1] - la_v_prev[1])
                if la_v and la_v_prev
                else 0.0
            )
            ra_accel = (
                math.hypot(ra_v[0] - ra_v_prev[0], ra_v[1] - ra_v_prev[1])
                if ra_v and ra_v_prev
                else 0.0
            )

            motions.append(motion)
            foot_vs.append(max(foot_v_left, foot_v_right))
            pivots.append(max(la_accel, ra_accel))
            xs.append(det.center[0])
            prox.append(max(0.0, det.bbox[3] - det.bbox[1]))

        if not motions:
            return Features(person_count=len(detections))

        # Aggregate.
        max_prox = max(prox) if prox else 0.0
        if max_prox > 0.0:
            weights = np.array(prox) / max_prox
            wsum = float(weights.sum())
            mean_x = float(np.dot(weights, xs) / wsum) if wsum > 0 else float(np.mean(xs))
        else:
            mean_x = float(np.mean(xs))

        return Features(
            motion_energy=float(np.mean(motions)),
            foot_velocity=float(np.max(foot_vs)),
            pivot_sharpness=float(np.max(pivots)),
            pan=float(np.clip(mean_x / width, 0.0, 1.0)),
            proximity=float(max_prox),
            person_count=len(motions),
        )
