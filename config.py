"""Tunables for Squeakerbockers.

The only file you should need to change to swap models, retune the control
mapping, or tweak UI defaults.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Models -------------------------------------------------------------
# Placeholder until the trained squeak model lands. Streaming-exported
# RAVE .ts from Intelligent-Instruments-Lab/rave-models on Hugging Face.
# See SPEC.md §9 for the naming convention and alternates.
RAVE_MODEL_PATH = PROJECT_ROOT / "models" / "organ_archive_b2048_r48000_z16.ts"
POSE_MODEL = "yolo11n-pose.pt"

# Inference resolution for YOLO pose. Lower = faster + lower-latency overlay.
# Overlay still draws on the full-res input frame, so skeleton looks sharp.
# 640 (ultralytics default) is overkill at ~29ms/inference; 416 is ~15ms.
POSE_IMGSZ = 416
POSE_CONF = 0.15  # detection confidence threshold (default 0.25 is high for webcam)

# latent_dim and sample_rate are introspected from the loaded RAVE model
# at runtime (rave_engine.py). Do not hardcode them.

# Features -----------------------------------------------------------
FEATURE_WINDOW = 5
PIVOT_DECEL_WEIGHT = 1.0
PIVOT_ANGLE_WEIGHT = 1.0

# Control mapping ----------------------------------------------------
# Starting points; iterate by ear in P7.
PIVOT_GAIN = 3.0
MOTION_GAIN = 1.5
NOISE_FLOOR = 0.02
LATENT_SMOOTH_ALPHA = 0.2  # 0 = no smoothing, 1 = frozen
# Bring raw pixel-space feature values into the RAVE PCA range (~±3).
# These are empirical scales; tune in P7.
MOTION_ENERGY_SCALE = 1.0 / 25.0
PIVOT_SHARPNESS_SCALE = 1.0 / 50.0
NOISE_DIMS_SCALE = 0.1  # amplitude of low-amp noise in non-driven latent dims

# UI defaults --------------------------------------------------------
UI_SENSITIVITY = 1.0
UI_SQUEAK_INTENSITY = 1.0
UI_DRY_WET = 1.0
UI_NOISE_FLOOR = NOISE_FLOOR
