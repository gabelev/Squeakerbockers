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
RAVE_MODEL_PATH = PROJECT_ROOT / "models" / "birds_dawnchorus_b2048_r48000_z8.ts"
POSE_MODEL = "yolo11n-pose.pt"

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
LATENT_SMOOTH_ALPHA = 0.85  # 0 = no smoothing, 1 = frozen

# UI defaults --------------------------------------------------------
UI_SENSITIVITY = 1.0
UI_SQUEAK_INTENSITY = 1.0
UI_DRY_WET = 1.0
UI_NOISE_FLOOR = NOISE_FLOOR
