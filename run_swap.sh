#!/usr/bin/env bash
# Launch the face-swap Gradio app locally. Sets LD_LIBRARY_PATH so the headless
# OpenCV wheel can find libGL.so.1 (extracted to ~/.local/gl-shim during setup,
# since this box has no system libgl1 and no sudo). On an HF Space, packages.txt
# installs libgl1 system-wide and this shim is unnecessary.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/gl-shim:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
exec uv run python app_swap.py "$@"
