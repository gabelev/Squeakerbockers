# Training Squeakerbockers' RAVE model on RunPod

Per SPEC §8. Trains in 2-3 days on a single RTX 4090, ~$0.34/hr Community
Cloud. Ends with one streaming-exported `.ts` file we drop into `models/`.

## Pre-flight (local)

1. **Collect dataset.** Target 1-3 hours of shoe-squeak / court foley.
   - **Pixabay** (no API; browser-download): https://pixabay.com/sound-effects/
     - Searches that pay off: `basketball court`, `sneakers`,
       `basketball squeak`, `gym sounds`. SPEC notes a ~35 min full
       basketball court recording exists — bulk material.
     - No attribution required.
   - **Freesound** (scripted; CC0 only):
     ```bash
     export FREESOUND_API_KEY=<from https://freesound.org/apiv2/apply/>

     uv run python scripts/freesound_search.py "shoe squeak"      --out shoe.csv
     uv run python scripts/freesound_search.py "basketball court" --out court.csv
     uv run python scripts/freesound_search.py "sneaker squeak"   --out sneaker.csv

     uv run python scripts/freesound_download.py shoe.csv    data/raw/freesound/
     uv run python scripts/freesound_download.py court.csv   data/raw/freesound/
     uv run python scripts/freesound_download.py sneaker.csv data/raw/freesound/
     ```
   - Drop Pixabay downloads into `data/raw/pixabay/`.

2. **Preprocess.** Normalize to mono 48kHz WAV, trim silence, peak-normalize:
   ```bash
   uv run python scripts/preprocess.py data/raw/ data/clean/
   ```
   The script reports total duration; aim for 1-3 hours.

3. **Tar it for upload.**
   ```bash
   tar -czf squeak-data.tar.gz -C data clean
   ```

## Pod spec (RunPod)

- **GPU:** 1× RTX 4090 (24 GB; RAVE v2 only needs ~8 GB)
- **Container image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (any Python 3.11 + PyTorch ≥ 2.2 image works)
- **Pod type:** Community Cloud (cheapest; preemptible — fine because we
  checkpoint to persistent volume)
- **Persistent network volume:** 50 GB, mounted at `/workspace`
- **Container disk:** default

> **Persistent volume vs container disk:** Community Cloud pods can die.
> Container disk is ephemeral — losing it means losing the run. Network
> volume survives a dropped pod. Put `runs/` and `data/` on the volume.

## Setup (inside the pod)

```bash
cd /workspace

# Upload squeak-data.tar.gz (via RunPod web UI, runpodctl, or scp
# over the pod's direct-TCP SSH endpoint), then:
tar -xzf squeak-data.tar.gz
# tar warnings about uid/gid and LIBARCHIVE.xattr.* are benign
# (macOS-created tar; files extract fine).

# Strip macOS AppleDouble metadata (._*) — RAVE preprocess will
# try to read them as audio and silently die.
find clean -name '._*' -delete

# Apt deps: RAVE preprocess shells out to ffmpeg/ffprobe.
apt-get update -qq && apt-get install -y -qq ffmpeg tmux

# Official ACIDS/IRCAM RAVE package.
# --ignore-installed pushes past the distutils-installed `blinker`
# package that ships with Ubuntu 22.04 and otherwise aborts the install.
pip install --ignore-installed acids-rave

# acids-rave pulls in a torchaudio that wants CUDA 13 (libcudart.so.13)
# but the runpod/pytorch image is CUDA 12.4. Pin both back to 2.4.1+cu124
# (they have to match each other and the image's CUDA).
pip install --no-deps torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Smoke check
rave --help
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

## Preprocess with RAVE (one-time)

Turns the clean WAVs into a fast-loading LMDB database.

```bash
cd /workspace
rave preprocess \
    --input_path data/clean/ \
    --output_path data/preprocessed/ \
    --sampling_rate 48000 \
    --channels 1
```

Takes a few minutes for 1-3 hours of audio.

## Train

Run inside a tmux session so SSH drops don't kill the training:

```bash
tmux new -s train
cd /workspace
rave train \
    --config v2 \
    --db_path data/preprocessed/ \
    --name squeakerbockers \
    --out_path runs/ \
    --gpu 0 \
    --channels 1 \
    --override SAMPLING_RATE=48000 \
    2>&1 | tee /workspace/train.log
# detach: Ctrl-b d
# reattach later: tmux attach -t train
```

`--config v2` is the same RAVE v2 architecture used by the placeholder
model. Checkpoints go to `runs/squeakerbockers/`.

Two flags that are easy to miss:

- **`--channels 1`** — `rave train --help` lists the default as `0`,
  not `1`. Omitting it makes `GeneratorV2`'s final Conv1d have 0 output
  channels, and `weight_norm` aborts with
  *"cannot reshape tensor of 0 elements into shape [0, -1]"*.
- **`--override SAMPLING_RATE=48000`** — `v2.gin` hardcodes
  `SAMPLING_RATE = 44100`. If the preprocessed LMDB is at 48000 and the
  config is at 44100, validation crashes on the first epoch with
  *"size of tensor a (236) must match the size of tensor b (237)"* —
  the multi-scale STFT comes out one frame short.

### Listen to checkpoints — don't wait for a step count

RAVE writes audio samples periodically. Listen; pull when it sounds
right. Narrow squeak timbre typically arrives in ~2-3 days.

```bash
# From your local machine (runpodctl installed from https://docs.runpod.io/runpodctl/install)
runpodctl receive /workspace/runs/squeakerbockers/version_0/audio
```

### Resume after a pod restart

If your pod dies:

1. Spin up a new pod, same spec, mount the same persistent volume.
2. Re-run the full Setup block above (the four installs — ffmpeg, the
   `--ignore-installed` acids-rave, the torch/torchaudio pin, the tmux
   session). The persistent volume keeps `runs/` and `preprocessed/`;
   the container disk does not keep the apt/pip installs.
3. Re-run the same `rave train` command — it auto-resumes from the
   latest checkpoint in `runs/squeakerbockers/`.

## Export with streaming (CRITICAL)

When the checkpoint sounds right, export with `--streaming` — without it
the model clicks and isn't realtime-safe:

```bash
rave export --run runs/squeakerbockers/ --streaming
```

That writes `squeakerbockers.ts` next to the run directory.

## Pull the export and swap it in

```bash
# Local
runpodctl receive /workspace/runs/squeakerbockers/squeakerbockers.ts
mv squeakerbockers.ts /Users/gabriel/new_development/Squeakerbockers/models/
```

Then in `config.py`:

```python
RAVE_MODEL_PATH = PROJECT_ROOT / "models" / "squeakerbockers.ts"
```

Restart `app.py`. `rave_engine.py` introspects sample rate, latent dim,
and hop size from the model — no other code change.

## Publish for the Well-Tuned badge

```bash
huggingface-cli login
huggingface-cli upload <your-hf-username>/squeakerbockers squeakerbockers.ts
```

Write a model card describing the dataset (sources, hours, license),
training config (`v2`), and intended use. Link it from this repo's
README.

## Budget guardrails

~$0.34/hr × 24 × 3 days = ~$25 worst case. **Stop the pod when not
actively training** — Community Cloud bills by the second.
