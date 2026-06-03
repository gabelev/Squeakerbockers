# SPEC.md — Squeakerbockers

A realtime generative-audio toy for the Hugging Face "Build Small" hackathon
(Track Two: An Adventure in Thousand Token Wood).

People walk around in front of a webcam. A pose model tracks how they move. Those
movement features drive a RAVE neural audio model trained on shoe-squeak / court
foley, which synthesizes squeak textures in realtime. The room sounds like a
basketball court. The audio is generated, not sampled.

Named for the New York Knickerbockers, whose sneakers squeak on the Garden floor.
Alternates if you want them: Sneakerbockers, Madison Squeak Garden.

---

## 1. What this is and is not

- This is an **inference + control-mapping + streaming** app. The only training in
  the whole project is one RAVE run, which happens on a rented GPU, not here. See
  Section 8.
- The RAVE model **generates** the waveform via latent-space navigation. We are not
  triggering audio samples.
- Realtime is the goal. Local will feel tight. The deployed HF Space will be laggier
  because of the WebRTC round-trip. Build and tune locally, capture the demo video
  locally, ship the Space as the required hosted artifact.

## 2. Constraints (hackathon)

- Total model params ≤ 32B. We are nowhere near it. Pose model is ~3-10M, RAVE is
  tens of M. Note this in the README; it is a non-issue.
- Must be a Gradio app hosted as a Hugging Face Space.
- Submission needs a short demo video and a social post.
- Free bonus badges we should collect: **Off the Grid** (no cloud APIs, fully local
  models), **Well-Tuned** (publish the fine-tuned RAVE model on HF), **Off-Brand**
  (custom court-themed UI past default Gradio), **Field Notes** (a short blog post).

## 3. Architecture

Two loops sharing state. This decoupling is the most important design decision.

```
                 ┌─────────────── VIDEO LOOP (webcam FPS) ───────────────┐
  webcam frames ─► pose model ─► movement features ─► update SHARED STATE │
                 └───────────────────────────────────────────────────────┘
                                                            │ (latest features)
                                                            ▼
                 ┌─────────────── AUDIO LOOP (audio rate) ───────────────┐
  SHARED STATE ─► control mapping ─► RAVE latent slice ─► RAVE.decode ─► audio out
                 └───────────────────────────────────────────────────────┘
```

- The **video loop** runs at camera FPS. It does pose estimation, computes features,
  and writes them to a small shared state object. It may also draw a skeleton overlay
  on the returned video.
- The **audio loop** runs independently at audio rate. Each time it needs the next
  audio chunk, it reads the latest features, turns them into the next RAVE latent
  slice, decodes, and yields audio. It never blocks on the video loop.
- Decoupling matters because RAVE produces a fixed number of samples per latent frame,
  and that cadence has nothing to do with webcam FPS. Do not try to generate one audio
  blob per video frame.

Transport is **FastRTC** with `modality="audio-video"`, `mode="send-receive"`. Video
in, audio out, video-with-overlay back to the client.

Concretely, this lives inside `AsyncAudioVideoStreamHandler`, which exposes four
async methods on independent cadences (plus `start_up` and `copy`):

- `video_receive(frame)` — runs at camera FPS. Pose, features, write shared state,
  stash latest overlaid frame.
- `video_emit()` — runs at video-send rate. Returns the latest overlaid frame.
- `emit()` — runs at audio rate. Reads shared state, produces the next audio chunk
  via control → RAVE decode.
- `receive(audio_frame)` — incoming mic from the client. Unused; implement as a no-op.

Heavy model loads (YOLO weights, RAVE TorchScript) live at module level so they happen
once per process, not on every new client connection — `copy()` should be cheap.

## 4. Components / repo layout

```
squeakerbockers/
  app.py            # Gradio Blocks + FastRTC Stream entry point, wires loops
  pose.py           # pose model wrapper: frame -> list[person_keypoints]
  features.py       # keypoints -> movement features (per person + aggregate)
  control.py        # features -> RAVE latent trajectory (the sound-design layer)
  rave_engine.py    # load exported RAVE .ts, decode latent slices -> audio chunks
  state.py          # thread/async-safe shared state between the two loops
  config.py         # model paths, latent_dim, sample_rate, all tunables
  ui/               # custom CSS + court assets
  models/           # RAVE .ts checkpoints (gitignored, downloaded at setup)
  requirements.txt
  README.md
  SPEC.md           # this file
```

## 5. Component contracts

### pose.py
- Input: a single video frame (numpy HxWx3).
- Output: a list of detections, each with named keypoints including both ankles/feet,
  hips, and a bounding box or center.
- Model: **YOLO pose nano** (ultralytics, e.g. yolo11n-pose). Chosen because "people
  walking around" is multi-person and the API is trivial. Must run locally with no
  cloud call (Off the Grid badge). Note ultralytics is AGPL; fine for a hackathon.

### features.py
Per person, from the ankle/foot keypoints over a short rolling window, compute:
- `motion_energy`: overall keypoint speed.
- `foot_velocity`: per-ankle frame-to-frame displacement.
- `pivot_sharpness`: magnitude of sudden deceleration or direction change of a foot.
  This is the basketball screech signal. It matters most. Tune it hardest.
- `pan`: body center x-position in frame, mapped to stereo L/R.
- `proximity`: body bbox height / scale, mapped to volume.
Aggregate across people into a single feature vector for the control layer (sum of
energies, max pivot, mean pan weighted by proximity, person count).

### control.py
- Input: the aggregate feature vector.
- Output: the next RAVE latent slice, shape `[1, latent_dim, T]`.
- Approach: drive the top few PCA latent dimensions from features (pivot_sharpness and
  motion_energy are the loudest knobs), fill the remaining dims with smoothed low-amp
  noise, and **smooth the trajectory over time** to avoid zipper/click artifacts.
- This file is where the character lives. Expect to iterate by ear. Keep the mapping
  parameters in config.py so they are tunable without touching logic.

### rave_engine.py
- Load an exported, **streaming** RAVE TorchScript model with `torch.jit.load`.
- Introspect the model for latent dimension and sample rate; do not hardcode. Expose
  them so config and the audio loop can read them.
- `decode(z) -> audio_chunk`. Maintain any rolling buffer needed so emitted chunks are
  continuous.
- Must work identically for the placeholder pretrained model and the trained squeak
  model. Model path is the only thing that changes.
- Load the model once at module import time (it is heavy). The handler's `copy()`
  should not re-load it.

### state.py
- Two single-slot, last-write-wins, async-safe holders:
  - `latest_features`: aggregate movement features. Written by `video_receive`, read by
    `emit` (audio loop).
  - `latest_overlay_frame`: most recent annotated video frame. Written by
    `video_receive`, read by `video_emit`.
- Stale reads are fine; never block. No queue — we only ever care about the most
  recent value on each slot.

## 6. Build phases (do them in this order)

De-risk streaming first, sound quality last.

1. **Scaffold.** Repo, requirements, config.py, empty module stubs. Runnable Gradio app
   that opens.
2. **FastRTC echo (two passes).**
   - **2a.** Pure-video echo: webcam in, same video back. Confirm WebRTC plumbing
     (camera permissions, TURN, local connection) works at all.
   - **2b.** Audio-video echo: implement `AsyncAudioVideoStreamHandler` directly with
     `video_receive` → stash → `video_emit` pass-through, and `emit` returning a fixed
     sine wave. This proves the *actual* handler shape everything else gets built
     inside. If audio clicks or stalls here, you know it is not pose or RAVE.
3. **Pose + overlay.** Add pose.py, draw skeletons on the returned video. Confirms
   detection works and gives a satisfying visual.
4. **Features.** Add features.py. Log the feature vector live so it can be eyeballed
   while moving around.
5. **RAVE static test.** Add rave_engine.py. Load the placeholder model, decode a fixed
   latent, confirm clean audio streams out over FastRTC. Decoupled from video.
6. **Wire it together.** features -> control -> latent -> decode -> audio, via shared
   state. First end-to-end sound driven by movement.
7. **Sound design.** Iterate control.py mapping by ear. Make pivots screech.
8. **UI.** Court-themed custom CSS, control sliders (sensitivity, squeak intensity,
   dry/wet, noise floor).
9. **Deploy.** HF Space: requirements, TURN credentials for the WebRTC connection in
   the hosted environment, README with badges, model card link.

## 7. Known gotchas (read before coding)

- **Streaming export is mandatory.** A RAVE model exported without cached convolutions
  will click and is not realtime-safe. The audio loop assumes a streaming model.
- **latent_dim and sample_rate are model-specific.** Read them from the loaded model.
  The placeholder model and the trained model may differ; config must adapt.
- **Smooth latents.** Interpolate between feature updates and low-pass the latent
  trajectory or you get zipper noise.
- **FastRTC handler is `AsyncAudioVideoStreamHandler`** with the methods listed in §3.
  This was unknown when the spec was first drafted; if FastRTC's API changes again,
  re-verify against https://fastrtc.org/userguide/audio-video/ before writing it.
- **Local vs Space latency.** Expect the Space to lag. That is the WebRTC round-trip,
  not the model. Demo video is captured locally.

## 8. Parallel workstream: training (NOT part of this repo's code)

This runs on RunPod the same day, independently. The app is built against a placeholder
pretrained RAVE model so it does not block on training.

- GPU: single RTX 4090, RunPod Community Cloud (~$0.34/hr). No A100 needed; RAVE v2
  wants 8GB and the 4090 has 24GB.
- Dataset: pull from public sources; do not record it yourself. Target 1-3 hours of
  shoe-squeak / court-shoe audio. Good sources:
  - **Pixabay** sound effects (royalty-free, no attribution required). Search
    "basketball court" and "sneakers"; it includes long court field recordings
    (there is a ~35 min basketball rec) that make ideal bulk material.
  - **Freesound** (filter to CC0 to skip attribution bookkeeping; has a bulk-download
    API). Search "sneaker squeak", "basketball court", "shoe squeak".
  - Orange Free Sounds and similar free libraries for extra variety.
  Concatenate, resample to one rate, trim obvious junk. Tradeoff: public clips are less
  homogeneous than a self-recording (varied rooms, mics, levels). That suits the lo-fi
  aesthetic fine, but expect a grittier model. Keep crowd noise and commentary out where
  you can; RAVE faithfully learns whatever is in the corpus.
- Preprocess (resample) up front; do not use lazy loading on a cheap pod.
- Train with a v2-class config. Checkpoint to a **persistent network volume**, not the
  pod's ephemeral disk, so a dropped Community Cloud machine can resume.
- Listen to checkpoints; do not wait for a step count. Pull when it sounds good. For a
  narrow squeak timbre, a usable checkpoint should arrive in roughly 2-3 days.
- **Export with the streaming flag.** The exported `.ts` is the only artifact the app
  needs. Drop it in `models/`, point config at it, done.
- Publish the trained model on HF for the Well-Tuned badge.

## 9. Placeholder model for development

Until the squeak model is trained, build against
[`Intelligent-Instruments-Lab/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)
on Hugging Face. Default: `birds_dawnchorus_b2048_r48000_z8.ts` — chosen because the
streaming export is confirmed, the file is small (~67 MB) so iteration is fast, the
latent dim is low (z=8) so control mapping has less variance to wrestle, and the
sample rate (48000) and block size (2048) are standard.

Other models in that repo (guitar, organ, water, voice) follow the same
`<source>_b<block>_r<sample_rate>_z<latent_dim>.ts` naming. `rave_engine.py` reads
block, sample rate and latent dim *from the loaded model*, so trying a different
placeholder is a config path change.

Download into `models/` (gitignored). Swapping in the trained squeak model later is
the same one-line config change — the whole point of the contract in `rave_engine.py`.

## 10. Definition of done

- Local: walk in front of the camera, hear movement-driven squeaks, pivots screech,
  panning tracks position, latency feels live.
- Trained squeak model swapped in and clearly better than placeholder.
- HF Space deployed and functional (laggier is acceptable).
- Demo video (local capture) + social post drafted.
- Badges wired: local-first, published fine-tuned model, custom UI, blog post.
