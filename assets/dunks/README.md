# Target clip library

Candidate target videos for the face-swap product, all **CC0 / Pexels License**
(free for commercial use, no attribution required). Curated with
`scripts/probe_clips.py`, which reports face-detection rate and median face
width (the key swap-quality signal — `inswapper_128` aligns to a 128px crop, so
a face smaller than ~10–15% of frame width swaps blurry).

## Verdict: swap quality depends entirely on face size

Empirically confirmed on this box (see `/tmp` sample renders during build):
**face-to-camera clips swap cleanly; action dunk/dribble clips do not** because
the player's face is only 2–3% of frame width (~17px) and the swap comes out as
mush.

| File | Source clip | Face rate | Face width | Use |
| --- | --- | --- | --- | --- |
| `cand_5319070.mp4` | man looking at camera | 100% | **39%** | ✅ swap target (proven clean swap) — but generic portrait, not basketball |
| `pexels_8979081_sd.mp4` | dunk toward camera | 49% | 2.7% | ⚠️ b-roll only — face too small to swap |
| `pexels_5192026_sd.mp4` / `_1080p` | man dunking | 56% | 3.1% | ⚠️ b-roll only |
| `pexels_5192076_sd.mp4` | playing basketball | 43% | 2.1% | ⚠️ b-roll only |
| `cand_5586522.mp4` | man playing basketball | 51% | 1.6% | ⚠️ b-roll only |

`✅` = good face-swap target. `⚠️` = usable as un-swapped action/montage b-roll
(e.g. the wide dunk shot in a composite), but not as a swap target.

## What's still needed

A **basketball-context clip with a large face facing camera** (single player
holding a ball / celebrating / posing). These exist on Pexels/Pixabay (e.g.
Pexels `8693767` "man holding a ball and looking at camera") but the sites'
bot protection blocked automated download of their CDN URLs. To get them
reliably, use a free **Pexels API key** (https://www.pexels.com/api/) — then
clip sourcing can be fully scripted — or download a few by hand and drop them
here, then run `scripts/probe_clips.py` to confirm face width ≥ ~0.15.
