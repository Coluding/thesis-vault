# Latent cache: skip the pixel resize on a cache hit (~15 s/step recovered)

Date: 2026-07-08
Files: `src/generative_flow_adapters/data/wan_batch_preprocessor.py`,
`src/generative_flow_adapters/data/wan22_batch_preprocessor.py`
Commits: `6bf05a1` (parent), + follow-up subclass fix (see below)

## Symptom

With the latent cache fully hitting (`[cache] miss 0/12 present=True`, VAE never
runs), `[prof] preprocess` was **still 15.4 s/step** at 97f/832²/batch-12 — and
`nvidia-smi` showed **GPU util 0 %** during that window. The H100 sat idle while
the CPU did work whose result was thrown away.

## Cause

`WanBatchPreprocessor.__call__` resized+normalized the **whole batch** up front
(`_normalize_video` → PIL LANCZOS upsample 128→832 of all 12×97 = 1164 frames)
*before* consulting the cache. On a hit, `z0` comes from disk and those resized
pixels are discarded. Pure waste, scaling with frames × batch × resolution
(was ~3.2 s at 41f/768²).

## Fix

Made `_encode_z0` **cache-first**, keying off the *raw* frames:

1. New `_output_hw(src_h, src_w)` predicts the resized H×W via `best_output_size`
   (deterministic) — mirrors every branch of `_normalize_video`, no resize.
2. `_latent_keys` now takes explicit `(t, h, w)` instead of a resized tensor.
3. `_encode_z0(raw_video, …)`: build keys from raw dims → look up cache → resize
   + normalize + upload + VAE-encode **only the miss indices** (`raw_video[miss_idx]`).
   On a full hit the pixel path is never entered.
4. `__call__` / `precompute` pass `batch["video"]` (raw uint8 [B,T,H,W,3]) straight
   in; no pre-normalize.

Miss path is byte-identical to before (hit rows bf16 from cache, miss rows fresh
fp32 — unchanged semantics). Non-uint8 input (dataset never emits it) falls back
to resize-up-front, so nothing regresses.

## Verified

- Full hit: `last_encoded=0`, `_normalize_video` called **0** times, z0 == old
  cached path (diff = pre-existing bf16 quantization, ~1e-4).
- Partial miss: `_normalize_video` called once (only the miss), order preserved,
  miss written back to disk. Hit rows == bf16 fresh, miss row == fp32 fresh.
- Predicted `_output_hw` == actual resized dims.
- 18 wan/preprocessor tests pass; the 2 `test_wan22_i2v.py` failures are
  pre-existing (missing `configs/diffusion_wan22_i2v_metaworld.yaml`, renamed to
  the avid/gated variants — unrelated).

## Expected impact

Cached-step preprocess 15.4 s → ~0.5 s (disk read + noise/interp/cond). GPU stops
idling. Per-step wall ~28.7 s → ~13.8 s at 97f/832²/b12 (~2× throughput). Bigger
the more frames/resolution/batch. No method or numerical change.

## Follow-up: the fix missed the class actually used (subclass override)

The first fix (`6bf05a1`) only touched the **parent** `WanBatchPreprocessor`. But
training runs `Wan22DiffusionForcingPreprocessor(WanBatchPreprocessor)` — the
diffusion-forcing variant — which **overrides `__call__`** (it builds per-frame
timesteps + `frame_mask`). Its `__call__` still did
`video = self._normalize_video(batch["video"])` up front and passed the
*normalized* tensor into `_encode_z0`. So on the cluster the code update "existed"
yet preprocess stayed ~15 s.

Two failure modes from that:
1. The parent's cache-first `__call__` was dead code for the subclass.
2. The pre-normalized (float `[B,C,T,H,W]`) video failed `_encode_z0`'s
   `standard = … dtype==uint8` guard → fell into the `else` branch → resized
   **again**. (Symptom the user caught: execution jumping into that `else`.)

Fix: apply the same 2-line change to the subclass `__call__` — pass raw
`batch["video"]` to `_encode_z0`, drop the up-front `_normalize_video`. All of
`_encode_z0` / `_normalize_video` / `_output_hw` / `_latent_keys` are inherited,
so nothing else changes.

Verified through the real class: full hit → `last_encoded=0`, `_normalize_video`
called **0** times, and the diffusion-forcing batch (`x_t`, `frame_mask`, `t`,
`target`, `x0`, `cond`) still builds correctly.

**Lesson / gotcha:** there are two preprocessors — the base `WanBatchPreprocessor`
and the `Wan22DiffusionForcingPreprocessor` subclass that overrides `__call__`.
Any change to the batch-building path must be applied to (or verified against) the
**subclass**, since that's what the Wan2.2 training script instantiates. Grepping
for `_output_hw` proves the *parent* fix is present but says nothing about whether
the subclass path is cache-first — check `wan22_batch_preprocessor.py:__call__`.

## Cache-key fields — change ANY of these and you MUST re-precompute

Key = `env | episode | start | frame_stride | T×H×W` (`data/latent_cache.py`). A
change to any field silently invalidates every cached latent (→ full misses, the
VAE re-encodes on every step). The non-obvious ones:

- **`temporal_length`** — sets `T` in the key AND shifts every window's `start_idx`
  (window starts = evenly spaced over `[0, L − temporal_length]`, so 41f and 97f
  produce *different* start positions). Changing 97→41 gave a dir full of stale
  `97x768x768` latents that never match `41x768x768` keys. Bit us 2026-07-08.
- **`max_area`** — sets `H×W` (via `best_output_size`). 589824→768², 692224→832².
- **`--num-windows`** — the deterministic window pool (which `start_idx` values exist).
- **`--frame-stride`**, **`--hdf5`** (env name).

Different geometries **coexist** in the same dir (T/H/W are in the key, no
collision), so an A/B over frame counts just needs one precompute pass per value —
but stale sets pile up as dead disk. Diagnose a mixed dir by latent shape:
`(48, 11, 48, 48)` = 41f, `(48, 25, 48, 48)` = 97f.

Symptom of a coverage gap (vs a format mismatch): **hits and misses share the same
`T×H×W`** in the debug line — the key format is right, those specific windows just
aren't on disk. A format mismatch instead shows a `present=False` key whose
`T×H×W` differs from what precompute wrote.

## Deploy note (cluster)

Cluster config diverges from committed `main` (`temporal_length: 97` in-place vs
`41` committed), so a full `git pull` would break the cache-key geometry. Deploy
surgically instead:
```
git fetch origin
git checkout origin/main -- src/generative_flow_adapters/data/wan_batch_preprocessor.py
git checkout origin/main -- src/generative_flow_adapters/data/wan22_batch_preprocessor.py
```
Verify a warm step shows `[cache] miss 0/N present=True` AND `[prof] preprocess
< 1000 ms` (was ~15000 ms).

## Remaining bottleneck (unchanged)

`model forward` base 5B ≈ 11.2 s/step is now the floor (97 frames, 16900 tokens).
Levers: lower `temporal_length`/`max_area`, `torch.compile` the frozen base, or a
faster attention backend. Shortcut prep adds 2 more base forwards on ~40% of steps
(`shortcut_anchor_prob 0.6`) — method knob, left alone for the A/B.
