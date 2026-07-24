---
date: 2026-07-03
tags: [wan2.2, performance, vae, latent-cache, training]
---

# VAE encode is the training bottleneck → bf16 + latent caching

Profiling (`GFA_PROFILE=1`, env-gated `_prof` timers with CUDA sync) of a training
step at 768² / 41 frames:
```
preprocess (VAE encode + cond):  ~3320 ms  (~67%)   <- bottleneck
  base forward (frozen 5B):       ~1480 ms
  adapter forward (delta):          ~46 ms
  backward (adapter-only):          ~92 ms
shortcut target prep (when it fires): ~3060 ms       (distillation teacher = 2 extra base fwds)
```
The VAE encode is pure recomputation (frozen VAE, deterministic clip pixels), so it
should be done once, not every step.

**#3 bf16 VAE** (`training.extra.vae_dtype: bf16`, default): the Wan VAE runs
encode/decode under `amp.autocast(dtype=self.dtype)`, default fp32. Setting bf16 cut
the encode ~25% only (4366→3320 ms) — the encoder's early full-res 3D convs are
**bandwidth-bound**, not compute-bound, so low precision barely helps. Latent still
returned as fp32. Also speeds the eval VAE decode (helps the earlier decode OOM).

**#2 latent cache** (the real fix): `data/latent_cache.py` `LatentCache` (RAM+disk),
keyed on `env_name|episode_idx|start_idx|frame_stride|TxHxW` (resolution in the key,
so max_area variants coexist). Latents stored bf16 (halves disk). Preprocessor
`_encode_z0` checks the cache: hit → skip VAE; miss → encode only the misses + write
back (`write_latent_cache`, on by default → lazy fill during training). New
`preprocessor.precompute(batch)` entry point + `--precompute-latents` mode on
`scripts/train_wan22_i2v_metaworld_external.py` fills the whole dataset in one pass.
Cache dir defaults to `<hdf5>.latents/`; disable with `--no-latent-cache`.

Effect: anchor step ~4950 ms → **~1650 ms** (VAE cost → ~0, just model fwd+bwd).
Shortcut steps still pay the teacher's 2 base forwards (~3s) — reduce via higher
`shortcut_anchor_prob` or lower `max_area`. Base forward (~1.5s, frozen 5B) is then
the floor. Tested CPU (fake VAE): miss encodes+writes, hit skips VAE, resolution in
key, disk persists across processes.

Usage: `python scripts/train_wan22_i2v_metaworld_external.py --precompute-latents`
once, then train normally. Re-run precompute if `max_area`/frame count changes.

## GOTCHA: random window sampling defeats a per-window cache

First cut cached per `(env,episode,start,frame_stride,TxHxW)` but training runs with
`--sampling random` (default), whose `TranslatedClipDataset.__getitem__` draws a
**fresh random `start` on every access** (dataset.py:101). So `start_idx` — part of
the key — changes every step → every access misses → re-encodes → cache grows
unboundedly (490→663→…). Symptom: `preprocess` stays ~3.2s even after "precompute".
Exhaustive sampling is deterministic but at stride-1 it's **127,400 windows / 298 GB**
— infeasible.

**Fix: fixed K-window pool.** `TranslatedClipDataset(num_windows=K)`: in random mode,
each episode exposes K deterministic evenly-spaced starts (`_fixed_starts`), and
`__getitem__` random-picks among them — so windows are a finite stable set (K-way
augmentation preserved) that the cache can hit. `fixed_window_enumeration()` yields
every (episode, fixed-start) pair; `--precompute-latents` iterates it to cache exactly
what training samples. Plumbed via `--num-windows` (default 16) + `data.num_windows`.
K=16 → 7,840 windows (~18 GB, ~4.4h one-time encode); K=8 → 3,920 (~9 GB, ~2.2h).
`--num-windows 0` = old unbounded random (cache can't hit).

Also found: `Wan22DiffusionForcingPreprocessor` (subclass) had its own `__call__` with
a direct `self.vae.encode` that bypassed the cached `_encode_z0` — fixed to call it.

Workflow now: delete any stale cache (old random-start keys are orphans), then
`--precompute-latents` (writes the K·N windows), then train (all hits). `GFA_DEBUG_CACHE=1`
prints per-step `[cache] miss m/B key0=… present=…` to confirm hits.

**Standalone precompute:** `scripts/precompute_latents.py` loads **only the VAE**
(`Wan2_2_VAE`, no 5B DiT / adapter / trainer / wandb) and reuses the same dataset
build + resize + `_encode_z0`, so latents/keys are identical to training's — fast
start, low memory. Enumerates exactly the K-window pool; `--limit N` for a partial
pass. Must pass the same `--num-windows`/`--max-area` as training. Latents are
deterministic (frozen VAE), so a cache built anywhere (e.g. locally) is valid on the
cluster — build + `rsync` the `.latents/` dir.

Cluster note: saw a run at 9.2s base-fwd / 52s VAE (bad/shared/MIG node); a healthy
A100 node does 0.3s fwd / 1.9s VAE. If the slow numbers recur, it's node allocation
(check `nvidia-smi -L`), not the pipeline.
