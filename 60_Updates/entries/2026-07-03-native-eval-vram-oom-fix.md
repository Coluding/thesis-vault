---
date: 2026-07-03
tags: [wan2.2, vram, oom, eval, offload, training]
---

# Native eval OOM on 24 GB (RTX 3090): diagnosis + fix

The native eval grid (`trainer._native_eval_grid` → `_native_clip_rollout`) OOM'd
during training on a 24 GB card, while the standalone generation script ran fine.

## Diagnosis (the useful traceback)
- The crash is in the **VAE decode** (`i2v` → `self.vae.decode(x0)`), *after* the
  DiT denoise loop finished (`100%|██████████| 1/1`), not in the DiT forward.
- The error reports memory **genuinely allocated** (~21–22 GiB), only ~360 MiB
  "reserved but unallocated" → **not** a caching/reserved problem, so
  `torch.cuda.empty_cache()` alone does nothing. (First fix attempt — empty_cache
  at the start of the eval methods — therefore didn't help.)
- Root cause: the **~10 GB frozen DiT is resident on GPU during the decode**. A full
  clip decode at native res (960×928 × 41 frames) needs ~10 GB *on top* of the
  resident training model → no room. The standalone script works because it runs
  with `offload_model=True` (its default), and upstream i2v then does
  `self.model.cpu()` **before** `self.vae.decode(...)` (textimage2video.py:627-633).
  Training set `offload_model=False` to keep the DiT resident for the training step.

## Why offload alone still OOM'd
Passing `offload_model=True` for eval generation (via a new `generate(offload_model=)`
override, forwarded through `AdaptedModel.generate`) let the first few rollouts
decode — then a later one OOM'd from **fragmentation**: shuffling the 10 GB DiT
CPU↔GPU every rollout fragments the allocator until no 1.3 GB *contiguous* block is
free. The error explicitly recommends `expandable_segments:True`.

## Fix (three parts)
1. `WanTI2VVideoModel.generate` gained `offload_model: bool | None` (override);
   `_native_clip_rollout` passes `offload_model=True`. After the eval loop, both
   `_native_eval_grid`/`_native_quality_eval` **free cache then `self.model.to(device)`**
   (order matters — the restore itself OOM'd when it ran before freeing).
2. **Eval-only resolution knob** `training.extra.inference_max_area` (script now
   `setdefault`s it, so YAML wins; default = training `max_area`). Set to `262144`
   (512²) so the eval decode needs a ~3.4× smaller block and fits alongside the
   resident model. Fidelity of the *monitoring grid* only — training stays native.
3. Run with **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to kill the
   offload-shuffle fragmentation.

Also: `inference_every_n_steps: 2` makes the grid run almost every step (with per-
rollout offload that's very slow) — raise it (e.g. 100–500).

## CORRECTION — the real hog is the VAE decode, not the DiT (2026-07-03, later)
A memory probe in `_native_clip_rollout` (`_eval_mem`, first grid only) settled it:
```
rollout-start:      alloc=13.42G  dit=cuda:0     (DiT resident, training)
after-base-gen:     alloc=3.42G   dit=cpu        (offload WORKS: 10 GB freed)
after-base-release: alloc=3.42G   reserved=3.70G dit=cpu   (~20 GB free)
```
So `offload_model=True` **does** move the DiT to CPU before decode — offload was
never broken, and it's not the limit. The OOM is the **VAE decode itself**: a
full-clip decode at native res (960×928 × 41 frames) peaks ~15-18 GB of conv
feature maps + `feat_cache`. The *base* rollout's decode fits by a hair; the
*adapted* one misses by a **fragmentation sliver** (needed 2.57 GB, 2.62 free but
non-contiguous). `expandable_segments` was already active (the "deprecated"
warning means the old `PYTORCH_CUDA_ALLOC_CONF` name is still honored) and still
missed — because it's raw decode size, not just fragmentation.

Real fix: **`inference_max_area: 589824` (768²)** — decode cost scales with pixel
area, so this shrinks the overflowing thing ~35% for comfortable margin while
staying near-native (vs the washed 512²). Added an `except torch.cuda.OutOfMemoryError`
guard around both native eval methods so a decode overflow **skips the grid/metrics
and frees memory instead of killing training** (then the DiT is still restored to GPU).
Superseded parts 2-3 above: the 262144 value was too low (washed), and
expandable_segments alone is insufficient.

Related: [[2026-07-03-wan22-text-injection-and-cfg-memoization]].
