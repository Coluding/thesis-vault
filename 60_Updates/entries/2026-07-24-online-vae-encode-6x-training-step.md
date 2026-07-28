---
date: 2026-07-24
category: finding
deliverable: D1
meeting:
sources: ["[[entries/2026-07-08-latent-cache-skip-resize-on-hit]]", "[[../20_Tickets/bug-data-acwm-decord-dataloader-fork-deadlock]]"]
---

# Online VAE encode is ~6× the training step at 768² → keep the latent precompute

## What

Profiled the WAN-2.2 (TI2V-5B) adapter training loop at the real ACWM
push_block geometry (768², 65-frame windows → latent 17×48×48, bs=1) with
CUDA-synced per-phase timers, comparing **online VAE encode** vs **cached
latents** in the same loop. The question was whether the pre-encode step could
be dropped at 768² now that the native-resolution OOM is gone.

Steady-state per step:

| phase | online encode | cached latents |
|---|---|---|
| preprocess (VAE encode + cond) | **~3665 ms** | ~100 ms |
| training_step (fwd+bwd+opt) TOTAL | ~580 ms | ~580 ms |
| ┗ frozen 5B base forward | 480 ms | 480 ms |
| ┗ adapter forward / backward | 28 / 54 ms | 28 / 54 ms |
| **per-step wall** | **~4.26 s** | **~0.69 s** |

## Why it matters

- **Keep precompute — the "drop it" hypothesis is dead.** The VAE encode
  (~3.6 s) is **~6× the entire training step** (~0.58 s), not ≪ it. Online
  encoding is a ~6.1× throughput hit (0.24 vs 1.44 steps/s). Memory was never
  the blocker (encode coexists with the resident 5B, ~26 GB reserved on an
  80 GB H100) — **wall-clock is.** Extrapolated to 50k steps: ~59 h online vs
  ~12.6 h cached (incl. one-time ~3 h precompute).
- **The frozen-base forward is the real throughput ceiling.** On the cached
  path, the frozen 5B forward is **480 ms — ~83% of the 580 ms step** — while
  the *trainable* adapter is only 28 ms fwd + 54 ms bwd. It can't be cached
  like the latents (it depends on freshly-sampled `x_t`, `t`), so any base-side
  speedup (kernel / `torch.compile` on the frozen DiT) is the lever that
  actually moves training throughput.

## Evidence / sources

- Online-encode run: wandb `hswppa8s` (project `Wan2.2-avid-xattn-acwm-pushblock`),
  config `diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml`,
  `scripts/train_wan22_i2v_metaworld_external.py --no-latent-cache`, 15 steps,
  elapsed 213.6 s.
- Cached-latents run: wandb `8cug8wfq` (same config + script,
  `--latent-cache-dir …/latents.shared`), 15 steps, elapsed 76.3 s.
- Profiler: `GFA_PROFILE=1` phase timers (CUDA-synced) via
  `jobs/experiments_cluster/infra/profile_vae/submit_profile_train_step.sh`.
  Isolated VAE-encode profile (memory + no-batch scaling) from
  `scripts/profile_vae_encode.py`.

## Next

- No code change to the training path — precompute stays the default.
- Low-priority `perf` follow-ups: (1) the WAN VAE `encode` does not batch
  (time scales linearly, memory flat) — matters only for *precompute* wall
  time now; (2) `torch.compile` / attention-kernel on the frozen base forward —
  the 83%-of-step throughput ceiling. Both to be filed as tickets.
- Numbers feed the thesis framework/method chapter (D1) as the justification
  for the latent-precompute design.
