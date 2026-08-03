---
type: exp
scope: backbone
status: in-progress
priority: medium
created: 2026-07-25
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-backbone-wan-robotarm-run]]", "[[exp-backbone-skyreels-pushblock-run]]", "[[exp-backbone-dc-robotarm-run]]"]
---

# exp: SkyReels-V2-1.3B · Robot Arm (matrix run 3 — weak flow base)

## Hypothesis

Base-strength axis: a **weaker flow base** (SkyReels-V2-1.3B, Wan2.1-lineage,
16-ch VAE) leaves a *larger* residual than Wan2.2-5B, so the adapter has more to
learn and is less prone to cloning. Robot Arm is the clean arena for this test —
the SkyReels probe (2026-07-25) showed the frozen 1.3B base **holds the
realistic-3D arm domain** (coherent, correct neutral palette, real articulated
motion) once the prompt is de-branded. On flat Push Cube it style-drifts instead,
so Robot Arm is where the strong-vs-weak comparison is apples-to-apples.

## Procedure

- Base: SkyReels-V2-I2V-1.3B-540P (weights already downloaded by the probe).
- Adapter: output, feature_dim **16** (SkyReels z_dim), mask_mix + cross-attn +
  base-input — same recipe as the Wan robot_arm baseline, at 16-ch width.
- Data: ACWM robot_arm, input_dim 7.

## Decision rule

- **Adapter learns action-conditioned dynamics with lower cloning than Wan-5B**
  ⇒ supports "weaker base → more residual → easier adaptation" for D2.
- **Same cloning behaviour** ⇒ base strength is not the lever; residual is set by
  the domain, not the base.

## Build status

**Core provider DONE (2026-07-25, smoke-green); training path still BLOCKED.**

Done + smoke-tested (no weight load):
- `src/generative_flow_adapters/models/base/skyreels_video.py` —
  `SkyReelsVideoModel(BaseVideoModel)` + `_ComposedSkyReelsDiT` seam. denoise
  uses the batched-tensor + `context`/`clip_fea`/`y` convention (not Wan2.1
  list+seq_len). generate() delegates to the native `Image2VideoPipeline`,
  swapping in the composed DiT; additive residual composes across both CFG
  branches.
- `factory.py` — `provider == "skyreels"` branch.
- `scripts/precompute_latents.py` — provider-aware VAE (`_vae_spatial_stride`
  map; skyreels→16-ch/stride-8; **Wan2.2 path byte-identical**).
- Configs `configs/skyreels/diffusion_skyreels_xattn_acwm_{pushblock,robotarm}.yaml`
  (feature_dim 16). Load clean.
- Text context: SkyReels has its OWN T5 → encoded live in generate(); Wan
  `*.contexts.pt` are NOT reusable.

Training path now CODE-COMPLETE + smoke-green (2026-07-25):
- `src/generative_flow_adapters/data/skyreels_batch_preprocessor.py` —
  `SkyReelsI2VPreprocessor(WanBatchPreprocessor)`; classic-i2v batch emitting
  `x_t`/`t`/`target`/`x0`/`frame_mask` + `cond.{act,action_seq,context,clip_fea,y}`.
  Only `z0` is cached; `context`(SkyReels T5)/`clip_fea`/`y`(20-ch) built live.
- `scripts/train_skyreels_acwm.py` entrypoint (mirrors the Wan external trainer;
  `--dataset acwm_phys`).
- sbatch: `jobs/experiments_cluster/acwm_phys/skyreels/submit_train_skyreels_{pushblock,robotarm}.sh`
  (weights auto-download from HF cache; no --ckpt-dir).

Remaining = **GPU validation only** (8 inline `# GPU-VALIDATE` items): full
model construction + one real `denoise()`; T5 batch-encode of B prompts;
`clip.encode_video` shape; `y` 20-ch vs DiT `in_dim` (16+20=36); VAE batched
encode shape; `t=[B]` vs per-frame loss; stride-8 geometry (384²/384×512 are
guesses; probe ran 960×544); `build_experiment` wiring the adapter at
feature_dim 16. Base coherence on both domains already shown by the probes.

## Notes

Partner: [[exp-backbone-skyreels-pushblock-run]] (same base, flat domain — expect
style-drift confound). Base-family peers: Wan
([[exp-backbone-wan-robotarm-run]]), DynamiCrafter
([[exp-backbone-dc-robotarm-run]]). Axis 5 in
[[../../30_Knowledge/writing/ablation-axes]].

## Cleanup 2026-08-01 — **DELIVERED**

Run as `8zjjn7wl` (0.0013) — [[../../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]; the same backbone reaches 0.045 on RT-1 ([[../../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]).

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
