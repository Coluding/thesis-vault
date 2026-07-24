---
type: experiment
date: 2026-07-15
config: external_repos/avid/latent_diffusion/configs/train/avid/avid_11M_metaworld.yaml
commit:
wandb_run_id: pg3x72uc
ckpt_path:
status: running
deliverable: exploratory
metrics:
  train_loss_step_early: 0.12374
  train_loss_step_late: 0.01342
  val_loss_early: 0.03404
  val_loss_late: 0.01743
  mask_mean_init_theoretical: 0.5
  mask_mean_early: 0.51978
  mask_mean_late: 0.63262
notes: >
  Real, unmodified AVID reference code (AVIDAdapter + train_avid.py), MetaWorld
  data via a new MetaworldVideoDataModule wiring. Positive control for the
  gate-saturation diagnosis — see bug-adapter-gate-saturation-mask-mix.md and
  why-adapter-underlearns-diagnosis.md.
---

# AVID native reference run on MetaWorld — healthy, positive control

**Run:** [`coluding/avid-metaworld/pg3x72uc`](https://wandb.ai/coluding/avid-metaworld/runs/pg3x72uc)
· 11M-tier action-conditioned UNet · frozen DynamiCrafter-512 base ·
`init_mask_bias: 0.0` · state `running` at time of writing (step ~1169,
~800 logged train steps).

The **real, unmodified upstream AVID code** (`AVIDAdapter.apply_model`,
`scripts/train_avid.py` — no changes to model/training logic) trained on our
MetaWorld frames via a newly-wired `MetaworldVideoDataModule`
([[../../20_Tickets/experiments/exp-adapter-avid-native-reference-run]]).

## Result — clean, monotonic convergence

| Metric | early (steps 0–79) | late (steps 478–804) | trend |
|---|---|---|---|
| `train/loss_step` | 0.1237 | 0.0134 | **~9.5× drop**, monotonic across 4 windows |
| `val/loss` | 0.0340 | 0.0174 | tracks train, no train/val divergence |
| `mask_mean` (gate) | 0.5198 | 0.6326 | **steady, substantial climb off init** |

`mask_mean` starts at 0.5198, matching the theoretical `σ(init_mask_bias=0.0)
= 0.5` almost exactly — confirms the composition math is behaving as
documented — and climbs steadily rather than staying pinned. Per-pixel
`mask_std` ≈ 0.057 (a real spatial gate, not a flat scalar). Full generative
quality metrics are also logging cleanly (PSNR/SSIM/LPIPS/MSE/FID/FVD across
16 rollout steps) with the expected degrade-with-rollout-distance pattern
(PSNR ~36 at step 0 → ~24 at step 15) — no sign of collapse or divergence.

## Why this matters

This is the first genuine **positive control** in this investigation: the
frozen-base + gated-adapter composition converges cleanly, on the actual task
we care about (MetaWorld), on code we did not write. It substantially
strengthens the case (previously only a static, code-read comparison — see
[[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] §"External
validation") that our `gate_bias: 4.0` (σ(4)≈0.982, gate stuck near "keep
base") — not something more fundamental about the adapter approach — is the
dominant confound behind our own runs' weak/flat denoising loss.

**Caveats:** still `running`, only ~800 logged steps — not a finished result,
watch for regression at longer horizons. Losses are latent-space diffusion
ELBO terms, not directly numerically comparable to our own pixel-space
metrics — the comparison that matters is the **shape** of the trajectory
(monotonic, high-SNR descent; gate actively moving), not absolute values.

## Related

- [[../../20_Tickets/experiments/exp-adapter-avid-native-reference-run]] — the setup ticket
- [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] — the hypothesis this supports
- [[../tech/why-adapter-underlearns-diagnosis]] — the full diagnosis this feeds into
