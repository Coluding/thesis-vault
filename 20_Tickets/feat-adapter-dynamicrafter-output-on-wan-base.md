---
type: feat
scope: adapter
status: in-progress
priority: high
created: 2026-07-09
updated: 2026-07-14
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[../50_Decisions/open/d2-default-adapter]]", "[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[bug-adapter-gate-saturation-mask-mix]]", "[[bug-infra-wan-script-provider-mismatch]]"]
---

# feat: DynamiCrafter output adapter on the WAN base (high-capacity, cross-arch)

> **Promoted to next active experiment (2026-07-13):** the cross-attention
> injection arm did not improve results
> ([[feat-adapter-wan-action-cross-attention]]), pointing at the **capacity**
> hypothesis over the injection-mechanism one. This ticket is the direct test.

## Idea (user, 2026-07-09)

Train the WAN2.2 TI2V-5B frozen flow base with a **DynamiCrafter output adapter**
(the unified output adapter's `backbone: unet` = `DynamicCrafterOutputAdapter`, a
full 3D video-diffusion UNet) as `Δ_φ`, instead of the lightweight AVID/Wan output
adapter used so far.

## Why

1. **Capacity lever for the "adapter helps but not enough" finding**
   ([[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]).
   If the current adapter's action effect is too weak relative to the strong
   frozen base, a UNet-scale adapter is the direct test of the capacity hypothesis.
2. **D1 contribution — heterogeneous adapter-on-backbone.** DynamiCrafter-UNet
   adapter on a WAN-DiT flow base = adapter architecture ≠ base architecture. A
   concrete demonstration of the "plug-and-play, wrap any backbone" thesis claim,
   and a data point for [[../50_Decisions/open/d2-default-adapter]].

## Key technical risk — latent/output-space bridge

Composition `f = f_base + g·Δ` requires the adapter to emit a delta in the **WAN
base's output space** (WAN velocity, Wan2.2 VAE latent layout: channels,
patchification, temporal stride 4). A DynamiCrafter UNet natively operates in
**DynamiCrafter's** latent space (different channels/resolution/temporal
structure). So it needs **input/output projection layers**: WAN-latent → DC-UNet
input → DC-UNet → DC-output → WAN-latent delta. This bridge is the make-or-break
piece — verify what `build_adapter` + `AdaptedModel` already support for a
DC-UNet adapter delta composing onto a WAN base (the existing
`diffusion_output_dynamicrafter.yaml` pairs the DC adapter with a *DC* base; the
novelty here is DC-adapter + WAN-base).

## Design fork (decide before building)

- **(a) Architecture-only DC UNet** (random/scratch init) — cleanest test of the
  *capacity* hypothesis; **recommended start.**
- **(b) Pretrained-DC-initialised** (`ckts/dynami512.ckpt`, 10GB) — leverages DC's
  video prior, but that prior lives in a different latent space → transfer value
  questionable, and it's heavy. Defer.

## Scope / cost

WAN 5B frozen + UNet-scale adapter is large — watch 24GB (3090) VRAM with base
offload. Expect a bigger, slower run than the AVID-adapter baseline.

## Scoping outcome (2026-07-13) — no bridge code needed, it IS a config

Verified against `adapters/factory.py` + `adapters/output/dynamicrafter.py`:

- The factory already routes `adapter.extra.backbone: unet` →
  `DynamicCrafterOutputAdapter` (requires `unet_config_path`).
- **The feared "WAN↔DC latent bridge" is a channel setting, not a projection
  layer.** `DynamicCrafterOutputAdapter` reads `in/out_channels` straight from the
  UNet config; `openaimodel3d.UNetModel` is fully-convolutional in the channel dim,
  so pointing them at the Wan2.2-VAE latent width (z_dim = **48**) *is* the bridge.
  With `condition_on_base_outputs: true` the adapter auto-bumps in_channels 48→96
  (x_t ⊕ base_output). No new code.
- **Capacity caveat:** the current AVID Wan adapter is `wan_adapter_34m` (34M), so a
  34M DC-UNet would NOT test the capacity hypothesis. Config is based on the
  **145M**-tier DC UNet (`model_channels: 96`).

**Configs written (in the code repo, draft — NOT yet smoke-validated):**

- `configs/base/act_cond_diffusion_wan48_145M.yaml` — 48-ch, context-free DC UNet
  (145M-tier). Deltas vs `act_cond_diffusion_145M.yaml`: in/out_channels 8/4→48/48,
  temporal_length 16→11, `image_cross_attention`+`addition_attention` → false.
- `configs/diffusion_wan22_dcunet_output_metaworld.yaml` — copy of the AdaLN
  baseline `diffusion_wan22_avid_i2v_metaworld.yaml` with ONLY the `adapter` block
  swapped to `backbone: unet`. Scratch init (fork **a**); everything else held
  fixed for a directly-comparable base-vs-adapted delta.

## Steps

1. ~~Verify feasibility / find the projection gap~~ **done — see scoping outcome; no
   bridge code needed.**
2. **Smoke-run** the random-weight path first — it must confirm THREE things (see
   the base UNet config header):
   (i) spatial latent dims divide the UNet downsample factor (8 for `channel_mult
   [1,2,4,4]`; drop to `[1,2,4]` if a non-square Wan `best_output_size` latent
   fails); (ii) spatial-transformer cross-attn tolerates `context=None` (self-attn
   fallback) with `image_cross_attention: false`; (iii) the wan22 training path
   routes the action condition-encoder output to the adapter's `embedding` key.
3. Train; compare action-following + fidelity vs the AVID-adapter baseline
   (base-vs-adapted delta, NFE-row grid). Quantify (hard rule 8). **Match the AVID
   run's step budget + MetaWorld subset** — the xattn run was killed @ 2661 steps on
   a different subset, so a short/off-subset run won't distinguish capacity from
   training-length.

## Smoke-test outcome (2026-07-13) — training step RUNS; it was NOT "just a config"

Ran `scripts/train_wan22_i2v_metaworld_external.py` with the 145M config. The model
builds (**133.8M trainable / 5.13B total = 2.61%**), the 128px source resizes to
Wan-native 768×768 → **11×48×48** latent (48 divides the factor-8 downsample ✓), and
**a training step completes end-to-end: `loss≈0.48` at step 1** (adapter forward +
shortcut self-consistency + flow loss all fine).

Getting there needed **three vendored-DynamiCrafter code fixes** (not config) —
flagged in [[../10_now/architecture]] §"Vendored code boundary". The DC UNet was
never written for a WAN diffusion-forcing base:

1. **Per-frame timesteps.** DC UNet assumed one scalar timestep/sample broadcast
   across frames; WAN diffusion forcing feeds per-frame `[b, t]` (clean obs frame @
   t=0). `openaimodel3d.forward` now handles both.
2. **`context=None` guard** in the UNet's context-reshape block.
3. **Skip `attn2` (cross-attn) when no context** — attn2's `to_k/to_v` are
   `context_dim`-sized so it can't self-attend; block → self-attn + FF. Plus the
   adapter drops the WAN base's *list* text-context → None.

**(This retires the "no bridge code needed" claim above: the channel bridge was
free, but the timestep + context integration was not.)**

### Remaining blocker: VRAM (expected — this is the "watch 24GB" risk)

After step 1, the per-step Wan-VAE **encode at native 768×768** hit
`torch.OutOfMemoryError` — tried 4.75 GiB with **4.19 GiB free** on the 23.56 GiB
3090 (only ~0.56 GiB short). Not a bug — the flagged capacity/VRAM cost. Knobs, in
rough order of least-perturbing:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (gap is tiny — may suffice).
- Swap to the **34M / 11M** tier adapter (`act_cond_diffusion_wan48_{34M,11M}.yaml`
  already written; 11M is factor-4 downsample, cheapest).
- Lower `max_area` (512²=262144 / 384²=147456) — but the base goes off-distribution
  ("washed") below native, so prefer this only for a smoke run, not the real deltas.
- Confirm base-DiT offload is on during the VAE encode/decode.

**Resolved (2026-07-14):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
fixed it — training ran cleanly through 10+ steps + evals firing on the 11M tier.

### Per-frame action conditioning added + validated (2026-07-14)

Implemented AVID-style per-frame additive action conditioning (previously the
action was aggregated/summed into one vector broadcast to all 11 frames —
diverging from original AVID's per-frame `[B,T,A]`). New `action_per_frame` flag,
default False (aggregated). Smoke-validated end-to-end on `_external.py`:
`action=per-frame[B,11,A]`, training steps run, loss descending. **Only wired
into `train_wan22_i2v_metaworld_external.py`** — see
[[bug-infra-wan-script-provider-mismatch]], the plain script doesn't read this
flag at all. Full detail: [[../10_now/architecture]] §"Action injection:
aggregated vs per-frame".

### Paired diagnostics added (2026-07-14)

`training/trainer.py`: `denoise_adapter_delta` (per-step, paired base-vs-adapted
denoising loss on the same batch — cancels sampling noise) and
`probe_denoise_{base,adapted,delta}` (same, on a frozen low-variance probe batch,
logged every eval). Motivated by the observation that the proper-base denoising
loss is only weakly decreasing (SNR ≈0.25 over ~4700 steps) vs. a broken-base
control run that clearly learns (SNR ≈1.9) — these metrics isolate the adapter's
marginal contribution from the aggregate loss's near-floor noise. See
`AdaptedModel.forward(..., return_base=True)` in `models/adapted_model.py`.

### Wide diagnosis: why is the adapter under-learning? (2026-07-14)

Ran a 21-agent investigate→verify→synthesize exploration into why the (Wan and
DC-UNet) adapters under-learn and how to strengthen the action signal. Full
write-up, ranked causes, and a concrete do-now experiment order:
[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]. Headline: **not one
cause** — a low-headroom regime (expected) stacked on top of two independent,
confirmed bugs: gate saturation
([[bug-adapter-gate-saturation-mask-mix]], ~50× gradient throttle) and a
training-script/provider mismatch
([[bug-infra-wan-script-provider-mismatch]]) that also silently drops the
per-frame action fix above on most configs. **Also invalidates the standing
"cross-attention didn't help" conclusion** — see the correction on
[[feat-adapter-wan-action-cross-attention]].

**Before running the real DC-UNet capacity experiment on this ticket:** apply
the gate-saturation fix and confirm the correct script/provider is used
(`_external.py`, not plain) — both confound the capacity question the same way
they confounded the cross-attention question.

### "Crazy experiment" — full composition override, smoke result (2026-07-14)

User's idea: keep the base's prediction as the adapter's **input**
(`condition_on_base_outputs: true`, unchanged) but let the adapter's output
**fully replace** the base at composition (`composition: replace` — an
existing, previously-unused branch in `AdaptedModel._compose`, zero new code).
Isolates "does gradient flow at all" from "is the mask_mix gate throttling it,"
without discarding the base's information the way the from-scratch control run
did. New config:
`configs/diffusion_wan22_dcunet_replace_metaworld.yaml` (34M-tier DC UNet — the
adapter now has to reconstruct the *whole* field, not a residual, so sized up
from 11M; `output_mask: false`, no gate needed under `replace`).

**Smoke-validated, ~15 steps, not a citable run (no wandb id captured — short
diagnostic only):** 25.9M trainable params, loss starts high as expected (no
zero-init on the DC UNet's final conv → an untrained full prediction, not a
small residual) at **1.85**, and **descends briskly and monotonically to 1.55
by step 15** (steps: 1.85→1.84→1.82→1.81→1.79→1.77→...→1.55). `eval_probe_denoise_base`
is stable ~0.04–0.05 across evals (the frozen base's own loss on the fixed
probe, as expected). `eval_probe_denoise_delta` (base − adapted) starts very
negative (−1.72, since an untrained full-override adapter is far worse than
the base at step 5) but is **shrinking fast**: −1.72 → −1.61 → −1.51 over just
10 eval steps.

**Read (early signal only, 15 steps ≠ a result):** this is a much higher-SNR
loss curve, much faster, than the mask_mix run's near-flat descent over
thousands of steps — consistent with gate saturation being a real, significant
contributor to the weak learning signal. Needs a real, longer, wandb-logged
run to confirm the trend holds and to see whether `probe_denoise_delta`
actually crosses zero (adapter matches/beats the base). **Not a proposed
architecture** — `replace` throws away the plug-and-play composition that is
the thesis's actual contribution; this run is diagnostic only, per the config's
own header.

## Unblocked (2026-07-15) — the real capacity experiment can now run

`diffusion_wan22_dcunet_output_metaworld.yaml` (the real `mask_mix`
composition, not the `replace` diagnostic above) is fixed:
`gate_bias: 4.0 → 0.0`, `grad_accum_steps: 4`, `linear_warmup_steps: 250`
added, matching the AVID-validated settings. The actual capacity-vs-gate
experiment this ticket was blocked on: [[experiments/exp-adapter-dcunet-gatelow-capacity-run]].
