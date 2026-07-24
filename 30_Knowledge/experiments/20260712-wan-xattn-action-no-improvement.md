---
type: experiment
date: 2026-07-12
config: diffusion_wan22_avid_xattn_i2v_metaworld
commit: 1c77db61fa929e9f5271df70e69948a77c3819fb
wandb_run_id: xb76ptw2
ckpt_path:                # _nv_ — not captured
status: killed            # killed at step 2661 (~15.9h runtime)
deliverable: D2
metrics:
  train_base_loss: 0.0722
  eval_base_loss: 0.0950
  eval_shortcut_direction_loss: 0.00077
  base_psnr: 18.793
  adapted_psnr: 18.739
  base_ssim: 0.8668
  adapted_ssim: 0.8454
  base_mse: 0.01320
  adapted_mse: 0.01337
  base_lpips: 0.4219
  adapted_lpips: 0.5457
  base_fvd_i3d: 2007.5
  adapted_fvd_i3d: 2133.8
  base_fid: 166.3
  adapted_fid: 409.6
notes: >
  Cross-attention (per-frame action tokens) injection — negative result on eval
  metrics. CORRECTION 2026-07-14: config never set action_seq_len, so tokens
  were unbinned/unmasked, violating the design's own precondition
  (50_Decisions/open/action-conditioning-injection-mechanism.md). Numbers are
  real; the injection-mechanism-vs-capacity causal conclusion is retracted
  pending a clean re-run. See 30_Knowledge/tech/why-adapter-underlearns-diagnosis.md.
---

# WAN xattn action injection — adapted worse than base on every eval metric

**Run:** [`coluding/Wan2.2-avid-xattn-i2v-metaworld/xb76ptw2`](https://wandb.ai/coluding/Wan2.2-avid-xattn-i2v-metaworld/runs/xb76ptw2)
· commit `1c77db61` · **killed** at `_step 2661` (~57.2k s ≈ 15.9 h).

Tests the cross-attention action-injection arm built in
[[../../20_Tickets/feat-adapter-wan-action-cross-attention]]: the action is moved
out of the AdaLN broadcast into the WAN output adapter's existing `t2v_cross_attn`
context path as per-frame action tokens (AVID output adapter, WAN2.2 frozen base,
MetaWorld i2v).

## Result — negative

Base-vs-adapted delta (wandb summary at kill):

| Metric | base | adapted | Δ (adapted − base) | direction |
|---|---|---|---|---|
| PSNR ↑ | 18.793 | 18.739 | **−0.055** | tied / slightly worse |
| SSIM ↑ | 0.8668 | 0.8454 | **−0.021** | worse |
| MSE ↓ | 0.013203 | 0.013370 | **+0.000167** | tied / slightly worse |
| LPIPS ↓ | 0.4219 | 0.5457 | **+0.124** | worse |
| FVD (i3d) ↓ | 2007.5 | 2133.8 | **+126** | worse |
| FID ↓ | 166.3 | 409.6 | **+243 (2.5×)** | much worse |

The adapter is **worse than the frozen WAN base on all six eval metrics**. Losses
behave normally (`train/base_loss` 0.072, `eval_base_loss` 0.095; per-rung
`shortcut_direction_loss` monotone: N001 0.182 → N064 0.0002) — consistent with the
standing pattern that **loss is fine, generation is the blocker**.

> **⚠️ Correction (2026-07-14):** the config never set `action_seq_len`, so the
> action tokens fed to cross-attention were the **unbinned raw per-frame
> sequence** against an 11-latent-frame query grid, with **no temporal
> masking** — violating the explicit precondition in
> [[../../50_Decisions/open/action-conditioning-injection-mechanism]] ("the
> cross-attn arm MUST use a per-frame action-token sequence" binned to the
> latent grid, else "cross-attn ≈ AdaLN and the ablation is uninformative").
> The numbers above are real and unchanged. The causal reading below them
> (cross-attn as a mechanism vs. AdaLN) is **not** established by this run —
> see [[../tech/why-adapter-underlearns-diagnosis]] for the full diagnosis
> and the corrected next step (re-run with `action_seq_len` pinned to the
> latent frame count).

## Why this matters — contrast with the AdaLN button-press run

The earlier **AdaLN** action injection
([[20260907-flow-shortcut-weak-action-signal]], button-press subset) *beat* the
base on reconstruction: PSNR 15.6 → 16.8, SSIM → 0.833, MSE → 0.021, with the gap
**widening over training**; it only regressed the perceptual/distribution metrics
(LPIPS/FVD/FID) = regression-to-the-mean blur.

Cross-attn injection **loses that reconstruction gain** (PSNR/SSIM/MSE now tied-or-
worse than base) while keeping/worsening the perceptual regression (FID 2.5× base).
So moving the action into the cross-attn context path is **not** the fix for the
weak-action-signal finding — if anything it is a step back from AdaLN.

## Confounds (honest read)

1. **Killed early** — 2661 steps only. The AdaLN run's advantage *widened with
   training*, so a short cross-attn run may understate it. Not a like-for-like
   training budget.
2. **Different data subset** — base PSNR here (~18.8) ≫ button-press base (~15.6),
   so this is a different / broader MetaWorld i2v subset, not the button-press task.
   Cross-run delta comparisons are therefore indicative, not rigorous; the valid
   read is the **within-run** base-vs-adapted delta above.
3. `ckpt_path` not captured (`_nv_`).

## Implication

Evidence for the **capacity** hypothesis over the **injection-mechanism**
hypothesis in
[[../../50_Decisions/open/action-conditioning-injection-mechanism]]: changing
*where* the action enters (AdaLN → cross-attn) did not help and removed the
reconstruction gain. Next lever is adapter **capacity** —
[[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]] (UNet-scale
`Δ_φ` on the frozen WAN base).

## Related

- [[../../20_Tickets/feat-adapter-wan-action-cross-attention]] — the build ticket
- [[20260907-flow-shortcut-weak-action-signal]] — the AdaLN baseline it regresses from
- [[../../50_Decisions/open/action-conditioning-injection-mechanism]] — the decision this informs
- [[../../20_Tickets/feat-adapter-dynamicrafter-output-on-wan-base]] — next lever (capacity)
