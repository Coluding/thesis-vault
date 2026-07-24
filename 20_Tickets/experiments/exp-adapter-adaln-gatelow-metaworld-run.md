---
type: exp
scope: adapter
status: open
priority: high
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[../bug-adapter-gate-saturation-mask-mix]]", "[[../feat-training-grad-accumulation-warmup]]", "[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]"]
---

# exp: AdaLN Wan adapter, all three confounds removed — the core validation run

**Config ready:** `configs/diffusion_wan22_avid_gatelow_metaworld.yaml`.
Code-mechanics smoke-validated 2026-07-15 (wandb `7uakyuad`, ~10 steps —
grad accumulation + LR warmup confirmed working correctly, no crash). **Not
yet run for real** — that's this ticket.

## Why

The single most important next experiment. Combines all three confirmed
confounds' fixes together on our own AdaLN Wan adapter:

- `gate_bias: 4.0 → 0.0` ([[../bug-adapter-gate-saturation-mask-mix]])
- `grad_accum_steps: 4`, `linear_warmup_steps: 250` (effective batch 8,
  matching the AVID reference run — [[../feat-training-grad-accumulation-warmup]])
- Pre-training baseline eval now runs automatically (no config change needed)

This is the controlled retest of our own adapter/composition against the
positive control already established:
[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]
(real AVID code, our MetaWorld data, `init_mask_bias: 0.0`, healthy ~9.5×
loss drop + actively-moving gate over ~800 steps).

## Procedure

Run to at least the same step count as the reference comparison (~800+ steps,
ideally further since our loss started at a much rougher point — see
"expected behaviour" below). Log to wandb. Prerequisite:
[[exp-training-single-clip-overfit]] should pass first.

## Decision rule

- **`train/loss` / `eval_base_loss` descend cleanly, high SNR** (comparable
  shape to `pg3x72uc`'s trajectory) ⇒ gate saturation + optimization-SNR
  confounds were the dominant cause. Update
  [[../../50_Decisions/open/action-conditioning-injection-mechanism]] and the D2
  default-adapter decision with this as real evidence.
- **Still flat/noisy** ⇒ the confounds fixed here weren't sufficient — next
  suspect is the deeper per-frame-AdaLN-collapse issue
  ([[../feat-adapter-wan-per-frame-adaln]]), or something not yet found. Don't
  skip straight to that big lever without this result in hand — it changes
  how urgently it needs building.

## Expected behaviour (not a bug if you see it)

At `gate_bias=0.0`, the composed prediction starts as a genuinely poor 50/50
blend (unlike `gate_bias=4.0`'s near-identity "mostly base" cold start) — the
smoke test already showed loss jumping around 4–13 in the first 10 steps
(before warmup even reaches 5% of target LR). Don't read early noise as
failure; the AVID reference comparison also needed real training time to show
its trend clearly.

## Related

- [[exp-adapter-xattn-gatelow-metaworld-run]] — the cross-attention sibling run
- [[../../10_now/training-hyperparameters]] — full hyperparameter reference
