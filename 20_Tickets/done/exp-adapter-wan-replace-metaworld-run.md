---
type: exp
scope: adapter
status: done
priority: medium
created: 2026-07-15
updated: 2026-07-21
resolution: shipped
resolution_note: >
  Ran twice: 5cxstyh4 (2026-07-16, eval invalid — generation-eval
  conditioning bug, see bug-adapter-replace-generation-flat-since-init) and
  y1jrgxqp (2026-07-20/21, no-shortcut variant, FIXED eval). Final answer
  to the decision rule: gradient flow through the ungated Wan tiny-DiT
  adapter is fine (loss descends briskly, generation coherent after the
  eval fix, FID ≈ base by step 600), but it converges to a base-clone
  (delta ≈ 0 from step 900, rel contribution shrinking to 0.06) and the
  action probe shows the clone is fully action-blind. The
  "catastrophic generation" result sections below were artifacts of the
  eval bug. Successor: exp-adapter-replace-nobase-overfit. Numbers:
  30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe.md
closed_at: 2026-07-21
related: ["[[feat-adapter-dynamicrafter-output-on-wan-base]]", "[[bug-adapter-gate-saturation-mask-mix]]", "[[../experiments/exp-adapter-adaln-gatelow-metaworld-run]]", "[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]"]
---

# exp: Wan tiny-DiT adapter, full composition override — gradient-flow diagnostic

**Config:** `configs/diffusion_wan22_avid_xattn_replace_metaworld.yaml`. Not
yet run for real (mechanics untested at this scope — the DC-UNet sibling was
smoke-tested, this Wan-adapter version has not been).

## What this config tests (current state, re-verified 2026-07-15)

- `composition: replace` — the adapter's output **fully replaces** the base
  at composition; base is still given as an **input**
  (`condition_on_base_outputs: true`). No gate at all — isolates
  gradient-flow/optimization from the mask_mix throttle entirely, same spirit
  as the already-smoke-tested DC-UNet "crazy experiment"
  ([[feat-adapter-dynamicrafter-output-on-wan-base]] §"Crazy experiment").
- `action_injection: cross_attention`, `action_per_frame: false` — the
  cross-attention token binning fix is **off** here too (same as the
  gatelow-xattn sibling — see [[../experiments/exp-adapter-xattn-gatelow-metaworld-run]] for
  why this matters).
- `shortcut_anchor_prob: 1.0` — shortcut self-consistency effectively
  disabled, pure single-step flow-matching loss.

## Why — and the key difference from the DC-UNet replace run

Isolates "does gradient flow at all, absent any gate throttle" for the **Wan
tiny-DiT adapter specifically** (not DynamiCrafter). Important architectural
difference already documented in the config: the Wan adapter's final layer
**is zero-initialised** (`Wan21OutputAdapter`'s own docstring — "the delta is
~0 at init"), unlike the DC-UNet's final conv (not zero-init). So under
`replace` here, the adapter's first prediction is literally the **zero
tensor** — a different, more standard cold start than the DC-UNet run's
untrained-but-nonzero output. Expect initial loss to reflect "predict zero
velocity everywhere," then descend as the zero-init head bootstraps away from
zero — not a bug, just a different starting regime. Don't directly compare
absolute loss magnitudes against the DC-UNet replace run's numbers.

## Decision rule

- **Loss descends briskly, high SNR** (like the DC-UNet replace smoke test:
  1.85→1.55 in 15 steps) ⇒ gradient flow through the Wan tiny-DiT adapter is
  fine once ungated — strengthens the case that gate saturation, not adapter
  capacity, is the dominant issue for this backbone too.
- **Still flat/slow** ⇒ something specific to the Wan adapter's architecture
  (possibly the per-frame-AdaLN-collapse issue,
  [[feat-adapter-wan-per-frame-adaln]] — though note `replace` bypasses the
  gate, not the adapter's own internal AdaLN modulation, so that collapse
  could still bite here) is limiting it independent of composition.

## Guardrails

- **Diagnostic only** — throws away the plug-and-play composition that's the
  actual thesis contribution. Don't report as D2/D4 evidence.
- Not yet smoke-tested at all (unlike the DC-UNet sibling) — run a short
  version first (few dozen steps) to confirm it doesn't crash before
  committing real time/cluster budget.

## Result (2026-07-16, wandb `5cxstyh4`, running, 1178+ steps logged)

Ran (well past smoke-test scale). **Loss descends briskly** — matches the
first decision-rule branch on the training curve alone (`train/loss` first-20
mean 1.63 → last-20 mean 0.109, i.e. gradient flow through the ungated Wan
tiny-DiT adapter works). But the descent is to ≈the frozen base's own
single-step denoising loss (0.109 vs `denoise_base_only` 0.106), not below
it — same clone-base convergence seen in the gated runs
([[../experiments/exp-adapter-xattn-gatelow-metaworld-run]]).

**The training loss is misleading here.** Out-of-distribution probe delta is
−0.112 (vs −0.003 to −0.005 for the mask_mix runs — a 20-30× bigger true
gap), and decoded eval-video quality is **catastrophic**: PSNR 10.98 vs base
15.99 (−5dB), SSIM 0.299 vs 0.806, FID 521 vs 66 (8×), FVD 4877 vs 1260 (4×),
LPIPS 0.796 vs 0.360 (2×). A small single-step prediction gap compounds over
the 25-50-step iterative sampler into genuinely bad generation — visible only
via the probe delta / decoded-quality metrics, not `train/loss`. Full
numbers: [[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]].

**Revised interpretation vs. the original decision rule:** "loss descends
briskly" no longer cleanly means "gradient flow is fine, gate saturation was
the dominant issue." It's compatible with a *different* explanation — the
adapter finding the same clone-base local optimum as the gated runs, just
without a gate to filter how much of the (weak) clone shows up in the
composed output. Doesn't yet distinguish gradient-flow-health from
weak-action-signal as the cause; see
[[../experiments/exp-conditioning-action-shuffle-ablation]] for that.

## Result update (2026-07-18) — eval quality flat at the noise floor for the whole run

Full eval-quality history pulled: `eval/adapted/fid`/`psnr`/`ssim` at step
2400 are statistically indistinguishable from step 0 (FID 520.9 vs 530.2,
PSNR 11.3 vs 11.4, SSIM 0.33 vs 0.33) — no measurable improvement across
2400+ steps, despite `train/loss` visibly descending over the same window.
Contrast: the sibling gatelow run (same adapter `hidden_dim: 256`, same
backbone/dataset, but `mask_mix` composition with a base fallback) goes from
the same noise-level FID (483) to near-base quality (81) within 300 steps.
Leading (unconfirmed) hypothesis: the adapter is a viable residual/gating
learner but not a viable standalone generator at this capacity/step budget —
see the full writeup and prioritized investigation plan:
[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]],
[[bug-adapter-replace-generation-flat-since-init]].
