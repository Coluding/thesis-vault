---
type: exp
scope: adapter
status: open
priority: high
created: 2026-07-21
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../done/exp-training-single-clip-overfit]]", "[[../done/exp-adapter-wan-replace-metaworld-run]]", "[[../bug-adapter-gate-saturation-mask-mix]]", "[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"]
---

# exp: single-clip overfit, replace composition, NO base-output input — trap vs capacity

Successor to [[../done/exp-training-single-clip-overfit]]. Every composition so
far converges to base-parity and a base-copy because each has a cheap route
to reproduce the base: mask_mix's gate saturates toward keep-base even from a
balanced init (`uxrst2k5`), and replace receives `base_output` as an *input*,
making identity-on-that-input a near-optimal low-effort function (`y1jrgxqp`
asymptotes at parity from below). This run removes the identity shortcut
entirely: `condition_on_base_outputs: false` — the adapter sees only
`(x_t, t, action tokens)` and must denoise on its own, on ONE memorizable
clip.

**Config (ready, in repo):**
`configs/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml` —
replace composition, no base input (verified: `ActionWanModel` input channels
drop 96→48, `base_output` ignored), shortcut terms zeroed (pure masked
denoise MSE), eval every 75 steps, σ ~ U(0,1) deliberately unshifted (isolate
the one variable; `sigma_shift` is a separate controlled change).

**Launch (remote):**

```bash
python scripts/train_wan22_i2v_metaworld_external.py \
  --config configs/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml \
  --overfit-index 0 --num-windows 1 --steps 2000 \
  --wandb-run-name overfit-single-clip-replace-nobase
```

Prerequisite: commit + pull the 2026-07-20/21 fixes (eval `action_seq`
threading, wan.py fail-loud raise, sigma_shift plumbing) on the training box.

## Decision rule

- **Mid/high-σ loss drives well below the frozen base's ~0.05** ⇒ the
  `base_output` pass-through input is confirmed as the trap; the 34M DiT has
  the capacity, the optimization just prefers the identity. Fix direction:
  remove/attenuate the copy path (detach, dropout on base_output, staged
  pretrain).
- **Loss parks at base level again (~0.05-0.08)** ⇒ with no copy path
  available, the small DiT itself can't beat the base on even one clip —
  capacity/architecture limit (per-frame timestep conditioning and
  action-latent alignment gaps are the next suspects; see
  [[../feat-adapter-wan-per-frame-adaln]],
  [[../chore-data-action-frame-alignment-audit]]).

## Guardrails

- Diagnostic only — no base at composition OR input abandons the thesis rule
  `f = f_base + g·Δ`; don't report as D2/D4 evidence.
- Single-clip overfit tests capacity/optimization, NOT action usage (one
  trajectory is memorizable from the anchor frame alone) — the
  action-informativeness question is the dataset decision, not this run.

## Cleanup 2026-08-01 — **OBSOLETE**

See the note on `exp-adapter-gatelow-nobase-overfit` — same superseded hypothesis.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
