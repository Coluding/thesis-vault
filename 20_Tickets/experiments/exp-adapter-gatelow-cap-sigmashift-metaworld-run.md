---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-07-21
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../bug-adapter-gate-saturation-mask-mix]]", "[[exp-adapter-gatelow-nobase-overfit]]", "[[../../50_Decisions/open/second-dataset-action-informativeness]]", "[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"]
---

# exp: full-data MetaWorld gatelow + gate_cap 0.9 + sigma_shift 5.0

User-requested (2026-07-21 grilling): the full-data intervention test. Both
landed countermeasures at once — the anti-saturation gate cap and the
high-σ-concentrated training distribution — on the standard gatelow
architecture (`condition_on_base_outputs: true` kept deliberately; the
input-conditioning question belongs to the nobase overfit pair).

**Config (ready):**
`configs/diffusion_wan22_avid_xattn_gatelow_cap_sigmashift_metaworld.yaml`
(pure denoise: step conditioning + shortcut terms off; quality eval at 10
steps; launch command in header). Eval σ stays U(0,1) → loss curves remain
comparable to bcipghvw/y1jrgxqp.

## Readout / decision rule

Primary metric is NOT the loss delta — it's **action sensitivity of the
resulting checkpoint** (`--sigma-sweep --action-probe` on it):

- **Shuffle-gap becomes measurably nonzero** (vs <1e-5 on the un-intervened
  replace checkpoint) ⇒ the interventions unlock action usage on MetaWorld;
  MetaWorld is back as a candidate for D2 claim (a).
- **Still ≈0** ⇒ strongest evidence yet that MetaWorld scripted demos don't
  reward actions — locks in the ACWM-Phys move for claim (a)
  ([[../../50_Decisions/open/second-dataset-action-informativeness]]), and this
  run becomes the negative-control row for the diagnostic chapter (c).

Secondary: gate histogram under the cap (does it pin at exactly 0.9? then
the pull is still there and the cap is doing real work), adapter grad norm
staying alive past step 150 (uxrst2k5 died there).

## Sequencing

After the overfit triangle (short runs) on the remote GPU; lower priority
than the ACWM-Phys port. Prerequisite: 2026-07-20/21 code commit pulled
(eval action_seq fix, gate_cap, sigma_shift).

## Guardrails

- σ-shift is train-only; if eval losses look oddly stable vs train, remember
  the train/eval σ distributions now differ by design.
- Don't compare this run's train-loss magnitude to unshifted runs (different
  σ mix ⇒ different loss scale); compare eval losses and probe deltas only.

## Cleanup 2026-08-01 — **OBSOLETE**

Same superseded hypothesis; note also that `gate_cap` itself turned out to be a freeze trap ([[../bug-adapter-gate-cap-equals-init-freezes-gate]]).

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
