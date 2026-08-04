---
type: bug
scope: adapter
status: open
priority: high
created: 2026-08-04
updated: 2026-08-04
resolution:
resolution_note:
closed_at:
related: ["[[bug-adapter-gate-cap-equals-init-freezes-gate]]", "[[../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]", "[[../30_Knowledge/experiments/20260803-concat-injection-does-not-help]]"]
---

# bug: the gate learns INTO `gate_cap` and freezes there — campaign-wide

## Symptom

Four of five Wan arms sit at the cap (`gate_cap: 0.9`), with `gate_std`
collapsing toward zero as they train:

| run | `gate_mean` | `gate_std` | `rel_contrib` | `effect_rel` |
|---|---|---|---|---|
| TOKENNORM (oracle) `52o3uxz8` | 0.89789 | 0.009577 | 0.047 | 0.00619 |
| TOKENNORM-NOBASE `vy9tcuco` | 0.89883 | 0.008479 | 0.058 | 0.00771 |
| GATEFIX `tny84p7k` | 0.89979 | 0.002165 | 0.040 | 0.00197 |
| CONCAT `6ruz55f6` | **0.89990** | **0.000949** | 0.059 | 0.00975 |
| ACTIONONLY-blind `gyy817pl` | **0.900000** | **0.000000** | 0.082 | 0.00213 |
| SIMPLE `7bmzwv6u` | **0.2997** | 0.056841 | 0.032 | 0.00220 |

## Mechanism

`mask_mix` blends `base·σ(gate) + adapter·(1−σ(gate))`, and **"gate → 1 means all
base"** (`models/adapted_model.py:64`). At the cap the adapter contributes ~10%
— and, per that same comment, **its gradient is scaled by `(1−gate)` = 0.1**, so
it learns an order of magnitude slower than an ungated arm. Then
`clamp(max=gate_cap)` zeroes the gate's own gradient once it reaches the bound,
so **it cannot come back down**: an absorbing boundary.

This is the **same bug class** as
[[bug-adapter-gate-cap-equals-init-freezes-gate]], reached by *learning into* the
cap rather than being born on it. Our guard only checks `gate_cap > σ(gate_bias)`
at init, so it never fires.

⚠ Uncapping is **not** the fix — the cap is the countermeasure. Removing it lets
the gate reach 1.0 = the adapter fully suppressed. The fix is to remove the
opt-out: `composition: add`, or a much lower cap.

## Why this matters beyond one run

It is uniform across cross-attention, concat, adaLN and token-norm. So it is an
**alternative explanation for the campaign's central puzzle** — that every
injection-site experiment landed in the same 0.006–0.011 `effect_rel` band. The
current write-up attributes that convergence to a shared *data* ceiling
(conditional-variance argument). A throttled adapter would produce the same
convergence for an entirely different reason.

**The two are distinguishable**, and the test is running: ungated arms
`25192313` (token-norm NOBASE + `add`) and `25192286` (action-only + `add`).

- ungated clears the band ⇒ the ceiling was **partly the gate**; the shared-data-ceiling
  claim in [[../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]
  and [[../30_Knowledge/experiments/20260803-concat-injection-does-not-help]] must be
  re-qualified.
- ungated lands in the same band ⇒ the gate was not the binding constraint, and the
  data-ceiling argument survives a serious challenge.

**`SIMPLE` is the informative outlier**: the 7.5M adapter never saturated
(gate 0.30). Worth understanding — smaller adapters may not out-compete the base
hard enough to trigger the collapse, which would make the saturation a
capacity-driven pathology rather than a universal one.

## Also fixed alongside (2026-08-04)

`training/trainer.py` now logs `adapter_out_rms`, `adapter_out_mean` and
`adapter_out_const_frac` (=|mean|/rms). Under `add` the head is zero-init, so a
dead adapter sits at exactly 0 and every ratio metric is silently vacuous. These
separate three cases the campaign has been unable to distinguish: **dead**
(rms≈0) vs **constant pedestal** (const_frac≈1, the DC failure mode) vs
**genuinely varying** (const_frac≈0).

## Proposed guard

The init-time check is not enough. Warn (or fail) when `gate_std` collapses below
a threshold mid-run while `gate_mean` sits within ε of `gate_cap` — that is the
signature, and it is cheap to detect since both are already logged every eval.
