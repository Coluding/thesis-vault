---
type: exp
scope: shortcut
status: in-progress
priority: high
created: 2026-08-02
updated: 2026-08-02
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-d3-fewstep-vs-noshortcut-control]]", "[[../bug-eval-stepsize-probe-runs-in-train-mode]]", "[[../../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]", "[[../../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]", "[[../../00_Inbox/2026-08-01-effect-rel-is-a-gain-metric]]"]
---

# exp: step-size blindness — the D3 analog of the action-blindness probe suite

## Why

D3 claims one adapter serves **any** step budget `d`. That claim needs the same
scrutiny D2's action claim got — and right now it has far less.

**Live evidence that this is not hypothetical:** the D3 shortcut arm
(`zw5wd1p6`, `dc-shortcut-fewstep-d3`) reads at step ~200:

```
eval_stepsize_effect_rel = 0
eval_stepsize_cos        = 0.9999999
eval_stepsize_base_null_violation = 0     # valid now, post-fix
```

Perfectly step-size blind. Early in training, so not yet a verdict — but the
metric is finally trustworthy ([[../bug-eval-stepsize-probe-runs-in-train-mode]])
and this is the number that decides whether few-step is real or fake.

## What exists vs what is missing

| | actions (D2) | step size (D3) |
|---|---|---|
| sensitivity + null | ✅ `effect_rel`, base-null | ✅ `eval_stepsize_effect_rel`, null (fixed 08-01) |
| **embedding pedestal** | ✅ `probe_action_embedding_standalone.py` — found the 48× pedestal | ❌ **never checked** |
| **structure / control** | ✅ steering, temporal, spatial (`evaluation/action_structure.py`) | ❌ **none** |
| **monotonicity** | n/a | ❌ **none** |

## The three probes being built

1. **Step-level embedding pedestal.** Direct analog of the probe that cracked
   D2: measure `realised/RMS` of the step-level embedding across `d`. DC's
   *action* embedding was 14× the time embedding but **99.7% constant**
   (0.005 vs the reference's 0.238). The step-level path is the same shape — a
   small MLP (`step_level_hidden_dim: 64`, `step_level_transform: log2`) — so
   the same failure is available to it and has never been looked for.
   CPU-only if possible.

2. **Step-size structure (`consistency_cos`).** `effect_rel ≠ 0` says the
   prediction *moved*, not that it moved *correctly* — exactly the D2 trap,
   where Wan had non-zero `effect_rel` and chance-level structure on all three
   axes. Measure whether `pred(2d)` moves toward the two-half-step composition
   `½·s(x,t,d) + ½·s(x_{t+d},t+d,d)`:

   `consistency_cos = cos( pred(2d) − pred(d), target_2d − pred(d) )`

   ≈ 0 ⇒ **step-size sensitivity without step-size control** — the few-step
   claim is fake even if quality looks fine. Must reuse the codebase's own
   `compute_self_consistency_target_v` / `shortcut_consistency_target`
   (`endpoint_inversion` for diffusion, `v_average` for flow) rather than
   re-deriving; a verbatim cross-model port was mathematically dead once
   already.

3. **Monotonicity across the dyadic ladder.** `effect_rel` is a *max* over
   levels, so it cannot distinguish a systematic response to `d` from a random
   one. Report ‖pred(d) − pred(d_ref)‖ over d ∈ {1,2,4,8,…} and whether it is
   monotone.

All three require a **frozen-base null at chance by construction**, and must
flag a zero-response adapter as `degenerate` rather than scoring it as a
plausible number.

## Run plan

Job `jobs/experiments_cluster/acwm_phys/shortcut/submit_probe_stepsize_blindness.sh`
runs all three on the retained checkpoints of **both** D3 arms, answering
directly: *does the shortcut objective actually buy step-size conditioning, or
does the control match it?* Built but **not submitted** — the D3 arms are
mid-flight and it wants review first.

## Decision rule

- **shortcut arm above chance on 2+3, control at chance** ⇒ D3 has a
  mechanism, and the few-step quality curve has something behind it.
- **both at chance** ⇒ few-step is degenerate collapse; any quality gap is
  coming from somewhere other than step-size conditioning, and the D3 claim
  must be rewritten.
- **pedestal detected in probe 1** ⇒ the same fix family as D2
  (`condition_center`-style decorrelation) applies, and D3's blindness has the
  same root cause as D2's.
