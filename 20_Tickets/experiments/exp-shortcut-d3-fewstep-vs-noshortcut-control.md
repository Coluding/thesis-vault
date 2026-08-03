---
type: exp
scope: shortcut
status: in-progress
priority: high
created: 2026-08-01
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/storyline-experiment-requirements]]", "[[../../30_Knowledge/writing/thesis-storyline]]", "[[exp-eval-shortcut-fewstep-videos]]", "[[../bug-adapter-gate-cap-equals-init-freezes-gate]]", "[[../../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]]"]
---

# exp: D3's missing positive result — few-step quality, shortcut vs no-shortcut

## Why

D3 has **no clean positive evidence**. The June runs had poor samples; the
2026-07-29 curvature signature is **confounded** by the gate-freeze bug. The
thesis claims few-step rollout is the payoff of the shortcut objective — that
requires showing a shortcut-trained adapter **degrades gracefully as N drops
where an otherwise-identical adapter does not**.

## Arm A/B — the few-step claim (jobs 25141979 / 25141980)

Two DC action-free arms on ACWM Robot Arm, differing **only** in the two
shortcut loss weights (diff verified: 2 lines + name/outdir/project):

| | `shortcut_direction_weight` | `multistep_consistency_weight` |
|---|---|---|
| **A — shortcut** (`..._d3arm_...`) | 1.0 | 1.0 |
| **B — control** (`..._noshortcut_control_...`) | **0.0** | **0.0** |

Same architecture, same **step-level conditioning input** (so B still *receives*
the step size — it simply was never trained to be consistent across step
sizes), same data, same eval grid. Both log to wandb project
`dc-shortcut-fewstep-d3`.

**Measurement:** the configs already sweep `eval_step_schedule`
N ∈ {1, 2, 4, 8, 25, 50} with the matching `step_level`. Compare quality
(FID/FVD/PSNR/LPIPS + the `eval_step_grid` videos) **per N**, both arms,
against the frozen base's 50-step rollout as the ceiling.

**Decision rule.** Plot quality vs N for both arms:
- **A degrades gracefully, B collapses as N drops** ⇒ the D3 claim, measured.
  The gap at N ∈ {1,2,4} is the headline number.
- **Both collapse equally** ⇒ the shortcut objective buys nothing here; D3
  must be re-scoped (report honestly — it would be a real negative result
  about shortcut adapters on a frozen base).
- **Both fine at low N** ⇒ the base itself is few-step-capable on this data;
  the comparison must move to a harder setting or a stronger fast-sampler
  baseline is required.

**Also required for an honest claim (old R6, not yet ticketed):** a
**DPM-Solver / fast-sampler baseline at matched NFE**. Without it "diffusion
rollout is slow" is a strawman an examiner will name immediately.

## Arm C — the curvature re-run (job 25141988)

Flow (Wan) vs diffusion (DC) shortcut consistency, **both with live gates** —
this is the confound repair. The 07-29 comparison ran Wan with a frozen gate
(`gate_cap: 0.5` == σ(bias), `gate_std ≡ 0`) against a free DC gate, so its 68×
gap conflated curvature with composition
([[../bug-adapter-gate-cap-equals-init-freezes-gate]]).

- Wan: `diffusion_wan22_shortcut_actionfree_robotarm.yaml`, `gate_cap: 0.9`
  (fixed 2026-07-31) — the cap does not bind (DC's free gate settles ~0.60).
- DC: the Arm-A run above doubles as the diffusion side (no `gate_cap`).

**Remaining caveat to state in the chapter:** the two sides still use different
consistency targets (`v_average` for flow, `endpoint_inversion` for diffusion)
because each is the correct target for its model class. A third arm — **DC with
the naive `v_average` target** — would isolate the sagitta bias explicitly and
is the cleanest version of the curvature argument. Deferred; ticket it if the
A/B/C results warrant.

## Arm D — the FLOW control (job 25150730, added 2026-08-02)

**Gap found by the user:** shortcut models were formulated for **flow
matching**, and the thesis spine ends *DC → planning → too slow → **flow →
shortcut***. Running the few-step A/B only on DC (diffusion) would put D3's
positive evidence on the wrong side of the story.

The Wan shortcut arm (25141988) was running **with no control**, so the flow
side could only produce a degradation curve with nothing to compare against —
which is not a claim. Arm D is its matched twin:

| | `shortcut_direction_weight` | `multistep_consistency_weight` |
|---|---|---|
| Wan shortcut (25141988) | 1.0 | 1.0 |
| **Wan control (25150730)** | **0.0** | **0.0** |

Verified by a flattened-YAML key diff: **only** `name`, `output_dir` and the
two weights differ. Step-level conditioning stays ON (the control *receives*
`d`, it was simply never trained for consistency across it), same
`eval_step_schedule` N ∈ {1,2,4,8,25,50}, and **both arms log to the same wandb
project** (`Wan2.2-shortcut-actionfree-acwm-robotarm`) so the curves are
directly comparable.

This makes the D3 design a **2×2**: {DC, Wan} × {shortcut, control}. If the
few-step gap appears on flow but not diffusion — the prediction, given the
shortcut derivation is native to flow — that is a much stronger D3 result than
either backbone alone, and it also feeds the curvature comparison.

## Scope decision 2026-08-02 (user) — D3 moves to Wan; DC keeps only curvature

**Rationale:** shortcut models are formulated for **flow matching**, and the
spine ends *DC → planning → too slow → flow → shortcut*. The few-step claim
therefore belongs on Wan. DC's remaining job is the **curvature argument**
(diffusion is harder to make self-consistent).

| arm | job | status |
|---|---|---|
| **Wan shortcut** | 25141988 | **running** — few-step claim + flow side of curvature |
| **Wan control** | 25151917 | **queued** — the few-step A/B partner |
| **DC shortcut (arm A)** | 25141979 | **kept** — it *is* the diffusion side of the curvature comparison |
| DC control (arm B) | ~~25141980~~ | **killed @3h30** — served only the DC few-step A/B, which is cut |

Note arm A is deliberately retained: the curvature comparison needs a diffusion
shortcut run, and arm A doubles as it. Cutting it would remove the diffusion
half of the only D3 argument DC still carries.

**Ops note:** the Wan control's first submission (25150730) was `CANCELLED by 0`
(root) one second after starting, with no log output — a node/prolog failure,
not a job defect (our own cancels appear as `CANCELLED by 75100`). Resubmitted
as 25151917.

## Early readings (both at ~step 400, far too early to conclude)

| | `eval_stepsize_effect_rel` | `cos` | base null |
|---|---|---|---|
| **Wan shortcut** @400 | **0.0726** | 0.9975 | 0 ✓ |
| DC shortcut | step-0 baseline only (0.000) | 1.000 | 0 ✓ |

Wan already responds to `d` — but at `cos` 0.9975 the response is **nearly pure
magnitude rescaling**, not a change of direction. A shortcut adapter should move
`pred(2d)` *toward* the two-half-step composition, which is a directional
change. So the open question on Wan is not "does it respond" but "does it
respond **correctly**" — exactly what `consistency_cos` (probe 2 of
[[exp-shortcut-stepsize-blindness-probe-suite]]) measures.

⚠ Correction: an earlier reading of DC's `effect_rel = 0` as "step-size blind"
was wrong — that value is the **step-0 baseline**. DC evals every 500 steps and
had not yet logged a trained value.

## Status

Launched 2026-08-01: A `25141979`, B `25141980`, C `25141988`. Compute freed by
killing `wan-rt1-tn-nobase` (26 h, plateau established, checkpoints retained) —
the Wan mechanism branch is complete and no longer needs GPU.
