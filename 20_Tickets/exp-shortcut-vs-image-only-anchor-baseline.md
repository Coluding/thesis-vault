---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-06-04
updated: 2026-06-04
resolution:
resolution_note:
closed_at:
related:
  - "[[exp-conditioning-add-actions-to-shortcut-adapter]]"
  - "[[refactor-shortcut-deprecate-twostep-add-heun-smoothness]]"
---

# exp: Shortcut adapter vs image-only baseline (anchor_prob=1)

## Context

The shortcut adapter looks step-count robust qualitatively (clean output from
1→25 NFE). To claim that robustness comes from the **shortcut self-consistency
training** — and not just from the adapter being a good image predictor — we need
the control: an otherwise-identical adapter trained **without** the
self-consistency supervision.

## The lever: `shortcut_anchor_prob = 1.0`

From `training/trainer.py:564-568`: with `shortcut_anchor_prob >= 1.0` every
training step runs **anchor mode** — `step_level` pinned to the finest step
(`step_schedule.smallest()`), **no self-consistency target returned**, standard
diffusion/flow loss as the sole supervision. The model never sees larger step
levels and never gets a consistency target.

This is the **cleanest possible isolation**: same step-level-conditioned adapter,
same conditioning plumbing, identical architecture/backbone/seed/budget — the
*only* difference vs the real shortcut run is whether the model is ever
supervised at coarse step levels. (Preferable to zeroing the loss weights, which
would also remove the `step_level` input.)

## Arms

| Arm | `shortcut_anchor_prob` | Sees coarse step levels + consistency target? |
|---|---|---|
| Shortcut (treatment) | 0.5 (current run)      | yes |
| Image-only (control)  | 1.0                    | no — finest step + standard loss only |

Everything else identical. Same config except the one field.

## Hypothesis

The image-only control will be fine at the **finest** step it was trained on but
**degrade at low NFE** (1–4 steps), where it was never supervised — blur, drift,
or wrong dynamics. The shortcut arm should hold quality roughly flat across NFE.
If the control is *also* flat, the shortcut loss is buying little here and that's
a finding worth surfacing (and reframing) — see the affine-vs-direct concern that
the adapter may be doing all the work regardless.

## Metrics

- **MSE/quality vs NFE curve** for both arms (the headline plot): x = sampling
  steps {1,2,4,8,16,25}, y = val MSE on held-out MetaWorld rollouts.
- Qualitative side-by-side at NFE=1 and NFE=25 for both arms.
- Same seed, same step budget, same eval script across arms.

## Done when

Both arms have logged runs (wandb id + ckpt + commit) at equal budget, the
MSE-vs-NFE curves are reported for both, and we can state plainly whether the
shortcut training buys few-step quality over a plain image adapter. No numbers
recorded before the runs execute (hard rule 8).
