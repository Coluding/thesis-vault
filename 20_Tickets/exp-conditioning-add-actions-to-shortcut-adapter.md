---
type: exp
scope: conditioning
status: open
priority: high
created: 2026-06-04
updated: 2026-06-04
resolution:
resolution_note:
closed_at:
related:
  - "[[exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../50_Decisions/open/output-format-affine-vs-direct]]"
---

# exp: Add action conditioning on top of the step-level shortcut adapter

## Context

The current shortcut adapter conditions **only on step level `d`** — no action
conditioning. Qualitatively it already produces clean, step-count-robust
MetaWorld frames (1→25 NFE), but with actions off the right column is near-static
across rows, so the figure shows **step-invariance, not action-following**. To
make this a D4 result (action-conditioned *and* shortcut) the adapter needs to
take `a_t` as well: `f(x_t, t, a_t, d) = f_base + g(d)·Δ_φ(x_t, t, a_t, d)`.

## Goal

Jointly condition the shortcut adapter on `(step_level, action)` and verify the
predicted next-frame actually responds to the action (not just reconstructs a
static target), while keeping the step-count robustness intact.

## Setup (machinery already exists — this is wiring, not building)

- Action machinery is already in the adapter families: `action_embed`,
  `null_action_emb`, `action_dropout_prob`, `dropout_actions` in
  `adapters/hidden_states/unicon.py`, `adapters/hypernetworks/hyperalign.py`,
  `adapters/output/dynamicrafter.py`. Native action-conditioned bases receive
  `cond["act"]` through their own action head (`hyperalign.py:637`).
- Current shortcut run has `action_conditioned=False` (or actions dropped). Step:
  enable `action_conditioned`, feed `cond["act"]`, keep step-level conditioning.
- Keep action dropout (CFG-style null action) so the model still has an
  unconditional path — needed for the anchor baseline comparison and for
  action-guidance at inference.

## Things to decide / watch

- **Interaction of the two conditionings.** `step_level` and `act` both enter the
  adapter — confirm they compose (additive embeddings vs concatenation) and that
  step-level anchoring still grounds the model. _needs verification_ on which
  injection path the shortcut config uses.
- **Does the self-consistency target stay valid with actions?**
  `compute_self_consistency_target_v` chains two no-grad calls of the *adapted*
  model; the action must be held fixed across the micro-step (same `a_t` for both
  half-steps), otherwise the consistency target is ill-defined. Verify the action
  is threaded through `cond_half` identically.

## Metrics

- Action-following: does next-frame change with `a_t`? Qualitative rollout videos
  with varied actions on a fixed start frame; ideally a quantitative
  action-sensitivity metric (frame-delta vs action-delta).
- Step-count robustness preserved: MSE/quality vs NFE still flat after adding
  actions.
- No regression vs the action-free shortcut adapter at matched NFE.

## Done when

A logged run (wandb id + ckpt + commit) of the action-conditioned shortcut
adapter exists, action-following is demonstrated (qual + ideally quant), and
step-count robustness is shown to survive the addition. No numbers recorded
before the run executes (hard rule 8).
