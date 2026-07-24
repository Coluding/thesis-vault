---
type: exp
scope: conditioning
status: open
priority: high
created: 2026-06-04
updated: 2026-06-05
resolution:
resolution_note:
closed_at:
related:
  - "[[exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../../50_Decisions/open/output-format-affine-vs-direct]]"
  - "[[../../50_Decisions/open/frame-stride-increase-for-action-dynamics]]"
---

# exp: Make action conditioning effective in the shortcut adapter

## Context

**Correction (2026-06-05): the shortcut adapter already conditions on `a_t`.** The
earlier assumption that it was "step-level only" was wrong — the AVID config
(`diffusion_avid_shortcut_metaworld.yaml`) already declares the `act` condition,
and the AVID adapter UNet is natively `action_conditioned: True`, so `a_t` is
passed to its action head. Yet the predicted dynamics still look **somewhat
random / not action-following** in the local runs. So this is no longer an
"add wiring" task — it's a **diagnose-why-actions-aren't-effective** task.

The target remains the D4 shape: `f(x_t, t, a_t, d) = f_base + g(d)·Δ_φ(x_t, t,
a_t, d)` where the prediction genuinely *responds* to `a_t`.

## Goal

Find out why the (already-wired) action conditioning isn't producing
action-following dynamics, fix it, and verify the predicted next-frame actually
responds to `a_t` — while keeping the step-count robustness intact.

## Leading hypotheses for the ineffective action signal

- **Actions null/dropped in practice.** Batch may not supply real `act`, or
  `dropout_actions` / eval may feed the null action — nominal conditioning only.
  _Check first: dataloader yields non-null `act`; eval conditions on the real
  action sequence._
- **Action not held fixed across the self-consistency micro-step**, so the
  shortcut target fights the action signal (see below).
- **Signal too weak / undertrained** — action head barely moves the output;
  local runs may simply be too short.

## Concrete plan (2026-06-05): local combined run first

Config added: **`configs/diffusion_avid_shortcut_action_metaworld.yaml`** in the
codebase. It is `diffusion_avid_shortcut_metaworld.yaml` (the anchor_prob=1.0
baseline) with **`shortcut_anchor_prob: 1.0 → 0.5`** and a new name/output_dir/
wandb_project — everything else held identical, so it's a clean A/B against the
baseline.

Grounded discovery while building it: the AVID adapter UNet
(`act_cond_diffusion_11M.yaml`) is **natively `action_conditioned: True`
(`action_dropout_prob: 0.0`)**, and `adapters/output/dynamicrafter.py:167-172`
already passes `act=cond.get("act")` into its action head. So action conditioning
is **architecturally wired in this config**.

**Status (2026-06-05): the earlier sharp `anchor_prob=0.5` figure was produced by
this same action-wired config** — so the local sanity check is effectively done.
The dynamics looked somewhat random there, but that run was **small data + small
batch (local)**, so undertraining/scale is the leading suspect rather than dead
actions. The committed config above makes that 0.5+action run reproducible.

Sequence: the next step is a **proper Snellius (HPC) run with more data and a
bigger batch size** — needed both to get a fair shortcut-vs-baseline comparison
*and* to see whether actions become effective at scale. If dynamics are *still*
random after a proper run, fall back to the action-effectiveness diagnostics
below.

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
