---
type: feat
scope: training
status: done
priority: medium
created: 2026-05-25
updated: 2026-05-25
resolution: shipped
resolution_note: Implemented in-session 2026-05-25; tests green.
closed_at: 2026-05-25
related:
  - risk-shortcut-eval-steplevel-out-of-distribution.md
  - ../30_Knowledge/tech/shortcut-training-modes.md
---

# Per-term loss logging + multi-step-size eval grid for shortcut training

Two improvements requested while training shortcut models.

## 1. Log each loss term separately alongside the total

**Before:** `trainer.training_step` logged only the combined `loss`, plus
`shortcut_direction_loss` via a *redundant extra forward pass* (and never
logged the local-consistency / multistep terms).

**After:** each term is captured at forward time (pre-backward, so it
reflects the params actually used) into a `loss_components` dict and
logged next to the total. wandb keys (under the `train/` prefix):

- `loss` — combined total (backward target), unchanged
- `base_loss` — diffusion/flow loss before any consistency term
- `shortcut_direction_loss`, `local_consistency_loss`,
  `multistep_consistency_loss` — each raw (unweighted) term, present only
  when its weight `> 0`

The redundant re-forward of the shortcut-direction loss was removed.
Rationale: for shortcut training the *relative magnitudes* of base vs.
consistency terms are the main signal for spotting collapse or a
mis-weighted term — invisible when only the sum is logged.

## 2. Multi-step-size eval visualization grid

**Goal (user):** during eval, test whether the shortcut adapter degrades
more gracefully than the frozen base as the number of sampling steps drops
— by sampling at several step counts and visualizing side by side.

**Implementation:** opt-in via `training.extra.eval_step_schedule`, an
ordered list of `{num_steps, step_level}` pairs (step_level optional). At
the existing eval cadence (`inference_every_n_steps`), if the schedule is
set, `trainer._generate_step_size_grid` samples the adapted model (with
`step_level` injected into cond) and the frozen base (no step_level) at
each step count, **all from one shared noise draw**, and logs a stacked
grid via the new `WandbLogger.log_step_size_grid` (rows = step counts,
cols = `gt | base | adapted`, key `eval_step_grid/sample_i`). Configs
without the schedule keep the existing single-step eval unchanged.

## Design decisions (asked the user)

- **step_level per N:** "configurable per-N list" — explicit pairs in YAML
  rather than hardcoding `1000/N`.
- **step counts:** "configurable in YAML" — `eval_step_schedule` drives
  both the counts and their step_levels; default ladder added to
  `diffusion_avid_shortcut_metaworld.yaml`: `1,2,4,8,25` →
  `step_level 1000,500,250,125,40`.

## ⚠️ Caveat that came out of this

The default eval step_levels (40–1000) are **far outside the trained range
{1,2,4}**. The grid will sample fully out-of-distribution on the `d` axis,
so few-step rows are expected to look poor until training is fixed. This
is the honest before-picture; the underlying train/eval mismatch is its
own open risk → [[risk-shortcut-eval-steplevel-out-of-distribution.md]].

## Touch points

- `src/generative_flow_adapters/training/trainer.py` — `loss_components`
  in `training_step`; `_eval_step_schedule`, `_generate_step_size_grid`;
  dispatch added to `_maybe_generate_samples`.
- `src/generative_flow_adapters/training/wandb_logger.py` —
  `log_step_size_grid`; `_format_action_block` factored out of
  `_format_caption`.
- `configs/diffusion_avid_shortcut_metaworld.yaml` — `eval_step_schedule`.
- `tests/test_video_logging.py` — `WandbLoggerStepSizeGridTest`,
  `EvalStepScheduleParseTest`. All green.
