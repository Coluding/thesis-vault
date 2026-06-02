---
type: feat
scope: shortcut
status: done
priority: high
created: 2026-05-25
updated: 2026-05-25
resolution: shipped
resolution_note: Implemented in-session 2026-05-25; schedule + transform tests green (88 tests pass).
closed_at: 2026-05-25
related:
  - risk-shortcut-eval-steplevel-out-of-distribution.md
  - feat-training-separate-loss-logging-and-multistep-eval-grid.md
  - ../30_Knowledge/related-work/shortcut-models.md
  - ../30_Knowledge/tech/shortcut-training-modes.md
---

# Configurable, paper-faithful shortcut step-size schedule + step_level embedding transform

Implements options **B + C** from
[[risk-shortcut-eval-steplevel-out-of-distribution.md]]: a single config-driven
schedule of step sizes on a normalised scale, plus a well-conditioned embedding
of that scale.

## Canonical scale: normalised `s ∈ (0,1]`

Step sizes are now canonicalised to a **fraction of the trajectory** (paper
convention: `s=1` = one-step, `s=1/128` = finest), decoupled from `T=1000`.
This fixes the unit/magnitude problems the risk ticket identified — train and
eval share one scale, and `d=1` hits 1-step exactly regardless of `T`.

## Config (`training.extra.shortcut_step_schedule`)

```yaml
shortcut_step_schedule:
  units: normalized      # normalized (0,1]  |  timesteps [1, T] (divided by T)
  mode: log2             # log2 | explicit | uniform
  min: 1/128             # log2/uniform bounds, in `units` (fraction strings ok)
  max: 1
  base: 2                # log2 only
  # values: [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]   # explicit only
```

Bounds/values accept **readable fraction strings** (`1/128`) as well as plain
ints/floats and decimal/scientific strings — YAML parses `1/128` as a string,
which `_to_float` converts via `fractions.Fraction`.

- **log2** → dyadic ladder min→max (the AVID config uses `1/128 … 1` = the
  paper's 8 sizes).
- **explicit** → an arbitrary list (in either unit system).
- **uniform** → continuous sampling in `[min, max]`.
- `units: timesteps` lets you still write `1,2,4,…,512`; they're divided by `T`.

One schedule drives **both**:
- **Training** (`trainer._maybe_prepare_shortcut`, distillation): sample `s`
  per batch; finest level = anchor/grounding (standard loss); larger `s` →
  self-consistency vs. two chained `s/2` calls, with `s/2` converted to a
  timestep jump `round(s/2·T)` for the DDIM micro-step.
- **Eval grid** (`trainer._eval_step_schedule`): each level `s` →
  `num_steps = round(1/s)`, injecting `step_level = s`. Explicit
  `eval_step_schedule` still overrides (e.g. to probe untrained levels).

## Embedding transform (fixes the `Linear(1,·)` asymmetry)

`step_level_transform` on the adapter (`output/dynamicrafter`, `hyperalign`):
- `linear` (default, back-compat) — feed `s` directly; now bounded since `s∈(0,1]`.
- `log2` — feed `log2(s)`, spreading a dyadic ladder into ~`[-7,0]` instead of
  crushing the fine end near zero. AVID config uses this.

Applied in `conditioning/utils/dynamicrafter_conditioning._apply_step_level_transform`;
threaded through the factory (default `linear`).

## Back-compat

No `shortcut_step_schedule` block → legacy raw-timestep dyadic path
(`shortcut_step_level_max`, `_sample_dyadic_d`) is unchanged, and
`step_level_transform` defaults to `linear`. Existing configs/runs are
bit-identical.

## Touch points

- **new** `training/step_schedule.py` — `ShortcutStepSchedule` (parse / sample /
  `to_timestep_jump` / `discrete_levels`).
- `training/trainer.py` — parse + cache `self.step_schedule`; schedule-driven
  distillation branch; schedule-derived eval grid.
- `conditioning/utils/dynamicrafter_conditioning.py` — `_apply_step_level_transform`,
  `step_level_transform` param.
- `adapters/output/dynamicrafter.py`, `adapters/hypernetworks/hyperalign.py`,
  `adapters/factory.py` — thread `step_level_transform`.
- `configs/diffusion_avid_shortcut_metaworld.yaml` — paper-faithful schedule
  (`log2 1/128…1`) + `step_level_transform: log2`.
- **tests** `tests/test_step_schedule.py` (new) + `tests/test_video_logging.py`
  (schedule-derived grid). 88 tests pass.

## Still pending (empirical, not code)

Run training with the full schedule and inspect the eval grid — does the
adapter now degrade gracefully at few steps? Watch the collapse modes
([[risk-shortcut-self-consistency-collapse.md]]) at large `s`. This is the
remaining item on the risk ticket.
