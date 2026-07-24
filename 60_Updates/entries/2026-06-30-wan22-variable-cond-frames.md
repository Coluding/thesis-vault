---
date: 2026-06-30
category: added
deliverable: D2
meeting:
sources:
  - "[[../../50_Decisions/decided/wan22-temporal-window-stride1]]"
  - "[[2026-06-30-wan22-stride1-121frames]]"
---

# Variable-history training: categorical distribution over cond_frames (Wan2.2 DF)

## What

Diffusion forcing lets the model condition on any number of leading clean
observation frames `k`, not just 1. Added a config-driven **categorical
distribution** over `k` so training sees mixed history lengths.

- New `training.extra.cond_frames_dist` — a `{k: weight}` map (weights must sum
  to 1; validated, raises on mismatch). Keys = number of clean observation
  *latent* frames.
- `Wan22DiffusionForcingPreprocessor` draws `k` **per-sample** (each batch row
  gets its own history length) via `torch.multinomial`, clamped to `t_lat-1` so
  at least one predicted frame always remains.
- **Train-only:** sampling fires when `train=True`. Eval (`train=False`, used by
  the trainer's logging path) stays deterministic at the scalar `cond_frames`
  (default 1) → stable, interpretable eval panel.
- Backward compatible: no `cond_frames_dist` → fixed `cond_frames` as before.

Current configs set `{1: 0.4, 2: 0.3, 4: 0.2, 8: 0.1}` (tunable).

## Why it matters

A world model that only ever sees 1 context frame can't exploit longer history
at inference (and vice-versa). Randomising `k` during training makes the model
robust to variable context length — useful for planning rollouts where the
amount of observed history varies. This is the `cond_frames > 1` revisit
anticipated in [[../../50_Decisions/decided/wan22-temporal-window-stride1]].

## Evidence

- `tests/test_wan22_i2v.py`: new tests cover per-sample mixed-`k` sampling,
  the contiguous-prefix mask invariant, clamping to `t_lat-1`, deterministic
  eval, and the sum-to-1 validation. Full wan22 suite: 17 passed.
- Verified the config loader preserves int-keyed YAML maps into
  `training.extra` and the empirical sampled distribution matches the weights.

## Files

- `data/wan22_batch_preprocessor.py` — `cond_frames_dist` arg, `_parse_cond_frames_dist`,
  `_sample_cond_frames`; per-sample mask via `(frame_idx >= ks)`.
- `scripts/train_wan22_i2v_metaworld.py` — reads `cond_frames_dist` from config.
- `configs/diffusion_wan22_{i2v,avid_i2v}_metaworld.yaml` — example distribution.

## Eval grid: history-length sweep (added same day)

The open question below is now answered in code. A new eval video grid sweeps a
configured set of history lengths and lays them out **horizontally**:

    cols = [ ground_truth | base | adapted@k1 | adapted@k2 | … | adapted@kn ]

- One GT, one **base** reference (at the eval `cond_frames`), then **one
  adapted column per k** — so 2+n columns. The swept `k` values are named in the
  caption ("subtitle").
- Config: `training.extra.wandb.eval_cond_frames: [1, 2, 4]` (also accepted at
  `training.extra.eval_cond_frames`). Omit → the old single
  `[gt | base | adapted]` panel.
- All columns share one noise draw (differences are purely conditioning-driven);
  each adapted column re-clamps the first `k` latent frames clean. `k` is clamped
  to `t_lat-1` and clamped duplicates are deduped (small clips).
- Implemented as the diffusion-forcing analogue of `log_step_size_grid`:
  `WandbLogger.log_cond_frames_grid` + `Trainer._generate_cond_frames_grid` /
  `_eval_cond_frames` / `_df_frame_mask`; dispatched in `_maybe_generate_samples`
  ahead of the step-size grid (opt-in, so shortcut configs are unaffected).

## Fix: eval grid is now 2-D (k × N), not k-only

The first cut of the cond-frames sweep **dropped the vertical step-count axis**:
because `_maybe_generate_samples` dispatches the cond grid *ahead of* the
step-size grid and returns, setting `eval_cond_frames` silently suppressed the
`eval_step_schedule` sweep — you got history lengths across columns but only a
single step count. Fixed by folding the step schedule into the cond grid so it
renders a full 2-D grid:

    rows (top→bottom)  = sampling step count N (from eval_step_schedule)
    cols (left→right)  = [ gt | base | adapted@k1 | … | adapted@kn ]

- Each **row** re-runs base + all `k` at that row's `N`, injecting the row's
  `step_level` into the adapted cond (shortcut horizon signal); the base never
  sees `step_level`. All cells still share one noise draw.
- No `eval_step_schedule` → a single unlabelled row (previous 1-D behaviour).
- Caption names both axes: `cols: gt | base | adapted@(k=1, k=2, …)` and
  `rows top→bottom: N=1, N=4, …`.
- `log_cond_frames_grid` signature changed from `(base_latents, adapted_by_k)` to
  a single `rows=[(num_steps, base_latents, adapted_by_k), …]` list;
  `_generate_cond_frames_grid` builds those rows from `_eval_step_schedule()`.
- Tests: added `test_grid_stacks_step_rows_vertically_and_names_them` (2-D
  H/W + row labels) and `test_grid_sweeps_step_schedule_as_rows` (one row per
  step count, columns preserved); existing 1-D tests updated to the `rows=` API.
  Cond-grid tests: 7 passed.

## Next

- Decide the actual training distribution for the thesis run (front-loaded vs
  uniform); current i2v config: `{1: 0.6, 2: 0.25, 4: 0.15}`.
- Read the history-length sweep panel once a real run is up: does prediction
  visibly sharpen as `k` grows? That curve is a thesis figure candidate.
