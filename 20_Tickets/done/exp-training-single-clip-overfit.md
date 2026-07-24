---
type: exp
scope: training
status: done
priority: high
created: 2026-07-15
updated: 2026-07-21
tooling_landed: 2026-07-15
resolution: shipped
resolution_note: >
  The gate-fixed overfit finally ran 2026-07-20 (wandb uxrst2k5, gate_bias
  0.0 confirmed by logged gate_mean 0.5 = sigma(0) at step 1; wandb's
  experiment name shows the stale xattn_i2v label from a remote yaml edit).
  Decision-rule outcome: loss did NOT drive toward its floor — it converged
  to the frozen base's level (delta ~ +0.0008) while the gate saturated
  0.5 -> 0.99 in ~70 steps and adapter grad norm collapsed 4.4 -> 0.003.
  The run failed to overfit ONE clip: the copy-through attractor is an
  optimization trap, not a data-diversity issue, and balanced gate init
  does not prevent it (evidence appended to
  bug-adapter-gate-saturation-mask-mix). Per the ticket's own closing
  guidance the shuffle ablation was run to interpret this — result: total
  action-blindness. Successor experiment (removes the copy path entirely):
  exp-adapter-replace-nobase-overfit. Numbers:
  30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe.md
closed_at: 2026-07-21
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[bug-adapter-gate-saturation-mask-mix]]", "[[../experiments/exp-conditioning-action-shuffle-ablation]]", "[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]"]
---

# exp: single-clip overfit — capacity/plumbing sanity check

**Run this first**, before any of the confound-removal runs below. Cheapest,
highest-discriminating-power test in the whole debugging plan.

## Hypothesis

With `gate_bias: 0.0` + grad accumulation + LR warmup (all landed
2026-07-15 — [[bug-adapter-gate-saturation-mask-mix]],
[[feat-training-grad-accumulation-warmup]]), the adapter should now be able to
drive loss toward its floor on a **single repeated clip** within a few
hundred/thousand steps — the action is effectively a constant offset the
network only has to memorize, so this removes the low-headroom/dataset-scale
confound entirely.

## Procedure

Take `diffusion_wan22_avid_gatelow_metaworld.yaml` (or the xattn-gatelow
variant), point the data loader at **one clip repeated** (e.g. `--num-windows
1` / a dataset wrapper that always returns the same index — check what the
MetaWorld dataset builder supports for this; may need a small script flag, not
a config change). Train for a few hundred–thousand steps, watch `train/loss`
and (once shipped) `train/probe_denoise_delta`.

## Decision rule

- **Loss drives toward its floor on the single clip** ⇒ the adapter,
  composition, and gradient path are all fundamentally sound — any weakness
  seen on the full dataset is a genuine headroom/generalization issue, not a
  capacity/plumbing bug. Green light to trust the other confound-removal runs.
- **Loss does NOT descend even on one clip** ⇒ there's still a real bug
  somewhere in the adapter/composition/gradient path, independent of dataset
  scale or the confounds already fixed. Stop and re-debug before running
  anything else in this plan — the other experiments would just produce more
  ambiguous flat curves.

## Guardrails

- Use the **already-fixed** config (`gate_bias: 0.0`, `grad_accum_steps`,
  `linear_warmup_steps`) — running this on the old `gate_bias: 4.0` baseline
  would conflate "gate throttle" with "capacity/plumbing bug" and defeat the
  point.
- Cheap enough to run locally before committing cluster time to the rest of
  this plan.

## Tooling — how to run it (landed 2026-07-15)

Confirmed the "one clip repeated" setup **cannot** be done from config alone.
Config (`DataConfig`) exposes `env` / `camera` filters, but those narrow to a
*set* of demo episodes, not one; in `random` sampling `len(dataset)` = number
of episodes and `__getitem__(idx)` returns episode `idx`, so the loader draws
different clips each batch. There is no single-episode filter, and
`num_windows` is not config-reachable (unknown YAML keys fall into
`data.extra`; the builder reads it via `getattr(data, "num_windows")` → always
`None`) — it is settable only through the `--num-windows` CLI flag. So the
plan's "may need a small script flag" caveat was correct.

Added `--overfit-index N` to `scripts/train_wan22_i2v_metaworld_external.py`.
Mechanics:

- Wraps the train dataset in `torch.utils.data.Subset(dataset, [N] * repeat)`,
  so every DataLoader access aliases `dataset[N]` — the same episode each step.
- `repeat = max(batch_size, 8)`: a bare `Subset([N])` has len 1 < batch_size
  and `drop_last=True` would empty the loader (zero steps). Repeating the index
  fills full batches; content is still identical.
- Pins nothing about the *window start* by itself — pair with `--num-windows 1`
  so `_fixed_starts` returns `[0]` and the clip is byte-identical every draw
  (prints a warning if `--num-windows != 1`). Bounds-checks `N` against
  `len(dataset)`.
- **Eval/generation runs on the overfit clip itself** (`eval_dataset =
  train_dataset`). The native generation grid + all quality metrics read their
  conditioning batch from `eval_loader`, and the Trainer gates every one of them
  on `eval_loader is not None` — so an overfit run with eval disabled shows NO
  inference grid. Pointing eval at the same clip is also the ideal overfit
  signal: watch the step-size grid regenerate the exact clip it memorized. Needs
  `--eval-gen` (default on) + nonzero `inference_every_n_steps`. `want_eval =
  False` only skips the held-out *split* branches; the eval loader is still
  built from `eval_dataset` below. (Held-out loss eval also lands on that clip
  at `eval_every_n_steps` cadence — harmless, just the memorized-clip loss.)

**Invocation:**

```bash
python scripts/train_wan22_i2v_metaworld_external.py \
  --config configs/diffusion_wan22_avid_gatelow_metaworld.yaml \
  --overfit-index 0 --num-windows 1 \
  --steps 2000 --batch-size 2 --no-eval-gen
```

Note this is not a zero-loss target: the base is frozen and the clip is fixed,
but the flow-matching timestep `t` and sampled noise are still random per step,
so the adapter must learn the correct velocity field for that clip across all
timesteps — an easy target, but not a constant scalar.

Alternative (zero code): point `--hdf5` at a one-episode HDF5 — data prep, not
config.

## Result so far (2026-07-16, wandb `uea10230`, crashed @ 319 steps) — DID NOT test the intended config

**This run did not satisfy the ticket's own guardrail.** Verified via
`run.config["experiment"]` against the API (not memory): it used
`diffusion_wan22_avid_xattn_i2v_metaworld`, which on disk still has
`gate_bias: 4.0` and `shortcut_anchor_prob: 0.6` — **not** the gate-fixed
config the guardrail requires. So the "does loss drive toward its floor with
the fixed adapter" question this ticket asks is still technically
unanswered.

That said, the run is still informative: `train/denoise_adapter_delta`
collapsed from −0.45 to ~+0.0006 within ~70 steps, nearly identical in shape
and magnitude to the properly gate-fixed xattn sibling
([[../experiments/exp-adapter-xattn-gatelow-metaworld-run]], `bcipghvw`) despite the 28×
difference in nominal gate weight — meaning the composed output converged to
≈base regardless of which config was actually running. Eval quality: adapted
is **worse than base on all 6 logged metrics**. Full analysis in
[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]].

**Still pending:** an actual run of `diffusion_wan22_avid_xattn_gatelow_metaworld.yaml`
with `--overfit-index` to properly close this ticket's decision rule. Given
the gate-fixed sibling run showed the same clone-base convergence, temper
expectations — a flat/non-descending overfit loss on the *properly fixed*
config would still point at a real capacity/plumbing issue, but a
loss-converges-to-base-floor result (as seen here) would need the
shuffle/action-free ablations, not another gate tweak, to interpret.
