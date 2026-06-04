---
type: tech-note
status: living
last_updated: 2026-06-04
sources:
  - "code: src/generative_flow_adapters/training/trainer.py"
  - "code: src/generative_flow_adapters/training/metrics_logger.py"
  - "code: src/generative_flow_adapters/training/checkpoint.py"
  - "code: src/generative_flow_adapters/training/builders.py"
  - "code: scripts/train_avid_shortcut_metaworld.py"
relevance: T1  # core training infra
---

# Training run I/O — JSONL metrics, checkpoints, periodic eval

Before this, the trainer logged only to wandb + stdout; nothing was persisted
for offline runs and there was no checkpointing or held-out eval. Added three
opt-in, config-driven facilities that no-op unless `training.output_dir` is set
(so existing smoke/wandb-only runs are unaffected).

## What lands where
`output_dir/`
- `metrics.jsonl` — one JSON object per line. Every training step plus each eval
  cycle. Fields: `step`, `split` ("train"/"eval"), `wall_time`, and all scalar
  metrics (the `generated_samples` tensor is dropped, same filter as wandb).
- `checkpoints/`
  - `step_XXXXXXXX.pt` — step cadence, rotated to the most recent
    `keep_last_checkpoints` (None = keep all).
  - `best.pt` — written whenever the eval `eval_metric` improves. Never rotated.
  - `final.pt` — last state at end of `train()`. Never rotated.

## Config (TrainingConfig)
`output_dir`, `log_metrics_jsonl` (default True), `checkpoint_every_n_steps`,
`keep_last_checkpoints`, `eval_every_n_steps`, `eval_num_batches` (default 8),
`eval_metric` (default `base_loss`).

## Design decisions
- **Checkpoints store only trainable weights** (`requires_grad` params: adapter +
  condition encoder) + optimizer state — the frozen base backbone is rebuilt
  from its own pretrained checkpoint, so writing it every cadence is pure waste.
  Load with `strict=False` (`CheckpointManager.load`); the frozen-base keys show
  up as "missing" and that is expected. This is the PEFT-style pattern.
- **Best-checkpoint selection uses `base_loss`, not the total loss.** The
  self-distilled shortcut/consistency terms can collapse toward 0 without the
  model actually improving (see [[risk-shortcut-self-consistency-collapse]]), so
  the honest denoising loss is the safer selection signal. Configurable via
  `eval_metric`.
- **`evaluate()` reuses the exact training forward+loss path** (`_forward_and_loss`,
  extracted from `training_step`) under `model.eval()` + `no_grad`, so eval
  numbers are directly comparable to train. It averages over `eval_num_batches`
  batches; because timesteps/noise are resampled each call the eval loss is
  noisy — averaging over several batches is what makes the comparison meaningful.
- A **final eval** runs at the end of training unless the last step already hit
  the eval cadence (guard avoids double-compute).
- The preprocessor is now called with `train=False` for eval batches (disables
  CFG condition dropout etc.).

## Wiring
`builders.build_experiment` constructs the `JsonlMetricsLogger` +
`CheckpointManager` from config (`_maybe_build_run_io`) and exposes them on
`ExperimentComponents`. `Trainer.__init__` takes them as optional kwargs;
`Trainer.train(..., eval_loader=...)` orchestrates the cadences. Backward
compatible — all existing `Trainer(...)` call sites (positional, no loggers)
keep working.

`scripts/train_avid_shortcut_metaworld.py` is the reference wiring: CLI flags
`--output-dir --eval-every --eval-batches --checkpoint-every
--keep-last-checkpoints --val-fraction --resume` override the YAML, and it builds
the held-out eval loader via `random_split`.

## ⚠️ Caveat — eval split is in-distribution, not leak-free
The AVID script splits at the **window** level with `random_split`. Adjacent
sliding windows from the same episode can land on opposite sides of the split,
so `eval_*` tracks progress / mild overfitting but is **not** a clean held-out
test split. A leak-free split would need an episode-level partition in the
dataset/translator — open follow-up if rigorous generalization numbers are
needed for the thesis.

## Resume
`Trainer.load_checkpoint(path)` restores trainable weights + optimizer +
`global_step` + `best_eval_metric`. Exposed in the AVID script via `--resume`.
