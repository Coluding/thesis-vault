---
type: update
date: 2026-06-04
tags: [training, infra, checkpointing, eval]
---

# 2026-06-04 — JSONL logging, checkpointing, periodic eval

Added persistent training I/O alongside the existing wandb path. All opt-in via
`training.output_dir`; runs without it are unchanged.

- **JSONL metrics** (`metrics.jsonl`): per-step + per-eval scalar records.
- **Checkpoints** (`checkpoints/`): step-cadence (rotated to `keep_last`), plus
  never-rotated `best.pt` and `final.pt`. Only trainable adapter/encoder weights
  + optimizer state are stored (frozen base rebuilt from its own ckpt; load
  `strict=False`).
- **Periodic eval cycle**: `evaluate()` averages the training loss over N
  held-out batches in `eval()`/`no_grad`; `best.pt` saved when `eval_metric`
  improves. Selection defaults to `base_loss` (not the total) because
  self-distilled shortcut terms can collapse to ~0 without real improvement
  ([[risk-shortcut-self-consistency-collapse]]).

Refactor: extracted `_forward_and_loss` from `training_step` so train and eval
share one loss path. New modules `training/metrics_logger.py`,
`training/checkpoint.py`; builders auto-construct them; AVID script wires
everything with `--output-dir/--eval-every/--checkpoint-every/--resume` flags.

Full details + the in-distribution-split caveat: [[training-run-io]].

Verified: 135/135 tests pass (CPU); dummy-backbone smoke run produces correct
JSONL splits, checkpoint rotation, best-tracking, and load roundtrip.
