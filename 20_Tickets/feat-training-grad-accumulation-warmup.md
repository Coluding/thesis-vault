---
type: feat
scope: training
status: in-progress
priority: medium
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[bug-adapter-gate-saturation-mask-mix]]", "[[../10_now/training-hyperparameters]]"]
---

# feat: gradient accumulation + LR warmup in the trainer

## Why

Found via the 2026-07-15 AVID-vs-ours structural comparison
([[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] §"AVID-vs-ours
structural comparison"): `training/trainer.py` has **zero gradient
accumulation support** (`optimizer.step()` fires every physical batch) and
**no LR warmup/scheduler at all** (`training/builders.py:43-47`, flat AdamW
from step 0).

Our healthy reference comparison run (`pg3x72uc`, real AVID code on our
MetaWorld data) used `batch_size=2, accumulate_grad_batches=4` (effective
batch 8) and AVID's active 250-step linear warmup — both live in the config
chain that produced the clean, monotonic loss curve we're using as a positive
control. Comparing our own runs' loss-curve *shape* against that reference
without matching these two settings is confounded — noisier per-step
gradients (no accumulation) and full LR from step 0 onto an already-adverse
gate init could each independently make a curve look flatter/noisier than the
underlying learning actually is.

**Note:** the `accumulate_grad_batches=4` value was our own choice adapting
AVID's code to local hardware (`avid_11M_metaworld.yaml`), not part of AVID's
original published methodology (`accumulate_grad_batches=1` at their real
`batch_size=16`). The confound is real regardless — just don't cite "4" as an
AVID design choice, it's an artifact of our own config.

## What

- **Gradient accumulation:** loop N micro-batches in `Trainer.training_step()`/
  `train()`, scale loss by `1/N`, only call `optimizer.step()`/`zero_grad()`
  every N micro-batches. Add `grad_accum_steps` (or similar) to
  `TrainingConfig`.
- **LR warmup:** add a `linear_warmup_steps` config field + a `LambdaLR` (or
  equivalent) wrapping the optimizer, matching AVID's schedule
  (`ddpm3d.py:1407-1423`) for a controlled comparison.
- **Log LR per step** — currently unlogged entirely; cheap and directly useful
  once warmup exists.

## Validate

Rerun the `gate_bias: 0.0` fix ([[bug-adapter-gate-saturation-mask-mix]]) at
effective batch 8 (matching the reference-comparison run) with and without
warmup; compare curve smoothness/slope against the current (no-accum,
no-warmup) baseline and against `pg3x72uc`.

## Guardrails

- Do this **before** drawing conclusions from any gate_bias ablation — it's a
  confound on the comparison, not just an isolated improvement.
- Keep accumulation simple (fixed N, no dynamic batching) — this is a parity/
  confound-removal fix, not a new capability to over-engineer.

## Progress (2026-07-15) — code landed, awaiting smoke validation

Implemented in `training/config.py` (`grad_accum_steps`, `linear_warmup_steps`
fields) and `training/trainer.py`:

- `Trainer.__init__` builds a `LambdaLR` warmup scheduler when
  `linear_warmup_steps` is set.
- `training_step()` scales the backward-pass loss by `1/grad_accum_steps`
  (reported `loss` metric stays unscaled), only calls
  `optimizer.step()`/`zero_grad()`/scheduler-step every `grad_accum_steps`
  micro-batches, and only increments `self.global_step` on a completed
  optimizer step. Returns `metrics["optimizer_stepped"]`.
- `train()`'s post-step block (jsonl logging, `on_step` callback, print, all
  three eval cadences, checkpointing) is gated on `optimizer_stepped` so
  cadence checks don't fire once per micro-batch while `global_step` is held
  flat mid-accumulation.
- New config `configs/diffusion_wan22_avid_gatelow_metaworld.yaml` applies
  this + `gate_bias: 0.0` together, sized to match the reference-comparison
  run's effective batch (8).

Documented in the new living reference:
[[../10_now/training-hyperparameters]].

**Smoke-validated (2026-07-15, run `coluding/Wan2.2-avid-gatelow-i2v-metaworld/7uakyuad`) — both mechanics confirmed working:**

- **Accumulation:** `optimizer_stepped: 1` on every logged row (no spam from
  intermediate micro-batches); `eval_every_n_steps: 5` fired exactly at global
  step 5, confirming cadence checks are correctly keyed off real optimizer
  steps, not micro-batches.
- **Warmup:** pulled the `train/lr` history directly from wandb — a perfectly
  linear ramp, `lr = 1e-4 * step/250` to the step (0.4e-6 per step, exactly
  matching `linear_warmup_steps: 250`).
- No crash (the native-quality-eval OOM at step 5 is the pre-existing, known
  issue, unrelated to this change — degrades gracefully as designed).
- Loss is noisy at 10 steps (8.0→13.3→...→6.9), no trend yet — expected this
  early (`gate_bias=0.0` starts as a genuinely poor 50/50 blend, and warmup
  keeps LR <5% of target this early). Too early to read the science; this
  smoke test validated the code, not the outcome.

**Status: code done and verified. Ready for a real (longer, cluster-scale) run
of `diffusion_wan22_avid_gatelow_metaworld.yaml`** to see whether `gate_bias:
0.0` + these confound fixes recover the healthy learning signal seen in the
AVID reference run (`pg3x72uc`).
