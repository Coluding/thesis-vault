---
last_updated: 2026-07-16
status: living
---

# Training Algorithm & Hyperparameter Reference

> **Living doc — update this whenever `training/trainer.py`, `training/builders.py`,
> `config.py`'s `TrainingConfig`, or a live config's training hyperparameters
> change.** This is the "what does our training loop actually do" reference —
> code layout lives in [[architecture]] §"Training"; this doc is about the
> algorithm and its knobs.

## Optimizer

Single flat `torch.optim.AdamW` over **all trainable parameters** (frozen base
excluded via `requires_grad` filtering) — `training/builders.py:43-47`.

- **No per-parameter-group LR scaling** (unlike AVID's `set_model_lr`, which
  scales LR by `num_rank * batch_size` — we don't do this).
- **Betas / eps:** PyTorch defaults (`(0.9, 0.999)`, `1e-8`) — never overridden.
- **Weight decay:** `TrainingConfig.weight_decay`, default **`0.0`**.
- **Learning rate:** `TrainingConfig.learning_rate`, default `1e-4`. Every live
  WAN22 config currently sets `1.0e-4`.

## LR schedule

- **Warmup:** `TrainingConfig.linear_warmup_steps` (added 2026-07-15). `None`/`0`
  → flat LR from step 0 (was the *only* behaviour before this date — no warmup
  existed anywhere in the codebase until now). When set, linear ramp
  `0 → learning_rate` over that many **optimizer steps** (not micro-batches),
  then flat at `learning_rate` — implemented as a `LambdaLR` constructed inside
  `Trainer.__init__` (`training/trainer.py`), stepped once per completed
  optimizer step.
- **No decay** of any kind after warmup (no cosine, no step decay) — matches
  AVID's own schedule (warmup-then-constant, confirmed by reading
  `ddpm3d.py:1407-1423`), not an oversight on our side.
- **LR is now logged** per optimizer step (`metrics["lr"]`, only when a
  scheduler is configured) — previously not logged at all.

## Gradient accumulation

Added 2026-07-15 (`TrainingConfig.grad_accum_steps`, default `1` = no
accumulation, previous behaviour unchanged). When `> 1`:

- Loss is scaled by `1/grad_accum_steps` **only for the backward pass** — the
  `loss`/`base_loss` etc. metrics reported/logged are the **true, unscaled**
  per-micro-batch values, so logged numbers stay directly interpretable.
- `optimizer.step()` / `zero_grad()` / LR-scheduler `.step()` fire only once
  every `grad_accum_steps` micro-batches. `self.global_step` — and everything
  indexed by it (`eval_every_n_steps`, `checkpoint_every_n_steps`, wandb
  logging cadence, sample generation, the `train()` progress print) — only
  advances on a **completed optimizer step**, not every micro-batch. This
  matches PyTorch-Lightning's `trainer.global_step` / `accumulate_grad_batches`
  semantics (and AVID's, since it's Lightning-driven).
- `training_step()` returns `metrics["optimizer_stepped"]: bool` so callers
  can tell which micro-batches actually stepped.

**Effective batch size = `training.extra`/CLI `--batch-size` × `grad_accum_steps`.**
There is no dedicated `effective_batch_size` field — compute it from the two.

## Gradient clipping

`TrainingConfig.grad_clip_norm` (default `None` = disabled). Applied via
`torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)` —
**note:** this passes *all* `model.parameters()`, including frozen ones, but
frozen params never populate `.grad` so this is a no-op for them in practice
(verified during the 2026-07-15 AVID comparison — not a bug).

## Precision / AMP

`TrainingConfig.extra["amp_dtype"]` (e.g. `bf16`) drives `torch.autocast` around
the **forward pass only** (`Trainer._autocast()`) — loss computation and
`backward()` run outside autocast, in full precision. This is standard practice
and was independently verified correct/symmetric against AVID's `precision: 16`
(fp16 AMP via Lightning) during the 2026-07-15 comparison.

## EMA (exponential moving average of weights)

**None.** No EMA mechanism exists anywhere in the trainer, unlike AVID's
`use_ema: True` + `LitEma` + `ema_scope` (EMA weights swapped in for
eval/sampling). Known, not yet built — see
[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] (bundled with the
optimization-SNR row, secondary priority).

## Loss objective

- **Flow matching** (WAN): `losses/flow_matching.py`. Timestep/sigma sampling:
  the diffusion-forcing preprocessor's own **flat-uniform**
  `sigma = torch.rand(...)` (`wan22_batch_preprocessor.py:128-145`) is what
  actually runs on every live `diffusion_wan22_*.yaml` config
  (`use_batch_timesteps_for_flow: true`). `FlowMatchingTrainingObjective`'s own
  logit-normal + shift-schedule sampler (`sample_timesteps`, lines 53-106)
  exists but is **dead code on this path** — see
  [[../20_Tickets/bug-losses-flow-boundary-sampling-unused]].
- **Diffusion** (DynamiCrafter/AVID-arch bases): `losses/diffusion.py` +
  `DiffusionTrainingObjective`, v-parameterisation, zero-SNR rescale optional
  per-base (`diffusion_rescale_betas_zero_snr`).
- **Shortcut self-consistency** (D3, flow and diffusion): opt-in via
  `shortcut_direction_weight > 0`, `shortcut_anchor_prob` gates anchor vs.
  self-consistency steps. No AVID analogue — AVID's loss is always a single
  denoising objective. See
  [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] for the
  not-yet-resolved question of whether this term interacts badly with a
  saturated/throttled composition gate.

## Composition (gate) hyperparameters

Not strictly "training" hyperparameters but load-bearing for how gradient
reaches the adapter — see [[architecture]] and
[[../20_Tickets/bug-adapter-gate-saturation-mask-mix]] for the full story.

- `adapter.composition`: `add` | `mask_mix` | `gated_residual` | `replace` |
  `adapter_only`.
- `adapter.gate_bias`: only meaningful for `mask_mix`/`gated_residual`. **As of
  2026-07-15, both values are live across configs** — legacy configs still use
  `4.0` (σ(4)≈0.982, ~98% base at init); newer `*_gatelow_*` configs use `0.0`
  (σ(0)=0.5, balanced — matches AVID's `init_mask_bias: 0.0`). Check the
  specific config, don't assume.
- **Gate value is now logged (2026-07-16).** `AdaptedModel._compose()` stashes
  the post-sigmoid gate tensor on `self._last_gate` (mask_mix/gated_residual
  only; `None` for add/replace) and the Trainer reads it each step —
  `adapter_gate_mean`/`adapter_gate_std` (scalar, every step) plus a full
  `adapter/gate_hist` wandb histogram (every optimizer step, subsampled to
  100k values). Shipped:
  [[../20_Tickets/feat-training-adapter-contribution-magnitude-logging]].

## Pre-training baseline eval

Added 2026-07-15. On a **fresh run** (`global_step == 0` at the start of
`train()`), a full eval cycle — loss eval, native generation grid, distribution
quality metrics (whichever are configured) — runs **before the first gradient
update**, not just at the first `eval_every_n_steps` cadence boundary (which
previously only fired *after* at least one optimizer step, so there was never
a genuinely untrained-model reference point in the logs). Skipped automatically
on resume (`global_step > 0` already). As a side effect, the lazily-captured
`self._probe_batch` (the fixed low-variance eval batch — see [[architecture]])
is now captured from the pristine, untrained model on every fresh run, not from
whatever step happened to hit the first eval cadence.

## What we log, per step

`base_loss` (or `loss`), any active consistency-loss components
(`shortcut_direction_loss` etc., also bucketed per-rung as
`shortcut_direction_loss/N{steps}`), `denoise_adapter_delta` /
`probe_denoise_{base,adapted,delta}` (paired base-vs-adapted diagnostics, added
2026-07-14 — see [[architecture]]), `lr` when a warmup scheduler is configured,
and, added 2026-07-16:

- `adapter_rel_contribution` = `‖composed − base‖ / ‖base‖` (batch-norm ratio,
  every step) — composition-agnostic "how much did the adapter move the
  output", works even for `add`/`replace` where there's no gate to read.
- `adapter_gate_mean` / `adapter_gate_std` (every step) + `adapter/gate_hist`
  (wandb Histogram, every optimizer step) — gated compositions only.
- `adapter_grad_norm` — L2 norm over `model.adapter.parameters()` only
  (excludes the frozen base and the condition encoder), captured after all
  grad-accumulation micro-steps have summed into `.grad` but before
  `zero_grad()` — the norm the optimizer is about to apply. Logged every
  optimizer step (not every micro-step).

**Not yet logged:** per-parameter statistics beyond the adapter-wide norm.

**Read-together note:** `adapter_rel_contribution` staying small while
`adapter_grad_norm` is healthy/nonzero would mean the adapter is receiving
real gradient signal but converging to a small-magnitude correction anyway —
narrows "dead gradient path" out as an explanation, leaving "found a
low-magnitude optimum" (see
[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]).
If `adapter_grad_norm` is ~0 instead, that's the dead-gradient-path signature
directly, independent of loss curves.

## Live config hyperparameters (spot-check table — extend as configs change)

| Config | `gate_bias` | `grad_accum_steps` | `linear_warmup_steps` | `learning_rate` | `grad_clip_norm` |
|---|---|---|---|---|---|
| `diffusion_wan22_avid_i2v_metaworld.yaml` (legacy AdaLN baseline) | 4.0 | 1 (unset) | none | 1e-4 | 1.0 |
| `diffusion_wan22_avid_gatelow_metaworld.yaml` (2026-07-15 controlled retest) | 0.0 | 4 | 250 | 1e-4 | 1.0 |
| `diffusion_wan22_avid_xattn_i2v_metaworld.yaml` (legacy, unbinned tokens — invalidated) | 4.0 | 1 (unset) | none | 1e-4 | 1.0 |
| `diffusion_wan22_avid_xattn_gatelow_metaworld.yaml` | 0.0 | 1 (unset) | none | 1e-4 | 1.0 |
| `diffusion_wan22_avid_xattn_replace_metaworld.yaml` | n/a (`replace`) | 1 (unset) | none | 1e-4 | 1.0 |
| `diffusion_wan22_dcunet_output_metaworld.yaml` | 4.0 | 1 (unset) | none | 1e-4 | 1.0 |
| `diffusion_wan22_dcunet_replace_metaworld.yaml` | n/a (`replace`) | 1 (unset) | none | 1e-4 | 1.0 |

**Reference comparison run** (real AVID code, not our trainer):
`avid_11M_metaworld.yaml` — `init_mask_bias: 0.0`, `batch_size=2`,
`accumulate_grad_batches=4` (effective batch 8 — this specific value was our
own choice adapting to local hardware, not AVID's original methodology),
`linear_warmup_steps: 250` (AVID's own, active).

## Related

- [[architecture]] §"Training" — code layout (where things live, not values)
- [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] — the investigation
  that motivated most of the recent additions here (grad accum, warmup, LR
  logging)
- [[../20_Tickets/feat-training-grad-accumulation-warmup]] — the ticket that
  shipped grad accumulation + warmup
- [[../20_Tickets/bug-adapter-gate-saturation-mask-mix]] — gate_bias history
- [[../20_Tickets/feat-training-adapter-contribution-magnitude-logging]] — the
  open gate/mask logging gap
