---
type: chore
scope: shortcut
status: open
priority: medium
created: 2026-05-15
updated: 2026-05-15
resolution:
resolution_note:
closed_at:
related: []
---

# Add pytest coverage for output shortcut training

## What's missing

No `pytest` tests under `tests/` exercise the output-adapter **shortcut training** code path end-to-end. The only thing touching the surface today:

- `examples/shortcut_training_test.py` and `examples/hyper_shortcut_training_test.py` — runnable demos, not collected by pytest. Easy to bit-rot.
- `tests/test_video_logging.py` only spot-checks that `shortcut_direction_loss` is filtered correctly by the wandb logger.

This is fine for D2 work but **D3 (shortcut adapters) is the deliverable that depends most on these paths**, and the current state is "the code runs end-to-end on a smoke script, but nothing fails loudly when one of the consistency losses is silently off."

## Surface that needs coverage

Three loss functions in `src/generative_flow_adapters/losses/consistency.py`:

- `local_consistency_loss(shortcut_prediction, one_step_target)`
- `shortcut_direction_loss(shortcut_prediction, shortcut_target)`
- `multistep_self_consistency_loss(prediction, detached_target)`

Three weighted code paths in `Trainer.training_step` (`src/.../training/trainer.py:137-154`):

- `config.local_consistency_weight > 0` + `batch["shortcut_target"]` → adds `local_consistency` term.
- `config.shortcut_direction_weight > 0` + `batch["shortcut_target"]` → adds `shortcut_direction` term **and** emits `metrics["shortcut_direction_loss"]`.
- `config.multistep_consistency_weight > 0` + `batch["self_consistency_target"]` → adds `multistep_self_consistency` term.

Target construction in `src/.../training/shortcut_targets.py` (helper that builds `shortcut_target` and `self_consistency_target` from a base prediction).

Synthetic data in `src/.../testing/fake_shortcut_data.py`.

Composition: the output-family shortcut adapters (`shortcut_direction` in `adapters/output/...`) that consume step-size conditioning.

## What the tests should assert

Minimum useful coverage, in priority order:

1. **Each loss function in isolation.** Input shape contracts, that the gradient flows w.r.t. the prediction only (`detached_target` actually detaches), and the trivial-input case (zero loss when prediction == target).
2. **Trainer step gating.** With weight=0, the corresponding term is *not* added to the loss. With weight>0 but the batch key missing, the term is silently skipped (current behaviour — assert it stays that way, otherwise we'd regress to a confusing user-facing error). With both set, the loss equals the documented sum.
3. **Metric emission.** When `shortcut_direction_weight > 0`, `metrics["shortcut_direction_loss"]` is a finite float matching the computed loss (compare against a direct call to the function, no in-place mutation).
4. **End-to-end smoke** on a tiny diffusion + output-shortcut-adapter model using `fake_shortcut_data`. Single train step, assert that:
   - `loss` is finite,
   - `model.adapter.parameters()` receive non-zero gradients,
   - `model.base_model.parameters()` receive **no** gradients (frozen base invariant — easy to break when refactoring the loss wiring).
5. **Two-step shortcut consistency.** Two consecutive `step_size=d` predictions vs one `step_size=2d` prediction should converge in the loss as training progresses on the synthetic data. One short training loop (~50 steps), assert the consistency loss strictly decreases over the first / last 10 step average.

(1)–(3) are unit-scoped (CPU, sub-second). (4) is integration-scoped but still cheap (dummy backbone). (5) is the only one with a non-trivial run cost; gate it behind a `slow` marker or env var if needed.

## Why this matters now

D3 (shortcut adapters) currently has zero regression protection beyond the example scripts. Any refactor of `Trainer.training_step`, the `LossRegistry`, or `adapters/output/shortcut_direction.py` can silently zero out one of these terms with no test failure. The thesis story for D3 depends on these losses being load-bearing — they need to fail loudly when broken.

## Plan

- Day 1: write (1) + (2) + (3) — pure unit tests, no synthetic data needed beyond `torch.randn`. Should fit in a single `tests/test_consistency_losses.py` and a `tests/test_trainer_shortcut_paths.py`.
- Day 2: write (4) using the existing `testing/fake_shortcut_data.py`. Convert one of the `examples/` smoke scripts into a pytest fixture.
- Optional follow-up: write (5) as a convergence check, marked `@pytest.mark.slow`.

## Files to touch / create

- New: `tests/test_consistency_losses.py`
- New: `tests/test_trainer_shortcut_paths.py`
- New: `tests/test_output_shortcut_training_smoke.py`
- Read-only references: `src/generative_flow_adapters/losses/consistency.py`, `src/.../training/trainer.py`, `src/.../training/shortcut_targets.py`, `src/.../testing/fake_shortcut_data.py`, `examples/shortcut_training_test.py`.

## Not in scope

- Shortcut **inference** rollout tests — separate ticket if needed (the `DiffusionInferenceSampler` doesn't currently handle step-size conditioning explicitly).
- Hyper-shortcut variant (`examples/hyper_shortcut_training_test.py`) — D3-extension, only worth covering after D3-core is locked in.
