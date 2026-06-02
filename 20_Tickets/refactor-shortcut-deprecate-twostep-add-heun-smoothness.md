---
type: refactor
scope: training
status: open
priority: high
created: 2026-05-28
updated: 2026-05-28
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/deprecate-twostep-shortcut-mode]]"
  - "[[../30_Knowledge/theory/heun-smoothness-regularizer]]"
  - "[[../30_Knowledge/theory/heun-shortcut-target]]"
  - "[[bug-training-shortcut-twostep-no-stepsize-variation]]"
  - "[[bug-training-shortcut-target-timestep]]"
---

# Delete `two_step` shortcut mode; add Heun-smoothness regularizer as a separate loss

Implementation ticket for the decision in
[[../50_Decisions/decided/deprecate-twostep-shortcut-mode]]. Two
intertwined changes: removing dead/misnamed code and adding the
replacement capability under a more honest name.

## Change 1 — delete `two_step` from the shortcut training dispatch

### Code to remove

- `Trainer._compute_two_step_target_v`
  (`src/generative_flow_adapters/training/trainer.py:510-534`).
- The `two_step` branch in `_maybe_prepare_shortcut`
  (`trainer.py:372-383`).
- `compute_two_step_target_v` in
  `src/generative_flow_adapters/training/shortcut_targets.py:33-51`
  (the standalone utility).
- The `"two_step"` value from the docstring + validation of
  `shortcut_target_method` (`trainer.py:438-441`).
- `_resolve_step_level` (`trainer.py:443-473`) **iff** it is no longer
  reachable after the deletion. Currently it's only called from the
  `two_step` branch (`trainer.py:372-383`), so it goes too. Confirm
  with a grep before deletion.
- Stale `shortcut_target_method: "linear"` default in
  `TrainingConfig` (`config.py:68`) — already documented as broken in
  [[../30_Knowledge/tech/shortcut-training-modes]] "Gotchas". Set the
  default to `"distillation"` and remove the `"linear"` branch from
  `shortcut_targets.py:44-62` if it still exists after the cleanup.

### Config migration

- `configs/diffusion_avid_shortcut_metaworld.yaml:81` — change
  `shortcut_target_method: two_step` to `shortcut_target_method:
  distillation`, and add `shortcut_step_schedule` (see
  [[../30_Knowledge/tech/shortcut-training-modes]] §"Mode 2" for the
  knobs).
- `configs/diffusion_hyperalign_shortcut_metaworld.yaml:97` — same
  edit.
- Decide for each config whether the *intent* is shortcut training (→
  `distillation`) or just smoother trajectories (→ no
  `shortcut_direction_weight`, just `heun_smoothness_weight`). The
  decision note flags this as a per-config call; default to
  `distillation` since the configs have `_shortcut_` in their name.

### Tests / examples to update

- `examples/shortcut_training_test.py` — switch to `distillation`.
- `examples/hyper_shortcut_training_test.py` — switch to
  `distillation`.
- Any test asserting that `shortcut_target_method == "two_step"` is
  accepted needs to become a "rejected with ValueError" assertion.

## Change 2 — add the Heun-smoothness regularizer

### New loss function

In `src/generative_flow_adapters/losses/consistency.py`:

```python
def heun_smoothness_loss(
    current_v: Tensor,
    future_v_detached: Tensor,
) -> Tensor:
    """Discrete material-derivative penalty along the predicted
    trajectory; see theory/heun-smoothness-regularizer.md eq. (S)."""
    return F.mse_loss(current_v, future_v_detached)
```

Register in `LossRegistry`'s consistency-loss table for parity with
`local_consistency` and `shortcut_direction`.

### New trainer method

In `Trainer`:

```python
def _compute_heun_smoothness(
    self,
    *,
    x_t: Tensor,
    t: Tensor,
    cond: object | None,
) -> Tensor:
    """Returns the regularizer term L_heun-smooth from
    theory/heun-smoothness-regularizer.md."""
    if self.step_schedule is not None:
        s    = self.step_schedule.sample()
        jump = self.step_schedule.to_timestep_jump(s)
    else:
        jump = 1

    v0     = self.model(x_t, t, cond)                   # with grad
    alphas, scale_arr = self._diffusion_tables(device=x_t.device, dtype=x_t.dtype)
    prev_t = (t - jump).clamp_min(0)
    x_prev = ddim_micro_step_v(
        x=x_t, v=v0, t=t, prev_t=prev_t,
        alphas_cumprod=alphas, scale_arr=scale_arr,
    )

    was_training = self.model.training
    self.model.eval()
    try:
        with torch.no_grad():
            v1 = self.model(x_prev, prev_t, cond)
    finally:
        self.model.train(was_training)

    return heun_smoothness_loss(current_v=v0, future_v_detached=v1.detach())
```

Mirrors `_compute_self_consistency_target_v` (`trainer.py:536-565`) in
structure — same `eval()`/`no_grad` plumbing, same use of the
`ShortcutStepSchedule`.

### Wire into `training_step`

After the existing shortcut-direction block at `trainer.py:200-214`,
add:

```python
if self.config.heun_smoothness_weight > 0.0:
    heun_loss = self._compute_heun_smoothness(x_t=x_t, t=t, cond=cond)
    loss_components["heun_smoothness_loss"] = float(heun_loss.detach().cpu())
    loss = loss + self.config.heun_smoothness_weight * heun_loss
```

### Config field

In `TrainingConfig` (`config.py`):

```python
heun_smoothness_weight: float = 0.0
```

The regularizer is opt-in; default zero preserves current behaviour
for all configs that don't set it.

## Test coverage

In addition to flipping `two_step` assertions to "rejected" as noted
above, add:

1. **Unit test for `heun_smoothness_loss`**: shape contract, that
   gradient flows through `current_v` only (not through the detached
   second arg), and the zero-loss case (`current_v == future_v_detached`).
2. **Trainer-step gating test**: with
   `heun_smoothness_weight = 0`, the metric is absent from
   `loss_components`; with `> 0`, the metric is present and finite.
3. **End-to-end smoke** on a tiny diffusion + dummy adapter with
   `heun_smoothness_weight = 0.1` and no shortcut loss — assert (a)
   loss decreases over a short training loop, (b) the base model has
   no gradients (frozen invariant).
4. **Frozen-base preservation**: assert that
   `model.base_model.parameters()` see no gradients regardless of the
   `heun_smoothness_weight` value. The regularizer goes through the
   adapter only.
5. **Schedule fallback**: with no `shortcut_step_schedule` configured,
   `_compute_heun_smoothness` runs with `jump = 1` and produces a
   finite loss.

Land tests in `tests/test_consistency_losses.py` (new) and extend
`tests/test_trainer_shortcut_paths.py` (also new — see
[[chore-shortcut-add-pytest-coverage-for-output-training]]).

## Migration timeline

1. **Patch 1 — Add Heun smoothness, keep `two_step`.** Land the new
   regularizer with `heun_smoothness_weight = 0.0` default. No
   behaviour change for any existing config. Tests for the new path
   land here.
2. **Patch 2 — Migrate live configs.** Update the two shortcut configs
   to `distillation`, run a short smoke training on each to confirm
   they don't blow up. The diff is small and the configs are tracked,
   so this is a separate commit for reviewability.
3. **Patch 3 — Delete `two_step` code.** Remove the
   `compute_two_step_target_v` functions, the `two_step` dispatch
   branch, `_resolve_step_level`, and the stale `"linear"` plumbing.
   Update the validation message at `trainer.py:438-441` to list only
   `"distillation"`.

Splitting it this way keeps each commit reviewable and gives a
rollback point if (2) surfaces an unexpected interaction with the
shortcut configs' other knobs.

## Documentation updates

The vault notes have already been updated to reflect this decision:

- [[../30_Knowledge/theory/heun-shortcut-target]] §6 — proposed fix
  is now annotated as "superseded by deprecation decision."
- [[../30_Knowledge/theory/heun-smoothness-regularizer]] — formal
  derivation of the replacement (new note).
- [[../50_Decisions/decided/deprecate-twostep-shortcut-mode]] — the
  decision record.

The following still need updates **after** the code change lands:

- [[../30_Knowledge/tech/shortcut-training-modes]] — collapse to one
  active mode (`distillation`); move `two_step` to a "deprecated
  modes — historical" section and add a `heun_smoothness` section.
- [[../30_Knowledge/theory/shortcut-training]] §4.2 — rewrite from
  "Two supported regimes" to "One shortcut regime (`distillation`)
  + an orthogonal Heun-smoothness regularizer". The current §4.2
  framing of `two_step` as a co-equal option is wrong post-decision.
- [[../30_Knowledge/writing/explainer-shortcut-training]] and
  `figure-shortcut-training` — if they show the two_step target's
  Heun construction as part of *shortcut training*, relabel to
  "smoothness regularizer" or remove from the shortcut explainer
  entirely.

## Open questions (resolve during patch 1)

- **Default schedule fallback when no `shortcut_step_schedule` is
  configured.** Plan above uses `jump = 1`. Acceptable for vanilla
  diffusion runs that just want some smoothness; not particularly
  informative. Alternative: refuse to start with
  `heun_smoothness_weight > 0` and no schedule, and force the user to
  configure one. Lean toward the lenient default for ergonomics.
- **`step_level` injection inside the regularizer.** `distillation`
  injects `step_level = s` into `cond` for both `v0` and `v1`. The
  smoothness regularizer arguably shouldn't — it's a property of
  the model's velocity field at a given conditioning state, and the
  step size is purely the smoothness scale. Default: do **not** inject
  `step_level` in the regularizer; let the model see the same `cond`
  it sees during standard training. Document and revisit if there's a
  reason to.
- **Should the regularizer also be available for the
  `flow_matching_loss` path?** Yes — the construction is generic. The
  `_diffusion_tables` lookup is the only diffusion-specific piece; for
  flow matching, the predicted-endpoint integration step is direct
  (no `α_t` rescaling needed). Refactor `_compute_heun_smoothness` to
  branch on `model_type`, or wrap the integration step in a
  trainer-level helper that picks the right primitive. Defer to patch
  1.5 if it slows down landing patch 1.

## Anti-scope

- Do **not** also introduce a different smoothness term (Lipschitz,
  Jacobian, score-curvature) in this ticket. Those are separate
  proposals that compose differently — track separately if anyone
  wants to ablate.
- Do **not** also change `distillation`'s structure. It already does
  the right thing; touching it here muddles the diff.
- Do **not** also rewrite the eval-grid logging. The same eval grid
  measures regularized and unregularized runs on identical step
  counts; that's the point.

## Files to touch

| File | Change |
|---|---|
| `src/generative_flow_adapters/config.py` | Add `heun_smoothness_weight`; remove `"linear"` default for `shortcut_target_method` |
| `src/generative_flow_adapters/training/trainer.py` | Delete `_compute_two_step_target_v`, `_resolve_step_level`, the `two_step` branch in `_maybe_prepare_shortcut`; add `_compute_heun_smoothness` and the loss-application block |
| `src/generative_flow_adapters/training/shortcut_targets.py` | Delete `compute_two_step_target_v`; keep `ddim_micro_step_v` (still used by `distillation` and the new regularizer) |
| `src/generative_flow_adapters/losses/consistency.py` | Add `heun_smoothness_loss`; register in `LossRegistry` |
| `configs/diffusion_avid_shortcut_metaworld.yaml` | Switch to `distillation` |
| `configs/diffusion_hyperalign_shortcut_metaworld.yaml` | Switch to `distillation` |
| `examples/shortcut_training_test.py` | Switch to `distillation` |
| `examples/hyper_shortcut_training_test.py` | Switch to `distillation` |
| `tests/test_consistency_losses.py` (new) | Heun-smoothness loss unit tests |
| `tests/test_trainer_shortcut_paths.py` (new) | Trainer-step gating, frozen-base, smoke |

## Done when

- `pytest` passes with the new tests above.
- Both shortcut configs run for ≥100 steps under `distillation` mode
  without errors; loss curves don't blow up.
- `git grep "two_step"` returns zero hits in `src/` and `configs/`
  (only historical mentions in the vault).
- `heun_smoothness_weight` is documented in `docs/` and surfaces in
  at least one example config.
- The follow-up documentation listed under "Documentation updates"
  above is complete.
