---
type: bug
scope: training
status: superseded
priority: high
created: 2026-05-28
updated: 2026-05-28
resolution: wont-fix-superseded
resolution_note: |
  Superseded by the decision to deprecate `two_step` as a shortcut
  training mode entirely. See
  [[../50_Decisions/decided/deprecate-twostep-shortcut-mode]]. The
  reasoning: even with the step-size-conditional fix proposed in this
  ticket, the frozen base cannot produce real chord-velocity
  supervision at the scales few-step inference asks about; Heun's
  averaging is only accurate near the current point. `distillation`
  does not have this ceiling. The Heun construction is preserved as a
  separate, generally-applicable smoothness regularizer — see
  [[refactor-shortcut-deprecate-twostep-add-heun-smoothness]] for the
  replacement.
closed_at: 2026-05-28
related:
  - "[[bug-training-shortcut-target-timestep]]"  # superseded — see §1
  - "[[risk-shortcut-eval-steplevel-out-of-distribution]]"
  - "[[../50_Decisions/decided/deprecate-twostep-shortcut-mode]]"
  - "[[refactor-shortcut-deprecate-twostep-add-heun-smoothness]]"
---

# `two_step` shortcut target supervises a single hardcoded step size

## Symptom

In `_compute_two_step_target_v`
(`src/generative_flow_adapters/training/shortcut_targets.py:33-51` and the
inline copy at `src/generative_flow_adapters/training/trainer.py:510-534`),
the DDIM micro-step is hardcoded to a jump of one timestep:

```python
v0    = base_model(x_t, t, cond=cond)
prev_t = (t - 1).clamp_min(0)                 # <-- jump = 1 timestep, always
x_mid = ddim_micro_step_v(x=x_t, v=v0, t=t, prev_t=prev_t, ...)
v1    = base_model(x_mid, prev_t, cond=cond)
return ((v0 + v1) / 2.0).detach()
```

Consequences:

- The Heun-averaged target is the chord velocity over the **finest
  possible interval** (one timestep, e.g. `1/1000` of the trajectory on
  a `T = 1000` schedule).
- The `step_level` value injected into `cond` is sampled from
  `[shortcut_step_level_min, shortcut_step_level_max]`
  (`_resolve_step_level`, `trainer.py:443-473`) but the supervisory
  target is **independent of it**. The adapter is regressed against
  the same target regardless of what `step_level` it sees.
- At inference, taking jumps of size `s > 1 timestep` (which is what
  any few-step rollout asks for — `N = 50` on `T = 1000` is a
  20-timestep jump) is **out-of-distribution**. The adapter has never
  been supervised for such jumps.

`two_step` as currently implemented is a base-distilled second-order
velocity prior at the finest scale, not a shortcut training mode in
the sense of Frans et al. 2024.

## Relationship to the earlier "t vs t+d" bug

The earlier ticket [[bug-training-shortcut-target-timestep]] flagged
that the second base call was evaluated at the wrong time argument
(`(x_mid, t)` instead of `(x_mid, t-1)`). **That issue is structurally
fixed** — the current code uses `ddim_micro_step_v` and passes
`prev_t` to the second `base_model(...)` call. The time argument
advances with the state, as Heun requires.

The present ticket is the *different* limitation that has taken its
place: the time argument now advances correctly, but only by a fixed
single timestep. So `two_step` is now textbook-Heun *at the finest
scale*, but it still does not produce a step-size-conditional target.

→ Once this ticket lands, mark
[[bug-training-shortcut-target-timestep]] as resolved with
`resolution: superseded` referencing this ticket.

## Why it matters now

D3 is the deliverable that depends on few-step generation. The headline
D3 story for `two_step` was "base-as-teacher avoids self-consistency
collapse while still giving step-size-aware behaviour." The
implementation gives the first half but not the second half.

Both live shortcut configs currently use `two_step`:

- `configs/diffusion_avid_shortcut_metaworld.yaml:81`
- `configs/diffusion_hyperalign_shortcut_metaworld.yaml:97`

So **every shortcut run logged so far has been trained on this
no-step-size-variation regime**. The few-step rollouts that the
`log_step_size_grid` visualisation produces are the model
extrapolating to step counts it was never supervised for.

## Plan

The fix has three pieces. The minimum viable change is small (one
function); the conditioning wiring is what takes the actual work.

### 1. Sample `s` per training step

In `_maybe_prepare_shortcut` (the `two_step` branch,
`trainer.py:372-383`), replace the per-sample uniform-int draw with a
per-batch draw from the existing `ShortcutStepSchedule`:

```python
if self.step_schedule is not None:
    s = self.step_schedule.sample()
    jump = self.step_schedule.to_timestep_jump(s)
else:
    # Legacy fallback: single fixed step size.
    s, jump = 1.0 / self.config.extra.get("timesteps", 1000), 1
```

The `ShortcutStepSchedule` is already wired into the trainer
(`trainer.py:71-72`) and consumed by `distillation`, so this is
"reuse what's there".

### 2. Use `jump` in the target computation

Change the signature of `compute_two_step_target_v` to accept the jump
size, and pass `prev_t = (t - jump).clamp_min(0)`. The DDIM
micro-step and second base call use the same `prev_t`. Two-line
change in `shortcut_targets.py`.

### 3. Inject the matching `step_level = s` into cond

Replace the `_resolve_step_level` call (which currently samples a
decorative integer) with direct injection of the sampled normalised
`s`:

```python
step_level = torch.full(
    (batch_size,), float(s), device=device, dtype=dtype
)
new_cond = self._inject_step_level(cond, step_level_key, step_level)
```

This is the same pattern `distillation` uses for `step_level_full`
(`trainer.py:407-409`), so the cond format is consistent across
modes — adapters can be agnostic to which target produced the
supervision.

### 4. Regression tests

- Backward-compat: with no `shortcut_step_schedule` configured and
  `shortcut_step_level_max=1`, the new code path should produce a
  target byte-identical to the current implementation. Add a test that
  pins this.
- Step-size variation: with a `log2` schedule over `[1/64, 1/2]`, the
  sampled `jump` should be a power of two between 1 and 16 for
  `T=1000` (`to_timestep_jump` rounds to the nearest int ≥ 1). Assert
  the empirical distribution of jumps over 1000 sampled batches.
- Target finite-ness: for every `s` in the schedule, the target
  should be finite and have the same shape as `x_t`.

## Open questions

- **Per-batch vs per-sample `s`.** The paper uses one `s` per batch,
  matching the `distillation` convention. Keeping per-batch for
  `two_step` parity is the safe default; per-sample would let one
  training step cover multiple step sizes but is a deviation from the
  paper. Decide before landing.
- **Heun fidelity at large `s`.** Heun's local error is `O(s^3)` with
  a constant scaling with the curvature of `v` along the chord. At
  `s = 1` (one-step generation) the Euler predictor lands far from
  the true endpoint and the target degrades to a noisy chord
  estimate. Mitigations: cap the schedule (e.g. `max = 1/8`), or use
  a higher-order quadrature (RK4) for the large-`s` rungs. Worth a
  small ablation after the fix.
- **Backward compatibility of the configs.** Both live shortcut
  configs would silently switch to step-size-variation supervision
  once this lands. Decide whether to (a) add an explicit
  `shortcut_step_schedule` block to the live configs first and ship
  the trainer change second, or (b) ship them together with a clear
  changelog note. The current state (no schedule configured) should
  preserve old behaviour via the fallback branch in step 1 above.

## Anti-scope (do *not* do here)

- Do **not** also switch the schedule sampler to be per-sample within
  the same patch. That's an orthogonal axis; separate ticket if
  warranted.
- Do **not** also rewrite `_resolve_step_level` to take the schedule.
  The current function still has a use under
  `shortcut_target_method=two_step` *without* a configured schedule
  (i.e. the legacy fallback). Touch it minimally.
- Do **not** also touch `distillation`. It already does this.

## Related

- [[../30_Knowledge/theory/heun-shortcut-target]] §5–6 — derivation of
  the limitation and the proposed fix in math notation
- [[../30_Knowledge/theory/ddim-step-v-parameterisation]] — the
  primitive `ddim_micro_step_v` operates on; gives the "why DDIM" for
  the predictor step
- [[../30_Knowledge/tech/shortcut-training-modes]] — code-side
  catalogue; the gotcha list now flags this limitation
- [[bug-training-shortcut-target-timestep]] — earlier ticket on the
  same code path; structurally superseded
- [[risk-shortcut-eval-steplevel-out-of-distribution]] — the
  eval-side counterpart of this issue (the eval grid asks the model
  about step sizes it was never trained on); both should close
  together
- Code: `src/generative_flow_adapters/training/shortcut_targets.py:33-51`
- Code: `src/generative_flow_adapters/training/trainer.py:372-383`
  (mode dispatch), `:443-473` (`_resolve_step_level`), `:510-534`
  (inline `two_step` target), `:71-72` (`step_schedule` init)
- Code: `src/generative_flow_adapters/training/step_schedule.py`
  (`ShortcutStepSchedule.sample`, `to_timestep_jump`)
