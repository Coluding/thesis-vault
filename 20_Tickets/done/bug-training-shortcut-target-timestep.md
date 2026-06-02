---
type: bug
scope: training
status: superseded
priority: medium
created: 2026-05-19
updated: 2026-05-28
resolution: superseded
resolution_note: |
  Structural fix landed via the switch to `ddim_micro_step_v` in the
  `two_step` teacher path (see commits prior to 5796149). The current
  code passes `prev_t` (not `t`) to the second `base_model(...)` call,
  so the time argument now advances with state as Heun requires. A
  *different* limitation has taken its place — the DDIM jump is
  hardcoded to one timestep, which means there is still no
  step-size-conditional supervision. Tracked at
  [[bug-training-shortcut-twostep-no-stepsize-variation]].
closed_at: 2026-05-28
related:
  - "[[bug-training-shortcut-twostep-no-stepsize-variation]]"
---

# `two_step` shortcut target uses `t` instead of `t + d` for the second base call

## Symptom

`_compute_two_step_shortcut_target` in
`src/generative_flow_adapters/training/shortcut_targets.py:122`
evaluates the second base prediction at the **same** timestep as the
first:

```python
# Second prediction (at stepped-forward state)
v_b2 = base_model(x_t2, t, cond=cond)  # Use same t (simplified)
```

The paper's shortcut formulation (see
[[../30_Knowledge/related-work/shortcut-models]]) advances time along
with state:

```
s(x_t, t, 2d, a_t) ≈ ½ · s(x_t, t, d, a_t) + ½ · s(x_{t+d}, t+d, d, a_t)
                                                          ^^^^
```

So the second base call should be at `t + d`, not `t`.

The inline comment (`# Use same t (simplified)`) confirms this is a
known shortcut taken at implementation time.

## Why it matters

- This is the **single divergence from the paper-faithful target
  construction** in the shortcut training path. The headline D3
  experiments will be argued against the paper's formulation, so an
  unintentional simplification here weakens the argument.
- The simplification quietly biases the target: at high-noise timesteps
  the base's velocity prediction is approximately consistent across a
  small window in `t`, so the effect may be small in practice — but
  "may be small" isn't an argument we want to make in a thesis chapter
  unless we've measured it.
- The cost of the fix is one tensor add (`t + d`) at the second base
  call; no architectural change.

## Plan

1. Patch `_compute_two_step_shortcut_target` to compute
   `t2 = t + d` (with appropriate dtype/device handling — `d` may be
   per-sample) and pass `t2` to the second `base_model(...)` call.
2. Make sure the shape broadcasting works for the per-sample step-size
   case (the rest of the function already handles `d` as `[B]` or
   `[B, 1]`).
3. Add a regression test: target with `d = 0` must equal the single
   base prediction at `t` (sanity check that the `t + d` change does
   not affect the trivial case).
4. Re-run the relevant shortcut smoke tests / training tests; check
   that the `shortcut_direction_loss` curve is qualitatively similar
   on a small run before launching anything bigger.

## Open questions

- Should `t2` be clipped to the schedule's valid range
  (`0 ≤ t2 ≤ T_max`)? In the paper, `t + d` is bounded by the
  diffusion schedule. Worth checking what happens at the boundary
  (`t` near max, `d` non-trivial) under the current sampler.

## Related

- [[../30_Knowledge/writing/figure-shortcut-training]] — the figure now
  shows the paper-faithful `t + d` (decision 2026-05-19); when this
  bug is fixed the figure and the code will agree.
- [[../30_Knowledge/related-work/shortcut-models]] — paper formulation
- [[../30_Knowledge/theory/heun-shortcut-target]] — numerical-methods
  framing; the simplification breaks textbook Heun by freezing the
  time argument of the second slope sample.
- Code: `src/generative_flow_adapters/training/shortcut_targets.py:122`
- Code: `src/generative_flow_adapters/training/trainer.py:327`
  (parallel inline implementation with the same divergence)
