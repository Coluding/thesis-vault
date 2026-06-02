---
type: feat
scope: shortcut
status: done
priority: high
created: 2026-05-21
updated: 2026-06-02
resolution: wont-do-deferred
resolution_note: |
  Not implementing now. Decided 2026-06-02 to rely on the data anchor
  through the standard diffusion/flow loss (`shortcut_anchor_prob: 0.75`)
  as the primary collapse defense — it rules out both named collapse
  modes at convergence, making the architectural g(d→0) gate largely
  redundant. The gate's only residual value (exact d→0 guarantee +
  early-transient smoothing) overlaps with
  [[feat-training-shortcut-anchor-step-warmup]]. Kept as a deferred,
  not-rejected option — see decision
  [[../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]
  for the revisit triggers. NOTE: the touch point named below
  (ShortcutDirectionOutputAdapter) is now DEPRECATED; if revived, the
  gate must target the live dynamicrafter/unicon step_level branch.
closed_at: 2026-06-02
related:
  - "[[done/risk-shortcut-self-consistency-collapse]]"
  - "[[feat-training-shortcut-anchor-step-warmup]]"
  - "[[../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]"
---

# Add architectural `g(d)` zero-asymptote gate to the shortcut adapter

## Motivation

Mitigation **Option A** from
[[risk-shortcut-self-consistency-collapse.md]] — structurally force the
adapter's contribution to vanish as `d → 0`, so the composed predictor
reduces to the frozen base at small step sizes by construction. This
removes one trivial fixed point of the self-consistency loss (cancellation
collapse at `d ≈ 0`) without touching the loss formulation.

Top of the risk-mitigation queue: cheap, local change, blocks one named
failure mode of the shortcut training before we start running it.

## Current state (verified 2026-05-21)

In `src/generative_flow_adapters/adapters/output/shortcut_direction.py`,
`ShortcutDirectionOutputAdapter` consumes `step_size` as a concatenated
input feature to a 3-layer MLP (lines 38-46, 82-99):

```python
if include_step_size:
    input_dim += 1
...
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.SiLU(),
    nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
    nn.Linear(hidden_dim, feature_dim),
)
...
chunks.append(step_size.to(device=x_t.device, dtype=x_t.dtype))
features = torch.cat(chunks, dim=-1)
shortcut_direction = self.net(features)
```

So `d`-dependence is learned through the MLP. The final
`nn.Linear(hidden_dim, feature_dim)` uses default init, **not** zero-init,
and there is no multiplicative scaling on the output. Nothing in the
architecture forces `Δ_φ(·, d=0) ≡ 0`.

The sigmoid in `models/adapted_model.py:93` is the AVID mask-mix gate
(`sigmoid(adapter.gate + gate_bias)`), gated on the adapter's learned
per-feature mask — not on `d`. The HyperAlign `_cached_gate` is the same
mask-mix path. The `* step_size` multiplications in
`training/shortcut_targets.py:91` and `testing/fake_shortcut_data.py:81`
are on the *target* side, not on the adapter output.

⇒ Option A is not implemented today.

## Proposed change

Wrap the adapter's output in a scalar gate `g(d)` with `g(0) = 0`:

```
s(x_t, t, d) = base(x_t, t) + g(d) · Δ_φ(x_t, t, a, d)
```

Default gate form (per the risk ticket): `g(d) = sigmoid(α·d + β)` with
`β ≪ 0` so the adapter is near-off at `d = 0` and saturates to ~1 for
larger `d`. `α` and `β` start fixed, optionally promotable to learnable
scalar `nn.Parameter`s if needed.

Concrete touch points:

- `adapters/output/shortcut_direction.py`
  - Add `d_gate: Literal["none", "sigmoid", "linear", "tanh"] = "none"`
    config flag (default `"none"` preserves current behaviour).
  - Add `d_gate_alpha: float` and `d_gate_beta: float` fields (only used
    when `d_gate != "none"`); for `"sigmoid"` start with values that give
    `g(1) ≈ 0.1`, `g(K) ≈ 0.9` for whatever `K` we end up running.
  - At the end of `forward`, multiply `shortcut_direction` by `g(d)`
    *before* returning the `OutputAdapterResult`.
- `adapters/factory.py:47-55` — thread the new `extra` keys through to
  the constructor.
- `config.py` — extend the shortcut-relevant config block with the gate
  options.
- `tests/` — add a unit test asserting that the adapter output is
  numerically zero at `d = 0` when `d_gate ≠ "none"`, regardless of
  random initialization of the MLP.

Roughly ~30 lines of code + one test. Default-off keeps existing configs
working unchanged.

## Acceptance criteria

1. `d_gate = "sigmoid"` produces an adapter output that is `≤ 1e-6` in
   absolute value at `d = 0` for any random init of the MLP weights.
2. With `d_gate = "none"`, behaviour and numerics are bit-identical to
   today (regression test on a small fixture).
3. The diagnostic `‖s_composed − base‖` from the risk ticket can be
   computed and stays small for small `d` when the gate is on.
4. One config (to be named) opts the new shortcut runs into
   `d_gate = "sigmoid"` so we can A/B against `d_gate = "none"` on the
   same backbone + adapter + dataset.

## Out of scope (separate tickets if needed)

- Option A2 (zero-init the `step_level_embed` final projection +
  bias-free skip). Different mechanism; defer until we have data on
  whether A1 alone moves the diagnostics.
- Option B (`d = 1` data anchor in the loss). That is the default path
  in the risk ticket and gets its own implementation ticket.
- Option C (A + B together). Compose once both are landed.

## Diagnostics to track once shipped

From [[risk-shortcut-self-consistency-collapse.md]]'s diagnostic list:

- `‖Δ_φ(x_t, t, a, d=1)‖` averaged over held-out batch — gate-on should
  not crush this to zero away from `d = 0`.
- `‖s_composed − base‖` at increasing `d` — should now grow monotonically
  from ~0 at `d = 0` to non-trivial at large `d`.
- Self-consistency loss vs. (eventual) `d = 1` anchor loss — gate-on
  alone does not address the "satisfied off-data" failure mode; expect
  it to help only with the small-`d` corner.

## Related

- [[risk-shortcut-self-consistency-collapse.md]] — Option A motivation
  and the full collapse-mode taxonomy.
- [[bug-training-shortcut-target-timestep.md]] — target-side bug;
  orthogonal to this loss/architecture change but lives in the same
  shortcut training path.
- Code anchor:
  `src/generative_flow_adapters/adapters/output/shortcut_direction.py`
  lines 38-46, 82-99 (the place the gate slots in).
