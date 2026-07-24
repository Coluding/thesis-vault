---
type: bug
scope: losses
status: open
priority: high
created: 2026-06-22
updated: 2026-06-22
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[feat-shortcut-displacement-target-parametrisation]]"
  - "[[chore-writing-fix-paper-faithful-wording]]"
---

# bug: shortcut self-consistency target is biased for v-prediction — ship endpoint inversion (Option A)

## Progress — 2026-06-24 (code landed; awaiting a real run for closure)

Endpoint inversion (Option A) is **implemented and tested**. Displacement
(Option B) was explicitly deferred to
[[feat-shortcut-displacement-target-parametrisation]] (user decision this
session — "endpoint inversion only" for now).

What landed in `generative-flow-adapters`:
- **`invert_ddim_v`** (`training/shortcut_targets.py`) — exact algebraic inverse
  of `ddim_micro_step_v`, **including the `scale_arr` dynamic-rescale branch**.
  The DDIM v-step is affine in v (`x_end = A·x + B·v`); the inverse is
  `v = (x_end − A·x)/B` with `A,B` built from `√ᾱ`, `√(1−ᾱ)` and the rescale
  ratio — i.e. the *reproducing* velocity, **not** the chord (chord is Option B).
- **`compute_self_consistency_target_v`** gained a `target_kind` arg:
  `v_average` (default, biased baseline — unchanged) or `endpoint_inversion`
  (takes the second sub-step to `x_end`, inverts the `2d` recompose).
- **Config selector** = new sibling field `TrainingConfig.shortcut_consistency_target`
  (`v_average` | `endpoint_inversion` | `displacement`→raises NotImplementedError).
  Kept separate from `shortcut_target_method` (which selects the *teacher path*,
  still `"distillation"`) — confirmed by reading the code, per the ticket's note.
  Diffusion-only; flow paths untouched (κ=0 ⇒ unbiased by construction).
- **Docstring wording** corrected (no longer calls v-averaging "paper-faithful")
  — partially closes [[chore-writing-fix-paper-faithful-wording]].

Regression test `tests/test_shortcut_endpoint_inversion.py` (4 tests, green):
- `invert_ddim_v` round-trips to **<1e-9** (float64) with **and without**
  `scale_arr`.
- On a synthetic consistent ray, the `endpoint_inversion` target reproduces the
  two-step landing to **~1e-16**; `v_average` lands measurably off and the
  ticket's "assert old average fails at the 2-step rung" holds.
- Measured `v_average` relative landing error grows with step size: **0.1 % (d=50)
  → 4.5 % (d=200) → 16.2 % (d=300)** over `t=800` — matches the theory note's
  5–24 % prediction.
- Full suite: my 4 + the 24 existing consistency/wan tests pass.

**Remaining for closure (per Done when):** a *real* shortcut run with
`shortcut_consistency_target: endpoint_inversion` showing the **coarse per-rung
losses converging** where they plateau today (per-rung logging from 2026-06-17,
Case A confirmed in
[[../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]). No
numbers before that run logs them (hard rule 8). Keep open until then.

## Context

`compute_self_consistency_target_v` returns `((v1 + v2) / 2).detach()`. That
average is exact for flow matching (Frans et al. 2024, eq. 4) but **biased for
v-prediction + DDIM**: on a perfectly consistent ray a single `2d` step lands
5–24 % off the true endpoint (worst in the few-step regime D3 targets), and the
true field is **not a fixed point** of the rule, so more training cannot remove
it. Full derivation + numbers:
[[../30_Knowledge/theory/shortcut-v-averaging-bias]]; the DDIM primitive being
inverted: [[../30_Knowledge/theory/ddim-step-v-parameterisation]] §8. Decision:
[[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]] (Option A).

## What to build

1. **`invert_ddim_v`** mirroring `ddim_micro_step_v` exactly, **including the
   `scale_arr` / dynamic-rescale branch**. The DDIM step is a rotation
   `x_{t'} = cos(Δ)·x_t + sin(Δ)·v_t`; the inverse for a target endpoint is
   `v = (x_end − cos(Δ)·x_t) / sin(Δ)` — **the reproducing velocity, not the
   chord `x_end − x_t`** (the chord is Option B's displacement target).
2. **Endpoint target** in `compute_self_consistency_target_v`: take the second
   sub-step too (`x_end = DDIM(x_mid, v2, t−d → t−2d)`), then invert the `2d`
   DDIM recompose for the single velocity that lands on `x_end`.
3. **Config selector** for the shortcut target method
   (`endpoint_inversion` | `v_average` baseline; `displacement` added by the
   sibling ticket). Confirm how this relates to the existing
   `shortcut_target_method` field (which currently selects teacher paths)
   before extending it vs. adding a sibling selector — **do not assume the
   current values; read the code first.**
4. **Keep `v_average`** reachable as the biased baseline arm (for the ablation).

## Before implementing

- Confirm whether the **DynamiCrafter** configs pass a non-`None` `scale_arr`
  (dynamic rescale). If so, `invert_ddim_v` must mirror the affine `x̂₀` rescale
  between decompose/recompose, not just the rotation.

## Regression test

- On a synthetic consistent ray, one `2d` step with the inverted target
  reproduces the two-step landing to **<1e-5**.
- **Assert the old `v_average` fails the same check at the 2-step rung**
  (guards against silently re-biasing).

## Done when

The inverted target + `invert_ddim_v` land with the test green, the config
selector switches between `endpoint_inversion` and `v_average`, and a real run
shows the **per-rung coarse shortcut rungs converging** where they plateau
today (per-rung logging from 2026-06-17). No numbers recorded before the run
logs them (hard rule 8).
