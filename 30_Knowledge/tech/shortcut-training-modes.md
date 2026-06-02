---
type: tech-note
status: living
last_updated: 2026-05-28
sources:
  - "code: src/generative_flow_adapters/training/trainer.py"
  - "code: src/generative_flow_adapters/training/shortcut_targets.py"
  - "code: src/generative_flow_adapters/training/step_schedule.py"
  - "code: src/generative_flow_adapters/losses/consistency.py"
  - "code: src/generative_flow_adapters/adapters/output/shortcut_direction.py"
  - "code: src/generative_flow_adapters/config.py"
  - "config: configs/diffusion_avid_shortcut_metaworld.yaml"
  - "config: configs/diffusion_hyperalign_shortcut_metaworld.yaml"
commit: 5796149  # WIP — refactor pending per the deprecation decision
relevance: D3  # shortcut adapters
---

# Shortcut training mode + Heun-smoothness regularizer — catalogue

> What the trainer can actually do, post-decision (2026-05-28). The
> conceptual framing of "what a shortcut adapter is" lives in
> [[../theory/shortcut-training]]; the formal derivation of the
> smoothness regularizer in [[../theory/heun-smoothness-regularizer]];
> the decision to deprecate `two_step` in
> [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]]. This
> note is the code-side catalogue: knobs, costs, collapse risks, line
> numbers.

## TL;DR

| Component | Role | Target source | Adapter passes per step | Risks | Anchor mechanism |
|---|---|---|---|---|---|
| **`distillation`** | Shortcut training (only mode after deprecation) | Adapted model — two no-grad calls at `step_level = s/2` averaged, supervising at `step_level = s` | 1 (forward), 2 (target, no-grad) | Self-consistency can collapse to zero | Anchor branch fires with prob `shortcut_anchor_prob`; runs the standard diffusion/flow loss at the schedule's smallest step |
| **`heun_smoothness`** *(planned)* | Velocity-field smoothness regularizer (opt-in for any run, not specific to shortcut) | Composed model `f_θ`'s own velocity at the predicted endpoint, stop-grad | 1 (forward), 1 (target, no-grad) | None — anchored by `L_base` (or `distillation`'s anchor when composed) | N/A |
| ~~`two_step`~~ | **Deprecated 2026-05-28** | n/a | n/a | n/a | n/a |

Both active components share the same DDIM micro-step primitive
(`shortcut_targets.py:81-126`, `ddim_micro_step_v`) and the same
`ShortcutStepSchedule` (`step_schedule.py`). The supervisory target is
detached in both cases so no gradient flows through the teacher path.

## Where it lives

| Piece | File | Lines (commit 5796149) |
|---|---|---|
| `shortcut_target_method` config field | `src/generative_flow_adapters/config.py` | 68 — needs default `"distillation"` post-refactor |
| `heun_smoothness_weight` config field *(planned)* | `src/generative_flow_adapters/config.py` | new |
| Loss weights `shortcut_direction_weight`, `local_consistency_weight` | `src/generative_flow_adapters/config.py` | 65–66 |
| `_needs_shortcut_target` gate | `src/generative_flow_adapters/training/trainer.py` | 319–324 |
| Mode dispatch (`_maybe_prepare_shortcut`) | `src/generative_flow_adapters/training/trainer.py` | 326–441 — the `two_step` branch (372–383) is deleted in the refactor |
| `distillation` target — paper-faithful self-consistency | `src/generative_flow_adapters/training/trainer.py` | 385–436 (with-schedule path: 392–414; legacy fallback: 416–436) |
| Self-consistency target helper | `src/generative_flow_adapters/training/trainer.py` | 536–565 (`_compute_self_consistency_target_v`) |
| Heun smoothness helper *(planned)* | `src/generative_flow_adapters/training/trainer.py` | new |
| Anchor branch (standard loss at the schedule's finest step) | `src/generative_flow_adapters/training/trainer.py` | 399–403 |
| Step-size sampler | `src/generative_flow_adapters/training/step_schedule.py` | 142 (`sample`), 134 (`to_timestep_jump`) |
| Loss application (direction + consistency) | `src/generative_flow_adapters/training/trainer.py` | 200–214 |
| Standalone target utilities | `src/generative_flow_adapters/training/shortcut_targets.py` | 33–78 (`compute_two_step_target_v` 33–51 to be deleted) |
| Loss kernels | `src/generative_flow_adapters/losses/consistency.py` | 7–17 (+ new `heun_smoothness_loss`) |
| Adapter consuming the targets | `src/generative_flow_adapters/adapters/output/shortcut_direction.py` | full file |

Line numbers refer to the WIP commit 5796149. The refactor pending in
[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]
deletes the `two_step` lines and adds the new helper / loss / config
field.

## Active mode — `distillation` (paper-faithful self-consistency)

Implements Frans et al. 2024, eq. 4. Each training step picks one of
two branches by Bernoulli on `shortcut_anchor_prob` (default `0.75`).

### Anchor branch (prob = `shortcut_anchor_prob`)

```
step_level = schedule.smallest()           # the finest rung
target     = (none — fall through to standard diffusion/flow loss)
```

The adapter is supervised by the regular `LossRegistry.get_loss(...)`
head at the smallest scale in the schedule. The shortcut direction
loss does not fire on these steps. This anchor is what prevents the
bootstrap target in the other branch from collapsing to zero — without
it, `target = 0` and `adapter = 0` is a trivial fixed point of the
self-consistency objective. Tracked separately in
[[../../20_Tickets/risk-shortcut-self-consistency-collapse]].

### Self-consistency branch (prob = `1 - shortcut_anchor_prob`)

With a configured `shortcut_step_schedule`:

```
s_full     ∈ schedule.sample(exclude_smallest=True)              # normalised (0,1]
s_half     = s_full / 2
jump       = schedule.to_timestep_jump(s_half)                   # round(s_half · T)
step_level = s_full                                              # for the adapter
cond_half  = cond with step_level = s_half                       # for the teacher

# no-grad, model in eval() mode for these calls:
v1    = adapted_model(x_t,   t,        cond_half)                # step_level=s_half
x_mid = ddim_micro_step_v(x_t, v1, t → t - jump)                 # s_half-sized DDIM step
v2    = adapted_model(x_mid, t - jump, cond_half)                # step_level=s_half
target = (v1 + v2) / 2                                           # detached
```

The adapter is regressed against this target *at* `step_level = s_full`.
The adapter is its own teacher; the frozen base contributes only
implicitly through the additive composition inside `self.model`. The
`eval()` toggle in `_compute_self_consistency_target_v`
(`trainer.py:552-564`) makes sure dropout / norm stats don't differ
between teacher and student passes.

The legacy fallback (no schedule configured) uses the dyadic-`d`
sampler at `trainer.py:494-508`: `d ∈ {1, 2, 4, …, max/2}` per batch,
supervised at `step_level = 2d` against two calls at `step_level = d`.
Equivalent in structure; raw-timestep units instead of normalised.

- **`s` is per-batch, not per-sample.** Matches the paper's per-step
  batch-split.
- **Cost.** Two no-grad adapted-model forwards for the target (each is
  a full base + adapter pass), plus the one forward for the adapter
  prediction. So ≈3× the per-step cost of standard training when the
  self-consistency branch fires.

## Regularizer — `heun_smoothness` *(planned)*

Opt-in `λ_hs L_heun-smooth` term added to the trainer loss, available
for any run (vanilla diffusion, vanilla flow, or shortcut). At a
sampled `(x_t, t, c)`:

```
s     ∈ schedule.sample()                                 # falls back to s = 1 if no schedule
jump  = schedule.to_timestep_jump(s)
v0    = f_θ(x_t, t, c)                                    # WITH grad
x_prev = ddim_micro_step_v(x_t, v0, t → t - jump)
# no-grad, model in eval() mode:
v1    = f_θ(x_prev, t - jump, c)
L_heun_smooth = ‖v0 - sg(v1)‖²
```

The regularizer is a discrete material-derivative penalty on the
composed model's velocity field along its own predicted trajectory.
See [[../theory/heun-smoothness-regularizer]] for the derivation, the
stop-grad justification, and the three sampling-regime trade-offs.

- **`step_level` is NOT injected.** The smoothness scale `s` is a
  property of the regularizer, not a task knob. The model sees the
  same `cond` it sees during standard training.
- **Cost.** One extra no-grad forward per step. Cheaper than
  `distillation`'s teacher branch.
- **Composes orthogonally with `distillation`.** Both terms can be
  on; they supervise different things. Likely redundant at the
  schedule's finest step (anchor branch + smoothness both reduce to
  consistency at `s_min`), independent elsewhere.

## Deprecated — `two_step` (base-anchored Heun)

**Removed from the shortcut mode dispatch on 2026-05-28.** Decision:
[[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]];
implementation: [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].

Historical summary (for cross-reference when reading old configs / commits):

```
v0    = base(x_t,   t,   cond)               # no-grad, frozen
x_mid = ddim_micro_step_v(x_t, v0, t → t-1)  # jump hardcoded to 1
v1    = base(x_mid, t-1, cond)               # no-grad, frozen
target = (v0 + v1) / 2                       # detached
```

This is Heun's method (2nd-order Runge-Kutta) applied to the frozen
base over one DDIM micro-step. The two structural problems that drove
deprecation:

1. **Jump hardcoded.** `prev_t = (t - 1).clamp_min(0)`. Heun's
   averaging window was always `[t-1, t]` regardless of any sampled
   step size. `step_level` was decorative.
2. **Base is the wrong teacher for chord velocities at large `s`.**
   The frozen base is a pointwise velocity estimator; Heun's averaging
   of two base samples only approximates the true chord velocity in a
   2nd-order Taylor sense around the current point. At large `s` the
   Euler predictor lands far from the true endpoint and the target
   degrades to noise. `distillation`'s adapted-model teacher does not
   have this ceiling.

Both live shortcut configs ran under `two_step` until the
2026-05-28 decision; runs from that period are valid as a **baseline
control** for "what does the adapter learn when the only signal is
single-scale smoothness?" — not as evidence of few-step generation
capability.

The Heun construction itself is preserved as the
**`heun_smoothness` regularizer** above — same primitive, different
teacher (composed model instead of base), different role (regularizer
instead of primary objective), different step-size handling
(sampled instead of hardcoded).

## Loss application

After the refactor, the trainer adds up to three terms on top of the
standard model loss:

```python
# distillation target lands in batch["shortcut_target"]
if config.local_consistency_weight > 0.0 and "shortcut_target" in batch:
    loss += config.local_consistency_weight * local_consistency_loss(prediction, target)

if config.shortcut_direction_weight > 0.0 and "shortcut_target" in batch:
    loss += config.shortcut_direction_weight * shortcut_direction_loss(prediction, target)

# heun smoothness is independent — no shared batch key
if config.heun_smoothness_weight > 0.0:
    loss += config.heun_smoothness_weight * heun_smoothness_loss(current_v=v0, future_v_detached=v1)
```

The first two MSE heads share the `distillation` target tensor and
differ only in name (historical: `local_consistency_loss` originally
targeted `target = base_output * step_size`, `shortcut_direction_loss`
the chord-velocity target). Live configs currently set
`shortcut_direction_weight = 1.0` and leave `local_consistency_weight = 0.0`,
so only one of them actually contributes; reconsider whether the
duplicate exposed knob is worth keeping when the configs are revisited
in patch 2 of the refactor.

## Config plumbing — knobs the components read

| Knob | Path | Read by | Notes |
|---|---|---|---|
| `shortcut_target_method` | `training.shortcut_target_method` | mode dispatch | Post-refactor: only `"distillation"` accepted. Default `"linear"` (`config.py:68`) is removed alongside `"two_step"`. |
| `shortcut_direction_weight` | `training.shortcut_direction_weight` | loss application | MSE coefficient on the direction head. |
| `local_consistency_weight` | `training.local_consistency_weight` | loss application | MSE coefficient on the consistency head (same target). |
| `heun_smoothness_weight` | `training.heun_smoothness_weight` | regularizer | **New post-refactor.** Default `0.0`. Recommended ≤ 0.1 when turned on. |
| `shortcut_step_schedule` | `training.extra.shortcut_step_schedule` | `distillation`, `heun_smoothness` | Defines the normalised step-size support; see `step_schedule.py` docstring. Falls back to the legacy dyadic sampler for `distillation`, to `s = 1` for the regularizer. |
| `shortcut_step_level_key` | `training.extra.shortcut_step_level_key` | `distillation` | Conditioning-dict key under which `step_level` is injected (default `"step_level"`). Adapters consuming step size must use the same key. |
| `shortcut_step_level_max` | `training.extra.shortcut_step_level_max` | `distillation` legacy fallback only | Dyadic cap when no `shortcut_step_schedule` is set. Inert otherwise. |
| `shortcut_anchor_prob` | `training.extra.shortcut_anchor_prob` | `distillation` only | Probability of taking the anchor branch on a given step (default `0.75`). |

## Adapter consumption

The dedicated `ShortcutDirectionOutputAdapter`
(`adapters/output/shortcut_direction.py:11-99`) is the adapter most
naturally paired with the `distillation` target:

- Consumes `cond["embedding"]` (the action / context vector), optional
  `x_t`, optional normalised `cond["base_direction"]` (the frozen
  base's local prediction — see [[../theory/shortcut-training]] §4.1),
  and optional `cond["step_size"]` (or `cond["horizon"]` as fallback).
- Outputs an `OutputAdapterResult` with `output_kind="prediction"` —
  a *direct* chord prediction in output space, **not** an additive
  residual. The one place in the framework where additive composition
  is bypassed; the composition rule is replaced by
  `prediction = adapter(...)`.

The general `DynamiCrafterOutputAdapter` is also usable with
`distillation` (it can ingest the base's local prediction via channel
concatenation; see [[../theory/shortcut-training]] §4.1 for the
framing). Both live configs use that one rather than
`ShortcutDirectionOutputAdapter` — re-confirm from the YAML if this
matters for a downstream claim.

The `heun_smoothness` regularizer does not require any specific
adapter — it regularizes the composed model's velocity field at
whatever conditioning state `cond` is currently being trained on.

## Gotchas worth flagging

- **Stale dataclass default until refactor patch 3 lands.**
  `TrainingConfig.shortcut_target_method` defaults to `"linear"`
  (`config.py:68`) which the trainer already rejects. Until patch 3
  removes this, any code path that constructs a `TrainingConfig`
  without overriding this and turns on a shortcut weight will fail at
  the first training step with a `ValueError`.
- **No `step_level = smallest` in supervised distillation steps.**
  The schedule's smallest level is reserved for the anchor branch.
  If `shortcut_anchor_prob = 0` you never see that step in training at
  all — and the model also never sees standard diffusion/flow
  supervision, which is the collapse path. The default `0.75` is
  load-bearing.
- **The adapted model is put into `eval()` for both
  `_compute_self_consistency_target_v` (distillation) and
  `_compute_heun_smoothness` (regularizer) teacher passes.**
  Re-entering `train()` afterwards is wrapped in `try/finally`. If a
  future change replaces `self.model` with something that doesn't
  tolerate mode toggling, both branches need review.
- **DDIM micro-step is the v-prediction code path.**
  `ddim_micro_step_v` decomposes via the v-parameterisation algebra
  (see [[../theory/ddim-step-v-parameterisation]]). Flow-matching
  backbones still work (the `α` table degenerates), but `*_v` in the
  function names is not just stylistic. Noise- or `x_0`-parameterised
  variants are a new function.
- **Heun smoothness at `s = 1` (no-schedule fallback) is
  cheap-but-weak.** It reduces to "the velocity at adjacent timesteps
  should agree", which is a very loose constraint on the velocity
  field. If you turn on `heun_smoothness_weight` without configuring
  a schedule, you get the regularizer in name only. Configure a
  schedule covering `s ∈ [1/64, 1/4]` or so for a real signal.

## Open follow-ups

- [ ] **Patches 1–3 of the refactor** (add regularizer, migrate live
      configs, delete `two_step` code). Tracked in
      [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].
- [ ] First A/B between **`distillation` alone** and
      **`distillation` + `heun_smoothness`** at fixed adapter family
      + dataset; measure shortcut-rollout fidelity (1-step / 2-step /
      4-step prediction error vs. a 50-step base rollout). _no run
      logged yet._
- [ ] Confirm whether `DynamiCrafterOutputAdapter` or
      `ShortcutDirectionOutputAdapter` is the right architectural
      pairing for the D3 headline; current configs pick the former.
      _design choice not yet decided — candidate for
      `50_Decisions/open/`._
- [ ] Resolve whether the duplicate-MSE-heads
      (`shortcut_direction_weight` vs. `local_consistency_weight`)
      pair is worth keeping post-refactor. They share the
      `distillation` target tensor; one knob with one name would be
      less confusing. Defer to patch 2 of the refactor.
- [ ] Decide whether `heun_smoothness` should refuse to start without
      a configured `shortcut_step_schedule`, or accept the
      `s = 1` fallback silently (current plan: accept silently with a
      "gotcha" callout above). Revisit after the first regularized
      run.

## Related

- [[../theory/shortcut-training]] — conceptual framing of D3;
  §4.2 now reflects `distillation` as the only mode and points to
  this catalogue.
- [[../theory/heun-shortcut-target]] — numerical-methods view of the
  Heun construction; §5 documents the deprecation rationale, §6
  superseded by the decision.
- [[../theory/heun-smoothness-regularizer]] — derivation of the new
  regularizer.
- [[../theory/ddim-step-v-parameterisation]] — DDIM single-step
  primitive both `distillation` and the regularizer use.
- [[../theory/prediction-objectives]] — v / ε / x_0 parameterisation
  context.
- [[../related-work/shortcut-models]] — the Frans et al. 2024 paper
  `distillation` implements faithfully.
- [[../related-work/consistency-models]] · [[../related-work/self-distillation]]
  — neighbouring approaches in the few-step-sampling cluster.
- [[dynamic-rescale]] — the DynamiCrafter `scale_arr` that the DDIM
  micro-step in both teacher pathways routes through.
- [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] —
  the decision that drove this rewrite.
- [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]
  — implementation plan.
- [[../../10_now/architecture]] — should reference this note from the
  shortcut section once that section is fleshed out.
