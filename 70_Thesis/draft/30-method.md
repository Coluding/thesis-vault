---
section: method
status: drafting
deliverable: D1
sources:
  - "[[../../10_now/architecture]]"
  - "[[../../30_Knowledge/tech/structural-encoder]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../50_Decisions/decided/shortcut-anchor-schedule]]"
  - "[[../../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[../../60_Updates/entries/2026-07-24-online-vae-encode-6x-training-step]]"
last_updated: 2026-07-24
---

# 3. Method (D1 — Framework)

## 3.1 Composition interface
_Stub — `f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)`.
Source: [[../../10_now/architecture]]._

## 3.2 Adapter taxonomy and the shared conditioning path
_Stub — the structural encoder as the single conditioning path every adapter
family consumes. Source: [[../../30_Knowledge/tech/structural-encoder]]._

## 3.3 Conditioning interfaces

_Stub — action conditioning, step-size conditioning. The frame-stride (`fs`)
boundary: base-model conditioning vs. framework conditioning. Source:
[[../../30_Knowledge/tech/frame-stride-conditioning]]._

### 3.3.x The frame-stride boundary (drafting)

The composition rule
`f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)` lists
the framework's conditioning variables explicitly: state `x_t`, time `t`,
action `a_t`, step-size `d`. The frozen base may consume additional
inputs that are part of *its own* training contract — in the
DynamiCrafter case, a frame-stride / fps scalar `fs` routed into the
base UNet's time embedding via a learned `fps_embedding` MLP. These
base-private inputs sit **outside** the framework's conditioning
surface: they reach `f_base` but do not enter `Δ_φ`.

For the audited HyperAlign and UniCon adapter wirings (see
[[../../30_Knowledge/tech/frame-stride-conditioning]]), `fs` reaches the
frozen base only. The adapter's explicit conditioning inputs remain
`(a_t, d)`; any temporal-resolution information `Δ_φ` consumes is
**indirect**, mediated by the base's activations that the adapter
reads via its feature hooks. We adopt this as a deliberate framework
choice: the contribution rule is action-and-step-conditioned, not
fps-conditioned. Should a later result motivate making `fs` an
explicit input to `Δ_φ`, the change would extend the composition rule
and belongs in a separate decision rather than mid-draft (CLAUDE.md
Part 12). The current revisit triggers are recorded in
[[../../50_Decisions/decided/per-sample-frame-stride-sampling]].

The *numerical value* fed into the base's `fs` channel is therefore a
data-side detail, not a framework parameter; the dataset choice is
documented in §4.1.

## 3.4 Shortcut training

_Stub — local consistency + multi-step self-consistency; two_step vs.
distillation modes. Sources:
[[../../30_Knowledge/tech/shortcut-training-modes]],
[[../../50_Decisions/decided/shortcut-anchor-schedule]]._

### 3.4.x Target construction for the shortcut loss (drafting)

D3's training signal compares the adapted model's velocity at a stepped
`step_level` against a self-supervised target that the frozen base (or
the adapter itself) provides. We use two target-construction families;
both are written for **diffusion v-prediction** (`model_type=diffusion`,
`prediction_type=velocity` in the codebase), and both rely on a
single-step DDIM micro-step under the training schedule —
`ddim_micro_step_v` in
`src/generative_flow_adapters/training/shortcut_targets.py`. Per-sample
timesteps `t, t' ∈ [0, T_train)` are supported so a batch can carry
heterogeneous `(t, t')` pairs without scheduler-side bookkeeping.

**Base-anchored two-step (Heun-corrected).** Given the frozen base
velocity `v₀ = f_base(x_t, t)`, advance one schedule step to
`t' = max(0, t - 1)` via the DDIM v-prediction update `x_mid =
ddim_micro_step_v(x_t, v₀, t, t')`, evaluate the base again at the
advanced state and time, `v₁ = f_base(x_mid, t')`, and form
`y_target = (v₀ + v₁) / 2`. This is the
average-of-velocities-at-endpoints construction (Heun-style), anchored
on the frozen base — no collapse risk, and the adapter's `step_level`
is decorative under this mode (the supervision does not depend on it).
The construction uses the *adapted* state pair `(x_t, t)` the adapter
sees at training step, so target and prediction live in the same
schedule frame.

**Paper-faithful self-consistency** (Frans et al. 2024, eq. 4). Given
a sampled `d` (raw timesteps or, under a configured paper-faithful
step schedule, normalised `s ∈ (0, 1]` mapped to a timestep jump), the
target is the average of two no-grad calls of the **adapted** model
at the *half* step `d/2`, chained across one `d/2`-sized DDIM
micro-step: `v₁ = f_model(x_t, t, cond_{d/2}); x_mid = ddim_micro_step_v
(x_t, v₁, t, t - d); v₂ = f_model(x_mid, t - d, cond_{d/2});
y_target = (v₁ + v₂) / 2`. The adapter is its own teacher; the frozen
base contributes only implicitly through the composition inside
`f_model`. Anchoring at the finest step is provided by a separate
standard diffusion loss with probability `shortcut_anchor_prob` (see
[[../../50_Decisions/decided/shortcut-anchor-schedule]] for why the
anchor matters).

Both formulations honour the temporal advance — the second velocity
sample is taken at the **advanced** time, not at `t` — which is the
single mathematical statement the paper-faithful self-distillation
relies on. An earlier intermediate implementation evaluated the second
base call at the same `t`, which we have since removed
(`bug-training-shortcut-target-timestep`, closed: no live or dead path
in the codebase still simplifies in that way). Documenting this here so
the eventual D3 claim — that we evaluate against the paper-faithful
target — is unambiguous.

## 3.5 Computational profile of the composition

The composition rule `f_base(x_t, t) + g(d)·Δ_φ(·)` has a direct
consequence for training cost that shapes two framework-level design
choices: the frozen base is evaluated **on every optimiser step**, and
the pixel→latent VAE encode is factored **out** of the loop. Both are
justified empirically below. The numbers are from the WAN-2.2 TI2V-5B
base with the output adapter on the ACWM push_block geometry
(768²-pixel windows, 65 frames → latent 17×48×48, batch size 1),
profiled with CUDA-synchronised per-phase timers (`GFA_PROFILE`), and
they should be read as **operating-point characterisation for this
base/geometry**, not architecture-independent constants.

**The frozen base dominates the step.** Because `Δ_φ` is a *residual*
on `f_base`, the frozen base forward cannot be cached across steps — it
depends on the freshly sampled `(x_t, t)` — so it is recomputed every
step under `no_grad`. Measured per step (cached-latent path, run
`8cug8wfq`): frozen 5B base forward ≈ 480 ms, adapter forward ≈ 28 ms,
backward ≈ 54 ms, optimiser ≈ 3 ms, for a total step of ≈ 580 ms. The
frozen base is thus ≈ **83 %** of the step while the *trainable* adapter
— the only part carrying gradients — is well under a fifth. This is the
defining cost signature of plug-and-play adaptation on a large frozen
prior: parameter efficiency (here 34.97 M trainable of 5.03 B total,
0.69 %) does **not** translate into a proportionally cheap step, because
the frozen forward is on the critical path. The practical implication —
that the throughput ceiling lives on the base side, so any speedup must
target the frozen forward (kernel / compilation) rather than the
adapter — is taken up in the Discussion.

**Latent precompute is a throughput decision, not a memory one.** The
online pixel→latent VAE encode at this geometry costs ≈ 3.66 s per
clip (run `hswppa8s`, `--no-latent-cache`), against the ≈ 0.58 s
training step — i.e. the encode alone is ≈ **6×** the entire step, and
online encoding runs at ≈ 0.24 steps/s versus ≈ 1.44 steps/s from a
warm latent cache (≈ 6× wall-clock). The encode transient coexists
comfortably in memory with the resident 5B (peak ≈ 26 GB reserved on an
80 GB device), so — unlike the earlier native-resolution regime, where
precompute was an out-of-memory workaround — the justification here is
purely wall-clock. We therefore keep an offline latent-precompute stage
as the framework default and treat online encoding as a fallback for
cases a static cache cannot cover (e.g. unbounded random-window
sampling). Full numbers and run provenance:
[[../../60_Updates/entries/2026-07-24-online-vae-encode-6x-training-step]].
