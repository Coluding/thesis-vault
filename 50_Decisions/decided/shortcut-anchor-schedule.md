---
type: decision
status: decided
created: 2026-05-21
decided_at: 2026-05-21
updated: 2026-05-21
target_date:
scope: training
related:
  - "[[avid-adapter-init]]"
  - "[[../../20_Tickets/risk-shortcut-self-consistency-collapse]]"
  - "[[../../20_Tickets/feat-shortcut-add-d-zero-gate]]"
  - "[[../../30_Knowledge/related-work/shortcut-models]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
---

# Decision: Step warmup on the d=1 data anchor probability

## Status

**Decided 2026-05-21.** Downstream of [[avid-adapter-init]] (resolved
to D — status quo, so step-0 prediction is `0.5·base + 0.5·random`).

## Context

D3 shortcut training has two loss components:

1. **d=1 data anchor** — standard diffusion/flow loss at the smallest
   non-trivial step. Supervises the composed predictor `s` against
   empirical velocity samples. This is the only term in the loss that
   touches the data; per
   [[../../20_Tickets/risk-shortcut-self-consistency-collapse]] it is
   what rules out the trivial fixed points of the self-consistency
   loss (cancellation collapse, constant-field collapse,
   self-consistent-but-data-inconsistent solutions).
2. **Self-consistency at d ∈ {2, 4, ..., K}** — recursive bootstrap.
   The teacher target is built from chained no-grad calls of the
   *adapted* model: `s(x'_{t+d}, t+d, d)`.

The shortcut-models paper uses a fixed batch split ~3/4 anchor /
~1/4 consistency throughout training (see
[[../../30_Knowledge/related-work/shortcut-models]]).

Under [[avid-adapter-init]] = D, step-0 prediction is
`0.5·base + 0.5·random_adapter_noise`. The self-consistency loss is
then recursively bootstrapping its teacher target from chained calls
of a model that is itself producing random output. **Early
self-consistency gradient is anti-informative** until the adapter
learns to predict structured velocities. The paper didn't face this
because they trained the entire model from scratch (no frozen base);
our setup is different.

## Decision

**Step warmup.** For the first **N** optimizer steps, set the
self-consistency loss weight to zero — train only on the d=1 data
anchor. After step N, restore the configured weight and recover the
paper's batch split (~3/4 anchor, ~1/4 consistency).

**Initial N = 5000 steps.** _Analysed estimate_, not a measured
optimum. Reasoning: order-of-magnitude budget that should let the
adapter's d=1 prediction error drop below the base's reconstruction
error on MetaWorld before the consistency teacher comes online. Ablate
against {1000, 5000, 20000} once the implementation lands.

## Why step warmup over alternatives

- **Constant ratio (status quo / paper):** wastes early-training
  compute on a noisy-teacher self-consistency target. Defensible
  *only* if you also commit to a clean-step-0 init (which we did not
  — [[avid-adapter-init]] resolved to D).
- **Smooth decay with floor (cosine / linear from p_start → p_floor):**
  strictly more flexible. Adds three hyperparameters (start, end,
  shape) for a payoff that the step warmup also delivers
  structurally. Reach for this only if the binary on/off proves too
  crude on the d=1 anchor loss curve.
- **Step warmup (chosen):** one new knob (N). Structurally identical
  to the curriculum the self-distillation literature already uses:
  pretrain student against data, *then* introduce the distillation
  term. Clean to defend in the thesis.

## Consequences

- **New config key:** `training.shortcut_anchor_warmup_steps: int`
  (default 0 in non-shortcut configs to preserve behaviour; 5000 in
  live shortcut YAMLs).
- **Loss wiring:** the self-consistency loss weight is gated to 0 for
  `global_step < shortcut_anchor_warmup_steps`. After the threshold,
  it switches to the configured weight (the existing
  `shortcut_direction_weight` in `src/generative_flow_adapters/config.py:65`).
- **Anchor floor (~3/4) is retained post-warmup.** Never decay anchor
  below the paper's ratio — collapse fixed points need the data term
  active throughout training, not just during warmup.
- **One extra hyperparameter to ablate.** {1000, 5000, 20000} ×
  {two_step, distillation} on MetaWorld, captured in the follow-up
  experiment ticket once a baseline run exists.
- **Departure from the paper's fixed split (in the early phase).**
  Defensible because we have a frozen base (paper trained from
  scratch); our early adapter is acting on top of an already-good
  one-step predictor, so the d=1 fit is the right initial target.

## Follow-ups (tickets)

- [[../../20_Tickets/feat-training-shortcut-anchor-step-warmup]] —
  implementation of the warmup gate in the training loop.
- Future ablation ticket (open once baseline run exists): warmup-N
  sweep × {two_step, distillation} targets.

## Open questions

- **N too short ⇒ same noisy-teacher problem; N too long ⇒ wasted
  compute on a converged anchor.** Reasonable upgrade path if fixed-N
  proves brittle: switch when an EMA of the d=1 anchor loss flattens
  below a threshold. Carry as "adaptive N" follow-up.
- **Interaction with AVID's mean-mask trajectory.** AVID's Fig. 4d
  shows the mean mask drifting per-timestep at convergence. The
  warmup phase trains only the d=1 loss — does that bias the early
  mask trajectory in a way the steady-state phase has to undo? _needs
  verification — log per-timestep mean-mask curve through the warmup
  boundary._
- **Does the warmup apply to two_step targets too?** The two_step
  shortcut target is derived from the *base* (Heun average of base
  velocities), not from the adapter — so it is not subject to the
  noisy-teacher problem in the same way. The distillation mode is
  the primary motivation here. Possibly the warmup should be
  loss-mode-conditional, only active under `shortcut_target_method:
  distillation`. Carry to the implementation ticket.

## Related

- [[avid-adapter-init]] — the 0.5+0.5 step-0 state under D is the
  precise motivation for this schedule. The two decisions are joined
  at the hip.
- [[../../20_Tickets/risk-shortcut-self-consistency-collapse]] — the
  data anchor floor (~3/4 post-warmup) is the structural mitigation
  for cancellation collapse. Do not let the warmup logic accidentally
  decay the anchor past that floor.
- [[../../20_Tickets/feat-shortcut-add-d-zero-gate]] — Option A from
  the same risk. Orthogonal to this schedule; both could coexist but
  the schedule is load-bearing first.
- [[../../30_Knowledge/related-work/shortcut-models]] — paper's fixed
  3/4-1/4 split (the steady-state baseline we recover after warmup).
- Code anchors:
  - `src/generative_flow_adapters/config.py:65-68` — existing shortcut
    loss-weight knobs; new warmup key sits adjacent.
  - `src/generative_flow_adapters/training/trainer.py` — warmup gate
    implementation site (exact lines TBD).
