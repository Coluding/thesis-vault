---
type: bug
scope: losses
status: open
priority: low
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]"]
---

# bug: purpose-built flow-matching boundary-sampling mitigation sits unused on the live WAN path

## What

`FlowMatchingTrainingObjective.sample_timesteps`
(`losses/flow_matching.py:53-106`) implements logit-normal + shift-schedule
timestep sampling — a purpose-built mitigation to avoid flat-uniform sampling
near the flow-matching boundary. Every live `diffusion_wan22_*.yaml` config
sets `use_batch_timesteps_for_flow: true`, which routes training through the
diffusion-forcing preprocessor's own flat-uniform sampler instead
(`sigma = torch.rand(...)`, `wan22_batch_preprocessor.py:128-145`) — the
shift-schedule sampler is dead code on the path that actually runs.

Found via the 2026-07-15 AVID-vs-ours structural comparison
([[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]).

## Correction to the original framing (from verification)

The initial hypothesis ("v-parameterization degenerates to an easy target
near t→0, so flat sampling there wastes gradient") does **not** hold —
v-parameterization has *constant* variance across the schedule by
construction, not a low-magnitude/easy region near t=0. Don't cite that
physics story — this is unrelated to
[[bug-losses-shortcut-v-averaging-target]] (a different, diffusion-side bias
issue).

What survives: this codebase built an explicit, purpose-designed mitigation
for exactly this training regime (flow matching near its boundary) and it is
silently switched off wherever the diffusion-forcing preprocessor's own
timesteps are used instead. Worth testing on its own narrower merits — "does
shifted/logit-normal sampling change anything here" — not the disproven
physics justification.

## Fix

Add a shifted/logit-normal sigma-sampling option directly into the
diffusion-forcing preprocessor (`wan22_batch_preprocessor.py`) — can't just
flip `use_batch_timesteps_for_flow` off, since the diffusion-forcing
`x_t`/`q_sample` construction there is specific to that preprocessor's own
per-frame timestep layout (observation frames pinned at 0, future frames at
sampled σ) and doesn't route through `FlowMatchingTrainingObjective` at all.

## Validate

A/B run: current flat-uniform sampling vs. the new shifted/logit-normal
option, same config otherwise. Cheap, decisive either way.

## Guardrails

Low priority relative to the gate-saturation fix and grad-accumulation/warmup
parity items — sequence after those per the diagnosis note's do-now order.
