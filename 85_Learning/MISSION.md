# Mission: Parallel Decoding Distillation (PDD) and few-step distillation of flow models

> Draft — written 2026-08-03 from session context, **awaiting Lukas's confirmation**.
> If the "Why" is wrong, say so and I'll rewrite before the next lesson.

## Why

Deliverable **D3** of the thesis needs few-step generation to actually work, and
the current shortcut/self-consistency recipe is learnable but not yet producing
good few-step video. PDD (arXiv:2607.26004) is a newer, simpler, *regression-only*
route to 4–8 NFE that claims to beat the VSD/GAN distillation family on exactly
the model families in use here (Wan, flow matching). A `teacher_rollout` target
inspired by it is already implemented in `generative-flow-adapters` and a run is
queued on the extra 10k SBU. The goal is to understand PDD well enough to know
**what was actually adopted, what was not, and which of the differences matter**
before spending the remaining compute — and to defend the choice in the thesis.

## Success looks like

- Can state the PD loss (Eq. 11) from memory and name every symbol in it.
- Can say precisely, without re-reading, where `compute_teacher_rollout_target_v_flow`
  diverges from PDD and what each divergence costs.
- Can decide whether the adapter should get PDD's replicated-head architecture, with
  a reason that survives an advisor asking "why not just condition on `d`?".
- Can position PDD against shortcut models, consistency models, Pi-Flow, and DMD2 in
  a related-work paragraph — from understanding, not from the paper's own framing.
- Can read the next trajectory-distillation paper in ~30 min and slot it into that map.

## Constraints

- **Compute is nearly exhausted.** ~10k SBU left. Lessons must sharpen decisions
  about what to run, not generate new things to run.
- Thesis deadline pressure — lessons stay short, one idea each.
- Strong existing background: flow matching, rectified flow, shortcut models
  (Frans et al.), diffusion/flow prediction types, adapter composition. Do not
  re-teach these. See [learning-records/0001](./learning-records/0001-prior-knowledge-flow-and-shortcut.md).

## Out of scope

- GAN/adversarial distillation internals (ADD, LADD, APT) beyond "why PDD avoids them".
- VSD / DMD2 derivations — needed only as a named contrast.
- Implementing PDD's full training loop from scratch. The question is what the
  *adapter* should borrow.
