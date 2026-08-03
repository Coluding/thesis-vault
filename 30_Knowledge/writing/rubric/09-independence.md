---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: independence
category: attitude
current_band: "8-9"
target_band: "9"
sources:
  - "[[_index]]"
  - "[[../../experiments/_index]]"
  - "[[../../../10_now/positioning]]"
---

# Rubric 9 — Independence

## The rows

| | |
|---|---|
| 10 | Actually did not need any assistance. Completely aware of the problem and the thesis |
| **9** | **Has knowledge and insight on a scientific level, i.e. explores solutions on their own, increasing their skills and knowledge where necessary** |
| 8 | Able to adopt new skills mostly independently, asks for assistance from the supervisor if needed |
| 7 | Selects and plans tasks together with the supervisor, performs them alone |
| 6 | Supervisor mainly responsible for setting out tasks; student performs them mostly independently |
| 1–5 | Can only perform the project properly after instructions and with help |

## What it actually asks

This is scored by the **supervisor and committee from the working
relationship**, not from the document. It is largely already determined by
how the last months went. But it is not *entirely* outside the document:
a committee forms an independence impression from **whether the thesis
shows its own reasoning**.

The 9-row's operative phrase is "**explores solutions on their own,
increasing their skills and knowledge where necessary**" — i.e. the student
went and learned what the problem needed.

## Evidence from the record

The campaign supports 9 on its face:

- **Self-designed instruments.** The probe suite (propagation trace,
  structure triad, Jacobian sensitivity, rollout-swap) was built because
  nothing existing answered the question. That is the 9-row literally.
- **Self-detected confounds.** `effect_rel`'s gain ambiguity, the in-sample
  evaluation, the silent 98.5% episode drop, the frozen-gate bug — all
  found internally, not flagged by a supervisor.
- **Skills acquired as needed.** Diffusion *and* flow-matching
  parameterisations; UNet *and* DiT internals; a published method ported to
  a new base family inside its own repository.
- **Self-corrected framing.** The AVID repositioning and the 08-02 reframe
  were both initiated from the evidence, against earlier commitments.

## How the document supports it

Independence is read from **whether design choices are attributed to
reasoning or presented as given**. Every instance of *"we chose X because
Y, having ruled out Z"* is an independence signal; every unexplained
default is a missed one.

Concretely, the thesis should show its own reasoning at these points:

- Why the output adapter family (the two selection criteria in Ch3 —
  weight/internals access, gradients through the frozen base)
- Why ACWM as a second dataset, and what "action-informative" means
- Why a probe suite rather than the standard readouts — *and how we knew we
  needed one*
- Why the endpoint-inversion target, derived rather than adopted
- Why each hypothesis was tested in the order it was

## Optimisation queue

- [ ] **Attribute design decisions in-text**, not just in
      `50_Decisions/`. The decided notes contain the reasoning; the thesis
      currently would not.
- [ ] **Show the instrument-building as a decision**, not as infrastructure:
      "the standard readouts could not distinguish X from Y, so we built…"
      This is the clearest independence signal available and it is the same
      text as [[05-reflection]] Q6.
- [ ] **Keep the supervisor-facing narrative consistent** with the thesis's
      claims — the weekly `60_Updates/` entries are where this impression
      is formed over time. Log the reframe and its evidence.
- [ ] **Own the negative results in the first person.** "We could not
      distinguish…" reads as independence; passive constructions read as
      distance from the work.

## Where it lands in the thesis

- Ch3 — the family selection, derived
- Ch4 §4.5 — the target derivation
- Ch5 — the probe suite as a reasoned response to a measurement problem
- Throughout — "we chose X because Y, having ruled out Z"
- Outside the document: `60_Updates/` and the advisor meetings
