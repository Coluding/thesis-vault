---
type: paper
status: living
last_updated: 2026-05-15
title: "Self-Distillation — _exact title needs verification_"
authors: []
venue:
year:
url:
local_pdf: docs/paper/self_distillation.pdf
relevance: theory
deliverable: D3
---

# Self-Distillation

> Self-distillation as a route to few-step sampling. Methodological cousin
> of consistency / shortcut. Likely a relevant ancestor for the loss form
> behind D3.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/self_distillation.pdf`. Exact
title, authors, venue, year, URL — _needs verification from the PDF_.
There are multiple papers in this space (e.g. progressive distillation,
self-distillation through time, etc.) — confirm which one this is before
citing.

## Why it matters for the thesis

- Self-distillation is the third leg of the few-step-sampling triangle
  (the other two being [[consistency-models]] and [[shortcut-models]]).
- The D3 chapter must distinguish between the three approaches:
  - **Consistency Models** — learn an `s(x_t, t)` function with a
    consistency property.
  - **Shortcut Models** — learn an `s(x_t, t, d)` function with multi-step
    self-consistency.
  - **Self-distillation** — distil a many-step model into a few-step
    model.
- This paper is the spot to anchor the "we use the loss but not the
  distillation regime" framing — *if* the loss form turns out to match
  ours, which it might.

## Key relationships to capture

- Whether self-distillation requires a *teacher* (a separately trained
  many-step model) or whether it's online (the model distils from itself
  across training steps). Affects compatibility with the frozen-base
  setup.
- How the loss form relates to consistency / shortcut. The thesis's
  theory note should triangulate all three.

## Open questions for the chapter

- Exact title, authors, venue. _needs verification_.
- Whether it uses a teacher or is self-bootstrap. _needs verification_.
- Reported few-step quality numbers for honest comparison.
- Whether any of the codebase's losses (`local_consistency`,
  `multistep_self_consistency`, `shortcut_direction`) directly implement
  this paper's objective or a near-cousin.

## Related

- [[_MOC]]
- [[shortcut-models]] · [[consistency-models]] · [[dpm-solver]] — the few-step-sampling cluster
- [[../../10_now/positioning]] — D3 deliverable
- `30_Knowledge/theory/` — derivations (to be populated)
