---
type: paper
status: living
last_updated: 2026-05-15
title: "CAFM — _needs verification_"
authors: []
venue:
year:
url:
local_pdf: docs/paper/cafm.pdf
relevance: framework
deliverable: D1
---

# CAFM

> Working assumption: a paper in the conditional-adapters-for-flow-matching
> neighbourhood, vendored at `docs/paper/cafm.pdf`. **Title, authors,
> venue, contribution all _need verification from the PDF_** before this
> note can be cited from the thesis. Do not put this paper into a chapter
> draft until that's done.

## Status of this note

**Stub — placeholder until the PDF is read.** Per [[../../CLAUDE]] hard
rule 7(a), this entire note is currently `_needs verification_` and
should not be summarised as fact by downstream synthesis.

If the PDF turns out to be unrelated to flow-matching adapters, this note
should be retitled and re-categorised — or moved out of the seed eight.

## Why it might matter for the thesis (hypothesis only)

If CAFM is what the filename suggests:
- It would be a direct precedent for the flow-matching half of D1.
- It would constrain whether the thesis's flow-matching coverage is
  novel or a replication with extensions.
- It would belong in the same "frozen base × adapter" quadrant as AVID,
  HyperAlign, UniCon.

All of the above is **conditional on PDF verification**.

## Open questions for the chapter

- Exact title, authors, venue, year, URL. _needs verification_.
- Adapter type (output? hidden-state? LoRA? hypernetwork?). _needs
  verification_.
- Whether the base model stays frozen. _needs verification_.
- Whether the paper considers any conditioning beyond text (action,
  step-size, multimodal). _needs verification_.

## Related

- [[_MOC]]
- [[avid]] · [[hyperalign]] · [[unicon]] — the other adapter-side neighbours, all currently better-mapped to the codebase than CAFM
- `docs/paper/cafm.pdf`
