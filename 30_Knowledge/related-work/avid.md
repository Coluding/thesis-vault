---
type: paper
status: living
last_updated: 2026-05-15
title: "AVID — Adapter-based Video Diffusion"
authors: []
venue:
year:
url:
local_pdf: docs/paper/avid.pdf
relevance: framework, baseline
deliverable: D1, D2
---

# AVID

> Output-level residual adapter for pretrained video diffusion. Closest
> precedent for the thesis's "additive correction on a frozen base"
> architecture; we generalise the idea to multiple adapter families and
> add action / step-size conditioning.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/avid.pdf`. Title, authors, venue
year, URL — _needs verification from the PDF_. The "AVID" working name is
how the codebase refers to it (e.g.
`configs/diffusion_output_avid_training_test.yaml`,
`src/external_deps/avid_utils/`).

## Why it matters for the thesis

- The thesis's framework (D1) explicitly mirrors AVID's
  "frozen-base + trainable residual" shape, then generalises it across
  four adapter families. The thesis must cite AVID as the direct
  architectural precedent for the output-adapter family.
- AVID does not condition on actions or step-size. The D2 and D3
  contributions of the thesis are exactly the two axes AVID does not
  cover.
- The codebase contains an explicit AVID-style starting point:
  - `configs/diffusion_output_avid_training_test.yaml` — replicates the
    AVID setup on top of DynamiCrafter.
  - `src/external_deps/avid_utils/` — vendored AVID evaluation utilities.

## How it maps to our adapter taxonomy

- AVID's adapter is **output-level** in our terminology: it adds a
  trainable correction on top of the frozen base's output (the noise
  prediction or velocity).
- In our factory it lives under `adapters/output/`, with the AVID-flavoured
  variant most closely matched by `dynamicrafter` (the
  `DynamicCrafterOutputAdapter`, see `adapters/output/dynamicrafter.py`).
- The composition mode used by AVID (whether the adapter adds, replaces,
  or mask-mixes the base output) — _needs verification from the PDF_.

## What this paper does that we don't (yet)

- AVID is video-specific. Our framework is backbone-agnostic.
- AVID's evaluation protocol on video — _needs verification_. Worth
  matching where reasonable so our D2 comparison can include an
  AVID-replica row.

## What we do that AVID doesn't

- Action conditioning (D2).
- Step-size conditioning + consistency training (D3).
- Multiple adapter families behind one composition rule (D1).
- Coverage of flow matching, not just diffusion (D1).

## Open questions for the chapter

- Exact composition equation as written in the AVID paper. _needs verification_.
- Whether AVID reports any few-step-rollout numbers (likely not, but
  worth confirming).
- Whether AVID's residual is gated (similar to our `gate_bias` /
  `composition: mask_mix`) or pure-additive. _needs verification_.

## Related

- [[_MOC]]
- [[../../10_now/architecture]] — see Adapter families, output column
- [[hyperalign]] · [[unicon]] · [[cafm]] — the other adapter-side neighbours
- `src/external_deps/avid_utils/`
- `configs/diffusion_output_avid_training_test.yaml`
