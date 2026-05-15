---
type: paper
status: living
last_updated: 2026-05-15
title: "UniCon — Unified Hidden-State Conditioning for Diffusion"
authors: []
venue:
year:
url:
local_pdf: docs/paper/unicon.pdf
relevance: framework, baseline
deliverable: D1
---

# UniCon

> Hidden-state / skip-connection control adapter for diffusion. Direct
> reference for the thesis's hidden-state adapter family.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/unicon.pdf`. Title, authors, venue,
year, URL — _needs verification from the PDF_.

The repo README explicitly calls out a "UniCon-style Figure 3(d)
hidden-state starting point" via
`configs/diffusion_hidden_unicon_decoder.yaml`. The Figure 3(d) reference
is the diagram to anchor the chapter's description of this family.

## Why it matters for the thesis

- UniCon is the direct precedent for the **hidden-state adapter family**
  (D1).
- The codebase contains three UniCon-derived adapters in
  `adapters/hidden_states/unicon.py`:
  - `UniConHiddenStateAdapter` — the canonical replication.
  - `ReplaceDecoderHiddenStateAdapter` — the "replace decoder" variant.
  - `FullSkipLayerControlAdapter` — the full-skip controlnet-style
    variant.
- Corresponding configs:
  `diffusion_hidden_unicon_decoder.yaml`,
  `diffusion_hidden_replace_decoder.yaml`,
  `diffusion_hidden_full_skip_controlnet.yaml`.

## How it maps to our adapter taxonomy

- UniCon is a **hidden-state** adapter: it injects a learned signal into
  the base model's intermediate activations / skip connections rather
  than at the output. In our factory it lives under
  `adapters/hidden_states/` and reaches the base via
  `attach_base_model()`.
- The three variants above correspond to different injection points:
  - `unicon` → decoder hidden states.
  - `replace_decoder` → replaces the decoder hidden state entirely.
  - `full_skip_controlnet` → controlnet-style full-skip injection.

## What this paper does that we don't (yet)

- UniCon's exact target tasks (likely text-to-image or controlled
  generation rather than action-conditioned dynamics) — _needs
  verification_.

## What we do that UniCon doesn't

- Action conditioning for world-model use (D2).
- Step-size conditioning + consistency training (D3).
- Unified taxonomy across four adapter families (D1).

## Open questions for the chapter

- Exact diagram label for "Figure 3(d)" referenced in the codebase
  README — _verify from the PDF and pin the page number_.
- Whether the three connector types in our codebase
  (`zeroft`, `zeroconv`, …) come from UniCon directly or from a
  ControlNet-style adjacent paper. _needs verification_.

## Related

- [[_MOC]]
- [[../../10_now/architecture]] — see Adapter families, hidden-state column
- [[avid]] · [[hyperalign]] · [[cafm]] — the other adapter-side neighbours
- `src/generative_flow_adapters/adapters/hidden_states/unicon.py`
- `configs/diffusion_hidden_unicon_decoder.yaml` and siblings
