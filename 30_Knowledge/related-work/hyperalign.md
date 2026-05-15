---
type: paper
status: living
last_updated: 2026-05-15
title: "HyperAlign — Hypernetwork-Generated LoRA Adapters for Diffusion"
authors: []
venue:
year:
url:
local_pdf: docs/paper/hyper_align.pdf
relevance: framework, baseline
deliverable: D1, D2
---

# HyperAlign

> Hypernetwork that produces task-specific LoRA weights for a frozen
> diffusion model. The reference for the thesis's hypernetwork adapter
> family. Vendored as a starting point in the implementation repo.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/hyper_align.pdf`. Title, authors,
venue, year, URL — _needs verification from the PDF_. The thesis docs at
`docs/hyperalign-architecture-replication.md` are the most thorough
existing notes — extract the key sections into this file when fleshing
the stub out.

## Why it matters for the thesis

- HyperAlign is the direct precedent for the **hypernetwork adapter
  family** (D1).
- The codebase contains a working replication: the HyperAlign adapter
  class lives in `adapters/hypernetworks/`, the target-module set is
  cached as `PAPER_HYPERALIGN_TARGET_MODULES` in
  `adapters/low_rank/common.py`, and several configs build on it:
  `configs/diffusion_hyperalign_action.yaml`,
  `diffusion_hyperalign_fake_action.yaml`,
  `diffusion_hyperalign_metaworld.yaml`,
  `test_dynamicrafter_hyperalign_unet.yaml`,
  `diffusion_hyper_lora_action.yaml`.
- A standalone training entrypoint exists at
  `scripts/train_hyperalign_metaworld.py`. This is currently the closest
  path to a real D2 / D4 run.

## How it maps to our adapter taxonomy

- HyperAlign is a **hypernetwork** that emits **LoRA** weights for a set
  of target modules. In our factory it lives under
  `adapters/hypernetworks/HyperAlignAdapter` and reaches the base
  model's `nn.Linear` layers via the LoRA injection scaffolding in
  `adapters/low_rank/lora.py`.
- The thesis's framing: hypernetwork is a *meta*-adapter that
  parameterises another adapter family (LoRA). This composability is
  part of D1's contribution.

## What this paper does that we don't (yet)

- HyperAlign's exact conditioning interface (which task/context features
  it accepts) — _needs verification_. Ours generalises to action vectors
  and step-size; HyperAlign's conditioning surface is narrower.
- Empirical scope (which datasets / backbones / tasks they ran) — _needs
  verification_.

## What we do that HyperAlign doesn't

- Action conditioning for world-model use (D2).
- Step-size conditioning + consistency training (D3) — including the
  `diffusion_hyper_lora_action.yaml` / `flow_hyper_shortcut_stepwise.yaml`
  configs which lift HyperAlign into the shortcut regime.
- Unified framework spanning all four adapter families, not just
  hypernetwork → LoRA (D1).

## Open questions for the chapter

- Which exact base model HyperAlign uses — _needs verification_.
- Whether the hypernetwork is per-layer or shared — _needs verification_
  (our `HyperAlignAdapter` makes a specific choice, document the match).
- Whether HyperAlign trains end-to-end on the downstream task or in a
  separate alignment stage. _needs verification_.

## Related

- [[_MOC]]
- [[../../10_now/architecture]] — see Adapter families, hypernetwork column
- [[avid]] · [[unicon]] · [[cafm]] — the other adapter-side neighbours
- `docs/hyperalign-architecture-replication.md` — existing in-repo notes
- `src/generative_flow_adapters/adapters/hypernetworks/`
- `scripts/train_hyperalign_metaworld.py`
- `configs/diffusion_hyperalign_action.yaml` and siblings
