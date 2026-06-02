---
type: paper
status: living
last_updated: 2026-05-19
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

## Conditioning bandwidth: weight-modulation is the only path

**Analysed estimate** (not a measured result). Source: code inspection of
`src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py` at
commit `7680e82` (HEAD as of 2026-05-19), conversation with user 2026-05-19.

### The observation

In our HyperAlign replication, the action conditioning reaches the
adapted prediction **through exactly one channel**: the per-sample LoRA
factors installed onto the targeted attention projections. There is no
direct activation-path injection of the action embedding into the
adapted forward pass.

Concretely, in `HyperAlignAdapter.forward` (hyperalign.py:251-291):

1. The reference pass at line 269 calls
   `base_model(x_t, t, cond=_resolve_base_condition(cond))`.
   `_resolve_base_condition` (lines 538-543) **strips `cond["embedding"]`**
   before forwarding.
2. The hypernet pass (`build_hyper_input`, lines 146-179) is the **only**
   place the action embedding is consumed — it builds memory/query tokens
   for the transformer decoder which emits the LoRA factors.
3. The adapted pass at line 279 again calls
   `base_model(x_t, t, cond=_resolve_base_condition(cond))` with the
   embedding stripped. The only thing that differs from the reference
   pass is the dynamic LoRA modulation installed at lines 272-277 via
   `handle.wrapped.set_dynamic_hyper_factors(...)`.

This is **deliberate**, as documented in two comments in the file
(hyperalign.py:434-438 and the docstring of `_prepare_hyperalign_runtime`
at lines 630-636): the frozen base UNet was never trained to interpret
the adapter's condition embedding, so it is stripped from the base call
and the conditioning signal is routed through the hypernetwork instead.

### Information flow (simplified)

```
action embedding ──► hypernet decoder ──► (down, up) LoRA factors ──► attention proj weights
                                                                            │
                                                          x_t, t ───────────┴───► adapted output
```

### Two caveats — `context` and action-conditioned bases

- **`context` (text/image cross-attention) is not stripped.** Only the
  adapter-specific `embedding` key is removed. CLIP context and concat
  conditioning still flow directly into both base passes. The
  "weight-modulation-only" property therefore applies specifically to
  whatever we packed into the adapter's `embedding` — in practice the
  action signal.
- **`action_conditioned=True` bases are an exception.** At
  hyperalign.py:657-674, if the base UNet was itself trained with an
  action head, `cond["act"]` is still routed through `module.action_embed`.
  In that case the action enters via two paths (native head + LoRA
  modulation). For the pure HyperAlign use case (frozen non-action
  base), only via the LoRA path.

### Is this a bottleneck? — analysed estimate

**Most likely not a hard bottleneck for action conditioning, but
expressivity-limited for spatially/temporally heterogeneous action
effects.** Reasoning:

- **Shannon capacity is not the issue.** With our typical setup
  (`rank=8`, ~30 target attention projections, `aux_down_dim=aux_up_dim=16`)
  the hypernetwork emits roughly `8 × 30 × (16+16) ≈ 8k` learned numbers
  per sample. That is more than enough bandwidth for any action vector
  we plan to use.
- **Expressivity is the real question.** A static rank-`r` perturbation
  on Q/K/V/O can re-route attention based on `x_t` (since attention
  itself is `x_t`-dependent), but the perturbation is constant across
  the forward — it cannot be read differently at different layers or
  spatial positions in an `x_t`-dependent way.
- **Inductive prior cushions this.** A well-trained video base already
  models plausible dynamics; the LoRA only needs to *nudge* an existing
  competent prior, not teach new capabilities. For discrete or
  simple-continuous actions on a domain the base understands, this
  should be sufficient.
- **Where it would bite:** when the action's effect on the prediction
  is locally heterogeneous in a way that depends on `x_t` (e.g.
  "this gripper action only matters at the contact point on frame 7").
  A constant low-rank weight delta is a poor representation for that.

### Diagnostic + validating experiment

If the adapter learns coarse global shifts but fails on fine local
control during D2 (action-conditioned MetaWorld) runs, this is the
suspect. Cheapest test to settle it:

- **A:** HyperAlign as-is (baseline).
- **B:** HyperAlign + a small parallel `output` residual adapter (gives
  the action a direct activation-path pathway alongside the
  weight-modulation pathway).

If B lifts noticeably over A on action-following accuracy or local
prediction MSE, the LoRA bandwidth was the bottleneck. If they are a
wash, the weight-modulation path was sufficient. Note: existing
`condition_injection_mode=cross_attention` only conditions the
hypernet queries, **not** the base forward — it does not test this
hypothesis.

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
