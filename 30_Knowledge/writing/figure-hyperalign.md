---
type: figure-design
status: living
last_updated: 2026-05-19
target: advisor-presentation
source: figures/hyperalign.html
deliverable: D1, D2
---

# Figure — HyperAlign architecture

> One-slide figure for the advisor presentation showing the HyperAlign
> hypernetwork architecture as it is implemented in this repo, and the
> precise injection point of the predicted LoRA factors into the frozen
> base U-Net.

## What this figure asserts

The figure makes the following claims, each traceable to code:

1. **Two passes through the frozen base.** A reference pass with no LoRA,
   and an adapted pass with the predicted LoRA factors set on the
   attention projections. Source: `adapters/hypernetworks/hyperalign.py`
   line 268–291 — the adapter calls `self.base_model(...)` twice and the
   output is `adapted - reference` (when `output_composition == "add"`).
2. **The hypernetwork's encoder is the frozen base's own input blocks.**
   Memory tokens are built from the U-Net `input_blocks` outputs,
   spatially mean-pooled per frame, then linearly projected to
   `hidden_dim`. Source: `_build_encoder_memory` line 449–480 and
   `_pool_video_encoder_features` line 589.
3. **Memory tokens are a (block × frame) grid.** Each `(layer, frame)`
   token gets a factorised sinusoidal positional encoding (layer + frame).
   Source: `_compose_memory_tokens` line 503–536.
4. **Conditioning enters the hypernetwork at one of two points.**
   - `condition_injection_mode == "memory_tokens"` → projected condition
     embedding is concatenated to the memory tokens. Source: line 440–447.
   - `condition_injection_mode == "cross_attention"` → cross-attention
     between zero queries and condition tokens before the decoder.
     Source: line 166–171, `_inject_condition_via_cross_attention` line 238.
5. **Decoder consumes zero queries — one per adapted LoRA module.**
   `query_count = len(self._handles)`, where each handle corresponds to
   exactly one wrapped `nn.Linear` in the U-Net's attention projections
   (one of `to_q`, `to_k`, `to_v`, `to_out.0` in a specific attention
   layer). Each query gets a sinusoidal positional encoding to give it
   a unique identity. The query-to-LoRA-module association is what is
   made explicit in the zoom panel below the main diagram. Source:
   query buffer init at `hyperalign.py:96–97`, `query_count` resolution
   at line 315, handle list assembled by `inject_hyperalign_lora_layers`
   in `adapters/low_rank/common.py:140`.
6. **Shared factor head produces all module factors at once.** The
   `_factor_head` is a single `Linear(hidden_dim, hidden_dim)` applied to
   every decoded token; output is reshaped into `hyper_down` and
   `hyper_up`. Source: line 306 and `_split_hyper_factors` line 413–419.
   The split-and-reshape sequence (first `r·a` values → `hyper_down`,
   next `r·b` values → `hyper_up`, then assignment to each
   `WrappedLoRALinear.set_dynamic_hyper_factors`) is what the factor-head
   zoom panel makes explicit.
7. **Auxiliary-factorised LoRA at injection.** Each target module wraps
   its `nn.Linear` with `WrappedLoRALinear`, which holds static
   `down_aux ∈ R^{d_in × a}` and `up_aux ∈ R^{b × d_out}` (with `up_aux`
   zero-initialised), and receives the dynamic per-sample factors
   `hyper_down ∈ R^{a × r}` and `hyper_up ∈ R^{r × b}`. Source:
   `adapters/low_rank/common.py`, `set_dynamic_hyper_factors`.
   - Effective LoRA: `A = A_aux @ A_hyper`, `B = B_hyper @ B_aux`.
   - Effective update: `ΔW · x = α/r · (B @ A) · x`.
8. **LoRA targets are attention projections only.**
   `to_q, to_k, to_v, to_out.0` — `PAPER_HYPERALIGN_TARGET_MODULES` in
   `adapters/low_rank/common.py` line 9.
9. **Three update modes.** `S` (stepwise — regenerate every denoising
   step), `I` (initial — once, reuse), `P` (piecewise — at progress
   markers). Implemented in `_should_refresh_hyper_factors` line 372–388.

## What this figure deliberately omits

- The output composition options `replace` and `mask_mix` and the
  associated gate heads. We show the canonical `add` path for clarity;
  composition variants are a separate figure if needed for D2/D3.
- The classifier-free dropout path on the condition embedding (handled
  by the structural encoder, separately figured).
- The `step_level` conditioning fusion (lives in
  `conditioning/utils/dynamicrafter_conditioning.py`); referenced as a
  small annotation rather than expanded.
- The trajectory-cache invalidation logic (timestep ordering test). One
  small annotation noting "S / I / P select when to refresh."
- The video-specific `[batch, frames, ...]` → `[batch × frames, ...]`
  reshaping inside the U-Net. The figure shows tensors at the abstract
  shape level only.

## Layout intent

Two stacked diagrams:

### Main diagram (top)

Two-row architecture overview:

- **Top row (hypernetwork):** input → perception encoder → memory tokens
  → transformer decoder (marked with 🔍 to flag the zoom below) →
  factor head → reshape.
- **Bottom row (frozen base):** `x_t, t, cond` → frozen U-Net (with one
  attention block expanded to show `to_q/k/v/out`) → output.
- **Vertical arrow** from factor head down into the expanded attention
  block, showing the LoRA factors landing on `W_q, W_k, W_v, W_out`.
- **Conditioning** enters from the left as a tagged `c` token, branching
  into both the hypernetwork (with the two injection modes shown as a
  switch) and the frozen base (via the existing `cond["embedding"]`
  fusion path).

### Zoom panel 1: Transformer Decoder internals

Visually framed as a dashed-bordered panel with a 🔍 badge (CSS class
`.zoom-panel`). Shows what the Decoder node abstracts away. The panel
is **always rendered, not interactive** — an earlier
`click + securityLevel: 'loose'` attempt to make it open on click
failed to parse under Mermaid 10 (see "Attempted (reverted)" below).
Contents:

- **Memory tokens** at the top, with the two-source composition
  spelled out: `T = L·F` tokens come from per-block features pooled
  from the frozen U-Net's `input_blocks` (perception encoder), and
  `cond_len` tokens come from the structural encoder. This makes the
  "memory tokens are a combination of input blocks AND structural
  encoder" point explicit.
- **M zero queries** in a horizontal row inside a `subgraph QGroup`,
  each labeled with the LoRA target it produces: `Q₁ → block 1 to_q`,
  `Q₂ → block 1 to_k`, …, `Q_M → block K to_out.0`. An ellipsis node
  indicates the omitted intermediate queries.
- **Cross-attention block** with the standard Q ← queries, K, V ←
  memory tokens annotation. The "× N decoder layers" stacking is shown
  as a small caption inside the node, not as separate stacked boxes.
- **Decoded factor tokens** at the bottom, with the link to the shared
  factor head and the resulting `hyper_down`, `hyper_up` tensors.

### Zoom panel 2: Factor head + reshape

Same `.zoom-panel` framing as the decoder zoom. Shows the tensor
surgery the single `Linear(H → H)` performs on the decoded factor
tokens:

- **Input**: decoded factor tokens `[B × M × H]` from the Transformer
  Decoder.
- **Shared linear**: a single `Linear(H → H)` applied independently to
  each of the `M` tokens. `H = r · (a + b)` by construction (asserted
  at `hyperalign.py:76–80`).
- **Split last dim + reshape**: first `r·a` values reshape to
  `hyper_down [B × M × a × r]`; next `r·b` values reshape to
  `hyper_up [B × M × r × b]`. Source: `_split_hyper_factors`
  (`hyperalign.py:413–419`).
- **Assign to LoRA wrappers**: each `WrappedLoRALinear` in the frozen
  base receives its per-module factors via `set_dynamic_hyper_factors`,
  giving the effective update
  `ΔW = (α/r) · (B_aux · B_hyper)(A_aux · A_hyper)`. `B_aux` is
  zero-initialised so the effective `ΔW = 0` at the start of training
  (source: `adapters/low_rank/common.py`).

This makes the connection from "decoded factor tokens" to "LoRA factors
on attention projections" concrete; in the main diagram this whole
sequence is collapsed into the single `Factor head + reshape` node
plus the injection bridge.

### Attempted (reverted 2026-05-19): make the zoom open on click

`click Dec call toggleZoom()` + `securityLevel: 'loose'` was tried so
the zoom panel would be collapsed by default and expand on clicking
the Decoder node. The combination failed to render under Mermaid 10 —
likely the loose-security sanitizer path is less tolerant of the
HTML-in-labels we already use. Reverted to always-rendered. If
revisiting, render the zoom as inline SVG (as the shortcut explainer
does) to avoid Mermaid's sanitizer entirely.

## Decisions (2026-05-19)

- **Injection mode shown: `memory_tokens` only.** `cross_attention` is
  mentioned in a caption footnote so the figure isn't double-pathed.
- **Two-pass structure is explicit.** The frozen base is drawn once, but
  its output forks into `v_base` (LoRA off) and `v_adapted` (LoRA on),
  meeting at a subtraction node that yields the adapter's contribution
  `Δ_φ = v_adapted − v_base`.
- **`S/I/P` update modes**: caption annotation, no separate inset.
- **Decoder and Factor head are each rendered at two levels of
  abstraction**: a single node in the main diagram (marked with 🔍),
  and an always-visible expanded zoom panel below. The decoder zoom
  makes the Q-per-LoRA-target correspondence explicit and clarifies
  that memory tokens combine perception-encoder and structural-encoder
  tokens; the factor-head zoom makes the split-and-reshape sequence
  and the connection to `WrappedLoRALinear.set_dynamic_hyper_factors`
  explicit.

## Related

- [[../related-work/hyperalign]] — paper-side note
- [[../tech/structural-encoder]] — to be written
- [[figure-avid]] · [[figure-shortcut-training]] · [[figure-structural-encoder]]
- Code: `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py`
- Code: `src/generative_flow_adapters/adapters/low_rank/common.py`
- Repo doc: `docs/hyperalign-architecture-replication.md`
