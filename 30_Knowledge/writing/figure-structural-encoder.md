---
type: figure-design
status: living
last_updated: 2026-05-19
target: advisor-presentation
source: figures/structural-encoder.html
deliverable: D1, D2
---

# Figure — Structural (Structured) condition encoder

> One-slide figure showing the structural encoder pipeline and **where
> its output `c = cond["embedding"]` is injected into both** the
> hypernetwork adapter and the output adapter. This is the unifying
> conditioning-injection figure — it ties [[figure-hyperalign]] and
> [[figure-avid]] together at the conditioning interface.

## What this figure asserts

1. **One encoder, two consumers.** The same fused vector
   `c = cond["embedding"]` is the conditioning input to **both** the
   HyperAlign hypernetwork and the DynamiCrafter output adapter. The
   figure makes this single-path structure visually obvious.
2. **Inputs are a structured set of condition tensors.** Each item in
   `ConditioningConfig.conditions` is a `ConditionSpec(key, input_dim, …)`;
   examples shown in the figure: `action` (the action vector at time
   `t`), `step_size` (the shortcut step-size `d`), and a placeholder
   `…` to indicate that other modalities (goal, returns-to-go, …) plug
   in the same way.
3. **Per-key MLP branches.** For each spec, an independent
   `Linear → SiLU → Linear` produces a per-modality embedding.
4. **Concat + fuser MLP.** Branches are concatenated, then a fuser
   `Linear → SiLU → Linear` reduces to the single `output_dim`-wide
   embedding.
5. **CFG dropout via `null_embedding`.** A learnable null embedding
   replaces `c` per sample when classifier-free guidance dropout fires.
6. **Injection point in HyperAlign (consumer 1):** `c` is projected
   and concatenated to the perception-encoder memory tokens fed into
   the Transformer Decoder (`memory_tokens` mode).
7. **Injection point in AVID adapter (consumer 2):** `c` is projected
   by `adapter_condition_proj` (an `Linear → SiLU → Linear` inside the
   adapter's `UNetModel`) and **added to the time embedding** before
   the U-Net residual blocks. This is the precise injection point.

All claims are anchored in [[../tech/structural-encoder]] and the cited
code there.

## What this figure deliberately omits

- The `IdentityConditionEncoder`, `MLPConditionEncoder`, and
  `MultimodalConditionEncoder` classes in `conditioning/encoders.py` —
  they exist for backwards compatibility but the thesis uses
  `StructuredConditionEncoder` only.
- The `act` (raw action) and `fs` (frame-stride) paths into the
  DynamiCrafter time embedding — these are existing base-model
  conditioning, not the thesis's contribution.
- The `cond["context"]` cross-attention path (CLIP-style image / text
  context) — this is the base model's own context input and is
  orthogonal to the structural encoder's output.
- The `cross_attention` HyperAlign injection mode variant — shown as a
  caption note only; the figure draws the default `memory_tokens`
  mode.
- The shape-rank handling for video (rank-2 vs rank-3 condition
  embeddings) — implementation detail, not architectural.

## Layout intent

Three-column flow, left to right:

- **Left column:** structured inputs (action, step_size, …) stacked
  vertically. Each input feeds its own per-key MLP branch.
- **Middle column:** the encoder pipeline — branches concatenate, the
  fuser MLP reduces to `c`. The `null_embedding` parameter sits
  beside the fuser output with a CFG-dropout switch annotation.
- **Right column:** the two consumer injection points, stacked
  vertically. Each is rendered as a compact node containing the *exact*
  module where `c` lands (in HyperAlign: "concatenated to memory
  tokens"; in AVID: "MLP_φ → +time_embed").

## Decisions (2026-05-19)

- **Both consumers shown on the same figure** — that's the whole
  point of the figure: one encoder, two injection points.
- **HyperAlign injection mode shown**: `memory_tokens` (matches the
  default and what `figure-hyperalign` shows).
- **Inputs shown**: `action`, `step_size`, plus a generic placeholder
  `…` so the figure doesn't have to be redrawn when a new modality
  (goal, returns-to-go) is added.

## Related

- [[../tech/structural-encoder]] — full tech note with file:line anchors
- [[figure-hyperalign]] · [[figure-avid]] — consumer-side figures
- [[figure-shortcut-training]] — where the step-size input matters
- Code: `src/generative_flow_adapters/conditioning/encoders.py:109`
- Code: `src/generative_flow_adapters/config.py:9` (`ConditionSpec`)
