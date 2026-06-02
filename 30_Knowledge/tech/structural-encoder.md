---
type: tech-note
status: living
last_updated: 2026-05-19
deliverable: D1, D2
---

# Structural (Structured) Condition Encoder

> The single conditioning encoder consumed by every adapter family in
> the framework. Produces `cond["embedding"]`, the conditioning vector
> that lands at well-defined injection points inside both the
> hypernetwork adapter and the output adapter. The thesis names it the
> "structural encoder"; the code spells it
> `StructuredConditionEncoder` (in
> `src/generative_flow_adapters/conditioning/encoders.py`).

## Why this is the right object to centre the conditioning story on

The thesis composition rule `f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)`
implicitly assumes that the conditioning `c = (a_t, d, …)` enters the
adapter through *one* canonical path. The structural encoder is that
path: it absorbs heterogeneous conditioning tensors (action vectors,
step-size, optionally goal / context) and produces a single embedding
that downstream adapters consume the same way regardless of family.

Concretely this is what makes the framework (D1) more than a bag of
adapter variants — every adapter family is a different way of
*spending* the same conditioning vector.

## Architecture

Per `StructuredConditionEncoder.__init__` (`conditioning/encoders.py:109`):

- **Inputs.** A `list[ConditionSpec]` from
  `ConditioningConfig.conditions`. Each spec has fields
  `(key, input_dim, encoder, hidden_dim)` (`config.py:9`).
- **Per-key branch.** For every spec, an independent MLP:
  `Linear(input_dim, hidden) → SiLU → Linear(hidden, output_dim)`.
  Stored as `self.condition_encoders[spec.key]`.
- **Concat.** Outputs of all branches concatenated along the last
  dimension → shape `[B, N·output_dim]` where `N` is the number of
  conditions.
- **Fuser.** `Linear(N·output_dim, hidden) → SiLU → Linear(hidden, output_dim)`.
  Reduces back to a single `[B, output_dim]` vector.
- **`null_embedding`** parameter, shape `[output_dim]`. Used by
  `_apply_condition_dropout` to replace `c` with `null_embedding` when
  classifier-free guidance dropout fires.

Only `fuse_mode = "concat_mlp"` is currently supported; the class
raises if any other mode is requested (`encoders.py:119`). Only
`encoder = "mlp"` per branch is currently supported
(`encoders.py:123`).

## Where the output lands

The fused vector is exposed as `cond["embedding"]`. Two consumers:

### 1. HyperAlign hypernetwork

Two injection modes (`condition_injection_mode`):

- **`memory_tokens`** (default, paper-faithful). The projected
  embedding is concatenated to the memory tokens fed into the
  Transformer Decoder.
  Source: `adapters/hypernetworks/hyperalign.py:440–447`
  (`_build_condition_tokens` line 218–236, the type-embedding line 235
  is the learnable categorical embed that lets cross-attn distinguish
  memory tokens from condition tokens).
- **`cross_attention`.** Zero queries cross-attend to projected
  condition tokens *before* the Transformer Decoder.
  Source: `_inject_condition_via_cross_attention` line 238–249.

### 2. DynamiCrafter output adapter (AVID-style)

The embedding enters the adapter's time-embedding path:

- `adapter_condition_proj` is a `Linear → SiLU → Linear` projection
  defined inside the adapter's `UNetModel`
  (`backbones/dynamicrafter/modules/networks/openaimodel3d.py:460–467`).
- The projected `cond_emb` is **added to** the time embedding before
  the residual blocks consume it
  (`openaimodel3d.py:769–776`,
  `emb = time_emb + cond_emb`).

### Shared helper

The runtime payload (in particular the optional `step_level` fusion
into the embedding) goes through
`conditioning/utils/dynamicrafter_conditioning.py`
(`prepare_dynamicrafter_condition`). HyperAlign and the DynamiCrafter
output adapter both call this helper to keep their handling of
`cond["embedding"]` consistent.

## How conditions are dropped (CFG)

`_apply_condition_dropout` (`encoders.py:154–165`) selects between the
encoded vector and `null_embedding` per sample, gated by a `drop_mask`
the trainer constructs from `ConditioningConfig.drop_condition_prob`.
The drop happens at the *output* of the encoder, so downstream
consumers see either the real embedding or the learned null embedding
— same shape either way.

## What this note deliberately omits

- Other encoder classes in the same file (`IdentityConditionEncoder`,
  `MLPConditionEncoder`, `MultimodalConditionEncoder`) — they exist
  for backwards compatibility with earlier configs but the thesis
  uses the structured encoder.
- The `act` / `fs` paths into the time embedding inside the U-Net
  base — these are existing DynamiCrafter conditioning, not part of
  the thesis framework.

## Open questions

- Should the framework eventually support a second `fuse_mode`
  (e.g. a small transformer fuser) for richer multimodal conditions?
  Currently only `concat_mlp` exists; this is fine for action + step
  size but might be limiting if we add high-rate goal modalities. —
  not blocking D2 / D3.

## Related

- [[../related-work/hyperalign]] · [[../related-work/avid]] — consumers of `cond["embedding"]`
- [[../writing/figure-structural-encoder]] — figure
- [[../writing/figure-hyperalign]] · [[../writing/figure-avid]] — show the consumer-side injection points
- Code: `src/generative_flow_adapters/conditioning/encoders.py:109` — class
- Code: `src/generative_flow_adapters/conditioning/utils/dynamicrafter_conditioning.py` — runtime helper
- Code: `src/generative_flow_adapters/config.py:9` — `ConditionSpec`
- Code (consumer 1): `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py:440`
- Code (consumer 2): `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:769`
