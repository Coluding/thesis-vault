---
type: decision
status: open
created: 2026-05-25
decided_at:
updated: 2026-05-25
target_date:
scope: architecture
related:
  - "[[../../30_Knowledge/related-work/unified-world-models]]"
  - "[[../../30_Knowledge/related-work/avid]]"
  - "[[../../30_Knowledge/tech/mask-mix-gate]]"
  - "[[../../30_Knowledge/tech/structural-encoder]]"
  - "[[../decided/avid-adapter-init]]"
  - "docs: docs/avid_approach.png"
  - "docs: docs/composite (2).png"
  - "docs: docs/image (2).png"
---

# Decision: How to broaden the framework to multimodal adapters

## Status

**Open — captured 2026-05-25.** This is the "optional extension" named in
the vault CLAUDE.md (`(x^video_{t+d}, x^prop_{t+d}) = f(x_t, t, a_t, d)`),
elevated from a footnote to an actual design because the user wants a
broadening hedge **in case shortcut modeling (D3/D4) does not pan out**.
The three thesis-draft figures (`docs/avid_approach.png`,
`docs/image (2).png`, `docs/composite (2).png`) are the source vision:
generate additional modalities (proprioception, depth, tactile, lidar, …)
alongside video, ideally via a **compositional** one-adapter-per-modality
scheme with a learned mask.

### Resolutions (live discussion, 2026-05-25)

- **All three variants get built — this is a comparison, not a pick-one.**
  The user: *"the compositional adapter is the goal but all of them need to
  be implemented at some point because we want to compare them."* So the
  channel-stack and single-joint variants are **baselines**; the
  compositional adapter is the **contribution**.
- **Therefore the design is "one substrate, swappable fusion."** For the
  comparison to be clean, the three variants must differ *only* in the
  fusion / adapter-orchestration block. Everything else (data, noising,
  per-modality loss, eval metrics) is held constant, or the comparison
  confounds the variant with the plumbing.
- **Timestep scheme = independent per-modality timesteps + shared
  denoising objective.** Default lean was *shared `t`*; flipped after
  reading UWM ([[../../30_Knowledge/related-work/unified-world-models]]),
  which the user flagged as the inspiration. Each output modality gets its
  own timestep sampled `~U(0,T)`; one weighted denoising loss sums over
  modalities. This buys free conditional/marginal inference (policy,
  forward/inverse dynamics, video prediction) and co-training on
  incomplete-modality data — and it *is* the "denoising optional" (purple)
  boxes in `docs/composite (2).png`.
- **Multimodal and shortcut are alternatives, not layers.** User
  (2026-05-25): *"we would ignore the shortcut for now if we decide for this
  more general objective."* So if the multimodal objective is chosen, D3/D4
  (shortcut) is **shelved**, not composed on top. This **resolves
  sub-decision 5** (below): the shared substrate does **not** need to carry
  shortcut consistency terms through the per-modality loss aggregation, and
  `MultiModalAdaptedModel` does not need to interleave a step-size `d` axis.
  Decision is still conditional — multimodal-vs-shortcut as the thesis's
  primary objective is not yet finalised — but the *relationship* is settled:
  pick one focus, not both. Positioning implication (shortcut moves from
  primary deliverable to shelved) is flagged but not yet written into
  [[../../10_now/positioning]] pending the go/no-go.
- **The input-side multimodality already exists.** The
  `StructuredConditionEncoder` ([[../../30_Knowledge/tech/structural-encoder]],
  `conditioning/encoders.py:109-151`) is the per-modality-branch + fuse
  pattern, on the conditioning input. The compositional output adapter is
  its **output-side dual** — same skeleton, but the fusion is *inverted*:
  keep streams separate (one prediction per modality) and replace the
  MLP-fuser with the learned mask. Reuse the `ConditionSpec` pattern for an
  output-side modality spec.

## Context — the design progression in the three figures

The three figures are stages of one idea, not alternatives:

1. **`docs/avid_approach.png`** — vanilla AVID: frozen base + one action
   adapter, combined by a learned gate `m`:
   `ε_t = m⊙ε_adj + (1−m)⊙ε_pre`. **This is essentially already
   implemented** (see below).
2. **`docs/image (2).png`** — one **joint** adapter ingests action + all
   modality conditioning, emits a video adjustment + per-modality
   predictions, scalar mask mix.
3. **`docs/composite (2).png`** — one adapter **per modality**, each
   emitting *both* a video noise-adjustment `ε_adj,i` *and* its own modality
   prediction `ε_i`, all fused by a learned vector mask `m ∈ ℝ^{n+2}` over
   {base, action, modality₁…ₙ}:
   `ε_t = m_{n+2}⊙ε_pre + m_{n+1}⊙ε_adj + Σ_i m_i⊙ε_adj,i`.
   Per-modality denoising optional (purple) = UWM's `t=T` marginalisation.

## What the repo already implements (start point ≈ figure 1)

The single-modality AVID path is effectively done:

- **The gate/mask** is implemented: `adapted_model.py:88-94`
  (`mask_mix`/`avid_mask_mix`) does `base·gate + adapter·(1−gate)`, and the
  DynamiCrafter UNet returns `(prediction, gate)` when `output_mask=True`
  (`adapters/output/dynamicrafter.py:153-158`). See
  [[../../30_Knowledge/tech/mask-mix-gate]].
- **Multimodal *input* conditioning** exists (`StructuredConditionEncoder`,
  `MultimodalConditionEncoder`, `encoders.py:68-151`).
- **The data already carries every figure-3 modality.** `data/schema.py:55-65`
  defines `proprio (T,7)`, `depth (T,H,W)`, `tactile (T,2,Hₜ,Wₜ)`,
  `force_torque`, `gripper`, `ee_xyz`, … as MetaWorld optional keys. The
  data side anticipates this work.

## The three bottlenecks blocking multimodal *output*

Everything downstream of the base model assumes **one output tensor**:

1. **Adapter contract is single-stream.** `OutputAdapterResult` is
   `adapter_output: Tensor` + one `gate` (`adapters/output/interface.py:11-15`);
   `AdaptedModel.forward → Tensor`, `_compose → Tensor`
   (`models/adapted_model.py:55,74`).
2. **Loss is single-target.** `trainer.py:108-110` hard-requires a single
   `batch["target"]` tensor; `q_sample` runs on it; `loss = loss_fn(pred,
   target)` (`trainer.py:139,176`). Shortcut targets are single-tensor too.
3. **No `m ∈ ℝ^{n+2}` fusion.** The current gate mixes base against *one*
   adapter only; nothing emits a separate modality prediction alongside the
   video adjustment.

## The design — one substrate, three swappable fusion strategies

### Shared substrate (built once, held constant across all variants)

1. **Multi-stream output contract** — model returns `dict[str, Tensor]` =
   `{"video": …, "proprio": …, "depth": …}`. Generalises the single-tensor
   `OutputAdapterResult`.
2. **Per-modality + per-timestep noising and loss** — each modality gets an
   independent timestep `t_m ~ U(0,T)` (UWM scheme); the trainer noises each
   stream at its own `t_m`, computes a per-modality diffusion/flow loss, and
   sums them with weights `w_m`. Today's single-tensor loss becomes the
   degenerate 1-modality case.
3. **Multimodal batch/data** — a preprocessor that normalises / latent-encodes
   each schema modality into `batch["targets"][name]`. ~90% there already
   (`schema.py`), needs the encode/normalise step.
4. **Output modality spec** — the output-side mirror of `ConditionSpec`:
   name, channels/shape, latent-vs-raw, loss weight `w_m`, `denoised` flag
   (the purple-optional), adapter assignment.

### Three fusion strategies behind one interface

| Variant | Source fig | Adapters | Fusion | Role |
|---|---|---|---|---|
| **Channel-stack** | — (PAD-like) | 1, extra output channels | one gate, masked loss | cheapest baseline; validates substrate items 1–4 end-to-end |
| **Single-joint** | `image (2).png` | 1, all-modality cond + per-modality heads | scalar / per-stream mask | mid baseline; reuses `DynamicCrafterOutputAdapter` |
| **Compositional** | `composite (2).png` | N (one/modality) + action | learned `m ∈ ℝ^{n+2}` over video adjustments | **the target / contribution** |

Recommended build order = table top-to-bottom. Channel-stack first proves
the substrate; each later variant is then just a new fusion module on the
same substrate, so eval ([[../../30_Knowledge/related-work/avid]] →
`src/external_deps/avid_utils/metrics.py`) stays constant.

> **UWM prior:** the channel-stack baseline is PAD-shaped, and UWM shows PAD
> is the weakest of its baselines precisely because of the shared-timestep /
> channel-concat conditioning. Expect channel-stack to underperform — which
> is the *point* of having it as the floor of the comparison.

## Open sub-decisions

1. **Per-modality loss weighting `w_m`.** Proprio / depth / tactile live at
   wildly different scales and dims. Fixed scalars (UWM-style), uncertainty
   weighting (Kendall), or gradient normalisation? Load-bearing for the
   compositional variant where n is large. _open._
2. **Latent vs raw per modality.** Video uses the frozen VAE. Depth could
   reuse it (treat as image); proprio/force-torque are low-dim vectors that
   want a small per-modality (V)AE or just raw diffusion. The `latent-vs-raw`
   field in the modality spec must resolve per modality. _open._
3. **Does timestep-as-mask survive a frozen base?** UWM marginalises by
   setting `t=T`; that relies on the joint model having seen all timestep
   combos. Our video base is frozen and only knows the video schedule, so
   "marginalise video via `t_video=T`" may misbehave. Modality streams (new,
   trainable adapters) are fine; the *video* stream is the question. _needs
   experiment._
4. **Where does cross-modal coupling live?** Joint self-attention + register
   tokens (UWM) is not available in the adapter setting. Candidates: (a) the
   shared conditioning embedding only; (b) the mask `m` coupling each
   modality's video-adjustment back into the video latent; (c) explicit
   cross-attention between modality adapters. The compositional figure
   implies (b). _open._
5. **How does this compose with shortcut (D3/D4)?** ~~open~~ **Resolved
   2026-05-25: they don't compose — they're alternatives.** If the
   multimodal objective is chosen, shortcut is shelved for now (user). So the
   substrate ignores shortcut entirely: no step-size `d` axis in
   `MultiModalAdaptedModel`, no consistency terms in the per-modality loss
   aggregation. The only residual open part is the **go/no-go** itself
   (multimodal vs shortcut as the thesis's *primary* objective) — a
   positioning call, not a substrate-design call.

## Consequences (if we proceed)

- New model class (`MultiModalAdaptedModel`) sibling to `AdaptedModel`;
  multi-stream output contract; per-modality+timestep trainer path; a
  multimodal batch preprocessor; an output modality spec in `config.py`.
- New extension points registered the same way as the existing taxonomy
  (new fusion strategy = new entry, like a new adapter family).
- New ablation surface: variant (channel-stack / joint / compositional) ×
  modality set × loss weighting × shared-vs-independent timestep (we can run
  shared-`t` as an ablation to reproduce the UWM-vs-PAD finding on our base).
- **Shortcut is shelved, not extended.** `MultiModalAdaptedModel` is a
  clean separate class that never touches the shortcut/`d` path; the existing
  single-modality shortcut code stays **byte-unchanged** (and dormant) rather
  than being interleaved with the multimodal loss. If the go/no-go later
  flips back to shortcut, nothing was lost.

## Follow-ups (derive on decide)

- Feat ticket (gated on this decision + sub-decision 1–2): scaffold the
  shared substrate (items 1–4) + the channel-stack variant.
- Feat tickets (later): single-joint variant; compositional variant.
- Ablation plan note in `30_Knowledge/experiments/` once the substrate
  exists: the variant comparison protocol, mirroring
  [[../../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]].
- Resolve sub-decision 5 (multimodal × shortcut relationship) with the user
  — it was unanswered when this note was written.

## Related

- [[../../30_Knowledge/related-work/unified-world-models]] — the timestep
  scheme this design adopts.
- [[../../30_Knowledge/related-work/avid]] — figure 1 (already implemented).
- [[../../30_Knowledge/tech/mask-mix-gate]] — the single-stream gate the
  multi-stream mask `m` generalises.
- [[../../30_Knowledge/tech/structural-encoder]] — the input-side dual of
  the compositional output adapter.
- Code: `src/generative_flow_adapters/models/adapted_model.py:55,74-96`
- Code: `src/generative_flow_adapters/adapters/output/interface.py:11-15`
- Code: `src/generative_flow_adapters/adapters/output/dynamicrafter.py:153-158`
- Code: `src/generative_flow_adapters/training/trainer.py:108-110,139,176`
- Code: `src/generative_flow_adapters/conditioning/encoders.py:68-151`
- Code: `src/generative_flow_adapters/data/schema.py:55-65`
- Figures: `docs/avid_approach.png`, `docs/image (2).png`, `docs/composite (2).png`
