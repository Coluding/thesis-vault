---
type: decision
status: open
created: 2026-05-25
decided_at:
updated: 2026-06-10
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
- **(2026-06-03) Not a hedge — a parallel exploration under one unifying
  object.** User clarified: multimodal is *not* a cheap fallback triggered by
  shortcut failing. It is something to explore **simultaneously**, just with a
  *different framing* (focused on multimodality rather than shortcuts).
  *"Whatever works is what the thesis will be mostly about."* The unifying
  object underneath both framings is stated explicitly: **the thesis will be
  about output adapters of diffusion models.** So the structure is: one
  substrate (output adapters on a frozen diffusion base) → two research
  framings explored in parallel (step-size/shortcut vs. compositional
  multimodality) → whichever yields the stronger results becomes the
  headline. This supersedes the "hedge" framing in the Status header above.
  **Open positioning consequence (next grill):** this narrows the thesis
  scope vs. [[../../10_now/positioning]], which currently claims *four* adapter
  families (LoRA/hidden-state/hypernetwork/output) spanning *both* diffusion
  and flow matching. "Output adapters of diffusion models" demotes the
  four-family taxonomy comparison (current D1/D2 headline) and drops flow
  matching from the headline. Not yet written into positioning — pending
  confirmation that the narrowing is intended.
- **(2026-06-03) Final framing deliberately held open.** User: *"The final
  framing is still open since we do not know what will work. I want to keep it
  as open as possible."* So we do **not** narrow positioning.md yet, and we do
  **not** pick a headline (four-family taxonomy vs output-adapter-shortcut vs
  output-adapter-multimodal) now. The thesis stays in exploratory mode; the
  headline is chosen *after* there are real results to choose from. The only
  fixed point is the unifying object: **output adapters on a frozen diffusion
  base.** Consequence for this decision note: it stays `open` by design, not
  by neglect — it is the live record of an exploration, not a stalled call.
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

## What "works" means for the multimodal line (2026-06-03)

The story-driven frame needs a definition of "works" to recognise the result.
User's target is the conjunction of three things:

1. **The modalities are predicted properly** — the adapters emit plausible,
   well-formed proprio/depth/etc. streams (existence proof).
2. **Video prediction improves** — adding the extra modalities measurably
   helps the *primary* video prediction vs. a video-only adapter (coupling
   pays off on the main task).
3. **The modalities are demonstrably used** — we can show the model actually
   relies on the modality streams for its prediction (not ignoring them);
   the learned mask `m` / an ablation makes the contribution visible.

4. **(added 2026-06-03) Control — multimodal policies that use the
   predictions.** The multimodal world model is used downstream for control:
   policies that consume the predicted modalities. This is flavor 4 (a
   capability single-modal can't give) and is the most thesis-worthy spine
   because it does not depend on the video-improvement bet.

This is flavors 1 + 3 + 2 + 4 from the grill, combined. Note **(2)** is the
empirically riskiest — an outcome bet that may not break our way — while
**(1)**, **(3)**, and **(4)** are more within design control. How the story
degrades if **(2)** fails is an open risk. How **(3)** gets *shown* (mask
inspection vs. drop-the-modality ablation vs. attribution) is a methods
sub-question to pin before the run.

**Positioning tension (next grill, 2026-06-03).** Adding control/policies as
a success criterion collides with [[../../10_now/positioning]]'s explicit
anti-positioning: *"Not a control / RL paper. The output is a world model
usable for planning. Demonstration via planning is a sanity check, not the
contribution."* Either the anti-positioning needs rewriting (control becomes
a first-class result) or control stays a sanity-check demo. Unresolved.

## Conceptual motivation (2026-06-04) — predictive processing

The user wants this direction motivated by **predictive processing** (Andy
Clark, *Surfing Uncertainty*, OUP 2016 —
[[../../30_Knowledge/related-work/surfing-uncertainty]]). The argument that
transfers cleanly: a cognitive system is a **generative model predicting
multiple sensory modalities forward over a temporally-extended trajectory**,
and **action is part of the same predictive machinery** ("action-oriented"
PP). That is structurally our object — a frozen generative prior + output
adapters predicting coupled future modalities, used downstream for control
(success criterion 4). Per-modality precision-weighting in PP is a loose
analogue of the open `w_m` weighting (sub-decision 1) and the learned mask
`m`. **Use as intro/discussion narrative only** — it is motivation, not
evidence (hard rule 7/8); and note the standing critique (Rescorla, NDPR) that
active inference "presupposes" the expected trajectory rather than explaining
it — the likely advisor objection if PP is leaned on too hard.

**Resolved (2026-06-04):** PP is adopted as the thesis's **narrative/framing
spine** (intro + discussion), as inspiration/motivation only — not as
empirical justification. Story-arc decision recorded in
[[../../30_Knowledge/writing/narrative-predictive-processing]]. Note this is a
soft lean toward the multimodal direction as headline (PP maps strongly onto
multimodal, weakly onto shortcut), even though the headline stays formally
open.

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

## Engineering approach — parallel `multimodal/` package (2026-06-03)

User's chosen structure: build a **parallel line in the same repo** under
`src/generative_flow_adapters/multimodal/`. It **imports** the existing
building blocks (adapter modules, encoders, conditioning, base-model loading)
and adds *only* a new model + its own training script + its own configs. It
does **not** modify `AdaptedModel`, `trainer.py`, or the output-adapter
contract.

**Verified clean (2026-06-03, repo `main`):** the dependency graph supports
this without circular imports —
- `adapters/base.py` imports only `models/base/interfaces`.
- `conditioning/encoders.py` imports only `config`.
- **nothing in `adapters/`, `conditioning/`, or `models/` imports
  `training/`** — so the trainer can be forked without dragging the rest.
- `OutputAdapterResult` (single-tensor; `interface.py:11-15`) does **not** need
  changing: the compositional variant uses **one existing single-stream output
  adapter per modality**, and the multi-stream fusion (mask `m`) lives inside
  the new `MultiModalAdaptedModel`. The primitives are imported untouched.

**Cross-line fairness is NOT a requirement (clarified 2026-06-03).** Initial
grill instinct was that the parallel package risks confounding a
"multimodal vs shortcut vs plain output adapter" head-to-head. User overruled:
*"they don't need to be compared fairly. The thesis is about writing a story.
Whatever works, we will build our main story on."* So this is **not** a
benchmark/ablation thesis where the contribution is a fair cross-line
comparison. It is **story-driven exploration**: run the directions in
parallel, and whichever produces a compelling result becomes the thesis's
main narrative. The parallel package needs apples-to-apples reuse only for
*convenience* (less code), not for *scientific validity of a cross-line
table*. Practical implication: fork freely; the multimodal line can have its
own trainer/eval without that being a methodological problem.

> Residual (not yet pushed on): even a story thesis's *chosen* line will want
> *some* internal baseline in its own results chapter (e.g. compositional vs
> channel-stack floor, or vs no-adapter) to show the winning idea actually
> works — fairness can be dropped *across* lines but is still cheap insurance
> *within* the line that becomes the story. Flagged, not resolved.

## Build status (2026-06-10) — substrate + compositional variant shipped

Despite this note still being **open**, the engineering moved: the parallel
`multimodal/` package landed in commit `b09e8d5` ("cleaned configs and added
multimodal model"). What's built vs. the plan above:

**Shared substrate (items 1–4) — done.** Multi-stream output contract
(`MultiModalAdaptedModel.forward(x_t: dict, t: dict, cond) -> dict`),
per-modality independent-timestep noising + summed weighted loss
(`MultiModalTrainer`, the UWM scheme), multimodal batch preprocessor, and
`OutputModalitySpec` (kinds `video`/`vector`/`map`, per-stream `loss_weight`,
codec selection) all exist. Codecs: `IdentityCodec`, `ResizeCodec`.

**Fusion variants — 1 of 3 + substrate floor.**

| Variant (plan) | Built? | Code |
|---|---|---|
| Channel-stack (baseline floor) | ❌ not yet | — |
| Single-joint (mid baseline) | ❌ not yet | — |
| **Compositional (the contribution)** | ✅ | `LearnedMaskFusion` — softmax mask `m ∈ ℝ^{n+2}`, base-biased init |
| *(extra)* additive substrate | ✅ | `TrivialFusion` — `ε_pre + Σ contributions` |

So the **contribution is built first**, not the baselines (the plan's
recommended order was channel-stack → single-joint → compositional). The
"cheap insurance" internal baseline flagged in the residual note above is the
gap: the compositional variant has no channel-stack floor to compare against yet.

**Tested, not run.** `tests/test_multimodal_substrate.py` (7 tests, passing)
overfits both `TrivialFusion` and `LearnedMaskFusion` on the `DummyVectorField`
base — multi-stream learning, codec roundtrips, spec validation, config
partition all verified. **No DynamiCrafter / real-data run has happened**, so
none of the three "what works" criteria (modalities predicted / video improves /
modalities demonstrably used) is tested yet. Diffusion-only —
`MultiModalTrainer` raises `NotImplementedError` for flow bases.

**What this does and doesn't settle.** It confirms the substrate design is
buildable and the imports-clean approach (parallel package, untouched
`AdaptedModel`) held. It does **not** settle the go/no-go (sub-decision 5
residual): multimodal-vs-shortcut as the thesis headline is still open, and the
positioning tensions (control as success criterion; PP-as-spine soft lean)
remain. Decision stays `open`.

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

- ~~Feat ticket: scaffold the shared substrate (items 1–4) + the channel-stack
  variant.~~ **Substrate done (2026-06-10, commit `b09e8d5`)** — but the
  variant built was *compositional*, not channel-stack. See Build status above.
- **Open: build the channel-stack and single-joint baseline variants** so the
  compositional contribution has a floor to beat (the "cheap insurance" residual).
  → needs a `feat-adapter-*` ticket.
- **Open: first real-backbone run.** Take the substrate off the dummy base onto
  DynamiCrafter + a real proprio/depth modality; only then can an experiment
  note be written. → needs an `exp-adapter-*` ticket.
- Ablation plan note in `30_Knowledge/experiments/` (variant comparison
  protocol), mirroring
  [[../../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
  — write once the baseline variants exist.
- ~~Resolve sub-decision 5 (multimodal × shortcut relationship).~~ Resolved
  2026-05-25 (alternatives, not layers). The residual **go/no-go** (which line
  is the thesis headline) stays open.

## Related

- [[../../30_Knowledge/related-work/surfing-uncertainty]] — predictive-processing
  motivation (Andy Clark): generative model predicting multimodal sensory
  trajectories forward + action-oriented control.
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
