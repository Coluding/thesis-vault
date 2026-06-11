---
date: 2026-06-10
category: progress
deliverable: exploratory
meeting:
sources:
  - "[[../../50_Decisions/open/multimodal-adapter-broadening]]"
  - "[[../../10_now/architecture]]"
  - "[[../../10_now/product-state]]"
  - "repo: commit b09e8d5 — src/generative_flow_adapters/multimodal/"
  - "repo: tests/test_multimodal_substrate.py"
---

# Multimodal multi-stream output substrate is built

## What

The multimodal world-model line moved from design to code. A new parallel
package `src/generative_flow_adapters/multimodal/` (commit `b09e8d5`) implements
the multi-stream substrate: `MultiModalAdaptedModel` predicts a *dict* of
coupled output streams; the frozen base carries the video stream and trainable
per-modality heads predict the rest from scratch. The **compositional**
contribution (`LearnedMaskFusion` — a learned softmax mask `m ∈ ℝ^{n+2}` over
{base, action, modality streams}) is built, alongside an additive
`TrivialFusion` substrate baseline. Training uses independent per-modality
timesteps + a summed weighted denoising loss (the UWM scheme).

## Why it matters

This is the "broadening" line that runs **in parallel** to shortcut (D3/D4) —
whichever produces the stronger story becomes the thesis headline. We now have
a working substrate to put that bet to the test, and the contribution variant
(not just a baseline) exists. The build also confirmed the clean-imports
approach held: `AdaptedModel`, the trainer, and the output-adapter contract were
left untouched.

## Evidence / sources

- `tests/test_multimodal_substrate.py` — **7 tests passing** on the lightweight
  `DummyVectorField` base: multi-stream contract, codec roundtrips, spec
  validation, config partition, per-stream noising, and **overfit of both
  `TrivialFusion` and `LearnedMaskFusion`** (both fit; the compositional mask
  receives gradient and stays a normalised softmax).
- These are unit/overfit checks on a toy base — **not** a real-data result. No
  metric is reported because no real-backbone run has happened.

## Next

- **Not yet run on a real backbone** (DynamiCrafter + a real proprio/depth
  modality) — that gates any experiment note or "does it work" claim.
- **Diffusion-only**; flow-matching multimodal is deferred (`NotImplementedError`).
- **Baseline variants missing:** channel-stack and single-joint aren't built, so
  the compositional variant has no internal floor to beat yet.
- The **go/no-go** — multimodal vs. shortcut as the thesis headline — stays open
  in [[../../50_Decisions/open/multimodal-adapter-broadening]].
