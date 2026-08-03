---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: writing
category: thesis
current_band: "not yet determined"
target_band: "8-9"
sources:
  - "[[_index]]"
  - "[[../thesis-style-guide]]"
  - "[[../thesis-formal-rules]]"
---

# Rubric 8 — Writing

## The rows

| | |
|---|---|
| 10 | Perfectly written, no errors. Method and experiments flawlessly written with great balance and visualizations. Analyses provide new insights to the thesis **and its larger subfield** |
| **9** | **Well written, practically no errors. Clear explanation of method. Detailed and thought-provoking experiments and insights. Clear on first sight, visually pleasing, with novel analyses that further enhance the thesis** |
| **8** | **Well written, few mistakes. Clear explanation of method and clear structuring of experiments with great visualizations. Interesting lessons learned** |
| 7 | Clearly written and visualized, minor issues. Standard analyses |
| 6 | Decently written but with errors and typos. Method/experiment parts not always in logical order. **Insights minimal** |
| 1–5 | Poorly written, details lacking, poor presentation |

## What it actually asks

Read the rows carefully — **this item is not mainly about prose quality.**
Every row from 7 up mentions *visualizations*, and the 8→9→10 ladder is
driven by **insight and analysis**, not grammar:

- 7 → 8: "appropriate visualizations + standard analyses" → **"great
  visualizations" + "interesting lessons learned"**
- 8 → 9: → **"novel analyses that further enhance the thesis"**, "clear on
  first sight"
- 9 → 10: → insights to "**its larger subfield**"

So *Writing* double-counts the same thing [[05-reflection]] and
[[01-originality]] reward: the transferable lesson, plus figures that carry
it. Prose correctness is table stakes; figures and insight are the score.

## Where we stand

Nothing is drafted at quality yet (541 lines of rough Markdown across 8
files, mostly stubs), so this item is **fully controllable** — and it is
governed by two documents written 2026-08-03:

- [[../thesis-style-guide]] — voice, the claim ladder, paragraph
  architecture, banned phrasings, negative-result rhetoric
- [[../thesis-formal-rules]] — LaTeX mechanics, notation, provenance
  receipts, figure/table conventions *(pending)*

## Figures — the item's real currency

The figure material here is unusually strong and almost entirely unbuilt.
For a thesis whose contribution is **diagnostic**, the probe figures
*carry the argument* — which means the 9-row's "clear on first sight"
applies literally: **they must be legible without the surrounding text.**

Priority figures, in order of argumentative weight:

| Figure | Claim it carries | Status |
|---|---|---|
| **Propagation trace** — action-driven share vs depth, with the residual add marked | *where* the signal dies | data exists (23-depth trace) |
| **The pedestal** — embedding magnitude and varying-fraction over training | the DC failure is *learned*, not initialised | data exists |
| **Structure triad vs chance** — steering / temporal / spatial, per cell | sensitivity ≠ control | data exists |
| **Blindness contrast** — perceptual metrics improving while probes sit at chance | the headline reflection | data exists |
| **Pathway A/B** — per-frame vs pooled at matched contribution and mask, over steps | the causal pathway result | data exists |
| **Curvature** — target error vs step size, averaged vs endpoint | the proven node | numerics exist |
| `[[FIG:hvxlbfjx-eval_step_grid]]` | qualitative base-parity | **export pending** |

Existing figure specs live in `30_Knowledge/writing/figure-*.md`.

## Optimisation queue

- [ ] **Q10 — Export the priority figures above.** Each must have: axes
      labelled with units, the chance level or null drawn as a reference
      line, n stated, and a claim-first caption.
- [ ] **Caption discipline.** Every caption: *claim* → *what is plotted* →
      *provenance* (wandb id + ckpt + commit). A caption that only says
      what is plotted wastes the highest-attention real estate in the
      document.
- [ ] **Single-seed honesty in every figure.** Where there is one seed, the
      caption says "single seed" — the rubric asks for uncertainty
      treatment and a silent single seed reads as concealment.
- [ ] **Sample grids need fixed seed + identical prompt/action across
      arms**, stated in the caption, or they are decorative.
- [ ] **Write the method chapter first** (outline order): the 8-row asks
      specifically for "clear explanation of method", and it is the section
      least dependent on in-flight results.
- [ ] **Run the [[../thesis-style-guide]] §8 pre-commit checklist** before
      marking any section `draft-complete`.
- [ ] **Proofing pass at the end** — the 9-row says "practically no
      errors". British English throughout; verbatim identifiers keep their
      source spelling.

## Where it lands in the thesis

Everywhere. The artifacts are the style guide, the formal rules, and the
figure set.
