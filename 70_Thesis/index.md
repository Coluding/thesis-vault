---
last_updated: 2026-05-28
status: living
---

# Thesis Writing Hub

> The **assembly layer** for the thesis. This directory holds the rough
> Markdown draft (`draft/`) and links every vault note that feeds it. It
> does **not** duplicate content — atomic writing material stays in
> `30_Knowledge/writing/`, `related-work/`, `theory/`, `experiments/`, and
> decided notes in `50_Decisions/decided/`. This index is the map from
> chapters → their sources.

**Thesis:** *Adapting Pretrained Generative Models into Action-Conditioned
World Models via Plug-and-Play Adapters.*

**Composition rule:** `f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)`

## How to write into the draft

- The draft is **deliberately rough Markdown** — extend it freely, no
  compile step. Convert to LaTeX/Typst/PDF (pandoc/typst) only when mature.
- Use `/thesis-write {section}` (e.g. `/thesis-write 40-experiments`) to
  have an agent draft or extend a section from recent `60_Updates/` +
  the linked sources below.
- **Deliverable hygiene (CLAUDE.md Part 12):** D1=framework, D2=action world
  models, D3=shortcut adapters, D4=combined. Do not mix deliverable evidence
  across chapters.
- **No unsourced numbers** — every metric cites a run (hard rule 8).

## Chapters → status → sources

| # | Chapter | Draft file | Status | Primary deliverable | Source notes |
|---|---|---|---|---|---|
| 0 | Abstract | [[draft/00-abstract]] | stub | — | (write last) |
| 1 | Introduction | [[draft/10-introduction]] | stub | — | [[../10_now/positioning]] |
| 2 | Related Work | [[draft/20-related-work]] | stub | theory/baseline | `30_Knowledge/related-work/*`, [[../30_Knowledge/related-work/hyperalign]] |
| 3 | Method | [[draft/30-method]] | drafting | D1 | [[../30_Knowledge/tech/structural-encoder]], [[../30_Knowledge/tech/shortcut-training-modes]], [[../30_Knowledge/tech/frame-stride-conditioning]], [[../50_Decisions/decided/shortcut-anchor-schedule]], [[../50_Decisions/decided/per-sample-frame-stride-sampling]], [[../10_now/architecture]] |
| 4 | Experiments | [[draft/40-experiments]] | drafting | D2/D3 | `30_Knowledge/experiments/*`, [[../50_Decisions/decided/param-matched-adapter-comparison-definition]], [[../50_Decisions/decided/per-sample-frame-stride-sampling]] |
| 5 | Results | [[draft/50-results]] | stub | D2/D3/D4 | `30_Knowledge/experiments/*` (sourced runs only) |
| 6 | Discussion | [[draft/60-discussion]] | stub | all | `50_Decisions/decided/*`, open questions |
| 7 | Conclusion | [[draft/70-conclusion]] | stub | all | — |

## Cross-links

- **Outline + per-section status:** [[outline]]
- **Progress narrative (what's new to write about):** [[../60_Updates/index]]
- **Figures:** `30_Knowledge/writing/figure-*.md`
- **Positioning / contribution framing:** [[../10_now/positioning]]
- **Current architecture (for the Method chapter):** [[../10_now/architecture]]
