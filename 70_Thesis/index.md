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

> **⚠ CHANGED 2026-08-03 — LaTeX is now the source of truth.** The faculty
> template (UvA MSc AI) arrived and the thesis is written in
> `70_Thesis/latex/`. The Markdown in `draft/` is **historical**; the real
> prose it already contains must be **ported, not rewritten** (see below).

- **Write into `70_Thesis/latex/chapters/*.tex`.** Skeleton, notation
  macros and provenance macros are in place. Conventions:
  [[../30_Knowledge/writing/thesis-formal-rules]].
- **⚠ 40-page upper limit** (appendices additional). The chapter structure
  below is the **proposed** 6-chapter fit — see
  [[../30_Knowledge/writing/rubric/07-thesis-organization]].
- **Rhetoric and the claim ladder:**
  [[../30_Knowledge/writing/thesis-style-guide]]. **What is graded:**
  [[../30_Knowledge/writing/rubric/_index]].
- Use `/thesis-write {section}` to draft or extend a chapter from recent
  `60_Updates/` + the linked sources.
- **Deliverable hygiene (CLAUDE.md Part 12):** D1=framework, D2=action world
  models, D3=shortcut adapters, D4=combined.
- **No unsourced numbers** — every metric carries `\prov{run}{ckpt}{commit}`
  (hard rule 8).

## Chapters → status → sources

**6-chapter structure for the 40-page limit — signed off 2026-08-03.**

| # | Chapter | File | Budget | Status | Deliverable | Source notes |
|---|---|---|---|---|---|---|
| — | Abstract | `main.tex` | — | stub | — | write last; must carry the control limit |
| 1 | Introduction | `chapters/10-introduction` | 4–5 pp | stub | — | [[../10_now/positioning]], [[../30_Knowledge/writing/thesis-storyline]] |
| 2 | Related work | `chapters/20-related-work` | 5 pp | stub | theory/baseline | `30_Knowledge/related-work/*` — **blocked on `lora`, `controlnet`, FiLM/adaLN notes** |
| 3 | Method | `chapters/30-method` | 10 pp | stub | D1 (+D3 theory) | [[../30_Knowledge/tech/structural-encoder]], [[../30_Knowledge/tech/shortcut-training-modes]], [[../30_Knowledge/tech/frame-stride-conditioning]], [[../30_Knowledge/theory/shortcut-v-averaging-bias]], [[../10_now/architecture]] · **port prose from [[draft/30-method]]** |
| 4 | Experiments | `chapters/40-experiments` | 7 pp | stub | D2/D3 | [[../30_Knowledge/writing/ablation-axes]] (**stale — rewrite first**), [[../30_Knowledge/writing/rubric/03-experimental-evaluation]] |
| 5 | Results | `chapters/50-results` | 10 pp | stub | D1/D2/D3 | [[../30_Knowledge/experiments/_index]] (sourced runs only) |
| 6 | Discussion & conclusion | `chapters/60-discussion` | 4 pp | stub | all | [[../30_Knowledge/writing/thesis-storyline]] §8–9, `50_Decisions/*` |
| A | Appendix | `chapters/90-appendix` | +N | stub | — | run inventory, probe definitions |

**Superseded:** the 8-chapter outline with a separate adapters chapter
(`draft/25-adapters.md`, never created) and a separate Discussion and
Conclusion. Both merged to fit 40 pages.

## Cross-links

- **Outline + per-section status:** [[outline]]
- **Progress narrative (what's new to write about):** [[../60_Updates/index]]
- **Figures:** `30_Knowledge/writing/figure-*.md`
- **Positioning / contribution framing:** [[../10_now/positioning]]
- **Current architecture (for the Method chapter):** [[../10_now/architecture]]
