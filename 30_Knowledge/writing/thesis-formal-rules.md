---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-style-guide]]"
  - "[[thesis-grading-rubric]]"
  - "[[rubric/07-thesis-organization]]"
  - "[[../../70_Thesis/outline]]"
---

# Thesis formal rules — LaTeX, notation, and sourcing mechanics

> **Scope.** *The mechanics.* How the prose argues is
> [[thesis-style-guide]]; what the grader rewards is
> [[thesis-grading-rubric]]. This doc is the checkable layer: if a rule
> here is broken, a grep or a compile can find it.

## 1. The template and its hard constraints

Source: the UvA MSc AI faculty template (`titlepageai.tex`,
`bibentries.bib`, `uvalogo_regular_p_nl.eps`), supplied 2026-08-03 and
kept verbatim at `70_Thesis/latex/TEMPLATE-original.tex`.

| Constraint | Value | Status |
|---|---|---|
| Document class | `report`, `12pt` | **fixed by template** |
| Page geometry | A4, `margin=2cm` | **fixed by template** |
| **Page limit** | **40 pages**, upper limit | **hard** |
| Appendices | *additional* optional pages, outside the 40 | allowed |
| Citation style | free choice; template ships `plain` | we use `natbib`+`plainnat` |
| Language | English (`babel`), British spelling in prose | our choice |
| Front matter | title page → TOC (roman) → abstract → body (arabic) | template order |

**⚠ Never gain pages by changing geometry, font size, or line spacing.**
The limit is defined against the template's settings; margin-gaming is
detectable and reads badly. Pages come from the appendix, from cutting, or
from tighter figures — never from the preamble.

`_needs verification_`: whether the 40 pages count the roman-numbered front
matter (TOC + abstract) or only the arabic body. **Assume body-only, plan
for the stricter reading**, and confirm with the supervisor.

Title-page fields still unknown, all marked `\nv{}` in `main.tex`: student
number, number of credits, research period, supervisor, examiner, second
reader.

## 2. Source layout

```
70_Thesis/latex/
  main.tex                 % document skeleton, title page, \include list
  preamble.tex             % packages — template's first, ours after
  macros.tex               % notation + provenance + draft markers
  refs.bib                 % bibliography (from the template's bibentries.bib)
  chapters/
    10-introduction.tex  20-related-work.tex  30-method.tex
    40-experiments.tex   50-results.tex       60-discussion.tex
    90-appendix.tex
  figures/                 % PDF/EPS only; uvalogo lives here
  TEMPLATE-original.tex    % the faculty template, untouched, for reference
```

- **One file per chapter**, `\include`d (not `\input`) so `\includeonly`
  can be used for fast partial compiles while drafting.
- **The Markdown draft in `70_Thesis/draft/` is now historical.** LaTeX is
  the source of truth as of 2026-08-03. The Markdown prose that already
  exists (the `fs` boundary, shortcut target construction, the
  computational profile) must be **ported**, not rewritten.
- Figures are **PDF or EPS** (vector). No PNG for plots; PNG only for
  sample grids and rendered video frames.

**No LaTeX toolchain is installed locally** (no `pdflatex`, no `latexmk`).
Compile in Overleaf, or install TeX Live.

## 3. Sectioning, labels and cross-references

`report` gives `\chapter` → `\section` → `\subsection`. **Do not go below
`\subsubsection`** — a fourth level in a 40-page document is a symptom of
misplaced material, not of precision.

**Label every chapter, section, figure, table and equation you reference.**
Prefixes, always:

| Prefix | For |
|---|---|
| `ch:` | chapter |
| `sec:` | section / subsection |
| `fig:` | figure |
| `tab:` | table |
| `eq:` | equation |
| `app:` | appendix |

Reference with **`\cref` / `\Cref`** (cleveref), never a hand-written
"Section~\ref{...}" — cleveref supplies the word and keeps it consistent.
`\Cref` at the start of a sentence, `\cref` mid-sentence.

## 4. Notation

Defined once in `macros.tex`; **never write the symbol by hand**. If a
symbol is not in the table, it does not go in the thesis until it is added
here and to the macros.

| Macro | Renders | Meaning |
|---|---|---|
| `\fbase` | $f_{\mathrm{base}}$ | the frozen base model |
| `\fadapted` | $f$ | the adapted model |
| `\adapter` | $\Delta_\phi$ | the trainable adapter |
| `\gate` | $g$ | the gate, $g(d)$ |
| `\xt` | $x_t$ | state at diffusion time $t$ |
| `\at` | $a_t$ | action |
| `\dstep` | $d$ | step size |
| `\params` | $\phi$ | adapter parameters |
| `\vel` | $v$ | velocity (flow / v-prediction) |
| `\noisepred` | $\epsilon$ | noise prediction |
| `\curv` | $\kappa$ | curvature of the probability-flow arc |
| `\composition` | the full rule | $f(\ldots) = f_{\mathrm{base}}(\ldots) + g(d)\cdot\Delta_\phi(\ldots)$ |

**Code identifiers are not prose.** Config keys, run names, file paths and
function names go in `\code{}` / `\cfg{}`: `\cfg{condition\_center}`,
`\code{ddim\_micro\_step\_v}`. This is also what lets American spellings
inside identifiers (`normalize`, `center`) coexist with British prose
without looking like errors.

**Prediction types are always explicit.** Whenever a loss, target or
shortcut construction is stated, name `model_type` and `prediction_type`.
CLAUDE.md Part 12; enforced because it is the domain's most-graded trap
([[rubric/04-knowledge-of-domain]]).

## 5. Numbers and provenance — the enforceable form of hard rule 8

**Rule: every measured numeral in the body carries provenance.** Two
permitted forms, no others:

1. **Inline** — `\prov{wandb-id}{ckpt}{commit}` immediately after the
   number, rendering a footnote.
2. **Table-level** — the table caption carries the run identifiers for
   every row, so individual cells do not repeat them.

A numeral with neither is a rule-8 violation. This is checkable:

```bash
# body numerals with no \prov and no provshort on the line — review each
grep -nE '[0-9]+\.[0-9]+' 70_Thesis/latex/chapters/*.tex \
  | grep -v 'prov' | grep -v '^\s*%'
```

Numbers that are **not** measurements — page budgets, counts of families,
equation constants, hyperparameters quoted from a config — do not need a
receipt, but hyperparameters should cite the config path.

**Formatting.** Use `siunitx` (`\num`, `\SI`) for anything with a unit or
needing consistent grouping. Significant figures follow the source: do not
render `0.01747` as `0.017` in one place and `0.0175` in another. Pick the
precision the measurement supports and hold it across the document.

**Uncertainty.** Where an interval exists, report it (`0.302
[0.251,0.356]`). Where a test was run, report the statistic and $n$.
Where there is **one seed**, the caption or sentence says so — silence
reads as concealment ([[rubric/03-experimental-evaluation]]).

## 6. Figures and tables

- **Tables use `booktabs`** — `\toprule`, `\midrule`, `\bottomrule`. No
  vertical rules, no `\hline`.
- **Figures are vector**, referenced from `figures/`, with the source
  script or notebook recorded in the figure's vault note
  (`30_Knowledge/writing/figure-*.md`).
- **Caption structure, always: claim → what is plotted → provenance.** The
  first sentence states what the reader should conclude
  ([[thesis-style-guide]] §7).
- Every float is `\cref`-referenced and **interpreted** in the text.
- Axes labelled with units; chance level or null drawn as a reference line
  on every probe plot; $n$ stated.
- `\begin{figure}[t]` by default. Avoid `[H]` except where placement
  genuinely breaks the argument.
- **Wide material goes to the appendix**, not into a shrunken font. Never
  use `\small`/`\footnotesize` on a table to make it fit — that is
  page-gaming by another route.

## 7. Citations

- `natbib` with `plainnat` (numeric, `sort&compress`). Switch to
  `unsrtnat` if citation-order numbering is preferred; the choice is free
  per the template.
- **`\citep` when the idea carries the sentence; `\citet` when the actors
  matter** — a disagreement, a direct comparison, work we build on
  directly ([[thesis-style-guide]] §6).
- **⭐ Peer-reviewed first, and this is a hard rule.** Cite the
  peer-reviewed version wherever one exists; the arXiv preprint is the
  citation of record **only** where no peer-reviewed publication does, and
  the entry says so. Many papers sit on arXiv for a year before their
  venue, so an arXiv listing is never evidence of status on its own. Check
  the arXiv `comments` field, the DOI, DBLP and the proceedings site. A
  bibliography full of preprints where proceedings exist reads as a
  literature search that stopped at the first result.
- **Bib hygiene.** Every entry needs a complete author list, title, year,
  and venue *or* an arXiv id. Any field that could not be confirmed from a
  primary source carries a `note` (`Venue unverified`, `Author list
  incomplete`). Those notes render in `plainnat` output, which is the
  point: an unverified entry says so rather than quietly asserting. No entry enters `refs.bib` before its
  `30_Knowledge/related-work/{slug}.md` note exists — the note is where
  the claim about the paper is verified (hard rule 7a).
- `refs.bib` currently holds **one** template entry (`lecun2015deep`).
  It is a placeholder; the real bibliography is built from
  `30_Knowledge/related-work/` ([[rubric/06-literature]]).

## 8. Draft markers — all must be gone at submission

| Macro | Meaning |
|---|---|
| `\nv{...}` | unverified — the claim has no source yet |
| `\todo{...}` | section not written |
| `\fig{...}` | figure not exported |
| `\provisional{...}` | rung-3 claim: a competing explanation is not excluded |
| `\inflight{...}` | run not settled; carries the snapshot date |
| `\estimate{...}` | analysed estimate, not a measurement |

The first three are **drafting scaffolding** and must be empty before
submission. The last three encode the claim ladder
([[thesis-style-guide]] §2) — they render in colour while drafting and
should be made no-ops (not deleted) for the final PDF, so the distinction
survives in the source.

```bash
# submission gate
grep -rn '\\todo{\|\\nv{\|\\fig{' 70_Thesis/latex/chapters/ && echo "NOT READY"
```

## 9. Pre-compile checklist

1. No `\todo`, `\nv`, `\fig` remaining.
2. Every measured numeral has `\prov` or a provenance-carrying caption.
3. Every float is labelled, `\cref`-referenced, and interpreted in text.
4. Every citation key resolves; every cited paper has a vault note.
5. No symbol used that is not in `macros.tex`.
6. Page count within 40 (body); overflow moved to the appendix, not shrunk.
7. British spelling in prose; identifiers untouched.
8. No `\subsubsection` nesting below the third level.
9. Compiles clean — no overfull `\hbox` warnings over 5pt, no undefined
   references.
