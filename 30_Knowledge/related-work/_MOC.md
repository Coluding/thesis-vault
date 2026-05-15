---
last_updated: 2026-05-15
status: living
---

# Related Work — Map of Content

> Index for [[30_Knowledge/related-work/]]. One file per paper that shapes
> the thesis. AI overwrites this when individual papers' notes change.
>
> **Initial population: 2026-05-15.** Eight seed notes created from the
> vendored PDFs under `docs/paper/`. Each per-paper note starts as a stub
> ("title, venue, why it matters here") and gets fleshed out via Discovery
> mode when the related-work chapter or methods chapter needs it.

## How this maps to the thesis

The four deliverables in [[../../10_now/positioning|positioning]] each have
their own related-work cluster. The columns:

- **D1/D2/D3/D4** — which deliverable's chapter cites this paper.
- **Relevance** — `framework` (informs D1 design), `baseline` (D2/D3/D4
  honest comparison), `theory` (provides the loss / parameterisation),
  `backbone` (we wrap their model), `negative` (we explicitly position
  against it).

## The seed eight (from `docs/paper/`)

### Adapter / control architecture

| Paper | D | Relevance | One-line |
|---|---|---|---|
| [[avid]] | D1 / D2 | framework, baseline | Output-level residual adapter for pretrained video diffusion. Closest precedent for the "additive correction on a frozen base" architecture; we generalise + add action / step-size conditioning. |
| [[unicon]] | D1 | framework, baseline | Hidden-state / skip-connection control adapter for diffusion. The reference for our hidden-state adapter family. |
| [[hyperalign]] | D1 / D2 | framework, baseline | Hypernetwork that produces task-specific LoRA weights for diffusion. Vendored as a starting point in the repo; the reference for our hypernetwork adapter family. |
| [[cafm]] | D1 | _needs verification_ | Paper title starting with "cafm" in `docs/paper/cafm.pdf`. Likely conditional adapters for flow matching — *verify exact title / authors / venue before citing*. |

### Few-step sampling / shortcut

| Paper | D | Relevance | One-line |
|---|---|---|---|
| [[shortcut-models]] | D3 | theory, framework | Source of the step-size-conditioned consistency formulation we use for D3. The closest direct precedent for our shortcut adapter. |
| [[consistency-models]] | D3 | theory, baseline | Standalone few-step generative models trained with consistency. We borrow the loss form but apply it on a frozen base via the adapter. |
| [[self-distillation]] | D3 | theory | Self-distillation as a route to few-step sampling. Methodological cousin of consistency / shortcut. |
| [[dpm-solver]] | D3 / D4 | baseline | Solver-side few-step inference for diffusion. Honest non-trained baseline against which our shortcut adapter must compete at matched step budgets. |

## Adjacency clusters (visual)

```
                            Frozen base?
                  yes (this thesis)        no
              ┌──────────────────────┬─────────────────────┐
   Adapter   │ avid                  │ unicon (control      │
   surface   │ hyperalign            │   adapters retrain   │
              │ cafm (?)              │   the model)         │
              │ ours: action + d      │                      │
              ├──────────────────────┼─────────────────────┤
   Few-step  │ ours: D3 / D4         │ consistency-models   │
   sampling  │ shortcut-models       │ self-distillation    │
              │   (in part)           │ dpm-solver           │
              └──────────────────────┴─────────────────────┘
```

Two papers sit closest to the thesis's actual contribution surface
(frozen base × shortcut behaviour): **shortcut-models** and **avid +
{shortcut sampling}** combined. Neither is exactly the thesis — the gap
between them is what D3/D4 is about.

## Anti-positioning health check

The thesis's [[../../10_now/positioning|positioning doc]] makes several
anti-positioning claims. Each is re-evaluated against the related-work
record:

| Claim | Health | Evidence |
|---|---|---|
| "Not a new diffusion/flow algorithm. We adapt the prior, not replace it." | **Holds.** | All eight seed papers operate either as adapters on a frozen base or as full-model retraining — no overlap with "new prior" framing. |
| "Not a fine-tuning paper. The base stays frozen." | **Holds.** | AVID, HyperAlign, UniCon, CAFM all keep the base frozen too — anti-positioning against fine-tuning papers, not against these neighbours. |
| "Not a pure consistency-models paper. We use the loss but on a frozen base." | **Holds, but the boundary is the key contribution claim.** | Consistency Models, Shortcut Models, Self-Distillation all retrain the model. The novelty of D3 is *applying shortcut training to a frozen base via the adapter*, which none of the cited papers do. |
| "Not a control/RL paper." | **Holds.** | None of the seed eight are RL papers; comparison is to world-model / planning lit which lives in the "coverage gaps" section below. |
| "Not an architecture-search paper." | **Holds.** | Taxonomy is fixed up-front by the proposal. |

## Coverage gaps the thesis still has to add later

Notes worth adding when the related-work chapter is drafted but not in
`docs/paper/` yet:

- **Diffusion world models** — Dreamer / DreamerV3 / DIAMOND / GAIA-style
  papers that use a generative model as a world model. Direct comparison
  surface for D2.
- **Flow matching foundations** — Lipman et al., Albergo & Vanden-Eijnden,
  Rectified Flow. The theoretical grounding for the flow-matching half of
  D1.
- **LoRA / adapter-tuning in NLP** — Hu et al. LoRA, AdapterHub. The
  conceptual ancestor that motivates the framework framing.
- **Video diffusion backbones** — DynamiCrafter, OpenSora, Cosmos — already
  reflected in the codebase (`backbones/dynamicrafter`, vendored OpenSora,
  `docs/cosmos-predict2.5-integration-report.md`). Backbones we *wrap*
  rather than compete with; each deserves a short backbone-note when the
  D2 / D4 chapters describe the empirical setup.
- **MetaWorld / robotic benchmarks** — Yu et al. MetaWorld and any
  follow-on benchmark the thesis evaluates on. Lives partially in
  `30_Knowledge/datasets/` once that folder gets populated.

These belong in this MOC's table once the corresponding per-paper notes
exist. The eight seed notes above are the strict subset that already has
PDFs on disk.

## Gaps in the seed eight (what these notes do *not* know yet)

A discovery-mode prompt list. Each gap is something a single read-through
of the paper PDF would fill. They are flagged in the per-paper notes too.

- **CAFM** — full title, authors, venue, exact contribution. The note is
  a stub until the PDF is read. _needs verification_.
- **HyperAlign** — exact LoRA-target-module set (we cache the paper's
  module list as `PAPER_HYPERALIGN_TARGET_MODULES` in
  `adapters/low_rank/common.py`; the rationale for each module belongs in
  the note).
- **UniCon** — exact diagram reference for "Figure 3(d) hidden-state"
  cited in the repo README. Tie it back to the note.
- **AVID** — clarify whether the AVID output adapter conditions on the
  base output or replaces it, and how that maps to our `composition` modes
  (`add` / `replace` / `mask_mix`).
- **Shortcut Models** — derive the loss form vs. consistency-models
  form. The note's "theory" section is the place; ties to
  `30_Knowledge/theory/shortcut-loss-derivation.md` once written.
- **DPM-Solver** — figure out at how few steps it stops being competitive,
  so the D3 chapter's baseline curve can be honestly drawn.

## Refresh discipline

Every entry is `status: living`. Update triggers:

- A paper note's relevance changes (e.g. `framework` → also `theory`) → update the file + this MOC's table.
- A new paper joins the eight (a friend recommends one, a reviewer points to one, a new arXiv drop changes the SOTA) → add a per-paper note + add to the table.
- A paper turns out to be **less** relevant than first thought → keep the
  note (don't delete) but mark relevance `peripheral` or `negative` and
  note why in the body.

Refresh cadence: a focused sweep when the related-work chapter is drafted
(read all eight PDFs, fill the stub bodies); a quarterly sweep
thereafter; reactive updates whenever the thesis's framing in
[[../../10_now/positioning]] changes.

## Related

- [[../../10_now/positioning]] — thesis framing that cites these papers
- [[../../10_now/architecture]] — codebase modules that implement ideas from these papers
- `docs/paper/` — the canonical PDF copies
- `30_Knowledge/theory/` — derivations referencing these papers (folder to be populated)
- `30_Knowledge/writing/` — thesis-section drafts (folder to be populated)
- [[../../50_Decisions/]] — open scope decisions derived from this MOC (folder to be populated)
