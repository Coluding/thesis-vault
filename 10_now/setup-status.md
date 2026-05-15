---
last_updated: 2026-05-15
status: living
---

# Second-brain Setup — Status + What's Missing

Living inventory of the vault's coverage for the thesis. What's done, what's
empty, what's next.

**Scope of this doc:** the experimental backlog lives in `20_Tickets/`; the
codebase state lives in [[architecture]]; the thesis framing lives in
[[positioning]]. This doc inventories everything **else** — paper notes,
theory derivations, writing scaffolds, advisor cadence, vault operations.

## ✅ Done

### Operating spec
- [[../CLAUDE]] — three modes, hard rules, frontmatter contracts, naming
  conventions tuned for the thesis.
- [[../Home]] — vault home page with generative queries.

### Living docs
- [[architecture]] — scaffold reflecting the codebase at HEAD. Honest about
  what's wired vs. _needs verification_.
- [[positioning]] — four-deliverable framing (D1–D4) with evidence
  requirements per deliverable and current state.
- [[product-state]] — explicitly pre-results; empty result table by
  hard-rule-8 discipline.

### Reference material on disk (not yet vault-ised)
- `docs/thesis-plan/Updated_Thesis_Proposal.pdf` — the proposal verbatim.
- `docs/paper/*.pdf` — 8 vendored reference papers (AVID, CAFM, consistency
  models, DPM-Solver, HyperAlign, self-distillation, shortcut models,
  UniCon). One per related-work note.
- `docs/foundation-architecture.md`, `docs/hyperalign-architecture-replication.md`,
  `docs/open-video-base-models.md`, `docs/open_sora_analysis.md`,
  `docs/opensora_vs_dynamicrafter_architecture_report.md`,
  `docs/shortcut_action_summary.md`,
  `docs/cosmos-predict2.5-integration-report.md` — existing technical
  reports written before this vault existed. Each is a candidate seed for
  a `30_Knowledge/tech/` or `30_Knowledge/related-work/` note.

## 🟡 Stub-only — needs content

Folders prescribed by [[../CLAUDE]] that are unpopulated. Each gap blocks a
specific synthesis query.

| Folder | What it should hold | Blocks | Priority |
|---|---|---|---|
| `30_Knowledge/related-work/` (per-paper) | One note per paper in `docs/paper/`. Currently only `_MOC.md` exists. | Related-work chapter draft; "what do we know about X" queries. | 🔴 critical |
| `30_Knowledge/theory/` | Derivations: shortcut-loss form, diffusion vs. flow loss equivalence, consistency loss variants, velocity vs. noise parameterisation. | Methods chapter; reviewer-question robustness. | 🟠 high |
| `30_Knowledge/experiments/` | One note per real run with wandb id, ckpt path, metrics. Currently empty. | All result-shaped queries. Required by [[../CLAUDE]] hard rule 8 before any number lands in the thesis. | 🔴 critical (once first run finishes) |
| `30_Knowledge/tech/` | Implementation notes — adapter family deep-dives, backbone choice rationale, training-loop quirks. | Methods chapter; future-me reading the repo. | 🟠 medium |
| `30_Knowledge/datasets/` | MetaWorld task list + preprocessing, video dataset choice, action normalisation. | D2 reproducibility; thesis dataset section. | 🟡 medium |
| `30_Knowledge/writing/` | Thesis outline, chapter drafts, figure list, bibliography. | Producing thesis chapters. | 🟠 medium (becomes critical near submission) |
| `30_Knowledge/advisor/` | Meeting notes + action items. | Coherent "what did the advisor say" recall across months. | 🟡 medium |
| `30_Knowledge/sessions/` | End-of-session logs per [[../CLAUDE]] Mode A. | Cross-session continuity. | 🟢 low (auto-fills with use) |
| `20_Tickets/` | Experiment / writeup / bug tickets. | Prioritised "what next" decisions. | 🟠 high |
| `00_Inbox/` | Daily fleeting-thought files. | Mode A safety valve. | 🟢 low (auto-fills with use) |
| `50_Decisions/open/` and `decided/` | Non-trivial design choices (primary backbone, D2 default adapter, shortcut target method, multimodal scope). | Coherent thesis scope decisions. | 🟠 high |

## 🟡 Operational gaps

Vault and tooling that [[../CLAUDE]] assumes exist but don't yet:

- **Vault is not its own Git repo** (it lives inside the implementation
  repo's working tree at `vellux-vault/`). Two cleanups possible:
  (a) initialise as a nested Git repo and add to outer `.gitignore`, or
  (b) accept that vault snapshots land in the main repo's history. (a) is
  what `90_Meta/scripts/snapshot.sh` assumes.
- **Vault folder is still named `vellux-vault/`** — the user will rename
  later. References inside files use relative `[[wikilinks]]` only, so the
  rename should be a straight folder rename with no breakage.
- **No `20_Tickets/_index.md`** template yet.
- **No `00_Inbox/{today}.md`** yet — auto-fills on first inbox entry.
- **No Obsidian Templater config visible** — vault works as plain
  Markdown; templates are optional.
- **No `50_Decisions/` folder yet** — create on first decision.

## 🟢 Optional — nice to have

- **Figure registry** in `30_Knowledge/writing/figures/` so every figure
  the thesis uses has a regeneration script and source data path.
- **Bibliography file** (`thesis.bib`) under `30_Knowledge/writing/` —
  populated as related-work notes accumulate; one BibTeX entry per
  related-work note.
- **Glossary** for the thesis-specific notation (`f_base`, `Δ_φ`, `g(d)`,
  `s(x_t, t, d, a_t)`, etc.) under `30_Knowledge/writing/glossary.md`.
- **Standing experiment-launch checklist** under `30_Knowledge/tech/`:
  config locked → ckpt path set → wandb project set → ticket open → run
  → finished → experiment note → ticket closed.

## What I'd do next, ranked

This list is **non-experiment work that should happen in parallel with
running the actual experiments**. The experimental backlog itself lives in
`20_Tickets/exp-*.md` as it gets prioritised.

1. **Backfill the related-work notes.** Eight papers in `docs/paper/`,
   one vault note each in `30_Knowledge/related-work/`. The `_MOC.md`
   already has the index — fill in the per-paper bodies. **This is the
   highest-leverage personal work** because the related-work chapter is
   the lowest-risk piece to write in parallel with experimental work.

2. **Write the first three theory notes:**
   - `30_Knowledge/theory/shortcut-loss-derivation.md` — how the multi-step
     self-consistency objective is derived and what it converges to.
   - `30_Knowledge/theory/diffusion-vs-flow-parameterisation.md` — noise
     vs. velocity prediction, why both fit the same composition rule.
   - `30_Knowledge/theory/adapter-taxonomy.md` — what makes output /
     hidden-state / LoRA / hypernetwork *different categories* rather
     than degrees of the same thing. This is the spine of the D1 chapter.

3. **Open the four scope decisions** as `50_Decisions/open/` notes:
   - `primary-backbone.md`
   - `d2-default-adapter.md`
   - `shortcut-target-method.md`
   - `multimodal-scope.md`

   Each one is short — title, question, options, current lean, target
   decision date. They unblock D2/D3/D4 framing in [[positioning]].

4. **Backfill one experiment note** for any past run with logged outputs
   the user remembers. This populates `30_Knowledge/experiments/` with a
   non-empty seed and validates the schema in [[../CLAUDE]] §Part 8.

5. **Write the thesis outline** as `30_Knowledge/writing/outline.md`. One
   page, chapter-by-chapter. Cross-link to [[positioning]] deliverables.
   Useful even at this stage because it forces the four deliverables to
   map cleanly onto chapter boundaries.

6. **Create an advisor-meeting template** at
   `30_Knowledge/advisor/_template.md`. One per meeting.

**What I'd defer:**
- Figures, bibliography, and writing infrastructure beyond the outline.
  Premature until results exist.
- Open-source / public-release thinking. Not the primary deliverable;
  decide once the thesis is well underway.
- Multimodal extension as a chapter. Open the decision (#3) but don't
  invest until D2 + D3 evidence is in.

## Trigger points for this doc's rewrites

- **First experiment note lands.** Drop the gold-set / experiment-note
  gap from critical priority.
- **All eight related-work notes drafted.** Drop the related-work item
  from #1; replace with whatever the next-highest gap is.
- **Thesis outline lands.** Add a "writing scaffold" section to "Done".
- **First D2 ablation run finishes.** Update [[product-state]] *and* update
  this doc's "Done" section to reflect that the framework's empirical
  case has its first data point.
- **D4 first combined run finishes.** This entire doc gets a serious
  rewrite — the focus shifts from "what's missing" to "what does the
  thesis need to defend."

## Related

- [[architecture]] · [[positioning]] · [[product-state]] — living docs
- [[../30_Knowledge/related-work/_MOC]] — paper index
- [[../CLAUDE]] — operating spec and hard rules
- `docs/thesis-plan/Updated_Thesis_Proposal.pdf` — the proposal verbatim
