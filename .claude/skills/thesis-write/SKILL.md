---
name: thesis-write
description: Draft or extend a section of the rough thesis draft in 70_Thesis/draft/, pulling from recent 60_Updates/ entries and the linked vault sources. Use when the user says "write the thesis section on X", "extend the method chapter", "draft about the newest changes", "update the thesis with what we did this week". Respects deliverable separation (D1-D4) and the no-unsourced-numbers rule.
---

# thesis-write

Draft/extend one section of the rough Markdown thesis draft. The draft is
deliberately rough — prioritise getting sourced content down over polish.

## Args
- Target section, e.g. `30-method` or `40-experiments` (maps to
  `70_Thesis/draft/{file}.md`). If omitted, ask which section.
- Optional focus, e.g. "the frame-stride conditioning boundary".

## Steps

1. **Read the map + sources.**
   - `70_Thesis/outline.md` (what this section should contain + its status).
   - `70_Thesis/index.md` (the chapter → source mapping).
   - The target `draft/{file}.md` (what's already there).
   - The linked sources for that section: relevant `30_Knowledge/*`,
     `50_Decisions/decided/*`, `30_Knowledge/experiments/*`.
   - **Recent `60_Updates/entries/`** — this is the "what's new to write
     about" source when the task is "write about the newest changes." The
     deck slide-specs in `60_Updates/presentations/*.slides.md` are also fair
     game for already-structured talking points.

2. **Gap-check (CLAUDE.md Mode C).** If the sources don't actually support
   the section, STOP and grill the user (Discovery mode) before writing —
   don't generate generic filler.

3. **Write** into `draft/{file}.md`. Extend, don't clobber: append/expand
   sections, preserve existing prose unless asked to rewrite. Use Markdown,
   `[[wikilinks]]` to vault sources, and keep paragraphs short.

4. **Update status** in `70_Thesis/outline.md` and the `70_Thesis/index.md`
   chapter table (`stub` → `drafting` → `draft-complete`).

5. **Show** what changed and ask for one round of revision feedback.

## Hard constraints (thesis integrity — CLAUDE.md Part 12)
- **Deliverable separation.** D1=framework, D2=action world models,
  D3=shortcut adapters, D4=combined. Never mix deliverable evidence across
  chapters. Method (ch.3) is D1; Experiments/Results (ch.4–5) carry D2/D3/D4
  evidence separately.
- **No unsourced numbers** (hard rule 8). Every metric cites a real run
  (wandb id + ckpt + commit). Planned runs are *not* results — keep them in
  the protocol framing until they land.
- **Diffusion vs. flow matching.** Be explicit which side you mean; the loss
  target and shortcut formulation differ (`model_type`, `prediction_type`).
- **Don't introduce a new contribution mid-draft.** New adapter types /
  scope changes go through a `50_Decisions/` note first, not the prose.
- Mark anything you couldn't source with `_needs verification_` rather than
  asserting it.
