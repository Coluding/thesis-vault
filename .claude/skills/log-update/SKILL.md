---
name: log-update
description: Capture a project/thesis progress update into the 60_Updates/ log — a curated, outward-facing entry for the weekly advisor meeting. Use when the user says "log an update", "note our progress", "record what we found", "we just added X", or describes a finding/blocker/decision worth surfacing at the weekly meeting. Distils the internal layers (product-state, sessions, experiments, decisions) into one chronological entry; feeds the /weekly-deck presentations.
---

# log-update

Create one curated progress entry in `60_Updates/entries/` and register it in
`60_Updates/index.md`. This is the **outward-facing** layer — written for the
weekly meeting, not a raw work log.

## When to use
- User: "log an update", "record this for the meeting", "we found X", "we
  added Y", "I'm blocked on Z".
- Proactively at the end-of-session ritual (offer to distil the session into
  an update entry — see CLAUDE.md).

## Steps

1. **Pick category + deliverable.** Category ∈ `progress | finding | added |
   blocker | decision`. Deliverable ∈ `D1 | D2 | D3 | D4 | exploratory`.

2. **Gather sources** (do not invent). Pull from `10_now/product-state.md`,
   recent `30_Knowledge/experiments/*`, `50_Decisions/*`, and relevant
   tickets. Any metric MUST cite a real run (wandb id + ckpt + commit) per
   CLAUDE.md hard rules 7–8. If a number isn't sourced, write
   `_needs verification_` — never fabricate.

3. **Create** `60_Updates/entries/{YYYY-MM-DD}-{slug}.md` with frontmatter:
   ```yaml
   ---
   date: 2026-05-25
   category: finding          # progress | finding | added | blocker | decision
   deliverable: D2            # D1 | D2 | D3 | D4 | exploratory
   meeting:                   # target meeting date, if known
   sources: []                # [[links]] to runs/tickets/decisions/notes
   ---
   ```
   Body sections: **What** (1–2 sentences) · **Why it matters** (the meeting
   takeaway) · **Evidence / sources** (links, sourced numbers) · **Next**
   (what this unblocks or what's needed). Keep it tight — this is a talking
   point, not an essay.

4. **Register in the index.** Prepend a row to the "Entries (newest first)"
   table in `60_Updates/index.md`:
   `| {date} | {category} | {deliverable} | [[entries/{date}-{slug}]] |`
   Keep the table reverse-chronological.

5. **Confirm** in chat: "Logged update: [path]" + the one-line takeaway.

## Guardrails
- Never write an unsourced number (hard rule 8). 
- Curate, don't dump — if it's a raw work log, it belongs in a session log
  (`30_Knowledge/sessions/`), not here.
- One topic per entry. Multiple topics → multiple entries.
