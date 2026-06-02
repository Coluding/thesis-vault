---
name: weekly-deck
description: Build a polished, self-contained HTML presentation for the weekly advisor meeting from the 60_Updates/ log. Use when the user says "make the weekly deck", "build the presentation", "prep slides for the meeting", or "turn the updates into a deck". Reads recent update entries + product-state + recent experiments + open decisions, writes a slide-spec, and runs build_deck.py to emit an offline-portable themed .html in 60_Updates/presentations/.
---

# weekly-deck

Generate `60_Updates/presentations/{date}.html` — a single self-contained
(offline, no external deps) themed slide deck for the weekly meeting.

## Inputs to read (sparingly)
- `60_Updates/index.md` + the `entries/` since the last deck (default: last 7
  days; ask if ambiguous).
- `10_now/product-state.md` (current state).
- Recent `30_Knowledge/experiments/*` (for sourced result numbers).
- `50_Decisions/open/*` (blockers / decisions needing the advisor).
- High-priority open tickets (for "next steps").

## Steps

1. **Decide the date** (the meeting date; default today) and the window of
   entries to include.

2. **Write the slide-spec** to `60_Updates/presentations/{date}.slides.md`
   using the mini-syntax below. Suggested arc:
   - Agenda
   - Progress since last week (from `progress`/`added` entries)
   - New findings (from `finding` entries)
   - Results (a `::: metrics` slide — **sourced numbers only**, with run ids)
   - Blockers & open decisions (from `blocker`/`decision` entries + open decisions)
   - Next steps (from high-priority tickets)
   Keep ~6–10 slides. The title slide is auto-generated — do not write one.

3. **Build** (the script is stdlib-only Python 3):
   ```bash
   python3 .claude/skills/weekly-deck/build_deck.py \
     --input 60_Updates/presentations/{date}.slides.md \
     --output 60_Updates/presentations/{date}.html \
     --title "Weekly Update" --subtitle "<one-line theme>" \
     --date {date} --author "Lukas Bierling" --deliverable "<e.g. D2/D3>"
   ```

4. **Register** the deck in the "Presentations" table of
   `60_Updates/index.md` and tell the user the output path. Mention they can
   open it in a browser (arrow keys / on-screen ‹ › to navigate) and
   File → Print → Save as PDF for a portable copy.

## Slide-spec mini-syntax (what build_deck.py understands)

- Slides separated by a line containing only `---`.
- `# H1` (slide title), `## H2`, `### H3` (eyebrow). A slide whose only
  content is an `# H1` renders as a full-bleed **section divider**.
- `- bullet` / `* bullet`; indent by **2 spaces** per nesting level.
  `1.` etc. → ordered list.
- ` ```lang … ``` ` → code block. `> quote` → accent **callout** box.
- `![caption](path/to/img.png)` → figure (local images are **base64-embedded**
  so the file stays self-contained).
- Inline: `**bold**`, `*italic*`, `` `code` ``, `[text](url)`.
- Metric cards (use for results):
  ```
  ::: metrics
  val MSE | 0.087 | run abc123 · ckpt e5/last.ckpt
  FID | 24.3 | run abc123
  :::
  ```
- Two columns:
  ```
  ::: columns
  left column markdown
  |||
  right column markdown
  :::
  ```
- Speaker notes: `<!-- notes: ... -->` (hidden on screen, shown when printing).

## Guardrails
- **No unsourced numbers on a slide.** Every metric card needs a run
  citation (CLAUDE.md hard rule 8). If you can't source it, don't put a
  number — state the qualitative status instead.
- Don't promote planned experiments to results (Part 12). A planned run is a
  "next step", not a metric.
- Keep the slide-spec `.md` next to the `.html` so the deck can be regenerated.
