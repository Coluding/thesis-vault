---
last_updated: 2026-06-01
status: living
---

# Project Updates — chronological progress log

> The **curated, outward-facing** layer of the vault. This is the narrative
> of how the thesis is going, written for the weekly advisor meeting — *not*
> a raw work log. It distils the internal layers:
> `10_now/product-state.md` (living snapshot),
> `30_Knowledge/sessions/` (raw session logs),
> `30_Knowledge/experiments/` (actual runs) and `50_Decisions/`.
>
> Each line below points to a detail file in `entries/`. The
> `/weekly-deck` skill reads this index + recent entries to build the
> meeting presentations in `presentations/`.

## How this directory works

- **One entry per topic**, not per day: `entries/{YYYY-MM-DD}-{slug}.md`.
  Use `/log-update` (or just tell Claude "log an update: …").
- **This index is reverse-chronological** — newest first. Each entry gets a
  one-line pointer here with its category and the deliverable it touches.
- **Categories:** `progress` (work advanced) · `finding` (we learned
  something) · `added` (new capability/code) · `blocker` (stuck/needs a
  call) · `decision` (a choice was made).
- **Honest-numbers rule applies** (CLAUDE.md hard rules 7–8): any metric in
  an entry needs a run citation. Decks inherit this — no unsourced numbers
  on a slide.

## Presentations

Generated decks live in `presentations/{YYYY-MM-DD}.html` (self-contained,
offline-portable). The slide source is kept alongside as
`presentations/{YYYY-MM-DD}.slides.md` so a deck can be regenerated or
hand-edited.

| Date | Deck | For meeting |
|---|---|---|
| 2026-06-01 | [presentations/2026-06-01.html](presentations/2026-06-01.html) · [source](presentations/2026-06-01.slides.md) | 2026-06-01 |
| 2026-05-25 | [presentations/2026-05-25.html](presentations/2026-05-25.html) · [source](presentations/2026-05-25.slides.md) | — |

## Entries (newest first)

| Date | Category | Deliverable | Entry |
|---|---|---|---|
| 2026-06-04 | added | T1 | [[entries/2026-06-04-training-run-io]] |
| 2026-06-01 | finding | D1 | [[entries/2026-06-01-flash-attention-sdpa-bf16]] |
| 2026-06-01 | finding | D1 | [[entries/2026-06-01-mask-mix-gate-composition-surface]] |
| 2026-06-01 | finding | D3 | [[entries/2026-06-01-shortcut-steplevel-ood-mitigated]] |
| 2026-05-29 | added | D1 | [[entries/2026-05-29-output-format-affine-vs-direct]] |
| 2026-05-28 | progress | D1 | [[entries/2026-05-28-unicon-shortcut-integration]] |
| 2026-05-28 | decision | D3 | [[entries/2026-05-28-deprecate-twostep-add-heun-smoothness]] |
| 2026-05-28 | finding | D3 | [[entries/2026-05-28-twostep-shortcut-no-stepsize-variation]] |
| 2026-05-28 | decision | D2 | [[entries/2026-05-28-frame-stride-decision-anchor-at-avid]] |
| 2026-05-28 | progress | D3 | [[entries/2026-05-28-shortcut-target-refactor]] |
| 2026-05-25 | blocker | D2 | [[entries/2026-05-25-hyperalign-oom-h100-blocker]] |
| 2026-05-25 | finding | D2 | [[entries/2026-05-25-frame-stride-conditioning-audit]] |
