---
last_updated: 2026-08-06
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
| 2026-06-30 | [presentations/2026-06-30.html](presentations/2026-06-30.html) · [source](presentations/2026-06-30.slides.md) | first samples on the flow-matching base (loss ok · samples poor · flow faster) |
| 2026-06-27 | [presentations/2026-06-27-project-summary.html](presentations/2026-06-27-project-summary.html) · [source](presentations/2026-06-27-project-summary.slides.md) | project summary (AVID adapter · shortcut geometry · multimodal) |
| 2026-06-19 | [presentations/2026-06-19.html](presentations/2026-06-19.html) · [source](presentations/2026-06-19.slides.md) | 2026-06-19 |
| 2026-06-09 | [presentations/2026-06-09.html](presentations/2026-06-09.html) · [source](presentations/2026-06-09.slides.md) | 2026-06-09 |
| 2026-06-01 | [presentations/2026-06-01.html](presentations/2026-06-01.html) · [source](presentations/2026-06-01.slides.md) | 2026-06-01 |
| 2026-05-25 | [presentations/2026-05-25.html](presentations/2026-05-25.html) · [source](presentations/2026-05-25.slides.md) | — |

## Entries (newest first)

| Date | Category | Deliverable | Entry |
|---|---|---|---|
| 2026-08-07 | finding | D3 | [[entries/2026-08-07-pdd-the-adapter-cannot-decode-and-the-loss-cannot-tell]] |
| 2026-08-06 | finding | D2 | [[entries/2026-08-06-paired-control-turns-a-false-positive-into-a-small-true-one]] |
| 2026-08-06 | finding | D2 | [[entries/2026-08-06-objective-governs-action-specificity-not-capacity]] |
| 2026-08-05 | finding | D2 | [[entries/2026-08-05-turbo-action-effect-without-accuracy]] |
| 2026-08-04 | decision | D3 | [[entries/2026-08-04-efficiency-axis-becomes-the-thesis-spine]] |
| 2026-07-31 | finding | D2 | [[entries/2026-07-31-wan-blindness-located-and-the-scale-calibration-principle]] |
| 2026-07-30 | finding | D2 | [[entries/2026-07-30-action-embedding-is-a-learned-pedestal]] |
| 2026-07-30 | finding | D2 | [[entries/2026-07-30-avid-follows-actions-on-our-data-its-our-implementation]] |
| 2026-07-29 | finding | D2 | [[entries/2026-07-29-avid-rt1-follows-actions-blindness-is-data]] ⚠ *superseded by 2026-07-30* |
| 2026-07-25 | finding | D2 | [[entries/2026-07-25-base-parity-is-flat-visuals-robotarm-residual]] |
| 2026-07-24 | finding | D1 | [[entries/2026-07-24-online-vae-encode-6x-training-step]] |
| 2026-07-21 | finding | D2 | [[entries/2026-07-21-replace-eval-bug-fixed-adapter-action-blind]] |
| 2026-07-11 | finding | D4 | [[entries/2026-07-11-adapter-beats-base-on-prediction-accuracy]] |
| 2026-07-01 | added | D2 | [[entries/2026-07-01-quality-metrics-activated]] |
| 2026-07-01 | added | D2 | [[entries/2026-07-01-wan22-i2v-eval-loader]] |
| 2026-06-30 | added | D2 | [[entries/2026-06-30-wan22-variable-cond-frames]] |
| 2026-06-30 | change | D2 | [[entries/2026-06-30-wan22-stride1-121frames]] |
| 2026-06-29 | finding | D3 | [[entries/2026-06-29-flow-base-much-faster-train-sample]] |
| 2026-06-29 | finding | D3 | [[entries/2026-06-29-flow-base-first-samples-loss-ok-quality-poor]] |
| 2026-06-19 | finding | D3 | [[entries/2026-06-19-shortcut-v-averaging-bias-resolved]] |
| 2026-06-19 | decision | D3 | [[entries/2026-06-19-pivot-flow-matching-base]] |
| 2026-06-19 | finding | D1 | [[entries/2026-06-19-attention-grid-overflow-not-oom]] |
| 2026-06-17 | finding | D3 | [[entries/2026-06-17-shortcut-overfit-larger-data-volatile-loss]] |
| 2026-06-17 | progress | exploratory | [[entries/2026-06-17-multimodal-real-backbone-smoke]] |
| 2026-06-10 | progress | exploratory | [[entries/2026-06-10-multimodal-substrate-landed]] |
| 2026-06-05 | finding | D3 | [[entries/2026-06-05-anchor-baseline-confirms-shortcut-fewstep-gain]] |
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
