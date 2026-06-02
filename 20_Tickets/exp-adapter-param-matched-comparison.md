---
type: exp
scope: adapter
status: blocked
priority: medium
created: 2026-05-22
updated: 2026-05-22
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[feat-adapter-flops-per-step-estimator]]"
  - "[[writeup-writing-preregistered-protocol-paragraph]]"
  - "[[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]"
---

# FLOPs-matched comparison across adapter families

Run a head-to-head comparison between adapter families (AVID-style output
residual, LoRA, HyperAlign-style hypernetwork, UniCon-style hidden-state
+ full-FT-of-a-subset as a single-point sanity baseline) on the same
backbone + task + training schedule, with **training-time FLOPs held
constant** across families.

Sweep grid (per
[[../50_Decisions/decided/param-matched-adapter-comparison-definition]]):
**3 FLOPs budgets x 4-5 families x 3 seeds = 36-45 runs**, plus
**1 full-FT cell x 3 seeds = 3 runs** at the anchor budget. Total
~39-48 runs.

## Status: blocked

Blocked on [[feat-adapter-flops-per-step-estimator]]. The sweep cells
are computed from per-family FLOPs estimates; until each family exposes
a `flops_per_step(input_shape)` method and the three budget brackets
are pinned, the protocol cannot be pre-registered.

The methodology decision itself closed on 2026-05-22:

- Matched axis: training-time FLOPs (primary). Trainable params,
  inference FLOPs, peak memory reported per row, not matched.
- Chapter placement: D1 + D2 split (methodology in D1, headline
  numbers in D2).
- Sweep shape: full sweep, 3 budgets x families x 3 seeds.
- Full-FT row: single FLOPs point at the anchor budget; sanity
  baseline only, framed as such.

## Protocol

Lives at
[[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]].
This ticket tracks the *run*; the protocol holds the methodology and
needs to be updated to reflect the FLOPs-matched framing once the
estimator lands.

## Pre-launch checklist

Before the ticket moves from `blocked` → `open`:

- [x] Methodology decision closed (matched axis, sweep shape, chapter
      placement, full-FT inclusion).
- [ ] FLOPs estimator landed for all five rows
      ([[feat-adapter-flops-per-step-estimator]]).
- [ ] Three FLOPs budget brackets pinned: anchor at AVID-shortcut
      training-FLOPs/step at the live config, one lower, one higher.
- [ ] Per-family adapter sizes computed so each family hits its
      target FLOPs budget in each cell (this is the inverse of the
      old "match trainable params" step — now derived, not assumed).
- [ ] Trainable-param counts at each cell recorded for the *reported*
      column of the chapter table (replaces the user-reported 0.78%
      AVID figure as the source-of-truth secondary metric).
- [ ] Pre-registered methods paragraph drafted
      ([[writeup-writing-preregistered-protocol-paragraph]]) and
      timestamped before any cell launches.
- [ ] GPU-hour estimate for the full 39-48-run grid; sanity check
      against available budget.

## Definition of done

- All sweep cells run to a logged outcome (wandb run ids + ckpt paths).
- Primary metric (one of MSE / rollout-error / inference-cost) reported
  per (family, budget, seed) cell with cross-seed std.
- Secondary metrics (the remaining axes from the protocol) recorded.
- Per-cell run notes promoted to
  `30_Knowledge/experiments/{slug}.md` *only after* logged outputs
  exist — see hard rule 6 in [[../CLAUDE]] Part 3.
- One figure committed to the vault that the chapter will eventually
  reuse.
- The chapter section it feeds drafts off `protocol-…` + the run notes.

## Out of scope (separate tickets if needed)

- Shortcut-on variant of this comparison (would couple to consistency
  loss dynamics; revisit as a follow-up after the shortcut-off ranking
  is established).
- Cross-backbone replication (DynamiCrafter vs. another base).
- Multimodal coupled-dynamics variant.

## Related

- Decision: [[../50_Decisions/open/param-matched-adapter-comparison-definition]]
- Protocol: [[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
- Inbox origin: [[../00_Inbox/2026-05-21]] (16:17)
- Adapter family notes: [[../30_Knowledge/related-work/avid]],
  [[../30_Knowledge/related-work/hyperalign]],
  [[../30_Knowledge/related-work/unicon]]
