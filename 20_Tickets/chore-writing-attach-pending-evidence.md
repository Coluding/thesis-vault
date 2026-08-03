---
type: chore
scope: writing
status: open
priority: high
created: 2026-08-02
updated: 2026-08-02
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/writing/thesis-storyline]]", "[[../30_Knowledge/writing/storyline-experiment-requirements]]"]
---

# Attach pending evidence for three claims the storyline now makes

Three claims are **stated in the storyline but have no artefact in the vault**.
Lukas confirmed the underlying runs exist and will supply them (2026-08-02).
Until then they must not appear in the draft as numbers or figures.

| # | Claim | Where it is asserted | What is needed |
|---|---|---|---|
| 1 | **Planning through the DC world model works** | [[../30_Knowledge/writing/thesis-storyline]] §2 | run id · checkpoint · planner config · the result (success metric or action-recovery numbers) · wall-clock per planning step |
| 2 | **241 frames on H100 → ~35–40 s decoded**, vs ~9–10 s for DynamiCrafter | storyline §10 (new) | run id · frames · resolution · VRAM · decoded seconds · the matched DC number |
| 3 | **Few-step quality: shortcut arm beats the no-shortcut control** | [[../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]] §few-step | run id · which N values · the comparison (metric or labelled qualitative panel) |

## Why this ticket exists

Each is load-bearing:

- **(1)** is the second link of the main chain; everything after it ("too slow →
  flow → shortcut") is motivated by it.
- **(2)** is the concrete practical argument for the flow/Wan branch. The largest
  `temporal_length` in any committed config is **97**, so nothing in the repo
  currently supports 241.
- **(3)** is the only quality evidence D3 would have; without it the D3 claim is
  strictly about the objective's learnability.

## Rule while pending

Draft prose may **describe** these qualitatively and mark them as forthcoming.
It may **not** state a number, plot a figure, or cite a run id that does not
exist. Per [[../CLAUDE]] hard rules 7–8, a number without a traceable run is a
fabrication regardless of whether the run happened somewhere.

## Close when

All three have an experiment note (or a labelled observation, for 3) plus a row
in [[../30_Knowledge/experiments/_index]].
