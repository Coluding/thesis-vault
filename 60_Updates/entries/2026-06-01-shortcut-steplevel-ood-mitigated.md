---
date: 2026-06-01
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../20_Tickets/risk-shortcut-eval-steplevel-out-of-distribution]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../30_Knowledge/related-work/shortcut-models]]"
---

# Shortcut eval would have run fully out-of-distribution on step size — now fixed in code

## What

Building the multi-step eval grid surfaced a train/eval mismatch on the
shortcut step-size axis: the adapter was only ever trained on
`step_level ∈ {1, 2, 4}` (raw timesteps), but any realistic few-step
rollout needs much larger jumps — a 25-step rollout implies `step_level ≈ 40`
(10× the trained max), and the 1/4/8-step regime where a shortcut is
actually valuable needs `step_level ∈ {125, 250, 500, 1000}`, all untrained.
Worse, the adapter embeds `step_level` through a raw `Linear(1, hidden)`, so
feeding 250–1000 (vs. the 1–4 it was tuned on) saturates SiLU and produces
garbage — a scaling bug on top of the data-coverage gap.

## Why it matters

Without this fix, poor few-step rows in the eval grid would have been a
**training-coverage artifact, not evidence about whether shortcut modeling
works** — exactly the kind of result that would mislead the D3 conclusion.
The premise of D3 (condition on larger `d` → predict the averaged direction →
stay accurate at few steps) was literally never exercised in its trained
regime.

## Evidence / sources

- Train range, inference range, and the 10–250× gaps are arithmetic from
  the config and DDIM scheduler, worked out in
  [[../../20_Tickets/risk-shortcut-eval-steplevel-out-of-distribution]]
  (verified 2026-05-25 against `configs/diffusion_avid_shortcut_metaworld.yaml`
  and `docs/paper/shortcut_models.pdf`).
- Grounding: the [[../../30_Knowledge/related-work/shortcut-models]] paper uses
  a dyadic ladder `d ∈ {1/128, …, 1}` (`step_level ≈ 8…1000`); our old max of 4
  sat *below* the paper's finest step.
- **Mitigation shipped (B + C):** step sizes canonicalised to normalised
  `s ∈ (0,1]` via a configurable `shortcut_step_schedule`, with a `log2`
  transform on the adapter input so a dyadic ladder spreads to ~`[-7, 0]`
  instead of a raw 1→1000 scalar; the same schedule drives both training
  sampling and the eval grid (one source of truth). AVID config now trains the
  paper's full `log2 1/128…1` ladder. _Code-shipped; no run numbers yet._

## Next

The OOD/units/embedding gaps are closed in code — the **only remaining item is
empirical**: run training with the full schedule and read the multi-step eval
grid (does the adapter degrade gracefully at few steps vs. the base?). Watch
the self-consistency collapse modes at large `s`, where the two-coarse-step
teacher is weakest.
