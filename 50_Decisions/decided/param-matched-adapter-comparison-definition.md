---
type: decision
status: decided
created: 2026-05-22
updated: 2026-05-22
decided_at: 2026-05-22
target_date:
scope: adapter
related:
  - "[[../../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]"
  - "[[../../20_Tickets/exp-adapter-param-matched-comparison]]"
  - "[[../../20_Tickets/feat-adapter-flops-per-step-estimator]]"
  - "[[../../20_Tickets/writeup-writing-preregistered-protocol-paragraph]]"
  - "[[../../10_now/positioning]]"
---

# Decision: Definition + scope of the param-matched adapter comparison

## Status

**Decided 2026-05-22.** Unblocks the protocol at
[[../../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
once the new prerequisite (FLOPs estimator) lands. The run-ticket
[[../../20_Tickets/exp-adapter-param-matched-comparison]] remains
`blocked`, now on the estimator rather than on this decision.

## Context

The user proposed comparing adapter families (AVID-style output residual,
LoRA, HyperAlign-style hypernetwork, UniCon-style hidden-state — possibly
also full-FT-of-a-subset) on the same backbone + task + training schedule,
holding **trainable parameter count** roughly equal across families
instead of letting each family run at its default size. The motivating
observation: AVID-shortcut on MetaWorld currently sits at ~0.78% of
backbone params trainable (user-reported, _needs verification_), and a
LoRA / hypernetwork at their defaults would land at very different
fractions — so comparing them "out of the box" conflates *which family*
with *how much capacity*.

Promotion of the inbox idea into a runnable protocol surfaced four
methodology questions that have to be answered together, because each
choice changes what the others mean.

## Questions to resolve

### Q1. What does "matched" mean?

Candidates, each gives different rankings:

| Definition | Pros | Cons |
|---|---|---|
| Total trainable params | Easiest to state, easiest to match exactly | Ignores that 1M LoRA params ≠ 1M hypernet params in either training cost or expressivity |
| Trainable params per backbone block | Controls for *where* the capacity sits across depth | Hypernetwork doesn't fit this framing — it's a separate network, not a per-block addition |
| Training-time FLOPs | Honest about actual training budget | Harder to match precisely; depends on input shape |
| Inference-time FLOPs | Honest about deployment cost (the planning use case in D4) | Hypernetwork is almost-free at inference; this metric would massively over-budget it relative to LoRA |
| Training-time peak memory | Honest about hardware feasibility | Memory profiling is noisy; matching is approximate |

Reasonable picks: trainable-params as **primary** (simplest, easiest to
state in the chapter), with FLOPs and memory **reported but not matched**
so the reader sees the secondary trade-offs. But this is a real call,
not obvious.

**Chosen (2026-05-22): training-time FLOPs as the primary matched axis;
trainable params, inference FLOPs, and peak memory reported per row but
not matched.** Rationale: the D1+D2 split (Q3) puts the protocol in the
methodology chapter, so the matched axis has to defend itself against
the "different families, different param meanings" objection a reviewer
will raise about a param-matched comparison. FLOPs is the honest cost
axis. Caveats this introduces:

- The existing 0.78% trainable-param AVID-on-MetaWorld anchor is **not**
  the operative anchor anymore. Need to compute AVID-shortcut training
  FLOPs/step at the live config and use that as the FLOPs anchor.
  Treat the 0.78% number as one of the *reported* secondary columns.
- Matching FLOPs across families requires a per-family FLOPs estimator
  in the framework. LoRA and AVID-output are simple (extra matmul +
  extra output projection); hypernetwork has a separate-network pass
  that has to be accounted for; UniCon-style hidden-state needs the
  per-block injection cost. The framework should expose
  `adapter.flops_per_step(input_shape)` so this isn't ad-hoc.
- Inference FLOPs being a separate reported column matters for D4
  (planning use case): hypernetwork is almost-free at inference, so the
  inference-FLOPs column flips the ranking relative to training-FLOPs.
  That is itself a chapter finding worth highlighting.

### Q2. Single budget or sweep?

Anchor at ~0.78% (the AVID number) as a single matched budget, or sweep
across {low, mid, high}, e.g. {0.1%, ~0.8%, 3%}.

- **Single anchor.** 4–5 families × ≥3 seeds = 12–15 runs. Cheap. Tells
  us the ranking at one point. Vulnerable to the falsification condition
  "ranking flips across budgets" because we can't see whether it would.
- **Sweep.** 4–5 families × 3 budgets × ≥3 seeds = 36–45 runs. Tells us
  whether the ranking is budget-dependent. Stronger contribution.

The sweep is the honest answer; the question is whether it fits the
GPU-hour budget.

**Chosen (2026-05-22): full FLOPs sweep, 3 budgets x 4-5 families x 3
seeds = 36-45 runs.** Consistent with the D1+D2 framing (Q3) and the
FLOPs-matched primary axis (Q1). The methodology chapter needs the
ranking-robustness claim. Budget grid TBD when the FLOPs estimator
(Q1 follow-up) is in place; tentative anchor at AVID-shortcut
training-FLOPs/step, with one lower and one higher budget bracketing it.

Prerequisite (gating the run-ticket move from `blocked` to `open`):
the per-family `flops_per_step(input_shape)` estimator must land in the
framework so the three budget cells are computable and the protocol is
pre-registrable.

### Q3. Where does this study live in the thesis?

Options:

- **D1 only.** Framing: "the framework supports an apples-to-apples
  comparison; here's what that comparison shows." Methodology piece.
- **D2 only.** Framing: "the headline empirical result on adapter
  families for action-conditioned dynamics is X." Result piece.
- **D1 + D2 split.** Methodology in D1's framework chapter; the
  action-conditioned instance + result in D2.
- **D2 + D4 reach.** Could fold the shortcut-on variant into D4 as a
  follow-up table.

This affects how the study is framed in the chapter and how much space
it gets, but not how it's run.

**Chosen (2026-05-22): D1 + D2 split.** The protocol lives in the D1
framework chapter as methodology; the action-conditioned instance and
its headline numbers live in D2. Consequence: the protocol has to be
defensible as methodology on its own merits, which raises the bar on
Q2 (a single anchor is harder to justify in a methodology chapter) and
on Q4 (the obvious-reviewer-baseline row is harder to omit).

### Q4. Include full fine-tuning of a parameter-matched subset as a row?

Full fine-tuning unfreezes the base, which conflicts with the thesis's
core "frozen base" stance ([[../../10_now/positioning#anti-positioning]]).
But "unfreeze a parameter-matched subset of the base" is the obvious
trivial baseline a reviewer will ask for. Options:

- **Include.** Honest baseline, blunts the obvious reviewer objection.
  Costs a row of the sweep.
- **Exclude.** Cleaner story alignment with the anti-positioning. Cite
  the choice explicitly so it doesn't read like avoidance.
- **Include as a single point only**, not across budgets. Compromise:
  one cell as a sanity baseline, not a full row.

**Chosen (2026-05-22): include as a single FLOPs point only, at the
anchor budget. +3 runs (1 budget x 3 seeds).** Reframed under FLOPs
matching: "unfreeze the cheapest-to-update subset of the base such that
training FLOPs match the adapter-row anchor cell." Blunts the obvious
reviewer baseline while keeping the visual weight of the table on the
adapter families. Methods paragraph commits explicitly to "reported for
completeness; the thesis position remains frozen-base" so the row does
not read as a hedge.

## Decision

**Decided 2026-05-22.** Four-part decision:

1. **Q3 (chapter placement): D1 + D2 split.** Protocol + ranking
   methodology in the D1 framework chapter; action-conditioned instance
   and headline numbers in D2.
2. **Q1 (match axis): training-time FLOPs primary.** Trainable params,
   inference FLOPs, and peak memory reported per row but not matched.
3. **Q2 (sweep shape): full sweep, 3 FLOPs budgets x 4-5 families x 3
   seeds = 36-45 runs.** Tentative anchor at AVID-shortcut training
   FLOPs/step (computed via the new estimator, not the 0.78% param
   figure). One lower and one higher budget bracket the anchor.
4. **Q4 (full-FT row): include as single FLOPs point at the anchor
   budget only.** +3 runs. Framed as a sanity baseline.

Total run count: **~39-48** (36-45 adapter cells + 3 full-FT cells).

## Consequences

### Derived tickets

- [[../../20_Tickets/feat-adapter-flops-per-step-estimator]] — **prereq.**
  Per-family `flops_per_step(input_shape)` estimator. Required before
  the sweep grid is computable and the protocol is pre-registrable.
  Gates the run-ticket move from `blocked` → `open`.
- [[../../20_Tickets/writeup-writing-preregistered-protocol-paragraph]]
  — pre-registered methods paragraph for the D1 chapter. The
  "what-we-commit-to-before-running" block has to land before any run
  finishes, or the methodology contribution is post-hoc.
- Update to [[../../20_Tickets/exp-adapter-param-matched-comparison]]:
  reflects FLOPs-matched sweep grid (4-5 families x 3 budgets x 3 seeds
  + 1 full-FT cell x 3 seeds), keeps `blocked` status, points blocker
  to the estimator ticket.

### Downstream

- Live config trainable-param verification (the old "verify 0.78%"
  consequence) is **downgraded** — under FLOPs matching, this number is
  a *reported* secondary column, not the matched axis. Still useful for
  the chapter table but no longer gating.
- Inference-FLOPs reporting becomes a chapter finding in its own right
  (hypernet is almost-free at inference; ranking by inference-FLOPs is
  expected to differ from ranking by training-FLOPs).
- Frozen-base anti-positioning ([[../../10_now/positioning#What this thesis is not]])
  remains intact: the full-FT row is single-point, framed as baseline.

## Recommendation (historical, kept for record)

Not strong enough to fill in. The trade-offs above are real and the
user's framing of the chapter (D1 emphasis vs. D2 emphasis) matters
heavily for the right answer. _Closed via grilling on 2026-05-22._

## Related

- Protocol: [[../../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
- Run ticket: [[../../20_Tickets/exp-adapter-param-matched-comparison]]
- Inbox origin: [[../../00_Inbox/2026-05-21]] (16:17)
- Positioning, anti-positioning: [[../../10_now/positioning#What this thesis is not]]
- Adapter family notes: [[../../30_Knowledge/related-work/avid]],
  [[../../30_Knowledge/related-work/hyperalign]],
  [[../../30_Knowledge/related-work/unicon]]
