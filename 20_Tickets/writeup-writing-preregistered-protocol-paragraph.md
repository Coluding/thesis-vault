---
type: writeup
scope: writing
status: open
priority: medium
created: 2026-05-22
updated: 2026-05-22
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[experiments/exp-adapter-param-matched-comparison]]"
  - "[[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]"
---

# Pre-registered protocol paragraph for the param-matched comparison

Draft the "what we commit to before running" paragraph for the D1
framework chapter's methodology section, covering the FLOPs-matched
cross-family adapter comparison.

## Why

The decision
[[../50_Decisions/decided/param-matched-adapter-comparison-definition]]
puts the protocol in the **D1 framework chapter as methodology**. The
methodology contribution is "here's the fair-comparison protocol; here's
what we commit to before running." That commitment has to be written
*before* any cell of the sweep finishes, or the contribution reads as
post-hoc rationalisation.

## What goes in the paragraph

The protocol's commitment block: family list, matched axis, sweep
budgets (set after the FLOPs estimator lands), seed count, primary
metric, success criterion, falsification condition, decision rule.
Anti-positioning sentence about why full-FT is a single sanity point,
not a row. Inference-FLOPs reporting framed as a separable chapter
finding.

Draft lives at `30_Knowledge/writing/draft-preregistered-protocol-paragraph.md`
once written; final source-of-truth in the chapter draft.

## Definition of done

- Paragraph drafted (~250-400 words, one tight block).
- Reviewed against
  [[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
  to confirm every committed claim has protocol-side machinery.
- Pre-registration date stamped (so post-hoc additions are detectable).
- Cited from the run-ticket
  [[experiments/exp-adapter-param-matched-comparison]] so the launch trigger
  references it.

## Not in scope

- Results/discussion paragraphs for the same section — those land after
  the sweep finishes.
- Methodology paragraphs covering other sections of D1 — separate
  tickets if needed.

## Related

- Decision: [[../50_Decisions/decided/param-matched-adapter-comparison-definition]]
- Protocol: [[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
- Run ticket: [[experiments/exp-adapter-param-matched-comparison]]
