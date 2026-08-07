---
type: writing
status: living
last_updated: 2026-08-07
sources:
  - "[[scfm-shortcutting-pretrained-flow-models]]"
  - "[[lcm-lora]]"
  - "[[pdd-parallel-decoding-distillation]]"
  - "[[../../50_Decisions/open/d3-positioning-vs-weaver-reflow]]"
  - "[[../writing/open-experiments-for-thesis]]"
---

# The few-step baseline set — three tiers

> **Resolves the standing gap** flagged since 2026-07-26: the D3 comparison
> risks being against many-step sampling alone, which measures that few
> steps are faster than many steps. That is definitional, not a result.
> Source: literature scan 2026-08-07.

## Tier 1 — training-free, mandatory, nearly free

**DPM-Solver** ([[dpm-solver]], arXiv:2206.00927, NeurIPS 2022) and
**DPM-Solver++** (arXiv:2211.01095) on our frozen base at **matched NFE**
(4, 8, 16).

No training. Runs on the exact checkpoint we already have. This is the
honest floor:

> **If the adapter does not beat DPM-Solver++ at matched NFE, there is no
> few-step contribution.**

Non-negotiable, and skipping it is the single most likely reviewer
objection. It is also the cheapest item on the entire experiment queue.

## Tier 2 — the frozen-base neighbours, the real competition

Methods sharing our constraint (base untouched, acceleration in an adapter),
so a comparison isolates our actual claim rather than the architecture.

| Method | Why it belongs |
|---|---|
| **SCFM** (arXiv:2510.17858) | Frozen weights + LoRA, and it *explicitly declines* explicit step-size conditioning. The paper that most directly tests whether our differentiator buys anything. Reports ~1 A100-day, so it is affordable |
| **LCM-LoRA** (arXiv:2311.05556) | Few-step entirely in LoRA, plugs into unmodified checkpoints. **Fixed** step count, cannot trade NFE at inference, which is precisely our differentiator |
| ArcFlow (arXiv:2602.09014) | Secondary; lightweight adapters, <5% of parameters |

## Tier 3 — base-retraining methods, context only, clearly labelled

Shortcut models trained on our data; optionally progressive distillation or
DMD-style. These **violate the frozen constraint**, so they are an *upper
reference*, not a head-to-head baseline.

Report them anyway. *"Shortcut models reach X but require retraining the
prior; we reach Y with the prior frozen"* converts a limitation into a
quantified trade-off, and it is much stronger than an omission a reader
discovers.

## Two framing rules

**Report quality-versus-NFE curves, not single points.** The step-size
claim is specifically about *varying* the budget at inference. A single-NFE
table cannot show it, and LCM-LoRA structurally cannot produce the curve at
all. That curve is our best evidence and it is the comparison that separates
us from tier 2.

**Exclude adversarial and distribution-matching distillation explicitly**
(ADD, LADD, DMD, DMD2, SDXL-Lightning): they replace the base and need
adversarial training infrastructure. Give the citable reason rather than
omitting silently; the PDD abstract notes such losses are "notoriously hard
to optimize and suffer from mode collapse".

## Where this lands

- **Ch4 §4.4** metrics and baselines — the tier structure
- **Ch5 §5.7** — the quality-vs-NFE curve
- **§6.2** — tier 3 as the stated trade-off
- Resolves [[../../50_Decisions/open/d3-positioning-vs-weaver-reflow]]

## Cost note

Tier 1 is hours and no training. Tier 2 is the expensive part and SCFM is
the one worth paying for. Tier 3 can be cited from the papers rather than
run, provided the setting is comparable and that is stated.
