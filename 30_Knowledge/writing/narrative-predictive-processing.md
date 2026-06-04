---
type: writing
status: living
last_updated: 2026-06-04
sources:
  - "[[../related-work/surfing-uncertainty]]"
  - "[[../../50_Decisions/open/multimodal-adapter-broadening]]"
---

# Story arc: predictive processing as the thesis's narrative frame

**Decision (2026-06-04).** The thesis's **introduction and overall narrative**
are framed around **predictive processing** (Andy Clark, *Surfing Uncertainty*
— [[../related-work/surfing-uncertainty]]). The user: *"I would use it as
motivation, intro and narrative. It is a theory that we can get inspiration
from and build our framing around it. Would be a great narrative."*

## What this commits to

- PP is the **framing spine**: open the thesis with the "predictive brain"
  picture — a generative model that predicts multiple sensory modalities
  forward over a trajectory, with action as part of the same predictive
  machinery — and cast the technical work as an artificial instance of that
  idea. Return to it in the discussion as interpretation.
- It is used as **inspiration / motivation / narrative**, explicitly **not as
  empirical justification**. The technical contribution stands on its own ML
  terms; nothing in the results depends on PP being the correct theory of the
  brain. (Hard rules 7–8: motivation ≠ evidence.)

## Guardrails (so the frame doesn't backfire)

- **Quarantine the active-inference objection.** Rescorla (NDPR) argues active
  inference "presupposes" the expected trajectory rather than explaining it
  (Bernstein's redundancy problem). Pre-empt this in one place; don't let the
  thesis's correctness appear to hinge on the active-inference account.
- **Don't over-attribute the *multimodal* claim to Clark.** The
  forward-in-time-over-trajectories claim is explicit in the source (pp.
  111/158/184); coupling *distinct* modalities is partly our extension. Phrase
  accordingly.
- **Flag the framing as inspiration, not derivation.** PP supplies no
  architecture; the adapter design comes from the ML lineage (AVID, UWM, etc.).

## Coherence note — PP tilts the headline toward multimodal

The PP narrative maps **strongly onto the multimodal-coupled-dynamics
direction** (predict many modalities forward + act) and **weakly onto the
shortcut/few-step direction** (step-distillation has no natural PP reading).
The thesis headline is still deliberately open
([[../../50_Decisions/open/multimodal-adapter-broadening]]), but adopting PP as
the spine is a **soft lean toward multimodal becoming the story**. If shortcut
ends up the winning result instead, the PP frame would need a different bridge
(or demote to a minor motif). Not a blocker — just keep the framing and the
eventual headline from silently diverging.

## Where it goes in the draft

- `70_Thesis/draft/10-introduction.md` — PP opening + motivation.
- `70_Thesis/draft/60-discussion.md` — PP as interpretive lens + the
  active-inference caveat.
- Not yet written in; flagged for `/thesis-write` when the intro is drafted.

## Related

- [[../related-work/surfing-uncertainty]] — the source + sourced quotes.
- [[../../50_Decisions/open/multimodal-adapter-broadening]] — the direction it
  most naturally motivates.
- [[../../10_now/positioning]] — anti-positioning ("not a control/RL paper")
  may need revisiting if the PP-action framing makes control first-class.
