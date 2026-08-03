---
type: decision
status: open
created: 2026-08-01
target_date: 2026-08-08
deliverable: D3
related: ["[[../../30_Knowledge/related-work/weaver]]", "[[../../30_Knowledge/writing/thesis-storyline]]", "[[../../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]]"]
---

# Open: how do we position D3/D4 now that WEAVER-ReFlow exists?

## The situation

[[../../30_Knowledge/related-work/weaver]] (found 2026-08-01) is an
action-conditioned robot world model trained with **rectified flow** on
**per-frame SD3 latents**, with a released **ReFlow-distilled few-step**
checkpoint, MIT-licensed, weights public. Verified from the repo README; no
paper yet.

That is the destination D4 describes: *fast, action-conditioned trajectory
prediction usable for planning*. It arrived while our D3 still has **no clean
positive result** ([[../../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]]).

**This is a positioning question, not a panic.** The mechanism differs in two
ways that look defensible: we adapt a **frozen** pretrained base with a small
plug-and-play module, and our few-step mechanism is a **step-size-conditioned**
adapter trained with self-consistency, not a fixed-budget distilled student.
But the framing has to change from "we build a fast action-conditioned world
model" to something that names the mechanism.

## Options

**A. Sharpen to the frozen-base / adapter axis.** The contribution becomes
"*step-size conditioning can be added to a frozen pretrained base by an
adapter*" — nobody's shown that, and it's what D1 uniquely enables. WEAVER
becomes related work and possibly a skyline reference. Cheapest; consistent
with the existing anti-positioning.

**B. Add WEAVER as an explicit baseline.** Strongest defence, real cost: a
different codebase, dataset (DROID), and action interface. Only worth it if
the D3 arms produce a positive result worth defending.

**C. Lean into the any-`d` property.** A shortcut adapter serves *all* step
budgets from one set of weights; a distilled student serves the budget it was
distilled for. If that holds, it is a clean, cheap, defensible axis — and the
already-running D3 sweep over N ∈ {1,2,4,8,25,50} is exactly the evidence.
**Requires verifying whether WEAVER's ReFlow student is step-size conditioned**
— currently unknown and the single most important open fact.

## Prerequisite — ANSWERED 2026-08-02, and it favours us

Both facts were read out of the WEAVER repo overnight:

- **(i) The dynamics transformer is trained from scratch** and is *already*
  action-conditioned (8-D joint deltas as a per-frame token). So WEAVER is not
  an adapter-on-a-frozen-base result at all — **D1 and D2 are untouched by it.**
- **(ii) `WEAVER-ReFlow` is a plain 2-rectification ReFlow — NOT step-size
  conditioned.** It is a fixed-budget student, not one set of weights serving
  any `d`.

**✅ VERIFIED AT THE SOURCE 2026-08-02** — `scripts/reflow.sh` shows
`rectified_teacher_steps=50`, `rectified_student_rollout_steps=4`,
`val_steps=4`, and **no** self-consistency or shortcut objective. A 50-step
teacher distilled into a **fixed 4-step** student. This is citable.

**Consequence: the shortcut contribution is NOT scooped, and Option C is
live.** The any-`d` axis is exactly the axis on which WEAVER does not compete.

**Recommendation: A + C.** Sharpen to the frozen-base/adapter axis (A) and make
the any-`d` property explicit (C); cite WEAVER as related work and a skyline
reference. **Skip B** — a DROID-based baseline is a large cost for a defence
against a system that turns out not to overlap on mechanism.

## Consequences either way

- `70_Thesis/` related-work must cite WEAVER; the D3/D4 framing in
  [[../../30_Knowledge/writing/thesis-storyline]] needs a pass.
- Raise it with the advisor at the next meeting — a released system on the
  contribution surface is exactly the kind of thing they should hear from us
  first rather than at the defence.
