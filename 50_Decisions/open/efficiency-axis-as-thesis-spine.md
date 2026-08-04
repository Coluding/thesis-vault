---
type: decision
status: open
created: 2026-08-04
decided_at:
updated: 2026-08-04
target_date:
scope: shortcut
related:
  - "[[../../30_Knowledge/writing/thesis-storyline]]"
  - "[[../../30_Knowledge/writing/rubric/01-originality]]"
  - "[[../../30_Knowledge/writing/evidence-map]]"
  - "[[../../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]]"
  - "[[../../30_Knowledge/experiments/20260802-rollout-wallclock-vs-steps]]"
  - "[[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]]"
  - "[[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]]"
  - "[[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]"
  - "[[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]]"
---

# Decision: make **efficiency of adapted world models** the thesis spine, with three acceleration levels

## Context

The storyline already routes through rollout cost — *DC + adapter works →
planning → too slow → flow → shortcut*. Until now "shortcut" was a single
node with one cell behind it, and the thesis climaxed in a **diagnostic**
(the injection-pathway principle plus the objective economics).

**Proposal (Lukas, 2026-08-04):** promote efficiency from a transition to
the thesis question, and expand the single shortcut node into **three
acceleration levels** — shortcut adapter, PDD/LoRA distillation, and an
already-distilled base. Experiments are planned for all three.

## The storyline this implies (barely changed)

```
DC + adapter works  →  AVID  →  too slow  →  flow models  →  rectified flow
     →  DISTILLATION TO FEW STEPS  (three axes)
     →  analysis: what worked, what did not, and why
```

Only two nodes move. "Shortcut" widens into three placements, and the
ending becomes a **comparative analysis with a recommendation** rather than
a diagnosis. The D2 mechanism campaign is not displaced — it lands in the
final node as the *conditioning* half of the analysis, which is where its
20+ runs already point.

## The three levels — a taxonomy, not three implementations

They differ in **where the few-step behaviour lives**, and therefore in
whether the two adaptations (conditioning, acceleration) are entangled.

| Level | Speed lives in | Base | Entanglement | Ticket |
|---|---|---|---|---|
| **L1 Shortcut adapter** | the adapter | frozen | **one adapter learns actions *and* step size** | [[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]] |
| **L2 PDD / LoRA distil** | a second adapter on the base | LoRA-modified | two adapters, sequential | [[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]] · [[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]] |
| **L3 Distilled base** | the base itself | externally distilled | acceleration is free; only conditioning is learned | [[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]] |

This mirrors the D1 adapter taxonomy — *where does the trainable capacity
sit relative to the frozen prior?* — asked of the **speed** axis instead of
the **conditioning** axis. That symmetry is what makes it a chapter rather
than a sweep.

## The prediction — and why this is the strong version

The framing is not a shoot-out. **The economics bound predicts the
ordering before any of the three runs:**

Actions explain ~0.45 % of a teacher-forced denoising loss, so the action
signal is already outbid by appearance correction. Adding a consistency
objective to the **same** adapter introduces a second, larger claim on the
same gradient budget. Therefore:

> **Predicted:** L1 (entangled) degrades action-following relative to a
> conditioning-only adapter; L2 and L3 (separable) preserve it. Separating
> acceleration from conditioning buys action-following back.

Consequences if it holds:
- A concrete, transferable design recommendation: *do not ask one adapter
  to learn both; separate acceleration from conditioning.*
- It retroactively justifies running D3 **action-free** — that isolation
  becomes a design decision rather than a caution
  ([[../../30_Knowledge/writing/thesis-storyline]] §6).
- It converts the economics bound from a *limit* into a *predictive* tool,
  which is a materially stronger form of the same result
  ([[../../30_Knowledge/writing/rubric/04-knowledge-of-domain]]).

**If it fails** — L1 works fine — the framing survives and the
recommendation simply inverts. Either outcome is a result, which is the
property a good axis should have.

## Why this improves the rubric position

| Item | Effect |
|---|---|
| **1 Originality** | Climax moves from *diagnostic* to *design question with three tested answers and a recommendation*. The 8-row's "tackled from a fresh perspective"; the theory-predicts-the-ordering structure reaches toward 9 |
| 3 Experimental evaluation | Three cells discriminating a **pre-registered prediction** — the strongest shape this item takes |
| 4 Knowledge | The economics bound becomes predictive rather than descriptive |
| 7 Organization | The two-axis structure (conditioning × acceleration) gives Ch5 a spine that is not "20 D2 runs then one D3 run" |

## Costs and risks

1. **This is a shape change, not a number change.** The schedule rule
   ([[../../30_Knowledge/writing/rubric/10-keeping-to-schedule]]) sends
   those to a decision note before drafting continues — hence this note.
   Deciding it *now*, before Ch5 prose exists, is cheap; deciding it after
   is not.
2. **Three runs are outstanding.** L1 has never been trained (D4); L2 and L3
   are ticketed but unrun. Schedule exposure is real.
3. **Rollout wall-clock is measured but the baseline is not honest yet** —
   the comparison must include DPM-Solver and consistency sampling, not only
   many-step sampling ([[d3-positioning-vs-weaver-reflow]]).
4. ⚠ **A claimed L-level result exists only in conversation.** Lukas reports
   data for the levels. Until run ids and checkpoints are in the vault, the
   framing may be built but **no level may be written as having worked**
   (hard rules 7–8). Section skeletons and the prediction cost nothing and
   do not depend on outcomes.
5. **DC is 2.8× faster per step than Wan because it is 1.4B vs 5B, not
   because of the objective.** Any cross-base speed statement must say so
   ([[../../30_Knowledge/experiments/20260802-rollout-wallclock-vs-steps]]).

## What each level needs to be citable

Common protocol, so the three are comparable at all:

- [ ] **A matched conditioning-only control** per level — otherwise
      "action-following degraded" has no referent.
- [ ] **The same action-structure readout** across levels (the structure
      triad, not `effect_rel` alone — it is monotone in gain).
- [ ] **Wall-clock and NFE** at matched quality, against an honest
      fast-sampler baseline.
- [ ] **Few-step quality** actually measured — currently *not measured* for
      the one shortcut cell that exists.
- [ ] Pre-registration of the predicted ordering **before** the runs land.

## The decision

**Adopt the efficiency spine with three levels, and pre-register the
predicted ordering?**

- **Adopt** — stronger Originality, a two-axis Ch5, and a prediction that
  makes three planned runs into a designed experiment rather than a sweep.
- **Decline** — keep efficiency as a transition and the diagnostic as the
  climax; lower ceiling, lower schedule risk.
- **Adopt-with-fallback (recommended)** — build the framing and the section
  skeletons now, pre-register the prediction, and state in the chapter how
  many levels landed. A two-level comparison still supports the
  entangled-versus-separable claim; a one-level result reverts to the
  current storyline with no rewrite, because the D2 material is unchanged
  either way.

➜ **Derived tickets on adoption:** the four above, plus a protocol note
fixing the common per-level measurement set.
