---
type: decision
status: decided
created: 2026-08-04
decided_at: 2026-08-04
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

## Pre-registration — write this down before any level lands

**Timestamped 2026-08-04, before L1, L2 or L3 have been trained.** Its value
depends entirely on preceding the results: pre-registered, three cells
discriminate a theory-derived prediction; written afterwards, it is a
post-hoc rationalisation of whatever happened, and the difference is
immediately visible to a reader.

### The primary account: gradient budget, not parameter budget

> **H-E (entanglement).** A single adapter trained on both objectives
> allocates a shared gradient budget between them. Actions explain ~0.45 %
> of the teacher-forced denoising loss and are therefore already outbid by
> appearance correction; a consistency objective adds a second and larger
> claim on the same budget. **Predicted: L1 shows lower action-structure
> than a matched conditioning-only control on the same base and data, while
> L2 and L3 — where the two adaptations occupy separate parameter sets —
> do not.**

Note this is about **allocation**, not about size. That distinction is what
keeps it consistent with the D2 findings (below).

### Both branches, stated in advance

| Outcome | Reading | Status as a result |
|---|---|---|
| **L1 below its control; L2/L3 at control** | H-E supported. Recommendation: *separate acceleration from conditioning* | ✅ the predicted result |
| **L1 at its control** | H-E is wrong at this scale; entangling is affordable and separability is unnecessary machinery | ✅ **equally a result** — and the better outcome for D4 |
| **All three below control** | The cost is acceleration itself, not entanglement; the recommendation becomes a cost statement about few-step adaptation in general | ✅ a result |
| **L1 below, L2/L3 also below but less** | Partial support; report the ordering, not a binary | ✅ |

**Do not pre-commit the prose to failure.** If L1 matches its control, D4 is
delivered and the thesis has its punchline; §5.x must be written so it can
hold that outcome. A section that can only report a negative is a structure
that cannot accommodate success.

### ⚠ The capacity account is *secondary* and must be scoped

The obvious alternative explanation — *"the adapter was too small to learn
both"* — **collides with our own H4**, which is recorded as **killed**: a
structurally clean 7.5M adapter settled *below* the DiT-clone arms
(~0.0025 vs 0.008–0.011), so parameter count was tested as an axis and
eliminated as the D2 explanation
([[../../30_Knowledge/writing/ablation-axes]] H4). Reinstating it at D4
without scoping invites the obvious objection: *you ruled this out in the
ablation; why is it back?*

The defensible scoped form is different from the killed one and must be
written as such:

> **H-C (scoped capacity).** Capacity sufficient for one objective may be
> insufficient for two. This is a claim about the *joint* task, not about
> action conditioning alone, and it is therefore not the hypothesis H4
> eliminated.

**H-C is testable and must be tested rather than asserted**, or it becomes
an explanation that absorbs any outcome — which is exactly the kind a
committee probes. The test already exists in the arm sizes: if capacity is
the binding constraint, the **47M** arm should recover what the **34.97M**
arm loses. If it does not, H-C is eliminated and H-E stands alone.

**Discriminator:** H-E predicts the deficit is *insensitive* to adapter size
and *sensitive* to whether the parameter sets are shared. H-C predicts the
reverse. L2 at fixed total parameter count separates them directly.

## First data on L3 (2026-08-05) — and what it does not yet settle

[[../../30_Knowledge/experiments/20260805-turbo-action-tokens-binned-to-latent-grid]]
— an action adapter on a frozen **4-step distilled Wan-Turbo** base is L3 by
construction: acceleration lives in the base, the adapter learns only
conditioning.

**Suggestive in the predicted direction.** `effect_vs_adapter` reaches
**0.18 → 0.31** on the distilled base, against **0.047** on the best
comparable non-distilled Wan arm — roughly 4×. H-E predicts exactly this:
where the two adaptations occupy separate parameter sets, conditioning is
not crowded out.

**But it is not yet evidence.** Three variables move at once — 100M vs
34.9M adapter, binned vs unbinned action tokens, distilled vs full base — so
it cannot be attributed to separability. **This is precisely what the
matched conditioning-only control exists to prevent**, and its absence here
is the first practical demonstration of why that requirement is
load-bearing rather than bureaucratic.

**And L3 is not a working cell as it stands.** The action changes the
prediction without changing it *correctly*: `eval_action_loss_gap` is ~0 at
all ten evals and `eval_action_cos` never leaves 0.9998. The adapter also
**hurts denoising at every eval** and overfits from step 1200. So L3
currently shows *the action reaching the adapter*, not *the adapter having
learned action-conditioned dynamics* — a distinction the level comparison
must preserve, or the axis will read as three cells that all "sort of work".

➜ **Consequence for the protocol below:** the per-level readout must include
an *accuracy* measure (loss gap / action error), not only an *effect*
measure. `effect_vs_adapter` alone would have scored L3 as the best level
while it learned no correct dynamics at all.

## What each level needs to be citable

Common protocol, so the three are comparable at all:

- [ ] **A matched conditioning-only control per level — the load-bearing
      requirement.** Without it "action-following degraded" has no referent,
      and the whole pre-registration above is unmeasurable. The control is
      the *same* adapter, *same* base, *same* data, *same* depth, with the
      consistency objective off. For L1 this is a config flag
      (`anchor_prob: 1.0` / consistency weight 0); for L2 and L3 it is the
      conditioning adapter without the acceleration stage.
- [ ] **The same action-structure readout** across levels (the structure
      triad, not `effect_rel` alone — it is monotone in gain).
- [ ] **An accuracy readout as well as an effect readout.** Added 2026-08-06
      after the first L3 data: the action reached the adapter
      (`effect_vs_adapter` 0.18–0.31, the best of any cell to date) while
      `action_loss_gap` stayed at ~0 across all ten evals — the adapter
      consumed the action **without learning correct dynamics**. An
      effect-only comparison would have ranked L3 first on a cell that
      learned nothing, which is the single most misleading outcome this
      protocol can produce.
- [ ] **Wall-clock and NFE** at matched quality, against an honest
      fast-sampler baseline.
- [ ] **Few-step quality** actually measured — currently *not measured* for
      the one shortcut cell that exists.
- [ ] Pre-registration of the predicted ordering **before** the runs land.

## ✅ DECISION — adopt-with-fallback (Lukas, 2026-08-04)

**Adopted.** The thesis spine is **efficiency of adapted world models**, with
three acceleration levels, and the ordering prediction is **pre-registered
as of 2026-08-04, before any level ran**.

*Adopt-with-fallback* specifically: the framing, the section skeletons and
the pre-registration are built **now**, because none of them depend on
outcomes. The chapter then states how many levels landed. A two-level
comparison still supports the entangled-versus-separable claim; a one-level
result reverts to the previous storyline with no rewrite, since the D2
material is unchanged either way.

Options not taken: *decline* (keep the diagnostic as the climax — lower
ceiling) and *adopt-unconditionally* (which would make the thesis hostage to
three outstanding runs).

## Consequences

**Thesis structure.** Ch5 becomes two axes rather than one deliverable list:
**conditioning** (the D2 campaign — injection pathway, scale calibration,
economics, the LoRA family comparison) and **acceleration** (L1/L2/L3), with
the interaction between them as the integration section. The storyline chain
is updated in [[../../30_Knowledge/writing/thesis-storyline]]; §§1–9 there
stand unchanged.

**Contribution framing.** *An extensible framework, and step-size /
distillation modelling through adapters for action-conditioned world models
— because rollouts are otherwise too slow to plan through.* D1 + D3 lead;
D2 supplies the conditions. Recorded in
[[../../30_Knowledge/writing/rubric/01-originality]].

**The economics bound changes role** — from a descriptive limit to a
**predictive** tool (it predicts the L1/L2/L3 ordering). This is the same
measured quantity in a materially stronger form.

**The action-free D3 isolation is retroactively a design decision**, not a
caution: it is the matched conditioning-free arm the comparison needs.

**Writing discipline.** No level is written as having worked until run ids
and checkpoints are in the vault (hard rules 7–8). Both branches of the
pre-registration are results; §5.x must be able to hold either.

**Schedule.** Three runs outstanding. This is a shape change made *before*
Ch5 prose exists, which is the cheap moment
([[../../30_Knowledge/writing/rubric/10-keeping-to-schedule]]).

## Derived tickets

- **L1 / D4 (entangled)** — [[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]] · tracked as A6/A7 in [[../../30_Knowledge/writing/open-experiments-for-thesis]]
- **L2 PDD / LoRA distil** — [[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]] · [[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]]
- **L3 distilled base** — [[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]
- **Matched conditioning-only control per level** — the load-bearing
  requirement; without it the pre-registration is unmeasurable
- **Common measurement protocol** across levels — ➜ **still to be written**
- **Honest fast-sampler baseline** — [[d3-positioning-vs-weaver-reflow]]
