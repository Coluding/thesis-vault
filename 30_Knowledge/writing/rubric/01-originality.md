---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: originality
category: research
current_band: "7-8"
target_band: "8"
sources:
  - "[[_index]]"
  - "[[../thesis-storyline]]"
  - "[[../../../10_now/positioning]]"
  - "[[../../experiments/20260802-avid-wan-cleanroom-perframe-causal]]"
  - "[[../../experiments/20260802-shortcut-works-on-flow-not-diffusion]]"
---

# Rubric 1 — Originality of the research

## The rows

| | |
|---|---|
| 10 | Daring and high-risk; steers the sub-field. Top paper at top conference |
| 9 | High originality and thought-provoking. Publishable at top tier |
| **8** | **New and original. The problem is tackled from a fresh perspective. Publishable at second tier** |
| 7 | Original elements, as part of a larger existing framework. Workshop at best |
| 6 | A variation of existing work, applying an existing approach to another problem |

## What it actually asks

**Not** whether the method worked. The 6-row and the 8-row differ by
*perspective*, not by outcome: "applying an existing approach to another
problem" (6) versus "the problem is tackled from a fresh perspective" (8).

The discriminating question a committee asks is: **after reading this,
does the reader know something they could not have predicted, that
transfers beyond this thesis?**

## ⭐ The frame this rests on (corrected 2026-08-03)

**The thesis has a positive result, not only a diagnosis.** On
DynamiCrafter × ACWM the adapter is a working action conditioner — 3.9× the
AVID reference, all three structure probes above chance on held-out data,
and a *lower* denoising loss than the untreated control. Wan is the matched
negative: 6/6 quality metrics beaten with structure at chance.

That combination — **a positive cell, a negative cell, and a causal
architectural variable that discriminates them** — is a materially stronger
originality position than "we measured why it fails". Structure the
contributions in that order.

## ⭐⭐ The efficiency axis — the largest available move on this item

**Adopted 2026-08-04, decided:
[[../../../50_Decisions/decided/efficiency-axis-as-thesis-spine]].**

Expand the single "shortcut" node into **three acceleration levels**, and
make *where few-step behaviour should live* the thesis question:

| Level | Speed lives in | Entanglement |
|---|---|---|
| **L1** shortcut adapter | the adapter | **one adapter learns actions *and* step size** |
| **L2** PDD / LoRA distil | a second adapter on the base | two adapters, sequential |
| **L3** distilled base | the base | acceleration free; only conditioning learned |

This mirrors the D1 taxonomy — *where does trainable capacity sit relative
to the frozen prior?* — asked of the **speed** axis rather than the
**conditioning** axis. The symmetry is what makes it a chapter.

**Why it moves this item more than anything else available.** The current
climax is a *diagnostic* (pathway + economics), and diagnostics cap near the
8-row: they explain, they do not answer a design question. Three tested
placements with a recommendation is what "the problem is tackled from a
fresh perspective" describes.

**And the economics bound *predicts the ordering* before the runs.** Actions
are worth ~0.45 % of the loss, so they are already outbid; adding a
consistency objective to the *same* adapter adds a second, larger claim on
the same gradient budget. Predicted: **L1 degrades action-following, L2/L3
preserve it.** A theory-derived prediction discriminated by three cells is a
stronger structure than any shoot-out, and it reaches toward the 9-row
("thought-provoking") rather than merely securing the 8.

It also converts the economics bound from a *limit* into a *predictive
tool* — the same result in a materially stronger form
([[04-knowledge-of-domain]]) — and retroactively makes the action-free D3
isolation a design decision rather than a caution.

⚠ **Three runs are outstanding** (L1 has never been trained; L2 and L3 are
ticketed but unrun), and a claimed result exists only in conversation.
Build the framing and the skeletons now — they cost nothing and do not
depend on outcomes — but **no level is written as having worked** until run
ids are in the vault. If the prediction fails, the framing survives and the
recommendation inverts; that is the property a good axis should have.

## ⭐ The contribution spine (Lukas, 2026-08-04): **efficient rollouts**

The positive, forward-facing statement of what the thesis contributes —
distinct from the *study* framing above, which describes the method:

> **An extensible framework, and step-size / distillation modelling through
> adapters for action-conditioned world models — because rollouts are
> otherwise too slow to plan through.**

This is the better spine for §1.4, and it reorders the deliverables: **D1 +
D3 lead, D2 supplies the conditions.** Rollout cost is the motivating
problem, action conditioning is the setting, and the adapter-borne
few-step behaviour is the contribution. It also makes the D2 mechanism
campaign *load-bearing rather than incidental* — you cannot build a fast
action-conditioned rollout without knowing which injection pathway survives.

**Three routes to the efficient rollout are in scope**, all ticketed:

| Route | Ticket |
|---|---|
| Shortcut adapter (step-size conditioning on a frozen base) | [[../../../20_Tickets/experiments/exp-shortcut-action-free-isolation]] |
| **PDD / LoRA distillation** of the base, action adapter on top | [[../../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]] · [[../../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]] |
| Action adapter on an already-distilled base | [[../../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]] |

### ⚠ What is and is not established on Wan

The tempting claim is *"we trained an action-sensitive in-domain shortcut
adapter on Wan"*. **That is D4, and it has not been run.** The two halves
are validated **separately, on the same backbone**:

| | Cell | Actions | Shortcut |
|---|---|---|---|
| S-w | `..._shortcut_actionfree_robotarm.yaml` | ❌ **stripped by design** | ✅ learns it, 9× vs matched control |
| W-a | clean-room per-frame AdaLN | ✅ 2.49×, structured | ❌ none |

The D3 arm is action-free *deliberately*, so that "does shortcut work" is
not confounded with "does action conditioning work"
([[../thesis-storyline]] §6). Writing the combination as done would collapse
that isolation and is checkable against our own config names.

**But the situation is far better than the vault records.** D4 was descoped
as "gated on a working D2 cell". Both halves now work on Wan independently,
so D4 is **no longer gated — it is simply unrun**, and it is the highest-value
remaining experiment for this item: it is the thesis's punchline, both
components are de-risked, and the ticket has existed since June
([[../../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]]).
➜ tracked as **A6** in [[../open-experiments-for-thesis]].

### The diffusion side

We show **that** the diffusion cell does not learn the consistency relation
(0.084 vs matched control 0.083, CIs coincident; gain exploding to 4e4 at
large `d`), and we **derive** why the published velocity-averaging target is
biased on a curved VP arc. Phrase these as two results — a measurement and a
derivation — rather than as one causal claim, since the DC arm ran
`endpoint_inversion`; A4 closes the gap between them config-only.

## Our answer to that question

Three candidates, ranked by how well they survive the test.

> ### 🛑 RESTATED 2026-08-07 — the conditioning literature scan
>
> Full verdict: [[../../related-work/pathway-claim-prior-art-verdict]].
>
> **The ranking is not novel.** ADM, DiT, and two 2026 preprints on *action
> conditioning specifically* (IOI arXiv:2606.23296; Nano World Models
> arXiv:2605.23993) already report modulation beating cross-attention. Do
> not write it as a finding.
>
> **The frozen-backbone version is nearly claimed.** Decoupled Action Expert
> (arXiv:2511.12101) ran it: cross-attention 61.3 → **19.8 %** under
> freezing, AdaLN 62.0 → **62.3 %**. Same conclusion, different explanation
> (projection staleness). **Cite it.** Their own additive row (57.0 → 45.3)
> bypasses backbone projections and still degrades, which their account does
> not explain and ours does. That is the opening.
>
> **One claim is refutable as loosely stated.** Motif-Video 2B
> (arXiv:2604.16503) measures the same ratio and gets 7.6 % mean / 5.2 %
> weakest block, concluding cross-attention is *not* a residual no-op. Ours
> is ~0.3–0.6 %. So "the cross-attention write is negligible" is false in
> general; the claim must be conditional on **frozen backbone +
> from-scratch adapter + low-dimensional signal**.
>
> **What is ours:** (i) the measured ratio *in that setting*, stated against
> Motif's numbers; (ii) **the localisation** — the signal survives all ten
> blocks at 44–56 % action-driven and dies in one residual addition, which
> is sharper than anything in the scan; (iii) the design rule, **scoped to
> low-dimensional spatially-global signals** or GenTron and DexAC-WM refute
> it.
>
> ➜ Rewrite the block below to lead with the mechanism and the localisation,
> not the ranking.

**(1) The injection-pathway principle — the strongest.**

> Conditioning must arrive **scale-free relative to the residual stream**.
> Per-frame modulation of *normalised* activations (AdaLN) carries an
> action signal; the same per-frame action supplied as **cross-attention
> tokens** does not — however well scaled (`action_token_norm`) and however
> well aligned to the latent grid (`action_seq_len`). It survives the
> attention and drowns at the residual add, because AdaLN multiplies a
> normalised activation and is scale-free by construction while
> cross-attention adds into an unnormalised stream.

Architecture-level, mechanism-level, and it transfers to *any* conditioning
problem on a DiT. Evidence: single-variable clean-room A/B with matched
adapter contribution and matched mask
([[../../experiments/20260802-avid-wan-cleanroom-perframe-causal]]),
corroborated by the depth trace that localises the loss
([[../../experiments/20260731-wan-action-trace-value-pathway-drowns]]).

**(2) The economics bound — the most honest, least architectural.**
Actions explain ~0.45% of a teacher-forced denoising loss, so appearance
correction always pays more per unit of gradient. This is *not* fixable by
architecture, and naming it bounds the whole method class. Originality here
is in the framing: a limit derived from the objective rather than observed
from a failure.

> 📌 **Open (Lukas): argue this with variance instead of loss share.**
> "Actions explain ~0.45 % of the loss" is a *share-of-objective* statement,
> and it invites the obvious objection that a small share can still be
> decisive — a rudder is a small fraction of a ship. The stronger form is a
> **variance decomposition**: how much of the *predictable* future variance
> is action-determined versus appearance-determined. That is measurable, and
> the **IDM ceiling** (B5 in [[../open-experiments-for-thesis]] — minutes of
> GPU, `(z_t, z_{t+1}) → a_t` on ground-truth transitions) gives the
> numerator directly. **Decide before §5.5 is written**; if the variance
> version holds it should replace the loss-share phrasing everywhere, since
> it is the version that survives the objection.

> ### 🛑 NARROWED 2026-08-07 by the literature scan
>
> **The architecture is taken.** [[../../related-work/scfm-shortcutting-pretrained-flow-models]]
> (arXiv:2510.17858) already does frozen-base LoRA shortcutting of flow
> models: `θ = θ₀ + Δθ`, base frozen, acceleration in the adapter.
> [[../../related-work/lcm-lora]] is a second precedent. So *"few-step
> behaviour carried entirely by the adapter while the prior stays frozen"*
> is **no longer an open claim** and citing it as one would not survive
> review.
>
> **What survives is narrower and sharper.** SCFM explicitly *declines*
> explicit step-size conditioning: "our method does not rely on an explicit
> step-size parameter". The open claim is therefore **explicit step-size
> conditioning in an adapter on a frozen base**, with a testable
> consequence: an explicitly conditioned adapter can trade NFE at inference
> and an implicitly trained one cannot. **The evidence for it is a
> quality-versus-NFE curve, not a single-NFE number** — LCM-LoRA
> structurally cannot produce that curve.
>
> **The curvature analysis also has prior art.**
> [[../../related-work/homo-high-order-shortcut-matching]]
> (arXiv:2502.00688) already argues shortcut models degrade in
> high-curvature regions and that a velocity-only treatment misses the
> geometry. Position our derivation as an **independent and more precise**
> statement of a known limitation; the sagitta form, the fixed-point
> argument and the endpoint-inversion escape are the additions. Do not claim
> the finding as novel.
>
> ➜ Rewrite the paragraph below accordingly; it is kept for its structure.

**(2b) Shortcut modelling *through an adapter* — the only uncovered axis.**

Added 2026-08-03. Distinct from the others in kind: (1) and (2) sharpen the
conditions on AVID's contribution, whereas **step-size conditioning is
something AVID does not do at all** ([[../../../10_now/positioning]] D3:
"this one *is* an uncovered axis"). Consistency and shortcut models retrain
the prior from scratch to be few-step; here the prior stays **frozen** and
the few-step behaviour is carried entirely by the adapter. That combination
— frozen base, adapter-borne step-size conditioning — is new.

Two results support it, at different rungs:

- **Measured (rung 1):** the shortcut objective is *learnable in the adapter
  setting* on a flow base, against a matched control at matched depth —
  `consistency_cos` 0.302 [0.251, 0.356] vs 0.034 [0.026, 0.042], 9× with
  non-overlapping CIs; and a second independent signal in the gain profile
  (flat O(1) for the treated arm, collapse for the control). Clip-nulls are
  near-identical across the pair, so it is not a genericity artefact, and
  the base-null is exactly 0.
  ⚠ The **cross-base** comparison is confounded (arms differ in consistency
  target *and* depth) — the within-arm results are what stand.
- **Measured, second independent statistic (rung 1):** the **gain ladder**
  `|dpred|/|dtarget|` per rung — Wan treated is **flat and O(1)** across the
  whole dyadic ladder (0.440 → 0.334), the matched Wan control **collapses**
  (0.483 → 0.026), and DC **explodes 4–5 orders of magnitude** (0.973 →
  40950) at exactly the large `d` that few-step rollout uses. This is the
  *magnitude* claim — the response is the right size for the jump being
  asked for — and it matters precisely because it is **not** a cosine: it is
  the scale the consistency cosine normalises away. Two statistics, each
  with its own matched control, same conclusion.

  ⚠ **`spearman vs ladder = +1.0000` is not this result.** It is +1.0000 for
  the *control* and for DC as well, and sits under *Nulls and controls* — a
  monotonicity sanity check that the ladder is ordered correctly. It cannot
  discriminate learned structure. Do not cite it as evidence of learning.
- **Qualitative, not measured (rung 5→pending):** few-step rollout
  *quality* is reported as improving over the no-shortcut control but **not
  competitive**; the experiment note records few-step quality as **not
  measured** (budget exhausted). Write this as *"directionally better,
  quality not measured, scoped to future work"* — **not** as a quality
  result. A measured mechanism plus an unmeasured quality claim in the same
  paragraph would put the whole D3 contribution at rung 3.

**The curvature argument is part of the contribution and stays**
([[04-knowledge-of-domain]]). It rests on its own two legs, both solid: the
**sagitta derivation** — eq. (4) is exact for a straight interpolant, biased
by ≈ κ·δ²/2 on a curved VP arc, and the true field is not a fixed point of
the averaging rule, so the bias compounds up the doubling tower — and
**numerical verification at zero model error**: the averaged target lands
5.1 / 16.1 / 24.1 % off at s = ¼, ½, ¾ where the endpoint target lands
0.000000. It is *proven*, not observed, and it fits the storyline's chain as
the node that motivates the flow pivot.

The **A4 2×2** (`v_average` vs `endpoint_inversion`, one base, one depth,
config-only) would convert theory into empirical confirmation cheaply —
see [[../open-experiments-for-thesis]].

**Framing rule:** present D3 as an *uncovered axis explored and partially
answered*, with the remaining work named. An honest "learnable, mechanism
measured, quality left to future work" is a stronger originality position
than an inflated few-step claim that the evidence cannot carry.

**(3) The metric-blindness result.** Loss, gate, FID, sample quality are
all blind to whether a conditioned model uses its conditioning — with the
sharpest possible demonstration: the Wan × ACWM cell improves on 6/6
perceptual metrics (FVD −64%) while all three structure probes sit at
chance, and the DC cell clears chance on all three with **no quality
metrics logged at all**. The same adapter design is a domain adapter on one
backbone and an action conditioner on the other. Scores mostly under
[[05-reflection]], but contributes here too.

## Where the band comes from

**8 is reachable; 9 is not.** Nothing here steers the sub-field — it
sharpens a design rule inside an existing paradigm (adapter-based world
models, post-AVID). Claiming 9-level novelty would be the wrong move
anyway: overclaiming is punished harder than the honest 8.

What *does* secure the 8 rather than the 7 is the pairing: the 7-row is
"original elements, as part of a larger existing framework", which is what a
pure mechanism study of a failure would be. A positive cell plus a matched
architectural negative plus the causal variable is a **finding**, and
findings are what the 8-row's "tackled from a fresh perspective" describes.

## The risk that drops this to 6

If the thesis reads as *"we applied AVID to Wan"*, the 6-row applies
literally: "a variation of existing work, by applying an existing approach
to another problem." This is a **real** risk because it is a fair
description of the first three months of work.

⚠ **The opposite risk is now live too.** With a working cell in hand, the
temptation is to present the thesis as a systems contribution — *"we built
an action-conditioned world model"*. That invites comparison against
AVID on AVID's own terms, where our delta is thin, and it collides with the
limits (no quality metrics on the DC cell, rollout control untested,
pre-convergence cancellation). **The working cell is evidence for the
pathway claim, not the headline in itself.**

The defence is structural, and must be **explicit in §1.4 rather than
implied**:

- AVID is the **starting point**, and action conditioning on a frozen base
  is *AVID's* contribution, not ours
  ([[../../../10_now/positioning]] D2 correction).
- Our delta is: **the conditions under which it works** (the pathway
  principle), **the bound on why it stops** (the economics), **step-size
  conditioning through the adapter** (D3 — *the one axis AVID does not
  touch at all*; learnable on flow against a matched control, mechanism
  measured, few-step quality scoped to future work), and **the framework**
  (D1, demonstrated by porting AVID's own recipe to a new base family
  inside AVID's own repository).
- --> Our contribution is extendable repo and the shortcut / distillation modelling through adapters based on action world models since the rollouts are too slow otherwise!!

**Note the shape of the delta:** three of the four *sharpen or bound* AVID's
contribution; **D3 is the one that extends it into new territory.** If a
reader takes only one thing away as "new", it should be that — so it must
not be buried behind the D2 mechanism campaign in §1.4 merely because D2 has
more runs behind it.

## Optimisation queue

- [ ] **Q7 — Write the design-principle statement for §1.4**, with an
      explicit **scope of validity**: which architectures, which injection
      sites, what we did *not* test. A principle with stated limits reads
      as science; one without reads as overreach. *(~1 h, biggest single
      move on this item.)*
- [ ] **Frame the thesis question as "when and why", not "can we".** The
      2026-08-02 reframe already did this — make sure the title, abstract
      and §1.2 all carry it. Working title: *On the Adaptability of Video
      Foundation Models to World Models*.
- [ ] **Recover the D1 originality the LoRA run unlocks.** Axis 2 of
      [[../ablation-axes]] currently says LoRA "is not run", which forced
      D1 to be a *complexity-analysis* contribution only
      ([[../../../70_Thesis/outline]] §3.3: "cost only — no quality
      comparison"). With the LoRA-vs-adapter comparison running, D1 can
      carry an **empirical** family claim — a materially more original
      chapter. ➜ requires Q2 (rewrite the axes note) and a decision update.
- [ ] **Do not add a new contribution to recover originality.** CLAUDE.md
      Part 12: new contributions go through `50_Decisions/` first. The
      surface is set; sharpen it, don't extend it.

## Where it lands in the thesis

- §1.2 problem statement — "when and why", not "can we"
- §1.4 contributions — the four-part delta vs AVID, explicit
- Ch3 (adapters, D1) — the family claim, now empirical if the LoRA run lands
- The pathway principle: stated in §1.4, derived in Ch6/§6.x, generalised in §7.1

## Open questions

- Does the LoRA comparison change [[../../../50_Decisions/decided/param-matched-adapter-comparison-definition]]? If it enables a quality comparison the decided note ruled out, that decision needs an update rather than a silent override.
- Should the title change to the working title? It affects how the first reader frames originality before reading a word.
