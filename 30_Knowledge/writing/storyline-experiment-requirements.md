---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-storyline]]"
  - "[[ablation-axes]]"
  - "[[../../10_now/positioning]]"
---

# Experiment requirements for the storyline

> **⚠ SUPERSEDED 2026-08-03 for status.** The live tracker of what is still
> needed is [[open-experiments-for-thesis]]. This note is kept for its
> 07-26 reasoning and tier derivation; its status table predates the 08-02
> results and is wrong in at least two places (it records D3 as having no
> clean positive, and rollout wall-clock as unmeasured — both have since
> landed). **Do not plan from the tables below.**

> **⚠ RE-SCORED 2026-08-01.** The table below (derived 07-26) is kept for its
> reasoning, but much of it is settled and the spine has changed. Read this
> section first; treat the older tiers as background.

## Current status against the corrected spine

Spine ([[thesis-storyline]] §chain): **DC works → planning → too slow → flow →
shortcut**, with Wan as the generality branch.

| spine link | evidence | status |
|---|---|---|
| **DC + adapter works** | arm E `condition_center` **0.106 = 3.6× the AVID reference**; mechanism (learned pedestal) measured; AVID-on-our-data control | ✅ **HAVE** |
| **Planning on it** | — | ❌ **MISSING — the one spine link with no evidence** |
| **Too slow** | cost arithmetic only; no measured wall-clock; no honest fast-sampler baseline | ⚠️ **ASSERTED, NOT MEASURED** |
| **Flow matching (κ=0)** | derivation + pivot decision; the per-rung prediction test is half-done | ⚠️ **theory strong, empirics partial** |
| **Shortcut / few-step (D3)** | June runs (poor samples), curvature signature **confounded** by the gate bug | ❌ **NO CLEAN POSITIVE RESULT** |
| **Wan branch (mechanism)** | full campaign: scale calibration, oracle 100:1, 0.45% economics, global-bag analysis, data-axis law | ✅ **COMPLETE — stop here** |

## What is actually still needed, in order

**T0 — the spine's missing link (blocks Ch5's headline demonstration)**

1. **IDM ceiling on DC latents** — `(z_t, z_{t+1}) → a_t` on ground-truth
   transitions. No world model, no new labels, minutes of GPU. Tells us whether
   action information is *present* in the data at all, and calibrates every
   number in the campaign. **Run first** — it can reframe §9 either way.
2. **Action recovery (inverse planning)** through DC arm E, with the three
   baselines (random · frozen action-free base · arm E). Ground truth is the
   true action sequence, so no reward model is needed.
   → [[../../20_Tickets/experiments/exp-eval-planning-through-dc-world-model]]
3. **Wall-clock per planning step** — falls out of (2) for free, and is the
   *quantitative* motivation for the entire back half of the thesis (currently
   asserted from arithmetic only; old **R5**).

**T1 — the D3 contribution needs one positive result**

4. **Few-step payoff on the working DC cell**: does the shortcut adapter
   reproduce the base's 50-step rollout in N ∈ {1,2,4,8} steps? Tooling exists
   ([[../../20_Tickets/experiments/exp-eval-shortcut-fewstep-videos]]) and has
   never been run on a cell that works. **D3 currently has no clean positive
   evidence** — this is the gap that most threatens the back half.
5. **Honest fast-sampler baseline** (DPM-Solver / consistency sampler at
   matched compute; old **R6**). Without it "diffusion rollout is slow" is a
   strawman an examiner will name immediately.

**T2 — repairs and strengthening**

6. **D3 curvature re-run with live gates** — the 68× flow-vs-diffusion result
   is confounded ([[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]])
   and cannot be cited until re-run.
7. **Per-rung v-averaging prediction test** (old **R7**) — converts the
   derivation into a validated prediction; the strongest purely-analytical node.
8. Resume DC arm E from step 3600 to pin peak-vs-plateau (cosmetic; the claim
   already holds).

**T3 — descope to future work if they do not land**

9. D4 combined cell (now unblocked in principle, since DC's D2 cell works).
10. Objective-level fix on Wan (action-CFG / rollout losses) —
    [[../../50_Decisions/open/wan-action-following-needs-objective-change]].

## What needs NO further experiments

The Wan mechanism branch. It is complete, internally consistent, and already
carries a methodological contribution (the probe suite + the negative results
that make the positive claim credible). Further Wan arms would improve numbers
inside a story that is already written.

---


What each node of [[thesis-storyline]] needs before the draft can assert it.
Derived 2026-07-26. **Requirement ≠ ticket** — the last column says whether a
ticket exists; several do not.

Status vocabulary: **have** (sourced run in the vault) · **partial** (run
exists but does not measure the required thing) · **missing** (no run) ·
**blocked** (cannot run until a prerequisite lands).

Priority tiers:
- **T0** — the storyline has no foundation without it. Do first.
- **T1** — a node is unsupported without it; the chapter cannot make its claim.
- **T2** — strengthens a claim that can already be made weakly.
- **T3** — descopable; state as future work if it does not land.

---

## Tier 0 — the foundation

| # | Requirement | Serves | Why the node fails without it | Status | Ticket |
|---|---|---|---|---|---|
| **R1** | **Action-sensitivity of the AVID/DynamiCrafter checkpoint.** Shuffled/zeroed-action gap (and ideally Action Error Ratio) on `pg3x72uc`. | Node 1 | Node 1 claims "AVID works." `pg3x72uc` shows convergence + healthy gate, **not action-use**. If the model is weakly action-sensitive, node 2 (planning) is impossible and the whole arc's starting point is unsupported. | **tooling built 2026-07-26, not yet run** | [[../../20_Tickets/experiments/exp-eval-action-sensitivity-avid-checkpoint]] |
| **R2** | **Finish the AVID native run.** `pg3x72uc` was `status: running` at ~800 steps. | Node 1 | A ~800-step snapshot is not a citable result for the thesis's starting point. | **partial** | [[../../20_Tickets/experiments/exp-adapter-avid-native-reference-run]] |

> R1 is one eval pass on a checkpoint you already have — no training. It is the
> cheapest experiment on this list and it gates the most. Run it first.
> Compounding risk: MetaWorld is classified **action-redundant**
> ([[ablation-axes]] Axis 1), the worst case for both R1 and node 2. If the gap
> is flat, the fix is to move node 1's anchor to Push Cube, not to abandon it.

---

## Tier 1 — per-node core evidence

### Node 2 — planning demo

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R3** | Reward model trained on rollout states. | No reward signal → no planning. | **missing** | **needs new ticket** |
| **R4** | Planner + env-grounded evaluation (does planning through the world model beat a no-model / random-shooting control?). | Without a control the demo shows nothing. | **missing** | **needs new ticket** |

Both **blocked on R1**. Time-box: this node is a *demonstration*, not a
contribution ([[../../10_now/positioning]] anti-positioning). The arc must
survive without it — node 3 stands on cost arithmetic alone.

### Node 3 — "too slow"

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R5** | **Rollout cost measurement**: NFE × wall-clock per rollout, on real hardware, for the AVID/DynamiCrafter setup. | The motivation for the entire back half of the thesis. Currently asserted, not measured. | **missing** | partially [[../../20_Tickets/feat-adapter-flops-per-step-estimator]] |
| **R6** | **Honest few-step baseline** — DPM-Solver (and/or a consistency-model sampler) at matched compute, not only 50-step DDIM. | Without it, "diffusion rollout is slow" is a strawman; an examiner will name DPM-Solver. Flagged as an open framing question in [[../../10_now/positioning]]. | **missing** | **needs new ticket** |

### Node 4 — the curvature bias

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R7** | **The theory's own prediction, tested.** [[../theory/shortcut-v-averaging-bias]] predicts the **coarse rungs plateau** under v-averaging and **converge** under an endpoint-consistent target. Measure per-rung shortcut loss both ways. | This is the highest-value experiment in the thesis: it converts a derivation into a *validated prediction*. It also makes node 4 an empirical result, not only an analytical one — so the node survives even if every adapter run fails. | **half done** — see below | [[../../20_Tickets/experiments/exp-shortcut-per-stepsize-loss-diagnosis]] (in-progress) + [[../../20_Tickets/bug-losses-shortcut-v-averaging-target]] (**stale — code shipped**) |

**R7 status, audited 2026-07-26.** More is done than "missing" implied:

| Piece | State |
|---|---|
| Zero-model-error analysis (5.1 / 16.1 / 24.1 % vs 0.000000) | **banked** — [[../theory/shortcut-v-averaging-bias]] §4 |
| Per-rung shortcut-loss logging | **shipped** 2026-06-17 |
| "Before" arm (`v_average`): fine rungs converge, coarse plateau, ~50× spread (`N064≈0.002` → `N001≈0.1`) | **measured** — run `diffusion_avid_shortcut_metaworld`, Snellius H100, bs 48, ~1600 steps |
| Endpoint inversion (`invert_ddim_v`, `target_kind="endpoint_inversion"`) | **shipped** — commit `279cdb7`, 2026-06-24 |
| Regression test that inversion reproduces the two-step landing, and `v_average` does not | **passing** — `tests/test_shortcut_endpoint_inversion.py`, 4 tests |
| Configs defaulting to `endpoint_inversion` | **5 DC configs**, incl. both ACWM ones |
| Displacement (Option B) | **not implemented** — raises `NotImplementedError`; ticket correctly open |
| **"After" arm: a training run with `endpoint_inversion` + per-rung curves** | **MISSING — this is all that R7 still needs** |

Two things that shape how to run it:

- **Do not compare a new run against the June baseline.** That data predates
  commit `279cdb7` by ~2 months of other changes, so the comparison would be
  confounded by everything except the target rule. `v_average` was deliberately
  kept as a config-selectable baseline arm — run **both arms at one commit**.
- **The June arm is not thesis-citable as it stands.**
  [[../../60_Updates/entries/2026-06-19-shortcut-v-averaging-bias-resolved]]
  records `wandb run id / commit / ckpt` as `_needs verification_`, and the
  artifacts (`data/results/20261706/`) are not in the local checkout. Hard rule 8
  blocks it. Re-running the A/B fixes the provenance as a side effect.

> So R7 is **one A/B run**, not an implementation task. If nothing else in the
> thesis works, R7 + the derivation is still a publishable-shaped contribution.

**Run it action-free.** Ticket:
[[../../20_Tickets/experiments/exp-shortcut-target-ab-actionfree]]. With actions
in the loop, "does the shortcut work" is confounded with "does action
conditioning work" — and the action side is the one currently failing. Tooling
built 2026-07-26: an action-free config (`conditions: []`), a
`--shortcut-consistency-target` CLI override, and
`jobs/experiments/exp_shortcut_target_ab_actionfree.sh` which runs both arms and
records per-run provenance.

**This also reorders the schedule.** Because the shortcut question can be asked
action-free, **D3 is not gated on D2** (only D4 is) — see the correction in
[[thesis-storyline]] §6. D3 evidence can be gathered *in parallel with* the D2
ablation rather than behind it, which is the cheapest available de-risking given
that D2 is the failing half.

### Node 5/6 — the Wan confound and the D2 collapse

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R8** | **ACWM dataset axis on Wan** — Push Cube / Robot Arm / Reacher, clean workhorse config. | The D2 headline. Tests hypothesis "data does not reward actions". | **partial / running** | [[../../20_Tickets/experiments/exp-backbone-wan-robotarm-run]], [[../../20_Tickets/experiments/exp-adapter-wan-cap50-warmup-pushblock-run]] |
| **R9** | **Weak flow base — SkyReels-1.3B.** Flow matching *and* weak. | **Resolves the confound.** Wan changed geometry *and* base strength together; SkyReels separates them. Without this, "Wan doesn't work" is ambiguous between "flow/shortcut adapters fail" and "we picked a base too strong for an adapter to matter" — and that ambiguity undercuts the whole back half. | **missing** | [[../../20_Tickets/experiments/exp-backbone-skyreels-pushblock-run]], [[../../20_Tickets/experiments/exp-backbone-skyreels-robotarm-run]] |
| **R10** | **DynamiCrafter on ACWM** — fills the empty cell of the boundary map. | Completes the 2×2 (base strength × action-informativeness) that node 8's claim rests on. | **missing** | [[../../20_Tickets/experiments/exp-backbone-dc-pushblock-run]], [[../../20_Tickets/experiments/exp-backbone-dc-robotarm-run]] |
| **R11** | **No-shortcut control on Wan** (`anchor_prob: 1.0`). | Proves the D2 collapse is **not** the consistency loss's fault — protects the D3 chapter from being read as "shortcut failed". Config exists; run does not. | **missing** | [[../experiments/wan22-avid-noshortcut-ablation]] (a *plan*, not a result), [[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]] |

### Node 7 — the ablation

One cell per hypothesis. Everything below is a *search*, not a factorial —
stop pulling levers as soon as a cell shows action-following.

| # | Hypothesis | Requirement | Status | Ticket |
|---|---|---|---|---|
| R9 | Base too strong | (above) | missing | skyreels tickets |
| R8 | Data does not reward actions | (above) | running | wan/ACWM tickets |
| **R12** | Gate saturation | `gate_cap` / AVID-warmup cells | **have (MetaWorld)** — did not unlock action-use | [[../../20_Tickets/experiments/exp-adapter-gatelow-cap-sigmashift-metaworld-run]] |
| **R13** | Identity-on-base-input shortcut | `condition_on_base_outputs: off`, one cell on Push Cube | **missing** | [[../../20_Tickets/experiments/exp-adapter-gatelow-nobase-overfit]] (verify scope) |
| **R14** | Wrong injection at high `d_a` | AdaLN cell on Robot Arm (da=7) | **missing** | [[../../20_Tickets/experiments/exp-adapter-adaln-gatelow-metaworld-run]] (MetaWorld — needs an ACWM variant) |
| **R15** | The shortcut objective itself | = R11 | missing | (above) |

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R16** | **Positive control for action-use.** One cell where the adapter provably learns something the frozen base *cannot* produce. | Without it, "every cell base-cloned" is **indistinguishable from a pipeline bug** — and that is the first question at the viva. `pg3x72uc` is a positive control for *composition mechanics*, not for action-use. Candidates already ticketed: the no-base overfit runs and single-clip overfit. | **missing** (candidates exist — verify one qualifies) | [[../../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]], [[../../20_Tickets/done/exp-training-single-clip-overfit]] |

### Ch3 — the adapter chapter

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R17** | **Cost table across adapter families**: params · FLOPs/step · peak VRAM · wall-clock/step, ± backprop through the frozen base, same base + task. | Ch3's "small experiments". Carries the **entire** adapter-taxonomy half of D1 — the Ch5 ablation only evidences backbone/dataset generality. Cost only; **no quality comparison** (ruled out by [[ablation-axes]] Axis 2). | **missing** | [[../../20_Tickets/experiments/exp-adapter-param-matched-comparison]] + [[../../20_Tickets/feat-adapter-flops-per-step-estimator]]; protocol: [[../experiments/protocol-param-matched-adapter-comparison]] |

### D3 / D4 — the back half

| # | Requirement | Why | Status | Ticket |
|---|---|---|---|---|
| **R18** | **Few-step rollout quality curve** — x = NFE, y = quality, vs a matched-compute many-step baseline. | D3's core evidence. [[../../10_now/positioning]] states plainly that no clean curve is in the vault. Without it D3 has *no* empirical result, only theory. | **missing** | [[../../20_Tickets/experiments/exp-shortcut-scale-episodes-longer-train]], [[../../20_Tickets/experiments/exp-shortcut-vs-image-only-anchor-baseline]] |
| **R19** | **D4 combined run** — action + step-size conditioning in one model, <10-step rollout. | The punchline chapter. **Blocked on a working D2 cell + R18.** | **blocked** | [[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]] |

---

## Minimum viable set

If only a handful of these run, run these — this is the smallest set where
every node of the arc is supported and the thesis ends on a claim:

1. **R1** — action-sensitivity on the AVID checkpoint _(eval only; gates node 1 and node 2)_
2. **R7** — the curvature theory's prediction, tested _(one A/B run: the code
   already ships both target rules; makes node 4 empirical; survives total
   adapter failure)_
3. **R9** — SkyReels weak flow base _(resolves the confound; without it the back half is ambiguous)_
4. **R16** — a positive control _(without it the ablation is uninterpretable)_
5. **R17** — the cost table _(carries D1's taxonomy claim; cheap, no convergence needed)_
6. **R5** — rollout cost _(the motivation for the entire back half)_

Note what is **not** on that list: the planning demo (R3/R4) and D4 (R19).
Both are descopable to future work without breaking the arc — the planning
node can be told as motivation, and D4 was always gated on D2+D3.

## Tickets that need creating

Requirements with **no** ticket today:

- ~~**R1**~~ — done 2026-07-26:
  [[../../20_Tickets/experiments/exp-eval-action-sensitivity-avid-checkpoint]],
  tooling built and unit-tested in the implementation repo.
- **R3 / R4** — reward model + planner demo (`exp-eval-…` or a new scope — the
  scope vocabulary is a closed set, so **ask before adding one**)
- **R6** — DPM-Solver / consistency-sampler baseline at matched compute
- Implementing **Action Error Ratio** as a metric (a `feat-eval-…` ticket) —
  needed for R1 to be AVID-comparable. The current probe measures
  prediction-space sensitivity, which answers "does the action matter" but is
  **not** the AVID-comparable number.
- Wiring the Wan2.2 / SkyReels backbones onto the shared probe core (today the
  CLI covers the DynamiCrafter family; Wan has its own probe inside
  `generate_wan22_i2v_compare.py --action-probe`, so the measurement exists but
  is duplicated).

## Related

- [[thesis-storyline]] — the arc these requirements serve
- [[ablation-axes]] — the axis/hypothesis design (node 7)
- [[../../10_now/positioning]] — deliverable evidence requirements
- [[../../70_Thesis/outline]] — the chapters that consume this evidence
