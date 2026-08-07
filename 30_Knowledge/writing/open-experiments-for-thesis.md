---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-storyline]]"
  - "[[ablation-axes]]"
  - "[[rubric/_index]]"
  - "[[../experiments/_index]]"
  - "[[../../70_Thesis/outline]]"
---

# Open experiments for the thesis

> **The single tracker for what is still needed while writing proceeds in
> parallel.** Supersedes the status half of
> [[storyline-experiment-requirements]] (dated 2026-08-01, written before
> the 08-02 results landed — it still records D3 as having no clean
> positive and wall-clock as unmeasured; both are now done).
>
> **Governing rule** ([[rubric/10-keeping-to-schedule]]): a new result may
> change a *number* in a section that exists. It may not change the *shape*
> of the thesis. Anything that would restructure chapters after drafting
> begins goes to `50_Decisions/open/` and becomes future work — unless it
> invalidates a claim already written.

## Status of the evidence base

| Area | State |
|---|---|
| **D1** framework | ✅ 3 backbone families + the AVID-repo port. LoRA comparison in flight |
| **D2** Wan mechanism campaign | ✅ **the contribution** — 13 axes, depth trace, decomposition, clean-room A/B |
| **D2** DC positive control | ✅ 3 readouts. Quality metrics + control measurement outstanding |
| **D3** shortcut | ✅ one clean positive (9×, disjoint CIs); cross-base confound unresolved |
| **D4** combined | ❌ descoped — state the descope explicitly |
| Rollout wall-clock | ✅ measured (linear in N; DC 2.8×/step faster, *model size not objective*) |

## Open experiments

Ordered by what they unblock in the draft.

### A — Blocking a written claim

| # | Experiment | Unblocks | Status |
|---|---|---|---|
| A1 | **Quality metrics on the DC cell** | §5.2 — currently *no* DC run logs quality metrics (all 18 checked), so the positive cell's output quality is unknown | 🔄 **support landing soon** (reported 2026-08-03) |
| A2 | **DC control measurement** — rollout-swap + per-dimension ablation, both generation-only | §5.2 upgrade from *structured response* → *action-following/control*; and the DC-vs-Wan contrast with the **same** probe on both sides | ☐ [[../../20_Tickets/experiments/exp-eval-rollout-action-swap-dc-arme]] |
| A3 | **RT-1 held-out re-eval** | every RT-1 number is quarantined as in-sample until this lands | ☐ [[../../00_Inbox/2026-08-01-rt1-heldout-split]] |
| A4 | **Shortcut target 2×2 — `v_average` vs `endpoint_inversion` within ONE base at ONE depth** | ⭐ **config-only, no code** (`training.shortcut_consistency_target` supports both). Decides whether the D3 result is about **geometry** (curvature) or about the **target construction**. Currently the confound runs *against* the curvature story: the DC arm used the theoretically exact `endpoint_inversion` and learned nothing. Until this runs, "shortcut works on flow because κ=0" is a hypothesis, not a result | ☐ **highest-value D3 run** |

### ⚠ A5 — Confirm the clean-room A/B's evaluation split (do this first, it is a read)

**The thesis's most original claim rests on this arm**, and its split status
is not recorded. The clean-room per-frame-vs-pooled A/B runs on **RT-1**,
and *all* RT-1 numbers are quarantined as in-sample (I1,
[[methods-integrity]]) — but that finding was about **our** trainer
(`--data-dir` = `--eval-data-dir`, the held-out branch never splitting). The
clean-room runs in **AVID's own repository**
(`external_repos/avid/wan_diffusion/`), which has its own evaluation path,
so the quarantine may or may not apply.
[[../experiments/20260802-avid-wan-cleanroom-perframe-causal]] does not say
either way — the words "held-out", "in-sample" and `ind_test` do not appear
in it.

**Why it may not be fatal even if in-sample.** The readouts are a
*sensitivity* probe and a *structure* probe, not generalisation metrics. A
memorised model can still be asked whether its prediction responds to an
action perturbation, and the diagonal-concentration result rests on a
*within-model* comparison (per-frame vs pooled) at matched contribution and
matched mask, where both arms share whatever memorisation is present.
**But structure measured on training clips is weaker evidence than on
held-out clips, and the difference must be stated either way.**

- [ ] Read the clean-room eval path; record the split status in the note.
- [ ] If in-sample: re-probe on held-out clips before §5.3 is finalised, or
      state the limitation explicitly in that section and in §6.2.

### ⭐ A6 — D4: the action-conditioned shortcut adapter on Wan

**The thesis's punchline, and it is no longer gated — just unrun.** D4 was
descoped as "gated on a working D2 cell". Both halves are now validated
**on the same backbone, separately**: the shortcut objective is learnable in
the Wan/flow cell (action-free by design), and per-frame AdaLN gives
structured action-following on Wan. The combination has never been trained.

Ticket exists since 2026-06-04:
[[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]].

**Why it is the highest-value remaining run for Originality**
([[rubric/01-originality]]): the contribution spine is *efficient rollouts
for action-conditioned world models via adapters*, and D4 is the only cell
that instantiates it. Everything else supplies conditions or components.

**Design note.** The per-frame AdaLN result says which injection to use, so
the D4 arm should not repeat the cross-attention pathway. Combine
per-frame AdaLN conditioning with the shortcut objective on the flow base.

⚠ Pre-register: what counts as success when *both* objectives are active —
action structure must not degrade relative to W-a, and consistency must not
degrade relative to S-w. A D4 cell that wins on neither axis is a negative
result and should be reported as one.

**Alternative routes to the same spine**, if D4-direct is too expensive:
PDD/LoRA distillation of the base with the action adapter on top
([[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]],
[[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]]),
or an action adapter on an already-distilled base
([[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]).
These split the two jobs rather than asking one adapter to learn both.

### ⭐ A0.4 — Few-step rollout QUALITY (reported 2026-08-07, not in the vault)

**The efficiency axis's missing outcome variable.** Lukas reports few-step
rollouts that look better and will supply the results. Until they land the
axis can say the shortcut objective is *learnable* and that a distilled base
*accepts* an action adapter, but not that any of it produces usable video in
few steps.

⚠ **EVIDENCE PENDING, NOT MISSING.** No level is written as producing usable
few-step video until this lands.

> **⭐ CLARIFIED 2026-08-07: the few-step quality result is ACTION-FREE, and
> the action-conditioned version does not work.**
>
> This is a **D3** outcome, not a D4 one, and it is the *predicted* result.
> H-E (pre-registered 2026-08-04, before any level ran) says the entangled
> level degrades while separable levels do not. Acceleration succeeding with
> actions stripped, and failing with actions present, is that prediction
> landing in its primary branch.
>
> **Consequence for the chapter:** D4 becomes a **reported negative with a
> mechanism**, not an absence. The efficiency axis reads: the shortcut
> objective is learnable *and* improves few-step quality when it is the only
> thing the adapter must learn; adding action conditioning to the same
> adapter breaks it; the economics predicted exactly that. That is a
> stronger chapter than a bare mechanism study and it does not require D4 to
> have succeeded.
>
> ⚠ **Three things must be established before it can be written that way**,
> because the prediction landing is worth nothing without the controls:
>
> 1. **A matched control on the action-free quality claim**: shortcut vs
>    no-shortcut at the *same step count*. Without it, "few-step looks
>    better" is not attributable to the objective.
> 2. **What form the action-conditioned failure took.** Degraded
>    action-following? Degraded few-step quality? Failed to train at all?
>    H-E predicts specifically the *first*. A different failure mode is a
>    different finding and must not be reported as H-E confirmed.
> 3. **A matched conditioning-only control for the combined arm**, or the
>    degradation has no referent. This is the requirement that the first L3
>    data already violated once.

**What makes it citable, in order of how much it matters:**

1. **Against the no-shortcut control at the same step count.** The only
   comparison that isolates the objective. "Better than the base at many
   steps" is a different claim; "better than earlier attempts" is not a
   claim at all.
2. **Which level** it came from (L1 / L2 / L3). The three currently sit on
   different bases, adapter sizes and clip lengths, so the level decides
   what the number may be compared against.
3. **A metric, not only visual inspection** (FVD / FID / LPIPS). A
   qualitative panel is legitimate as a figure but cannot be the outcome
   variable, and this campaign owns the counterexample: the Wan cell
   improved on 6/6 perceptual metrics while carrying no action information.
4. **Matched NFE**, against an honest fast-sampler baseline (DPM-Solver,
   consistency sampling), not many-step sampling alone
   ([[../../50_Decisions/open/d3-positioning-vs-weaver-reflow]]).
5. **Same clips, same seed** across arms, stated in the caption.
6. Run ids, checkpoints, step counts.

**With (1) and (3) the efficiency chapter has an outcome and the axis stops
being a mechanism study.** With only "looks better" it is a qualitative
figure and the gap stays open.

### ⭐ A7 — The three acceleration levels (the efficiency axis)

Open decision: [[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]].
**Pre-register the predicted ordering before any of these land.**

| Level | Run | Status |
|---|---|---|
| **L1** shortcut adapter + actions (entangled) | = **A6 / D4** | ☐ never trained |
| **L2** PDD / LoRA distil, adapter on top (separable) | [[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]] · [[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-wan]] | ☐ ticketed |
| **L3** action adapter on an already-distilled base (free) | [[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]] | 🔶 **first data 2026-08-05** — [[../experiments/20260805-turbo-action-tokens-binned-to-latent-grid]]. `effect_vs_adapter` 0.18–0.31 (~4× the best Wan arm) but `action_loss_gap` ~0 → **effect without accuracy**; confounded by 3 simultaneous changes; no matched control |

**Pre-registered 2026-08-04, before any level ran** — full statement with
both branches and the capacity discriminator in
[[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]] §Pre-registration.

- **H-E (primary):** the deficit is a **gradient-budget** effect, not a
  parameter-budget one. L1 below its matched control; L2/L3 at control.
- **H-C (secondary, scoped):** capacity sufficient for one objective may be
  insufficient for two. ⚠ Distinct from H4, which is **killed** — do not
  reinstate the unscoped version. Testable via the 34.97M vs 47M arms.
- **Discriminator:** H-E predicts the deficit is insensitive to adapter size
  and sensitive to parameter-set sharing; H-C predicts the reverse.
- **Both branches are results.** If L1 matches its control, H-E is wrong at
  this scale, separability is unnecessary — and D4 is delivered. Do not
  write §5.x so that it can only report a negative.

**Common protocol — without it the three are not comparable:**
- [ ] A **matched conditioning-only control** per level.
- [ ] The **same action-structure readout** across levels — the structure
      triad, not `effect_rel` alone.
- [ ] **Wall-clock and NFE at matched quality**, against an honest
      fast-sampler baseline (not many-step sampling alone).
- [ ] **Few-step quality actually measured** — currently *not measured* on
      the one shortcut cell that exists.
- [ ] **An accuracy readout, not only an effect readout** — L3 scored best
      of any cell on `effect_vs_adapter` while learning no correct dynamics
      (`action_loss_gap` ~0). Effect-only ranking is actively misleading.

⚠ Lukas reports data for the levels exists. Until run ids and checkpoints
are in the vault this is **A0-class evidence** — the framing may be built,
no level may be written as having worked.

### B — Strengthening a claim that already stands

| # | Experiment | Moves | Status |
|---|---|---|---|
| B1 | **Seeds on the headline clean-room A/B** | Technical skills → 9; the multiple-comparison exposure across 13 axes | 🔄 **planned** — see the seed policy below |
| B2 | **LoRA vs output adapter** | D1 from complexity-analysis to an *empirical* family claim | 🔄 in flight — [[../../20_Tickets/experiments/exp-adapter-lora-vs-output-comparison]] |
| B3 | **Action Error Ratio** (AVID §4.2) | an external, published readout so the D2 table is AVID-comparable — replaces reliance on our own `effect_rel` | ☐ |
| B4 | **Structure triad on the binned RT-1 checkpoint** (`0fqjrqjl`) | never run, and it is the intervention most likely to change the Wan verdict | ☐ |
| B5 | **IDM ceiling on DC latents** — `(z_t, z_{t+1}) → a_t` on ground-truth transitions | calibrates *every* number in the campaign: is action information present in the data at all? No world model, no new labels, minutes of GPU | ☐ carried over from [[storyline-experiment-requirements]] |

### A0 — Evidence that EXISTS but is not in the vault (collect, don't re-run)

| # | Evidence | Why it matters | Status |
|---|---|---|---|
| A0.1 | **Shortcut loss curves** (going down) | the basic learnability signal for D3 | ⚠️ reported 2026-08-03, **not in the vault** |
| A0.2 | ~~Spearman velocity-magnitude vs step size ≈ 1~~ | ✅ **ALREADY IN THE VAULT — and it is not what it looked like.** `spearman vs ladder = +1.0000` for **all three arms** (Wan treated, Wan control, DC), listed under *Nulls and controls*: a **monotonicity sanity check**, not evidence of learning. The control scores identically, so it cannot discriminate | ✅ found — [[../experiments/20260802-shortcut-works-on-flow-not-diffusion]] |
| A0.2b | **The gain ladder `\|dpred\|/\|dtarget\|` per rung** | ✅ **the discriminating magnitude evidence, already sourced.** Wan treated flat and O(1) across the ladder (0.440→0.334); Wan control **collapses** (0.483→0.026); DC **explodes** 4–5 orders (0.973→40950) at exactly the large `d` few-step rollout uses. Not a cosine — it is the scale the cosine normalises away | ✅ in the vault |
| A0.3 | **Step-size input sensitivity experiments** | the D3 analogue of the action-sensitivity probe — does perturbing `d` change the prediction? | ⚠️ reported, **not in the vault** |

> ⚠ **EVIDENCE PENDING, NOT MISSING.** Lukas reports these exist and will
> supply them (2026-08-03). Needed before any of it is written: run ids,
> checkpoints, the probe invocation, and the `n` behind the Spearman. **No
> D3 number enters the draft until they land.**

**Note the overlap with existing tooling.** A **step-size blindness probe
suite** was built and unit-tested on 2026-08-02 —
`evaluation/stepsize_structure.py`, `scripts/eval_stepsize_blindness.py`,
`scripts/probe_stepsize_embedding_standalone.py`, 21 CPU tests with
closed-form oracles for both model classes, plus a 2×2 Slurm job over
{DC diffusion, Wan flow} × {shortcut, matched control} — and **never
submitted** ([[../../00_Inbox/2026-08-02-stepsize-blindness-probe-suite]]).
If A0.1–A0.3 came from ad-hoc analysis rather than this suite, submitting it
would put D3's evidence on the *same instrument footing* as D2's, which is
worth more to the thesis than the numbers alone: it makes the probe
methodology a general contribution rather than an action-specific one
([[rubric/02-technical-skills]]).

⚠ Its `base_null_violation` on DC was root-caused to the frozen base running
in `train()` mode (dropout firing), **not** a conditioning leak — resolved
2026-08-01, [[../../00_Inbox/2026-08-01-stepsize-null-violation-rootcause]].
Confirm the fix is in before quoting DC step-size nulls.

### C — Future work / candidates (not blocking)

| # | Item | Note |
|---|---|---|
| C1 | **Exploration / mean-regression fixes on Wan** | Wan's action signal is measurable in the *prediction* but does not manifest in the *video*. Read as mean regression. Fix family Lukas calls **exploration** — action-CFG, rollout/multi-step losses, action-conditional consistency. ⚠ the mean-regression reading is currently *inferred* from the RT-1 metric split, not directly probed — measure the render-vs-prediction gap before writing it. [[../../00_Inbox/2026-08-03]], [[../../50_Decisions/open/wan-action-following-needs-objective-change]] |
| C2 | **Planning through the DC world model** | the natural follow-on once A2 lands — control is the precondition for planning. [[../../20_Tickets/experiments/exp-eval-planning-through-dc-world-model]] |
| C3 | **Structural repairs** from the bag analysis | enforced px→latent binning (done for the simple head), a localisable gate |

### Explicitly NOT running

| Item | Why |
|---|---|
| **ControlNet on Wan** | **AVID already published ControlNet and ControlNet-Small baselines on this problem** — cite theirs. ControlNet is a UNet construct (encoder copy + zero-convs into skips); porting it to a DiT invents a variant, so any result would be confounded by our port rather than testing ControlNet. The pathway axis is already settled causally by the clean-room A/B. ⚠ **`related-work/controlnet.md` is still required** for Ch2/Ch3 — the *note* is not optional, only the run |
| **The full FLOPs-matched family grid** | 39–48 runs, blocked since May on the estimator. [[../../20_Tickets/experiments/exp-adapter-param-matched-comparison]] |
| **Hypernetwork / hidden-state empirical arms** | stay in the complexity-analysis argument (D1) |

## Seed policy (noted 2026-08-03)

**Not treated as a blocker.** Additional experiments are running over the
coming days while the thesis is written, and those runs supply the extra
seeds. Consequences for writing:

- Draft against **single-seed** numbers now, with the caption or sentence
  saying so explicitly — silence about a single seed reads as concealment
  ([[rubric/03-experimental-evaluation]]).
- When multi-seed numbers land, they replace the number and the caption; no
  section is restructured.
- **Prioritise seeds on the headline clean-room A/B** — the one comparison
  carrying the pathway claim — over breadth across cells.
- State the **multiple-comparison exposure** honestly: 13 axes, and which
  claims were pre-registered vs exploratory.

## Related

- [[rubric/_index]] — the rubric queue these feed
- [[storyline-experiment-requirements]] — the 07-26 reasoning, kept as background
- [[../experiments/_index]] — the results ledger
