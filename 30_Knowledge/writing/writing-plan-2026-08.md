---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-storyline]]"
  - "[[../../70_Thesis/outline]]"
  - "[[../experiments/_index]]"
  - "[[../../50_Decisions/open/wan-action-following-needs-objective-change]]"
---

# Writing plan — start now, experiments continue in parallel

> **🛑 SUPERSEDED 2026-08-03.** This plan was written on 2026-08-01, before
> the 08-02 results and the reframe. Several of its statements are now
> **wrong**, not merely dated:
>
> - It frames **Wan as the collapse branch**. Wan beats its frozen base 6/6
>   on ACWM and, with per-frame conditioning, beats the AVID/DC reference at
>   matched depth. It is the *contribution*, not the failure.
> - Its headline — *"the adapter learns to correct the base, not to
>   incorporate actions"* — is contradicted by the DC cell, which follows
>   actions on three independent readouts.
> - Its §D2-e "two-factor law" and the 35× rest on the **retracted**
>   SkyReels cell.
>
> **Live replacements:** [[../../70_Thesis/outline]] (structure + per-section
> sources) · [[open-experiments-for-thesis]] (what is still needed) ·
> [[rubric/_index]] (what is graded) · [[thesis-storyline]] §9 (the result).
> Kept for its reasoning about sequencing and its sourcing discipline; **do
> not draft from its status claims.**

**Decision 2026-08-01 (user):** begin writing the thesis. The qualitative
results are clear enough to write against — *the adapter learns to correct the
base, not to incorporate actions* — and the mechanism behind that is measured.
Fix-experiments continue for ~2 weeks **in parallel**; they can only improve
numbers inside a story whose shape is already settled
([[thesis-storyline]] §8–9).

## What is safe to write TODAY (evidence closed)

| section | claim | sources |
|---|---|---|
| Ch1 arc | the chain AVID → planning → speed → shortcut → curvature → flow/Wan → D2 collapse → **mechanism stack** | [[thesis-storyline]] §1–9 |
| Ch3 methods | adapter families, composition, the probe suite (`effect_rel`, null control, `--emb-scale`, Jacobian, propagation trace, rollout-swap) | probe code + the 5 experiment notes |
| Ch5 §D2-a | the reference control: it is our implementation, not the data | [[../experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]] |
| Ch5 §D2-b | ⚠ **PROVISIONAL** — DC learned pedestal + `condition_center` (0.003 → 0.115 final). The *pedestal mechanism* is safe; the *fix* claim is not (gain-metric threat, see below) | [[../experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]], [[../experiments/20260731-dc-condition-center-accelerates-escape]] |
| Ch5 §D2-c | ⚠ **PROVISIONAL** — Wan value-pathway drowning is safe; `action_token_norm` as a *fix* is not (same gain-metric threat) | [[../experiments/20260731-wan-action-trace-value-pathway-drowns]] |
| Ch5 §D2-d | base-correction not action-use: oracle 100:1, 0.45% economics, global bag | [[../experiments/20260731-why-wan-copies-the-base-decomposed]], [[../experiments/20260731-wan-action-signal-is-a-global-bag]] |
| Ch5 §D2-e | 🛑 **BLOCKED** — in-sample eval + gain/instability confound. Do not write | [[../experiments/20260801-wan-rt1-indistribution-plateau]] |
| Ch6 discussion | the boundary claim + what would move it | [[thesis-storyline]] §9 |

**Methodological material worth its own subsection** (it is a contribution in
itself): loss/gate/FID/sample-quality are all blind to action-blindness; the
diagnosis required purpose-built probes. Include the failed hypotheses and the
pre-registered thresholds — the negative results are what make the positive
claim credible.

## What is NOT safe to write yet

- Final trajectory endpoints for in-flight runs (mark `in flight`, cite the
  snapshot date).
- The D3 flow-vs-diffusion curvature comparison — **confounded** by the
  frozen-gate bug ([[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]);
  needs a re-run before it can be cited.
- Anything from the SIMPLE / oracle-RT-1 arms until they settle.
- **Ch5 §D2-e (the two-factor law) — DOWNGRADED 2026-08-01.** The 35× "data
  sets the level" figure and "91% of the AVID reference" rest on `effect_rel`,
  which cannot distinguish action *information* from early-fit *instability*;
  both readings predict the observed peak-then-settle AND the data axis
  ([[../experiments/20260801-wan-rt1-indistribution-plateau]], confound
  section). Probe 25143284 settles it. Do not write §D2-e as a law until it
  returns.
- **🛑 ALL RT-1 AND OPENVID NUMBERS ARE IN-SAMPLE** — those jobs pass the same
  dir to `--data-dir` and `--eval-data-dir`, and the trainer's held-out branch
  (line 419) never splits it. ACWM is clean (`ind_train`/`ind_test`). Nothing
  RT-1 goes in the thesis until the held-out re-eval lands
  ([[../../00_Inbox/2026-08-01-rt1-heldout-split]]).
- **⚠ Ch5 §D2-b/c (the two mechanism fixes) are ALSO provisional.** `effect_rel`
  is monotone in action-pathway gain, and both `condition_center` and
  `action_token_norm` are gain increases — the vault's own gate control moved
  the metric 4.8× with the action path untouched
  ([[../../00_Inbox/2026-08-01-effect-rel-is-a-gain-metric]]). The DC structure
  triad (arm E vs arm 0, ~1 GPU-h) decides whether these are mechanism fixes or
  volume knobs. **Do not write §D2-b/c as fixes until it returns.**
- **DC arm E's "3.5× higher level" is retired** — the gap fell to 2.5× by the
  final eval and had not converged; the controls both exceed the AVID reference
  unaided. Quote the ~6× *acceleration* only
  ([[../experiments/20260731-dc-condition-center-accelerates-escape]]).
- **The RT-1 cells are a net quality regression** vs the frozen base on every
  perceptual metric (FID/FVD/LPIPS/SSIM) while better on pixel metrics
  (PSNR/MSE) — the mean-regression signature. Any D2 claim on RT-1 must state
  this; check whether the same split holds for DC arm E before relying on the
  spine cell.

## Order of writing (dependency-first)

1. **Ch3 methods + the probe suite** — stable, long, unblocks everything.
2. **Ch5 D2 results** in the §D2-a…e order above — the story is linear.
3. **Ch1 arc** — write once the results section exists, so the intro promises
   exactly what Ch5 delivers.
4. **Ch6 discussion / boundary** — last of the prose.
5. **Abstract** — last.

## Spine correction (2026-08-01) — DC is the main story

The narrative is carried by the **DC cell**: DC + adapter → planning → too slow
→ flow → shortcut. DC is the cell that *works* (`condition_center`, 0.115
final), so it is the one that can host planning and motivate the speed
argument. **⚠ 2026-08-01: "works" is now conditional.** The untreated control
also clears the AVID reference unaided (0.0456), the treatment gap was still
closing when the runs were cancelled, the DC cell logs **no quality metrics at
all**, and the structure triad has **never** been run on it. Whether the spine
has a working cell at its centre is decided by the arm-E-vs-arm-0 triad
(~1 GPU-h, job written, awaiting review). **Wan is the generality branch** — it collapses, and the
mechanism campaign explaining why is a separate (methodological) contribution.
Write it in that order; the reverse would leave the thesis with no working
system at its centre.

**Consequence for the schedule:** the only spine link without evidence is
**planning**, which has never been run
([[../../20_Tickets/experiments/exp-eval-planning-through-dc-world-model]]).
It is now the top experimental priority — above further Wan fixes — and its
wall-clock measurement is also the quantitative motivation for D3.

## Parallel experiment track (next 2 weeks, does not block writing)

- **Planning through the DC arm-E world model (TOP PRIORITY — spine link).**
- Objective-level fix (the open decision): action-CFG or rollout losses —
  the only lever the evidence says can raise the floor.
- Structural repairs from the bag analysis: enforced px→latent temporal
  binning (done for the simple head), localisable gate.
- D3 curvature re-run with live gates (unblocks a D3 claim).
- Rollout-action-swap on an RT-1 checkpoint (does the ~0.02 floor buy any
  *control*?).

Any of these that lands improves a number in a section that already exists —
none of them changes the storyline.

## Rules while writing

- Every number cites a run (CLAUDE.md hard rules 7–8); `in flight` where true.
- Keep D1–D4 evidence separated in the chapters even though Ch1 tells the arc.
- Prefer "the adapter corrects the base rather than incorporating actions" —
  the measured statement — over "action-blind", which the RT-1 floor now
  qualifies.
- Use `/thesis-write`; it reads [[../../70_Thesis/outline]] + recent
  `60_Updates/` and respects the sourcing rules.
