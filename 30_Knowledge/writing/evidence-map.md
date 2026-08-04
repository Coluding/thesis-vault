---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-storyline]]"
  - "[[ablation-axes]]"
  - "[[rubric/_index]]"
  - "[[../experiments/_index]]"
  - "[[../tech/probe-suite]]"
  - "[[../../70_Thesis/outline]]"
---

# Evidence map — cells → sections → topics → rubric items

> **The drafting lookup table.** For each evidence cell: what it shows, the
> runs behind it, where it lands in the thesis, which claim it licenses,
> which rubric items it moves, and the caveats that must travel with it.
>
> **Rule:** if a caveat is in this table, it appears in the same paragraph as
> the number ([[thesis-style-guide]] §2). No caveat gets deferred to §6.2
> alone.

## The core contrast — three D2 cells

| # | Cell | Injection | Verdict | § | Rubric |
|---|---|---|---|---|---|
| **W-x** | Wan × ACWM | cross-attention | **domain corrector** — beats frozen base 6/6, structure triad **at chance** | §5.3, §5.6 | 1, 3, **5** |
| **W-a** | Wan × RT-1 (clean-room) | **per-frame AdaLN** | **follows actions** — 2.49× @12000, diagonal concentration 0.409 vs chance 0.200 | §5.3 | **1**, 2, 3 |
| **D-c** | DC × ACWM | concat + `condition_center` | **follows actions** — triad above chance on all three axes | §5.2 | 1, 3 |

**W-x vs W-a is the decisive comparison** — same backbone, matched adapter
contribution and matched mask, only the pathway differs. It removes base
strength as a confound, which **D-c vs W-x cannot**. Lead with the
within-Wan contrast; use DC as corroboration on a second architecture.

---

## W-x — Wan × ACWM, cross-attention

**Runs.** `52o3uxz8` (oracle on), `vy9tcuco` (oracle off); structure triad
from probe job `25107536` on the NOBASE checkpoint.
Note: [[../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]],
[[../experiments/20260731-wan-action-signal-is-a-global-bag]].

**Numbers.** Against each arm's *own* frozen base: FID +36.3 % (90.15 →
57.43) and FVD-I3D +63.7 % (1118.4 → 406.3) with the oracle; +27.5 % and
+45.1 % without. `effect_rel` 0.0062 / 0.0077. Structure triad on the
NOBASE checkpoint: steering cos 0.00, temporal and spatial at chance.
Frozen-base null 0.

**Topics it licenses**
- *The adapter is a domain corrector on this pathway.* A 2.75× FVD
  improvement carrying no recoverable action information is domain
  adaptation, not conditioning.
- **The metric-blindness demonstration** — the single sharpest form we have.
- The **global bag**: the signal is present, measurable and structureless.
  Arbitrary in direction, at chance in time, uniform in space. *Structureless
  is a stronger and more interesting claim than absent.*

**Sections.** §5.3 (the negative half of the pathway contrast) · §5.6 (the
blindness section — **primary evidence**) · §1.4 (contributions, as the
exhibit) · §6.1 (the boundary).

**Rubric.** **5 Reflection** (this cell *is* the transferable lesson) ·
1 Originality · 3 Experimental evaluation (matched controls, verified nulls).

**Caveats that travel with it**
- The quality win is **ACWM-only**. On RT-1 the same adapter is a net
  perceptual regression (FID 143.1 vs base 120.8; LPIPS 0.427 vs 0.376),
  better only on pixel metrics — the mean-regression signature.
- Quote **% improvement over each arm's own base**; the two arms' bases
  differ. n=1 per arm, at slightly different steps (3315 vs 3054).
- The triad was run on the **NOBASE** checkpoint — quote the same
  checkpoint's quality numbers (+27.5 / +45.1 %) when making the
  same-checkpoint claim, not the oracle arm's larger ones.
- The triad has **never** been run on the binned RT-1 checkpoint
  (`0fqjrqjl`) — B4 in [[open-experiments-for-thesis]].

---

## W-a — Wan × RT-1, clean-room per-frame AdaLN

**Runs.** `avid_wan_rt1_47M_faithful` vs `..._pooled`; probes `25148083` /
`25148084`, re-probe `25148170`. AVID's own repository, third branch
(`external_repos/avid/wan_diffusion/`).
Note: [[../experiments/20260802-avid-wan-cleanroom-perframe-causal]].

**Numbers.** Single-variable A/B at **matched adapter contribution** (0.114
vs 0.111) and **matched mask** (0.953). At step 5000: effect_rel 0.01747 ±
0.00153 vs 0.01019 ± 0.00159, Welch t=3.30, n=16 batches; action-driven
share 15.3 % vs 9.1 %. Widening with depth: 2.46× @10000, **2.49× @12000**
(t=10.5, contributions identical to 0.2 %). Share 15.3 % → **25.3 %**,
exceeding AVID's 24.4 %; the pooled arm goes 9.1 % → 10.2 % — an
*information ceiling*, not under-training. Null 0.

**The confound resolution.** Diagonal concentration **0.390 vs 0.199**
(chance 0.200), and the pooled arm's per-frame response rows are
**bit-identical**. Gain scales a response; it cannot manufacture per-frame
differentiation. This is what takes the pathway claim off `effect_rel`.

**Topics it licenses**
- **The design principle** — conditioning must enter scale-free relative to
  the residual stream. *The* originality claim.
- **Wan is not the harder substrate** — at matched step 5000 the faithful
  arm beats the AVID/DC reference (0.01747 vs 0.01254); the famous 0.0495 is
  a step-15000 number.
- **D1 extensibility, demonstrated** — a new base family added to the
  official AVID repo, their recipe unchanged.

**Sections.** §5.3 (**primary**) · §5.1 (the D1 port) · §1.4 (the design
principle) · §6.1.

**Rubric.** **1 Originality** (the headline) · 2 Technical skills (matched
A/B + the temporal control) · 3 Experimental evaluation (confound named
*and* resolved) · 4 Knowledge.

**Caveats that travel with it**
- ⚠ **A5 — split status unrecorded.** This is on RT-1, and all RT-1 numbers
  from *our* trainer are quarantined as in-sample. The clean-room uses
  AVID's own eval path, so the quarantine may not apply — but the note does
  not say. **Read it before §5.3 is finalised**
  ([[open-experiments-for-thesis]] A5).
- Single seed. Welch t is over 16 evaluation batches, not over runs.
- Purity gap: AVID 24.4 % action-driven at step 15000 vs our 25.3 % at
  12000 — comparable, but the depths differ; state them.

---

## D-c — DC × ACWM, concat + `condition_center`

**Runs.** `6oyu1inq` (arm E), `tr0uovs5` (arm 0), `86kb01su` (arm F);
structure probe job `25144197` on held-out `ind_test`.
Notes: [[../experiments/20260731-dc-condition-center-accelerates-escape]],
[[../experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]].

**Numbers.** `effect_rel` @3500: arm E 0.11479 (3.9× the AVID reference
0.0295), arm 0 **0.04564** (1.55× unaided), arm F 0.05049 (1.71×). Triad on
held-out: steering +0.117 (chance 0.000), temporal 1.000 (chance 0.313),
spatial 0.470 (chance 0.100); null 0. Adapted loss 0.0357 vs control 0.0433.

**Topics it licenses**
- **The positive control** the ablation needed, and D2 target claim (a).
- **Action-following is not a trade against prediction quality** — it has
  the *lower* denoising loss.
- **Blindness is a long transient, not a state** — the untreated arms escape
  on their own, exactly as the 0.45 % economics predicted *in advance*.
- **The learned pedestal** as the DC-side scale failure (§5.4).

**Sections.** §5.2 (**primary**, keep compact) · §5.4 (the pedestal) ·
§5.6 (the mirror: follows actions, no quality metrics logged).

**Rubric.** 1 Originality (a positive result exists) · 3 Experimental
evaluation (matched control, measured noise floor 0.000245) ·
4 Knowledge (the economics prediction landing).

**Caveats that travel with it**
- **No DC run logs quality metrics** — all 18 checked. A1 pending.
- **Cancelled pre-convergence.** Quote the **~6× acceleration**; the level
  gap fell 3.7× → 2.5× and had not converged. Never quote a level.
- `condition_center` is an **accelerator, not an enabler** — arm 0 and arm F
  clear the reference unaided.
- **Rollout control not measured** here — A2.
- Keep it **compact**: Wan is the contribution, DC the positive control
  ([[rubric/_index]] weighting).

---

## The D3 cells

| # | Cell | Verdict | § | Rubric |
|---|---|---|---|---|
| **S-w** | Wan / flow / `v_average` | **learns the consistency relation** — `consistency_cos` 0.302 [0.251,0.356] vs control 0.034 [0.026,0.042], 9×, disjoint CIs; gain flat O(1) | §5.7 | 1, 2, 4 |
| **S-d** | DC / diffusion / `endpoint_inversion` | **does not** — 0.084 vs control 0.083, CIs coincident; gain explodes to 4e4 | §5.7 | 1, 4 |

**Topics.** Step-size conditioning through the adapter — *the one axis AVID
does not touch*. Two independent statistics (direction, magnitude), each
with a matched within-arm control. `consistency_cos` is a cosine and so
**gain-normalised by construction** — chosen precisely because `effect_rel`
was not.

**Caveats.** ⚠ **Cross-base comparison confounded** — target *and* depth
both vary; state it in the same paragraph as the 9×. Present the curvature
derivation as **theory with synthetic verification**; do not cite S-d as its
empirical confirmation (it ran the exact target). Few-step *quality* not
measured. A4 (the `v_average` vs `endpoint_inversion` 2×2, config-only) is
the decider.

---

## Reverse index — section → cells

| § | Cells | Also needs |
|---|---|---|
| §5.1 D1 framework | W-a (the AVID-repo port) | LoRA comparison (B2) |
| §5.2 working cell | **D-c** | quality metrics (A1), control (A2) |
| §5.3 pathway decides | **W-a** primary, W-x as the negative | A5 split check |
| §5.4 scale failures | D-c (pedestal), W-x (drowning) | — |
| §5.5 what it learns instead | W-x | — |
| §5.6 metrics are blind | **W-x** primary, D-c as the mirror | — |
| §5.7 shortcut | **S-w**, S-d | A4 |

## Reverse index — rubric → cells

| Item | Cells that move it |
|---|---|
| 1 Originality | **W-a** (design principle), **S-w** (uncovered axis), D-c (a positive exists) |
| 2 Technical skills | W-a (matched A/B + temporal control), S-w (gain-normalised metric) |
| 3 Experimental evaluation | all — matched controls, verified nulls, named confounds |
| 4 Knowledge | S-w/S-d + the curvature derivation, D-c (economics predicted the escape) |
| **5 Reflection** | **W-x** — the blindness demonstration |
