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

## Two cells added 2026-08-03

### L-c — LoRA vs output adapter, wall-clock (D1)

**Runs.** `25183721` (LoRA) vs `vy9tcuco` (output adapter), Wan × ACWM Robot
Arm. Note: [[../experiments/20260803-lora-costs-3x-the-walltime-of-an-output-adapter]].

**Number.** LoRA costs **3.1× the wall-clock per optimizer step** at matched
effective batch — 66 vs 205 steps/h — while being the **smaller** adapter
(26.0M vs 34.9M trainable). The cause is structural, not tuning: LoRA's
trainables sit *inside* the frozen base, so the forward pass must be
differentiable through all 30 blocks, and the resulting activation memory
forces gradient checkpointing.

**Why it matters to the spine.** This is the measured answer to *"why not
just fine-tune, or use PEFT?"* — the objection every frozen-base thesis
meets. It is a **cost** claim, not a quality claim, so it sidesteps the
underpowered-comparison problem that ruled out a quality shoot-out
([[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]).
Parameter count is the wrong axis: the *smaller* adapter is 3× more
expensive because of where its parameters live.

**§.** §5.1 (D1) · §3.2 (the family selection — this is the receipt for
criterion (b), gradients through the frozen base).
**Rubric.** 1 Originality (D1 becomes empirical, not just analytical) ·
3 Experimental evaluation.
**Caveats.** n=1 per arm; wall-clock is hardware- and
geometry-specific — state the device and the batch geometry.

### C-n — channel-concat injection (D2, negative)

**Runs.** `6ruz55f6` vs `vy9tcuco`, single-key config diff.
Note: [[../experiments/20260803-concat-injection-does-not-help]].

**Result.** Concat is **below** cross-attention at every matched step
(0.00975 vs 0.01056 @1200; share 0.168 vs 0.181) with identical `eval_loss`.
Does **not** reproduce GigaWorld-1's 2.2× concat-over-xattn.

**Why it is worth writing.** It is a *reasoned* negative: their concat
carries a spatially-aligned control **video**; ours broadcasts a 7-DoF
vector uniformly across H,W, so it adds no spatial information that
cross-attention lacked. Their concat tells the model *where*; ours only
tells it *when*. This sharpens the pathway principle — what matters is not
concat-vs-attention but whether the conditioning is **spatially grounded and
scale-commensurate**.

**§.** §5.3 (a third point on the injection axis) · §6.3 (future work:
rendering actions into image space).
**Rubric.** 3 Experimental evaluation (a literature-motivated prediction
tested and refuted, with the reason) · 5 Reflection.
**Caveats.** Killed at step 1567; concat was **rising** while xattn was
falling and they would cross around ~1500–2000 — the sign is unresolved.
⚠ The first test suite passed **vacuously** (zero-init head, zeros compared
to zeros) — belongs in [[methods-integrity]].

## Cells added 2026-08-05/06 — these REVISE earlier readings

### E-o — EasyAnimate V5 (diffusion) vs V5.1 (flow), objective comparison

**Runs.** slurm `25240257` (V5 diffusion) / `25241732` (V5.1 flow), project
`coluding/EasyAnimate-objective-acwm-robotarm`, ACWM Robot Arm. ⚠ **status
`running`, steps not matched (9000 vs 8200), n=1 per arm.**
Note: [[../experiments/20260806-objective-governs-action-specificity-not-adapter-capacity]].

**Two findings, and the first one revises the Wan story.**

1. **The Wan ceiling was not intrinsic to output adapters.** The *same*
   adapter family (34.9M, `composition: add`, cross-attention,
   `condition_on_base_outputs: false`) cuts denoising loss **−74.9 %** on EA
   and contributes **0.52** of the prediction, against **−3.3 %** and
   **0.047** on the best comparable Wan arm. Wan's `adapter_base_cosine`
   0.9989 means the adapted prediction was *numerically almost the frozen
   base* — the adapter was **cosmetic** there.
2. **The objective governs action-specificity, not adaptation capacity.**
   Loss reduction is tied (−74.9 % vs −73.6 %); `action_effect_rel` is
   **+36 %** and `effect_vs_adapter` **+26 %** for diffusion.
   `effect_vs_adapter` is the cleaner measure — it asks what fraction of the
   adapter's *own* output is action-driven, so a large adapter cannot
   inflate it.

**Cross-backbone pattern.** Both diffusion backbones (DC 1.4B, EA-V5 7B) sit
above both flow backbones (EA-V5.1 7B, Wan 5B) — **n=2 per objective across
independent model families**, which is materially harder to dismiss than one
pair. ⚠ But it is *"flow is weaker"*, not *"flow fails"*: EA-flow (0.029) is
~3× Wan-flow, so backbone still matters and the two are **not separable**
from four points.

**Topics it licenses.** The objective as the governing variable — which
**unifies with the 0.45 % economics**: both say *what the adapter learns is
set by how the objective allocates gradient*, not by capacity. Also: a
5th/6th backbone family for D1, and a same-video-backbone objective contrast
(V5 vs V5.1) that is the closest thing to a controlled comparison available.

**§.** §5.3 (the pathway/objective analysis) · §5.1 (D1 breadth) · §6.1.
**Rubric.** **1 Originality** · 3 Experimental evaluation · 4 Knowledge.

**Caveats that travel with it**
- ⚠ **Interim.** Both arms running; steps unmatched (9000 vs 8200); n=1.
  Quote the end-of-run matched-step value. Direction held for 8 consecutive
  hourly evals — quote *that*, not a single eval.
- **EA vs Wan differs in backbone, VAE, objective *and* text conditioning.**
- **V5 vs V5.1 do not share a text backbone** (BERT+T5 vs Qwen2VL), and CFG
  steers *through* text — not peripheral.
- **Only 6–8 % of the adapter's large contribution is action-driven**
  (`effect_vs_adapter` 0.076 / 0.060). "The adapter is powerful" is
  established; "powerful **at action conditioning**" is not — the
  domain-corrector reading may still hold, now at ~10× the operating scale.
- All `effect_rel` logged before the 2026-08-05 base fix are **void** (I8).

### T-l3 — Wan2.2-Turbo (distilled base) + action adapter = **efficiency level L3**

**Run.** `jlnl7s1k` (slurm `25240927`), 100M adapter (1.99 %), binned action
tokens, ACWM Robot Arm 49f. Frozen `Wan2.2-TI2V-5B-Turbo`, 4-step distilled
grid. Note: [[../experiments/20260805-turbo-action-tokens-binned-to-latent-grid]].

**This is the first data point on L3 of the efficiency axis**
([[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]]).

**Result 1 (sourced).** Binning action tokens to the latent grid raises
`effect_rel` **4.1×** and `effect_vs_adapter` **6.5×** (0.0277 → 0.179) at
matched step 1600, growing monotonically to 0.27–0.31 by step 2800–4000.
⚠ **Two variables changed** — binning *and* 34M→100M — so it is not
attributable to binning alone; a 34M+binned control is needed.

**Result 2 (sourced, negative) — the clean dissociation.**
`eval_action_loss_gap` is ~0 at every one of ten evals (|x| ≤ 0.00055) and
`eval_action_cos` never leaves 0.9998. **The action now changes the
prediction, but changing it to the *correct* value does not reduce error
versus shuffling or zeroing it.** Misalignment was real and worth fixing; it
was **not the binding constraint.**

**Result 4 (sourced).** Overfits from step 1200 (best `eval_loss` 0.1271 →
0.1918 @4000), and `eval_denoise_adapter_delta` never crosses zero — **the
adapter hurts denoising at every eval.** Gate healthy throughout.

**Why L3 matters to the axis.** `effect_vs_adapter` 0.18–0.31 on the
distilled base is ~4× the best comparable Wan arm (0.047) — *suggestive*
that putting acceleration in the base does not cost conditioning, which is
the direction H-E predicts. ⚠ **Confounded** (100M vs 34.9M, binned vs
unbinned, different base), so it is not yet evidence for the
pre-registration. The matched control is what would make it so.

**§.** §5.7 / the acceleration axis (L3) · §5.3 (the dissociation).
**Rubric.** 1 Originality (L3 has data) · 3 Experimental evaluation (a clean
dissociation with a matched-step comparison) · 5 Reflection.

**Caveats**
- ✅ **Result 3 (motion tracking) RESOLVED 2026-08-06** — and it resolved
  *against* the preliminary reading. See **T-mc** below.
- ⚠ **Reproducibility:** 135 uncommitted modified files at launch; the run
  used rsynced working-tree code, not commit `75721b7`.
- Also changed and not isolated: 49f clips (was 97), fp32 eval VAE decode.

### T-mc — the paired control on the Turbo arm (D2, methodological)

**Run.** `mo3k2639` (slurm `25259766`), same Turbo arm, 1200 steps, first run
of the instrumented motion metric.
Note: [[../experiments/20260806-motion-tracking-is-action-driven-but-the-base-control-was-wrong]].

**Result 1 (sourced).** The **paired shuffled-action** control — identical
weights, conditioning frame and seed, only another clip's actions — gives a
positive gain in **4/4 draws** (+0.069, +0.168, +0.122, +0.213; mean
**+0.143**; sign test p ≈ 0.06). Because it differs from the adapted rollout
in exactly one respect, this is the **causal** quantity.

**Result 2 (sourced, negative).** The **frozen-base** control does *not*
fire: adapted − base is +0.13, +0.10, +0.045, **−0.034** — sign-inconsistent
and centred near zero. The hand-measured preliminary gap of **0.66** does
**not** survive; the logged gap is an order of magnitude smaller.

**What it licenses.** A modest, consistently-signed action effect on **how
much the arm moves per clip** — and, more importantly, the design rule:
*a control must differ from the treatment in exactly one respect.* This is
the second time an instrument in this thesis was validated by finding it
wrong (the first being `effect_rel`'s gain confound).

**§.** §4.3 (probe suite — controls) · §4.6 (integrity, **I10**) · §5.3.
**Rubric.** **2 Technical skills** (instrument validation) · **5 Reflection**
· 3 Experimental evaluation.

**Caveats**
- **No interval on the gain.** The CIs reported are on `corr(adapted, GT)`,
  not on the gain; paired bootstrap is now wired but **cannot be applied
  retroactively** — the next run produces the first gain intervals.
- `corr(adapted, GT)` is **not distinguishable from zero** — its CI spans
  zero in 3 of 4 draws. The claim is about the *gain*, not the correlation.
- n=16 per draw, 4 draws, **one run**, no seed replication.
- ⚠ **No wandb dashboard** — metrics in stdout and `metrics.jsonl` only.
- Still **magnitude, not correctness**: `action_loss_gap` finished at
  0.00005, `action_cos` at 0.99995. The effect-without-accuracy dissociation
  is unchanged.

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
