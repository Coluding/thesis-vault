---
type: experiment
date: 2026-07-31
config: configs/dynamicrafter/diffusion_dc_acwm_robotarm_{armE_center,arm0_baseline,armF_nativeavidencoder}.yaml
commit: uncommitted working tree @ 2026-07-30 (launcher submit_train_dc_avid_parity.sh, arms relaunched with the condition_center code present)
wandb_run_id: 6oyu1inq (arm E), tr0uovs5 (arm 0), 86kb01su (arm F)
ckpt_path: /scratch-shared/lbierling/outputs/dc_acwm_robotarm_{armE_center,arm0_baseline_s0,armF_nativeavidencoder}/checkpoints/
status: killed
deliverable: D2
metrics:
  armE_final_effect_rel: 0.11479      # @step 3500, verified 2026-08-01 (the 0.106->0.092 dip was noise)
  arm0_final_effect_rel: 0.04564      # untreated control — 1.55x the AVID reference on its own
  armF_final_effect_rel: 0.05049      # AVID's exact encoder, no centering — 1.71x the reference
  armE_over_arm0_final: 2.51          # was 3.7x at step 3000 — the level gap is SHRINKING
  avid_reference_effect_rel: 0.029475
  armE_eff_over_adapter: 0.311
  armE_adapted_loss_vs_control: 0.0357
  arm0_adapted_loss: 0.0433
notes: "condition_center (arm E) takes DC action-following 0.003 -> 0.115 final (3.9x the AVID reference) within ~2500 steps. QUALIFIER: the untreated controls are NOT permanently blind — past step ~2500 arm 0 escaped to 0.0456 and arm F (AVID's exact encoder, no centering) to 0.0505 — BOTH above the reference. So the fix = ~6x faster escape; the level gap shrank 3.7x->2.5x between steps 3000 and 3500 and had NOT converged when the runs were cancelled, so the level advantage must not be quoted as a fixed number. Late escape is exactly what the 0.45%-loss-share economics predicts."
---

# DC `condition_center` — ~6× faster escape from the blind basin (D2)

> Long-horizon continuation of the parity arms
> ([[20260730-dc-parity-arms-null-action-embedding-pedestal]]). Three arms
> relaunched 2026-07-31 under the patched code: E (`condition_center: true`),
> 0 (untreated control), F (AVID's exact 2-layer encoder + concat, no
> centering). Runs still in flight; evals every ~500 steps.

## Trajectories (`eval_action_effect_rel`, null = 0 throughout)

| step≈ | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | **3500 (final)** |
|---|---|---|---|---|---|---|---|
| **arm E** | 0.0257 | 0.0631 | 0.0819 | 0.0920 | 0.1064 | 0.0917 | **0.11479** |
| arm 0 | 0.0033 | 0.0035 | 0.0037 | 0.0046 | 0.0120 | 0.0288 | **0.04564** |
| arm F | 0.0032 | 0.0036 | 0.0040 | 0.0058 | 0.0256 | 0.0392 | **0.05049** |

AVID reference (step 5000): **0.0295**.

**Final values verified 2026-08-01** from the last logged eval of each run
(n=11 evals; `6oyu1inq` / `tr0uovs5` / `86kb01su` @ `_step` 3500). Two
corrections to the readings below, which were written on the step-~3000
snapshot:

- **Arm E's 0.106 was NOT a peak** — the 0.092 dip was single-eval noise and it
  resumed to **0.11479**. Caveat 2 is resolved.
- **The level advantage is SHRINKING, not stable.** E ÷ arm 0 was 3.7× at step
  3000 and is **2.5× at 3500**, because the untreated controls are still
  climbing faster than E. Both controls now sit **above** the AVID reference
  (arm 0 = 1.55×, arm F = 1.71×). Reading 3's "~3.5× higher level" is
  therefore a snapshot of a moving quantity, and the runs were cancelled before
  it converged — **the level claim should not be quoted as a fixed number in
  the thesis.** What survives cleanly is the *acceleration* claim (arm E clears
  0.02 at its first eval; the controls need ~2500 steps).

## Readings

1. **The intervention works and overtakes the reference.** Arm E clears the
   0.02 pre-registered target at its *first* eval, peaks at **0.106 = 3.6× the
   reference**, with `effect ÷ adapter` 0.311 closing on AVID's 0.42 and an
   adapted loss ~18% *below* the control's (0.0357 vs 0.0433) — the
   action-following solution is better on the very objective the blind basin
   optimises. The 0.106 → 0.092 dip is one eval; peak-vs-noise unresolved.
2. **The untreated arms escape on their own — late.** Both controls sat at
   ≤0.005 through step 2000 (which is why the 07-30 parity verdict read
   "null"), then rose sharply: arm 0 to reference level, arm F *above* it.
   **DC blindness is a long transient, not a permanent state.** This was
   predicted by the loss economics before it happened
   ([[20260731-why-wan-copies-the-base-decomposed]] §3: actions are worth
   0.45% of the loss, so escape pressure appears only once the easy variance
   is exhausted).
3. **Framing for the thesis:** `condition_center` = **~6× faster escape**, not
   a binary unlock. The acceleration is the practically relevant property
   (2500 steps of H100 vs an unknown multiple). **REVISED 2026-08-01:** the
   companion "~3.5× higher level" claim does not survive the final evals — the
   gap fell to 2.5× by step 3500 and was still closing when the runs were
   cancelled. Quote the acceleration, not the level.
4. **The architecture question is reopened for long horizons.** Arm F
   (9,344-param AVID encoder) escapes ~2× faster than arm 0 (792k-param
   encoder) without any centering. At ≤2000 steps the two were
   indistinguishable — the 07-30 "architecture exonerated" claim holds only
   in that window.

## Runs closed 2026-07-31 (compute management)

All three arms cancelled by choice (arm 0 first, then E and F at ~step 3600;
checkpoints retained). Final numbers = the step-~3000 evals in the table.
**RESOLVED 2026-08-01 (arm E) / STILL OPEN (the controls):** arm E's 0.106 was
not a peak — it finished at 0.11479. But whether the untreated arms would have
converged to arm E's level is now the *more* pressing question, because the gap
closed from 3.7× to 2.5× over the final 500 steps. If the chapter needs the
acceleration-vs-level split pinned, resume all three from the retained
step-3500 checkpoints and run to convergence — this is a cheap, high-value
run.

## Caveats

- ~~All three runs in flight~~ — final values pulled 2026-08-01, table complete.
- ~~Arm E's dip (0.106 → 0.092) is a single eval~~ — resolved, it was noise.
- **The runs were cancelled before the arms converged**, so no statement about
  the asymptotic level of any arm is supported.
- **No quality metrics were logged on any DC run** (checked across all 18 runs
  in `dc-acwm-robotarm-avid-parity`, 2026-08-01). Whether arm E — the cell that
  carries the thesis spine — improves or degrades perceptual quality relative
  to the frozen base is **unknown**, while the RT-1 cells are known to degrade
  it ([[20260801-wan-rt1-indistribution-plateau]]). This must be measured
  before the spine is written up.
- Step counts are approximate (eval every 500 steps; wall-clock differs
  slightly per arm).

## Related

- [[20260730-dc-parity-arms-null-action-embedding-pedestal]] — the mechanism +
  the null parity arms this continues
- [[20260731-why-wan-copies-the-base-decomposed]] — the economics that
  predicted the late escape
- [[../../20_Tickets/experiments/exp-conditioning-decouple-encoder-bias]] —
  the intervention ticket
