---
type: experiment
date: 2026-07-31
config: probes of ncztxyyo/GATEFIX/TOKENNORM/NOBASE (no new training); attribution job 25097452 on ncztxyyo step_00001000
commit: uncommitted working tree @ 2026-07-31 (probe: generate_wan22_i2v_compare.py --base-attribution)
wandb_run_id: ncztxyyo · tny84p7k-family (GATEFIX) · TOKENNORM · TOKENNORM-NOBASE
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-gatelow-capshift-run/checkpoints/step_00001000.pt
status: completed
deliverable: D2
metrics:
  attribution_base_zero_drel: 1.129
  attribution_base_shuffle_drel: 0.837
  attribution_act_shuffle_drel: 0.0087
  cosine_asymptote_with_oracle_frozen_gate: 0.990
  cosine_asymptote_with_oracle_live_gate: 0.926
  cosine_asymptote_without_oracle: 0.866
  action_worth_share_of_loss: 0.0045
notes: "WHY Wan copies the base, decomposed into four measured components: (1) shared-target convergence — cosine reaches 0.87 even with NO base input; (2) the input oracle — the adapter's own prediction is ~100x more sensitive to the base_pred channel than to actions (drel 1.13 vs 0.009); (3) economics — actions are worth 0.45% of the denoising loss, appearance ~100x more; (4) composition weight modulates the pull (cosine 0.990 at 50% pred weight vs 0.93 at 10%)."
---

# Why the Wan adapter copies the base — the copy, decomposed (D2)

> Follow-up to [[20260731-wan-action-trace-value-pathway-drowns]]. Question:
> after the transport fix (token-norm), why does action-following *erode*
> (0.0113 → 0.0078) instead of climbing like DC's arm E? All numbers below are
> measured; no new training was run for this note except the already-launched
> arms.

## 1. Most of the "copy" is not copying — it is shared-target convergence

`train/adapter_pred_base_cosine` trajectories across the four Wan arms:

| arm | oracle in input? | step 10 | step 200 | asymptote |
|---|---|---|---|---|
| orig (gate frozen 0.5) | yes | 0.44 | 0.96 | **0.990** |
| GATEFIX (gate ~0.9) | yes | 0.44 | 0.95 | 0.943 |
| TOKENNORM | yes | 0.52 | 0.93 | 0.926 |
| **NOBASE** | **no** | 0.52 | 0.82 | **0.866** |

The arm with *no base input at all* still reaches cosine 0.87, and every arm
hits ~0.5 within ten steps. Two networks fitting the same target
(`noise − z0`) necessarily correlate — this component is inevitable and not a
defect. **A high pred–base cosine is therefore NOT by itself evidence of
copying** (correction to how earlier notes read the 0.985).

## 2. The real copying — the input oracle, quantified functionally

`condition_on_base_outputs: true` concatenates the base's prediction into the
adapter's input. Doctored-oracle probe (job 25097452, adapter's OWN prediction,
paired, σ ∈ {0.5, 0.83}):

| perturbation | drel |
|---|---|
| oracle blanked (`base_zero`) | **1.13 / 1.01** |
| oracle swapped for another clip's (`base_shuffle`) | 0.84 / 0.78 |
| actions swapped (`act_shuffle`) | **0.0087 / 0.0082** |

The adapter's computation is **~100× more sensitive to the oracle channel than
to the actions**. Blanking the oracle changes the prediction as much as
replacing it with an unrelated tensor. Functionally, the trained adapter is a
transform of `base_pred` with an action garnish — the mechanical form of the
copy. In cosine terms the oracle contributes the increment above
shared-target convergence: 0.87 → 0.93–0.99.

## 3. Why the optimiser prefers this — the economics, measured

From the σ-sweep (job 25085110): zeroing the actions costs a mean **0.45% of
the denoising loss** (range 0.26–0.59% across σ = 0.1…0.95). With clean
observation frames plus the partially-noised future in the input, the visuals
already pin the target; appearance/domain adjustment offers ~100× more loss
reduction per unit of gradient than action modelling. The action gradient sits
below the noise floor, so gradient descent invests elsewhere. Three
previously-separate observations follow from this one number:

- **Blindness is the default basin** for every adapter on this objective.
- **The DC control arms escaped late** (step ~2500: arm 0 0.0046 → 0.0120,
  arm F 0.0058 → 0.0256): once the easy variance is fit, the residual loss is
  increasingly action-explainable — exactly when the economics says escape
  begins.
- **It is a plateau, not a verdict on actions:** DC arm E's action-following
  solution has ~18% *lower* adapted loss than the blind control (0.0357 vs
  0.0433) — the better solution exists; the blind basin just offers no visible
  path to it. The fixes (clean transport, no shortcut) lower the barrier.

## 4. The composition weight modulates the pull

The loss pulls the adapter's prediction toward the base in proportion to how
much of the composed output the adapter owns: cosine asymptote **0.990** when
the frozen gate weighted pred at 50%, **0.93–0.94** at ~10% (live gate at the
0.9 cap). A second, independent knob on the copy strength.

## Predictions this makes (pre-registered for the running arms)

- **NOBASE** (token-norm + `condition_on_base_outputs: false`, job 25088945):
  with the oracle gone, the adapter must model dynamics itself; its
  action-following should **hold or climb** rather than erode. First eval
  opened at **0.01234** — the highest Wan first-eval yet, cosine asymptote
  0.87 vs 0.93. If its trajectory erodes anyway, the incentive story is
  incomplete and the residual ceiling is elsewhere (e.g. the 0.45% economics
  itself, which no architecture change fixes — that would point at the
  objective, e.g. action-CFG or rollout-based losses).
- Composed-output caveat: NOBASE pays a real price — the adapter must
  reproduce base-quality prediction alone before beating it; expect worse
  early denoise loss. Judge on effect_rel × loss jointly.

## Outcome of the pre-registered NOBASE prediction (3rd eval, 2026-07-31)

**Partially confirmed — CORRECTED at 6 evals (2026-07-31 evening).** At 3
evals NOBASE looked flat (0.0106) and this was logged as "erosion stopped".
Three further evals show slow monotone decline: NOBASE 0.0123 → 0.0090
(−27% over 6 evals) vs TOKENNORM 0.0113 → 0.0070 (−38%). Honest statement:
**removing the oracle SLOWS the erosion ~1.4×, it does not stop it** — the
residual decay needs no shortcut, only the absence of anything paying for
action retention (the 0.45% economics reclaiming the adapter). It also does
**not** produce arm-E-style takeoff — consistent with the 0.45%
economics providing no upward pressure once the leak is plugged. Wan's plateau
with both fixes: ~0.011 (7.5× the gatefix control; reference is 0.0295 on a
full separate UNet). Pushing higher likely requires changing the *objective*
(action-CFG, rollout losses), not the architecture.

Also: DC arm F (AVID's encoder, untreated) reached **0.0392 > reference** at
step ~3000, ahead of arm 0 (0.0288) — the small encoder escapes the blind basin
faster even without centering; the architecture question is reopened for the
long-horizon regime (it was settled only for ≤2000 steps).

## Related

- [[20260731-wan-action-trace-value-pathway-drowns]] — transport (problem B)
- [[20260730-dc-parity-arms-null-action-embedding-pedestal]] — DC mirror (problem A)
- [[20260721-replace-fix-validation-sigma-sweep-action-probe]] — the original
  "pure action-independent domain adjustment" verdict this mechanises
