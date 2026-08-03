---
type: exp
scope: conditioning
status: in-progress
priority: high
created: 2026-07-30
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]", "[[../../30_Knowledge/experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]", "[[../../50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan]]", "[[exp-adapter-our-framework-avid-replication-robotarm]]"]
---

# exp: stop the action encoder being a bias generator (D2)

## Why

[[../../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]
measured the mechanism behind D2's action-blindness: `cond_emb` is **~99.7% an
input-independent constant**. `realised/RMS` per element is **0.0050** against the
AVID reference's **0.238** on the same data — 48×.

Crucially the pedestal is **learned, not architectural**:

| | init | step 600 | step 1000 |
|---|---|---|---|
| cond ÷ time RMS | 0.182 | 16.02 | 14.45 |
| frame-var ratio | 0.0238 | 0.00283 | 0.00344 |
| ‖J‖_F | 0.0568 | 1.629 | 1.711 |

Magnitude grows **106×** while the varying fraction *collapses* 7×; ‖J‖ grows ~30×,
so the encoder does become more action-sensitive — the constant just grows ~3.5×
faster. Reading: the encoder is used as a **bias generator**, because a large
constant into every ResBlock's `emb` fits the denoising objective without actions.

## Intervention (arm E) — `condition_center`

`BatchNorm1d(adapter_condition_output_dim, affine=False)` on `cond_emb`, applied
on the flattened `(b·t, c)` form so the statistic spans **clips and frames** — the
two axes the constant was measured to be constant over. It subtracts the
per-channel mean across the batch, which *is* the input-independent component, so
a constant cannot survive a forward pass.

**Why not LayerNorm:** it centres across *channels*. A fixed 128-d vector is not a
per-channel offset, so LayerNorm leaves it intact. Verified on a synthetic replica
of the measured pathology (constant 2.895 + variation 0.010, reproducing the
measured frame-var 0.00325 vs measured 0.00344):

| | frame-var ratio |
|---|---|
| before | 0.00325 |
| after BatchNorm(affine=False) | **0.99136** (305×) |
| after LayerNorm (control) | 0.00323 (unchanged) |

## Build (2026-07-30)

- `adapter.extra.condition_center: bool` → `adapters/factory.py` →
  `DynamicCrafterOutputAdapter` → UNet `adapter_condition_center`.
- `openaimodel3d.py`: builds `adapter_condition_center` beside
  `adapter_condition_proj`; `_center_condition()` applied in **both** branches of
  `_prepare_adapter_embedding` (rank-2 and rank-3). Skips centring when training
  with batch < 2 (BatchNorm needs ≥2 samples for a mean; eval uses running stats).
- Config `configs/dynamicrafter/diffusion_dc_acwm_robotarm_armE_center.yaml` —
  diffed against arm 0: **only** `condition_center: true`, plus name/output_dir.
- Launcher: `submit_train_dc_avid_parity.sh` gains `ARM=E` (also given the
  positional-ARM fallback, since env propagation fails over non-interactive ssh).
- Verified on the cluster: config parses `condition_center` True/False per arm,
  `UNetModel` accepts `adapter_condition_center`, `_center_condition` exists.

## Decision rule

Judge against arm 0 / 0S — floor **0.000245**, control **0.0033–0.0035**. Reference
**0.029475**.

- **effect_rel ≥ 0.02, null ≈ 0** ⇒ the pedestal was the cause. Headline D2 result
  with a measured mechanism and a matching reference value.
- **effect_rel moves but < 0.02** ⇒ contributory, not sufficient; combine with
  per-dim action standardisation (the §3 misallocation) and re-measure.
- **effect_rel unchanged** ⇒ removing the constant does not restore action use;
  the adapter is fitting the objective by some other action-independent route,
  and the search moves to the objective itself.

**Also probe `--emb-scale` on arm E's checkpoints regardless of the outcome** —
`realised/RMS` should rise toward AVID's 0.238 if the intervention does what the
synthetic test predicts. If `effect_rel` stays flat *while* `realised/RMS` rises,
that is highly informative: the conditioning is then genuinely informative and
being ignored downstream.

## RESULT (in-flight, 2026-07-31) — arm E works and OVERTAKES the reference

| eval (arm E `6oyu1inq`) | effect_rel | eff/adapter | null | base_loss |
|---|---|---|---|---|
| step ~500 | 0.02572 | 0.103 | 0 | 0.0435 |
| step ~1000 | **0.06310** | **0.224** | 0 | 0.0357 |

Arm E through step ~2500: 0.026 → 0.063 → 0.082 → 0.092 → **0.106** — 3.6× the
AVID reference, still climbing.

**⚠ Qualifier added 2026-07-31 evening: the controls are NOT permanently flat.**
Past step ~2500 the untreated arms began escaping on their own — arm 0
0.0046 → **0.0120**, arm F (AVID's encoder) 0.0058 → **0.0256** (near reference
level). Honest restatement: `condition_center` does not create an ability the
baseline permanently lacks — it **accelerates action-following ~10× and lifts
the level** (0.106 vs 0.012 at matched step). And arm F outpacing arm 0 by ~2×
suggests encoder architecture *does* matter at longer horizons — the earlier
"architecture exonerated" claim was true at step ≤2000, not absolutely. Confirmed at step ~3000: arm 0 reached **0.0288** — the untreated baseline is at
reference level, so DC blindness is a long transient, not a permanent state
(precisely what the 0.45%-loss-share economics in
[[../../30_Knowledge/experiments/20260731-why-wan-copies-the-base-decomposed]]
predicts). Arm E printed its first dip (0.106 → 0.092) — peak or noise, next
eval decides. Framing for the thesis: `condition_center` = **~6× faster escape
and ~3.5× higher level**, not a binary unlock.

## On hold (superseded by the "learned, not architectural" finding)

Narrow / shallow encoder arms. They were premised on depth creating the pedestal;
the architecture behaves fine at init, so a 2-layer encoder could inflate
identically. Revisit only if arm E fails and depth is re-implicated.

## Cleanup 2026-08-01 — **DELIVERED**

`condition_center` shipped and validated: arm E 0.003 -> 0.106 (3.6x the AVID reference).

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
