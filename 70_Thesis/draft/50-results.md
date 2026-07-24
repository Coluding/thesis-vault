---
section: results
status: drafting
deliverable: D2/D3/D4
last_updated: 2026-07-24
sources: ["[[../../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]]", "[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]", "[[../../50_Decisions/open/second-dataset-action-informativeness]]"]
---

# 5. Results

> **Every number in this chapter cites a run** (wandb id + ckpt + commit).
> No unsourced metrics (hard rule 8). No promoting `_not yet run_` to a
> result (Part 12). Video placeholders `[[FIG:...]]` mark where the qualitative
> grids go — replace with the exported wandb `eval_step_grid` frames.

## 5.1 Adapter-family comparison (D2)
_Stub — cross-family comparison pending (see the ablation plan)._

## 5.1.x Diagnostic: why a naive adapter on a strong frozen base collapses to base-parity (D2)

> Source note (all numbers): [[../../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]].
> Backbone: frozen Wan2.2-TI2V-5B. Adapter: 34M ActionWan, cross-attention
> action injection. Dataset: MetaWorld `five_task_diverse`. Commit `07ec01bd`.

On MetaWorld, the action-conditioned adapter converges to a **base-clone at
base-parity loss** regardless of composition. We isolate *why* with a
single-clip overfit triangle (removes the data-scale confound) plus a
full-data run with both optimization countermeasures enabled. Three
independent failure factors emerge; addressing all of them still does not
produce action-following — because the benchmark's actions carry almost no
loss-reducing signal given the observation.

**Table 5.x-1 — Single-clip overfit triangle (training seam).** Gate and
gradient-norm trajectories show which trap is active.

| Run (wandb) | composition | gate | base-output as input | denoise Δ (base−adapted), start → end | gate mean, start → end | adapter grad-norm, start → end | pred-vs-base cosine, start → end |
|---|---|---|---|---|---|---|---|
| `uxrst2k5` | mask_mix | uncapped | yes | −0.90 → +0.0008 | 0.50 → 0.99 | 4.4 → 0.003 | — |
| `o79ki0ul` | mask_mix | uncapped | **no** | −0.86 → +0.0012 | 0.50 → **0.992** | 3.43 → **0.005** | 0.008 → **0.031** |
| `o9113j4h` | mask_mix | **cap 0.9** | no | _needs verification — run failed, re-run pending_ | | | |
| `rxzwh4ak` | replace | none | no | −0.86 → **−0.30** (worse than base) | — | — | — |

**Table 5.x-2 — Intervention run: gate-cap + σ-shift, full data
(`hvxlbfjx`, killed @900 steps).** Both optimization traps addressed.

| metric | adapted | frozen base | reading |
|---|---|---|---|
| denoise Δ (base−adapted) | — | — | +0.0015 (≈ 0) |
| gate mean (end) | 0.894 | — | held at the 0.9 cap (uncapped → 0.99) |
| adapter grad-norm (end) | 0.027 | — | **alive** (uncapped → 0.005) — cap works |
| pred-vs-base cosine (end) | 0.856 | — | **base-clone** |
| FID | **64.57** | 64.70 | hair-thin win |
| FVD (I3D) | 1124.9 | 1260.1 | |
| PSNR | 16.55 | 16.12 | |
| MSE | 0.0221 | 0.0244 | |
| SSIM | 0.839 | 0.821 | |
| LPIPS | 0.319 | 0.336 | |

`[[FIG:hvxlbfjx-eval_step_grid]]` — GT | base | adapted step-size grid,
full-data cap+σ-shift run (qualitatively indistinguishable from base).

**Interpretation.** Three factors, separated by the triangle:

1. **Gate saturation (optimization, real, fixable).** Uncapped runs drive the
   mask gate to ~0.99 within ~70–150 steps and the adapter gradient collapses
   to ~0.003–0.005 — the delta path receives no signal and learning stalls
   (`uxrst2k5`, `o79ki0ul`). The cap defuses it: `hvxlbfjx` holds the gate at
   0.9 and the gradient stays alive at 0.027 (5× larger).
2. **Identity-on-base-output (optimization, real).** With the base velocity
   given as an *input* (`condition_on_base_outputs`), the cheapest solution is
   to copy it — the pred-vs-base cosine reaches 0.86 (`hvxlbfjx`). Removing
   the input stops the copy (cosine 0.03, `o79ki0ul`), but then the gate trap
   dominates instead. The two traps are independent.
3. **Adapter capacity (real, but expected).** With neither gate nor base input
   (`replace`, `rxzwh4ak`), the 34M adapter denoises *worse* than the frozen
   5B (Δ −0.30); its overfit pixel metrics are the MSE-optimal-blur signature
   (MSE 0.0046 vs 0.0223 and PSNR 23.4 vs 16.5, but FID 409 vs 111). The small
   adapter genuinely needs the base — which the thesis composition retains.

**The load-bearing negative result.** Even with (1) and (2) addressed and
supervision σ-shifted toward high noise, the full-data adapter (`hvxlbfjx`)
**still converges to a base-clone at base-parity** (cosine 0.86, denoise
Δ ≈ 0, FID within 0.1 of the base). Its only measurable edge over the base is
action-*independent* domain calibration — consistent with the per-σ sweep,
where the adapter beats the base solely at σ≈0.05
([[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]).
Fixing the optimization does not create an action signal that the data does
not contain: on MetaWorld scripted demonstrations the frozen base already
predicts the future accurately without the actions, so the adapter has no
gradient pressure to use them. This motivates evaluating on an
action-informative benchmark (§ dataset decision), where the same
countermeasures — now validated as mechanisms — have signal to act on.

_Pending to finalize this subsection: (a) re-run `o9113j4h` for the third
triangle arm; (b) refresh `rxzwh4ak` final numbers on completion; (c) the
action-shuffle probe on `hvxlbfjx`'s checkpoint as the direct
action-blindness measurement; (d) export the `eval_step_grid` videos for the
`[[FIG:...]]` placeholder._

## 5.2 Shortcut few-step rollout (D3)
_Stub — no sourced runs yet._

## 5.3 Combined action + shortcut (D4)
_Stub — no sourced runs yet._
