---
type: experiment
date: 2026-07-16
config: configs/diffusion_wan22_avid_xattn_gatelow_metaworld.yaml, configs/diffusion_wan22_avid_xattn_i2v_metaworld.yaml, configs/diffusion_wan22_avid_xattn_replace_metaworld.yaml
commit:
wandb_run_id: bcipghvw, uea10230, 5cxstyh4
ckpt_path:
status: completed
deliverable: D2
metrics:
  gatelow_eval_base_loss: 0.1133
  gatelow_eval_denoise_adapter_delta: 0.00104
  overfit_eval_base_loss: 0.0793
  overfit_eval_denoise_adapter_delta: 0.00080
  overfit_eval_probe_denoise_delta: -0.00321
  replace_eval_base_loss: 0.1227
  replace_eval_denoise_adapter_delta: -0.00125
  replace_eval_probe_denoise_delta: -0.11193
notes: pulled directly via wandb API 2026-07-16, not eyeballed from dashboard
---

# exp: three Wan2.2 xattn-adapter runs — adapter converges to clone the base, not to use actions

## Runs

| label | wandb run | project | config (`config.experiment`, confirmed via API) | composition | gate_bias | state | train steps logged |
|---|---|---|---|---|---|---|---|
| gatelow | `bcipghvw` | `Wan2.2-avid-xattn-i2v-metaworld` | `diffusion_wan22_avid_xattn_gatelow_metaworld` | mask_mix | 0.0 | crashed | 624 |
| overfit | `uea10230` | `Wan2.2-avid-xattn-i2v-metaworld` | `diffusion_wan22_avid_xattn_i2v_metaworld` | mask_mix | **4.0 (unfixed)** | crashed | 319 |
| replace | `5cxstyh4` | `Wan2.2-avid-xattn-replace-i2v-metaworld` | `diffusion_wan22_avid_xattn_replace_metaworld` | replace | n/a (no gate) | running | 1178+ |

**Correction to the record:** the "overfit" run linked 2026-07-16 does **not** use
the gate-fixed config despite the prior session's decision to fix
`gate_bias→0.0` for this test ([[../../20_Tickets/experiments/exp-training-single-clip-overfit]]).
Its `config.experiment` field resolves to `diffusion_wan22_avid_xattn_i2v_metaworld.yaml`,
which on disk still has `gate_bias: 4.0` and `shortcut_anchor_prob: 0.6`. A
properly gate-fixed overfit run is still outstanding.

## Method

Pulled `run.config`, `run.summary`, and full `run.history()` for all three
directly via `wandb.Api()` — config identity confirmed from the logged
`experiment` field (cross-checked against the actual YAML on disk, since
configs in this project have been hand-edited mid-session before), not
inferred from run names.

## Finding

`train/denoise_adapter_delta` (= `denoise_base_only − adapted_loss`; negative
= adapted worse than base alone) for all three runs:

| run | first-20-step mean | last-20-step mean | last-20-step range |
|---|---|---|---|
| gatelow | −0.462 | **+0.00125** | [0.00103, 0.00164] |
| overfit | −0.454 | **+0.00063** | [−0.00049, 0.00090] |
| replace | −1.525 | **−0.00328** | [−0.00617, +0.00331] |

In all three, the delta decays from a large negative value (untrained
adapter actively hurts) to a value ~2–3 orders of magnitude smaller than the
loss scale (~0.1) within 60–150 steps. `train/loss` tracks
`train/denoise_base_only` almost exactly from that point on in every run.
**The "loss decreasing toward ~0.14" the user observed is the composed
output converging toward the frozen base's own denoising floor, not to a new
optimum clearly below it** — see the zero-variance probe trajectory below,
which shows this convergence is still slowly continuing (not a hard
plateau) but has not yet crossed into net-positive-vs-base in any of the
three runs.

**The gate ratio does not explain the transient.** Under `mask_mix`
(`adapted_model.py:161-169`), composition is
`base·σ(gate+gate_bias) + adapter_pred·(1−σ(gate+gate_bias))` — the adapter
output is a standalone competing prediction, not a residual. `gate_bias: 4.0`
(σ≈0.982) should, per the config's own comment, give "identity-at-init" —
i.e. a ~28× smaller initial deviation from base than `gate_bias: 0.0`
(σ=0.5). The observed initial deltas are nearly equal (−0.454 vs −0.462).
This means the collapse-to-base-clone behavior is being driven by what the
adapter's own prediction head learns, not by how much weight the gate
assigns it — gate_bias was a necessary fix (saturation is real, see
[[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]]) but is **not
sufficient** to make the adapter diverge from the base and use the action
signal.

**Downstream consequence differs by composition mode:**

- **mask_mix (gatelow, overfit):** adapter converges to predict ≈what base
  predicts → mixing at any gate ratio gives back ≈base → composed video ≈
  base's own rollout, action edits don't move it. Matches the user's direct
  observation ("overfit is not overwriting the base model at all, videos
  look like copies"). Eval quality metrics confirm no real gain: gatelow is
  a wash (adapted marginally better on MSE/PSNR/SSIM, worse on FID/FVD/LPIPS);
  overfit's adapted output is **worse than base on all six logged quality
  metrics** (MSE 0.0112 vs 0.0110, PSNR 19.50 vs 19.60, SSIM 0.866 vs 0.870,
  FID 272 vs 225, FVD 4287 vs 4057, LPIPS 0.466 vs 0.444).
- **replace (no gate, no escape hatch):** single-step training loss also
  converges to ≈base's floor (0.109 vs 0.106), suggesting the same
  "clone base" strategy — but the clone is much weaker than it looks on
  training loss. `eval_probe_denoise_delta` (out-of-training-distribution) is
  **−0.112**, a 20–30× larger gap than the mask_mix runs' probe deltas
  (−0.003 to −0.005). Decoded eval-video quality is catastrophic: PSNR 10.98
  vs base 15.99 (−5 dB), SSIM 0.299 vs 0.806, FID 521 vs 66 (8×), FVD 4877 vs
  1260 (4×), LPIPS 0.796 vs 0.360 (2×). ~~Small single-step prediction error
  compounds over the 25–50-step iterative sampler into genuinely bad video~~
  **Corrected 2026-07-19: these quality metrics are 1-STEP generations**
  (`quality_eval_num_steps: 1` in the config; passed at
  `trainer.py:575,632,704`) — they probe `v(pure noise, σ=1)` directly, the
  zero-leakage endpoint that flat-uniform training σ almost never
  supervises. No sampler compounding is involved in these numbers. Full
  reframe: [[../../20_Tickets/bug-adapter-replace-generation-flat-since-init]].
  Matches "the videos are bad."

## Probe-delta trajectory (zero-variance signal, added 2026-07-16 follow-up)

`eval_probe_denoise_delta` is scored on **one frozen** `(x_t, t, noise, target)`
triple fixed at the first eval call of each run and reused for every
subsequent eval — `probe_denoise_base` is therefore a per-run constant
(0.0862 gatelow / 0.1214 overfit / 0.0880 replace; not comparable across runs,
each run's probe clip differs). Since the input never changes, any change in
`probe_denoise_adapted` between two eval cycles is 100% attributable to
weight updates, not sampling variance — the cleanest signal available.

```
gatelow:  step 300: -0.00325  →  step 600: -0.00131
overfit:  step 75: -0.00738 → 150: -0.00418 → 225: -0.00370 → 300: -0.00321
replace:  step 300: -0.205 → 600: -0.131 → 900: -0.112
```

**Correction to the framing above:** this is not a dead plateau at zero — the
gap is shrinking monotonically at every logged checkpoint in all three runs.
The adapter keeps learning throughout; it has not stopped. What holds is
that it is closing the gap **from below** — still net-worse than base at the
last logged point in every run — and has not crossed into net-positive
(beats base on this fixed test case) in any of them. gatelow/overfit are
already close (~0.001-0.003, ~1-3% of the loss scale); replace remains far
off (~0.11, ~90% of its own scale) and decelerating hard (Δ per 300 steps:
-0.074, -0.019 — shrinking fast).

**Caveat:** all three runs crashed after only 2-5 eval cycles — not enough
points to tell "still ramping, would cross zero with more steps" apart from
"asymptoting just short of parity." Re-run to a comparable, non-crashed step
count before treating either reading as settled.

**Correction (2026-07-19), replace run only — full history pulled, not just
the first 3 points.** `5cxstyh4` ran to step 3600 (not 900, as the excerpt
above suggested). Over that fuller range, `eval_probe_denoise_delta` does
**not** keep closing — it plateaus at −0.067 to −0.11 from step ~900 onward
(900: −0.112, 1800: −0.101, 2400: −0.074, 3000: −0.072, 3600: −0.067),
noisy but flat, not trending toward zero. The "still closing" framing above
was based on an incomplete read of this run and should not be trusted for
`replace`. Separately, and more importantly: **`eval_denoise_adapter_delta`
(batch-averaged over fresh, randomly-resampled held-out batches) reaches
parity with base by step ~900 and is sometimes net-positive** (+0.0073 at
1800, +0.019 at 3000) — a completely different picture from the frozen
probe's persistent −0.07 to −0.11 gap over the *same* step range. Full
writeup: [[../../20_Tickets/bug-adapter-replace-generation-flat-since-init]].

## Eval-quality trajectory: replace is flat at the noise floor since step 0 (2026-07-18 follow-up)

Pulled full `run.history()` eval-quality columns for `5cxstyh4` (replace) and
`bcipghvw` (gatelow) directly via `wandb.Api()`, 2026-07-18:

| run | step | `eval/adapted/fid` | `eval/adapted/psnr` | `eval/adapted/ssim` |
|---|---|---|---|---|
| replace | 0 | 530.18 | 11.36 | 0.329 |
| replace | 900 | 521.44 | 10.98 | 0.299 |
| replace | 1500 | 524.35 | 11.16 | 0.318 |
| replace | 1800 | 522.68 | 11.21 | 0.330 |
| replace | 2100 | 520.70 | 11.08 | 0.315 |
| replace | 2400 | 520.88 | 11.32 | 0.333 |
| gatelow | 0 | 482.99 | 13.17 | 0.433 |
| gatelow | 300 | **81.34** | **16.12** | **0.797** |
| gatelow | 600 | 96.82 | 16.50 | 0.827 |

**Replace's decoded-video quality has not moved off its step-0 (effectively
untrained) value across 2400+ steps** — every metric at step 2400 is within
noise of step 0, despite `train/loss` descending 1.63→0.109 over the same
window (see Finding, above). This is a materially different picture from
"slowly closing the gap": the single-step training objective is visibly
improving while the full 25-50-step generation shows **zero** measurable
transfer.

**Gatelow is the sharp counter-example.** Same adapter family
(`hidden_dim: 256`, zero-init final layer — `configs/diffusion_wan22_avid_xattn_*`),
same backbone, same dataset. Its `eval/adapted/fid` collapses from
noise-level (483, matching replace's ballpark) to near-base quality (81, vs.
base's own 66-70) within just **300 steps** — 8x fewer steps than replace's
still-flat 2400. The only structural difference is composition: gatelow
always has the frozen base in the output (`base·gate + adapter·(1-gate)`);
replace has none (`adapter_output` alone).

**Leading hypothesis, revised 2026-07-18 (analysed estimate — not yet
confirmed):** exposure bias / off-manifold drift, not adapter capacity. A
plain capacity shortfall predicts graceful degradation (worse than base, not
noise) — it doesn't explain why matching the base's loss coexists with a
literally-noise-level video. The training loss is measured only on
ground-truth-anchored `x_t` (built from real encoded clips); `generate()`
integrates the adapter's **own** predictions for 25-50 steps from pure noise,
so every step after the first feeds the adapter an input the training
objective never constrained it on. A small systematic bias — fully
compatible with a low *averaged* L2 loss — compounds across the rollout.
Gatelow's composition (`base·gate + adapter·(1-gate)`) keeps the base
anchoring every step, which pulls drift back; `replace` has no such anchor.
Capacity may be a secondary factor but doesn't resolve the core paradox on
its own. Not ruled out: a wiring/parameterization bug specific to the
`replace` path through `generate()`/`_ComposedDiT` (`models/base/wan_ti2v.py`)
— single-step training loss and full-rollout generation have never been
directly compared on the *same* checkpoint (see the investigation ticket).

Investigation plan: [[../../20_Tickets/bug-adapter-replace-generation-flat-since-init]].

## Interpretation

The dominant failure mode across every composition mode tested is not
gate saturation (already fixed in the gatelow config) and not raw gradient
flow (replace bypasses the gate entirely and shows the same clone-base
convergence on training loss). It is that **the adapter's competing/residual
prediction has an easier gradient path to "reproduce the base's own strong
prior" than to "use the action conditioning to diverge and do better."**
This is a fresh, sourced confirmation — on the WAN2.2 xattn mask_mix/replace
runs specifically — of the weak-action-signal finding already logged in
[[20260907-flow-shortcut-weak-action-signal]].

## Next steps (not yet run)

- A genuinely gate-fixed overfit run (`gate_bias: 0.0`, matching
  `diffusion_wan22_avid_xattn_gatelow_metaworld.yaml`) is still outstanding —
  worth running to close the loop, but given gate_bias didn't distinguish
  the transient here, don't expect it alone to fix the clone-base behavior.
- The action-shuffle ablation ([[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]])
  and the action-free shortcut isolation
  ([[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]]) are the tests
  that can actually confirm/deny "the adapter ignores the action" as
  distinct from "the adapter has converged to a degenerate optimum
  regardless of action." Both are higher priority now than another
  gate-bias sweep.

## Resolution addendum (2026-07-21) — what later measurements settled

Read [[20260721-replace-fix-validation-sigma-sweep-action-probe]] alongside
this note; it supersedes several conclusions here:

1. **All generation-based metrics in this note are invalid.** The eval-video
   path fed the xattn adapter the aggregated `action` instead of the
   per-frame `action_seq` it trained on (silent fallback, output collapse
   cos 0.997→0.634). Every decoded-video number for `bcipghvw`, `uea10230`,
   `5cxstyh4` (and likely `xb76ptw2`) measures that bug, not the adapter.
   Training-seam losses/deltas here remain valid.
2. **"Clones base" is now measured, not inferred**: per-σ sweep shows
   cos-to-base ≥ 0.996 at every noise level, and the action probe shows
   shuffled/zeroed actions move the loss by <1e-4 — the adapter is fully
   action-blind; the shuffle ablation this note asked for has run.
3. The single-clip overfit rerun this note requested happened (`uxrst2k5`,
   gate_bias 0.0): gate saturated 0.5→0.99 from the balanced init, grad norm
   died — saturation is loss-driven, not init-driven.

## Related

- [[20260721-replace-fix-validation-sigma-sweep-action-probe]] — resolution + follow-up measurements (2026-07-21)
- [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] — the gate fix this session assumed would resolve the flat-loss pattern; it didn't, on its own.
- [[../../20_Tickets/experiments/exp-adapter-xattn-gatelow-metaworld-run]]
- [[../../20_Tickets/done/exp-training-single-clip-overfit]]
- [[../../20_Tickets/done/exp-adapter-wan-replace-metaworld-run]]
- [[20260907-flow-shortcut-weak-action-signal]]
- [[../../20_Tickets/done/exp-conditioning-action-shuffle-ablation]]
- [[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]]
