---
last_updated: 2026-07-21
status: pre-results
---

# Experiment State

> What experiments have **actually run** and what came out. Not roadmap, not
> vision — the literal current state. AI overwrites this when runs finish
> or get killed.
>
> **Status: pre-results.** The framework is mostly built ([[architecture]])
> but the thesis lacks evidence-backed numbers across the four
> deliverables. This doc is deliberately short until D2 has its first
> ablation result. Per [[../CLAUDE]] hard rule 8: do not list a result as
> measured without citing a real run.

## What "exists" (codebase / configs)

The configs in `configs/` enumerate **planned experiments**, not finished
ones. They are the most concrete inventory of what the framework can run.
Grouped by deliverable:

### D1 — framework (configs that exist to demonstrate the taxonomy)

| Config | Adapter family | Backbone | Notes |
|---|---|---|---|
| `diffusion_lora_action.yaml` | LoRA | diffusers (dummy or HF) | Smallest "action-conditioning works" demo |
| `diffusion_hyper_lora_action.yaml` | Hypernetwork → LoRA | diffusers | Hypernetwork emits LoRA weights |
| `diffusion_multimodal_hyper.yaml` | Hypernetwork | — | Multimodal conditioning smoke test |
| `diffusion_hidden_unicon_decoder.yaml` | Hidden-state (UniCon) | diffusers | UniCon-style decoder hidden-state adapter |
| `diffusion_hidden_replace_decoder.yaml` | Hidden-state | diffusers | Replace-decoder variant |
| `diffusion_hidden_full_skip_controlnet.yaml` | Hidden-state | diffusers | Full-skip controlnet-style |
| `diffusion_output_dynamicrafter.yaml` | Output | DynamiCrafter | Video diffusion + output adapter |
| `diffusion_output_avid_training_test.yaml` | Output | DynamiCrafter (AVID-style) | AVID replication starting point |
| `diffusion_hyperalign_action.yaml` | Hypernetwork (HyperAlign) | diffusers | HyperAlign replication |
| `diffusion_hyperalign_fake_action.yaml` | Hypernetwork (HyperAlign) | diffusers | HyperAlign with synthetic action |
| `diffusion_hyperalign_metaworld.yaml` | Hypernetwork (HyperAlign) | diffusers / MetaWorld | Closest D2 config |
| `opensora_output_adapter.yaml` | Output | OpenSora | OpenSora wired _partial_ |
| `test_dynamicrafter_hyperalign_unet.yaml` | Hypernetwork | DynamiCrafter U-Net | Test-only |
| `test_dynamicrafter_metaworld_unet.yaml` | — | DynamiCrafter U-Net | Test-only |

### D3 — shortcut configs

| Config | Notes |
|---|---|
| `flow_output_shortcut.yaml` | Flow matching + output adapter + shortcut losses |
| `flow_output_shortcut_velocity.yaml` | Flow matching with velocity prediction + shortcut |
| `flow_hyper_shortcut_stepwise.yaml` | Hypernetwork + step-wise shortcut |
| `diffusion_output_shortcut_noise.yaml` | Diffusion (noise pred) + shortcut output adapter |
| `diffusion_output_shortcut_velocity.yaml` | Diffusion (velocity pred) + shortcut output adapter |
| `diffusion_output_dynamicrafter_shortcut_test.yaml` | DynamiCrafter + shortcut adapter (test) |

### D2/D4 — combined / action-conditioned

The clearest entrypoint is `scripts/train_hyperalign_metaworld.py`, paired
with `configs/diffusion_hyperalign_metaworld.yaml`. Working tree is
modified at HEAD — there's in-flight work on this exact path.

### Multimodal extension — compositional wired to real DynamiCrafter (2026-06-26)

The multi-stream output world model (`multimodal/` subpackage) is **built and
contract-tested**, but has **still not been trained on a real backbone** — so it
produces no experimental result yet. It sits here, not in the "what has actually
run" table, deliberately.

- **Substrate (2026-06-10, commit `b09e8d5`):** `MultiModalAdaptedModel` +
  compositional `LearnedMaskFusion` + additive `TrivialFusion` + per-modality
  timestep diffusion trainer, overfit-tested on a dummy base.
- **Real-backbone TRUE compositional (2026-06-26, working tree):** the
  contribution (one AVID adapter per modality emitting Δ_m + a learned mask
  `m ∈ ℝ^{n+2}`, `docs/composite (2).png`) is now implemented against DynamiCrafter
  — `ModalityEncoder` (video←m: each modality's tokens injected into *only its own*
  adapter's `context` cross-attention) + per-modality `build_adapter` + `VideoReadout`
  (m←video) + `LearnedMaskFusion`. (A first cut wired a single shared adapter /
  workhorse path and was corrected to the per-modality structure.) Contract-tested
  in `tests/test_multimodal_real_backbone.py` (per-modality token routing, text
  boundary preserved, mask + bidirectional grads) and **smoke-runs end-to-end on
  the real DynamiCrafter UNet** (random weights, 35.2M trainable params) — **wiring
  that runs, not a trained result.**

Diffusion-only (`MultiModalTrainer` raises `NotImplementedError` for flow bases).
Channel-stack / single-joint baseline variants still not built. See
[[architecture]] §"Multimodal output adapters" and
[[../50_Decisions/open/multimodal-adapter-broadening]] (build status 2026-06-26).
**Next:** first real run (DynamiCrafter ckpt + GPU + real proprio loader) → an
`exp-adapter-*` ticket → an experiment note.

## What has actually run

| Experiment | Config | Status | wandb run id | ckpt | Result |
|---|---|---|---|---|---|
| Flow-matching base — shortcut, no-shortcut, + diffusion inversion-shortcut (first post-pivot sample batch, MetaWorld) | _nv_ (artifacts `data/results/20260629/`) | completed | _nv_ | _nv_ | base_loss converges cleanly on all three (flow ~0.13–0.15 @ ~15k steps; diffusion ~0.07 @ ~1.4k); shortcut term well-behaved per-rung. **Samples still poor** — blur/fog/colour drift, degrading over rollout. Loss is okay, generation is the blocker. See [[../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]] |
| AVID shortcut on larger MetaWorld (`anchor_prob=0.45`) | `data/results/20261706/config_run_anchor_prob=045.yaml` | running | _nv_ (project `avid-shortcut-metaworld-0.45`) | _nv_ | base_loss stable (~0.06–0.07); **shortcut_direction_loss volatile** (~0.01–0.12); prediction degraded on larger data. See [[../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]] |
| **Cross-attn action injection** (per-frame action tokens, AVID adapter, WAN2.2 i2v MetaWorld) (2026-07-12) | `diffusion_wan22_avid_xattn_i2v_metaworld` | killed @ step 2661 | `xb76ptw2` (project `Wan2.2-avid-xattn-i2v-metaworld`), commit `1c77db61` | _nv_ | **Negative.** Adapter **worse than frozen base on all 6 eval metrics** (PSNR 18.74 vs 18.79, SSIM 0.845 vs 0.867, MSE 0.0134 vs 0.0132, LPIPS 0.546 vs 0.422, FVD 2134 vs 2008, FID 410 vs 166 ≈ 2.5×). Loses the reconstruction gain the AdaLN arm had → moving action into cross-attn is a step back, not the fix. Evidence for **capacity** over **injection-mechanism** hypothesis. Confounds: early kill @ 2661 steps, different MetaWorld subset. See [[../30_Knowledge/experiments/20260712-wan-xattn-action-no-improvement]] |
| Flow shortcut, action-conditioned, **first REAL pretrained WAN base** (2026-07-09) | _nv_ (artifacts `data/results/20260907/`) | completed | _nv_ | _nv_ | First run with a genuinely pretrained WAN frozen base (prior runs loaded **random** WAN weights → adapter learned everything from scratch). Samples now **coherent** (base prior working). `base_loss` flat ~0.15 is *expected* with a strong frozen base — real metric is base-vs-adapted delta. On **button-press** (`button/`), the **quantified base-vs-adapted delta** (eval-metric screenshots, eyeballed): adapter **clearly beats the frozen base on reconstruction and the gap widens with training** — PSNR ~15.6→**16.8** (base flat ~15.6), SSIM →**0.833** (base ~0.80), MSE →**0.021** (base ~0.0275). But it **degrades perceptual/distribution** metrics (LPIPS/FVD/FID rise above base) = **regression-to-the-mean blur**. **Overturns "adapter isn't doing much"**: for a planning world model PSNR/MSE is what counts and the adapter helps; blur is a separable MSE-objective (possibly shortcut-distillation) tradeoff. Next: shuffle test (action-following vs task prior). See [[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]] |
| Wan2.2 xattn adapter, gatelow (mask_mix, gate_bias=0) + overfit (mask_mix, gate_bias=4 unfixed) + replace (2026-07-16) | `diffusion_wan22_avid_xattn_gatelow_metaworld` / `diffusion_wan22_avid_xattn_i2v_metaworld` / `diffusion_wan22_avid_xattn_replace_metaworld` | gatelow+overfit crashed, replace running | `bcipghvw` / `uea10230` / `5cxstyh4` (project `Wan2.2-avid-xattn-{i2v,replace-i2v}-metaworld`) | _nv_ | **Negative — gate_bias fix alone is insufficient.** In all three, `denoise_adapter_delta` collapses from a large initial penalty to ~0 within 60-150 steps, regardless of gate_bias (0.0 vs 4.0 give nearly identical transients) — the adapter's own prediction converges to **clone the frozen base** rather than diverge using the action signal. mask_mix runs: composed output ≈ base, eval quality is a wash (gatelow) or worse-than-base on all 6 metrics (overfit, which turned out to still be on the unfixed `gate_bias=4.0` config). replace: single-step loss also converges to ≈base, but out-of-distribution probe delta (−0.112) and decoded video quality (PSNR −5dB, SSIM 0.30 vs 0.81, FID 8× worse) reveal the clone is much weaker than training loss suggests. **Update 2026-07-18:** replace's eval-quality metrics (FID/PSNR/SSIM) are flat at the step-0 noise floor across 2400+ steps, vs. sibling gatelow's collapse from the same noise-level FID to near-base quality within 300 steps — re-analysis 2026-07-19: replace's rollout endpoint is statistically identical to the *untrained zero-init* endpoint (x never moves off the initial noise), every good metric is training-seam and every bad one generation-seam, gatelow's decent videos never certified the adapter's generation path (FID 81 vs base 66 is consistent with the adapter contributing ≈0 at generation and the mask letting base through) — plus a **verified train/inference conditioning mismatch**: the eval-video path passes only the aggregated `action` vector while the xattn adapter was trained on per-frame `action_seq` (falls back to a single OOD token, flagged by an in-code TODO). Investigation plan in [[../20_Tickets/bug-adapter-replace-generation-flat-since-init]]. See [[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]] |

| **Generation-eval conditioning bug found+fixed; σ-sweep + action probe; gatelow overfit** (2026-07-20/21) | `diffusion_wan22_avid_xattn_replace_metaworld` (+ local `--sigma-sweep`/`--action-probe` diagnostics) | y1jrgxqp @1500 running→converged; uxrst2k5 crashed @342 | `y1jrgxqp` / `uxrst2k5`; local ckpt `outputs/replace-metaworld-run/checkpoints/step_00001500.pt` | see note | **Root cause of "replace generation = noise": eval paths dropped the per-frame `action_seq`** the xattn adapter trained on (silent OOD fallback, cos vs base 0.997→0.63). Fixed + validated end-to-end: `y1jrgxqp` adapted FID 518→58 ≈ base by step 600. **⚠ Generation metrics of ALL earlier action_seq-xattn runs are invalid** (5cxstyh4, ostoa19d, 81wq3lwt, bcipghvw, uea10230, likely xb76ptw2 incl. its "worse on all 6 metrics" row above) — training-seam losses stay valid. Post-fix measurements: adapter converges to a **total base-clone** (per-σ sweep: cos ≥0.996 at every σ, flat −0.002 deficit; sole exception σ=0.05 where adapter beats base +0.005) and is **fully action-blind** (shuffled/zeroed actions move loss <1e-4 at every σ). Gatelow single-clip overfit (`uxrst2k5`): gate saturated 0.5→0.99 from balanced init, grad norm 4.4→0.003 — failed to overfit ONE clip ⇒ copy-through is an optimization trap, balanced init insufficient. Countermeasures landed: `sigma_shift: 5.0` training option (enabled in replace config), no-base-input overfit config queued ([[../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]]). See [[../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]] |

Per [[../CLAUDE]] hard rule 8, every row in this table must cite a real
run. Cells marked `_nv_` (needs verification) are still missing their
wandb run id / commit / ckpt and must be filled from the run's records.

Concrete first-write candidates (rows the user can fill from their own
records — each becomes one experiment note under
`30_Knowledge/experiments/`):

- HyperAlign on MetaWorld — the in-flight one in
  `scripts/train_hyperalign_metaworld.py`.
- DynamiCrafter sanity smoke test that produced the now-deleted
  `tests/_outputs/dynamicrafter_sanity/*.png` artefacts.
- Any flow-shortcut config run end-to-end (`flow_output_shortcut.yaml`).
- The video-logging fix run that motivated commits `44b214b` and
  `88e4430`.

## Tests passing today

`pytest` (no CI runner wired). Files:

- `test_hyperalign_architecture.py`
- `test_dynamicrafter_checkpoint_sanity.py`
- `test_dynamicrafter_integration.py`
- `test_hyper_step_size_conditioning.py`
- `test_batch_preprocessor.py`
- `test_metaworld_dataset.py`
- `test_null_caption.py`
- `test_video_logging.py`
- `test_multimodal_substrate.py` *(added 2026-06-10; 7 tests, all passing)*
- `test_multimodal_real_backbone.py` *(added 2026-06-26; 3 tests, all passing —
  real-backbone compositional context-injection wiring contract)*

Tests cover the architecture / shape / wiring contract for each adapter
family, the data pipeline, and the video logging path. They do **not**
constitute experimental results — they ensure the code runs, not that the
adapters learn anything. (The multimodal overfit tests show the streams
*can* be fit on a toy dummy base — still not a real-data result.)

## What's planned

The four-deliverable plan in [[positioning]] is the high-level roadmap.
The concrete experiment backlog lives in `20_Tickets/` (not yet
populated). When experiments are queued, they should appear as
`20_Tickets/experiments/exp-{scope}-{slug}.md`.

## How this doc evolves

Trigger points for substantial rewrites:

1. **First real run finishes with logged outputs.** Replace the empty
   "What has actually run" table with one real row. Create a real
   `30_Knowledge/experiments/{slug}.md`. Update [[positioning]] D-status
   from "no evidence yet" to "first evidence in."
2. **D2 ablation across all four adapter families finishes.** Add a
   summary table and a Pareto figure reference. This is the headline D2
   chapter result.
3. **First shortcut few-step rollout curve.** Add a section "Shortcut
   results" with the curve and the comparison to non-shortcut baseline.
4. **D4 combined run.** Final summary section becomes the thesis result.

Per [[../CLAUDE]] hard rule 8: do not write any number into this doc
without citing the run that produced it.

## Related

- [[architecture]] — codebase and what it can do
- [[positioning]] — the four deliverables this evidence has to back
- [[../30_Knowledge/experiments/]] — per-run write-ups (folder to be
  populated)
- [[setup-status]] — vault coverage gaps
