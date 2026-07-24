---
date: 2026-07-11
topic: 20260907-weak-action-signal diagnosis → cross-attention build → experiment program
duration_minutes: 
files_touched:
  - 30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal.md
  - 10_now/product-state.md
  - 50_Decisions/open/action-conditioning-injection-mechanism.md
  - "impl repo: backbones/wan/modules/action_model.py, adapters/output/wan.py, adapters/factory.py, data/wan_batch_preprocessor.py, configs/diffusion_wan22_avid_xattn_i2v_metaworld.yaml"
tickets_created:
  - feat-eval-base-vs-adapted-delta
  - exp-conditioning-action-shuffle-ablation
  - feat-training-adapter-contribution-magnitude-logging
  - chore-data-action-frame-alignment-audit
  - exp-shortcut-action-free-isolation
  - exp-shortcut-zero-weight-control-run
  - feat-adapter-dynamicrafter-output-on-wan-base
  - feat-eval-interactive-action-debug-ui
  - feat-adapter-wan-action-cross-attention
---

# Session: WAN action-signal diagnosis → cross-attention build

Spanned 2026-07-09 (20260907 results review) → 2026-07-11 (build + teaching).

## The arc

1. **20260907 results review.** User: "loss not going down, videos okayish,
   adapter not doing much, action signal not picked up." Read the loss screenshots
   + sample videos → flat `base_loss`, blur/ghosting, arm not tracking GT.
2. **Reframe 1 — real WAN base.** The changed variable was the base model: prior
   runs loaded **random** WAN weights; 20260907 is the **first genuinely pretrained
   Wan2.2-TI2V-5B frozen base**. → flat `base_loss` is *expected* (strong base near
   its floor); the real metric is the **base-vs-adapted delta**, not the loss trend.
3. **Reframe 2 — button task + rows=NFE.** Grid layout is **rows = denoising steps
   (1→50), cols = GT | base | adapted**. On button-press the adapter (col3) is more
   task-directed than the base. Few-step (top rows) still poor = the D3 gap.
4. **Quantified base-vs-adapted delta (the headline finding).** From
   `button/{adapted,base}_eval.png`: adapter **clearly beats frozen base on PSNR /
   SSIM / MSE and the gap widens with training**, but **degrades LPIPS / FID / FVD**
   = **regression to the mean** (pixel-error minimisation buys blur). Overturns
   "adapter isn't doing much." For a planning world model PSNR/MSE is the headline →
   adapter helps.
5. **Diagnosis → experiment program.** Cheap probes (shuffle, magnitude, delta) +
   deliverable-defining runs (action-free shortcut isolation for D3; zero-weight
   control) + capacity lever (DC-adapter) + the interactive debug UI.
6. **Conditioning-mechanism decision.** Confirmed the WAN adapter injects action via
   **global AdaLN broadcast** (`action_model.py`). Opened a decision: AdaLN vs
   **cross-attention** (the adapter already has an unused `t2v_cross_attn` slot).
   Coupled sub-lever: per-frame action tokens vs the single summed vector.
7. **Built the cross-attention feature** in the impl repo (see below).
8. **Teaching:** `/teach` → new `teaching/` workspace + Lesson 01 "which metric for
   which purpose."

## Code built (impl repo — UNCOMMITTED, partially verified)

Cross-attention action injection for the WAN output adapter, flag
`action_injection: adaln | cross_attention | both`:
- `wan_batch_preprocessor.py` — emits per-frame `cond["action_seq"]` `[B,L,4]`
  (+ `action_seq_len` binning; sum-preserving, verified).
- `action_model.py` — `action_embedding` (mirrors `text_embedding`) + temporal
  pos-emb → existing `t2v_cross_attn` context; AdaLN gated by mode; timestep +
  step_level stay in AdaLN.
- `adapters/output/wan.py` — pulls `action_seq` (falls back to aggregated `action`;
  **user added a TODO to remove that fallback** for train/inference consistency).
- `adapters/factory.py` — threads flags + `action_token_dim = conditioning.input_dim`.
- New config `diffusion_wan22_avid_xattn_i2v_metaworld.yaml`.
- **Verified:** all 3 modes build + forward, `|Δ|=0` at init (base untouched);
  binning sum-preserving. **Not completed:** the config→factory integration smoke
  test (interrupted). User will test-run.

## Open questions / parking lot

- Read the base-vs-adapted PSNR/SSIM numbers straight from wandb → cite them.
- Shuffle test: is the col3>col2 gap action-*following* or a task prior?
- Is the LPIPS/FID/FVD blur partly the shortcut `distillation` target?
- **Headline-metric decision** (prediction accuracy vs realism) — not yet a vault
  decision; the teaching learning-record feeds it.
- Adapter checkpoint path for the 20260907/button runs (blocks the debug UI).
- Confirm which WAN ckpt + that weights are genuinely loaded (reframe rests on it).
