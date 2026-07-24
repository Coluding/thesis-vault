---
type: experiment
date: 2026-07-21
config: configs/diffusion_wan22_avid_xattn_replace_metaworld.yaml
commit: uncommitted working tree (trainer/wan.py fixes of 2026-07-20; commit pending)
wandb_run_id: y1jrgxqp (Wan2.2-avid-xattn-replace-i2v-metaworld), uxrst2k5 (Wan2.2-avid-xattn-i2v-metaworld); local diagnostics have no wandb (artifacts on disk)
ckpt_path: outputs/replace-metaworld-run/checkpoints/step_00001500.pt (local); remote y1jrgxqp checkpoints on training box
status: completed
deliverable: D2
metrics:
  loss: "see per-sigma table"
  val_mse: "y1jrgxqp step 1500: adapted 0.0182 / base 0.0191"
notes: "Closes the replace-generation-noise investigation; establishes total action-blindness of the xattn adapter."
---

# Replace-noise root cause validated end-to-end + σ-sweep + action-sensitivity probe

Three connected measurements, 2026-07-20/21. Together they close
[[../../20_Tickets/done/bug-adapter-replace-generation-flat-since-init]] and
[[../../20_Tickets/done/exp-conditioning-action-shuffle-ablation]], and set up
the successor experiment
[[../../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]].

## 1. Root cause of "replace generation is noise" — found, fixed, validated

**Root cause (measured 2026-07-20, local debug session):** every
generation-eval site (`_native_eval_grid`, `_native_quality_eval`, and the
debug script's rollout) built `adapted_cond = {"action": ...}` **without the
per-frame `action_seq`** the cross-attention adapter was trained on. The
preprocessor emits `action_seq` unconditionally
(`wan_batch_preprocessor.py:383`) regardless of `action_per_frame` — that
flag only governs the *encoder* input — so training always cross-attended
over per-frame tokens while generation fed one sum-aggregated OOD token
through `Wan21OutputAdapter`'s silent fallback. Cond-variant test on the
local step-1500 replace checkpoint, identical input at σ=0.999:

| cond passed to adapter | cos(adapter out, base) | ‖out‖/‖base‖ |
|---|---|---|
| full (with `action_seq`) | 0.997 | 1.007 |
| without `action_seq` (old eval path) | 0.634 | 0.504 |

Under `composition: replace` the collapsed output IS the velocity → rollouts
never leave noise, at every step count, flat since init. An earlier code-read
had *retracted* this hypothesis on the grounds that `action_per_frame: false`
disables `action_seq` — that retraction was itself wrong (the flag doesn't
gate the xattn token path) and is corrected in the closed bug ticket.

**Fixes (2026-07-20, implementation repo):** `trainer.py`
`_native_batch_conditions` returns and threads `action_seq` into both eval
paths; `scripts/generate_wan22_i2v_compare.py` passes it in rollouts;
`adapters/output/wan.py` now **raises** instead of silently falling back when
a cross-attention adapter gets an aggregated action without `action_seq`.

**End-to-end validation (wandb `y1jrgxqp`, no-shortcut replace, first run on
the fixed code):**

| eval step | adapted FID | base FID | adapted FVD | base FVD | denoise Δ (base−adapted) | rel contribution |
|---|---|---|---|---|---|---|
| 0 | 518.8 | 64.7 | 4159 | 1384 | −1.78 | 1.00 |
| 300 | 82.9 | 59.6 | 1563 | 1432 | −0.062 | 0.216 |
| 600 | 58.1 | 55.5 | 1286 | 1259 | −0.016 | 0.117 |
| 900 | 60.0 | 60.0 | 1077 | 1090 | −0.0013 | 0.100 |
| 1200 | 61.0 | 60.6 | 1273 | 1244 | +0.0063 | 0.100 |
| 1500 | 59.2 | 61.0 | 978 | 1020 | −0.0004 | 0.060 |

Step 0 is the expected replace-at-zero-init noise; from step 300 the adapted
rollouts are coherent. The old symptom (FID pinned at ~520 forever) is gone.

**Consequence:** all *generation-based* metrics (FID/FVD/PSNR/SSIM/LPIPS
grids and quality evals) from every earlier run whose adapter cross-attended
over `action_seq` are invalid — `5cxstyh4`, `ostoa19d`, `81wq3lwt` (replace),
`bcipghvw` (gatelow), `uea10230` (overfit), and most likely `xb76ptw2` (the
2026-07-12 xattn negative result — same eval code path; its "worse on all 6
metrics" conclusion should not be cited until re-run). Training-seam losses
from those runs remain valid.

## 2. Per-σ loss breakdown — the copy is total at every noise level

Question: does the uniform-σ average hide an adapter advantage at high σ
(where actions/dynamics carry information)? Tool: new `--sigma-sweep` mode in
`scripts/generate_wan22_i2v_compare.py` — rebuilds `x_t`/`t` at fixed σ from
preprocessed eval batches (diffusion-forcing convention, obs frames clean),
frame-masked MSE, 6 clips × 2 paired noise draws per σ. Checkpoint: local
replace `step_00001500.pt`. Artifacts:
`outputs/replace_debug/sigma_sweep.{csv,png}`.

| σ | adapted | base | Δ (base−adapted) | rel dev | cos to base |
|---|---|---|---|---|---|
| 0.05 | 0.2848 | 0.2900 | **+0.0053** | 0.106 | 0.996 |
| 0.10 | 0.1806 | 0.1795 | −0.0010 | 0.074 | 0.998 |
| 0.20 | 0.1094 | 0.1073 | −0.0021 | 0.067 | 0.998 |
| 0.30 | 0.0805 | 0.0783 | −0.0022 | 0.061 | 0.998 |
| 0.50 | 0.0554 | 0.0534 | −0.0020 | 0.055 | 0.999 |
| 0.70 | 0.0457 | 0.0436 | −0.0020 | 0.053 | 0.999 |
| 0.90 | 0.0473 | 0.0451 | −0.0022 | 0.056 | 0.999 |
| 0.99 | 0.0995 | 0.0973 | −0.0022 | 0.061 | 0.998 |

**No hidden advantage anywhere**: flat ≈−0.002 deficit at every σ ≥ 0.1,
cos-to-base ≥ 0.996 everywhere. The only sign flip is σ=0.05, where the
adapter *beats* the base (+0.0053) — plausibly domain calibration at very low
timesteps Wan's shifted pretraining rarely visits (analysed estimate, not
proven). This likely also explains the small persistent generation-metric
edge of `y1jrgxqp` (adapted PSNR > base at every eval ≥ 300) — low-σ solver
steps benefit — rather than any action effect (see §3).

## 3. Action-sensitivity probe — the adapter is completely action-blind

Tool: `--action-probe` (same script): adapted loss under (a) the clip's own
actions, (b) a *different clip's* actions, (c) zeroed actions — same noise
per comparison (paired). Same checkpoint.

| σ | adapted (true) | shuffled gap | zeroed gap |
|---|---|---|---|
| 0.10 | 0.18078 | −0.00000 | +0.00009 |
| 0.30 | 0.08045 | −0.00000 | +0.00009 |
| 0.50 | 0.05551 | +0.00000 | +0.00007 |
| 0.70 | 0.04537 | +0.00000 | +0.00006 |
| 0.90 | 0.04723 | +0.00001 | +0.00007 |
| 0.99 | 0.11596 | +0.00008 | +0.00004 |

Another clip's actions change the loss by <1e-5 at every σ; zeroing costs
<1e-4. The nonzero zero-gaps prove the conditioning pathway is live (the
override reaches the model) — the model just doesn't use it. Meanwhile the
adapter's ~5% deviation from base persists unchanged across variants: the
deviation it does make is **pure action-independent domain adjustment**. This
hits the shuffle-ablation ticket's decision-rule branch "adapter is not using
actions" exactly.

Caveat: the 6 donor clips are MetaWorld scripted demos, so shuffled actions
may resemble true ones — but the zero variant rules that confound out.

## 4. Single-clip overfit (wandb `uxrst2k5`) — gate saturation from a balanced init

Gatelow-variant overfit (logged `adapter_gate_mean` = 0.5 at step 1 = σ(0),
i.e. `gate_bias: 0.0`; the wandb `experiment` name shows the stale
`..._xattn_i2v_metaworld` label — remote yaml edit, user-confirmed gatelow
settings). Crashed at step 342 (~6 h; no synced log). Findings:

- Gate 0.5 → 0.99 in ~70 steps, pinned; adapter grad norm 4.4 → 0.003;
  `denoise_adapter_delta` −0.90 → ≈+0.0008; rel contribution 0.71 → 0.022.
- Adapted FID plateaued ~257 vs base 225 with no further movement.
- **The run failed to overfit a single clip** — it slid into base-copy and
  gradients died. Since one memorized clip is an easy target, this locates
  the failure in optimization (a copy-through attractor that closes behind
  itself), not data diversity. Balanced gate init does **not** prevent
  saturation — evidence appended to
  [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]].

## Interpretation (discussed 2026-07-21; analysis, not raw measurement)

Base-parity is the convergence point of every composition tried because each
gives the optimizer a cheap route to reproduce the base (gate→base for
mask_mix; identity on the `base_output` input for replace), while everything
beyond parity is expensive: the σ~U(0,1) objective barely pays for
action/dynamics information (it concentrates near σ→1), and the residual may
be small in this data at all (MetaWorld scripted demos: the anchor frame
largely determines the future — see the ACWM-Phys comparison discussion,
related-work note pending). Countermeasures landed 2026-07-21:
`sigma_shift: 5.0` training option (SD3/Wan shift, train-only, enabled in the
replace config), and the no-base-input overfit config
([[../../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]]) to discriminate
trap vs capacity.

## Reproduce

```bash
# σ sweep + action probe (local 3090, 41-frame windows to hit the latent cache)
python scripts/generate_wan22_i2v_compare.py \
  --config configs/diffusion_wan22_avid_xattn_replace_metaworld.yaml \
  --checkpoint outputs/replace-metaworld-run/checkpoints/step_00001500.pt \
  --sigma-sweep --action-probe --temporal-length 41 --loss-batches 0
```
