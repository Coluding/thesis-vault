---
date: 2026-08-05
category: finding
deliverable: D2
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260805-turbo-action-tokens-binned-to-latent-grid]]"
  - "[[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]]"
  - "[[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]"
---

# On a distilled base the action reaches the adapter — and still is not used correctly

## What

An action adapter on a frozen **4-step distilled Wan-Turbo** base — which is
**level L3** of the efficiency axis, where acceleration lives in the base
and the adapter learns only conditioning. Binning the action tokens to the
latent grid raised action *effect* **4–6.5×**, the highest of any cell to
date. Action *accuracy* did not move at all.

## Why it matters

This is a **clean dissociation**, and it is the sharpest statement of the
thesis's core problem we have:

> The action now changes the prediction, but changing it to the **correct**
> value does not reduce error versus shuffling or zeroing it.

Two consequences. The token misalignment was real and worth fixing, but it
was **not the binding constraint**. And more importantly for the efficiency
comparison: an *effect*-only readout would have ranked L3 **best of all
three levels** on a cell that learned no correct dynamics. The per-level
protocol has been changed to require an **accuracy** readout alongside the
effect readout — the single most misleading outcome the axis could produce,
caught before the comparison ran.

## Evidence

Run `jlnl7s1k` (slurm `25240927`), 100M adapter (1.99 % trainable), ACWM
Robot Arm 49f, frozen `Wan2.2-TI2V-5B-Turbo` at a 4-step distilled grid.

At matched step 1600 versus the unbinned predecessor (`zcjvjj5a`):

| metric | unbinned, 34M | binned, 100M | ratio |
|---|---|---|---|
| `eval_action_effect_rel` | 0.00313 | **0.01268** | 4.1× |
| `eval_action_effect_vs_adapter` | 0.0277 | **0.179** | 6.5× |

Growing monotonically to 0.27–0.31 by steps 2800–4000.

Against that: `eval_action_loss_gap` is ~0 at **all ten evals**
(|x| ≤ 0.00055, no trend) and `eval_action_cos` never leaves 0.9998.

Also sourced: the arm **overfits from step 1200** (`eval_loss` 0.1271 best →
0.1918 @4000), and `eval_denoise_adapter_delta` never crosses zero — the
adapter still *hurts* denoising at every eval. The gate is healthy
throughout, so the earlier gate-freeze signature is gone.

⚠ **Two variables changed at once** (binning *and* 34M→100M), so the 4–6.5×
is not attributable to binning alone — a 34M+binned control is needed.
⚠ Reproducibility: 135 uncommitted modified files at launch; the run used
rsynced working-tree code, not the recorded commit.

🛑 A promising motion-tracking signal (adapted rank order matches ground
truth in both independent draws) is **hand-measured from 6 mp4s and must not
be cited**. The open question is causal, not statistical — is the adapter
tracking the *action* or the conditioning frame? The paired shuffled-action
control is now instrumented and fires on the next run.

## Next

- 34M + binned control to split binning from capacity.
- Cap `--time` at ~3 h on this arm — the last 2800 steps were past the
  optimum.
- The instrumented motion-tracking control decides whether the qualitative
  signal is real.
