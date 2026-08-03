---
type: experiment
date: 2026-07-31
config: configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml (probe target: ncztxyyo ckpt step_00001000)
commit: uncommitted working tree @ 2026-07-31 (probe: scripts/generate_wan22_i2v_compare.py --action-trace / --sigma-sweep --action-probe)
wandb_run_id: probe of ncztxyyo (no new training run; job 25085110)
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-gatelow-capshift-run/checkpoints/step_00001000.pt
status: completed
deliverable: D2
metrics:
  action_tokens_drel: 0.973
  xattn_out_drel_range_lo: 0.29
  xattn_out_drel_range_hi: 0.56
  block_out_drel: 0.0085
  composed_drel: 0.0096
  xattn_out_rms: 0.01
  stream_rms_lo: 1.8
  stream_rms_hi: 3.0
  sigma_sweep_effect_rel_lo: 0.0084
  sigma_sweep_effect_rel_hi: 0.0102
notes: "WHERE Wan loses the action, located: the signal survives cross-attention intact (xattn output 44-56% action-driven in ALL 10 blocks) then DROWNS at the residual add — xattn RMS ~0.01 vs stream RMS 1.8-3.0, drel 0.44 -> 0.0085 in one addition. qk_norm rescues attention logits but the VALUE pathway inherits the raw tiny token scale (RMS 0.006). Mirror image of DC's pedestal. sigma-sweep: effect_rel flat 0.0084-0.0102 across sigma -> train/eval mismatch hypothesis dead."
---

# Wan action trace — the signal survives attention, then drowns at the residual add (D2)

> Overnight investigation 2026-07-31 (autonomous session). Probe: paired
> true-vs-shuffled-action forwards, identical x_t/noise/σ, forward hooks at 23
> depths of the `ncztxyyo` adapter. `drel = ‖A_true − A_shuffle‖/‖A_true‖` per
> depth. 3 batches × bs 2, σ ∈ {0.5, 0.83}; table below σ = 0.83 (train median
> under `sigma_shift: 5`); σ = 0.5 is qualitatively identical.

## The propagation profile

| depth | drel (shuffle) | activation RMS |
|---|---|---|
| action tokens | **0.973** | 0.006 |
| block00.xattn out | **0.438** | 0.031 |
| block01–09.xattn out | 0.29–0.56 | 0.006–0.015 |
| **block00–09 out** | **0.0085** | **1.8–3.0** |
| head (pred) | 0.0093 | 1.39 |
| composed | 0.0096 | — |

**The action signal survives cross-attention intact** — the xattn output stays
~50% action-driven through all ten blocks. It then **drowns at the residual
junction** `x = x + cross_attn(norm3(x), context)` (`model2_2.py:254`): a
contribution of RMS ~0.01 joins a stream of RMS 1.8–3.0, and drel collapses
0.44 → 0.0085 in a single addition. The chain closes: block-out 0.0085 ≈
composed 0.0096 ≈ the measured `effect_rel` — the collapse fully accounts for
the blindness.

## Root cause

`norm_q`/`norm_k` exist (qk-norm; RMS ≈ 1.0 in the checkpoint), so the attention
*logits* are scale-rescued and attention routes correctly — which is why the
xattn output is action-faithful. But the **value pathway is unnormalised** and
inherits the raw action-token scale: joint deltas (RMS ~0.06) through the token
MLP give tokens of RMS ~0.006, so V (and hence the attention output) is a
whisper. The text-token path arrives at O(1); the action path arrives at O(0.01).

**Mirror image of the DC pedestal**: DC's conditioning was too *large* and
constant (99.7% pedestal swamping the signal); Wan's is faithful but ~250× too
*small* (stream swamping the signal). Both are scale mis-calibrations at the
injection interface — a candidate unifying D2 finding: plug-and-play conditioning
fails on *scale calibration*, in either direction, not on information content.

## Hypotheses killed tonight (each by direct measurement)

1. **Positional-readout** (pre-registered before the trace: attention learns to
   read `action_pos_emb` and ignore action content; predicted collapse *at* the
   xattn output). **Refuted** — xattn drel is 0.44–0.56, not ~0.
2. **Train/eval σ mismatch** (`sigma_shift: 5` trains at median σ 0.83, probe
   evals U(0,1)). **Refuted** — per-σ `effect_rel` is flat 0.0084–0.0102 from
   σ=0.1 to 0.95, slightly *higher* at low σ. No hidden band (mirrors the DC
   timestep stratification).
3. **Attention untrained** (yesterday's Xavier-RMS observation). **Refuted
   properly** — checkpoint-to-checkpoint movement (step 600→1000):
   cross_attn 0.88%, self_attn 1.70%, ffn 1.74%, comparable; only
   `gate_head` is frozen (0.006%, the [[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate|gate_cap bug]],
   confirmed independently). Fastest mover: `action_pos_emb` at 17.5%/400 steps.

## Also measured (σ-sweep, same job)

- The adapter is **strictly harmful at eval**: Δ(base−adapted) < 0 at every σ
  (−0.117 @ σ=0.1 → −0.022 @ σ=0.95). With the gate frozen at 0.5 the composed
  output is ½·base + ½·(degraded near-clone) everywhere.
- Prediction-space `effect_rel` ≈ 0.009 at every σ — the in-training aggregate
  (0.0056) was representative.

## Intervention launched — `action_token_norm` (Wan analogue of DC arm E)

`nn.LayerNorm(dim)` on the action tokens after embedding + pos-emb
(`action_model.py`, flag `adapter.extra.action_token_norm`), bringing them to
O(1) like the text path. Verified pre-launch: token RMS **0.004 → 0.757**; param
delta exactly the LayerNorm affine (896).

- **Arm:** `wan-robotarm-tokennorm` (job 25085598), config
  `diffusion_wan22_avid_xattn_tokennorm_acwm_robotarm.yaml`, wandb run name
  `acwm-robotarm-wan-TOKENNORM`.
- **Control:** the GATEFIX re-run (job 25083978, `tny84p7k`) — identical config
  minus the flag, both with the live gate (`gate_cap: 0.9`).
- Read `eval_action_effect_rel` on both; the DC precedent for this fix class is
  arm E's 0.0033 → 0.0257.

## Early causal readout (first evals, step ~500 — 2026-07-31 morning)

| run | `eval_action_effect_rel` | gate | rel_contrib | null |
|---|---|---|---|---|
| GATEFIX control (job 25083978) | 0.00117 → 0.00126 | 0.899 | 0.041 | 0 |
| **TOKENNORM** (job 25085598) | **0.01131 → 0.00900** | 0.900 | 0.043 | 0 |

**Sustained ~7–10× the matched control across the first two evals** — the scale fix unblocks the injection, but unlike DC's arm E (0.026→0.063→0.082, still climbing) it does not take off. Remaining suspects for the ceiling: the cloning shortcut (`condition_on_base_outputs: true`) and the gate re-saturating at its 0.9 cap. Single-variable next arms, not run tonight. — the trace's causal prediction
holding. Note both gates ran 0.5→0.899 immediately (the `uxrst2k5` saturation
dynamics reasserting once the cap moved to 0.9), so the control's 0.00117 is
mechanically lower than the frozen-gate original (adapter weight ~10% vs 50%);
the like-for-like number is the ratio at matched gate state. ⚠ one eval point
each — confirm at steps 1000/1500 before promoting.

## Caveats

- Trace on 3 batches × bs 2, 2 σ values, one checkpoint (step 1000, frozen-gate
  run — deliberate: diagnosing the model that measured 0.0056; the gate only
  affects composition, not the interior depths).
- drel at the action tokens is 0.973 *by construction* under whole-clip shuffle;
  the informative rows are the interior ones.
- The residual-junction reading is from RMS magnitudes + the drel collapse; the
  token-norm arm is the causal test.

## Related

- [[20260730-dc-parity-arms-null-action-embedding-pedestal]] — the DC mirror image
- [[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]] — found en route
- [[20260728-acwm-robotarm-matrix-action-blind]] — the number this explains
