---
type: experiment
date: 2026-07-29
config: configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml; configs/dynamicrafter/diffusion_dc_shortcut_actionfree_robotarm.yaml
commit: uncommitted working tree @ 2026-07-29
wandb_run_id: pzmc2orq (Wan flow), t4bp8nki (DC diffusion)
ckpt_path: remote (cluster)
status: running (Wan @913, DC @684 — snapshot)
deliverable: D3
metrics:
  wan_multistep_consistency_loss: 0.0126
  dc_multistep_consistency_loss: 0.861
  wan_adapter_pred_base_cosine: 0.963
  dc_adapter_pred_base_cosine: 0.710
  wan_eval_base_loss: 0.184
  dc_eval_base_loss: 0.048
notes: "Early flow-vs-diffusion shortcut signal: DC (diffusion) consistency loss ~68x Wan (flow) — the curved-trajectory theory in the loss. Few-step quality + step-size sensitivity NOT yet resolved (runs predate the step-size probe; only N=8/50 grids)."
---

# Shortcut flow-vs-diffusion — early curvature signature (D3)

> **⚠ SECOND CONFOUND FOUND 2026-07-31 — the 68× gap is not attributable to
> curvature alone.** Wan's gate was **frozen**: `pzmc2orq` shows
> `eval_adapter_gate_mean` = 0.5 and `eval_adapter_gate_std` = **0** exactly,
> because its config pairs `gate_bias: 0.0` with `gate_cap: 0.5` — the gate is
> born on its clamp boundary and its gradient dies (see
> [[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]). DC's
> config has **no** `gate_cap`, so `t4bp8nki` ran with a live, spatially-varying
> gate (mean 0.596, std 0.0789) and ~2× the adapter contribution (0.260 vs
> 0.138).
>
> So the two arms differed in composition as well as in objective: Wan's output
> was pinned to a fixed 50/50 blend with a frozen, self-consistent base — which
> would *by itself* depress a consistency loss — while DC's adapter was free to
> diverge. Combined with the already-noted target mismatch (`v_average` vs
> `endpoint_inversion`), **the direction of the effect is no longer clean
> evidence for the curvature argument.** Re-run Wan with `gate_cap` above
> σ(`gate_bias`) before citing this anywhere.

> Snapshot of live runs (Wan `pzmc2orq` @913, DC `t4bp8nki` @684), wandb API
> 2026-07-29. Action-free shortcut on ACWM Robot Arm. Early — read as trend, not
> final.

## The signal

**Diffusion's self-consistency loss is ~68× flow's** — the curved-trajectory
theory made visible:

| metric | Wan (flow) | DC (diffusion) |
|---|---|---|
| `multistep_consistency_loss` | 0.0126 | **0.861** |
| `shortcut_direction_loss` | 0.0126 | 0.861 |
| `eval_denoise_adapter_delta` (base−adapted) | −0.032 | +0.042 |
| `adapter_pred_base_cosine` | 0.963 | 0.710 |
| `adapter_rel_contribution` | 0.138 | 0.260 |
| `eval_base_loss` (adapted) | 0.184 | 0.048 |

Flow (straight trajectory) satisfies the shortcut consistency easily; diffusion
(curved) does not — the thesis's "diffusion ill-suited for shortcut" argument
appearing empirically, even at these early steps.

**Caveat:** the two use different consistency targets (`v_average` for flow,
`endpoint_inversion` for diffusion), so the absolute 68× isn't strictly
apples-to-apples — but the direction (diffusion much harder to make
self-consistent) is the expected curvature signature.

## Not yet resolved (the two open questions)

1. **Few-step quality.** `eval_step_schedule` was only `{8, 50}`, so only N=8/N=50
   grids exist (`eval_step_grid/*` videos in both runs, gt|base|adapted). Wan's
   aggregate quality is *worse* than base at 913 steps (FID 119 vs 87, FVD 1699
   vs 1011) — early. Need the full N∈{1,2,4,8,25,50} sweep to see the few-step
   trade-off (the actual D3 payoff).
2. **Condition-blindness.** **No `eval_stepsize_effect_rel`** — these runs predate
   the step-size probe (added 2026-07-29). Wan's high `pred_base_cosine 0.96`
   could be cloning-collapse OR benign anchor-step behaviour; only the step-size
   probe disambiguates. Re-run/resume with the updated trainer (probe is
   default-on for shortcut) to get it.

## Next

- Expand `eval_step_schedule` → {1,2,4,8,25,50} so runs store the full few-step
  video comparison.
- Get `eval_stepsize_effect_rel` on these (resume with updated code, or an
  offline step-size probe on the checkpoints).
- The flow-vs-diffusion consistency-loss gap is the D3 result to watch — if it
  holds while Wan's few-step quality beats base at small N and DC's doesn't, the
  curvature story is made.
