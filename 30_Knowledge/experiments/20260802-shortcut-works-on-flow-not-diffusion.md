---
type: experiment
date: 2026-08-02
config: configs/wan22/diffusion_wan22_{shortcut,noshortcut_control}_actionfree_robotarm.yaml · configs/dynamicrafter/diffusion_dc_shortcut_d3arm_actionfree_robotarm.yaml
commit: uncommitted working tree @ 2026-08-02
wandb_run_id: wan-shortcut-actionfree-robotarm / wan-noshortcut-control-actionfree-robotarm / dc-shortcut-D3-arm-run
ckpt_path: /scratch-shared/lbierling/outputs/{wan-shortcut-actionfree-robotarm,wan-noshortcut-control-actionfree-robotarm}/checkpoints/step_00000800.pt · dc-shortcut-D3-arm-run/checkpoints/step_00000400.pt
status: completed
deliverable: D3
metrics:
  wan_shortcut_consistency_cos: 0.30246
  wan_shortcut_ci95: "[0.25051, 0.35586]"
  wan_control_consistency_cos: 0.03369
  wan_control_ci95: "[0.02602, 0.04150]"
  wan_separation_ratio: 9.0
  wan_shortcut_gain_at_half: 0.3337
  wan_control_gain_at_half: 0.0255
  dc_shortcut_consistency_cos: 0.08406
  dc_control_consistency_cos: 0.08292
  dc_shortcut_gain_at_half: 40950.0
  dc_control_gain_at_half: 17300.0
  clip_null_dc_shortcut: 0.00379
  clip_null_dc_control: 0.05488
  clip_null_wan_shortcut: 0.01698
  clip_null_wan_control: 0.01664
  base_null: 0.0
notes: "⚠ CONFOUNDED for the cross-base claim — the arms differ in consistency target (DC endpoint_inversion vs Wan v_average) and depth. WITHIN-ARM results stand. On a FLOW base the shortcut objective produces genuine step-size consistency (cos 0.302 vs control 0.034, 9x, non-overlapping CIs, depth-matched at step 800) and a flat O(1) gain profile. On a DIFFUSION base it produces none (0.084 vs control 0.083 at matched step 400, CIs coincident) and the gain explodes to 4e4 at large d. Two independent signals, same conclusion: shortcut works on flow, not on the published velocity-averaging target over a curved VP arc."
---

# Shortcut is learnable in the Wan/flow cell, not in the DC/diffusion cell (D3)

> # 🛑 CORRECTION 2026-08-02 (later same day) — THE CROSS-BASE CLAIM IS CONFOUNDED
>
> The two families do **not** share a consistency target:
>
> ```
> DC  arm: shortcut_consistency_target: endpoint_inversion
> Wan arm: shortcut_consistency_target: v_average
> ```
>
> So "flow vs diffusion" is confounded with "endpoint-inversion vs
> velocity-averaging" — **and with depth** (DC step 400, Wan step 800). The
> curvature attribution below is **NOT established by this experiment**.
>
> **The confound runs against the curvature story.** DC ran the *theoretically
> exact* target (`endpoint_inversion` — the "fix the coordinates" escape, which
> lands 0.000000 error in the synthetic `ddim_micro_step_v` test) and still
> learned nothing. If curvature of the averaging target were the explanation,
> that arm should have worked.
>
> **What still stands** (each has a matched within-arm control, same target, same
> base, same depth):
> - Wan / flow / `v_average` **learns** the consistency relation: 0.302 vs
>   control 0.034, 9x, disjoint CIs, flat O(1) gain.
> - DC / diffusion / `endpoint_inversion` **does not**: 0.084 vs control 0.083,
>   CIs coincident, gain exploding to 4e4.
>
> **What does NOT stand:** "shortcut works on flow *because* kappa=0", and the
> title of this note. The right title is *"shortcut is learnable in the Wan/flow/
> v_average cell and not in the DC/diffusion/endpoint-inversion cell"*.
>
> **What would decide it:** vary ONE thing. Within a single base, at one depth,
> run `v_average` vs `endpoint_inversion`. The codebase supports both via
> `training.shortcut_consistency_target`, so this is a config-only 2x2 — no code.
> Until then §"Why this is the curvature node" below is a hypothesis, not a result.


> The first clean **positive** D3 result in the vault, and the first empirical
> backing for [[../theory/shortcut-v-averaging-bias]] — which until now rested on
> the derivation plus a 68x consistency-loss ratio that
> [[20260729-shortcut-wan-vs-dc-curvature-signature]] flags as gate-bug
> confounded.

## The measurement

`scripts/eval_stepsize_blindness.py` (probe A, *consistency direction*):

```
consistency_cos = cos( pred(2d) - pred(d),  target_2d - pred(d) )
```

where `target_2d` is the two-half-step composition the shortcut objective itself
supervises, built by the codebase's own
`compute_self_consistency_target_v{,_flow}`. **0 = the model responds to `d`
arbitrarily; +1 = it responds exactly as the objective asks.** Gain-normalised by
construction (it is a cosine), so it cannot be inflated by a louder step-size
pathway — the failure mode that made `effect_rel` unreadable on the D2 side.

Held-out `ind_test`, action-free arms (D3 isolated from D2 per
[[../writing/thesis-storyline]] §6), full dyadic ladder 1/128 … 1.

## Result — the 2x2

| base | arm | `consistency_cos` | CI95 | gain at `d=1/2` |
|---|---|---|---|---|
| **Wan (flow, `v_average`)** | shortcut | **+0.30246** | [0.251, 0.356] | **0.334** |
| | control | **+0.03369** | [0.026, 0.042] | 0.026 |
| **DC (diffusion, `endpoint_inversion`)** | shortcut | +0.08406 | [0.067, 0.102] | **40950** |
| | control | **+0.08292** | [0.067, 0.100] | 17300 |

**Both pairs are now depth-matched** — Wan at step 800, DC at step 400.
(An earlier unmatched DC control read `best.pt` and gave 0.090; the matched
step-400 figure is 0.08292 and is the one to quote.)

**Wan: 9x separation, non-overlapping CIs. DC: 1.01 — none.** DC's intervals are
essentially coincident ([0.067, 0.102] treated vs [0.067, 0.100] control).

## Two independent signals, same conclusion

**1. Direction.** The shortcut objective buys real consistency on flow
(0.034 → 0.302) and nothing on diffusion (0.090 → 0.084).

**2. Magnitude.** The gain companion `|dpred|/|dtarget|` per rung:

```
                d=1/128  1/64   1/32   1/16    1/8    1/4     1/2
Wan treated:     0.440   0.380  0.350  0.344  0.420  0.468   0.334
Wan control:     0.483   0.373  0.285  0.193  0.118  0.063   0.026
DC  treated:     0.973   0.549  0.283  360.5  1926   3288   40950
```

- **Wan treated is flat and O(1)** at every step size — the response is about the
  right size for the jump being asked for.
- **Wan control collapses** as `d` grows (0.48 → 0.026): without the objective the
  adapter barely moves at large steps.
- **DC explodes by 4–5 orders of magnitude** at exactly the large `d` few-step
  rollout uses.

The second signal matters because it is not a cosine — it is the scale the
cosine deliberately normalises away. Both agree.

## Why this LOOKS like the curvature node (hypothesis — see the correction above)

[[../theory/shortcut-v-averaging-bias]]: Frans et al. eq. (4) averages two
half-step velocities. That is **exact for a straight interpolant** (flow, κ=0)
and **biased on a curved VP arc** by the sagitta ≈ κ·δ²/2, a bias that is not a
fixed point of the averaging rule and so compounds up the doubling tower.

Prediction: shortcut should be learnable on flow and not on the published
velocity-averaging target over diffusion. **That is what the table shows**, in
two independent statistics, each with its own matched control.

It also isolates half of the confound flagged in [[../writing/thesis-storyline]]
§5 ("the flow pivot changes two variables at once — flow *and* a much stronger
base"). This measurement is about the *objective's learnability on the
geometry*, and the matched within-base controls remove the base-strength axis
entirely.

## Nulls and controls

| | Wan shortcut | Wan control | DC shortcut |
|---|---|---|---|
| random-direction null | −0.00007 | +0.00010 | +0.00022 |
| mismatched-clip null | 0.01698 | 0.01664 | 0.00379 (DC control: 0.05488) |
| frozen-base null | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| spearman vs ladder | +1.0000 | +1.0000 | +1.0000 |

Clip-nulls are near-identical across the Wan pair (0.0170 vs 0.0166), so the 9x
separation is **not** a clip-genericity artifact. On DC they differ (treated
0.0038, control 0.0549); clip-corrected that leaves ~0.080 vs ~0.028 (~2.9x),
against Wan's ~0.285 vs ~0.017 (~17x) under the same correction. Either way the
ordering is the same and the magnitude gap is large. Frozen-base null is exactly 0
by construction (the base cannot see `d`) and was verified, not assumed.

## Caveats

- **Shallow.** Wan at step 800, DC at 400. Both arms were <1 day old.
- **Wan's wrong-rung null is 0.291** against a pooled 0.329 (at step 1200), so
  *"points the right way"* is solid but *"points the right way for this specific
  `d`"* is not separable — neighbouring rungs' targets are 0.65–0.86 correlated
  and the probe explicitly warns this null is unreadable in that regime. The
  **control** comparison is unaffected: the control shares the same rung
  structure and still scores 0.034.
- This measures the **objective's learnability**, not few-step sample quality.
  No rollout/FID evidence — [[../../20_Tickets/experiments/exp-eval-shortcut-fewstep-videos]]
  remains open.

## Few-step quality — status and scope (2026-08-02)

**Qualitative observation (Lukas, from run inspection — _not_ a sourced metric):**
few-step rollouts do **not** yet reach good quality, but are "going in the right
direction". Compute budget for the thesis is exhausted, so systematic
quality/tuning work is **scoped to future work** rather than attempted now.

⚠️ This paragraph is an *observation*, not a measurement. Nothing in this vault
contains FID/LPIPS/PSNR for few-step rollouts on these arms. To make it citable
the missing run is: {Wan, DC} x {shortcut, control} x N in {1,2,4,8,50},
decoded, scored against the 50-step reference — inference-only on the existing
checkpoints, no training. `scripts/generate_shortcut_fewstep.py` exists and has
never been run on these cells.

**Why the mechanism result is unaffected.** This note's claim is that the
shortcut *objective* is learnable on flow and not on the published
velocity-averaging target over diffusion. That is established by
`consistency_cos` and the gain profile, both with matched controls. Sample
quality is a *separate and downstream* question — a model can satisfy the
self-consistency relation and still generate poorly if it is under-trained or
mis-tuned. Stating the mechanism result without a quality result is therefore
honest, provided the thesis does not imply few-step generation works.

**Future work, scoped concretely:** longer training on the flow cell (these arms
were <1 day old, steps 800-1200); tuning the shortcut/consistency weights and
the step-size ladder; and the N-grid quality evaluation above.

## Sources

Probe jobs `25155284` (Wan shortcut @1200, +0.32943), `25156497` (Wan pair @800),
`25155716` (DC shortcut @400), `25156498` (DC control @400).
Suite: `scripts/eval_stepsize_blindness.py` +
`src/generative_flow_adapters/evaluation/stepsize_structure.py`.
Fix required to run it at all on the vendored Wan provider:
`eval_action_sensitivity.py` assumed the external provider's `.wan.vae` and
crashed on `Wan22DiTWrapper`; now resolves the VAE per provider.

## Related

- [[../theory/shortcut-v-averaging-bias]] — the derivation this confirms
- [[20260729-shortcut-wan-vs-dc-curvature-signature]] — the confounded 68x precursor this supersedes
- [[../writing/thesis-storyline]] — §4 (the proven node), §5 (the two-variable confound)
