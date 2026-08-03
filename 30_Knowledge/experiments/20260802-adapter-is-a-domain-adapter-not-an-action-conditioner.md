---
type: experiment
date: 2026-08-02
config: configs/wan22/diffusion_wan22_avid_xattn_tokennorm_acwm_robotarm.yaml · ..._tokennorm_nobase_... · configs/dynamicrafter/diffusion_dc_acwm_robotarm_armE_center.yaml
commit: uncommitted working tree @ 2026-08-02
wandb_run_id: 52o3uxz8 · vy9tcuco (Wan × ACWM) · 6oyu1inq (DC arm E) · 0fqjrqjl (Wan × RT-1 binned)
ckpt_path: /scratch-shared/lbierling/outputs/{acwm-robotarm-wan-tokennorm*,dc_acwm_robotarm_armE_center}/checkpoints/
status: completed
deliverable: D2
metrics:
  wan_acwm_fvd_improvement_pct: 63.7      # 1118.39 -> 406.34, oracle arm 52o3uxz8
  wan_acwm_fvd_improvement_nobase_pct: 45.1   # 1282.29 -> 703.74, vy9tcuco
  wan_acwm_effect_rel: 0.0062             # 52o3uxz8 final; 0.0077 nobase
  wan_acwm_steering_cos: 0.00             # all three structure axes at chance
  dc_arme_steering_cos: 0.117             # chance 0.000
  dc_arme_temporal_alignment: 1.000       # chance 0.313
  dc_arme_spatial_concentration: 0.470    # chance 0.100
notes: "The headline D2 result, stated per-cell. The SAME adapter design is a strong DOMAIN adapter on Wan (FVD -64% on ACWM) and a working ACTION conditioner on DC (structure above chance) -- but never both on either backbone. On Wan the actions enter faithfully and drown in the residual stream."
---

# The adapter is a domain adapter on Wan and an action conditioner on DC — never both

## The claim

> The same adapter design is a **strong domain adapter on Wan** and a **working
> action-conditioner on DC** — but not both on either. On Wan the actions enter
> faithfully and drown in the residual stream; the base-correction path is far
> cheaper under a loss where actions are worth 0.45%.

## Evidence

### Wan × ACWM — excellent domain adaptation, no action control

Both token-norm arms beat the **frozen base on all six** quality metrics
(% = improvement over each arm's *own* base, since the two arms' bases differ):

| | oracle ON (`52o3uxz8`) | oracle OFF (`vy9tcuco`) |
|---|---|---|
| FID | **+36.3%** (90.15 → 57.43) | +27.5% (82.42 → 59.74) |
| **FVD-I3D** | **+63.7%** (1118.4 → 406.3) | +45.1% (1282.3 → 703.7) |
| LPIPS | +19.7% | +16.0% |
| PSNR / MSE / SSIM | +11.4 / +35.2 / +1.5% | +9.7 / +31.0 / +1.7% |
| `effect_rel` | 0.0062 | 0.0077 |
| `pred_base_cosine` | 0.914 | 0.851 |

And on the *same* NOBASE checkpoint the structure triad came back **at chance on
all three axes** — steering cos 0.00, temporal alignment at chance, spatial
concentration at chance ([[20260731-wan-action-signal-is-a-global-bag]]).

**A 2.75× FVD improvement carrying no action information is a domain
correction.**

### DC arm E — genuine action control

Structure triad on held-out `ind_test`
([[20260731-dc-condition-center-accelerates-escape]], probe job 25144197):

| probe | arm E | arm 0 | chance |
|---|---|---|---|
| steering cos | **+0.117** | +0.106 | 0.000 |
| temporal alignment | **1.000** | 1.000 | 0.313 |
| spatial concentration | **0.470** | 0.489 | 0.100 |
| frozen-base null | 0.000 | 0.000 | — |

All three above chance. **DC is not action-blind.** (Note arm 0 matches it —
`condition_center` accelerates escape rather than creating control.)

## Two scope conditions the claim needs

1. **The Wan quality win is ACWM-specific.** On RT-1 the same adapter is a net
   *perceptual* regression — held-out FID 143.1 vs base 120.8, LPIPS 0.427 vs
   0.376 — improving only pixel metrics (`0fqjrqjl`,
   [[20260801-wan-rt1-indistribution-plateau]]). Domain adaptation works where
   the base is far from a **narrow** target distribution and fails on diverse
   real data.
2. **It is a Wan failure, not an adapter failure.** DC processes actions with
   the same adapter family. "The adapter cannot process actions" is false in
   general; the true statement is "**on Wan**".

## Mechanism (already established)

- Action tokens are informative on entry and survive all 10 blocks, then
  **drown at the residual add** — cross-attn out RMS ~0.01 vs a stream of
  1.8–3.0 ([[20260731-wan-action-trace-value-pathway-drowns]]).
- Actions are worth **0.45%** of the teacher-forced denoising loss, so
  base-correction is overwhelmingly the cheaper gradient
  ([[20260731-why-wan-copies-the-base-decomposed]]).
- The oracle makes that cheaper still: with `condition_on_base_outputs: true`
  the prediction is ~100× more sensitive to the base's prediction than to the
  actions.

## The composition-interface ablation (D1)

`condition_on_base_outputs` is a **single-flag change on the composition
interface** and it trades prediction quality against action-conditioning:
oracle ON wins every quality metric (FVD +63.7% vs +45.1%) and sits closer to
the base (cosine 0.914 vs 0.851) while following actions **25% less** (0.0062
vs 0.0077). That is a framework (D1) result: whether `Δ_φ` sees
`f_base(x_t, t)` is a real design axis with a measurable cost, not an
implementation detail.

⚠ n=1 per arm, at slightly different steps (3315 vs 3054), and the two arms'
bases differ — quote **% improvement over each arm's own base**.

## Caveats

- `action_token_norm` raised `effect_rel` ~6× at peak (0.0113/0.0123 vs the
  GATEFIX control's 0.0020) and ~3–4× sustained — but the structure stayed at
  chance, and token-norm is mechanically a **gain knob** while `effect_rel` is
  monotone in gain. It raised sensitivity, not control.
- No DC run logs quality metrics (checked all 18), so the DC cell's domain-
  adaptation performance is **unknown**.
- The structure triad has **never been run on the binned RT-1 checkpoint**
  (`0fqjrqjl`), which is the intervention most likely to change the Wan verdict.

## Related

- [[20260731-wan-action-signal-is-a-global-bag]] · [[20260731-wan-action-trace-value-pathway-drowns]]
- [[20260731-why-wan-copies-the-base-decomposed]] · [[20260731-dc-condition-center-accelerates-escape]]
- [[../writing/thesis-storyline]] §8–9
