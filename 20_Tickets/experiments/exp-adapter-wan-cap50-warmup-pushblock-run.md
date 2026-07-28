---
type: exp
scope: adapter
status: open
priority: high
created: 2026-07-25
updated: 2026-07-25
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-backbone-wan-robotarm-run]]", "[[exp-adapter-gatelow-cap-sigmashift-metaworld-run]]"]
---

# exp: Wan2.2 · Push Cube · gate_cap 0.5 + AVID warmup (matrix run 6 — intervention)

## Hypothesis

Can a **harder gate cap + adapter warmup** rescue the adapter from base-parity
collapse on the action-informative-but-flat 2D domain, *without* moving to Robot
Arm? Push Cube is the stress case: frozen-base masked denoise loss ~0.036 at 17f
(≈8.7× below Robot Arm's 0.314, measured 2026-07-25) → almost no residual, so the
mask_mix gate happily saturates toward a clone. Two levers vs that:

- `gate_cap` 0.9 → **0.5** (base keeps ≥50%; adapter forced to own the rest)
- `pretrain_steps` 0 → **500** (AVID pure-adapter warmup before composition)

## Procedure

- Config: `configs/wan22/diffusion_wan22_avid_xattn_cap50_warmup_acwm_pushblock.yaml`
  — minimal deltas on the capshift parent (`sigma_shift 5.0` kept). Reuses the
  **already-precomputed pushblock latents** (no new precompute).
- Diff against the parent
  (`diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml`) isolates
  cap+warmup.

## Decision rule

- **pred-base cosine stays low + adapted loss beats base** ⇒ interventions can
  hold the adapter open even on a near-zero-residual domain → strong toolbox
  result for D2.
- **Still clones (cosine ~0.85, gate pinned at cap)** ⇒ interventions are not
  sufficient when the residual is this small → the dataset (residual) is the
  binding constraint, reinforcing the Robot-Arm move.

## Build status

Config created & validated (loads: gate_cap 0.5, pretrain_steps 500).
**Launch-ready now** — pushblock latents already cached.

## Notes

Isolates Axis-3 interventions (gate_cap / warmup) from the dataset axis. If you
want to also drop `sigma_shift` to isolate purely cap+warmup, say so — currently
kept from the parent recipe. See [[../../30_Knowledge/writing/ablation-axes]] Axis 3.
