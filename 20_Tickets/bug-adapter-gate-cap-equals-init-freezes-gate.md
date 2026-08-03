---
type: bug
scope: adapter
status: in-progress
priority: high
created: 2026-07-31
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[bug-adapter-gate-saturation-mask-mix]]", "[[../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]", "[[../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"]
---

# bug: `gate_cap` equal to σ(`gate_bias`) freezes the gate permanently

## What

With `gate_bias: 0.0` and `gate_cap: 0.5`, the mask_mix gate is born **exactly on
its own clamp boundary** and dies there within a few steps.

`models/adapted_model.py:255-257`:

```python
gate = torch.sigmoid(adapter_result.gate + self.gate_bias)   # σ(0 + 0.0) = 0.5
if self.gate_cap is not None:
    gate = gate.clamp(max=self.gate_cap)                      # clamp(max=0.5)
```

The gate head is zero-initialised (`backbones/wan/modules/action_model.py:178-179`
zeroes weight *and* bias), so the pre-sigmoid logit is exactly 0 → σ = 0.5, and
the cap is 0.5.

## Mechanism (verified)

| gate logit | cap | `d(gate)/d(logit)` |
|---|---|---|
| 0.0 (init) | 0.5 | 0.25 — flows |
| **+0.001** | 0.5 | **0.000 — dead** |
| 0.0 | 0.9 (pre-2026-07-27) | 0.25 — flows |

The documented dynamics push the gate **up** — `uxrst2k5` went 0.5 → 0.99 in ~70
steps, which is *why* `gate_cap` was introduced
([[bug-adapter-gate-saturation-mask-mix]]). The moment the logit goes positive,
`clamp(max=)` zeroes its gradient, and with zero gradient it cannot come back
down. **One-way trap.**

So `gate_cap`, added to *counter* gate saturation, becomes a freeze when set
equal to σ(`gate_bias`).

## Evidence — wandb `ncztxyyo` (Wan × ACWM Robot Arm)

- `eval_adapter_gate_mean` = **0.5**, `eval_adapter_gate_std` = **0** exactly —
  spatially uniform, not a learned mask.
- Checkpoint `gate_head.head.weight` RMS **0.00061**, bias **0.00060**, against a
  zero init — it barely moved before freezing.
- Contrast, the AVID reference: `mask_std` ≈ 0.057 (a real spatial gate) with
  `mask_mean` climbing 0.5 → 0.63
  ([[../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]).

## Blast radius — audited 2026-07-31 across every config setting `gate_cap`

| config | bias | cap | σ(bias) | status |
|---|---|---|---|---|
| `wan22/..._gatelow_capshift_acwm_robotarm` | 0.0 | 0.5 | 0.500 | **FROZEN** — `ncztxyyo`, D2 matrix arm |
| `wan22/diffusion_wan22_action_rt1` | 0.0 | 0.5 | 0.500 | **FROZEN** — the RT-1 arm |
| `wan22/diffusion_wan22_shortcut_actionfree_robotarm` | 0.0 | 0.5 | 0.500 | **FROZEN** — `pzmc2orq`, **D3 curvature run** |
| `wan22/diffusion_wan22_shortcut_openvid` | 0.0 | 0.5 | 0.500 | **FROZEN** |
| `wan22/..._cap50_warmup_acwm_pushblock` | 0.0 | 0.5 | 0.500 | **FROZEN** |
| `wan22/..._gatelow_capshift_acwm_pushblock` | 0.0 | 0.5 | 0.500 | **FROZEN** |
| all SkyReels configs, `..._gatelow_cap_sigmashift_metaworld`, `..._nobase_gatecap_overfit_metaworld` | 0.0 | 0.9 | 0.500 | OK (cap well above init) |

**Six configs affected in total.** My first pass at this table was wrong in both
directions — the audit grep matched `gate_bias`/`gate_cap` values inside *comment*
lines documenting previous settings, which produced one false positive
(`..._gatelow_cap_sigmashift_metaworld` flagged as bias 4.0 / dead-at-0; it is
actually 0.0 / 0.9 and fine) and hid the two Push Cube configs above. Audits of
YAML must strip comments before matching. The two empirical claims below are
unaffected — they rest on wandb `gate_std` values, not on the config scan.

**D3 is affected, not just D2.** `pzmc2orq` (Wan) shows `gate_mean` 0.5 /
`gate_std` **0**; `t4bp8nki` (DC, no cap) shows 0.596 / 0.0789 with ~2× the
adapter contribution. The 2026-07-29 "DC consistency loss ~68× Wan's" curvature
signature therefore compared a **frozen 50/50 base-anchored blend** against a
**free, diverging** one — an asymmetry that would depress Wan's consistency loss
regardless of trajectory curvature. Caveat added to
[[../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]].

## Blast radius (original note)

`configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml:57`
carries the comment *"gate_cap: 0.5 — matched to the live pushblock run
(2026-07-27); was 0.9"*. So the change landed **the day before** the
2026-07-28 matrix, and Wan's `0.0056` and the Push Cube runs were measured with a
frozen gate. Any config pairing `gate_bias: 0.0` with `gate_cap: 0.5` is affected.

**DynamiCrafter is NOT affected** — `diffusion_dc_acwm_robotarm.yaml` states
"Clean-baseline intent: no gate_cap / warmup / sigma_shift". The 2026-07-30/31
parity arms are clean.

## Honest scoping — this is probably NOT the cause of the blindness

A gate frozen at 0.5 still gives the adapter **50% weight**; it is not starved.
What is lost is *spatial adaptivity* — the gate can never learn where to apply
the adapter. Wan's blindness looks better explained by the adapter's prediction
being 98.5% cosine-similar to the base (`eval_adapter_pred_base_cosine` 0.985),
i.e. the older cloning failure. Fix this because it is wrong, not because it is
expected to restore action-following.

## Fixed 2026-07-31

- **Guard in code** (`models/adapted_model.py`): construction now raises if
  `gate_cap <= sigmoid(gate_bias)`, naming both values and the consequence. A
  silent knife-edge is what made this invisible for four days.
- **Six configs** raised to `gate_cap: 0.9`, each with an inline note recording
  the previous value and why it changed.

## Fix (original analysis)

Require `gate_cap` **strictly greater** than σ(`gate_bias`), and fail loudly
otherwise — a silent knife-edge is exactly what made this invisible for four
days. Restoring `gate_cap: 0.9` with `gate_bias: 0.0` (the pre-2026-07-27
setting) satisfies it.

Consider also: clamping the **logit** rather than the post-sigmoid gate would
preserve gradient below the bound and avoid the dead zone entirely.

## Re-measure after fixing

`ncztxyyo` (Wan) and the Push Cube runs, since their gate was frozen throughout.

## Cleanup 2026-08-01 — **FIXED**

Guard added in `models/adapted_model.py` (raises if gate_cap <= sigmoid(gate_bias)); six configs corrected. Confounded D3 result flagged in its note.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
