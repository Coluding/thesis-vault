---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../feat-adapter-dynamicrafter-output-on-wan-base]]", "[[../bug-adapter-gate-saturation-mask-mix]]", "[[exp-adapter-adaln-gatelow-metaworld-run]]"]
---

# exp: DC-UNet capacity adapter, gate + optimizer confounds fixed

**Config:** `configs/diffusion_wan22_dcunet_output_metaworld.yaml` — fixed
2026-07-15: `gate_bias: 4.0 → 0.0`, `grad_accum_steps: 4`,
`linear_warmup_steps: 250` applied directly (no historical cited runs on this
config yet, safe to edit in place). `action_per_frame: true` already set (DC
UNet's per-frame action wiring, added earlier).

## Why

The DC-UNet capacity experiment
([[../feat-adapter-dynamicrafter-output-on-wan-base]] — the direct test of
capacity vs. injection-mechanism as the answer to the weak-action-signal
finding) was explicitly blocked pending this fix — its own ticket says "apply
the gate-saturation fix... before running the real DC-UNet capacity
experiment." That's now done at the config level; this ticket is the actual
run.

11M tier (`act_cond_diffusion_wan48_11M.yaml`) — cheapest capacity tier,
appropriate for a first real run before scaling to 34M/145M.

## Procedure

Run to a comparable step count as [[exp-adapter-adaln-gatelow-metaworld-run]]
(the tiny-DiT AdaLN sibling at the same gate_bias/accumulation/warmup
settings) for a capacity-vs-architecture comparison, and against
[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]
(the AVID reference) for the "is this a healthy training run at all" check.

## Decision rule

- **Healthy, and clearly better than the 34M tiny-DiT AdaLN sibling** ⇒
  capacity is a real, independent lever beyond just fixing the gate —
  supports scaling to 34M/145M tiers.
- **Healthy but no better than the tiny-DiT sibling** ⇒ once the gate
  confound is removed, capacity isn't the binding constraint — the earlier
  "adapter helps but not enough" framing was really about the gate the whole
  time, not adapter size. Redirect further capacity investment.
- **Still unhealthy despite the fix** ⇒ something DC-UNet-specific — check
  the vendored-DynamiCrafter integration fixes from
  [[../feat-adapter-dynamicrafter-output-on-wan-base]] (per-frame timestep,
  context guard, attn2 skip) didn't introduce a subtler issue.

## Guardrails

Same VRAM risk as previously flagged — watch 24GB headroom with the 11M
tier before considering 34M/145M. Smoke-validated for wiring correctness
already; this run is about the science, not plumbing.
