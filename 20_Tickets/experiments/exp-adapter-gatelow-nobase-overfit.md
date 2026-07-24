---
type: exp
scope: adapter
status: open
priority: high
created: 2026-07-21
updated: 2026-07-21
resolution:
resolution_note:
closed_at:
related: ["[[exp-adapter-replace-nobase-overfit]]", "[[../bug-adapter-gate-saturation-mask-mix]]", "[[../done/exp-training-single-clip-overfit]]", "[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"]
---

# exp: single-clip overfit, mask_mix gatelow, NO base-output input

User-proposed follow-up (2026-07-21) to the uxrst2k5 overfit failure: keep
the thesis composition (`mask_mix`, `gate_bias: 0.0`) but set
`condition_on_base_outputs: false` — the adapter no longer sees the base
velocity as an input, removing the identity-copy path while the gate/blend
stays intact. Complements [[exp-adapter-replace-nobase-overfit]] (no gate at
all): together the two runs separate the two measured traps — gate
saturation vs identity-on-input.

**Two arms, run in parallel (decided 2026-07-21, grilling):**

- **(i) raw** — `configs/diffusion_wan22_avid_xattn_gatelow_nobase_overfit_metaworld.yaml`:
  the pure one-variable ablation vs uxrst2k5 (only the base input removed).
- **(ii) capped** — `configs/diffusion_wan22_avid_xattn_gatelow_nobase_gatecap_overfit_metaworld.yaml`:
  identical + `gate_cap: 0.9` (new `AdapterConfig.gate_cap` option, clamp on
  the post-sigmoid mask_mix gate in `AdaptedModel._compose`) so pred keeps
  ≥10% of the gradient even if the gate saturates — guarantees the
  input-conditioning question gets answered.

Both: pure denoise loss, eval every 75 steps, σ unshifted for comparability;
launch commands in the config headers. With
[[exp-adapter-replace-nobase-overfit]] this forms the triangle
uxrst2k5 (gate, base-input) / replace-nobase (no gate, no input) /
gatelow-nobase ±cap (gate, no input) that separates the two traps.

## Decision rule

- **pred improves AND gate stays mixed** ⇒ the concat input was the crutch;
  the full composition can learn once it's removed. Best-case outcome.
- **Gate saturates 0.5 → ~0.99 in ~70 steps again with grad-norm collapse
  (uxrst2k5 pattern)** ⇒ the gate trap alone is sufficient and this run
  cannot speak to the input-conditioning question; a gate cap / penalty /
  staged warmup ([[../bug-adapter-gate-saturation-mask-mix]] fix candidates)
  must land first.
- **Gate stays mixed but pred stalls at base level** ⇒ capacity/architecture
  limit, same reading as the replace-nobase sibling's negative branch.

## Known confound (flagged before launch)

uxrst2k5 showed the gate closes in ~70 steps while pred is still garbage;
without a gate cap this run risks reproducing that trajectory regardless of
the input change. Watch `adapter/gate_hist` + `adapter_grad_norm` from step
1 — if the gate pins early, stop the run and land the cap first instead of
burning the full budget.

## Related design question (not this run)

How the adapter should consume the base output at all (channel-concat =
maximally copyable → base-dropout → cross-attention over base tokens →
pooled-AdaLN → none) is an open design surface — decision note pending.

## Guardrails

- Diagnostic; single-clip overfit tests capacity/optimization, not action
  usage. Don't report as D2/D4 evidence.
- Prerequisite: 2026-07-20/21 code fixes committed + pulled on the training
  box (same as the replace-nobase sibling).
