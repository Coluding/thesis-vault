---
type: bug
scope: adapter
status: open
priority: high
created: 2026-07-14
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[feat-training-adapter-contribution-magnitude-logging]]", "[[feat-adapter-dynamicrafter-output-on-wan-base]]", "[[../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]", "[[feat-training-grad-accumulation-warmup]]", "[[feat-adapter-wan-per-frame-adaln]]"]
---

# bug: `mask_mix` + `gate_bias=4.0` throttles adapter gradient ~50×

## What

Both live Wan configs (`diffusion_wan22_avid_i2v_metaworld.yaml`,
`diffusion_wan22_avid_xattn_i2v_metaworld.yaml`) use `composition: mask_mix` with
`gate_bias: 4.0`. At init, `gate = σ(gate_logit + 4.0) ≈ 0.982`. The composition is
`base·gate + pred·(1−gate)`, so the gradient reaching the adapter's own prediction
head scales by `(1 − gate) ≈ 0.018` — a **~50× attenuation of the only training
signal the adapter gets**, every step, for the whole run
(`models/adapted_model.py` composition math, `adapters/output/wan.py` gate head,
`adapters/factory.py`'s `predict_full=is_mask_mix` routing).

The gate has **no annealing schedule anywhere in the codebase** (grepped) — it
either stays near-saturated for the whole run, or has to fight its way off ~0.982
purely from a throttled gradient.

Confirmed independently by two lenses in the 2026-07-14 exploration workflow (see
[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]) and re-derived directly
from the composition code.

## Why it matters

Orthogonal to the headroom/low-motion story — this throttles whatever gradient
exists, action-relevant or not. It's the single highest expected-effect-per-effort
item in the diagnosis: removing it doesn't guarantee learning improves, but leaving
it in place confounds every other experiment (any negative result could just be
"the gradient never had a chance to move the gate").

## External validation (2026-07-14) — the reference AVID implementation confirms this

Read the vendored, unmodified upstream AVID repo directly
(`external_repos/avid/latent_diffusion/libs/dynamicrafter/lvdm/models/avid.py`,
`AVIDAdapter.apply_model`) and its real published configs
(`configs/train/avid/{avid_11M,avid_34M,avid_145M}.yaml`). Their composition is
**the exact same formula** we use: `combined = base*mask + adapter*(1-mask)`,
`mask = σ(raw_mask + init_mask_bias)` — confirms our `mask_mix` faithfully
replicates AVID's mechanism.

**But their `init_mask_bias: 0.0` in all three real configs** → `mask = σ(0) =
0.5` at init — a balanced 50/50 split between base and adapter. Our
`gate_bias: 4.0` → `σ(4) ≈ 0.982` — **98% base, 2% adapter** at init. This is a
much more conservative starting point than the paper authors' own reference
recipe, and it's the ~50× gradient throttle quantified above.

**Bonus finding:** `AVIDAdapter.prepare_adapter` also supports a
`pretrain_steps` param that forces `mask = 0` (pure adapter output, no base at
all) for the first N steps before switching to the learned blend — i.e. exactly
the `composition: replace` "crazy experiment" diagnostic already run this
session
([[feat-adapter-dynamicrafter-output-on-wan-base]] §"Crazy experiment"),
except AVID exposes it as a **staged warmup**, not just a diagnostic. None of
the three real AVID configs set it (`pretrain_steps` defaults to 0 — the
published runs don't use this warmup), so it's not what made AVID's original
results work, but it's an available, paper-precedented option worth
considering as a real fix, not just a diagnostic.

## Fix

- **Cheapest, most directly comparable to the paper:** if staying on
  `mask_mix`, change `gate_bias: 4.0` → `gate_bias: 0.0` to match the reference
  AVID recipe exactly (σ(0)=0.5). Single-value config change, zero risk, and now
  has an external, paper-precedented justification rather than just our own
  derivation.
- **Wan backbone (`backbone: wan`):** switch `composition: mask_mix` →
  `gated_residual` in the config. This routes through the existing
  `predict_full=is_mask_mix` factory logic
  (`adapters/factory.py`, `adapters/output/wan.py`) which already auto zero-inits
  the head for the residual composition — no code change needed, config-only.
- **DynamiCrafter backbone (`backbone: unet`):** no equivalent `predict_full`
  bypass exists for `DynamicCrafterOutputAdapter` yet
  (`adapters/output/dynamicrafter.py` — `output_mask` always implies the
  `mask_mix`-style gated prediction path). Needs a small code addition to support
  a `gated_residual`-equivalent composition, mirroring the Wan path.
- **Optional, paper-precedented staged variant:** a `pretrain_steps`-style full
  override (mask=0) for the first N steps, then switch to the learned
  `gate_bias: 0.0` blend — mirrors `AVIDAdapter.prepare_adapter`'s
  `pretrain_steps` param exactly. Not used in AVID's own published runs, so not
  necessary for a fair comparison, but a legitimate option if `gate_bias: 0.0`
  alone still isn't enough.

## Validate

- Log gate value over training (pairs with
  [[feat-training-adapter-contribution-magnitude-logging]]) — confirm it moves
  off ~0.982 substantially faster / further under `gated_residual` than under
  `mask_mix`.
- Short matched-step re-run comparing `denoise_adapter_delta` /
  `probe_denoise_delta` trend (already-implemented diagnostics, see
  [[../10_now/architecture]]) between the two compositions.

## Guardrails

- `gated_residual` changes what the adapter *emits* (a ~0-init delta vs. a
  standalone prediction) — re-check any eval/inference code path that assumes
  `output_kind="prediction"` (mask_mix) still handles `output_kind="delta"`
  correctly.
- Do this before drawing conclusions from any of the do-now experiments in
  [[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] — it confounds all
  of them.

## MAJOR UPDATE (2026-07-21): `gate_bias: 0.0` is NOT sufficient — the gate saturates dynamically

The single-clip overfit run (wandb `uxrst2k5`, gate_bias 0.0 — logged
`adapter_gate_mean` 0.5 = σ(0) at step 1) shows the saturation is
**loss-driven, not init-driven**: from the balanced 50/50 start the gate
marched to 0.99 within ~70 steps, adapter grad norm collapsed 4.4 → 0.003,
and learning stopped — on a single memorizable clip. Mechanism: at init
`pred` is near-garbage, so opening the gate toward the (excellent frozen)
base is the fastest loss reduction; once σ(g) ≈ 0.99, gradients into `pred`
scale by (1−σ(g)) ≈ 0.01 and the gate's own gradient by σ′(g) ≈ 0.01 — both
learning channels die together. Classic highway/MoE-style collapse. Numbers:
[[../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]].

The "Fix" list above therefore needs upgrading — the bias change (done) was
necessary but not sufficient. Live candidates, in rough order:

1. **Gate cap:** clamp σ(g) ≤ ~0.9 so `pred` always receives ≥10% of the
   gradient (cheap, one line in the composition).
2. **Gate penalty:** small loss term on mean σ(g) pushing away from
   pure-base.
3. **`pretrain_steps`-style staged warmup** (paper-precedented, see the
   AVID bonus finding above): force mask=0 (pure adapter) for the first N
   steps so `pred` becomes competent before the blend is learnable.

Note the AVID reference run on our data (2026-07-15 section below) did NOT
saturate (mask 0.52 → 0.63 and moving) — difference candidates: their base
is far weaker than Wan2.2-TI2V-5B (less incentive to hand everything to the
base), different data regime. The stronger the frozen base, the stronger the
copy-through pull.

## Real-run confirmation (2026-07-15) — upgraded from code comparison to matching evidence

[[../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]:
the reference AVID code, run on our own MetaWorld data (not just RT1), with
`init_mask_bias: 0.0`, shows a clean ~9.5× monotonic loss drop over ~800 steps
and `mask_mean` climbing steadily from its 0.5198 init (≈theoretical σ(0)=0.5)
to 0.6326 — actively moving, not stuck. This is no longer just a static
comparison against the paper's config values — it's a matching real run on our
task, on the composition mechanism we're diagnosing. Materially raises
confidence that `gate_bias: 4.0` (not something more fundamental about the
adapter approach) is the dominant confound in our own runs.

## Cleanup 2026-08-01 — **RESOLVED-BY**

The gate story is now complete: saturation is real (uxrst2k5) AND the `gate_cap` countermeasure was itself a freeze trap ([[bug-adapter-gate-cap-equals-init-freezes-gate]], fixed + guarded).

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
