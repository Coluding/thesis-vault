---
type: feat
scope: adapter
status: open
priority: medium
created: 2026-05-25
updated: 2026-05-25
resolution:
resolution_note:
closed_at:
related:
  - "[[../30_Knowledge/related-work/hyperalign]]"
  - "[[../30_Knowledge/tech/mask-mix-gate]]"
  - "[[../30_Knowledge/tech/structural-encoder]]"
  - "[[bug-training-hyperalign-oom-flash-attention]]"
  - "[[feat-adapter-flops-per-step-estimator]]"
  - "[[experiments/exp-adapter-param-matched-comparison]]"
  - "[[../50_Decisions/open/multimodal-adapter-broadening]]"
---

# Condition-only ("static") hypernetwork — generate the weight delta from the conditioning chunk alone

Build a lightweight hypernetwork adapter whose generated weights depend
**only on the conditioning chunk** — not on the noisy latent `x_t`, the
timestep `t`, or the base model's features/output. Feed the condition
embedding in, get a weight delta out, inject it into the frozen base, run
the base. A "super soft", maximally-cheap point on the hypernetwork
spectrum that should save a lot of memory.

## Why

The current HyperAlign adapter is **memory-heavy** and is the subject of an
open OOM ticket ([[bug-training-hyperalign-oom-flash-attention]], which
notes "HyperAlign runs the base forward TWICE per step"). Its hypernetwork
is conditioned on the base model's *internal features* and on `(x_t, t)`,
which forces:

- a feature-capture pass over the frozen base with **retained activations**
  for the hypernetwork's backward, and
- a transformer decoder that **cross-attends a large captured-feature
  memory** (many encoder blocks × spatial tokens).

If a much cheaper hypernetwork that conditions on the (small) conditioning
chunk alone is competitive, it's both a strong memory win **and** a clean
extra family/variant for the adapter comparison
([[experiments/exp-adapter-param-matched-comparison]]). It also makes an ideal
cheap per-modality adapter if the multimodal direction goes ahead
([[../50_Decisions/open/multimodal-adapter-broadening]]).

## Current state (verified 2026-05-25)

`HyperAlignAdapter.forward` (`adapters/hypernetworks/hyperalign.py:254`):

- **line 272** — `reference = base_output if base_output is not None else
  self.base_model(...)`: base reference pass (forward #1).
- **line 273 / 363** — `_resolve_hyper_factors(x_t=x_t, t=t, cond=cond)`:
  the generated LoRA factors **depend on `x_t`, `t`, and `cond`**.
- **line 282** — `adapted = self.base_model(...)`: base adapted pass with
  factors injected (forward #2).
- **lines 442–507** — the hypernetwork's memory is built **from captured
  base encoder features** (`_build_encoder_memory` →
  `_build_memory_from_captured_features`).
- **lines 91, 122, 698–703** — `_feature_store` + `register_forward_hook`
  capture and retain those base features.

So today: `ΔW = f(x_t, t, cond, base_features)`. The base is entangled with
the hypernetwork on the input side (features feed the hypernet) and the
output side (factors modify the base).

## What — the proposed variant

A new hypernetwork variant (registered under `adapters/hypernetworks/`,
new key in `adapters/factory.py`, e.g. `hyper` architecture
`condition_only` / `static`) where:

- `ΔW = h(c)` with `c = cond["embedding"]` (the
  [[../30_Knowledge/tech/structural-encoder]] output) — **no `x_t`, no `t`,
  no base features**.
- `h` is a small MLP / token-decoder over the condition chunk `[B, d_cond]`
  that emits the low-rank factors `(down, up)` per target module — *not* a
  transformer cross-attending a base-feature memory.
- **No feature-capture hooks** (`_feature_store` unused) ⇒ no retained
  base-encoder activations.
- The factors are injected into the frozen base exactly as today
  (`set_dynamic_hyper_factors`), then the base runs once for the prediction.

### "Run it once" — the key property

Because `ΔW` is independent of `t` and `x_t`, the hypernetwork is evaluated
**once per (sample, condition)** and the resulting weight delta is **reused
across the entire denoising trajectory**. Current HyperAlign must recompute
factors at every sampling step (they depend on `x_t, t`). This is a large
**inference-time** saving (one hypernet eval per rollout instead of one per
step) on top of the training-time memory saving.

## Memory / compute analysis (analysed estimate — to verify with a profiler)

Wins, roughly in order of expected impact:

1. **No retained base-feature activations.** Dropping the `_feature_store`
   capture removes activation memory the OOM ticket suspects is dominant.
2. **Tiny hypernetwork.** Cross-attention over a big captured-feature memory
   → a small MLP over a `[B, d_cond]` vector. Removes the `(N,N)` memory
   attention entirely.
3. **Inference: factors computed once per trajectory**, reused over all
   steps.
4. **Possibly one fewer base pass.** The reference pass (#1) only exists for
   the `add`/`mask_mix` composition; if we adopt a composition that reads
   the adapted output directly (`replace`) the reference can be skipped.
   _Needs the composition choice below._

Not a win / unchanged: the adapted base forward itself still runs (it is the
prediction).

## Trade-off (what "soft" costs)

The correction is a **fixed per-condition weight shift, blind to the noise
level and to the base's current output**. It cannot react to *how* noisy
`x_t` is or *what* the base predicted at this step. Expect it to underfit
dynamics that need timestep- or input-dependent correction. This is exactly
the axis the param-matched comparison should measure: capacity vs.
cost across the hypernetwork spectrum (full HyperAlign ↔ condition-only).

This is essentially **conditional LoRA**: `ΔW = h(c)`, a per-condition
low-rank delta — worth checking against the conditional-hypernetwork /
LoRA-generation literature for prior art before claiming novelty.
_needs verification._

## Open questions

- **Composition mode.** `add` (needs the no-grad base reference for
  `adapted − reference`) vs `replace` (read adapted output directly, skip
  the reference) vs `mask_mix` (needs base for the gate blend). Picking
  `replace` maximises the memory win but changes the comparison's
  composition axis vs the other families.
- **Condition on `t` after all?** A middle variant `ΔW = h(c, t)` keeps the
  per-step recompute (loses the inference reuse) but regains noise-level
  adaptivity — a natural ablation between this and full HyperAlign.
- **Does the structural encoder's `cond["embedding"]` carry enough signal**
  to drive useful weight deltas on its own? It was designed to, but it has
  never been the *sole* input to a hypernetwork.
- **Param/FLOPs accounting.** Add a `flops_per_step` for this variant per
  [[feat-adapter-flops-per-step-estimator]] so it can enter the matched
  comparison grid.

## Acceptance / done-when

- New hypernetwork variant builds via `adapters/factory.py` from a YAML
  config and trains on the existing action-conditioned MetaWorld setup.
- Verified (profiler) that peak VRAM is materially below full HyperAlign on
  the same base + batch, with the base feature hooks confirmed off.
- A config under `configs/` mirroring `diffusion_hyperalign_metaworld.yaml`
  but selecting the condition-only variant.
- One row added to the adapter-family comparison
  ([[experiments/exp-adapter-param-matched-comparison]]).

## Related

- [[../30_Knowledge/related-work/hyperalign]] — the full-fat hypernetwork
  this strips down.
- [[../30_Knowledge/tech/mask-mix-gate]] — composition options + how
  HyperAlign currently produces its output.
- [[bug-training-hyperalign-oom-flash-attention]] — the memory problem this
  variant sidesteps by construction.
- [[../50_Decisions/open/multimodal-adapter-broadening]] — candidate cheap
  per-modality adapter there.
- Code: `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py:254,272,273,282,363,442-507,698-703`
- Code: `src/generative_flow_adapters/adapters/factory.py` — register the new variant
- Code: `src/generative_flow_adapters/conditioning/encoders.py:109-151` — the `cond["embedding"]` source
