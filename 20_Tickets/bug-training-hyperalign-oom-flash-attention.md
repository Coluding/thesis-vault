---
type: bug
scope: training
status: open
priority: high
created: 2026-05-25
updated: 2026-05-25
resolution:
resolution_note:
closed_at:
related:
  - "[[done/bug-training-oom-after-image-cross-attn-wiring]]"
  - "[[feat-adapter-condition-only-hypernetwork]]"
---

# HyperAlign OOMs very quickly, even on an H100

> 🔴 **STILL OPEN — actively biting.** A fix was attempted but the OOM
> **still reproduces**. This is not resolved; do not close. Must be fixed —
> keeping it visible so it doesn't get forgotten. _What exactly was tried,
> and why it didn't take — needs to be filled in (see Attempted fixes)._

## Symptom

HyperAlign training runs go CUDA-OOM **very quickly, even on an H100**
(80 GB). User-reported; no profiler trace or run id captured yet —
_traceback / config / exact alloc size needs verification_.

## Suspected cause (user hypothesis — not yet confirmed)

Miscalibrated attention: the model **is not properly using flash
attention**, so the full `(N, N)` spatial-attention score matrix gets
materialized and blows up VRAM.

> ⚠️ This is the reported hypothesis, recorded as such — **not** a verified
> root cause. It also partially contradicts known state (see below), which
> is exactly why it needs investigation rather than a blind fix.

## Context that complicates the hypothesis

The SDPA + bf16-autocast Flash path was **already shipped** for the frozen
DynamiCrafter backbone — see [[../30_Knowledge/tech/flash-attention-sdpa-bf16]]
(commit `cca6a88`). That work moved spatial attention off the hand-rolled
`einsum` softmax onto `F.scaled_dot_product_attention`, and it closed the
earlier 24 GB-card OOM ticket
([[done/bug-training-oom-after-image-cross-attn-wiring]], shipped 2026-05-22).

So if HyperAlign is *still* OOMing, either (a) Flash is silently not firing
on the config/path actually being run, or (b) the dominant memory cost is
**not** the spatial-attention matrix at all. Both need ruling out.

## Investigation candidates (analysed estimates — shown reasoning, unverified)

Listed roughly by how HyperAlign-specific they are. None are confirmed;
each line says what to check.

1. **amp / Flash not actually engaged on the run's config.** Per the tech
   note, Flash only fires under bf16 autocast *and* the 4D SDPA layout;
   fp32 falls back to the mem-efficient backend (memory win but no Flash
   kernel). `diffusion_hyperalign_metaworld.yaml` was the config given
   `amp_dtype: bf16` — **verify the config used on the H100 actually sets
   `training.extra.amp_dtype: bf16`.** If it's fp32/none, that's the first
   thing to fix. _Which config was launched — needs verification._

2. **HyperAlign runs the base forward TWICE per step.** From
   [[../30_Knowledge/related-work/hyperalign]]: `HyperAlignAdapter.forward`
   does a reference pass (`base_model(...)`, hyperalign.py:269) *and* an
   adapted pass (hyperalign.py:279). Two full forwards through a ~1.4B UNet
   ≈ roughly double the activation memory of a plain LoRA step. This is
   HyperAlign-specific and a strong reason it would OOM "very quickly"
   where other adapters don't — and it is **orthogonal to flash attention**.
   _Whether both passes retain activations for backward — needs
   verification._

3. **Intermediate hidden-state capture for the hypernet.** HyperAlign
   captures intermediate UNet hidden states to feed the hypernetwork. Those
   captured tensors are extra resident activations on top of the two
   forwards. _Confirm whether captures are detached / freed — needs
   verification._

4. **Non-SDPA attention branches still build `(N, N)`.** The tech note notes
   the relative-position and explicit causal-mask branches keep the manual
   softmax path. Those are temporal (small `N`), so shouldn't dominate — but
   confirm the spatial path on the H100 config is genuinely hitting the 4D
   SDPA branch and not one of the fallbacks.

5. **Config scaled up vs. the 24 GB run.** "Even on H100" implies more
   headroom than before, yet it still OOMs — so something grew (batch_size,
   `temporal_length`, latent resolution) or activation/gradient
   checkpointing is off. _Compare the H100 config against the resolved
   24 GB run's settings — needs verification._

## Attempted fixes

- **2026-05-25 — a fix was attempted; OOM still reproduces.** The specific
  change(s) tried and the evidence it still fails are _not yet captured —
  needs to be filled in_. (Don't lose this: write down what was changed, on
  which config, and the trace it still threw, so we don't re-try the same
  dead end.)

## Definition of done

- Captured a real OOM trace (run id / config / offending alloc size).
- Determined whether Flash is firing (SDPA backend probe on the actual run
  path) — settles candidate 1 & 4.
- Quantified the double-forward + hidden-state-capture memory cost —
  settles candidate 2 & 3.
- A fix lands that lets the intended HyperAlign MetaWorld config train on an
  H100 without OOM, with the chosen mechanism documented.

## Related

- [[../30_Knowledge/tech/flash-attention-sdpa-bf16]] — the shipped SDPA+bf16 Flash work
- [[../30_Knowledge/related-work/hyperalign]] — double-forward + hidden-state capture
- [[done/bug-training-oom-after-image-cross-attn-wiring]] — the earlier (resolved) OOM
- code: `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py`
- code: `src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py`
- code: `src/generative_flow_adapters/training/trainer.py`
