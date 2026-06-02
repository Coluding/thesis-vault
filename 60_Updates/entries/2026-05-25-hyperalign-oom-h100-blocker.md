---
date: 2026-05-25
category: blocker
deliverable: D2
meeting:
sources:
  - "[[../../20_Tickets/bug-training-hyperalign-oom-flash-attention]]"
  - "[[../../20_Tickets/done/bug-training-oom-after-image-cross-attn-wiring]]"
  - "[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]"
  - "[[../../30_Knowledge/related-work/hyperalign]]"
---

# HyperAlign still OOMs quickly — even on an H100

## What

HyperAlign MetaWorld training goes CUDA-OOM **very quickly, even on an H100
(80 GB)**. A fix was attempted on 2026-05-25 but the OOM **still reproduces** —
this is open, not resolved. No OOM trace / run id / config has been captured
on the H100 yet, so the root cause is _not yet verified_.

## Why it matters

This blocks every HyperAlign MetaWorld run — which is the path to **D2
action-conditioned world-model evidence** and to the planned param-matched
adapter comparison. Until it's fixed, the hypernetwork adapter family produces
no empirical numbers. It's the top open training blocker.

What makes it puzzling: the earlier 24 GB-card OOM was **already resolved**
(shipped 2026-05-22, commit `cca6a88`) by moving spatial attention onto
`F.scaled_dot_product_attention` + bf16 autocast (the SDPA Flash path, see
[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]). So if HyperAlign still
OOMs with more headroom than before, either Flash isn't firing on the config
actually launched, or the dominant memory cost was never the spatial-attention
matrix.

## Evidence / sources

- Open ticket: [[../../20_Tickets/bug-training-hyperalign-oom-flash-attention]]
  (status: open, priority: high). 2026-05-25 fix attempt failed; **what was
  changed and the trace it still threw is not yet captured** — needs recording
  so we don't re-try the same dead end.
- Resolved precursor: [[../../20_Tickets/done/bug-training-oom-after-image-cross-attn-wiring]]
  — 24 GB-card OOM, shipped 2026-05-22 via the SDPA+bf16 work.
- Leading suspects (analysed estimates from the ticket, **unverified**):
  1. Flash not actually engaged on the launched config — verify
     `training.extra.amp_dtype: bf16` is set on the H100 run (fp32 keeps the
     mem-efficient backend but no Flash kernel).
  2. HyperAlign runs the base UNet forward **twice per step** (reference pass +
     adapted pass, see [[../../30_Knowledge/related-work/hyperalign]]) →
     roughly double the activation memory of a plain LoRA step. HyperAlign-
     specific and **orthogonal to flash attention**.

## Next

1. Capture a real OOM trace on the H100 (run id / config / offending alloc).
2. SDPA backend probe on the actual run path — settle whether Flash is firing.
3. Quantify the double-forward + hidden-state-capture memory cost.
4. Land + document a fix that lets the intended HyperAlign MetaWorld config
   train on an H100 without OOM.
