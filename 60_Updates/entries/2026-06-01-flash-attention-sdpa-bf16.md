---
date: 2026-06-01
category: finding
deliverable: D1
meeting:
sources:
  - "[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]"
  - "[[../../20_Tickets/bug-training-hyperalign-oom-flash-attention]]"
  - "[[entries/2026-05-25-hyperalign-oom-h100-blocker]]"
---

# Flash attention is now firing on the frozen backbone — but it's not what's OOMing HyperAlign

## What

Moved the frozen DynamiCrafter spatial attention off the hand-rolled `einsum`
softmax (which materialises the full `(N, N)` score matrix) onto PyTorch's
`F.scaled_dot_product_attention`. Getting the Flash kernel requires **two
independent levers**: a 4D `(B, H, N, D)` SDPA layout (the dispatcher only
routes to Flash/mem-efficient for rank-4 inputs — the big VRAM win), *and*
bf16 autocast in the trainer (Flash only fires under fp16/bf16; fp32 falls back
to the mem-efficient backend). Also removed a broken `xformers` routing path
that autocast couldn't cast into and that crashed under the env's torch build.

The important finding for the meeting: **this does not fix the HyperAlign
OOM.** Flash firing rules out the original "attention isn't using flash"
hypothesis; the dominant cost is almost certainly HyperAlign-specific —
it runs the ~1.4B base forward **twice** per step (reference + adapted pass)
plus captures intermediate hidden states for the hypernetwork — orthogonal to
attention.

## Why it matters

The frozen 1.4B backbone is now trainable on commodity GPUs (this closed the
earlier 24 GB-card OOM). But it reframes the still-open H100 blocker
([[entries/2026-05-25-hyperalign-oom-h100-blocker]]): the fix is **not** more
flash attention — it's the double-forward + hidden-state capture, which needs
quantifying before HyperAlign can train at the intended config.

## Evidence / sources

- Mechanism, the two levers, and the autocast/`prediction.float()` seam are
  documented in [[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]
  (commit `cca6a88`).
- Engineering-equivalence checks (RTX 3090, torch 2.11.0+cu130): SDPA vs.
  manual softmax `max|Δ| = 6.7e-08` in fp32; `training_step` smoke test bf16
  vs. fp32 loss matched to 5 decimals (1.01112 vs 1.01114); backend probe at
  the 4D layout confirms `FLASH_ATTENTION` is selected under bf16. _These are
  correctness/equivalence checks on synthetic inputs, not model-quality
  metrics._

## Next

Resolve [[../../20_Tickets/bug-training-hyperalign-oom-flash-attention]]
(still open, actively biting): capture a real OOM trace, probe the SDPA backend
on the *actual* HyperAlign run path, and quantify the double-forward +
hidden-state-capture memory cost. The condition-only hypernetwork variant
([[../../20_Tickets/feat-adapter-condition-only-hypernetwork]]) is a candidate
mitigation.
