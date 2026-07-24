---
date: 2026-06-19
category: finding
deliverable: D1
meeting: 2026-06-19
sources:
  - "code: src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py"
---

# A batch-size crash that read as OOM was actually a grid / index overflow in attention

## What

Raising the batch size crashed training and was first diagnosed as
out-of-memory. It was actually a **grid / index overflow** in the DynamiCrafter
attention path, **not** a memory limit. Fixed.

## Why it matters

Removes a false ceiling on batch size and a misleading failure mode: larger
batches were being treated as "OOM" when the real cause was indexing. Practical
rule for the project — **a bigger batch is not always an OOM**; confirm the crash
is memory before scaling the batch down or adding GPUs.

## Evidence / sources

Fix in `attention.py` (working tree at time of writing). Found while scaling
batch size on the shortcut / DynamiCrafter runs. Exact overflowing index /
commit `_needs verification_`.
