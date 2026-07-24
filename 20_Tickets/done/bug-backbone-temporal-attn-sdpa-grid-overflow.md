---
type: bug
scope: backbone
status: done
priority: high
created: 2026-06-17
updated: 2026-06-17
resolution: shipped
resolution_note: Fixed in the working tree on 2026-06-17 by chunking the batch axis in the only_self_att temporal path. User-reported fix.
closed_at: 2026-06-17
related:
  - "[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]"
  - "[[bug-training-oom-after-image-cross-attn-wiring]]"
---

# Temporal self-attention SDPA kernel launch fails (cudaErrorInvalidConfiguration) at large batch·resolution

## Symptom

In the **temporal** self-attention path of the DynamiCrafter backbone, the
batch dimension fed to the attention blocks is `(b·h·w)` — batch times spatial
resolution. PyTorch's Flash / memory-efficient SDPA CUDA kernels launch a grid
whose y/z dimension is `batch · n_heads`, and that dimension must stay below the
hardware limit of **65,535**. At large batch sizes / spatial resolutions this
was exceeded, so the kernel launch failed with **`cudaErrorInvalidConfiguration`**
— a grid/thread overflow, **not** an OOM.

> Distinct from [[bug-training-oom-after-image-cross-attn-wiring]] (a memory
> blow-up in the *spatial* cross-attention `(N,N)` matrix). This one is a kernel
> *launch* failure from too large a grid dimension, and lives on the temporal
> self-attention path.

## Root cause

`TemporalTransformer.forward`'s `only_self_att` branch ran each transformer
block over the full `(b·h·w)` batch at once. With SDPA's grid y/z dim =
`batch · n_heads`, a large `b·h·w` pushes `batch · n_heads ≥ 65,535` and the
launch is rejected.

## Fix (two parts)

`src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py`

1. **Store `n_heads` on the module** (`self.n_heads = n_heads`, attention.py:410)
   so the head count is available at forward time — it's needed to compute the
   safe chunk size. Added in `TemporalTransformer.__init__`:

```python
self.in_channels = in_channels
self.n_heads = n_heads          # <-- added
inner_dim = n_heads * d_head
```

2. **Chunk the batch axis in the `only_self_att` path** so each chunk's
   `chunk · n_heads` stays below 65,535.

### Before — single full-batch pass over each block

```python
if self.only_self_att:
    ## note: if no context is given, cross-attention defaults to self-attention
    for i, block in enumerate(self.transformer_blocks):
        x = block(x, mask=mask)          # x.shape[0] == b·h·w, passed whole
    x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
```

Here `x` has leading dim `(b·h·w)`. Each `block(x, …)` runs SDPA over that entire
batch in one kernel launch, so the launch grid's batch·n_heads dimension is
`(b·h·w) · n_heads` — unbounded as batch/resolution grow.

### After — split the batch into grid-safe chunks, then concatenate

```python
if self.only_self_att:
    ## note: if no context is given, cross-attention defaults to self-attention
    # ... (comment block explaining the 65,535 grid limit) ...
    bhw = x.shape[0]
    chunk = max(1, 65535 // self.n_heads)
    for i, block in enumerate(self.transformer_blocks):
        if bhw <= chunk:
            x = block(x, mask=mask)                  # fits → unchanged fast path
        else:
            outs = []
            for s in range(0, bhw, chunk):
                e = s + chunk
                m = mask[s:e] if mask is not None else None
                outs.append(block(x[s:e], mask=m))   # each launch ≤ 65,535·... grid
            x = torch.cat(outs, dim=0)
    x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
```

When the batch fits (`bhw <= chunk`) it runs exactly as before — same single
launch, zero overhead. When it's too large, the batch is split into slices of at
most `chunk` rows, each block is launched once per slice, and the outputs are
concatenated back along dim 0. This mirrors the per-batch loop already used in
the cross-attention path below (`for j in range(b)`, attention.py:498-503, whose
comment already flagged the same 65,535 limit).

## How this prevents the grid overflow (precisely)

SDPA's Flash / mem-efficient CUDA kernels parallelise over `(batch, n_heads)` by
mapping that product onto the launch grid's y/z dimension. CUDA caps each grid
dimension at **65,535**. So the *only* launch parameter that matters for this
error is:

```
grid_dim = effective_batch · n_heads      # must be < 65,535
```

- **Before:** `effective_batch = bhw = b·h·w`, so `grid_dim = (b·h·w)·n_heads`.
  Once `b·h·w ≥ 65535 / n_heads`, the launch is rejected with
  `cudaErrorInvalidConfiguration`. Example: `n_heads = 8` → the ceiling on
  `b·h·w` is `8191`; a single 64×64 latent (`h·w = 4096`) with `b = 2` already
  gives `b·h·w = 8192` and tips over.
- **After:** `chunk = floor(65535 / n_heads)` is the largest batch slice whose
  `chunk · n_heads` is still `< 65,535` (for `n_heads = 8`, `chunk = 8191`).
  Every kernel launch now uses `effective_batch = min(chunk, remaining) ≤ chunk`,
  so `grid_dim = effective_batch · n_heads ≤ chunk · n_heads < 65,535` by
  construction — the grid dimension can never reach the cap regardless of how
  large `b·h·w` is. `max(1, …)` guards the degenerate `n_heads > 65535` case so
  `chunk` is at least 1.

The fix doesn't shrink the *work*, it shrinks the *per-launch* batch so each
launch's grid is legal; the loop covers the whole batch across multiple launches.

## Why it's safe (no numerical / perf change)

- **Numerics unchanged.** Self-attention is independent across the batch axis —
  row `i`'s output depends only on row `i`'s tokens, never on other batch rows.
  Splitting `(b·h·w)` into contiguous slices and concatenating is therefore
  exactly equal to the single-pass result (the mask is sliced with the same
  `[s:e]` so each chunk keeps its own mask rows).
- **Flash preserved.** Each chunk still goes through the same `block(...)` → SDPA
  path, so the Flash / mem-efficient kernel still fires; chunking only changes
  how many times it's launched, not which backend runs.
- **Wall-clock / memory effectively unaffected.** The batch axis is
  embarrassingly parallel, so the same total FLOPs run as a few sequential
  launches instead of one; peak memory is no higher (it's a slice of the batch).

## Status / provenance

- Fix present in the working tree of `generative-flow-adapters` (uncommitted as
  of this write, on top of commit `b09e8d5`). `git diff` on `attention.py`:
  +20 / -1.
- The grid-limit fact (65,535 cap on the SDPA launch grid's batch·n_heads
  dimension) is reflected verbatim in the in-code comment at attention.py:474-481.

## Files

- `src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py`
  — `TemporalTransformer`: `self.n_heads` store (line 410), chunked
  `only_self_att` loop (lines 472-493).

## Related

- [[../../30_Knowledge/tech/flash-attention-sdpa-bf16]] — the SDPA + bf16 work this builds on
- [[../../30_Knowledge/tech/sdpa-vs-manual-attention-expressiveness]] — why this path uses SDPA at all (relative-position / causal flags off), and the SDPA expressiveness boundary
- [[bug-training-oom-after-image-cross-attn-wiring]] — the earlier (memory) attention bug, different failure mode
