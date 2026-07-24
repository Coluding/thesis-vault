---
title: "When increasing batch size crashes your training, but memory isn't the issue"
slug: cuda-grid-limits-in-pytorch
type: blog-brainstorm
status: drafting
created: 2026-06-17
last_updated: 2026-06-19
deliverable: exploratory
sources:
  - "[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]"
  - "[[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]]"
---

# Brainstorm — CUDA grid limits in PyTorch

**Angle:** A high-level PyTorch call (`F.scaled_dot_product_attention`) can fail
for a *hardware* reason that has nothing to do with your tensors being too big —
the CUDA launch grid has a per-dimension cap of 65,535, and Flash/mem-efficient
SDPA maps `batch · n_heads` onto it. Knowing one CUDA fact turns a cryptic
`cudaErrorInvalidConfiguration` into a two-line fix.

**Audience:** PyTorch practitioners comfortable with `nn.Module` and attention,
who treat CUDA as a black box. Assumed background: know what attention/heads are,
have seen a CUDA OOM. *Not* assumed: know what a kernel launch grid is — explain it.

**Hook:** "I thought I was hitting memory limits when increasing the batch size
crashed the program. I was way within my VRAM limit." — the surprise that a
*bigger batch* can crash training for a reason that has nothing to do with memory.

**Storyline (confirmed 2026-06-19):** Increasing batch size can crash your run
even when you're well within VRAM. The instinct is "OOM, shrink the batch" — and
shrinking *works*, which hides the real cause: the CUDA launch grid overflowed,
not memory. The fix is bounding the *per-launch* batch (chunk + loop), not using
a smaller batch overall.

**Length/voice (confirmed):** ~1,000–1,400 words, first-person debugging story,
concrete and a little wry. Minimal code (real before/after only). Teaches the
grid model via Fig 1; shows the overflow design space via Fig 2.

**Title (confirmed):** "When increasing batch size crashes your training, but
memory isn't the issue."

## Key beats
- The bug: temporal self-attention batches `(b·h·w)`; at large batch·resolution
  the SDPA kernel launch fails — `cudaErrorInvalidConfiguration`, not OOM.
  — [[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]]
- The CUDA fact: a kernel launches over a grid of blocks; each grid dimension is
  capped at 65,535 (gridDim.y/z). Flash/mem-efficient SDPA parallelises over
  `(batch, heads)` → that product lands on the capped dimension.
- Why the error is confusing: it surfaces from a `torch` call, says nothing about
  grids, and looks like it *should* be a memory problem (it isn't).
- The fix: chunk the batch axis so `chunk · n_heads < 65,535`; loop + concat.
  Mirrors a loop DynamiCrafter already had in the cross-attention path.
  — [[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]
- Why it's free: batch axis is embarrassingly parallel → identical numerics,
  Flash preserved, memory/wall-clock unaffected.
- The takeaway: the PyTorch ↔ CUDA boundary is leaky; a little knowledge of how
  kernels are dispatched pays off exactly at the scaling frontier.

**Depth decision (2026-06-17):** *Teach the grid model properly.* Middle of the
post is a real mini-lesson — threads → blocks → grid, `gridDim.x/y/z`, the
65,535 cap on y/z — then show how SDPA maps `(batch, heads)` onto it. Educational
for someone who has never written a kernel, not just a war story.

## Discovery story (real play-by-play — the spine of §1–3)
1. **The symptom looked like an OOM.** The crash only showed up at *large batch
   sizes* — the classic "bigger batch → crash" pattern that everyone reads as
   out-of-memory. But the error wasn't `torch.OutOfMemoryError`; it said
   **`cudaErrorInvalidConfiguration`**.
2. **Chased it as an OOM first — not pure waste.** Spent time playing with the
   batch size. The twist: batch size *was* the culprit — just not through
   memory. The instinct fingered the right variable for the wrong reason.
3. **Looked up the grid limit, then wrote the comment.** Searching what
   `cudaErrorInvalidConfiguration` means surfaced the 65,535 cap on a launch
   grid dimension; realising SDPA puts `batch·n_heads` on that axis closed the
   loop. (DynamiCrafter's cross-attention path already had a `for j in range(b)`
   loop with a "can't exceed 65,535" comment — confirmation, not the tip-off.)

## What I learned
**"Crashes at large batch" ≠ "out of memory."** The same knob (batch) can kill a
run two completely different ways: by exhausting VRAM, or by overflowing the
*launch grid*. The OOM reflex (shrink the batch) accidentally works here, which
is exactly what makes it a trap — you "fix" it without understanding it, and the
real fix is *bounding the per-launch batch* (chunk + loop), not shrinking the
batch you actually process. One CUDA fact (the 65,535 grid cap) is the difference
between a superstition and a fix.

## Verified technical facts (sourced — for accuracy of §3)
- **The temporal self-attention path does go through SDPA in the configs we
  run.** `CrossAttention.forward` uses SDPA iff `(not relative_position) and
  (not mask)` (`attention.py:124`). All base configs set
  `use_relative_position: false` *and* `use_causal_attention: False`
  (`configs/base/dynamicrafter_512.yaml:63-64`, `act_cond_diffusion_34M.yaml:79-80`)
  → no relative-pos bias, no causal mask → `use_sdpa = True`. So the overflowing
  kernel really is the Flash / mem-efficient SDPA kernel. ✅ confirms the angle.
- **Nuance worth a footnote.** Stock DynamiCrafter ships `use_relative_position:
  true`; with that on, the temporal blocks fall to the manual `einsum` softmax
  instead, and the *same* 65,535 cap is then hit by the **batched matmul
  (cuBLAS `bmm`)** over leading dim `bhw·n_heads` — which is exactly what the
  pre-existing cross-attention comment ("number in shape could not [be] greater
  than 65,535 for some package") was about. So the grid cap is a property of the
  *launch*, not of SDPA specifically; SDPA is just the backend our config hits.
- **Grid dim = `(b·h·w)·n_heads`**, sequence length there is `N = t` (temporal
  length), `n_heads = model_channels / num_head_channels`, `num_head_channels:
  64` → e.g. top level of the 512 model has `n_heads = 320/64 = 5`,
  `chunk = 65535//5 = 13107`. ✅ a real worked example, not invented.

## Scope decision (2026-06-17, revised): SDPA-only, NO flag explainer in the post
Keep the narrative strictly on the SDPA kernel. Do **not** add a teaching section
on `relative_position` / `causal_attention` (the full explanation was given to
the author directly, out of band — it is not blog material). The post may state
in *one line* the precondition — "our configs disable relative position and
causal attention, so the temporal path takes the SDPA branch" — and move on.
No bmm detour either.

## Figures (LOCKED 2026-06-19 — two ASCII-inline figures, real n_heads=5 numbers)

**Fig 1 — CUDA grid layout** (§2/§3): grid of thread blocks; `gridDim.x` ≈
uncapped (≤ 2,147,483,647), `gridDim.y` capped at 65,535. Annotate SDPA's
mapping: x ← query/sequence tiles, y ← `batch · n_heads` (the capped axis).

```
 A CUDA kernel launches a GRID of thread BLOCKS (each block = a tile of threads):

                    gridDim.x   ──────────────►   (≤ 2,147,483,647, basically free)
                 ┌───────┬───────┬───────┬───────┬─ ··· ─┐
       gridDim.y │ block │ block │ block │ block │       │
      (≤ 65,535) ├───────┼───────┼───────┼───────┼─ ··· ─┤
            │    │ block │ block │ block │ block │       │
            ▼    ├───────┼───────┼───────┼───────┼─ ··· ─┤
                 │ block │ block │ block │ block │       │
                 └───────┴───────┴───────┴───────┴─ ··· ─┘

 SDPA maps attention onto the grid like this:
      gridDim.x  ←  query / sequence tiles      ← large, but x is basically uncapped
      gridDim.y  ←  batch · n_heads             ← CAPPED at 65,535  ⚠
```

**Fig 2 — overflow design space** (§3): hyperbola `batch · n_heads = 65,535`
(curve, NOT a straight line — it's a constant product). x = batch (b·h·w),
y = n_heads. Region above/right = overflow. Concrete ticks, real n_heads=5
example (legal up to ~13k batch). Teaching points: product matters (grow
batch OR heads), and the fix = slice batch to stay left of the curve.

```
 n_heads
    ▲
 20 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │▓▓▓▓▓▓▓▓▓  OVERFLOW  ▓▓▓▓▓▓▓▓▓     batch · n_heads > 65,535
 15 ┤▓▓▓▓▓▓╲▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     →  cudaErrorInvalidConfiguration
    │▓▓▓▓▓  ╲╲                            (the kernel never launches)
 10 ┤▓▓▓▓    ╲╲╲___
    │▓▓          ╲╲╲╲____   ◄── red line:  batch · n_heads = 65,535
  5 ┤░░░░░░░░░░░░░░░░░╲╲╲╲╲________________
    │░░░░░░  LEGAL (fits the grid)  ░░░░╲╲╲╲╲╲╲__________
  1 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░╲╲╲╲╲╲╲╲╲
    └┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──►
    256    1k    3.3k    8k    13k          32k       65k
                                                batch  (b·h·w)
```

## Outline
1. The symptom — a launch failure that isn't an OOM (the "looks like OOM" trap)
2. What a CUDA launch grid is (the one concept you need: threads → blocks → grid,
   `gridDim.x/y/z`, the 65,535 cap on y/z)
3. How SDPA maps attention onto the grid → where the 65,535 sneaks in
   (`(b·h·w)·n_heads` on the capped axis; worked example, n_heads=5 → chunk=13107)
4. The fix — chunk the batch, loop, concat (before/after, mirrors the existing
   cross-attention `for j in range(b)` loop). One-line precondition note that our
   configs disable relative position + causal attention so SDPA is the active path.
5. Why it costs nothing (parallel batch axis → identical numerics, Flash
   preserved, no extra memory)
6. Takeaway — "crashes at large batch" ≠ OOM; the leaky PyTorch/CUDA boundary
