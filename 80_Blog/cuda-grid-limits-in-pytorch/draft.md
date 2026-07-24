---
title: "When increasing batch size crashes your training, but memory isn't the issue"
slug: cuda-grid-limits-in-pytorch
type: blog-draft
status: drafting
created: 2026-06-17
last_updated: 2026-06-19
deliverable: exploratory
sources:
  - "[[../../30_Knowledge/tech/flash-attention-sdpa-bf16]]"
  - "[[../../30_Knowledge/tech/sdpa-vs-manual-attention-expressiveness]]"
  - "[[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]]"
published_url:
---

# When increasing batch size crashes your training, but memory isn't the issue

The model was finally training. After a week of plumbing a video diffusion model
into shape, the loss was going down and the only thing left to do was the most
satisfying part of the whole job: make it go faster. The obvious lever was the
batch size. The GPU was sitting half empty, so I nudged the batch up to use the
room I was paying for, hit run, and went to get coffee.

It had crashed before the coffee was ready.

Fine, I thought. I got greedy, the bigger batch did not fit, classic out of
memory. So I did the thing the reflex tells you to do: I dropped the batch back
down and launched again. It trained. Reflex confirmed, lesson learned, move on.

Except it nagged at me. When the crash hit, the card was not full. Not close.
`nvidia-smi` had shown gigabytes free the whole time. An out of memory error does
not leave gigabytes on the table. And when I went back and actually read the
traceback instead of pattern matching it to "the batch crashed," it was not the
`torch.OutOfMemoryError` I had met a hundred times. It said something I had been
skimming straight past:

```
RuntimeError: CUDA error: cudaErrorInvalidConfiguration
```

That is not the GPU running out of room. That is the GPU refusing to start the
work at all. The batch size really was the trigger, but memory had nothing to do
with it. This is the story of how one small fact about the way CUDA launches
kernels turned a crash that made no sense into a two line fix.

## The instinct that hides the bug

Here is the trap, and it is a good one, because it defends itself.

"Bigger batch crashes the run" reads as "out of memory" to every practitioner,
because out of memory is the failure mode we meet first and meet most. And the
reflex fix, shrinking the batch, makes the crash disappear, which feels like a
confession. Cause identified, case closed.

But a smaller batch curing the crash does not prove the cause was memory. It only
proves the batch size was involved. The batch can crash a run through a
completely different door, and if you walk away the moment the symptom clears,
you never find out which door it was. The only clue that you went through the
wrong one is sitting in the exact text of the error. `cudaErrorInvalidConfiguration`
does not mean "I ran out of room." It means "the way you asked me to launch this
kernel is not a legal shape." To understand why a large batch can make a launch
illegal, you need one idea about how the GPU actually runs your code.

## The one concept you need: the launch grid

When PyTorch runs an operation on the GPU, it launches a CUDA kernel, and a
kernel does not run as one monolithic lump of work. It runs as a grid of thread
blocks. A block is a small group of threads that run together on a single one of
the GPU's streaming multiprocessors, sharing fast memory and scheduled in bundles
of 32 threads called warps. The grid is the collection of all those blocks for
one launch.

When the kernel is launched it declares the shape of that grid: how many blocks
along each of three axes, named `gridDim.x`, `gridDim.y`, and `gridDim.z`. You
can picture it as a three dimensional box of blocks, and the kernel is asking the
driver to please schedule a box of exactly this size.

The catch is that the three axes are not equal citizens. The hardware caps each
one, and the caps are lopsided in a way that is easy to never notice:

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
```

The `x` axis can hold over two billion blocks. The `y` and `z` axes top out at
65,535, which is exactly `2^16` minus one. That number is not a coincidence. Those two
dimensions were 16 bit quantities on the earliest CUDA devices and have stayed
that way. The `x` dimension was later widened to a full 32 bit range, so it grew
to billions, but `y` and `z` were left where they always were. The asymmetry is a
fossil, and most of the time it is harmless, because most kernels put their big
dimension on `x` and never come anywhere near 65,535 on the other two.

But if a kernel does ask for a grid whose `y` or `z` dimension is 65,536 or more,
the driver rejects the launch on the spot. The kernel never runs. No memory is
allocated, nothing overflows in the way we usually mean. You simply asked for a
shape the hardware cannot describe, and it told you so:
`cudaErrorInvalidConfiguration`. That is the whole error. It is a shape complaint,
not a space complaint.

## Where attention puts your batch

Now line that up with attention. The layer that was crashing was a temporal
attention block in the video model, and the attention there runs through
PyTorch's `scaled_dot_product_attention`, the fused path that dispatches to the
Flash and memory efficient kernels.

Those kernels get their speed from parallelizing across the independent pieces of
the attention problem, and the most independent pieces are the batch entries and
the attention heads. Every `(batch entry, head)` pair is its own self contained
little attention computation: its own queries, keys, and values, with no
interaction across pairs. So the kernel hands each pair its own slice of the grid
and lets them all run at once. There are `batch · n_heads` such pairs, and that
product lands on one of the grid axes. On one of the capped ones:

```
 SDPA maps attention onto the grid like this:
      gridDim.x  ←  query / sequence tiles      ← large, but x is basically uncapped
      gridDim.y  ←  batch · n_heads             ← CAPPED at 65,535  ⚠
```

And here is the part that actually bit me. In a temporal attention layer over
video, the "batch" the kernel sees is not your clip batch. Temporal attention
asks "how does each spatial location evolve over time," so the layer reshapes the
video to make every spatial location its own independent sequence over the time
axis. The number of those sequences is your clip batch times the spatial
resolution of the feature map. The effective batch is `b · h · w`. So the
quantity that has to stay under 65,535 is not `b · n_heads`, it is:

```
batch · n_heads  =  (b · h · w) · n_heads
```

Spatial resolution is large, and it multiplies into the batch before the heads
even enter the picture. That is the whole reason this masquerades as a memory
bug: you raise `b`, the product `(b · h · w) · n_heads` slides past 65,535, and
the launch dies, with the increase in `b` looking for all the world like the
increase that filled up your VRAM.

A concrete example from my model. At the top resolution level the feature map is
40 by 64, so `h · w = 2560`, and the layer has `n_heads = 5`. That means a clip
batch of just six gives `6 · 2560 · 5 = 76,800`, which is over the limit. A batch
of five fits, at exactly 64,000. One extra clip is the entire difference between a
legal launch and a crash, which is also why fiddling with the batch size felt so
much like tiptoeing around a memory ceiling.

The picture that finally made it click for me is the design space. The boundary
is a constant product, `batch · n_heads = 65,535`, so it is a curve, not a
straight line. Everything above or to the right of it is a launch the hardware
will refuse:

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

The curve says two things at a glance. First, it is the product that matters, so
you can fall off the edge by growing the batch or by growing the head count, and
a wide model with many heads hits the wall at a smaller batch than a narrow one.
Second, the legal region has plenty of room left under it. The point was never to
use a smaller batch. The point is to stay left of the curve on every individual
launch.

(One precondition worth naming so the story is honest: this layer takes the SDPA
path only because the model runs with relative position and causal masking turned
off. With either of those on, the code routes to a different, hand written
attention path instead. That path has its own brush with the same 65,535 limit,
but the SDPA version is the one I hit, and it is the cleaner story.)

## The fix: bound the launch, not the batch

Once you see that the illegal thing is the grid dimension, and not the memory,
the fix writes itself. You do not need a smaller batch overall. You need each
individual launch to stay under the cap. So you split the batch axis into chunks
small enough that `chunk · n_heads` is always legal, run each chunk through the
attention block on its own, and concatenate the outputs back together.

Before, the layer ran the whole batch through each block in one shot:

```python
for i, block in enumerate(self.transformer_blocks):
    x = block(x, mask=mask)
```

After, it chunks the batch only when running it whole would overflow, and
otherwise behaves exactly as it did:

```python
bhw = x.shape[0]
chunk = max(1, 65535 // self.n_heads)
for i, block in enumerate(self.transformer_blocks):
    if bhw <= chunk:
        x = block(x, mask=mask)
    else:
        outs = []
        for s in range(0, bhw, chunk):
            e = s + chunk
            outs.append(block(x[s:e], mask=mask[s:e] if mask is not None else None))
        x = torch.cat(outs, dim=0)
```

The chunk size is just `65535 // n_heads`: the largest batch slice whose product
with the head count still fits under the cap. When the batch already fits, the
fast path runs untouched and there is zero overhead. When it does not, the work
is spread across a small handful of legal launches instead of one illegal one. If
you have ever read this codebase's cross attention path, this will look familiar,
because the cross attention branch already looped over the batch one entry at a
time for exactly this reason, with a comment muttering about a number that "could
not be greater than 65,535." The temporal path had simply never been pushed hard
enough to need the same treatment, until the batch went up.

## Why this costs nothing

The reason this is a fix and not a tradeoff comes back to that word independent.
Each entry along the attention batch attends only to its own tokens. Entry `i`
never looks at entry `j`. So slicing the batch into contiguous chunks and gluing
the results back together is numerically identical to running it whole, down to
the last bit. There is no approximation, no averaging, no fuzz. The mask is
sliced the same way, so each chunk keeps its own rows.

It also keeps the fast kernel. Every chunk still flows through the same Flash
attention path, so you are not trading the speed kernel for a slow fallback. You
are running the same total arithmetic as a few back to back launches instead of
one, and the peak memory is if anything a touch lower, since each launch only
ever holds a slice of the batch at a time. Wall clock and memory are, for any
practical purpose, unchanged. The only thing that changed is that every launch
now describes a grid the hardware is willing to schedule.

## How to tell this apart from a real OOM

If you take one practical habit away from this, let it be this: when a bigger
batch crashes, read the error code before you trust the reflex. A genuine out of
memory failure is a `torch.OutOfMemoryError` (or a CUDA error that says as much),
it usually names how much it tried to allocate, and `nvidia-smi` will show the
card pinned near full right before it dies. A grid overflow is
`cudaErrorInvalidConfiguration`, it mentions no allocation size because nothing
was allocated, and the memory will have headroom to spare. Same symptom at the
level of "I raised the batch and it broke," two completely different diseases,
and the cure for one is not the cure for the other.

## The takeaway

The lesson I keep returning to is that "crashes when I raise the batch size" is
not a synonym for "out of memory." The batch can kill a run two completely
different ways: by exhausting VRAM, which is about how much fits, or by
overflowing the launch grid, which is about a shape the hardware can express. And
because the reflex fix, shrinking the batch, happens to relieve both, it is easy
to misdiagnose the second as the first and never know you were wrong.

PyTorch lets you write attention without ever once thinking about thread blocks,
and almost all of the time that abstraction holds beautifully. But it is leaky at
the edges, and the edge it leaks at is precisely the scaling frontier, where you
are pushing batch and resolution as high as they will go. That is the worst
possible moment to be running on a reflex. One small fact about how kernels get
launched, the 65,535 cap on a grid dimension, was the whole distance between a
superstition that happened to work and a fix that actually understood the
problem.
