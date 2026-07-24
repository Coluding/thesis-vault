---
type: tech-note
status: living
last_updated: 2026-06-17
sources:
  - "code: src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py"
  - "code: configs/base/dynamicrafter_512.yaml"
  - "[[flash-attention-sdpa-bf16]]"
  - "[[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]]"
relevance: D1  # framework / infrastructure (frozen backbone attention)
---

# Why SDPA can't do relative-position attention (but the manual path can), and how we *could* make it causal-aware

> The attention in `CrossAttention.forward` branches on one line:
>
> ```python
> use_sdpa = (not self.relative_position) and (not exists(mask))
> ```
>
> When `use_sdpa`, we call `F.scaled_dot_product_attention(q, k, v)` (the fast
> Flash / mem-efficient kernel). Otherwise we drop to a hand-written `einsum`
> softmax. This note explains **why** each of the two flags forces the manual
> path — and the key result is that the two flags are *not* the same kind of
> constraint: one is a genuine expressiveness limit, the other is just an
> implementation choice we could lift.

## The principle: SDPA is a fixed-function kernel

`F.scaled_dot_product_attention` computes exactly one formula and nothing else:

```
SDPA(Q, K, V) = softmax( Q·Kᵀ / √d  +  attn_mask ) · V
```

with two optional knobs — an additive/boolean `attn_mask`, and `is_causal` (a
built-in triangular mask). That rigidity is *why* it's fast: the Flash kernel
fuses this exact op sequence and **never materializes the `(N, N)` score
matrix** (that's the whole memory win — see [[flash-attention-sdpa-bf16]]).

The consequence that matters here: **anything that needs the score matrix, or
needs to add a term the formula has no slot for, cannot use SDPA.** You fall back
to the manual path, which builds `sim = QKᵀ` explicitly and can poke at it.

## Flag 1 — `relative_position`: a *genuine* limit (and a subtle one)

Relative positional encoding (Shaw et al., 2018) makes attention depend on the
**distance** `i − j` between a query position `i` and a key position `j`, rather
than absolute indices. The repo implements it with two learned tables,
`relative_position_k` and `relative_position_v`, and adds **two** terms in the
manual path. The subtlety: they enter at *different* points.

```python
# attention.py — manual path
sim = einsum("b i d, b j d -> b i j", q3, k3) * scale     # the QKᵀ scores

if self.relative_position:
    k2  = self.relative_position_k(len_q, len_k)           # (i, j, d) distance table
    sim2 = einsum("b t d, t s d -> b t s", q3, k2) * scale # PRE-softmax: a score bias
    sim += sim2                                            #   sim = QKᵀ + relbias

sim = sim.softmax(dim=-1)                                  # the attention weights

out3 = einsum("b i j, b j d -> b i d", sim, v3)            # softmax · V
if self.relative_position:
    v2  = self.relative_position_v(len_q, len_v)           # (i, j, d) distance table
    out2 = einsum("b t s, t s d -> b t d", sim, v2)        # POST-softmax: uses the WEIGHTS
    out3 = out3 + out2                                     #   out = softmax·V + relvalue
```

**Term A — `sim2`, the key-side bias (pre-softmax).** This is *expressible* in
SDPA. It's a tensor added to the scores before the softmax — exactly what
`attn_mask` does (`attn_mask` is added to `QKᵀ/√d` inside the kernel). `sim2`
depends only on `q` and the learned table, both available *before* the SDPA call,
so you could precompute it and pass it as an additive float `attn_mask`:

```python
sim2 = einsum("b h t d, t s d -> b h t s", q, k2) * scale   # precompute outside SDPA
out  = F.scaled_dot_product_attention(q, k, v, attn_mask=sim2)  # bias folded in
```

(You give up some of the Flash memory win — a `(b,h,N,N)` bias defeats the
never-materialize-`(N,N)` property — but it is at least *possible*.)

**Term B — `out2`, the value-side term (post-softmax).** This is the one that is
**fundamentally impossible** in SDPA. Look at what it needs:
`out2 = einsum("b t s, t s d -> b t d", sim, v2)` — it multiplies the
**post-softmax attention weights `sim`** by a learned value table. But SDPA's
entire reason for existing is that it **never gives you `sim`**: the Flash kernel
computes `softmax(...)·V` in a fused pass and discards the weight matrix. There
is no argument, no output, no hook that hands you the attention weights. No
weights → you cannot form `out2`. Full stop.

> **This is the clean lesson.** Relative position isn't "hard to fit into SDPA" —
> *half* of it (the score bias) fits fine via `attn_mask`. The other half (the
> value-side term) is excluded by the very optimization that makes Flash fast:
> the score matrix is never materialized, so anything downstream of it that needs
> the weights is off the table. Fixed-function speed and post-softmax
> expressiveness are mutually exclusive here.

So `(not self.relative_position)` in the gate is a *correctness* requirement, not
laziness: with relative position on, SDPA literally cannot produce the same
output.

### Tiny worked example (why `out2` needs the weights)

Take `t = 2`, ignore heads/`d` for intuition. After softmax, suppose the weights
for query 0 are `sim[0] = [0.7, 0.3]` (0.7 on itself, 0.3 on the other frame).
The relative-value table `v2` has an entry per distance: `v2[Δ=0]`, `v2[Δ=+1]`.
Then `out2[0] = 0.7·v2[Δ=0] + 0.3·v2[Δ=+1]`. The coefficients `0.7, 0.3` are the
softmax weights — produced *inside* the kernel and never returned. Without them
you can't even start the sum. (The key-side `sim2`, by contrast, is computed from
`q` before any softmax, so it doesn't have this problem.)

## Flag 2 — `causal_attention`: just an implementation choice

Causal masking (frame `t` may not attend to `t+1`) is enforced with a
lower-triangular mask. The repo builds it once
(`self.mask = torch.tril(torch.ones([1, T, T]))`), slices it to length `t`, and
the manual path applies it the textbook way:

```python
if exists(mask):
    mask = repeat(mask, "b i j -> (b h) i j", h=h)
    sim.masked_fill_(~(mask > 0.5), -inf)   # future positions → -inf before softmax
```

Unlike relative position, **SDPA supports this natively.** The gate excludes it
(`and (not exists(mask))`) only because the SDPA branch was wired for the common
no-mask case and the mask was never plumbed through — not because of any
hardware/kernel limit. Masking is a *pre-softmax* score edit, exactly SDPA's
`attn_mask` / `is_causal` territory.

### How to make the SDPA branch causal-aware (two ways)

**Option A — `is_causal=True` (simplest, fastest).** For self-attention where
query and key lengths match (our temporal self-attention does: both are `t`),
SDPA's built-in causal flag is a drop-in. No mask tensor to build, and the Flash
kernel applies the triangular mask internally without materializing it:

```python
use_sdpa = not self.relative_position          # masks no longer disqualify SDPA
if use_sdpa:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal_attention)
```

Caveat: `is_causal=True` assumes the standard lower-triangular layout aligned to
the last two dims. Fine for equal-length self-attention; do **not** use it for
cross-attention or unequal q/k lengths.

**Option B — pass the existing mask as `attn_mask` (general).** Handles arbitrary
masks, not just the triangular one. SDPA accepts a boolean mask where
`True = participate` (and fills the rest with `-inf` internally). The repo's mask
is `1 = keep`, so `(mask > 0.5)` is already the right boolean — just give it a
head axis to broadcast over `(b, h, t, t)` instead of pre-repeating to `(b*h,…)`:

```python
attn_mask = None
if exists(mask):
    attn_mask = (mask > 0.5).unsqueeze(1)      # (b, 1, t, t) — broadcasts over heads
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

(For a *soft*/additive mask use a float tensor of `0` / `-inf` instead of bool;
SDPA adds it to the scores.)

Either option keeps the Flash kernel and, as a bonus, would have kept the
grid-overflow fix relevant for causal runs too (the
`[[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]]` chunking
is orthogonal — it bounds the launched batch regardless of mask).

We have **not** made this change — our base configs disable causal attention
(`use_causal_attention: False`), so there's no live need. Documented here so the
"masks force the manual path" line in the gate is understood as a choice, not a
constraint.

## Summary table

| Flag | What it adds | In SDPA? | Why |
|---|---|---|---|
| `relative_position` — key side (`sim2`) | bias added to scores **before** softmax | ⚠️ possible | it's a pre-softmax score edit → fold into additive `attn_mask` (loses some Flash memory win) |
| `relative_position` — value side (`out2`) | term added **after** softmax, weighted by the attention matrix | ❌ impossible | needs the post-softmax weights `sim`, which Flash never materializes or returns |
| `causal_attention` (mask) | future scores → `-inf` **before** softmax | ✅ supported | pre-softmax mask = SDPA's `is_causal` / `attn_mask`; we just don't plumb it through (yet) |

## Related

- [[flash-attention-sdpa-bf16]] — how/why SDPA + bf16 enables the Flash kernel here, and why it never builds `(N,N)`
- [[../../20_Tickets/done/bug-backbone-temporal-attn-sdpa-grid-overflow]] — the 65,535 launch-grid overflow on this same SDPA temporal path
- code: `src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py:124-151`
