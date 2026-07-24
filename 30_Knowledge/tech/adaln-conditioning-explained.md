---
type: tech-note
status: living
last_updated: 2026-06-23
sources:
  - "code: src/generative_flow_adapters/backbones/wan/modules/model.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/action_model.py"
  - "ref: Peebles & Xie 2023, Scalable Diffusion Models with Transformers (DiT)"
relevance: D1 / D2  # how the Wan DiT (and our tiny-Wan adapter) is conditioned
---

# AdaLN conditioning, explained from scratch

This note explains **Adaptive LayerNorm (AdaLN)** and its **AdaLN-Zero**
variant — the mechanism the Wan DiT uses to inject the timestep (and, in our
adapter, the action + shortcut step) into every transformer block. It's the
single most important and least obvious part of how a DiT is conditioned.

## 1. The problem it solves

A transformer block processes a sequence of tokens `x` (here: patchified video
latents). But the network also needs to know **side information** that is the
same for the whole sequence: the diffusion/flow timestep `t`, and in our case
the action `a` and shortcut step size `d`. How do you feed one `[B, dim]`
vector into a block that operates on `[B, L, dim]` tokens?

Three classic options:
- **Concatenate** the condition to every token → wastes width, clumsy.
- **Cross-attention** to the condition (how *text* enters Wan) → extra params
  and compute per block; overkill for a single global scalar-ish signal.
- **Modulate the normalization** → cheap, global, expressive. This is AdaLN.

## 2. Recap: what LayerNorm does

A normal `LayerNorm` over the feature dim does:

```
LN(x) = (x - mean(x)) / std(x) * gamma + beta
```

`gamma` (scale) and `beta` (shift) are **fixed learned parameters** — the same
for every input. The key idea of AdaLN: **make gamma and beta depend on the
condition** instead of being fixed.

## 3. The core idea of AdaLN

> Predict the LayerNorm scale & shift *from the conditioning vector*, per sample.

```
AdaLN(x, c) = LayerNorm_no_affine(x) * (1 + scale(c)) + shift(c)
```

where `scale(c)` and `shift(c)` are produced by a small linear map from the
conditioning embedding `c` (timestep + action + step). So at low noise the
network can behave one way, at high noise another — the *same weights*, re-tuned
per timestep by scale/shift. `(1 + scale)` (not just `scale`) means "start from
identity, learn a delta," which trains stably.

## 4. AdaLN-Zero: add a gate

DiT (and Wan) add a third modulation: a **gate** that multiplies the *output*
of each sublayer before it's added back to the residual stream:

```
x = x + gate(c) * sublayer( AdaLN(x, c) )
```

If `gate ≈ 0`, the block is ~a no-op (the residual passes through unchanged).
This is "**AdaLN-Zero**": gates are initialised near zero so each block starts
as identity and *learns* how much to contribute. (It's also why our tiny-Wan
adapter's **head** is zero-init: delta ≈ 0 at start → `prediction == base`.)

## 5. Why **6** modulation vectors

A Wan block has **two** sublayers that get modulated — self-attention and the
FFN — and each needs **(shift, scale, gate)**. That's `2 × 3 = 6` vectors, each
of width `dim`. They are produced by `time_projection`:

```python
# action_model.py / model.py
e  = time_embedding(sinusoid(t))            # [B, dim]   (+ cond + step in our adapter)
e0 = time_projection(e)                      # [B, 6*dim]
e0 = e0.unflatten(1, (6, dim))               # [B, 6, dim]
```

`time_projection = SiLU → Linear(dim → 6*dim)`. One conditioning vector in,
six modulation vectors out.

## 6. Walk through a Wan block (with shapes)

`WanAttentionBlock.forward` (`model.py:281-318`), `x:[B,L,dim]`, `e0:[B,6,dim]`:

```python
# Each block adds its OWN learned bias, then splits into the 6 pieces:
e = (self.modulation + e0).chunk(6, dim=1)   # 6 × [B, 1, dim]
#   self.modulation = Parameter(randn(1,6,dim)/sqrt(dim))  -> block-specific

# --- self-attention sublayer ---
y = self_attn( norm1(x) * (1 + e[1]) + e[0] )   # AdaLN: scale e[1], shift e[0]
x = x + e[2] * y                                 # gate e[2] on the residual

# --- FFN sublayer ---
y = ffn( norm2(x) * (1 + e[4]) + e[3] )          # AdaLN: scale e[4], shift e[3]
x = x + e[5] * y                                 # gate e[5]
```

| piece | name | sublayer | role |
|---|---|---|---|
| `e[0]` | shift β | self-attn | bias after norm |
| `e[1]` | scale γ | self-attn | `·(1+γ)` after norm |
| `e[2]` | **gate** | self-attn | scales the residual add |
| `e[3]` | shift β | FFN | bias after norm |
| `e[4]` | scale γ | FFN | `·(1+γ)` after norm |
| `e[5]` | **gate** | FFN | scales the residual add |

The `[B,1,dim]` shape **broadcasts over the L tokens** — every token in a sample
gets the same scale/shift/gate (the condition is global to the sequence).

## 7. Shared signal + per-block offset

`e0` is computed **once** and the *same* `e0` is handed to all blocks. But each
block adds its own `self.modulation` bias before the chunk, so block *k* sees
`modulation_k + e0`. → all blocks share the timestep/action/step signal, yet
each can specialise its response. (Cross-attention to text is separate and
*not* gated — see `model.py:313`.)

## 8. How OUR conditioning rides this path

In `ActionWanModel` the conditioning vector is a **sum** before projection:

```python
e = time_embed(t) + cond_proj(c) + step_embed(log2 d)   # [B, dim]
e0 = time_projection(e)                                   # [B, 6, dim]
```

So action (`c`, the fused encoder embedding) and shortcut step (`d`) flow through
the *exact same* AdaLN machinery as the timestep — they modulate scale/shift/gate
of every block. That's why "inject action/step" = "add to `e`": no new attention,
no new tokens, just more signal into the modulation. (Contrast: text enters via
cross-attention, a different path.)

## 9. fp32 caveat (why our code forces it)

The modulation must be **fp32** — `WanAttentionBlock` asserts
`e.dtype == torch.float32` (`model.py:299`), because the multiplicative gating
`(1+scale)` compounds bf16 rounding. So the `e`/`e0` computation runs under
`torch.autocast(enabled=False)` even though the rest of the block is bf16. See
[[wan-vendoring-patches]] and the adapter-system guide.

## 10. One-paragraph summary

AdaLN conditions a transformer by **predicting each block's LayerNorm scale &
shift (and a residual gate) from a conditioning vector**, instead of feeding the
condition in as tokens. Wan produces **6** such modulation vectors per block
(shift/scale/gate for self-attn and FFN) via `time_projection`, broadcasts them
over all tokens, and adds a per-block learned bias. The gate (AdaLN-Zero) lets a
block start as identity and learn its contribution. In our adapter, time +
action + step are summed into one vector before this projection, so all three
condition the network through the same cheap, global modulation path.

See [[wan21-model-architecture]] and [[wan21-adapter-system-guide]].
