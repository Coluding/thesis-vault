---
type: tech-note
status: living
last_updated: 2026-05-25
sources:
  - "code: src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py"
  - "code: src/generative_flow_adapters/training/trainer.py"
  - "code: src/generative_flow_adapters/models/base/dynamicrafter.py"
  - "code: src/generative_flow_adapters/models/base/factory.py"
  - "config: configs/diffusion_hyperalign_metaworld.yaml"
  - "config: configs/diffusion_avid_shortcut_metaworld.yaml"
  - "config: configs/diffusion_hyperalign_shortcut_metaworld.yaml"
commit: cca6a88
relevance: D1  # framework / infrastructure (memory + throughput of the frozen backbone)
---

# Flash Attention via SDPA + bf16 autocast on the DynamiCrafter backbone

> Documents how attention in the frozen DynamiCrafter UNet was moved from a
> hand-rolled `einsum` softmax (which materializes the full `(N, N)` score
> matrix) onto PyTorch's `F.scaled_dot_product_attention` (SDPA), and how
> mixed-precision **bf16 autocast** in the trainer is what actually makes the
> Flash kernel fire. Two separate levers — the SDPA *layout* and the runtime
> *dtype* — both have to be right, which is the subtle part.

## TL;DR

- `CrossAttention.forward` in `attention.py` now reshapes Q/K/V to **4D
  `(B, H, N, D)`** and calls `F.scaled_dot_product_attention`. The 4D layout is
  mandatory: SDPA's dispatcher only routes to the Flash / mem-efficient
  backends for rank-4 inputs; anything else silently falls back to the math
  backend, which builds the full `(N, N)` matrix and costs multiple GB at high
  spatial resolution.
- The relative-position branch and the explicit causal-mask branch keep the
  **original manual-softmax path** (reshaped back to `(B*H, N, D)`), because
  SDPA can't express the relative-position bias. These paths are temporal
  (small `N = t`), so not materializing `(N, N)` doesn't matter there.
- **Flash only fires under fp16/bf16.** In fp32, SDPA drops to the
  *mem-efficient* backend: you keep the memory win (no `(N, N)`), but lose the
  fast Flash kernel. To get the speed kernel we run the model forward under
  **`torch.autocast("cuda", dtype=bf16)`** in `Trainer.training_step`.
- We deliberately did **not** use the config `model.extra.dtype` knob to
  downcast the backbone for training (see "Why not the dtype downcast" below).

## The two levers

1. **Layout (SDPA backend selection).** Lives in `attention.py`. Even in fp32
   the 4D SDPA path avoids the `(N, N)` blow-up by using the mem-efficient
   backend. This alone is the big VRAM saving.
2. **Dtype (which kernel + how fast).** Lives in the trainer via autocast.
   Verified on an RTX 3090 (Ampere, sm_86): with bf16 inputs at the 4D layout,
   `SDPBackend.FLASH_ATTENTION` is available; under fp32 it is not selected.

## bf16 autocast in the trainer

`Trainer` gained an opt-in knob `training.extra.amp_dtype` (`bf16` / `fp16` /
`none`, default off). When set, `training_step` wraps **only the model
forward** in `torch.autocast`:

```python
with self._autocast():
    prediction = self.model(x_t, t, cond)
prediction = prediction.float()   # upcast before loss/backward
loss = self.loss_fn(prediction, target_tensor)
```

Key design points:

- **Autocast operates at the op level, not on stored weights.** The
  DynamiCrafter wrapper casts its inputs to its *storage* dtype (fp32 if the
  `dtype` knob is unset), but autocast still runs the inner matmuls/conv/SDPA
  in bf16 regardless — so Flash fires without downcasting any parameters.
  Verified: an fp32 `nn.Linear` under autocast emits bf16 activations that feed
  bf16 (Flash-eligible) tensors into SDPA.
- **Upcast `prediction.float()` before the loss.** The autocast forward returns
  bf16; computing the loss/backward against an fp32 target without upcasting
  raises `RuntimeError: Found dtype Float but expected BFloat16` in the backward
  pass. Upcasting keeps the autograd dtypes consistent and the diffusion loss in
  full precision. It is a no-op when amp is off.
- **No GradScaler.** bf16 has fp32's exponent range, so loss scaling isn't
  needed. (fp16 *would* want a scaler; the current forward-only wrap doesn't
  wire one in, so fp16 is "use at your own risk" — bf16 is the supported path.)
- Backward + `optimizer.step()` run outside autocast, in fp32. Weights and the
  optimizer state stay fp32 throughout.

Enabled in `training.extra` of the three frozen-DynamiCrafter MetaWorld configs
(`diffusion_hyperalign_metaworld`, `diffusion_avid_shortcut_metaworld`,
`diffusion_hyperalign_shortcut_metaworld`) via `amp_dtype: bf16`.

## Mental model: what autocast actually does (and why the upcast)

This trips people up, so spelled out explicitly.

**Autocast is an op-level interceptor, not a "cast everything" switch.** While
active it routes each op to one of two lists:

- **bf16 list** — heavy, numerically-safe ops: `matmul`, `linear`, `conv`,
  `scaled_dot_product_attention`. Run in bf16.
- **fp32 list** — sensitive ops: softmax, layernorm, reductions, loss
  functions. Kept in fp32.

**Weights are never modified.** They stay fp32. When a `Linear` runs under
autocast, autocast casts the fp32 weight *and* the fp32 input to bf16 on the
fly, does the bf16 matmul, and emits a **bf16 activation**. So:

- **Intermediate activations *are* bf16** (the ones from bf16-list ops). This is
  the win: bf16 matmuls hit tensor cores, SDPA picks the Flash kernel (bf16/fp16
  only), and the activations saved for backward are ~half the memory.
- Outputs of fp32-list ops (e.g. layernorm) stay fp32 — so not *every*
  activation is bf16, just the heavy ones.

Demonstrated (RTX 3090): fp32 `Linear` under autocast → `weight.dtype=float32`,
post-`Linear` activation `bfloat16`, post-`layer_norm` activation `float32`.

**"Gradients in full precision" is a half-truth — two different things:**

- *Activation* gradients (grad w.r.t. intermediate tensors) flow in **bf16**
  inside the autocast region — the backward of a bf16 matmul is bf16.
- *Parameter* gradients (`weight.grad`) are accumulated in **fp32**: PyTorch
  casts the param grad up to fp32 when writing `.grad`. So the fp32 master
  weights + fp32 optimizer state — the thing the optimizer actually steps — stay
  full precision. "Heavy compute is bf16, but what the optimizer updates is fp32."

**Why `prediction.float()` — it is the *seam*, not a precision rollback.** We
wrap only the forward in autocast, not the loss. So `loss_fn(prediction,
target)` mixes a bf16 `prediction` with an fp32 `target` in a *non-autocast*
region:

- Forward math promotes to fp32 fine, but on backward the grad arriving at
  `prediction` is fp32 while the op that *produced* it (a bf16 autocast op)
  expects a bf16 grad → `RuntimeError: Found dtype Float but expected BFloat16`.
- `prediction.float()` inserts an explicit cast node: forward bf16→fp32 (clean
  fp32 loss), backward casts the fp32 grad back to bf16 before the model's last
  op — dtypes reconcile.

It does **not** throw away the benefit: by the time we upcast, every expensive
op (all matmuls, convs, the Flash SDPA) has *already executed in bf16* inside the
forward. We only cast the single final output tensor. Autocast's payoff is the
compute precision of the big ops, not the dtype of the final loss scalar.

**Idiomatic alternative.** The canonical AMP recipe puts the *loss inside* the
autocast block; autocast then keeps the loss fp32 automatically (loss fns are on
the fp32 list) and handles the seam internally, so no manual `.float()` is
needed:

```python
with autocast("cuda", bf16):
    prediction = self.model(x_t, t, cond)
    loss = self.loss_fn(prediction, target)   # auto-fp32 inside autocast
loss.backward()                                # outside
```

We use the forward-only wrap + explicit upcast instead because our loss spans
many lines (diffusion target + shortcut/consistency terms); the explicit upcast
keeps the bf16 region tight and self-documents "loss in fp32". Both are correct.

## Why not the `model.extra.dtype` downcast (for training)

`factory.py` has a `dtype` knob that downcasts the whole frozen UNet (and VAE)
to bf16/fp16. That is fine for **base-only inference**, but it breaks
**adapter training**:

- `AdaptedModel.forward` (`adapted_model.py:68-72`) feeds the base's output —
  and, for HyperAlign, captured intermediate hidden states — *into the adapter*.
- The adapter is trained and stays fp32; it is not downcast by the factory's
  `dtype` (which only touches the base wrapper). So a bf16 base output / hidden
  state hitting fp32 adapter weights throws a dtype-mismatch `RuntimeError`.

Autocast sidesteps this because it casts activations consistently across *both*
the frozen base and the fp32 adapter, while leaving all stored weights in fp32.

> Open question / future work: if we ever want pure bf16 *storage* (half the
> VRAM for the frozen 1.4B UNet, not just faster matmuls), we'd need to downcast
> the base **and** insert casts at the composition boundary in `AdaptedModel`
> (or downcast the adapter too). Not done yet — autocast was the lower-risk win.

## Verification (RTX 3090, torch 2.11.0+cu130)

- SDPA path vs. manual-softmax path: `max|Δ| = 6.7e-08` in fp32 — functionally
  identical.
- Backend probe at the 4D layout, fp16: `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`,
  and `MATH` all available; SDPA auto-selects Flash.
- `training_step` smoke test (synthetic flow model on CUDA): `amp=None` →
  fp32 forward, loss finite; `amp=bf16` → bf16 forward, backward succeeds, loss
  matches fp32 to 5 decimals (1.01112 vs 1.01114).

## xformers routing was removed (important)

`CrossAttention.__init__` used to swap spatial attention onto
`self.forward = self.efficient_forward` (the xformers
`memory_efficient_attention` path) whenever `xformers` was importable. **That
routing is now removed** — spatial attention always uses the SDPA `forward`.

Why it had to go: installing `xformers` flipped `XFORMERS_IS_AVAILBLE` to True
and re-activated `efficient_forward`, which then crashed in training with
`NotImplementedError: No operator found for memory_efficient_attention_forward`.
Two compounding causes:

1. The installed xformers wheel was built for a different torch
   (`2.10.0+cu128` vs the env's `2.11.0+cu130`) → effectively a CPU-only build,
   `"xFormers wasn't built with CUDA support"`, no CUDA operator.
2. Even with a correct CUDA build, **autocast does not cast into
   `xformers.ops.memory_efficient_attention`** (it's a custom op outside
   autocast's cast list), so its inputs stayed fp32 — which the flash/fa2 ops
   reject (`supported: {bfloat16, float16}`).

The SDPA `forward` supersedes xformers in every way that matters here: it
dispatches to Flash/mem-efficient itself, *does* respond to autocast (bf16 →
Flash), and also supports masks + relative position. `efficient_forward` and the
`xformers` import are left in the file only as dead reference code.

## Other gotchas / notes
- `flash-attn` (the standalone Dao-AILab lib) is not needed — PyTorch's SDPA
  ships its own Flash kernel.
- head_dim must be a multiple of 8 for the Flash backend; the DynamiCrafter
  configs satisfy this (the probe used `dim_head=40`).
- Disable per run by setting `amp_dtype: fp32` (or `none`) in `training.extra`.

## Related

- [[concat-condition]] — `cond["concat"]` channel pack that changes the UNet
  input the attention sees.
- [[dynamic-rescale]] — the other trainer-side numerical knob touching
  `trainer.py`.
