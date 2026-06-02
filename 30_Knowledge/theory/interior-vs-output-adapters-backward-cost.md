---
type: theory
last_updated: 2026-05-29
sources:
  - "[[../related-work/hyperalign]]"
  - "[[prediction-objectives]]"
  - "[[unicon-output-adapters-detached-backward]]"
---

# Interior vs. output adapters — why LoRA & HyperAlign pay a full-network backward pass

> **The deciding cost factor for an adapter family is *where* its trainable
> parameters sit relative to the frozen base model.** LoRA and HyperAlign
> inject parameters *inside* the base UNet (into attention projection
> `nn.Linear` layers), so their delta only exists after running the whole
> network with modified weights. Getting a gradient to those parameters
> therefore requires backpropagation through the *entire* base model —
> even though the base weights are frozen. Output- and (detached)
> hidden-state adapters operate on the base *output*, which can be treated
> as a constant, so their backward pass touches only the small adapter
> module. This is a first-order architectural trade-off the thesis (D1/D2)
> should state explicitly: parameter-efficiency in *parameter count* does
> not imply efficiency in *training compute or memory*.

## 1. The composition rule and where each family attaches

The framework's composition rule (see vault home):

```
f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)
```

The intent of the additive form is that `f_base(x_t, t)` can be a **fixed,
detached input** to the adapter — `Δ_φ` is supposed to be a small module
that reads `base_output` and emits a residual. That is exactly how
output-level adapters behave, and it is what makes them cheap to train.

LoRA and HyperAlign deliberately break that property. They do not read the
base output and emit a residual on top of it; they **re-parameterise the
base model itself** and re-run it.

## 2. LoRA — interior weight injection, two base passes

`adapters/low_rank/lora.py:35-38`:

```python
reference = base_output  # frozen pass (can be detached / no-grad)
with self._enabled():               # turn LoRA branches ON inside attention projections
    adapted = self.base_model(x_t, t, cond=...)
return adapted - reference
```

The LoRA factors are spliced into `nn.Linear` layers deep inside the UNet
(`inject_lora_layers` over `target_modules`). The "delta" the adapter
returns is `adapted - reference`, where `adapted` is the output of the
**full base model run with LoRA active**.

Because the only trainable signal (`A`, `B` low-rank factors) lives at an
*interior* weight, the chain rule must propagate the loss gradient
backward **from the output, through every layer downstream of each
injection point**, to reach those factors. For attention projections
scattered across the UNet, "downstream of the injection point" is
effectively the whole network.

## 3. HyperAlign — same shape, plus a hypernetwork bolted to the interior

`adapters/hypernetworks/hyperalign.py:271-285` is structurally identical,
with more machinery:

1. A reference pass on the unmodified frozen weights (`clear_dynamic_parameters`
   first, then `base_model(...)`).
2. A hypernetwork (transformer decoder + memory projections + factor head,
   `_prepare_architecture`, lines 296-321) generates **per-layer LoRA
   factors** from encoder memory tokens.
3. Those factors are written into the attention projections via
   `set_dynamic_hyper_factors` (lines 275-280).
4. The full UNet is run *again* to produce `adapted` (line 282).

The hypernetwork's output feeds an interior weight, so its gradient
**only** arrives by backprop through the UNet. You cannot train the
hypernetwork in isolation — it is coupled to the full-network graph.

## 4. Why backward is expensive *here* specifically

Four compounding reasons, particular to this codebase:

1. **The graph spans the whole base UNet, not just the adapter.** Frozen
   weights (`requires_grad=False`) save the *weight*-gradient compute, but
   autograd must still **retain the activations of every layer** between an
   injection point and the output in order to chain gradients back to the
   LoRA/hyper params. *Frozen ≠ free*: you skip weight grads but still pay
   the full activation graph.

2. **The backbone is a large video UNet.** Latents are
   `[batch, channels, frames, height, width]` and get reshaped to
   `batch*frames` for the spatial blocks (`hyperalign.py:466`). Activation
   memory scales with frames × spatial resolution × depth, so retaining the
   forward graph for backward is memory-heavy — which is precisely why the
   code depends on **gradient checkpointing**. The comment at
   `hyperalign.py:264-270` (factors are deliberately *not* cleared at the
   end of `forward`) exists so checkpoint recomputation reproduces the same
   graph as the original forward.

3. **Two base forward passes per step.** Composition is `adapted - reference`.
   The `reference` / `base_output` pass can be detached / no-grad, but the
   `adapted` pass **must** build the full autograd graph. HyperAlign adds a
   third encoder pass to build memory tokens — that one *is* wrapped in
   `torch.no_grad()` (`hyperalign.py:477`), so it is cheaper, but the main
   adapted pass is not.

4. **HyperAlign couples the hypernetwork to the UNet graph.** Its gradient
   path runs entirely through the frozen UNet (see §3 above), so there is no
   way to amortise it away.

## 5. Contrast: why output / detached hidden-state adapters are cheap

An output adapter computes `delta = adapter(x_t, t, cond, base_output)`
where `base_output` is a **detached constant**. Backward flows only through
the small adapter module — the base UNet is *never* in the gradient graph.
That is the whole appeal of the additive `prediction = base(x,t) + adapter(...)`
rule: it lets the base output be a fixed input. LoRA/HyperAlign reach
*inside* the base, forfeit that property, and pay full-network backprop on
every training step.

## 6. Thesis implication

The adapter taxonomy (D1) should not be ranked by trainable-parameter count
alone. The relevant axis for training cost is **interior vs. exterior
attachment**:

| Attachment | Examples | Trainable params | Backward cost |
|---|---|---|---|
| Exterior (on base output) | output (affine, shortcut_direction), detached hidden-state | small | adapter only — cheap |
| Interior (into base weights/hidden states) | LoRA, HyperAlign, (non-detached) UniCon-style | small–moderate | full base-model backward — expensive |

A LoRA adapter and an output adapter can have comparable parameter counts
yet differ by ~an order of magnitude in training memory/compute, purely
because of where they attach. This matters for the D2 inference-vs-training
cost analysis and for justifying which family the contribution centres on.

The companion note [[unicon-output-adapters-detached-backward]] works the
*other* side of this axis: UniCon-style and output adapters avoid the
frozen-base backward entirely via `.detach()` at the feature/output
boundary, so the real discriminator is not interior-vs-exterior but **"is
the frozen base in the gradient graph?"**.

_Code refs verified against `adapters/low_rank/lora.py` and
`adapters/hypernetworks/hyperalign.py` at commit 57244cc (working tree,
2026-05-29)._
