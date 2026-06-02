---
type: theory
last_updated: 2026-05-29
sources:
  - "[[interior-vs-output-adapters-backward-cost]]"
  - "[[../related-work/unicon]]"
  - "[[../tech/mask-mix-gate]]"
---

# UniCon & output adapters — detached base features mean no full-network backward

> **UniCon-style hidden-state adapters and output adapters both avoid the
> full-base-model backward pass that [[interior-vs-output-adapters-backward-cost|LoRA and HyperAlign pay]],
> and they do it by the same mechanism: the frozen base is run *once,
> detached*, and the adapter consumes its activations (or its final output)
> as a constant input.** Because the base contributes no edges to the
> autograd graph, gradient only flows through the adapter's *own*
> parameters. They differ only in *how large* that own network is — a tiny
> affine head, a full 3D-UNet output adapter (the
> `DynamicCrafterOutputAdapter` we actually train with), or a trainable
> decoder replica (UniCon). None of them backpropagates through the frozen
> backbone. This
> completes the backward-cost axis of the adapter taxonomy: it is not
> interior-vs-exterior alone, it is **"is the frozen base in the gradient
> graph?"** — and the answer is set by where `.detach()` sits.

See [[interior-vs-output-adapters-backward-cost]] for the LoRA/HyperAlign
side of this comparison; this note is its companion.

## 1. The mechanism: detach at the boundary

Every UniCon-family adapter attaches forward hooks to the frozen UNet that
**detach** as they capture (`adapters/hidden_states/unicon.py:384-394`,
`_UNetFeatureStore`):

```python
def capture_input(_module, _args, output):
    self.input_activations.append(output.detach())   # <- detached
def capture_middle(_module, _args, output):
    self.middle = output.detach()                    # <- detached
def capture_output(_module, _args, output):
    self.output_activations.append(output.detach())  # <- detached
```

`.detach()` severs the captured tensors from the base model's autograd
graph. When the adapter later reads them (`features = self._feature_store.require()`,
`unicon.py:73`), they enter the adapter's computation as **leaf constants**.
No gradient can flow back into the base through them.

Output adapters do the same thing one level coarser: they read the base's
*final* output as a detached constant (`adapters/output/affine.py:22`):

```python
reference = base_output if base_output is not None else x_t
```

And the detach is not even local to the adapter — `AdaptedModel` runs the
**entire base forward under `torch.no_grad()`** before handing `base_output`
to any adapter (`models/adapted_model.py:67-68`):

```python
with torch.no_grad():
    base_output = self.base_model(x_t, t, cond=cond)
...
adapter_result = self.adapter(x_t, t, adapter_cond, base_output=base_output)
```

So for *every* output and hidden-state adapter, `base_output` (and the
hook-captured features, which are additionally `.detach()`-ed) is a leaf
constant with no path back into the frozen base. The only families that
re-enter the base's graph are LoRA/HyperAlign, and they do it by running the
base model **themselves a second time, outside this `no_grad` block**, with
their interior weights live.

## 2. UniCon — a separate trainable replica, not interior surgery

The crucial structural difference from LoRA/HyperAlign: UniCon does **not**
modify the base's weights and re-run it. It builds its **own** trainable
copy of the decoder at attach time (`unicon.py:108-120`):

```python
self.decoder_blocks   = nn.ModuleList(copy.deepcopy(module.output_blocks))
self.out_head         = copy.deepcopy(module.out)
self.middle_connector = build_connector(...)   # zero-init connectors
self.skip_connectors  = nn.ModuleList(...)
self.decoder_connectors = nn.ModuleList(...)
```

Forward (`unicon.py:84-93`) runs *the replica*, fed by the detached
captured features:

```python
h = self.middle_connector(features.middle)              # detached base middle
for index, block in enumerate(self.decoder_blocks):     # trainable replica blocks
    skip = self.skip_connectors[index](features.input_skips[-(index + 1)])  # detached skips
    h = torch.cat([h, skip], dim=1)
    h = block(h, emb, context=context, batch_size=batch_size)
    h = self.decoder_connectors[index](h, features.output_activations[index])
prediction = self.out_head(h)
```

So the backward pass traverses **only the replica decoder + connectors** —
the trainable part — and stops at the detached feature boundary. The frozen
base is never in the graph. The base is executed once (frozen, no-grad) to
populate the feature store; that pass builds no graph and retains no
activations for backward.

### The sibling designs scale the replica differently

The same family spans a range of replica sizes (all detached, none touching
the frozen base's gradient):

| Adapter | Trainable replica | Backward graph size |
|---|---|---|
| `ReplaceDecoderHiddenStateAdapter` (`unicon.py:130`) | decoder blocks + out head, **no** connectors | ~decoder |
| `UniConHiddenStateAdapter` (`unicon.py:20`) | decoder replica + zero-init connectors reading captured features | ~decoder + connectors |
| `FullSkipLayerControlAdapter` (`unicon.py:224`) | **full** UNet replica (input + middle + output blocks) + connectors | ~whole UNet |

`FullSkipLayerControl` is the expensive end — its trainable replica is as
big as the base UNet, so its backward graph is comparable in size. But note
*why* it is expensive: the cost is the **adapter's own replica**, which the
designer chose, **not** a forced traversal of the frozen base. With LoRA/
HyperAlign you cannot opt out of the full-base backward; with the UniCon
family the backward size is a design knob.

## 3. Output adapters span a range — from a tiny head to a full UNet

"Output adapter" labels *where the adapter attaches* (it reads the detached
base output), **not** how big the adapter network is. The family spans the
same size range as the hidden-state family — and crucially, the heavy end is
the one we actually train with.

### 3a. `AffineOutputAdapter` — the cheap extreme

`affine.py:9-25` holds only a small context projector and a broadcast head:

```python
context = self.context(t, resolve_condition_embedding(cond))   # tiny MLP
scale, shift = self.head(context, reference)                   # tiny head
return OutputAdapterResult(adapter_output=reference * scale + shift, output_kind="delta")
```

`reference` (the detached base output) is treated as data. Backward touches
only the projector and head — a handful of layers. This is the cheapest
family on the backward axis, and the most literal realisation of the
additive composition rule's intent: *base output in, residual out.*

### 3b. `DynamicCrafterOutputAdapter` — a full 3D UNet as the adapter ← **the one in use**

`adapters/output/dynamicrafter.py:15` is the output adapter we are actually
working with. Its trainable network is a **complete AVID/DynamiCrafter 3D
UNet** (`UNetModel`), not a head (`dynamicrafter.py:62`):

```python
self.module = UNetModel(**params)   # a full video UNet, trained from scratch or from a checkpoint
```

It still belongs to the output family because it consumes the **detached
base output as input**, not as a weight injection. With
`condition_on_base_outputs=True` (the default) the base prediction is
concatenated onto `x_t` along the channel axis and fed to the adapter UNet
(`dynamicrafter.py:58-59`, `128-129`):

```python
params["in_channels"] += params["out_channels"]   # widen input to take base_output
...
adapter_input = torch.cat([x_t, base_output], dim=1)   # base_output is the no_grad constant
output = self.module(adapter_input, timesteps=t, context=context, act=act, fs=fs,
                     adapter_embedding=adapter_embedding)
return OutputAdapterResult(adapter_output=output, output_kind="prediction")
```

Consequences for the backward-cost picture:

- **Backward traverses the entire adapter UNet** — it is a large gradient
  graph, comparable in cost to training a video diffusion model. This is
  *not* a cheap adapter, despite the "output" label.
- **But the frozen base is still not in that graph.** `base_output` arrives
  detached (the `no_grad` block of §1, plus it is concatenated as plain
  data), so backprop stops at the channel-concat boundary. The cost is
  entirely the adapter's own UNet — a design choice, not a forced base
  traversal (same distinction as `FullSkipLayerControl` in §2).
- **It returns `output_kind="prediction"`**, so under `add` composition the
  framework computes `base_output + adapter_output`, and under `mask_mix`
  it gates the two (`adapted_model.py:81-94`). It also carries its own
  action / step-level conditioning (`act`, `fs`, `adapter_embedding`,
  `use_step_level_conditioning`) and an optional mask head
  (`output_mask` → `(prediction, gate)`), which is what wires it into the
  D2 action-conditioned and D3 shortcut paths. See [[../tech/mask-mix-gate]].

The takeaway: **`DynamicCrafterOutputAdapter` is expensive to train (full
UNet backward) yet shares the affine adapter's key property — zero gradient
through the frozen base.** It sits at the opposite size extreme of the same
family.

## 4. The unified picture (backward-cost axis)

Putting this together with [[interior-vs-output-adapters-backward-cost]]:

| Family | Trainable params live… | Frozen base in gradient graph? | Backward cost driver |
|---|---|---|---|
| Output — affine | a tiny head reading the detached base **output** | **No** | a handful of layers |
| Output — **DynamiCrafter** (in use) | a **full 3D UNet** reading the detached base **output** (channel-concat) | **No** | the adapter UNet (design knob) |
| Hidden-state / UniCon (replace_decoder, unicon, full_skip) | a **separate replica** reading detached base **features** | **No** | size of the replica (design knob) |
| LoRA | **interior** to frozen base weights | **Yes** | full base-model backward (forced) |
| HyperAlign | **interior** weights, generated by a hypernetwork | **Yes** | full base-model backward + hypernet (forced) |

Note how the first two rows are the *same family* at opposite size extremes:
"output adapter" is cheap (affine) or expensive (DynamiCrafter) entirely
depending on the adapter network, while both keep the frozen base out of the
graph. Backward cost and frozen-base-in-graph are genuinely independent axes.

The discriminating question is the third column. `.detach()` at the capture
boundary (UniCon) or at the `base_output` input (output adapters) keeps the
frozen base out of the graph; interior injection (LoRA/HyperAlign) forces it
in. Trainable-parameter count is orthogonal — `FullSkipLayerControl` may
have *more* trainable parameters than a LoRA adapter yet still avoid the
frozen-base backward.

## 5. Caveats worth flagging for the thesis

- **"Run once, detached" still costs a forward.** All families still execute
  the frozen base forward to produce features/output. The saving is purely
  in *backward* (no graph retention, no gradient through the base), not in
  the base's forward FLOPs.
- **Number of base passes differs.** UniCon/output run the base **once**
  (detached). LoRA/HyperAlign run it **twice** with grad on the second
  (`reference` + `adapted`), and HyperAlign a **third** time (no-grad
  encoder memory pass). This compounds the cost asymmetry beyond the
  backward-graph difference.
- **Connectors are zero-initialised** (`ZeroConvConnector`, `ZeroFTConnector`,
  `unicon.py:342-370`) so the adapter starts as an identity-ish residual —
  relevant to training stability, not to the backward-cost argument, but
  worth not conflating.

_Code refs verified against `adapters/hidden_states/unicon.py`,
`adapters/output/affine.py`, `adapters/output/dynamicrafter.py`, and
`models/adapted_model.py` at commit 57244cc (working tree, 2026-05-29)._
