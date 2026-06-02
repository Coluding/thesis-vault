---
type: tech-note
status: living
last_updated: 2026-05-29
sources:
  - "code: src/generative_flow_adapters/models/adapted_model.py"
  - "code: src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py"
  - "code: src/generative_flow_adapters/backbones/dynamicrafter/basics.py"
  - "code: src/generative_flow_adapters/adapters/output/dynamicrafter.py"
  - "code: src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py"
  - "config: configs/diffusion_avid_shortcut_metaworld.yaml"
  - "config: configs/diffusion_hyperalign_shortcut_metaworld.yaml"
  - "paper: docs/paper/avid.pdf §3.3"
commit: 7680e82  # plus uncommitted edits in training/trainer.py
relevance: D2 / D3 / D4  # composition is shared across deliverables
---

# Mask-mix gate — composition rule and how the two adapter families produce it

> Two different adapter families in this repo support the same outer
> composition rule —
> `prediction = base · sigmoid(gate + gate_bias) + adapter_y · (1 - sigmoid(gate + gate_bias))`.
> The composition itself is shared (one function in `AdaptedModel`),
> but **how `gate` is produced differs between AVID (output adapter on
> the DynamiCrafter UNet) and HyperAlign (hypernetwork adapter)**. This
> note catalogues both gate-production paths, their initialisation
> regimes, and what the gate "is" at step 0 in each case.
>
> See [[../../50_Decisions/open/avid-adapter-init]] for the open
> design decision about whether the current AVID-style asymmetric init
> is right for our shortcut-training setup.

## TL;DR

| Family | Where the gate is computed | Resolution at step 0 | Init regime | `y` at step 0 |
|---|---|---|---|---|
| AVID (`DynamicCrafterOutputAdapter`) | `self.out_mask` Conv head on the adapter UNet's **final shared feature map `h`** | Per-pixel, 1-channel (broadcast over `C`) | `zero_module(conv_nd(...))` — both `W=0` and `b=0` | Random (separate Conv head `self.out` is **not** zero-init) |
| HyperAlign `channel` | `_gate_head_channel` Linear on the transformer decoder's **mean-pooled tokens** | Per-channel, constant over `(T, H, W)` | `nn.init.zeros_(weight); nn.init.zeros_(bias)` | LoRA-perturbed base (= base × random small low-rank delta) |
| HyperAlign `spatial` | `_gate_head_spatial` Conv2d on the **shallowest base-UNet encoder feature** | Per-pixel + per-channel, constant over `T` | `nn.init.zeros_(weight); nn.init.zeros_(bias)` | LoRA-perturbed base |

All three produce `gate = 0` everywhere at step 0 ⇒ `sigmoid(0 + gate_bias)` is the per-batch starting mix. With the default `gate_bias = 0.0` this is `0.5` uniformly. See §"The asymmetric init story" for why the gate is zero-init but `y` is not.

## The shared composition

`src/generative_flow_adapters/models/adapted_model.py:74-96`

```python
def _compose(self, base_output: Tensor, adapter_result: Tensor | OutputAdapterResult) -> Tensor:
    if isinstance(adapter_result, Tensor):
        return base_output + adapter_result        # plain "delta" adapter, no gate
    output_kind = adapter_result.output_kind.lower()
    composition = self.output_composition.lower()

    if composition == "add":
        return base_output + adapter_result.adapter_output     # additive, no gate

    elif composition in {"replace", "adapter_only"}:
        return adapter_result.adapter_output                   # adapter replaces base

    elif composition in {"mask_mix", "avid_mask_mix"}:         # <-- AVID prediction-blend
        if output_kind != "prediction":
            raise ValueError("Mask-mix composition requires adapter outputs with output_kind='prediction'.")
        if adapter_result.gate is None:
            raise ValueError("Mask-mix composition requires an adapter gate output.")
        gate = torch.sigmoid(adapter_result.gate + self.gate_bias)
        return base_output * gate + adapter_result.adapter_output * (1.0 - gate)

    elif composition in {"gated_residual", "gated_add", "residual_mask"}:  # <-- added 2026-05-29
        if adapter_result.gate is None:
            raise ValueError("gated_residual composition requires an adapter gate output.")
        gate = torch.sigmoid(adapter_result.gate + self.gate_bias)
        return base_output + gate * adapter_result.adapter_output
```

Knobs:

- `output_composition` (YAML `adapter.composition`) — selects which
  branch above runs. The two mask-mix aliases (`mask_mix` and
  `avid_mask_mix`) are interchangeable.
- `gate_bias` (YAML `adapter.gate_bias`, default `0.0`) — additive
  offset before the sigmoid. `5.0` puts the gate near `1.0` at init
  ("trust the base"), `-5.0` near `0.0` ("trust the adapter").

Both AVID and HyperAlign (when `composition: mask_mix`) flow through
the mask-mix branch; they differ only in **what `adapter_result.gate` is**
and **how `adapter_result.adapter_output` was produced**.

### Two gated blends, by what `Δ` *means* (added 2026-05-29)

The gate is a mixing-layer concern and is agnostic to how the adapter
formed its output (direct delta, affine scale/shift, full UNet — see
[[affine-output-granularity]] and
[[../theory/unicon-output-adapters-detached-backward]]). What differs is
whether the adapter output is a **contribution** or a **standalone
prediction**:

- **`gated_residual`** — `base + σ(gate)·Δ`. Δ is a *contribution* (residual).
  This is the thesis core rule `f = base + g·Δ`; `add` is the `g≡1` case. It
  is **identity at init automatically** (Δ→0 ⇒ output=base, for any gate), so
  no `gate_bias` is needed. Recipe-agnostic — works for any output adapter
  that emits a contribution + a gate (e.g. the `output_v2` heads with
  `gate_kind` set).
- **`mask_mix`** — `base·σ(gate) + Δ·(1−σ(gate))`. Δ is a *standalone
  prediction* competing with the base; the gate selects between them. Needs
  `gate_bias` (or a learned bias toward base) for identity at init. This is
  the AVID/HyperAlign path and requires `output_kind="prediction"`.

The `output_v2` heads emit the gate via `gate_kind ∈ {none, channel, dense}`
(extra gate channels appended after the delta/format channels;
`format.prepare_gate` pools/broadcasts them). Note: the DynamiCrafter `affine`
path does **not** yet emit a gate — affine+gate is currently only wired for the
mlp/transformer `output_v2` backbones.

## Family 1 — AVID (output adapter on the DynamiCrafter UNet)

The gate is a **sibling Conv head** of the prediction head. Both heads
read from the same final feature map `h` of the adapter UNet, then
split at the very last conv.

### Adapter UNet output stage

`src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726`

```python
if not self.output_mask:                                          # no-mask branch (no gate)
    self.out = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),  # zero-init
    )
else:
    # zero-initialised mask with one output channel
    self.out_mask = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        zero_module(conv_nd(dims, model_channels, 1, 3, padding=1)),             # gate: zero-init
    )
    # don't zero initialize outputs
    self.out = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        conv_nd(dims, model_channels, out_channels, 3, padding=1),               # y: random init
    )
```

`zero_module` (`backbones/dynamicrafter/basics.py:21-27`) sets both
`weight` and `bias` of the wrapped module to zero in-place.

### Adapter UNet forward (heads run in parallel)

`openaimodel3d.py:811-820`

```python
y = self.out(h)                                              # velocity/noise prediction
y = rearrange(y, "(b t) c h w -> b c t h w", b=b)
if hasattr(self, "out_mask"):
    mask = self.out_mask(h)                                  # raw gate logits, 1-channel
    mask = rearrange(mask, "(b t) c h w -> b c t h w", b=b)
    return y, mask
return y
```

The two heads are **siblings on the same `h`** — `y` is *not* fed into
the gate, and the gate is *not* fed into `y`. They meet only outside
the UNet in the composition above.

### Adapter wrapper packaging

`src/generative_flow_adapters/adapters/output/dynamicrafter.py:146-151`

```python
if self.output_mask:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Expected DynamicCrafter output adapter with output_mask=True to return (prediction, mask).")
    prediction, gate = output
    return OutputAdapterResult(adapter_output=prediction, output_kind="prediction", gate=gate)
```

So at the composition site:

- `adapter_result.adapter_output ← y` (random structured noise at step 0)
- `adapter_result.gate ← mask` (zero everywhere at step 0)

### Net effect at step 0 (AVID live setup)

`gate_bias = 0.0`, `composition = avid_mask_mix`:

```
output = base · sigmoid(0) + y_random · (1 - sigmoid(0))
       = 0.5 · base + 0.5 · y_random
```

50/50 mix with random noise from a freshly-initialised 11M UNet body. Not base pass-through.

## Family 2 — HyperAlign (hypernetwork adapter)

The gate is produced from the **transformer-decoder side** of the
hypernetwork — the same `decoded` tensor that produces the LoRA
factors. Two flavours, selected by `adapter.extra.mask_mix_gate_kind`
(`channel | spatial`, default `channel`).

The architectural philosophy is different from AVID's. In AVID the
gate "knows" about local image structure via the shared `h` feature
map of the adapter UNet. In HyperAlign, the gate is a function of the
**hypernetwork's context-conditioned state** (action / context embedding
+ base-memory cross-attention), not of the noisy latent directly.

### Adapter forward — gate computed during hyper-factor pass

`src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py:172-178`

```python
decoded = decoder(tgt=queries, memory=memory)
factor_head = self._require_factor_head()
factor_tokens = factor_head(decoded)
# Compute the mask-mix gate while we have `decoded` and the captured base
# features in hand; cache it alongside the factors so forward() can pick
# it up without recomputation under stepwise/initial/piecewise modes.
self._cached_gate = self._compute_gate(decoded=decoded, x_t=x_t) if self.output_composition == "mask_mix" else None
```

Note the conditional — the gate is **only computed when
`output_composition == "mask_mix"`**. With `composition: add` (the
live HyperAlign YAML setting), `_cached_gate` stays `None` and forward
returns the additive delta `adapted - reference` instead. See
§"Important caveat" below.

### Flavour A — `channel` gate

`hyperalign.py:181-188`

```python
def _compute_gate(self, *, decoded: Tensor, x_t: Tensor) -> Tensor:
    batch_size, _, frames, height, width = x_t.shape
    if self.mask_mix_gate_kind == "channel":
        pooled = decoded.mean(dim=1)                              # mean-pool over query tokens
        gate = self._gate_head_channel(pooled.to(...))            # Linear(hidden_dim → output_channels)
        return gate.view(batch_size, self.output_channels, 1, 1, 1).to(dtype=x_t.dtype)   # broadcast (T,H,W)
```

- `decoded` is shaped `[B, num_queries, hidden_dim]` (output of the
  HyperAlign transformer decoder).
- Mean-pool the query tokens → `[B, hidden_dim]`.
- Linear `hidden_dim → output_channels` → `[B, output_channels]`.
- Reshape and broadcast → `[B, C, 1, 1, 1]` (will broadcast across
  `(T, H, W)` at the composition).

**Resolution:** per-channel, **constant over space and time**. A given
sample gets one scalar logit per output channel that says "use base
this much" globally.

### Flavour B — `spatial` gate

`hyperalign.py:189-205`

```python
if self._gate_head_spatial is None:
    raise RuntimeError("Spatial gate head was not initialized.")
expected_blocks = len(self._memory_projections) if self._memory_projections is not None else 0
captured = self._feature_store.get(expected_count=expected_blocks)
if captured is None or not captured:
    raise RuntimeError(
        "Spatial mask-mix gate requires captured base features from the frozen base pass."
    )
shallow = captured[0].to(dtype=self._gate_head_spatial.weight.dtype)
gate_flat = self._gate_head_spatial(shallow)                                  # Conv2d(shallow_C → output_C)
if gate_flat.shape[-2:] != (height, width):
    gate_flat = nn.functional.interpolate(
        gate_flat, size=(height, width), mode="bilinear", align_corners=False
    )
channels = self.output_channels or gate_flat.shape[1]
gate = gate_flat.view(batch_size, frames, channels, height, width)
return gate.permute(0, 2, 1, 3, 4).contiguous().to(dtype=x_t.dtype)
```

- Pulls the **shallowest captured base-UNet encoder feature** out of
  `_feature_store`. This store is populated by the forward hooks
  HyperAlign installs on the frozen base during the reference pass at
  line 269.
- 2D Conv `shallow_channels → output_channels`, then bilinear-upsample
  to `(H, W)` of `x_t` if the hooked feature map was downsampled.
- Reshape to `[B, C, T, H, W]`.

**Resolution:** per-pixel + per-channel, **constant over frames within
a sample** (frames inherit the same per-frame map because `shallow` is
`[B·T, C, h, w]` flattened from a per-frame view).

### Gate-head initialisation

Both gate heads are built (and zero-init'd) in `_prepare_mask_mix_gate_modules`:

`hyperalign.py:320-340`

```python
def _prepare_mask_mix_gate_modules(self, encoder_dims: list[int]) -> None:
    if self.output_composition != "mask_mix":
        return                                              # not built at all unless mask_mix is on
    if self.output_channels is None or self.output_channels <= 0:
        raise ValueError(
            "HyperAlign mask_mix composition requires a positive output_channels "
            "(set adapter.extra.output_channels or model.extra.latent_channels)."
        )
    if self.mask_mix_gate_kind == "channel":
        head = nn.Linear(self.hidden_dim, self.output_channels)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self._gate_head_channel = head
    elif self.mask_mix_gate_kind == "spatial":
        if not encoder_dims:
            raise ValueError("Cannot build spatial gate head without encoder feature dims.")
        shallow_channels = int(encoder_dims[0])
        conv = nn.Conv2d(shallow_channels, self.output_channels, kernel_size=3, padding=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        self._gate_head_spatial = conv
```

Both branches zero `weight` and `bias` of the final gate-producing
layer — the same end state as AVID's `zero_module(...)`, just written
without the helper.

### What `y` is at step 0 (mask-mix HyperAlign)

For HyperAlign, `adapter_result.adapter_output` is **not a fresh
random conv output** the way it is for AVID. It is the **base UNet
re-run with LoRA factors set from the hypernetwork**:

`hyperalign.py:268-291`

```python
self.clear_dynamic_parameters()
reference = base_output if base_output is not None else self.base_model(x_t, t, cond=...)
hyper_down, hyper_up = self._resolve_hyper_factors(x_t=x_t, t=t, cond=cond)
for index, handle in enumerate(self._handles):
    handle.wrapped.set_dynamic_hyper_factors(
        down_hyper=hyper_down[:, index, :, :],
        up_hyper=hyper_up[:, index, :, :],
        alpha=self.alpha,
    )
adapted = self.base_model(x_t, t, cond=...)                        # = y in the composition

if self.output_composition == "add":
    return adapted - reference                                     # additive path (no gate)
if self.output_composition == "replace":
    return OutputAdapterResult(adapter_output=adapted, output_kind="prediction")
gate = self._cached_gate                                           # mask-mix path
...
return OutputAdapterResult(adapter_output=adapted, output_kind="prediction", gate=gate)
```

The `_factor_head` Linear at line 306 is **default PyTorch init**
(Kaiming-ish on `weight`, uniform on `bias`) — *not* zero-init. So
at step 0:

- `hyper_down`, `hyper_up` are random small low-rank tensors.
- `adapted = base_model_with_random_LoRA(x_t, t, cond)` — a noisy
  perturbation of the frozen pretrained base.
- `adapted - reference` is a small random delta (only used in the
  `add` branch).
- `gate` logits are `0` everywhere (see above).

So under `mask_mix` HyperAlign with `gate_bias = 0`:

```
output = base · 0.5 + adapted · 0.5
       = base · 0.5 + (base + small_random_LoRA_perturbation) · 0.5
       = base + 0.5 · small_random_LoRA_perturbation
```

i.e. the corruption at step 0 is the *LoRA perturbation* of the
frozen pretrained base, not a random freshly-initialised UNet's
output. Structurally cleaner than the AVID case.

## The asymmetric init story (why zero the gate but not `y`)

The puzzle: if `sigmoid(0) = 0.5`, zero-init'ing the gate alone does
*not* give base pass-through at step 0. So what is zero-init buying?
Three things:

1. **Symmetry-breaking prior** — `gate = 0.5` is the "no commitment"
   starting point. The model has not pre-committed to "use the base"
   or "use the adapter" in any region.
2. **Maximum sigmoid derivative at logit 0** — `d/dx sigmoid(x) =
   sigmoid(x)(1-sigmoid(x))` peaks at `x=0` with value `0.25`. The
   gate moves fastest in early training from this starting point.
   Any non-zero init lands further out on the sigmoid, in
   smaller-gradient territory.
3. **Body gradient flow from step 0** — the gradient flowing back into
   the rest of the UNet body via the gate path is `∂L/∂h_gate =
   ∂L/∂gate · W_gate`. With `W_gate = 0` this is zero. If `W_y` were
   *also* zero, the body would receive **no upstream gradient at all
   on step 0** (1-step stall — see [[../../50_Decisions/open/avid-adapter-init]]
   §"Analysed estimate" for the walked chain rule). Keeping `W_y`
   random ensures the body still learns at step 0 via the `y`-path.

This rationale is mirrored in HyperAlign: the gate heads are zero-init,
but the `_factor_head` that drives `y = adapted` is not. The trade-off
is the same: clean gate prior + body gradient at step 0, at the cost
of step-0 noise from the random `y` head (or random LoRA factors).

## Comparison table

| Aspect | AVID `output_mask` | HyperAlign `channel` | HyperAlign `spatial` |
|---|---|---|---|
| Conv/Linear producing the gate | `Conv3d(model_channels → 1, k=3)` | `Linear(hidden_dim → output_channels)` | `Conv2d(shallow_channels → output_channels, k=3)` |
| Input to the gate head | adapter UNet's final `h` | decoder pooled tokens (mean over queries) | shallowest base-UNet encoder feature |
| Channel dim of gate | 1 (broadcast over `C`) | `output_channels` (per-channel) | `output_channels` (per-channel) |
| Spatial dim of gate | per-pixel `(H, W)` | constant `(1, 1)` broadcast | per-pixel `(H, W)` (after bilinear upsample) |
| Temporal dim of gate | per-frame `T` | constant `1` broadcast | constant `T`-broadcast (per-frame map reshaped) |
| Init helper | `zero_module(...)` | `nn.init.zeros_(weight); zeros_(bias)` | `nn.init.zeros_(weight); zeros_(bias)` |
| Step-0 logit | `0` everywhere | `0` everywhere | `0` everywhere |
| Step-0 sigmoid (default `gate_bias=0`) | `0.5` everywhere | `0.5` everywhere | `0.5` everywhere |
| What `y` (adapter prediction) is | random conv output | LoRA-perturbed frozen base | LoRA-perturbed frozen base |

## Important caveat — live HyperAlign YAML uses `add`, not `mask_mix`

`configs/diffusion_hyperalign_shortcut_metaworld.yaml:35` sets
`composition: add`. So in the live HyperAlign shortcut run the gate
heads are not even constructed (see line 321 short-circuit in
`_prepare_mask_mix_gate_modules`), and `forward()` returns the
additive delta:

```
prediction = base + (adapted - reference)
           = base + (base_with_random_LoRA - base)
```

The `mask_mix_gate_kind: channel` line in that YAML (line 43) is
**dormant** — only fires if you also flip `composition` to `mask_mix`.

The live AVID YAML, in contrast, uses `composition: avid_mask_mix`
(`configs/diffusion_avid_shortcut_metaworld.yaml:37`), so its gate is
active.

## Practical implications for D3

- **AVID composition is the only active mask-mix in the current
  training configs.** Anything we conclude about gate dynamics from
  AVID runs does not automatically transfer to HyperAlign unless we
  flip HyperAlign to `mask_mix` (and define `output_channels` properly).
- **The HyperAlign mask-mix case is structurally cleaner at init** —
  `y` is a LoRA-perturbed frozen base rather than a freshly random
  UNet. The case for changing init under HyperAlign mask-mix is
  weaker than the AVID case (less random noise to suppress).
- **The decision in [[../../50_Decisions/open/avid-adapter-init]]
  applies primarily to AVID.** If we later decide to run HyperAlign
  with `mask_mix`, that note's "Generalization" section should be
  written; for now it's flagged as a sibling concern.

## Gotchas

- **HyperAlign gate is not pixel-aware in `channel` mode.** Despite
  `mask_mix` reading visually similar to the AVID composition, the
  channel-flavour gate has the same value for every pixel of every
  frame in a sample — it is a per-channel global mix. Visualising it
  the way AVID Fig. 2/3 visualises masks would produce a uniform
  image; the "mask" is more like a learned per-channel SNR knob.
- **Spatial gate requires the frozen-base hooks to have fired.** The
  spatial branch reads from `_feature_store` populated by
  `self.base_model(x_t, t, ...)` at line 269. If that pass is skipped
  or hooks aren't registered, line 192-196 will raise.
- **`output_channels` must be set** for HyperAlign mask-mix, otherwise
  the gate head construction fails at line 323-327. The AVID adapter
  reads `out_channels` from the UNet config and never has this issue.
- **`gate_bias` is shared across families.** It's a property of
  `AdaptedModel`, not of the adapter. So if we ever want different
  `gate_bias` for AVID vs HyperAlign in a head-to-head sweep, that's a
  feature request (or two separate YAML configs, which is what we do
  today).
- **All zero-init paths produce `gate = 0`, not `gate = some-target`.**
  If we wanted "gate starts close to 1 to give clean base
  pass-through", we have to set `gate_bias` (not change the init).
  See [[../../50_Decisions/open/avid-adapter-init]] options B and C.

## Open follow-ups

- [ ] Once D3 selects a final composition for HyperAlign, update this
      note to reflect it — and write a sibling decision file if
      HyperAlign `mask_mix` is chosen, since the AVID-init decision
      doesn't transfer cleanly.
- [ ] If we add a third adapter family (LoRA-only, UniCon hidden-state,
      ControlNet, …) that also supports the mask-mix composition,
      extend the comparison table above.
- [ ] Consider whether `gate_bias` should be made adapter-side rather
      than `AdaptedModel`-side, so AVID and HyperAlign in the same
      sweep can use different values.

## Related

- [[../../50_Decisions/open/avid-adapter-init]] — open decision on
  AVID-specific step-0 init. This note documents the *current state*;
  the decision file proposes alternatives.
- [[shortcut-training-modes]] — supervises the adapter output that
  the gate then blends with the base.
- [[../theory/shortcut-training]] — conceptual framing of why the
  base contribution to the composition matters.
- [[../related-work/avid]] — AVID paper note; the paper defines the
  composition formula (eq. 5) but not the init.
- [[../related-work/hyperalign]] — HyperAlign paper note; the original
  HyperAlign work does not use a mask-mix composition, so the gate
  here is a local extension.
- Code: `src/generative_flow_adapters/models/adapted_model.py:74-96`
- Code: `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726, 811-820`
- Code: `src/generative_flow_adapters/backbones/dynamicrafter/basics.py:21-27`
- Code: `src/generative_flow_adapters/adapters/output/dynamicrafter.py:146-151`
- Code: `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py:172-205, 268-291, 320-340`
