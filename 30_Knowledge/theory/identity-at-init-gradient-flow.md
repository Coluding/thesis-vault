---
type: theory
status: living
last_updated: 2026-06-02
sources:
  - "code: src/generative_flow_adapters/models/adapted_model.py"
  - "code: src/generative_flow_adapters/adapters/output/output_head.py"
  - "code: src/generative_flow_adapters/adapters/output/format.py"
relevance: D1 / D2 / D3 / D4  # identity-at-init is a property of the core composition rule
---

# Identity at init — why a zero-init adapter still trains (zero output ≠ zero gradient)

> Every output adapter in the repo zero-inits its **final projection** so that
> at step 0 the contribution `Δ ≈ 0` and the composed model reproduces the
> **frozen base exactly**: `f = base + g·Δ ≈ base`. The recurring confusion is
> that this looks like a dead start — if the adapter outputs zero, how does it
> ever move? The resolution: **zero *output* is not zero *gradient*.** The
> final layer's weight gradient depends on its **input activations** (nonzero)
> and on `∂L/∂Δ` (nonzero, *because the frozen base is not a perfect predictor
> of the action-conditioned target*). So the adapter receives a real gradient on
> the very first step and learns a residual correction whose size is exactly
> "how wrong the base still is."

This is the trainability counterpart to [[../tech/mask-mix-gate]] (which
catalogues the *init regimes* of the gate heads) and the
ControlNet/zero-conv argument it inherits. See the "asymmetric init story"
in that note (§ why zero the gate but not `y`) for the AVID/HyperAlign
specifics; this note is the clean general statement.

## 1. The composition is identical to the base at init, not zero

`adapted_model.py` — `add` and the new `gated_residual` branch:

```
prediction = base_output + g · Δ        # g≡1 for `add`; g = σ(gate+bias) for gated_residual
```

- **`base_output` is the frozen pretrained model's prediction** — full,
  meaningful, nonzero, and unchanging. It is NOT zero.
- **`Δ` is the adapter's contribution**, ≈0 at init because the final
  projection is zero-initialised (`_MLPBackbone.head`, `_VideoTransformerBackbone.unpatch`,
  the DynamiCrafter `zero_module` out head — all `weight=0, bias=0`).

So "identity at init" means *identical to the frozen base*, **not** "outputs
zero." A world model that emitted zeros would be useless; one that starts as a
perfect copy of the pretrained base and learns the action-conditioned residual
on top is the whole design of the thesis core rule.

## 2. Why a zero *output* still produces a nonzero *gradient*

Take the final linear `Δ = W·h + b` with `W=0, b=0` at init, `h` = activations
feeding it. Backprop gives:

```
∂L/∂W = (∂L/∂Δ) · hᵀ      ← nonzero: depends on the INPUT h, not on the value of W
∂L/∂b =  ∂L/∂Δ            ← nonzero
```

A linear layer's weight gradient is set by its **inputs**, not its current
weights. `W=0` zeros the *forward* output but not `∂L/∂W`. The two ingredients
are both present at init:

- **`h ≠ 0`** — the adapter trunk runs on real data and produces real activations.
- **`∂L/∂Δ ≠ 0`** — *because* `prediction = base + 0 = base`, and the frozen
  base is not a perfect predictor of the action-conditioned target. There is
  residual error, so the loss has a nonzero slope w.r.t. the adapter output.

**The adapter only grows to the extent the base is wrong.** If the base were
already perfect, `∂L/∂Δ → 0` and the adapter would correctly stay near zero —
nothing to learn. This is the desired behaviour of a residual correction.

## 3. The one-step upstream stall (zero-conv property)

The gradient flowing *back through* the zero-init layer into the trunk is:

```
∂L/∂h = Wᵀ · (∂L/∂Δ) = 0     at init, since W=0
```

So for **exactly one step** the layers *upstream* of the zero-init projection
get no gradient — only the final projection updates. Once that step makes `W`
nonzero, `Wᵀ ≠ 0` and gradients propagate to the whole trunk on step 1. The
network "unblocks itself" after a single warm-up step. This is precisely the
ControlNet zero-convolution mechanism (Zhang et al.): harmless identity start
(no random garbage injected into the frozen base) **and** trainability, at the
cost of one step of upstream warm-up.

> Contrast with the AVID `mask_mix` init choice in [[../tech/mask-mix-gate]]:
> there `y` (the adapter prediction head) is deliberately left *random*-init
> while only the gate is zero-init, specifically so the UNet body still
> receives gradient on step 0 via the `y`-path and avoids the stall — at the
> cost of step-0 noise. Our `gated_residual` + zero-init-Δ path accepts the
> one-step stall in exchange for a clean base-pass-through start.

## 4. The gate learns *after* Δ, not before

For `gated_residual`, `prediction = base + σ(gate_raw)·Δ`, so:

```
∂L/∂(gate_raw) ∝ Δ · σ'(…)
```

With `Δ ≈ 0` at init the gate gets ~no gradient — it cannot meaningfully train
before there is a contribution for it to modulate. The learning order is:
**final projection moves first → Δ becomes nonzero → trunk and gate then begin
learning.** Note the gate sits at `σ(0) = 0.5` at init (not 0); the
base-pass-through start comes entirely from `Δ→0`, NOT from the gate — which is
why `gated_residual` needs no `gate_bias` for a neutral start, unlike
`mask_mix`. (See [[../tech/mask-mix-gate]] § "Two gated blends".)

## Practical implications

- **Don't read a flat adapter-output norm at step 0 as "not training."** Watch
  `‖Δ‖` *growing* over the first few hundred steps and the loss dropping below
  the frozen-base baseline — that's the signal the residual is being learned.
- **A base that's already good ⇒ slow/small adapter.** If `‖Δ‖` stays tiny, it
  may mean the base already explains the target on that data slice, not that
  training is broken.
- **Gate metrics are uninformative early.** The gate barely moves until `Δ`
  is nonzero; judge it only after the warm-up.

## Related

- [[../tech/mask-mix-gate]] — init regimes of every gate-production path; the
  "asymmetric init story" is the AVID/HyperAlign-specific version of §3 here.
- [[unicon-output-adapters-detached-backward]] — the *other* reason these
  adapters are cheap: the frozen base is detached, so no gradient flows
  through it at all.
- [[../../50_Decisions/open/avid-adapter-init]] — open decision on whether the
  asymmetric (gate-zero, y-random) init is right for our shortcut setup.
- [[shortcut-training]] — why the base contribution to the composition matters.
- Code: `src/generative_flow_adapters/models/adapted_model.py` (`_compose`, `gated_residual`/`add` branches)
- Code: `src/generative_flow_adapters/adapters/output/output_head.py` (zero-init `head`/`unpatch`)
