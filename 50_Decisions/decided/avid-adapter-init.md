---
type: decision
status: decided
created: 2026-05-21
decided_at: 2026-05-21
updated: 2026-05-21
target_date:
scope: adapter
related:
  - "[[shortcut-anchor-schedule]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../30_Knowledge/related-work/avid]]"
  - "config: configs/diffusion_avid_shortcut_metaworld.yaml"
  - "config: configs/diffusion_hyperalign_shortcut_metaworld.yaml"
---

# Decision: AVID adapter step-0 initialization

## Status

**Decided 2026-05-21 — Option D (status quo).** The early-training
noisy-teacher problem this decision was originally about is being
addressed downstream via a step warmup on the d=1 data anchor, see
[[shortcut-anchor-schedule]]. Init stays faithful to upstream AVID.

## Context

The D3 setup uses the AVID-style "mask-mix" composition:

```
prediction = base * sigmoid(gate + gate_bias) + adapted * (1 - sigmoid(gate + gate_bias))
```

Composition site: `src/generative_flow_adapters/models/adapted_model.py:88-94`.

The 11M adapter UNet is built from-scratch (no `checkpoint_path` in
either live shortcut YAML) — see
[[../../30_Knowledge/tech/shortcut-training-modes]] §"Adapter
consumption" and the AVID paper §3.3 (`docs/paper/avid.pdf`).

Two heads inside the adapter UNet drive the step-0 behaviour:

1. **Prediction head** `self.out` — the conv that emits the adapter's
   velocity/noise prediction.
2. **Mask/gate head** `self.out_mask` — the conv that emits the per-pixel
   logits fed through `sigmoid`.

Both heads live in
`src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726`.

## What the current code does (verbatim)

```python
# src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726
if not self.output_mask:
    self.out = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
    )
else:
    # zero-initialised mask with one output channel
    self.out_mask = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        zero_module(conv_nd(dims, model_channels, 1, 3, padding=1)),
    )
    # don't zero initialize outputs
    self.out = nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        conv_nd(dims, model_channels, out_channels, 3, padding=1),
    )
```

`zero_module(...)` is the standard "zero the parameters of a module
and return it" wrapper:

```python
# src/generative_flow_adapters/backbones/dynamicrafter/basics.py:21-27
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
```

So with `output_mask=True` (live AVID setup):

- `self.out_mask` final conv has W=0, b=0 → **gate logits = 0 everywhere
  at step 0** → `sigmoid(0) = 0.5`.
- `self.out` final conv has **random PyTorch init** (Kaiming-ish through
  `nn.Conv*d`) → prediction head emits random structured noise at step 0.

With `adapter.gate_bias: 0.0` in the YAML
(`configs/diffusion_avid_shortcut_metaworld.yaml:38`,
`configs/diffusion_hyperalign_shortcut_metaworld.yaml`), the composition
at step 0 is:

```
prediction = 0.5 · base_velocity + 0.5 · random_adapter_noise
```

## What the upstream AVID code does

`external_repos/avid/latent_diffusion/libs/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:690-708`
is **byte-identical** — same `# don't zero initialize outputs`
source comment. The vendored copy is faithful; this is AVID's deliberate
choice.

## What the AVID paper says

**Nothing about initialization.** §3.3 defines the architecture
(eq. 5: `ε_final = ε_pre ⊙ m + ε_adapt ⊙ (1 - m)`, with
`m = sigmoid(...)`) and Appendix B.2 / Table 9 lists optimizer / LR /
batch / steps. No "we initialize the mask to start at 1", no
"ControlNet-style zero-init", no rationale for the asymmetry.

The asymmetry between the no-mask branch (zero-init `self.out`) and the
masked branch (random `self.out`, zero `self.out_mask`) is therefore an
**implementation choice** present in the AVID code but not motivated in
the paper.

**Analysed estimate** for the rationale (refined after walking the
chain rule):

1. **Symmetry-breaking prior on the gate.** `mask_logit = 0 ⇒ gate =
   0.5` is the "no commitment" starting point. Any non-zero init would
   bias the optimization toward "trust base" or "trust adapter" as a
   prior the mask has to escape.
2. **Maximum sigmoid gradient at logit 0.** `d/dx sigmoid(x) =
   sigmoid(x)(1-sigmoid(x))`, max at `x=0` with value `0.25`. The mask
   moves fastest in early training from this starting point; any other
   init lands further out on the sigmoid where the gradient is smaller.
3. **Body gradient flow from step 0.** With `W_out_mask = 0` the mask
   path contributes zero to upstream body gradient (since
   `∂L/∂h via mask = ∂L/∂mask · W_out_mask = 0`). If `W_out` were also
   zero, the body would receive **no gradient at all** at step 0 — its
   parameters would not update. Keeping `W_out` random ensures the
   `y`-path carries gradient into the body from step 0 onward.

The body-gradient stall under dual zero-init (Option A or C) is in
fact **exactly one optimizer step long**: after step 0, both
`W_out` and `W_out_mask` move off zero (their own weight-gradients
are non-zero since `∂L/∂W = h ⊗ ∂L/∂output` with `h ≠ 0`), so body
gradient flows normally from step 1. Not catastrophic — just a
1-step delay — but worth knowing before choosing A or C.

Not verified against the AVID authors; the paper does not motivate
the asymmetry.

## Why this is a real question for our setup

We stack shortcut supervision on top of vanilla denoising, which adds
two pressures at step 0 that AVID did not face:

1. **Standard diffusion/flow loss vs. random adapter.** At step 0,
   `0.5·ε_pre + 0.5·random` is much further from `ε_data` than `ε_pre`
   alone. Early training mostly fights the random init rather than
   refining the base prior.
2. **Shortcut-direction loss vs. base-derived target.** Both `two_step`
   and `distillation` targets are anchored on the base velocity (Heun
   average for `two_step`; adapter-self-bootstrap that recurses to
   `step_level=1` for `distillation`). Random adapter prediction at
   step 0 is far from those targets too — the shortcut loss starts
   high and noisy.

A clean base pass-through at step 0 would make the standard loss start
near the base's residual (low) and the shortcut loss start near zero
(`prediction ≈ base ≈ Heun-average-of-base`). Specifically relevant
under `distillation`, where the self-consistency target is constructed
from no-grad calls of the *adapted* model — a noisy adapter at step 0
also corrupts its own teacher.

## Options

### A — Zero-init `self.out` (ControlNet-style)

Wrap the prediction head's final conv in `zero_module`:

```python
# proposed edit to backbones/dynamicrafter/modules/networks/openaimodel3d.py:722-726
self.out = nn.Sequential(
    normalization(ch),
    nn.SiLU(),
    zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),  # was: no zero_module
)
```

Step 0: `prediction = 0.5·base + 0.5·0 = 0.5·base`.

- ✅ Cleaner than current — no random noise from the adapter at step 0.
- ✅ Gradients still flow into `self.out` weights (zero output, non-zero
  gradient through `∂L/∂W = h ⊗ ∂L/∂output`).
- ⚠️ **Body parameters receive no gradient at step 0** (both `W_out` and
  `W_out_mask` are zero ⇒ both upstream-gradient paths are zero). Stalls
  body learning by exactly one optimizer step. See "Analysed estimate"
  in the §"What the AVID paper says" block above.
- ⚠️ The 0.5 factor still attenuates the base, so `prediction ≠ base`.
  Standard diffusion loss at step 0 is not the base's loss.
- 🔻 Departure from upstream AVID. Worth noting in the thesis if cited.

### B — Bump `gate_bias`

YAML-only change, no code touch:

```yaml
# configs/diffusion_avid_shortcut_metaworld.yaml
adapter:
  composition: avid_mask_mix
  gate_bias: 5.0   # was 0.0; sigmoid(5) ≈ 0.993
```

Step 0: `prediction ≈ 0.993·base + 0.007·random_noise ≈ base`.

- ✅ One YAML line. Reversible. No repo-code divergence from upstream
  AVID.
- ✅ Composition starts at near-exact base pass-through.
- ⚠️ The mask logits have to ride down from `+5` wherever the adapter
  should take over (robot arm, character, etc.). Mask gradients in the
  flat region of `sigmoid` are small — slower mask learning early on.
- ⚠️ Asymmetric vs. AVID's mask interpretation (mean mask drifts to
  ≈ 0.5 over training in AVID — see paper Fig. 4d). Starting at 0.993
  means the model has to learn that drift from a *biased* prior. Not
  obviously bad, but a structural choice.

### C — Both: zero-init `self.out` AND `gate_bias > 0`

Combine A and B. Pick `gate_bias` such that `sigmoid(gate_bias) ≈ 1`
(e.g. `gate_bias = 5.0`).

Step 0: `prediction ≈ 0.993·base + 0.007·0 ≈ base`. Exact identity to
the pretrained base.

- ✅ Cleanest start. Standard loss at step 0 *is* the base's loss.
  Shortcut loss at step 0 is near zero (modulo Heun second-derivative
  curvature, which is small).
- ✅ Both heads still receive non-zero gradients on their own conv
  weights (zero output, non-zero `∂L/∂W = h ⊗ ∂L/∂output`).
- ⚠️ **Same 1-step body-gradient stall as A.** With both heads zero,
  upstream gradient through the UNet body is zero at step 0. Recovers
  from step 1 onward.
- ⚠️ Two changes to roll back if it doesn't help. Mask early-learning
  same caveat as B (gate starts at `sigmoid(gate_bias) ≈ 1` so the
  sigmoid is saturated in low-gradient territory).

### D — Status quo (do nothing)

Keep current init. Justify by citing fidelity to upstream AVID.

- ✅ Cheapest. No risk of diverging from a known-working AVID setup.
- 🔻 Doesn't address the shortcut-training-specific pressure described
  above. May simply be wasted early-training compute.
- 🔻 No comparison evidence available either way until we run an
  ablation.

## Recommendation

Lean **C** for the first real D3 run, with **B** as the minimum-diff
fallback if we want to avoid touching `openaimodel3d.py` at all.

Justification ranked:

1. **D3 evaluates trade-offs between adapter families** ([[../../10_now/positioning]]
   etc.). Early-training compute spent fighting random init is noise in
   that comparison — it'll wash out eventually but biases the
   sample-efficiency story.
2. **Distillation mode bootstraps the teacher from the adapter** — a
   clean step-0 adapter ⇒ a clean step-0 teacher target ⇒ cleaner
   self-consistency learning. Random init poisons that recursion early.
3. **B alone is reversible in YAML**, so if C turns out to under-perform
   we can revert to B without recompiling. If C turns out to be a wash
   vs D, that's a thesis-worthy observation.
4. **The 1-step body-gradient stall for A/C is real but cheap.** One
   step out of (tens of) thousands. Worth the cleaner step-0 output as
   long as we know to look for it (e.g. don't be alarmed if `step=0`
   body-norm metric is identical to init under A/C).

## Decision

**D — Status quo.** Keep upstream-AVID's asymmetric init:
random-init `self.out`, `zero_module(self.out_mask)`, `gate_bias=0.0`.
Step-0 prediction stays at `0.5·base + 0.5·random_adapter_noise`.

Departs from the prior recommendation (lean C). Rationale:

1. **Thesis comparison fidelity.** D3 evaluates trade-offs between
   adapter *families* on the same base + dataset. Staying byte-faithful
   to upstream AVID removes one degree of freedom from the AVID
   baseline's interpretation. A clean-init variant would always invite
   the question "is this AVID, or is this a modified AVID."
2. **The shortcut-specific pressure is addressed downstream.** A step
   warmup on the d=1 data anchor (see [[shortcut-anchor-schedule]])
   gives the adapter time to learn structured predictions before the
   self-consistency teacher comes online. That curriculum solves the
   same noisy-teacher problem the C-init was solving, without changing
   the architecture.
3. **Reversibility.** D is the no-op. If the warmup turns out to be
   insufficient, we can revisit and adopt A/B/C without having already
   burned ablation budget on init variants.

## Consequences

- `configs/diffusion_avid_shortcut_metaworld.yaml` and
  `configs/diffusion_hyperalign_shortcut_metaworld.yaml` stay as-is
  (`gate_bias: 0.0`, no `zero_module` wrap on `self.out`).
- `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726`
  untouched. The vendored AVID code remains byte-identical.
- **Implicit dependency**: [[shortcut-anchor-schedule]] must resolve
  to *something* that handles the noisy-teacher problem. Leaving both
  decisions on "status quo" simultaneously is not a valid combination
  — the shortcut training would then take the full hit of
  `0.5·base + 0.5·random` corrupting its self-consistency teacher.
- The AVID mean-mask convergence trajectory remains an *observable*
  in the first AVID shortcut run, with one extra angle: the warmup
  boundary in the schedule may show up as a discontinuity in the
  per-timestep mean-mask curve. Worth logging.

## Follow-ups

- Add a one-paragraph note to
  [[../../30_Knowledge/tech/shortcut-training-modes]] explaining
  that the asymmetric init is retained despite the shortcut-specific
  pressure, with the warmup acting as the mitigation. _Pending: that
  knowledge note doesn't exist yet — fold the note in when the file
  is populated._
- If the schedule warmup proves insufficient after a baseline run,
  re-open this decision and reconsider C as a follow-up rather than
  a primary path.

## Open questions

- **What does AVID's mean-mask plot (Fig. 4d) actually trend toward at
  convergence?** The paper shows mask values < 0.5 at high-noise
  diffusion steps and > 0.5 at low-noise steps for RT1, with the
  opposite for Coinrun. If our MetaWorld setup converges to a
  fundamentally different mean-mask regime, option B's "start at 0.993"
  could be actively wrong. _needs verification — log as ablation
  observable._
- **Does `gate_bias > 0` interact badly with the per-pixel mask
  semantics?** AVID's Fig. 2/3 show the mask going to 0 around the
  agent/object that the adapter must inject. A biased starting point
  may simply add a constant shift the optimiser cancels — but could
  also slow per-region differentiation. No clean way to tell without an
  ablation.
- **Does B reproduce A's effect at all?** With B alone, the adapter's
  prediction head is still random — just gated down to 0.7% of its
  contribution. That 0.7% × `||random||` may still be larger than
  zero, and may still corrupt the shortcut target under `distillation`.
  A is structurally cleaner; B is operationally cheaper.

## Related

- [[../../30_Knowledge/tech/shortcut-training-modes]] — shortcut
  supervision setup that this decision interacts with.
- [[../../30_Knowledge/related-work/avid]] — paper note (verify it
  exists; populate if missing).
- AVID paper: `docs/paper/avid.pdf` §3.3 (architecture), App. B.2 +
  Table 9 (training details).
- Code: `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:708-726`
- Code: `src/generative_flow_adapters/backbones/dynamicrafter/basics.py:21-27`
  (`zero_module` definition)
- Code: `src/generative_flow_adapters/models/adapted_model.py:88-94`
  (composition site)
- Code: `src/generative_flow_adapters/adapters/output/dynamicrafter.py:33-66`
  (adapter constructor)
