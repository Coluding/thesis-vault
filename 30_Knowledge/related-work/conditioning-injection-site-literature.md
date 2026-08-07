---
type: writing
status: living
last_updated: 2026-08-07
sources:
  - "[[dit-peebles-xie]]"
  - "[[gentron]]"
  - "[[film]]"
  - "[[controlnet]]"
  - "[[../writing/rubric/01-originality]]"
---

# Where conditioning enters: what the literature settles, and what it does not

> **The scan that decides whether our pathway result reads as principled or
> as a lucky ablation** (2026-08-07). Short version: the ordering we measure
> is published, its opposite is also published, and the variable that
> reconciles them puts our setting at an untested point.

## The contradiction

| Paper | Setting | Verdict |
|---|---|---|
| **DiT** (Peebles & Xie, ICCV 2023, arXiv:2212.09748) | class-conditional image diffusion, **trained from scratch** | *"The adaLN-Zero block yields lower FID than both cross-attention and in-context conditioning"*, ~2× FID gap at 400K iterations. Concludes *"the conditioning mechanism critically affects model quality"* |
| **GenTron** (Chen et al., CVPR 2024, arXiv:2312.04557) | **text**-conditional, same architecture family | The reverse: *"Cross attention exhibits a distinct advantage in the text-conditioned scenario"*, T2I-CompBench mean **47.84 (cross-attn) vs 34.32 (adaLN-Zero)** |
| **I2V-Adapter** (SIGGRAPH 2024, arXiv:2312.16693) | image identity into a **frozen** T2V model | Explicitly rejects cross-attention embedding injection: it *"typically fails to preserve the identity of the input image"* |

**GenTron's own reconciliation is the key.** The deciding variable is not
the mechanism but **the nature of the signal**: adaLN's global modulation
suffices for a fixed class label, while text needs cross-attention because
it *"treats spatial positions with more granularity"*.

## Where our measurement sits

Sort conditioning signals by dimensionality and spatial extent:

```
class label  ──────────────  action vector  ──────────────  text  ──  dense control map
(global, low-dim)            (global, low-dim,             (high-dim,   (spatially aligned)
 fixed per sample)            PER TIMESTEP)                 spatial)
```

adaLN wins the left end (DiT), cross-attention wins the right (GenTron,
GLIGEN's boxes, ControlNet's dense maps). **A per-timestep low-dimensional
action vector sits near the left end and has not been tested there**, and
every study above trains the conditioning and the host network **together**.

**Our point on the axis is therefore doubly untested: a low-dimensional
per-timestep signal, injected into a base that is frozen.** That is the
claim to make. Not *"modulation beats cross-attention"*, which is contested
and would invite GenTron as a counterexample.

**A prediction worth stating**, because it makes the result explanatory
rather than anecdotal: GenTron's axis predicts our outcome. An action vector
is on the class-label side, so modulation should beat cross-attention there.
It does. Say that the result is consistent with the field's own explanation
of its contradiction, and that we supply the missing measurement.

## The second gap: zero-init gating

Every frozen-base adapter uses a near-identity or zero initialisation, and
always for the same stated reason:

| Method | Construction | Stated motivation |
|---|---|---|
| Houlsby adapters | near-zero projection init | *"initialized to an approximate identity function"* |
| ControlNet | zero convolutions | *"no harmful noise could affect the finetuning"* |
| AnimateDiff | zero-init output projections | identity mapping at start of training |
| GLIGEN | gated injection | preserve pretrained knowledge |
| LLaMA-Adapter | zero-init attention with zero gating | *"effectively preserves its pre-trained knowledge"* |
| adaLN-Zero | zero-init dimension-wise scale | identity block at init |

**The same construction that guarantees a safe start can leave a
conditioning branch permanently attenuated.** Only two works push back:
ControlNet-XS reframes the coupling as a bandwidth and delay problem, and
ControlNeXt goes further, proposing *"Cross Normalization (CN) as a
replacement for 'Zero-Convolution'"*. Neither frames it as *the conditioning
input is not being used*.

> **The gap between "stable initialisation" and "the gate never opens" is
> where our gate-saturation and drowning results live.**

Note ControlNeXt's replacement is a *normalisation-statistics alignment*,
which is mechanically adjacent to our finding that modulation of
**normalised** activations survives where addition into an unnormalised
stream does not. Worth citing as independent convergence.

## The third gap, and it is unclaimed

**Adapters that read the frozen base's own output are a recognised design,
and nobody measures their internal credit assignment.**

- **X-Adapter** (CVPR 2024) feeds a frozen model's decoder features into
  mapping layers. It *wants* that dependence; it is the transport mechanism.
- **AVID** feeds the pretrained model's prediction `ε_pre` directly into the
  adapter alongside the noisy video and the conditioning frame, and blends
  with a learned mask: `ε_final = ε_pre ⊙ m + ε_adapt ⊙ (1 − m)`.

Both report task-level wins. **Neither reports how much of the adapter's
output is explained by the base prediction versus by the nominal
conditioning input.** That is exactly our ~100:1 sensitivity measurement.

The nearest methodological tool in the literature is Minimal Impact
ControlNet's treatment of *"the asymmetry in the score function's Jacobian
matrix induced by ControlNet"*, and it is applied to control-versus-control
interference, not control-versus-base-prediction.

➜ **This validates the probe suite as a contribution in its own right**, not
merely as instrumentation for our results
([[../writing/rubric/02-technical-skills]], [[../tech/probe-suite]]).

## How to write it

1. **Open Ch2 §2.2 with the contradiction**, DiT versus GenTron. It is a
   real disagreement in the field and it organises the section.
2. **Name the reconciling variable** (signal nature) and place our setting
   on the axis, noting it is untested and that the base is frozen where
   theirs are not.
3. **State that our result is what GenTron's axis predicts.** Consistency
   with the field's own explanation is stronger than novelty against it.
4. **Then the zero-init tension**, ending on the "gate never opens" framing.
5. **Then the credit-assignment gap**, which motivates the probe suite.

⚠ Per-paper notes for DiT, GenTron, FiLM, ControlNet, ControlNeXt,
LLaMA-Adapter, I2V-Adapter and X-Adapter still to be written; metadata is
verified but only abstracts have been read.
