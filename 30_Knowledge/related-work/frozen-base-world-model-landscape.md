---
type: writing
status: living
last_updated: 2026-08-07
sources:
  - "[[avid]]"
  - "[[adapower]]"
  - "[[vid2world]]"
  - "[[../../10_now/positioning]]"
---

# The world-model landscape, sorted by what happens to the weights

> Literature scan 2026-08-07. The axis that matters for our positioning is
> **what happens to the generative model's parameters**, not what the model
> is. Sorted that way, the field has five tiers and we are in the fifth.

## The five tiers

| # | Regime | Papers | Cost / character |
|---|---|---|---|
| 1 | **From scratch, latent-space, per environment** | Ha & Schmidhuber; PlaNet; Dreamer V1/V2/V3 | No generative prior. Cheap per task, no visual transfer, no pixel-fidelity claims |
| 2 | **From scratch, pixel/token generative, per environment** | Oh et al. 2015; DIAMOND; Diffusion Forcing | Full diffusion/AR machinery, but every environment pays the full cost and inherits no prior |
| 3 | **From scratch at foundation scale** | Genie; iVideoGPT; UniSim; Cosmos | The prior is *built*, not borrowed. UniSim: 512 TPU-v3 × 20 days. Cosmos: 10,000 H100 × 3 months |
| 4 | **Initialised from a pretrained model, then fully fine-tuned** | GameNGen (SD 1.4); GR-1; **Vid2World** (DynamiCrafter); AdaWorld (SVD); Cosmos post-training; Genie Envisioner | **The dominant current recipe**, and the first thing a committee points at |
| 5 | **Pretrained base held frozen, small trained module carries the conditioning** | **AVID**; **AdaPower**; plausibly DWS; plausibly GigaWorld-1 | **Our cell** |

## Our cell is occupied, thinly and recently

- **AVID** (Rigter, Gupta, Hilmkil, Ma; arXiv:2410.12822; **RLC/RLJ 2025**)
  is the direct antecedent. Base treated as a black box, "without access to
  the parameters of the pretrained model". One adapter design, two domains,
  no step-size conditioning.
- **AdaPower** (arXiv:2512.03538, Dec 2025) is genuinely frozen-base on
  Cosmos-Predict2-2B, a DiT. Its contribution is test-time training and
  memory persistence, not an adapter taxonomy. ➜ [[adapower]]
- **DWS** (arXiv:2502.07825) claims a universal action-conditioned module
  for any pretrained model. ⚠ **Freezing regime unverified.** Verify before
  finalising positioning: it is either a third occupant of tier 5 or it
  belongs in tier 4.

**Consequence: the setting alone is no longer a contribution.** "We keep the
base frozen and train an adapter" places us in a populated cell.

## ⭐ UPDATE 2026-08-07b — the perimeter is tighter, and D1/D3 are safer

Second scan, focused on the immediate neighbourhood. Three corrections and
one warning.

**Corrections to the tier table above:**

- **DWS is AAAI 2025 and it fine-tunes.** It is **tier 4**, not a third
  occupant of tier 5. The ambiguity is resolved.
- **IOI (arXiv:2606.23296) is tier 5, on Wan.** Verbatim: *"Only the
  Kinematic Fusion Embedder, Alignment Embedder, and Kinematic Blocks are
  trainable. All DiT parameters remain frozen throughout."* It conditions on
  **rendered kinematics** (forward kinematics → 3-view orthographic renders
  → zero-init gated side branch), not on a raw action vector, which is the
  distinction to draw.
- **AVID's venue** is RLC 2025 / RLJ paper 64, not a preprint. Corrected in
  [[avid]].

**⚠ The warning: a frozen-backbone adapter cluster exists outside AVID's
citation graph.** None of these cite AVID and none were in the vault:

| Work | Base | Regime | Action interface |
|---|---|---|---|
| **Aero-World** (2605.19728) | video DiT | **frozen + LoRA** | IMU → MLP → **cross-attention token stream** |
| LOME (2603.27449) | Wan2.1-VACE-14B | frozen, LoRA r=128 on VACE only | 3D keypoints rasterised to 2D action maps |
| RealWonder (2603.05449) | Wan2.1-1.3B-InP | frozen, LoRA r=2048 | physics sim → optical-flow / RGB control images |
| EA-WM (2605.06192) | Wan2.2 DiT | frozen + LoRA r=32 | kinematic-to-visual action fields, projected to camera view |
| Generated Reality (2602.18422) | Wan2.2 14B I2V | LoRA r=32 | **ablates TokenConcat / AdaLN / CrossAttn / TokenAddition** |

**Aero-World is our exact configuration**: MLP(action) → cross-attention
into a frozen base. This is where a reviewer asking *"isn't this just
LoRA?"* will point, and we currently cite none of them.

Note the pattern: **four of the five condition on a spatially-aligned
rendering** (keypoint maps, optical flow, action fields, orthographic
renders) rather than on a raw action vector. That is the same conclusion
GigaWorld-1 reaches, and it is a real convergence in the field worth naming
in Ch2.

**✅ D3 is uncontested** among AVID citers: no citer implements a
step-size-conditioned or shortcut adapter. Nearest adjacent risk is
**MWM (arXiv:2603.07799)**, action-conditioned consistency post-training
plus "Inference-Consistent State Distillation" for few-step rollout. It is
distillation rather than step-size conditioning and says nothing about a
frozen base, but it is adjacent and **not in the vault**. Highest-priority
read.

**✅ D1's conjunction has no match found.** DWS explicitly notes AVID is
*"limited to diffusion-based models"*, which is an argument **for** our
flow-matching coverage rather than against it.

## The gap that is ours, stated as a conjunction

No paper in the scan does all three:

1. a **comparative taxonomy** of adapter families (LoRA / hidden-state /
   output-level / hypernetwork) under a **single composition interface**,
   evaluated on the same task. Every frozen-base paper above commits to one
   adapter design and does not ablate across families;
2. the same interface spanning **both diffusion and flow matching** backbones.
   All tier-5 works are diffusion-only;
3. **step-size conditioning** on a frozen base. No frozen-base adapter paper
   conditions the adapter on `d`; Diffusion Forcing is the closest
   sequence-level analogue and is explicitly *not* built on a pretrained
   backbone.

**Write the contribution as the conjunction.** Each element alone is either
taken or thin; together they are not.

## The composition distinction worth stating

AVID combines by a **convex mask-mix** of the two noise predictions,
`ε_final = ε_pre ⊙ m + ε_adapt ⊙ (1 − m)`. Ours is a **gated additive
residual**, `f_base + g(d)·Δ_φ`. That is a real difference and it is what
makes step-size conditioning expressible: the gate is a function of `d`,
which a convex mask over two predictions does not naturally accommodate.
State it explicitly rather than glossing it.

## ⚠ The objection, and it now has a name

**Vid2World** (ICLR 2026, arXiv:2505.14357) causalizes **DynamiCrafter**,
the same base AVID uses for RT-1, and evaluates on **RT-1**. So "why not
just fine-tune the backbone?" is no longer hypothetical: someone did it, on
our base, on our dataset. ➜ [[vid2world]]

**Three citable answers**, all from the literature rather than from us:

1. **Access.** The strongest video models are closed-weight; AVID's own
   stated motivation.
2. **Function, not just economy.** AVID's analysis finds the frozen base
   carries background texture while the adapter carries action-relevant
   regions, so the split is functional rather than a compute concession.
3. **Parameter economy.** AVID's adapters are 11M–145M against a 1.4B base.

**What the literature does not supply**, and what would settle it: a
head-to-head of frozen-base-plus-adapter against full fine-tuning on the
**same backbone and dataset**. Vid2World and AVID share both, which makes
that comparison unusually tractable and probably the single most valuable
experiment for defending the positioning.

## ⚠ Unverified, do not assert

Genie 2 and Genie 3 are **blog-only**, no technical report. UniSim's
from-scratch status is an argument from silence. IRASim's initialisation,
DWS's freezing regime, GigaWorld-1's architecture, and the NeurIPS 2018
version of Ha & Schmidhuber are all unverified in this scan.
