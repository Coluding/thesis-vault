---
date: 2026-08-07
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260806-pdd-parallel-decoding-works-on-the-lora-base-not-the-adapter]]"
  - "[[../../20_Tickets/experiments/exp-shortcut-parallel-decoding-adapter-dc]]"
  - "[[../../20_Tickets/experiments/exp-shortcut-pdd-lora-distill-dc]]"
---

# Parallel decoding: the plug-and-play adapter cannot do it, and no loss we have would tell us

**One line:** on Parallel Decoding Distillation, the paper-faithful student (LoRA inside
the base + N replicated heads) learns to generate a video in **one network call instead of
eight**; our adapter-over-a-frozen-base transposition does not, at any capacity tested,
and the training loss ranks the two the wrong way round.

## The setup

PDD (arXiv:2607.26004) replicates the final layer N times so one forward pass emits N mean
velocities, one per interval of the timestep grid. Two formulations on a frozen
DynamiCrafter base, action-free, N = 8:

- **Idea A** — N heads on a separate additive adapter. Our transposition onto the thesis's
  composition rule.
- **Idea B** — LoRA on the backbone plus its own final conv replicated N times. What the
  paper actually does, made parameter-efficient.

## The result

At **matched trainable parameters** (146,373,664 vs 146,405,408, 0.02% apart), identical
objective, grid, batch, learning rate and data:

| | training loss | 1-call rollout |
|---|---|---|
| **A** (adapter) | **0.128** | unusable: noise texture, no scene |
| **B** (LoRA'd base) | ~0.64 | **recognisable robot arm on the box** |

A fits the distillation targets **5x better** and cannot decode. B fits them 5x worse and
produces the video.

Capacity was the obvious objection and it has been excluded: A at 13x its original size
improved its loss 4x (0.524 to 0.128) and the rollout did not improve at all.

## Why this matters beyond PDD

**No scalar we have can rank these models.** Four independent quantitative signals point
the wrong way: PSNR (A 17.85 vs B 16.33), latent MSE to ground truth (A 0.44 vs B 2.0),
implied latent correlation (A ~0.78 vs B ~0), and the sanity check that a 50-step base
scores *below* an 8-step base. Only looking at decoded frames separates them.

Concretely, `best.pt` is selected on eval_loss, so on arm A the checkpoint chosen as
"best" is not the one that generates best, and no loss-based criterion would have caught
it. **Any future PDD work needs a generation-based selection criterion.**

## What we can and cannot say

Supported: an additive adapter over a frozen base did not learn parallel decoding here,
while the same parameter budget spent as LoRA inside the base did. The remaining
untested explanation is that **B's heads read features from the 1.4B pretrained backbone
while A's read from a network trained from scratch** — so the finding is about access to
pretrained features, not about additive-versus-in-place composition as such.

Not supported: any quantitative quality claim. The A-vs-B difference is "scene versus no
scene" on 16 clips per arm, judged by eye. A perceptual metric is the missing piece.

Also honest: a hypothesis I put on record was refuted by measurement. I predicted A had
collapsed into a low-variance region of the velocity field; the rollout latents measure
std 1.009-1.023 against ground truth on 16 of 16 clips, so there is no collapse. The
mechanism remains open.

## Next

1. A perceptual metric on the generations, to turn the headline into a number.
2. Head fusion (PDD Eq. 15), still deferred, which would let a rollout use fewer steps
   than heads.
3. Test the pretrained-features explanation directly, e.g. by letting A's heads read the
   frozen base's intermediate features rather than only its output velocity.
