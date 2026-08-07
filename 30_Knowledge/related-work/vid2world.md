---
type: paper
status: living
last_updated: 2026-08-07
title: "Vid2World: Crafting Video Diffusion Models to Interactive World Models"
authors: ["Siqiao Huang", "Jialong Wu", "Qixing Zhou", "et al."]
venue: ICLR 2026 (arXiv:2505.14357, May 2025)
year: 2026
url: https://arxiv.org/abs/2505.14357
local_pdf:
relevance: D2 — the fine-tuning counterfactual, on our base and our dataset
---

# Vid2World — the "why not just fine-tune?" objection, made concrete

> **The most important paper for our positioning**, because it removes the
> objection's hypothetical status. It takes **DynamiCrafter**, the same base
> AVID uses for RT-1, and evaluates on **RT-1**.

## What it does

Systematically *causalizes* a 1.1B DynamiCrafter video diffusion model:
causal masks on temporal attention, weight-transfer schemes for the temporal
convolutions, and a causal action guidance mechanism in which the embedding
of `a_{t-1}` is **added to the latent at temporal position `t`**, trained
with action dropout so classifier-free guidance can scale action influence
at inference.

**The backbone weights are updated.** Roughly 100k post-training steps, on
the order of 7 days on 4×A100, per domain. Evaluated on RT-1 manipulation,
CS:GO, and RECON navigation.

## Why it matters to us

**It is tier 4 on the same substrate as our tier 5.** Our thesis argues for
keeping the base frozen. Vid2World unfreezes it, on the same model family
and one of the same datasets, and reports results. A committee will ask what
we give up, and the honest answer is currently unmeasured.

**It also uses additive action injection at a temporal position**, which is
adjacent to our per-frame conditioning finding: the action embedding is
added to the latent at the frame it governs. Worth reading carefully against
our pathway result, because a method that adds at the latent (rather than
through cross-attention into the residual stream) may be another instance of
the same design constraint we identified, arrived at independently.

## The experiment it makes available

Vid2World and AVID share **DynamiCrafter** and **RT-1**. That makes a
head-to-head of *frozen-base-plus-adapter* against *full fine-tuning* on a
matched backbone and dataset unusually tractable, and it is the comparison
the literature does not currently supply
([[frozen-base-world-model-landscape]]).

⚠ Our RT-1 numbers are quarantined as in-sample until the held-out re-eval
lands, so this comparison is gated on that
([[../writing/open-experiments-for-thesis]] A3).

## What to write

Do **not** treat it as a competitor to beat. Treat it as the **quantified
trade-off**: full fine-tuning costs ~7 GPU-days per domain and modifies the
backbone; the frozen-base route costs a fraction and preserves the base for
reuse. State what it buys them and what we give up, with numbers if the
comparison runs and as a stated limitation if it does not.

⚠ **Not read in full.** Metadata and the mechanism summary are from the
literature scan of 2026-08-07 and were verified against the arXiv page. Read
before citing specifics.

## Related

- [[avid]] — same base, frozen instead
- [[frozen-base-world-model-landscape]] — the tier structure
- [[adapower]] — the other tier-5 occupant
