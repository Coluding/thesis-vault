---
type: paper
status: living
last_updated: 2026-08-07
title: "AdaPower"
authors: ["Yuhang Huang", "Shilong Zou", "Jiazhao Zhang", "et al."]
venue: arXiv:2512.03538
year: 2025
url: https://arxiv.org/abs/2512.03538
local_pdf:
relevance: D2 — a second occupant of our cell, frozen base on a DiT world model
---

# AdaPower — frozen-base adaptation of a world foundation model

> **A second occupant of our cell**, found 2026-08-07. Confirms the setting
> is live and being pursued, which helps relevance and removes
> "frozen base plus adapter" as a contribution in itself.

## What it does

Adapts **Cosmos-Predict2-2B** with the base genuinely frozen: *"only the
newly introduced layers … are trained, while the pre-trained WFM parameters
remain frozen."*

Two design points worth noting:

- **The world model's text encoder is replaced by an action encoder** of
  stacked perception layers. That is a different answer to the same question
  we ask, namely where the action should enter, and it is a *substitution*
  rather than an addition.
- Two modules, Temporal-Spatial Test-Time Training and Memory Persistence,
  are inserted **every seven DiT blocks**.

Reports >41% improvement in LIBERO task success without policy retraining.

## Why it matters to us

**The setting is not the contribution.** AdaPower plus AVID means tier 5 has
at least two published occupants, on two different base families. Our claim
has to be the conjunction of taxonomy, both objectives, and step-size
conditioning ([[frozen-base-world-model-landscape]]), not the frozen base.

**Its differentiators are orthogonal to ours.** Test-time training and
memory persistence are about inference-time adaptation and long-horizon
consistency. It presents no adapter taxonomy, no gated additive
composition, and no step-size component. So it does not take our claim; it
narrows the framing of it.

**The text-encoder replacement is a genuine design alternative.** We inject
alongside the base's conditioning; they replace a conditioning stream
entirely. Worth one sentence in Ch3 as a rejected alternative with a reason,
since it bears directly on the pathway question.

⚠ **Not read in full.** Metadata and quoted sentence verified from the arXiv
page via the literature scan; the module descriptions come from the
abstract. Read before citing specifics.

## Related

- [[frozen-base-world-model-landscape]] · [[avid]] · [[vid2world]]
