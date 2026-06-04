---
type: paper
status: living
last_updated: 2026-06-04
title: "Surfing Uncertainty: Prediction, Action, and the Embodied Brain"
authors: ["Andy Clark"]
venue: "Oxford University Press (book)"
year: 2016
url: https://www.thebsps.org/reviewofbooks/andy-clark-surfing-uncertainty-prediction-action-and-the-embodied-brain/
local_pdf:
relevance: motivation   # conceptual framing for the multimodal-coupled-dynamics direction (D2/multimodal)
---

# Surfing Uncertainty (Andy Clark, 2016) — predictive processing as a motivation for multimodal coupled dynamics

> **What this note is.** A conceptual-motivation source, *not* an ML
> related-work baseline. Andy Clark's book popularises the **predictive
> processing (PP)** account of the brain. The user wants the thesis's
> multimodal-coupled-dynamics direction
> ([[../../50_Decisions/open/multimodal-adapter-broadening]]) motivated by the
> PP argument that a cognitive system is a generative model predicting
> *multiple sensory modalities forward in time*, and that *action* is part of
> the same predictive machinery.
>
> **Sourcing.** Claims below are drawn from **Andrew Buskell's review** (BSPS
> Review of Books; full text supplied by the user 2026-06-04; also mirrored at
> philsci-archive 24299) with page references *as cited in that review*. The
> primary NDPR review (Rescorla) was also read and is cited for criticism. I
> have **not** read the book directly — page-level claims are second-hand via
> the reviews and are marked accordingly. The ML bridge at the bottom is an
> **analysed framing (the user's argument), not a claim the book makes.**

## Citation (note a subtitle discrepancy)

- Buskell's review and the BSPS page render the subtitle **"…and the Embodied
  Brain"**, Oxford: OUP, 2016, ISBN 9780190217013.
- OUP's catalogue, Goodreads, and the NDPR review render the subtitle **"…and
  the Embodied *Mind*"** with the same ISBN. _Which subtitle is canonical
  needs verification against the physical title page before it is cited in the
  thesis_ — same book, same ISBN, conflicting subtitle in secondary sources.

## Core thesis (per Buskell)

Clark calls his framework **"hierarchical action-oriented predictive
processing" (PP)** — a "mid-level organizational sketch" (p. 2) of the
nervous system. The brain "is in the business of acquiring and tuning a
**model that generates predictions about the temporally fluctuating distal
causes of sensory stimulation**." It inverts the classical bottom-up
"detection → enrichment → rich internal representation" picture:

- The nervous system realises a **generative model** that "produces hypotheses
  about the **spatiotemporal structure of sensory input**."
- "Higher levels pass predictions down to lower levels, and lower levels pass
  **prediction errors** back up to higher levels" (roots in Friston, Hinton,
  Knill, Pouget).
- Perception = the hypotheses produced by the generative model, inferred
  on-the-fly via Bayesian inference, *not* a mirror of the world.

### The two load-bearing claims for the thesis

1. **Prediction is forward in time, over trajectories.** "Movement up the
   representational hierarchy brings with it **increasing temporal and spatial
   scope** over sensory stimuli. At the higher levels … representations consist
   in **probability distributions over temporally extended trajectories of
   sensory stimulation**" (Buskell, citing pp. 111, 158, 184). This is the
   direct grounding for "predict modalities *into the future*."
2. **Action is part of the predictive machinery.** Ch. 4 covers "the
   integration of PP with action, planning, and the control of behaviour"; Ch.
   5 covers proprioception, mirror neurons, and "off-line rehearsal of action."
   The "action-oriented" in the framework's name is the bridge from prediction
   to control. (The stronger Friston "active inference" reading — *action acts
   to fulfil predictions / suppress sensory prediction error* — is in the NDPR
   review, see below.)

### Precision-weighting (≈ attention / uncertainty)

Ch. 2: "we are constantly engaged in attempts to **predict precision**, that
is, to predict the context-varying reliability of our own sensory prediction
error" (p. 58). Precision-weighting "modulates the influence of top-down
priors relative to bottom-up stimuli." → a per-signal, context-dependent
reliability weight on prediction error. (Conceptual cousin of the per-modality
loss weighting `w_m` open sub-decision in the multimodal note.)

## Criticisms to be aware of (so the thesis doesn't over-claim the analogy)

- **Buskell:** the "hierarchy" gets *three inconsistent characterisations* —
  increasing spatiotemporal scope (pp. 111/158/184) vs. "gists"/context (Ch. 5,
  p. 166) vs. "hyperpriors"/near-Kantian invariants (pp. 174–5, 188). These are
  "very distinct kinds of models." PP is presented as an **inference to the
  best explanation / agenda-setter**, a themata-level commitment, "not a
  knock-down argument."
- **Rescorla (NDPR):** predictive coding has produced "comparatively few
  successful models" of core perceptual phenomena vs. non-PP alternatives; and
  active inference is "much less explanatory than OFC [Optimal Feedback
  Control]" because it "simply assume[s] a prior over bodily trajectories" —
  it *presupposes* the expected trajectory rather than explaining it
  (Bernstein's redundancy problem).

> **Implication for us:** PP is a *motivation* and a *narrative frame*, not
> evidence. Use it in the introduction/discussion to explain *why* coupled
> multimodal forward prediction + action is a natural objective; do **not**
> present it as justification that the method *works* (hard rule 7/8). The
> empirical case still has to come from real runs.

## Bridge to the ML thesis (analysed framing — the user's argument, not the book's)

- **Inputs:** (a) PP's "generative model predicting temporally-extended
  trajectories of multimodal sensory input" (sourced above); (b) our object =
  a frozen generative prior + output adapters predicting coupled future
  modalities `(x^video, x^prop, x^depth, …)`
  ([[../../50_Decisions/open/multimodal-adapter-broadening]]); (c) the
  "action-oriented" reading → the multimodal world model feeds policies /
  control (success criterion 4 in the decision note).
- **Reasoning:** the multimodal coupled-dynamics adapter is, structurally, a
  PP-style generative model: it predicts multiple sensory streams forward over
  a horizon, and the same predictive object is used for action. The
  per-modality precision-weighting in PP maps onto the open `w_m` /
  uncertainty-weighting sub-decision; the learned mask `m` over
  {base, action, modality₁…ₙ} is a (very loose) analogue of precision-routing.
- **Output (label: analysed framing):** PP gives the thesis a principled
  *why* for predicting many modalities into the future and coupling them to
  action — a strong introduction/motivation narrative. It does **not** supply
  any architectural detail or empirical support; the criticisms above (esp.
  Rescorla on active inference "presupposing" the trajectory) are exactly the
  objection an advisor could raise if PP is leaned on too hard.

## Related

- [[../../50_Decisions/open/multimodal-adapter-broadening]] — the direction
  this motivates.
- [[unified-world-models]] — the ML instantiation closest to "one generative
  model over all modalities + timesteps" (independent per-modality timesteps).
- [[avid]] — the frozen-base + output-adapter mechanism we predict modalities
  through.
