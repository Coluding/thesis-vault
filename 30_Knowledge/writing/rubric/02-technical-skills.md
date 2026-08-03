---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: technical-skills
category: research
current_band: "8-9"
target_band: "9"
sources:
  - "[[_index]]"
  - "[[../../experiments/_index]]"
  - "[[../../experiments/20260731-wan-action-trace-value-pathway-drowns]]"
  - "[[../../experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]"
  - "[[../../experiments/20260731-wan-action-signal-is-a-global-bag]]"
---

# Rubric 2 — Technical skills

## The rows

| | |
|---|---|
| 10 | Complete control of the data; high-level analyses that took the committee by surprise |
| **9** | **Organize the data, perform thorough checks and perform advanced and original analyses** |
| 8 | Organize the data, perform commonly used checks and some advanced analyses |
| 7 | Major modifications to an existing tool or model, based on literature. Basic validation |
| 6 | Minor modifications. Superficial validation or none |

## What it actually asks

Note the ladder: 7 is about *modifying a tool*, 8–10 are about *analysing
data*. The rubric stops rewarding engineering at 7. Everything above is
about what you did with the measurements — which means **the codebase is
not what scores here; the probe campaign is.**

"Original analyses" (9) is the operative phrase. Standard analyses = FID,
loss curves, ablation tables. Original = instruments you built because
nothing existing answered the question.

## Evidence inventory — this is our strongest item

All sourced in [[../../experiments/_index]].

**Original instruments** (this *is* the 9-row):

| Probe | What it measures | Null / chance |
|---|---|---|
| `effect_rel` | relative response of the prediction to action perturbation | `base_null_violation` = 0 |
| `--emb-scale` sweep | whether the failure is a scale mismatch | flat ⇒ hypothesis dead |
| 23-depth propagation trace | *where* in the network the action signal dies | per-block action-driven share |
| Jacobian sensitivity | prediction sensitivity to base-pred vs to actions | ratio (measured ~100:1) |
| Rollout-action-swap | does action identity change the rollout? | true vs wrong vs zero actions |
| Structure triad — steering cosine | is the action→effect map directional? | cos ≈ 0 = arbitrary |
| Structure triad — temporal alignment | px→latent frame correspondence | chance 0.200 |
| Structure triad — spatial concentration | is the effect localised? | ≈ chance = uniform |

**Thorough checks** (the 9-row's other half):

- **Nulls verified, not assumed** — `base_null_violation` exactly 0
  throughout; frozen-base invariance checked *before every run*.
- **Matched controls on every headline** — clean-room A/B holds adapter
  contribution (0.114 vs 0.111) and mask (0.953) fixed so only the pathway
  varies; DC parity arms measured against a noise floor of 0.000245.
- **Statistics** — Welch t across depths (3.30 → 7.3 → 10.5), bootstrap CIs
  reported as disjoint intervals (0.302 [0.251,0.356] vs 0.034
  [0.026,0.042]).
- **Confound discrimination** — `effect_rel`'s gain-vs-information
  ambiguity identified *by us*, then settled by a temporal-control probe
  (diagonal concentration 0.390 vs 0.199, chance 0.200; the pooled arm's
  per-frame response rows are bit-identical).

**Root-causing to the tensor level** — the kind of detail that reads as
control of the data:

- DC pedestal: the action embedding grows **106×** during training into a
  **99.7%-constant** vector, **14×** the time embedding, carrying 0.5%
  action-driven variance vs the reference's 24% — and the pedestal is
  **learned**, not initialised (at init 0.182×, 2.4% varying).
- Wan drowning: cross-attention output RMS ~0.01 against a residual stream
  of 1.8–3.0; relative sensitivity falls 0.44 → 0.0085 across a **single
  addition**; the value pathway is unnormalised while `qk_norm` rescues
  only the logits.

## Gaps to 9

1. **Seed variance is unaddressed.** Every headline is single-seed. The CIs
   are over *evaluation batches* — a different source of uncertainty than
   training-run variance, and a committee will know the difference. This is
   the single concrete thing standing between the evidence and a clean 9.
2. **The probe suite is undocumented as an instrument.** It exists as
   scripts and scattered numbers. "Original analyses" only scores if the
   reader can see the instrument, its null, and its validation.
3. **Data organisation is not written anywhere.** The rubric says "organize
   the data" in both the 8 and 9 rows. We have 25+ runs across many axes
   and a ledger — that *is* organisation, but it must appear in the thesis
   (a run inventory / appendix table), not only in the vault.

## Optimisation queue

- [ ] **Q3 — Probe-suite instrument spec.** One subsection: for each probe,
      what it measures, its null, its chance level, its cost, and **the
      case where it misled us** (the `effect_rel` gain confound and how the
      temporal-control probe resolved it). *Validating your own instrument
      is exactly what "advanced and original analyses" means.* (~2 h)
- [ ] **Q8 — Two extra seeds on the headline clean-room A/B.** Converts
      "advanced analysis" into "thorough checks" in the rubric's sense.
      GPU-cheap relative to its return; do it on the *one* A/B that carries
      the pathway claim, not everywhere.
- [ ] **Run inventory table for the appendix** — every run, base × dataset ×
      axis, wandb id, steps, outcome, and whether it is cited. Derived
      mechanically from [[../../experiments/_index]]. Doubles as the
      "organize the data" evidence and as an honesty signal (the killed and
      retracted runs are *in* it).
- [ ] **Report the multiple-comparison exposure.** With this many axes
      (backbone, dataset, adapter size, injection, composition, base-pred
      injection, gate bias, gate pretraining, adapter family) and single
      seeds, some cells will look significant by chance. State how many
      comparisons were made and which claims are pre-registered vs
      exploratory. Pre-empting this is a 9-row move; being caught by it is
      a 6-row one.

## Where it lands in the thesis

- Ch5 methods — the probe suite as an instrument section (its own §)
- Ch5 — data organisation, splits, and the held-out discipline
- Appendix — run inventory
- Ch6 — the tensor-level root-cause figures carry this item visually
