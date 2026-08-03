---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: experimental-evaluation
category: research
current_band: "7-8"
target_band: "8-9"
sources:
  - "[[_index]]"
  - "[[../ablation-axes]]"
  - "[[../thesis-storyline]]"
  - "[[../../experiments/_index]]"
---

# Rubric 3 — Experimental evaluation

## The rows

| | |
|---|---|
| 10 | Precisely the right experiments with perfect execution |
| **9** | **Set up or modify an experiment exactly tailored to answering the research questions. Quantitative consideration of sources of error and uncertainty. Execution nearly flawless** |
| **8** | **Judge the setup of an existing experiment and include modifications if needed. Considers sources of error and uncertainty quantitatively** |
| 7 | Execute an experiment designed by someone else. Error/uncertainty qualitatively |
| 6 | Execute an experiment designed by someone else, without critical assessment |
| 1–5 | Errors are made in the process, invalidating (part of) the experiment |

## What it actually asks

Two independent things, and we are strong on one and exposed on the other:

1. **Design** — is the experiment *tailored to the research question*?
   (8: judge and modify someone else's; 9: design one exactly tailored.)
2. **Uncertainty** — treated **quantitatively** (8 and 9) rather than
   qualitatively (7) or not at all (6).

⚠ **The 1–5 row is the only failure row that could plausibly touch this
thesis**: "errors are made in the process, invalidating (part of) the
experiment." Parts of our campaign *were* invalidated — the retracted
SkyReels cell, the in-sample evaluations, the frozen-gate bug. **Whether
that reads as a 5 or as an 8 depends entirely on presentation**, because
the 8-row's "judge the setup and include modifications if needed" is a
description of *self-detected and corrected* errors. See "The integrity
strategy" below — this is the highest-stakes writing decision in the thesis.

## The axis inventory — what we actually ran

The design space is large. Reconciled from [[../ablation-axes]] (which is
**stale**, predating the whole 07-30 → 08-02 campaign) plus the campaign
itself. **This table is the raw material for Q2.**

| # | Axis | Values exercised | Hypothesis it discriminates |
|---|---|---|---|
| 1 | **Dataset** | MetaWorld · ACWM {Push Cube, Robot Arm, Reacher} · RT-1 · OpenVid | the data does not reward actions |
| 2 | **Frozen backbone** | Wan2.2-5B · SkyReels-1.3B · DynamiCrafter | the base is too strong → copy-through pull |
| 3 | **Adapter family** | output adapter · **LoRA (in flight)** | the family is wrong / PEFT would do better |
| 4 | **Adapter size** | 7.5M simple · 34.97M · 47M | insufficient capacity |
| 5 | **Action injection** | cross-attention · **per-frame AdaLN** | the pathway is wrong |
| 6 | **Composition** | `mask_mix` · `replace` | the composition traps the gate |
| 7 | **Base-prediction injection** | `condition_on_base_outputs` on/off | oracle-reading shortcut |
| 8 | **Gate bias / cap** | `gate_bias` · `gate_cap` {off, 0.9} | gate saturation |
| 9 | **Gate pretraining** | AVID warmup `pretrain_steps` {0, N} | pred head incompetent when the gate becomes learnable |
| 10 | **Action scale calibration** | `action_token_norm` · `condition_center` | the signal arrives too quiet / on a learned pedestal |
| 11 | **Action token binning** | `action_seq_len`, enforced px→latent | temporal misalignment of tokens to latent frames |
| 12 | **Noise schedule** | σ-shift {off, 5.0} | actions only carry signal at high σ |
| 13 | **Shortcut target × geometry** | `v_average` · `endpoint_inversion` × flow · diffusion | the curvature bias (D3) |

**Thirteen axes is coverage, and coverage is a liability if presented
axis-first.** [[../thesis-storyline]] §7 already names the two readings:
*"we varied model, dataset, adapter size…"* reads as flailing (and invites
the 1–5 row); *"the failure has N candidate explanations, here is the
design that discriminates each"* reads as science.

## The hypothesis-first regrouping (the presentation that scores)

Group the 13 axes under the explanations they kill. Every row ends in a
**mechanism claim**, not a result.

| Candidate explanation | Discriminating axes | Verdict |
|---|---|---|
| **Our data is too hard / not informative** | 1 | **Killed** — the unmodified AVID recipe follows actions on *our* ACWM Robot Arm where our adapters were blind: same frozen weights, same data, same probe |
| **The base is too strong; the adapter clones it** | 2, 7 | **Refined, not confirmed** — ~87% of the pred–base cosine is shared-target convergence present with *no* base input; high cosine ≠ copying. The removable part is oracle-reading |
| **Optimisation traps (gate)** | 6, 8, 9 | **Insufficient alone** — the traps are real and fixable, and fixing them did not unlock action use |
| **Insufficient capacity / wrong family** | 3, 4 | **Killed for capacity** — a structurally clean 7.5M adapter settles *below* the DiT-clone arms; the DiT inductive bias is worth 3–4×. **Family: LoRA comparison in flight** |
| **Signal arrives mis-scaled** | 10 | **Confirmed, two opposite failures at the same interface** — DC's learned pedestal, Wan's 250×-too-quiet stream. Both are scale calibration |
| **Tokens are temporally unaligned** | 11 | **Confirmed as a defect, insufficient as a cause** — binning alone buys nothing |
| **The pathway itself is wrong** | 5 | **✅ The answer** — per-frame AdaLN is causal at matched contribution and matched mask; cross-attention fails at every scale and alignment tested |
| **The objective does not pay for actions** | (analysis, not an axis) | **Confirmed as the bound** — actions explain ~0.45% of the teacher-forced denoising loss |
| **The consistency loss is at fault (D3 confusion)** | 13 + `anchor_prob: 1.0` control | **Killed** — the D2 failure is measured on runs with no consistency term |

That table is the experimental chapter's spine. It is also the single
strongest artifact this item has, and it does not exist in the draft yet.

## Uncertainty — where we stand

**Have (quantitative):** Welch t-tests (3.30 → 7.3 → 10.5), bootstrap CIs
reported as disjoint intervals, a measured noise floor (0.000245), verified
nulls (`base_null_violation` = 0), chance levels stated for every structure
probe (0.200 temporal, uniform spatial), pre-registered thresholds (the
NOBASE erosion prediction, marked ✓).

**Missing:** seed variance (all single-seed — batch CIs are a different
source); multiple-comparison exposure across 13 axes; and for the
in-flight LoRA comparison, a pre-registered decision rule *before* it
lands.

## The integrity strategy — the decisive call

Every invalidation in the campaign was **self-detected**:

| What was wrong | How it surfaced | What it invalidated |
|---|---|---|
| `dataset_size=76` — a silent 98.5% episode drop | config audit | the whole SkyReels cell: the 0.0450, the "91% of reference", the 35× |
| Evaluation was in-sample (`--data-dir` = `--eval-data-dir`) | trainer read-through | all RT-1 / OpenVid numbers |
| Frozen-gate bug (`gate_cap` = init) | gate telemetry | the D3 curvature comparison (68× superseded) |
| `effect_rel` monotone in gain | our own analysis | the mechanism-fix claims, until the temporal-control probe |
| Cross-base shortcut A/B varies target *and* depth | design review | the cross-base reading — **still unresolved**; within-arm stands |

**Write these, prominently, as a methods-integrity section.** The reasoning:
a reader who sees you caught your own in-sample evaluation stops hunting
for the one you missed. Concealment is the only path from here to the 1–5
row; disclosure is the path to 8. Use the five-part negative-result shape
([[../thesis-style-guide]] §5) for each.

## Optimisation queue

- [ ] **Q2 — Rewrite [[../ablation-axes]]** around the 13-axis inventory
      and the hypothesis-first regrouping. The current note predates the
      campaign, holds LoRA as "not run", and still frames the study as a
      search for a working cell. *(~2 h; unblocks Ch5.)*
- [ ] **Q4 — Methods-integrity inventory**, one row per invalidation:
      detection → scope of damage → what was re-run → what remains void.
- [ ] **Pre-register the LoRA comparison's decision rule now**, before it
      lands: what result would mean "the adapter is better", at what
      matched budget, on which metric. Pre-registration is worth more to
      this item than the outcome.
- [ ] **State the cross-base shortcut confound in the same paragraph as
      the 9×.** It is the one unresolved design flaw; unflagged, it takes
      the D3 claim with it.
- [ ] Q8 (seeds) and Q9 (Action Error Ratio as an external readout) — see
      [[02-technical-skills]] and [[06-literature]].
- [ ] **Report the axis count and the comparison count honestly**, with
      which claims were pre-registered and which were exploratory.

## Where it lands in the thesis

- Ch5 §5.4 — the hypothesis-first ablation design (the spine table above)
- Ch5 — metrics, chance levels, nulls, uncertainty treatment
- Ch5 or Appendix — the methods-integrity section
- Ch6 — each result reported against its control, not standalone
