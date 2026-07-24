---
last_updated: 2026-05-15
status: living
---

# Thesis Framing & Contributions

> Current framing of the thesis: the four deliverables from the proposal,
> what each one's contribution actually is, what counts as evidence for
> it, and where the surface gets blurry. AI overwrites this when framing
> shifts. Past framings live in Git history. The proposal itself lives at
> `docs/thesis-plan/Updated_Thesis_Proposal.pdf`.
>
> This doc is the **internal-quick-reference cut**. When the thesis itself
> needs an external framing (e.g. abstract, advisor pitch, defence
> introduction), that artifact lives in `30_Knowledge/writing/` and
> cross-links to this doc.

## One-paragraph statement

Pretrained diffusion and flow-matching models are strong generative priors
but cannot be used directly as action-conditioned world models — they
lack action conditioning and require many denoising steps per rollout.
This thesis introduces a unified adapter framework that augments any frozen
pretrained model with an additive trainable correction
`Δ_φ(x_t, t, a_t, d)`, makes that correction action-conditioned, and adds
step-size conditioning trained with consistency objectives so the resulting
model can roll out trajectories in few steps. The result is a single
framework supporting four kinds of adapter (LoRA-style, hidden-state,
output-level, hypernetwork), spanning both diffusion and flow matching,
suitable for planning workloads.

## The four deliverables

The proposal commits to four deliverables. The contribution per
deliverable, the evidence required, and the current state are:

### D1 — Adapter framework (software contribution)

**What:** A modular, extensible Python library that adapts pretrained
diffusion / flow-matching backbones into action-conditioned models via
plug-and-play adapters covering the full taxonomy.

**Evidence required:**

- All four adapter families implemented behind a single `Adapter` interface.
- Both `model_type=diffusion` and `model_type=flow` working under one
  `BaseGenerativeModel` interface.
- At least one non-trivial backbone wrapped (beyond the `dummy` MLP).
- Configuration-driven; new adapters / backbones land as registry entries.
- Tests for each adapter family's shape and composition rule.

**Current state:** taxonomy is mapped 1-to-1 onto
[[architecture#Adapter families|the codebase]]. Output, hidden-state,
hypernetwork, and LoRA families are all present. Diffusers and DynamiCrafter
backbones are wired; OpenSora vendored but provider wiring _needs
verification_. Tests cover HyperAlign architecture, DynamiCrafter
integration, batch preprocessor, MetaWorld dataset. **Status: mostly
landed; needs a clean diagram for the thesis chapter + a final
documentation pass.**

### D2 — Action-conditioned world models (empirical contribution)

**What:** Use adapters to learn action-conditioned dynamics
`f(x_t, t, a_t)`. Analyse the trade-off between adapter classes on:

- Prediction accuracy
- Stability over long rollouts
- Inference cost (FLOPs / wall-clock / steps)

**Evidence required:**

- A common benchmark dataset (MetaWorld is the implemented choice — see
  `tests/test_metaworld_dataset.py` and `train_hyperalign_metaworld.py`).
- Each adapter family trained to convergence on the same task with the
  same backbone.
- A common evaluation protocol with at least one rollout metric and one
  trajectory-quality metric.
- An honest cost comparison (parameters trained, GPU-hours, inference cost
  per step).
- Plots / a table that lets the reader pick a Pareto-optimal adapter for a
  given budget.

**Current state:** `train_hyperalign_metaworld.py` exists as the closest
running entrypoint. _Needs verification_ which adapters have actually
finished a real training run end-to-end with logged outputs in wandb. The
comparative ablation across all four adapter families on the same task is
the headline deliverable and is **not yet evidence-backed**.

**Evidence strategy (decided 2026-07-21, grilling session):** the target D2
claim is **(a) "adapters can make a frozen video model action-following on
at least one benchmark"** — one clean positive result. Fallback if (a)
fails after the current intervention round: shift to **(b) the comparative
trade-off analysis** plus **(c) the diagnostic contribution** (base-parity
convergence, copy-through traps, action-blindness measurement — already
banked, see
[[../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]).
Context: the current Wan2.2 xattn adapter is measured fully action-blind on
MetaWorld scripted demos; whether the benchmark itself rewards actions is an
open dataset decision (ACWM-Phys candidate).

### D3 — Shortcut adapters (methods contribution)

**What:** Extend the adapter to take a step-size argument `d`:

```
f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)
```

Train with **local consistency** and **multi-step self-consistency**:

```
s(x_t, t, 2d, a_t) ≈ ½ s(x_t, t, d, a_t) + ½ s(x_{t+d}, t+d, d, a_t)
```

So the adapter can approximate finite-step transitions — reducing the
number of denoising steps required at rollout time.

**Evidence required:**

- A working step-size-conditioned adapter (hypernetwork OR output-level
  with step-size in the conditioning).
- Both `local_consistency` and `multistep_self_consistency` losses
  exercised in training.
- A few-step-rollout quality curve: x = number of steps, y = some quality
  metric, swept over `d` schedules.
- An honest comparison to standard 50-step DDIM/Euler at matched compute.
- Both `shortcut_target_method=linear` and `=two_step` characterised.

**Current state:** the necessary scaffolding exists:
[[architecture#Losses|consistency losses]] are implemented; configs
`flow_output_shortcut.yaml`, `flow_hyper_shortcut_stepwise.yaml`,
`diffusion_output_shortcut_*.yaml` exist; `test_hyper_step_size_conditioning.py`
covers the step-size-conditioned hyper. A first real-backbone shortcut run
has now landed (AVID/DynamiCrafter, `anchor_prob=0.45`, larger MetaWorld) —
but as a **cautionary** result, not a clean win: the honest `base_loss` is
stable while the **`shortcut_direction_loss` is volatile**, and prediction
**degraded** on the larger dataset (the earlier qualitative NFE-robustness
looks like small-data overfit). See
[[../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]].
**A clean few-step-rollout quality curve is still not in the vault**, and
diagnosing the shortcut-loss instability (per-step-size logging in progress)
is currently the highest-leverage shortcut work.

### D4 — Combined: action-conditioned shortcut world models (integration)

**What:** Combine D2 + D3 — action-conditioned dynamics *and* step-size
conditioning *and* consistency training in one model. The output is a
planner-suitable world model that does few-step, action-conditioned
trajectory rollouts at high quality.

**Evidence required:**

- One end-to-end run on MetaWorld (or chosen benchmark) where:
  - the adapter takes both `a_t` and `d`,
  - the loss includes both task loss and consistency,
  - rollout is performed in <10 steps,
  - quality is competitive with a 50-step non-shortcut baseline.
- A short demonstration that the model is usable as a planning surrogate
  (e.g. one task where rollouts feed into a controller).

**Current state:** D4 is **gated on D2 + D3 evidence existing first.** No
combined run yet. This is the thesis's punchline result and the chapter
that closes the story.

### Optional extension — multimodal coupled dynamics

`(x^{video}_{t+d}, x^{prop}_{t+d}) = f(x_t, t, a_t, d)`

Predict coupled video + proprioceptive transitions in a single model.
Architecturally supported via `ConditioningConfig.modalities`. **No
multimodal config or run lives in the vault yet** — opening
[[../50_Decisions/open/multimodal-scope]] is the gate for whether this
ships as a chapter or as a future-work paragraph.

## What this thesis is *not* (anti-positioning)

Anti-positioning matters for the related-work section. The thesis is **not**:

- **Not a new diffusion or flow-matching algorithm.** The base models are
  taken as given (DDPM/EDM/flow-matching). The contribution is the
  *adaptation layer*, not the prior.
- **Not a fine-tuning paper.** Fine-tuning unfreezes the base. Here the
  base stays frozen; only `Δ_φ` is trained. Closer to LoRA / adapter-tuning
  in NLP than to fine-tuning.
- **Not a pure consistency-models paper.** Consistency models retrain the
  prior from scratch to be few-step. Here the prior is frozen and the
  shortcut behaviour is bolted on via the adapter (D3). The relationship
  to consistency / shortcut models is "borrows the loss form, applies it
  in a frozen-base adapter setting" — see
  [[../30_Knowledge/related-work/consistency-models]] and
  [[../30_Knowledge/related-work/shortcut-models]].
- **Not a control / RL paper.** The output is a *world model usable for
  planning*. Demonstration via planning is a sanity check, not the
  contribution.
- **Not an architecture-search paper.** The four adapter families are
  fixed up-front by the taxonomy; the contribution is unifying them, not
  discovering new architectures.

## The category we're claiming

**Adapter-first world models.** Frozen pretrained generative prior + a
lightweight, action-conditioned, step-size-conditioned trainable adapter,
all under one composition rule. The closest neighbours in the
related-work neighbourhood:

- [[../30_Knowledge/related-work/avid|AVID]] — adapter-style residual for
  pretrained video diffusion. Closest in *form* to the framework, but
  output-level and not action / step-size-aware.
- [[../30_Knowledge/related-work/hyperalign|HyperAlign]] — hypernetwork
  that produces task-specific LoRA weights. Closest in *method* on the
  hypernetwork adapter side. Vendored as a starting point in the repo.
- [[../30_Knowledge/related-work/unicon|UniCon]] — hidden-state /
  skip-connection control adapter for diffusion. Closest in *form* on
  the hidden-state adapter side.
- [[../30_Knowledge/related-work/shortcut-models|Shortcut Models]] —
  source of the step-size-conditioned consistency formulation behind D3.
- [[../30_Knowledge/related-work/consistency-models|Consistency Models]]
  + [[../30_Knowledge/related-work/self-distillation|Self-Distillation]]
  + [[../30_Knowledge/related-work/dpm-solver|DPM-Solver]] — the
  few-step-sampling family that motivates D3 and against which the
  shortcut adapter should be honestly benchmarked.
- [[../30_Knowledge/related-work/cafm|CAFM]] — _slot for a paper title
  starting with "cafm" in `docs/paper/cafm.pdf`; needs verification of
  the exact title and venue_.

## Open framing questions

Things to grill on before each chapter is drafted:

- **Should the D2 ablation table report on MetaWorld only, or include a
  second domain?** Single-domain evidence is the weaker version of the
  claim; multi-domain is more work but a stronger contribution. — _needs
  decision_.
- **Should the D3 chapter benchmark against DPM-Solver / consistency
  models directly, or only against naïve 50-step sampling?** The honest
  comparison is the harder one and matters for related-work credibility.
  — _needs decision_.
- **Does D4 need a real planning task (e.g. MetaWorld MPC) or just a
  rollout-quality demonstration?** This is the integration chapter's
  scope question. — _needs decision_.
- **How prominently should the "freezing the base" claim be marketed?** It
  is the defining choice but also where reviewers will push hardest. —
  _needs framing decision_.

## Audience

- **Primary:** the advisor + thesis committee.
- **Secondary:** future collaborators reading the repo as a research tool.
- **Tertiary:** the open-source community if any part is released
  publicly. Not the primary deliverable.

## Related

- [[architecture]] — what's actually built (codebase state)
- [[product-state]] — which experiments have actually run
- [[setup-status]] — vault coverage gaps
- [[../30_Knowledge/related-work/_MOC]] — paper notes
- `docs/thesis-plan/Updated_Thesis_Proposal.pdf` — the proposal verbatim
