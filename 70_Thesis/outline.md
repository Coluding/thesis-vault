---
last_updated: 2026-05-28
status: living
---

# Thesis Outline

> Full chapter/section breakdown with per-section status and the vault
> sources each section draws from. The `/thesis-write` skill reads this to
> know what to write and where the material lives. Keep section status
> current: `stub` → `drafting` → `draft-complete` → `revised`.

Status legend: **stub** (placeholder only) · **drafting** (partial prose) ·
**draft-complete** (full rough pass) · **revised** (edited at least once).

---

## 0. Abstract — `draft/00-abstract.md`
- Status: **stub** · Deliverable: — · _Write last._

## 1. Introduction — `draft/10-introduction.md`
- 1.1 Motivation: pretrained generative models as world-model priors — **stub** — src: [[../10_now/positioning]]
- 1.2 Problem: action-conditioning + fast rollout without retraining — **stub**
- 1.3 Contributions (D1–D4) — **stub** — src: [[../10_now/positioning]]
- 1.4 Thesis structure — **stub**

## 2. Related Work — `draft/20-related-work.md`
- 2.1 Adapters / PEFT (LoRA, hidden-state, output, hypernetwork) — **stub** — src: `related-work/`, [[../30_Knowledge/related-work/hyperalign]]
- 2.2 Action-conditioned video / world models (AVID, UniCon) — **stub** — src: `related-work/avid`, `related-work/unicon`
- 2.3 Few-step generation (shortcut models, consistency, distillation, DPM-solver) — **stub** — src: `related-work/shortcut-models`, `consistency-models`, `self-distillation`
- 2.4 Diffusion vs. flow matching (param/prediction types) — **stub** — src: `theory/`

## 3. Method (D1 — Framework) — `draft/30-method.md`
- 3.1 Composition interface `f_base + g(d)·Δ_φ` — **stub** — src: [[../10_now/architecture]]
- 3.2 Adapter taxonomy + shared conditioning path — **stub** — src: [[../30_Knowledge/tech/structural-encoder]]
- 3.3 Conditioning (action, step-size; the `fs` boundary) — **drafting** (fs boundary subsection written 2026-05-28) — src: [[../30_Knowledge/tech/frame-stride-conditioning]], [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
- 3.4 Shortcut training modes — **drafting** (target-construction subsection written 2026-05-28) — src: [[../30_Knowledge/tech/shortcut-training-modes]], [[../50_Decisions/decided/shortcut-anchor-schedule]]

## 4. Experiments (D2 / D3) — `draft/40-experiments.md`
- 4.4 Ablation design (the axes) — **drafting** — src: [[../30_Knowledge/writing/ablation-axes]] (dataset axis run-in-full × intervention search-toolbox; output-adapter only + complexity discussion for the rest)
- 4.1 Datasets (MetaWorld), preprocessing, action handling — **drafting** (MetaWorld windowing + `fs` anchor written 2026-05-28) — src: `30_Knowledge/datasets/`, [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
- 4.2 Protocol: param/FLOPs-matched comparison — **stub** — src: [[../50_Decisions/decided/param-matched-adapter-comparison-definition]]
- 4.3 Metrics + baselines — **stub**
- _Planned runs are tickets, not results — keep this section's claims to the protocol until runs land._

## 5. Results — `draft/50-results.md`
- 5.1 Adapter-family comparison (D2) — **stub** — src: `30_Knowledge/experiments/*` (sourced runs only)
- 5.1.x Diagnostic — base-parity collapse & the traps (D2) — **drafting** — src: [[../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]] (wandb: uxrst2k5, o79ki0ul, o9113j4h*, rxzwh4ak, hvxlbfjx). The negative result that motivates the dataset switch; `[[FIG:hvxlbfjx-eval_step_grid]]` pending export.
- 5.2 Shortcut few-step rollout (D3) — **stub**
- 5.3 Combined action+shortcut (D4) — **stub**
- _Every number here cites a run (wandb id + ckpt + commit). No exceptions._

## 6. Discussion — `draft/60-discussion.md`
- 6.1 Trade-offs across adapter families — **stub**
- 6.2 Limitations + open decisions — **stub** — src: `50_Decisions/open/*`
- 6.3 Multimodal coupled-dynamics extension — **stub**

## 7. Conclusion — `draft/70-conclusion.md`
- 7.1 Summary of contributions — **stub**
- 7.2 Future work — **stub**
