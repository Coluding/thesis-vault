---
last_updated: 2026-07-26
status: living
---

# Thesis Outline

> Full chapter/section breakdown with per-section status and the vault
> sources each section draws from. The `/thesis-write` skill reads this to
> know what to write and where the material lives. Keep section status
> current: `stub` → `drafting` → `draft-complete` → `revised`.

Status legend: **stub** (placeholder only) · **drafting** (partial prose) ·
**draft-complete** (full rough pass) · **revised** (edited at least once).

> **Narrative spine (settled 2026-07-26):**
> [[../30_Knowledge/writing/thesis-storyline]]. The arc is chronological —
> each step's limitation forces the next — but the **chapters stay
> deliverable-separated** (D1–D4). The arc is told in Ch1; the later chapters
> merely *order* themselves along it. Two rules this imposes:
> - **AVID is the working starting point, not a gap we fill.** Action
>   conditioning on a frozen base is AVID's contribution — see the D2
>   correction in [[../10_now/positioning]].
> - **What fails on Wan is D2 (base-parity), not D3 (shortcut).** Say so
>   plainly rather than letting the narrative imply the shortcut objective
>   failed there. **D4** is gated on a working D2 cell — **D3 is not**: the
>   action-free shortcut run tests it with actions stripped entirely, so D3
>   evidence can be gathered in parallel with the D2 ablation.

---

## 0. Abstract — `draft/00-abstract.md`
- Status: **stub** · Deliverable: — · _Write last._

## 1. Introduction — `draft/10-introduction.md`
- 1.1 Motivation: pretrained generative models as world-model priors — **stub** — src: [[../10_now/positioning]]
- 1.2 Problem: action-conditioning + fast rollout without retraining — **stub**
- 1.3 **The arc** — AVID works → planning → too slow → shortcut → curvature bias → flow/Wan → base-parity → the ablation → the boundary. **This is where the storyline is told end-to-end.** — **stub** — src: [[../30_Knowledge/writing/thesis-storyline]]
- 1.4 Contributions (D1–D4) — **stub** — src: [[../10_now/positioning]] (note the corrected D2 delta)
- 1.5 Thesis structure — **stub**

## 2. Related Work — `draft/20-related-work.md`
- 2.1 Adapters / PEFT — **pointer only**; the substance moved to Ch3 — src: `related-work/`
- 2.2 Action-conditioned video / world models (AVID, UniCon) — **stub** — src: `related-work/avid`, `related-work/unicon`
- 2.3 Few-step generation (shortcut models, consistency, distillation, DPM-solver) — **stub** — src: `related-work/shortcut-models`, `consistency-models`, `self-distillation`
- 2.4 Diffusion vs. flow matching (param/prediction types) — **stub** — src: `theory/`

## 3. Adapters for diffusion & flow-matching models (D1) — `draft/25-adapters.md`

**NEW 2026-07-26.** Deliberately **brief (~6–8 pp)**. Its job is to derive
**one decision** — the AVID-style output adapter — and hand off; everything
downstream builds on AVID. Anything not serving that decision belongs in §2.1.
Scope matches [[../30_Knowledge/writing/ablation-axes]] Axis 2: **D1 is a
software + complexity-analysis contribution**, so this chapter carries the
adapter-taxonomy claim while the Ch5 ablation carries only the
backbone/dataset-generality claim.

- 3.1 The families, one paragraph each + figure (LoRA · ControlNet · hidden-state/UniCon · hypernetwork/HyperAlign · output/AVID) — **stub** — src: `related-work/` (**needs `controlnet.md` + `lora.md` — do not exist yet**)
- 3.2 Two selection criteria — **stub**
  - (a) **Weight/internals access required?** Only the output adapter survives frozen-base-with-output-access-only. AVID's own argument (SOTA video models are closed-weight).
  - (b) **Gradients through the frozen base required?** Output adapters train detached; interior families cannot — src: [[../30_Knowledge/theory/interior-vs-output-adapters-backward-cost]], [[../30_Knowledge/theory/unicon-output-adapters-detached-backward]]
- 3.3 Measured cost table (the "small experiments") — **stub** — params · FLOPs/step · peak VRAM · wall-clock/step, ± backprop through base, same base+task. Cost only — **no quality comparison** (underpowered; ruled out by Axis 2). src: [[../50_Decisions/decided/param-matched-adapter-comparison-definition]], [[../30_Knowledge/experiments/protocol-param-matched-adapter-comparison]]
- 3.4 The choice: AVID-style output adapter — **stub**
  - Must answer explicitly: *if output adapters win on principle, why does the framework implement four families?* (Answer: so the selection is a demonstrated choice; the cost analysis rules the rest out.)
  - Must handle head-on: a `d`-dependent **weight update is definitionally a hypernetwork**, so the hypernetwork is the natural D3 family. We choose conditioning-as-input on cost grounds — receipt is §3.2(b). Do not let the chapter argue against D3.

## 4. Method (D1 — Framework) — `draft/30-method.md`
- 4.1 Composition interface `f_base + g(d)·Δ_φ` — **stub** — src: [[../10_now/architecture]]
- 4.2 Adapter taxonomy + shared conditioning path — **stub** — src: [[../30_Knowledge/tech/structural-encoder]] (taxonomy *argument* lives in Ch3; this is the implementation)
- 4.3 Conditioning (action, step-size; the `fs` boundary) — **drafting** (fs boundary subsection written 2026-05-28) — src: [[../30_Knowledge/tech/frame-stride-conditioning]], [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
- 4.4 Shortcut training modes — **drafting** (target-construction subsection written 2026-05-28) — src: [[../30_Knowledge/tech/shortcut-training-modes]], [[../50_Decisions/decided/shortcut-anchor-schedule]]
- 4.5 **The curvature bias in the shortcut target (D3)** — **stub, high priority** — the *proven* node of the arc: eq. (4) averages velocities, exact for straight interpolants, biased by the sagitta on a VP arc; the true field is not a fixed point, so the bias compounds up the doubling tower rather than training away. Two escapes: endpoint inversion (coordinates) or flow matching (geometry). **Phrase precisely** — "shortcut modelling fails on diffusion" is false and attackable. src: [[../30_Knowledge/theory/shortcut-v-averaging-bias]], [[../30_Knowledge/theory/heun-shortcut-target]], [[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]
- 4.6 Computational profile of the composition (frozen-base-dominated step; latent precompute as a throughput decision) — **drafting** (written 2026-07-24, WAN-2.2 sourced) — src: [[../60_Updates/entries/2026-07-24-online-vae-encode-6x-training-step]]

## 5. Experiments (D2 / D3) — `draft/40-experiments.md`
- 5.1 Datasets and preprocessing — **drafting** (updated 2026-07-24) — two dataset families (MetaWorld redundant anchor + ACWM-Phys informative) × base-backbone axis
  - 5.1.1 MetaWorld windowing + `fs` anchor (DynamiCrafter-specific) — **drafting** — src: [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
  - 5.1.2 ACWM-Phys (action-informative; Push Cube/Robot Arm/Reacher) — **drafting** — src: [[../50_Decisions/open/second-dataset-action-informativeness]]
  - 5.1.3 Wan2.2 diffusion-forcing preprocessing (768², 65f, primary base) — **drafting**
  - 5.1.4 Base backbones / base-strength axis (Wan2.2-5B · SkyReels-1.3B · DynamiCrafter) — **drafting** — src: [[../30_Knowledge/writing/ablation-axes]]
- 5.2 Protocol: param/FLOPs-matched comparison — **stub** — src: [[../50_Decisions/decided/param-matched-adapter-comparison-definition]]
- 5.3 Metrics + baselines — **stub** — **adopt AVID's Action Error Ratio** (§4.2 of the paper: action-predictor error on generated ÷ on real) as the primary action-sensitivity readout, so the D2 table carries an AVID-replica row against their baselines (ControlNet, ControlNet-Small, action-conditioned-from-scratch, Product-of-Experts, action-CFG) — src: [[../30_Knowledge/related-work/avid]]
- 5.4 Ablation design — **hypothesis-first, not axis-first** — **drafting** — open with the candidate explanations for the D2 collapse, then the axis that discriminates each; every row ends in a mechanism claim. src: [[../30_Knowledge/writing/ablation-axes]], [[../30_Knowledge/writing/thesis-storyline]] §7
- _Planned runs are tickets, not results — keep this section's claims to the protocol until runs land._

## 6. Results — `draft/50-results.md`

_Ordered along the arc; evidence stays deliverable-separated._

- 6.1 AVID reproduction — the working starting point (D2) — **stub** — src: [[../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]] (wandb `pg3x72uc`). **Gate:** this run shows convergence + gate health, *not* action-sensitivity — needs an action-sensitivity eval before §6.2 can stand.
- 6.2 Planning on the AVID world model (D2) — **stub** — explicitly labelled a **sanity-check demonstration, not a contribution** ([[../10_now/positioning]] anti-positioning). Framing: AVID names planning as its own future work.
- 6.3 Rollout cost — why few-step (motivation, not a result) — **stub** — needs sourced NFE × wall-clock; honest baseline includes DPM-Solver, not only 50-step DDIM.
- 6.4 Shortcut on DynamiCrafter — bounded gain (D3) — **stub** — **no run backs this yet**; if none lands, §4.5's theory carries the step. src: [[../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]
- 6.5 Shortcut adapter, **action-free** (D3) — **stub** — the pure D3 test: step-size conditioning only, actions stripped, so "does the shortcut work" is not confounded with "does action conditioning work". **This section is NOT gated on D2** — src: [[../30_Knowledge/writing/thesis-storyline]] §6 correction, [[../20_Tickets/experiments/exp-shortcut-action-free-isolation]], [[../20_Tickets/experiments/exp-shortcut-target-ab-actionfree]]
- 6.6 Diagnostic — base-parity collapse on Wan & the traps (**D2**) — **drafting** — src: [[../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]] (wandb: uxrst2k5, o79ki0ul, o9113j4h*, rxzwh4ak, hvxlbfjx). `[[FIG:hvxlbfjx-eval_step_grid]]` pending export. **Label the deliverable explicitly** — the narrative arrives here via the shortcut argument, so state that this is a D2 failure and that **D3 on Wan is gated on a working D2 cell**; the `anchor_prob: 1.0` no-shortcut control is what proves the consistency loss is not at fault.
- 6.7 Combined action+shortcut (D4) — **stub** — likely descoped; state the minimum viable D4 or descope explicitly.
- 6.8 The ablation — locating the boundary — **stub** — src: [[../30_Knowledge/writing/ablation-axes]]. **Needs a positive control** (a cell where the adapter provably learns something the frozen base cannot produce), or "everything base-cloned" is indistinguishable from a pipeline bug.
- _Every number here cites a run (wandb id + ckpt + commit). No exceptions._

## 7. Discussion — `draft/60-discussion.md`
- 7.1 **The boundary condition** — adapter-based action conditioning works when the base is weak enough relative to how much the data rewards actions; base-strength × action-informativeness map, with the mechanism named. **This is the ending — a claim, not an absence.** — **stub** — src: [[../30_Knowledge/writing/thesis-storyline]] §8
- 7.2 Trade-offs across adapter families — **stub** — cost/complexity analysis (Ch3), not a quality comparison
- 7.3 Limitations + open decisions — **stub** — src: `50_Decisions/open/*`; include the **Wan confound** (flow-matching *and* stronger base changed together) and how the ablation resolves it
- 7.4 Multimodal coupled-dynamics extension — **stub**

## 8. Conclusion — `draft/70-conclusion.md`
- 8.1 Summary of contributions — **stub**
- 8.2 Future work — **stub**
