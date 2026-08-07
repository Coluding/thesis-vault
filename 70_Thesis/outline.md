---
last_updated: 2026-08-03
status: living
---

# Thesis Outline

> Per-section function, page budget, status and vault sources.
> `/thesis-write` reads this to know what to write and where the material
> lives. Keep status current: `stub` → `drafting` → `draft-complete` →
> `revised`.

Status legend: **stub** (placeholder only) · **drafting** (partial prose) ·
**draft-complete** (full rough pass) · **revised** (edited at least once).

> **⚠ RESTRUCTURED 2026-08-03 — the 40-page limit.** The faculty template
> (UvA MSc AI, `report`/12pt) sets a **40-page upper limit** with
> appendices as *additional* pages. The previous 8-chapter outline did not
> fit. **Signed off: 6 chapters**, budgets below, appendix as the pressure
> valve. Superseded: the standalone adapters chapter
> (`draft/25-adapters.md`, never created) and the split of Discussion from
> Conclusion.
>
> **⚠ LaTeX is the source of truth** — write into
> `70_Thesis/latex/chapters/*.tex`. The Markdown in `draft/` is historical;
> its real prose must be **ported, not rewritten**.
>
> **Governing documents:**
> [[../30_Knowledge/writing/thesis-style-guide]] (rhetoric + the claim
> ladder) · [[../30_Knowledge/writing/thesis-formal-rules]] (LaTeX,
> notation, provenance) · [[../30_Knowledge/writing/rubric/_index]] (what
> is graded, per item).

## The framing everything rests on (2026-08-02 reframe)

An **empirical study of when and why adapter-based adaptation works**, not
a systems build. The headline:

> **The method works — on DynamiCrafter.** `effect_rel` **0.11479** (3.9×
> the AVID reference), **all three structure probes above chance** on
> held-out `ind_test` (steering +0.117 vs 0.000; temporal 1.000 vs 0.313;
> spatial 0.470 vs 0.100), and the action-following solution is **better on
> the denoising objective itself** (0.0357 vs the control's 0.0433). The
> untreated control clears the AVID reference unaided (1.55×), so the cell
> works *natively* — `condition_center` accelerates escape (~6×) rather
> than enabling it.
>
> **On Wan the verdict depends on the injection pathway** — state it per
> cell, never per backbone. With **cross-attention** (× ACWM) it is a
> *domain corrector*: 6/6 quality metrics beaten (FVD 1118 → 406) with all
> three structure probes at chance. With **per-frame AdaLN** (the clean-room
> arm) the same backbone *follows actions*: 2.49× at step 12000 (Welch
> t=10.5), diagonal concentration 0.409 against chance 0.200.
>
> **The decisive contrast is within Wan**, not between backbones — same
> base, same data, matched adapter contribution and matched mask, only the
> pathway differs. That removes base strength as a confound, which the
> DC-vs-Wan comparison cannot. Per-frame modulation of normalised
> activations carries the action; cross-attention tokens survive the
> attention and drown at the residual add.
>
> **A positive result with a matched architectural negative control** — tell
> it in that order. Limits, per cell: rollout-level control is null on Wan;
> on DC there is **qualitative evidence, artefacts pending** — nothing on
> DC control enters the draft until they land
> ([[../20_Tickets/experiments/exp-eval-rollout-action-swap-dc-arme]]). No
> DC run logs quality metrics; the runs were cancelled pre-convergence, so
> quote the acceleration, never a level.

Never write "the approach does not work on Wan" — it contradicts our own
tables ([[../30_Knowledge/writing/rubric/_index]]).

**Two standing rules:**
- **AVID is the working starting point, not a gap we fill.** Action
  conditioning on a frozen base is AVID's contribution — see the D2
  correction in [[../10_now/positioning]].
- **The arc is told once, in Ch1.** Later chapters keep evidence
  deliverable-separated (D1–D4) and merely *order* themselves along it.

---

## Abstract — `main.tex`
- **stub** · _Write last._
- Must contain, in order: the question (*when and why*), the framework, the
  pathway result, and — explicitly — **the control limit**. The limit in
  the abstract is what makes the rest credible
  ([[../30_Knowledge/writing/rubric/05-reflection]]).

## 1. Introduction — `chapters/10-introduction.tex` · **4–5 pp**

_Function: pose the question, tell the arc end-to-end, state the
contributions._

- 1.1 Motivation: pretrained generative models as world-model priors — **draft-complete** (2026-08-07) — src: [[../10_now/positioning]]
- 1.2 Problem: **when and why** adaptation works, not *can it* — **draft-complete** (2026-08-07)
- 1.3 **The arc** — **draft-complete** (2026-08-07, ~7 paragraphs) — AVID as the starting point, the cost argument, the **three-reason** flow pivot with its confound named, the pathway result, the economics, and the pre-registered acceleration prediction. Told end-to-end here and nowhere else. src: [[../30_Knowledge/writing/thesis-storyline]]
- 1.4 Contributions — **drafting** (prose written 2026-08-03, ~1 pp) — the four-part delta vs AVID, **explicit**: (a) the conditions under which it works (the pathway principle, *with its scope of validity*); (b) the bound on why it stops (the objective economics); (c) step-size conditioning (D3, genuinely untouched by AVID); (d) the framework (D1, demonstrated by porting AVID's recipe to a new base family in AVID's own repo). src: [[../10_now/positioning]], [[../30_Knowledge/writing/rubric/01-originality]]
- 1.5 Thesis structure — **draft-complete** (2026-08-07)

## 2. Related work — `chapters/20-related-work.tex` · **5 pp**

_Function: make our question the obvious next one. Organised by **tension
in the field**, not by paper; every paragraph ends in a limitation or a
delta ([[../30_Knowledge/writing/thesis-style-guide]] §6)._

- 2.1 Adapting frozen generative models (PEFT lineage) — **stub** — 🛑 **blocked**: `related-work/lora.md` and `related-work/controlnet.md` do not exist
- 2.2 **Conditioning mechanisms** (FiLM · adaLN-Zero/DiT · cross-attention) — **stub** — 🛑 **blocked**: notes do not exist. **This section underwrites the pathway result** — it turns an empirical finding into a principled one
- 2.3 Action-conditioned video and world models — **stub** — src: `related-work/avid`, `unicon`, `unified-world-models`, `dreamzero-wam`. Place our frozen-base setting against retrain-from-scratch
- 2.4 Few-step generation — **stub** — src: `related-work/shortcut-models`, `consistency-models`, `self-distillation`, `dpm-solver`. The honest D3 baseline is not 50-step DDIM alone; open decision [[../50_Decisions/open/d3-positioning-vs-weaver-reflow]]
- Cite the **negative space** with its search scope stated

➜ Gaps and the full candidate list: [[../30_Knowledge/writing/rubric/06-literature]]

## 3. Method (D1, + D3 theory) — `chapters/30-method.tex` · **10 pp**

_Function: what the system **is**. No results, no motivation._

- 3.1 The composition interface `\composition` — **stub** — src: [[../10_now/architecture]]
- 3.2 **Adapter families and the selection** — **stub** — *(was a standalone chapter; merged for the page limit.)* Derive **one** decision — the AVID-style output adapter — via two criteria: (a) weight/internals access required? (b) gradients through the frozen base required? src: [[../30_Knowledge/theory/interior-vs-output-adapters-backward-cost]], [[../30_Knowledge/theory/unicon-output-adapters-detached-backward]]. Must answer head-on: *if output adapters win on principle, why implement four families?* — and note that a `d`-dependent weight update is definitionally a hypernetwork
- 3.3 Conditioning interfaces (action, step-size, the `fs` boundary) — **drafting** — ⬅ **port from [[draft/30-method]]** — src: [[../30_Knowledge/tech/frame-stride-conditioning]], [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
- 3.4 Shortcut training and target construction — **drafting** — ⬅ **port from [[draft/30-method]]** — src: [[../30_Knowledge/tech/shortcut-training-modes]], [[../50_Decisions/decided/shortcut-anchor-schedule]]
- 3.5 **The curvature bias (D3 theory)** — **drafting** (prose written 2026-08-07, ~2 pp) — the *proven* node: **derive** the sagitta argument, don't assert it. Eq. (4) averages velocities: exact for straight interpolants, biased on a VP arc; the true field is not a fixed point, so the bias compounds up the doubling tower. Two escapes: endpoint inversion (coordinates) or flow matching (geometry). **Phrase precisely** — "shortcut modelling fails on diffusion" is false and attackable. src: [[../30_Knowledge/theory/shortcut-v-averaging-bias]], [[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]
- 3.6 Computational profile of the composition — **drafting** — ⬅ **port from [[draft/30-method]]** (WAN-2.2 sourced)

## 4. Experiments — `chapters/40-experiments.tex` · **7 pp**

_Function: **how we would know**. Protocol, metrics, controls, thresholds.
No findings._

- 4.1 Datasets and preprocessing — **stub** — MetaWorld (action-**redundant** control) × ACWM (informative) × RT-1. **State the held-out discipline explicitly**
- 4.2 Backbones — **stub** — Wan2.2-5B (flow/DiT) · SkyReels-1.3B · DynamiCrafter (diffusion/UNet)
- 4.3 **The probe suite** — **drafting** (prose written 2026-08-03, ~2 pp) — the instrument section. ✅ **source written**: [[../30_Knowledge/tech/probe-suite]] (the ladder sensitivity→structure→control, all four sensitivity variants, the triad with its chance levels, the seven controls, and §6 the case where the instrument misled us). src also: [[../30_Knowledge/writing/rubric/02-technical-skills]]
- 4.4 Metrics and baselines — **stub** — **Action Error Ratio** (AVID §4.2) as the external readout so the D2 table carries an AVID-replica row; `effect_rel` + the structure triad internally; FID/FVD/LPIPS/PSNR for quality
- 4.5 **Ablation design — hypothesis-first, not axis-first** — **stub** — ✅ **source rewritten 2026-08-03**: [[../30_Knowledge/writing/ablation-axes]] now carries the 11 hypotheses with verdicts, the 13-axis inventory, the metric ladder, and the Wan/DC dissociation table. Open with the candidate explanations, then the axis that discriminates each; every row ends in a mechanism claim
- 4.6 **Methodological integrity** — **drafting** (prose written 2026-08-03, ~1 pp) — ✅ **source written**: [[../30_Knowledge/writing/methods-integrity]] (I1 in-sample eval · I2 the 98.5% silent drop · I3 the frozen gate · I4 the gain-vs-information confound · I5 the *unresolved* cross-base confound), each as expectation → detection → damage → correction → what stands, plus the practices adopted. *Concealment is the only path to the rubric's failing row.*
- _Planned runs are tickets, not results._

## 5. Results — `chapters/50-results.tex` · **10 pp**

_Function: what we found. Ordered along the arc; evidence stays
deliverable-separated. **Every number carries `\prov{run}{ckpt}{commit}`.**_

- 5.1 The framework across backbones (**D1**) — **stub** — three backbone families behind one interface + the AVID-repo port; **+ the LoRA-vs-output comparison** once it lands ([[../20_Tickets/experiments/exp-adapter-lora-vs-output-comparison]])
- 5.2 **A working action-conditioned cell (D2)** — **stub** — *the positive result; write it before the negative.* DC × ACWM: 3.9× the AVID reference, all three structure probes above chance on held-out `ind_test`, and a **lower denoising loss** than the untreated control. The control clears the reference unaided → the cell works natively; `condition_center` is a ~6× accelerator. **This is also the ablation's positive control.** Includes a **qualitative output-quality figure** (arm E `6oyu1inq`, arm F `86kb01su`, adapted vs frozen base) — the *only* evidence about DC output quality, since no DC run logs quality metrics. ⚠ **Label it qualitative and make no action claim from it**: adapted-vs-base cannot separate action-following from a better temporal prior — the Wan cell beats its base 6/6 while at chance on structure. Provenance `\nv{}` until artefacts land ([[../20_Tickets/experiments/exp-eval-rollout-action-swap-dc-arme]]). Other limits: cancelled pre-convergence; rollout control **not measured on either cell**. src: [[../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]], [[../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]
- 5.3 **The injection pathway decides (D2)** — **stub** — *the mechanism.* The Wan contrast (6/6 quality, structure at chance = a domain adapter) against §5.2, then the clean-room A/B at matched adapter contribution and matched mask, and the temporal control that resolves gain-vs-information. src: [[../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]]
- 5.4 Two scale failures at the same interface (**D2**) — **stub** — DC's learned pedestal; Wan's drowning at the residual add. src: [[../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]], [[../30_Knowledge/experiments/20260731-wan-action-trace-value-pathway-drowns]]
- 5.5 **What the adapter extracts from the action** — **drafting** (prose written 2026-08-07, ~1.5 pp) — the scalar reading — shared-target convergence vs oracle-reading vs the ~0.45% economics; the global-bag structure result. src: [[../30_Knowledge/experiments/20260731-why-wan-copies-the-base-decomposed]], [[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]
- 5.6 **Standard metrics are blind** — **drafting** (prose written 2026-08-03, ~1.2 pp) — perceptual metrics improving while all three structure probes sit at chance. The single best writing investment in the thesis. src: [[../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]], [[../30_Knowledge/writing/rubric/05-reflection]]
- 5.7 Shortcut on flow versus diffusion (**D3**) — **stub** — the first clean D3 positive. ⚠ **state the cross-base confound in the same paragraph as the 9×**, or the number takes the claim with it. src: [[../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]]
- **Descoped:** D4 (combined action + shortcut) — no run exists; state the descope explicitly rather than leaving a hole. Planning: pending artefacts ([[../30_Knowledge/writing/thesis-storyline]] §2)

## 6. Discussion and conclusion — `chapters/60-discussion.tex` · **4 pp**

_Function: what it means and where it breaks. (Merged for the page limit.)_

- 6.1 **The boundary** — **stub** — the generalised claim: adapting a frozen prior to a new conditioning variable is governed by *how the conditioning enters relative to the residual stream* and *what the objective pays for it* — neither visible in any standard readout. **Write it at that altitude**; this is the rubric-9 move for Knowledge and Originality both. src: [[../30_Knowledge/writing/thesis-storyline]] §8–9
- 6.2 Limitations — **stub** — concentrated here, **once**, thoroughly: control demonstrated nowhere; single seeds; the Wan confound (flow *and* stronger base changed together); the unresolved cross-base shortcut design; the in-sample quarantine
- 6.3 Future work — **stub** — an objective that pays for actions (action-CFG, rollout losses, action-conditional consistency) + the structural repairs. src: [[../50_Decisions/open/wan-action-following-needs-objective-change]]
- 6.4 Conclusion — **stub**

## Appendix — `chapters/90-appendix.tex` · **+N, outside the 40**

_The pressure valve: detail that would unbalance a chapter._

- A.1 **Run inventory** — every run: base × dataset × axis, wandb id, steps, outcome, cited or not. Derived mechanically from [[../30_Knowledge/experiments/_index]]. Doubles as the "organize the data" evidence **and** as an honesty signal — the killed and retracted runs are *in* it
- A.2 **Probe definitions** — formal definitions, nulls, chance levels

---

## Writing order (dependency-first)

1. **Ch3 Method** — stable, long, unblocks everything; three subsections already have prose to port.
2. **Ch4 §4.3 probe suite + §4.6 integrity** — no GPU dependency, high rubric return.
3. **Ch5 §5.5 the blindness section** — the highest-return single section.
4. **Ch5 rest**, in the §5.1–5.6 order.
5. **Ch1 arc** — once Results exists, so the intro promises exactly what Ch5 delivers.
6. **Ch6**, then the **abstract**, last.

Ch2 proceeds in parallel as the missing literature notes land.

## Related

- [[../30_Knowledge/writing/chapter-checklists]] — **per-section gates: drafted / writable today / blocked and on what**
- [[../30_Knowledge/writing/evidence-map]] — **cells → sections → topics → rubric**, with the caveats that travel with each number
- [[index]] — chapter → source map
- [[../30_Knowledge/writing/rubric/_index]] — the ten graded items + queue
- [[../30_Knowledge/writing/thesis-storyline]] — the narrative spine
- [[../30_Knowledge/writing/ablation-axes]] — ⚠ **stale**, rewrite before Ch4
- [[../30_Knowledge/experiments/_index]] — the results ledger
