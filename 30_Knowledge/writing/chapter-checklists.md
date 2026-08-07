---
type: writing
status: living
last_updated: 2026-08-07
sources:
  - "[[../../70_Thesis/outline]]"
  - "[[evidence-map]]"
  - "[[thesis-storyline]]"
  - "[[open-experiments-for-thesis]]"
  - "[[thesis-style-guide]]"
---

# Chapter checklists — what each section needs before it can be written

> **Writing has started.** This is the per-section gate list: what is
> drafted, what is writable today, what is blocked and on what. Companion to
> [[../../70_Thesis/outline]] (function + budget) and [[evidence-map]]
> (which cell licenses which claim).
>
> **Headline: roughly 35 of the 40 pages are writable now.** The only true
> blocks are Ch2 (literature, deferred by decision) and the *outcome* of
> §5.7 (pending A0.4). Everything else is assembly.

## Legend

✅ drafted · 🟢 writable today · 🟡 writable with a stated caveat ·
🔴 blocked

---

## Ch1 Introduction — 4–5 pp

| § | State | Needs |
|---|---|---|
| 1.1 Motivation | 🟢 | [[../../10_now/positioning]]. No evidence dependency |
| 1.2 Problem | 🟢 | Frame as *when and why*, not *can it* |
| 1.3 The arc | 🟢 | [[thesis-storyline]]. Told end-to-end **here and nowhere else** |
| 1.4 Contributions | ✅ | drafted 2026-08-03 |
| 1.5 Structure | 🟢 | write last in the chapter |

**Checklist**
- [ ] §1.3 names the flow-pivot confound (flow *and* a stronger base) rather than glossing it
- [ ] §1.4 and §5.5 agree on the control claim (fixed 2026-08-07; re-check after any results change)
- [ ] AVID positioned as the **starting point**, not a gap, in §1.2 and §1.4

## Ch2 Related work — 5 pp

🔴 **Deferred by decision** ([[rubric/06-literature]]). Not blocking any
other chapter.

- [ ] `lora.md`, `controlnet.md` → §2.1 and Ch3 §3.2 citations
- [ ] FiLM / adaLN-Zero → §2.2. **Do this one first**: it is what makes the
      pathway principle read as principled rather than as a lucky ablation
- [ ] World-model coverage → §2.3
- [ ] Few-step / distillation lineage → §2.4, and it is what makes the D3
      baseline honest
- [ ] Cite the **negative space** with its search scope stated

## Ch3 Method — 10 pp

| § | State | Needs |
|---|---|---|
| 3.1 Composition interface | 🟢 | [[../../10_now/architecture]] |
| 3.2 Families and the selection | 🟡 | the *argument* is writable now; **citations** wait on Ch2. Add the **LoRA 3.1× wall-clock** result as the receipt for criterion (b) |
| 3.3 Conditioning interfaces | 🟢 | ⬅ **PORT** from `draft/30-method.md` |
| 3.4 Shortcut target construction | 🟢 | ⬅ **PORT** from `draft/30-method.md` |
| 3.5 Curvature bias | ✅ | drafted 2026-08-07 |
| 3.6 Computational profile | 🟢 | ⬅ **PORT** from `draft/30-method.md` |

**Checklist**
- [ ] Three sections are **ports, not rewrites**. Preserve the prose
- [ ] §3.5 states the attribution as a **hypothesis**, the derivation as
      proven ([[../theory/shortcut-v-averaging-bias]] banner)
- [ ] Every loss/target names `model_type` **and** `prediction_type`
- [ ] §3.2 answers head-on: if output adapters win on principle, why
      implement four families?

## Ch4 Experiments — 7 pp

| § | State | Needs |
|---|---|---|
| 4.1 Datasets | 🟢 | **state the held-out discipline explicitly**; ACWM clean, RT-1 quarantined |
| 4.2 Backbones | 🟢 | now **six**: Wan2.2, Wan-Turbo, SkyReels, DC, EA V5, EA V5.1. Note `prediction_type: velocity` across all, a control obtained by construction |
| 4.3 Probe suite | ✅ | drafted 2026-08-03 |
| 4.4 Metrics and baselines | 🟡 | Action Error Ratio not yet run; fast-sampler baseline missing. Write the protocol, mark the AER row pending |
| 4.5 Ablation design | 🟢 | [[ablation-axes]], hypothesis-first, 11 hypotheses × 13 axes |
| 4.6 Methodological integrity | ✅ | drafted 2026-08-03, now I1–I10 |

**Checklist**
- [ ] §4.5 leads with **candidate explanations**, never with the axis list
- [ ] §4.2 states the axis count (13) and the comparison count honestly
- [ ] §4.6 includes I8 (base rendered noise past every automated check) and
      I10 (the frozen-base control was the wrong control)
- [ ] Chance levels quoted **per cell**, since temporal chance varies with
      latent frame count

## Ch5 Results — 10 pp

| § | State | Needs |
|---|---|---|
| 5.1 Framework across backbones (D1) | 🟢 | 3+ families, AVID-repo port, LoRA 3.1× |
| 5.2 A working cell (D2) | 🟡 | sourced; **A1 quality landing**, A2 control open. Keep compact |
| 5.3 The pathway decides | 🟡 | sourced; ⚠ **A5 split status unverified.** Read before finalising |
| 5.4 Two scale failures | 🟢 | pedestal + drowning |
| 5.5 What the adapter extracts | ✅ | drafted 2026-08-07 |
| 5.6 Standard metrics are blind | ✅ | drafted 2026-08-03 |
| 5.7 Acceleration: three levels | 🔴 | **outcome pending A0.4**; L2 has no data; L3 confounded |

**Checklist**
- [ ] §5.2 **before** §5.3. Positive result first, negative second
- [ ] §5.2 carries: no quality metrics logged, cancelled pre-convergence
      (quote acceleration never a level), control not measured
- [ ] §5.3 states the **within-Wan** contrast as decisive, DC as
      corroboration; base strength is only eliminated by the within-Wan form
- [ ] §5.7 reports L1 action-free vs action-conditioned against the
      **pre-registration**, and says how many levels landed
- [ ] Every numeral carries `\prov{}{}{}` or a provenance-bearing caption
- [ ] `\nv{commit}` where the source note records an uncommitted tree

## Ch6 Discussion and conclusion — 4 pp

| § | State | Needs |
|---|---|---|
| 6.1 The boundary | 🟢 | all sourced |
| 6.2 Limitations | 🟢 | concentrated **once**, thoroughly |
| 6.3 Future work | 🟢 | objective-level fixes, structural repairs |
| 6.4 Conclusion | 🟢 | write last |

**Checklist for §6.1**, the chapter's argument:
- [ ] **The economics, in its sharpened form:** actions are worth ~0.45 % at
      the *margin* from inside the blind basin (`25085110`) and ~18 % at the
      *optimum* (arm E 0.0357 vs arm 0 0.0433). Not competing numbers: a
      local gradient signal below the noise floor, and a global gap. The
      claim is an **optimisation-landscape** one, not "actions are worth
      little"
- [ ] **Why the local signal is small:** teacher forcing puts a noised
      version of the answer in the input, so the visuals already pin the
      target. This says what the fix must be, and it is an objective, not an
      architecture
- [ ] **Diffusion versus flow: measured, unexplained.** Report the two
      eliminated explanations (parameterisation controlled by construction;
      noise-level distribution dead by the σ-sweep) and the third as a
      **labelled hypothesis** with its test
- [ ] Keep D3 curvature and the D2 hypothesis in **different paragraphs**.
      One is proven, one is a guess, and the shared word will merge them

**§6.2 must contain:** rung-3 control not demonstrated (null on Wan, not
measured on DC, modest magnitude effect on Turbo); single seeds throughout;
the flow-pivot confound; the unresolved cross-base shortcut design; the
in-sample quarantine; L3 confounded three ways.

## Appendix

- [ ] Run inventory from [[../experiments/_index]], **including** killed and
      retracted runs. It is the "organize the data" evidence and an honesty
      signal at once
- [ ] Probe definitions with nulls and chance levels

---

## Cross-chapter gates

- [ ] **A5** (clean-room split status) — a file read; W-a carries three
      rubric items
- [ ] **A0.4** (few-step quality, with its matched control) — the
      acceleration axis's outcome variable
- [ ] **Characterise the L1 action-conditioned failure.** H-E predicts
      *action-following* degrades specifically; a different failure mode is
      a different finding and must not be written as the prediction landing
- [ ] Submission gate: `grep -rn '\\todo{\|\\nv{\|\\fig{' chapters/`
- [ ] Rule-8 gate: numerals without `\prov`
