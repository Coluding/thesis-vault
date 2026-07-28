---
type: writing
status: living
last_updated: 2026-07-24
sources:
  - "[[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]]"
  - "[[../experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"
  - "[[../../50_Decisions/open/second-dataset-action-informativeness]]"
  - "[[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]]"
---

# Ablation axes — the design space for the D2 study

The organising map for the experimental chapter. Two kinds of axis, and the
distinction matters:

- **The dataset axis is a *reporting* axis** — run in full, every cell, because
  the informative-vs-redundant contrast is the D2 result itself.
- **Every other axis is a *search* toolbox** — levers we pull *adaptively* to
  find a configuration that escapes base-parity on an informative dataset. We
  do **not** run their full factorial; we pull them as the dataset runs
  demand. (Priority set 2026-07-24: **the dataset axis is what runs now.**)

The central question the whole space serves: **does an action-conditioned
adapter on a frozen video base learn to use actions, and under what
conditions?** MetaWorld already answered "no, when the data doesn't reward
actions and the optimization has a copy shortcut"
([[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]]). The
ablation space is built to find the "yes," or to characterise the "no"
precisely enough to be a contribution regardless.

---

## Axis 1 — Dataset  *(reporting axis; run in full; PRIORITY)*

Varies **action-informativeness × action-dimensionality × physics regime**.

| Cell | da | regime | informative? | status |
|---|---|---|---|---|
| MetaWorld `five_task` | 4 | manipulation (scripted) | **no** (redundant) | **done anchor** — base-parity, §5.1.x |
| ACWM Push Cube | 2 | rigid-body | yes | ready + base-validated |
| ACWM Robot Arm | 7 | kinematics | yes | needs base-validation |
| ACWM Reacher | 2 | kinematics | yes | needs base-validation |

Why these: MetaWorld is the redundant-action control (the negative result we
own); the three ACWM envs are action-informative by construction (the
commanded action determines the future). Push Cube (da=2) vs Robot Arm (da=7)
gives an **action-dimensionality** contrast that also probes the injection
choice (the ACWM paper finds cross-attention helps at high da); Reacher is a
cheap second-kinematic control at da=2. Particle/deformable envs skipped —
physics-regime breadth is the ACWM paper's claim, not ours, and they carry
the highest base-coherence risk.

Per-env cost: each new ACWM env needs its own **base-coherence validation**
(does the frozen Wan base produce coherent video on that visual domain?) —
Push Cube took real work (letterbox/prompt debugging); Robot Arm (realistic
3D) is likely *easier*, Reacher (flat 2-link) unknown. Validate before
training.

---

## Axis 2 — Adapter family  *(held: output only, empirically)*

The framework (D1) implements four families — output, hidden-state (UniCon),
hypernetwork (HyperAlign), LoRA — but the **empirical study uses the output
adapter only**. UniCon and the hypernetwork are covered by a **computational-
complexity discussion** in the writing (they are too expensive to run the full
comparison); LoRA (a PEFT weight-update baseline, not a frozen-base additive
adapter) is not run. This makes **D1 a software + complexity-analysis
contribution** and **D2 the empirical output-adapter study**. No DynamiCrafter
taxonomy runs, no LoRA config.

_Held constant across the whole study (the "clean" workhorse):_ composition
`mask_mix`, injection `cross_attention`, `condition_on_base_outputs: true`,
`gate_bias: 0`, 65-frame windows, 768² (max_area 589824).

---

## Axis 3 — Training interventions  *(search toolbox)*

Levers against the two base-parity traps (gate saturation; identity-on-base-
input). Pulled adaptively; **the clean baseline has all of them OFF**, so the
dataset gets the purest possible chance first.

| Lever | values | mechanism | status | note |
|---|---|---|---|---|
| **gate_cap** | {off, 0.9} | clamp σ(gate) ≤ 0.9 so the adapter branch keeps ≥10% gradient | implemented | validated on MetaWorld (gate held 0.9, grad-norm alive 0.027 vs 0.005) but did not unlock action-use there |
| **σ-shift** | {off, 5.0} | concentrate training noise at high σ (where actions carry signal); matches Wan pretraining | implemented | its own axis (2026-07-24) — not baked into the workhorse |
| **AVID warmup** (`pretrain_steps`) | {0, N} | force mask=0 (pure adapter) for the first N steps so the pred head is competent *before* the gate is learnable — a **different** escape from gate-saturation than the cap | **implemented** (2026-07-24): `AdapterConfig.pretrain_steps`; the trainer feeds `global_step` via `AdaptedModel.set_train_step`; while step<N `_compose` returns the standalone adapter prediction (gate gets no gradient, stays at init). Off by default (0) | AVID's own mechanism |

**The clean test (the run-now ACWM baseline):** mask_mix + cross-attn +
base-input, **no cap, no σ-shift, no warmup**. Rationale: if the adapter
escapes base-copying on an informative dataset *with zero interventions*, that
is the strongest possible result — the dataset alone did it. If it base-clones
even here, we pull the levers (each isolable because the baseline is bare).

---

## Axis 4 — Adapter mechanism probes  *(search toolbox; single cells, not axes)*

MetaWorld settled composition (replace can't stand alone at 34M; mask_mix
gate-saturates) and base-input-as-crutch diagnostically. On ACWM only two
mechanism questions need the informative data to answer, as **single probe
cells** (not full axes):

- **`condition_on_base_outputs` off** (one cell, Push Cube) — does informative
  data let the adapter *drop* the base-output crutch and denoise on its own,
  or does removing it just hurt? (implemented)
- **AdaLN injection** (one cell, Robot Arm da=7) — validate the cross-attn
  choice where the ACWM paper predicts it matters most. (implemented)

---

## Axis 5 — Frozen base backbone  *(search toolbox; the base-strength lever)*

Swap the frozen base (2026-07-24: **base backbone**, not adapter architecture).
This is the most *theoretically* motivated lever, because it varies exactly
what the base-parity diagnosis blames: **the strength of the frozen base sets
the copy-through pull.** Wan2.2-TI2V-5B is near-optimal on these domains, so
the adapter has almost nothing to earn beyond cloning it — hence base-parity.
A **weaker / smaller base leaves the adapter more to do**, so the same adapter
may develop a real (action-using) residual.

Direct evidence this is the right lever: the AVID reference run on
DynamiCrafter (a ~1.5B video UNet, much weaker than Wan-5B) on *our* MetaWorld
data showed a **healthy, non-saturating gate** (mask mean 0.52 → 0.63,
actively moving) and a ~9.5× loss drop — where Wan-5B gate-saturates to ~0.99
([[../experiments/20260715-avid-metaworld-native-gate-healthy]] vs
[[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]]). The
copy-through pull scales with base strength.

Wired base providers (candidates, no new integration needed):
`dynamicrafter` (weaker, AVID-native — the leading candidate), `wan2.1`,
`opensora` (partial), `diffusers`. The clean comparison holds the adapter +
dataset fixed and swaps only the base.

Caveat: a weaker base is worse at the *domain* to begin with (lower ceiling on
generation quality), so this axis trades "adapter has room to matter" against
"absolute quality is lower" — report both. Pulled if Axes 3–4 don't escape
base-parity on Wan-5B + ACWM, or run alongside as the base-strength contrast.

---

## Run plan (proposed — the "cross", not the grid)

**Run now (dataset axis, clean workhorse):**
1. Clean baseline × {Push Cube, Robot Arm, Reacher} — 3 runs (the D2 headline).
   Prereq: base-validate Robot Arm + Reacher.

**Then, adaptively, only if base-parity persists (toolbox):**
2. On Push Cube: {gate_cap, σ-shift, cap+σ-shift, AVID-warmup} — up to 4 runs.
3. Probe cells: base-off (Push Cube), AdaLN (Robot Arm) — 2 runs.
4. Axis-5 model swap — only if 1–3 all base-clone.

MetaWorld anchor: **done** (`hvxlbfjx` cap+σ-shift; the overfit triangle).

Total if everything is needed: ~9 ACWM runs. But this is a *search*, not a
factorial — stop as soon as a cell shows action-following (nonzero
shuffle-gap), then that config becomes the D2/D4 workhorse.

---

## Metric per cell (identical everywhere, so any outcome is a result)

1. **Action-sensitivity** — two readouts, both *primary*:
   - **Action Error Ratio** (AVID §4.2, added 2026-07-26): train an action
     predictor on *real* videos; report its error on generated videos ÷ its
     error on real videos. **AVID-comparable**, so the D2 table can carry an
     AVID-replica row against their published baselines (ControlNet,
     ControlNet-Small, action-conditioned-from-scratch, Product-of-Experts,
     action-CFG). See [[../related-work/avid]] §Evaluation protocol.
   - **Shuffled/zeroed-action loss gap** — cheaper, no auxiliary model, good
     for fast screening across cells.
2. **Prediction accuracy** — masked-MSE / PSNR vs GT.
3. **Generation quality** — FID / FVD.
4. **Base-parity** — pred-vs-base cosine (how much it clones).
5. **Cost** — trainable params, FLOPs/step, GPU-hours.

---

## If nothing escapes base-parity — the thesis still stands

The axes are built so the study is a contribution either way:
- **(a) success:** one cell shows action-following → the D2 headline + D4.
- **(b) comparative:** the cross yields a trade-off table across datasets and
  interventions — cost/stability/base-parity behaviour, even without (a).
- **(c) diagnostic:** the mechanism of base-parity collapse (gate saturation,
  identity shortcut, data-redundancy condition), measured — already partly
  banked (§5.1.x, the σ-sweep + action probe).

## Related
- [[thesis-storyline]] — the narrative these axes serve; §7 turns this axis
  list into the **hypothesis-first** framing the ablation chapter needs, and
  §8 states the boundary-condition ending
- [[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]] — the MetaWorld result these axes respond to
- [[../../50_Decisions/open/second-dataset-action-informativeness]] — the dataset decision
- [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] — gate trap; cap + AVID-warmup are the two escapes
- [[../../70_Thesis/draft/40-experiments]] — the chapter this feeds
