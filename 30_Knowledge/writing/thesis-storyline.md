---
type: writing
status: living
last_updated: 2026-08-02
sources:
  - "[[../related-work/avid]]"
  - "[[../theory/shortcut-v-averaging-bias]]"
  - "[[ablation-axes]]"
  - "[[../experiments/20260715-avid-metaworld-native-gate-healthy]]"
  - "[[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]]"
  - "[[../experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]"
  - "[[../experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]"
  - "[[../experiments/20260731-wan-action-trace-value-pathway-drowns]]"
  - "[[../experiments/20260731-why-wan-copies-the-base-decomposed]]"
  - "[[../experiments/20260731-wan-action-signal-is-a-global-bag]]"
  - "[[../experiments/20260731-dc-condition-center-accelerates-escape]]"
  - "[[../experiments/20260801-wan-rt1-indistribution-plateau]]"
  - "[[../experiments/20260802-avid-wan-cleanroom-perframe-causal]]"
  - "[[../experiments/20260802-shortcut-works-on-flow-not-diffusion]]"
  - "[[../../20_Tickets/experiments/exp-adapter-wan-rt1-action-token-binning]]"
  - "[[../../20_Tickets/chore-writing-attach-pending-evidence]]"
  - "[[../../10_now/positioning]]"
---

# Thesis storyline — the narrative spine

## ⭐ REFRAME 2026-08-02 — an empirical study, not a systems build

**Working title: _On the Adaptability of Video Foundation Models to World
Models_.**

Decided in conversation 2026-08-02. The thesis is framed as an **empirical
study of when and why adapter-based adaptation works**, rather than as the
construction of a working system. This matches what the evidence actually
supports, and it turns the campaign's negative results from a liability into the
substance: each is a *matched-control* negative with a null battery, which is
what makes a diagnostic claim citable rather than a postmortem.

### The claim set

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Adapting a frozen video FM into an action-conditioned world model is possible, and **the injection pathway decides whether it works** | clean-room A/B, 2.49x @12000, Welch t=10.5, matched adapter contribution + mask | ✅ measured |
| 2 | **Cross-attention injection fails** even with correctly binned per-frame tokens | binned RT-1 run: no benefit at any depth to 2800 steps, declines from 1600 | ✅ measured |
| 3 | **The backbone is not the limit *for addressability*** — flow/DiT with 4x-compressed latents beats the published UNet baseline at matched depth | 0.0175 vs 0.0125 @step 5000; 25.3% vs 24.4% action-driven @12000 | ✅ measured · ⚠️ **scoped 2026-08-06** — the backbone **is** a limit on how much the adapter can reshape at all; see claim 11 |
| 4 | **Shortcut is learnable on flow, not on a curved diffusion arc** | 0.302 vs control 0.034 (9x, disjoint CIs); DC 0.084 vs 0.083; gain flat O(1) vs 4e4 blow-up | ✅ measured |
| 5 | **Horizon**: flow + a stronger base gives multi-second single rollouts vs ~1 s | ~241 frames / ~35-40 s vs ~9-10 s | ⚠️ pending |
| 6 | **The framework makes base-swapping a configuration change** | 3 backbone families behind one interface; a published method ported to a new base family inside its own repo | ✅ demonstrated |
| 7 | **Planning through the world model works, and is slow** — which motivates 4-5 | planning demo + wall-clock | ⚠️ pending |
| 8 | Few-step rollouts **improve over the no-shortcut control but are not competitive**; more training and tuning needed | qualitative | ⚠️ pending |
| **9** | **The adapter reads the action as a scalar ("how much movement"), not as a directional command** — which is why MSE cannot see it and the structure triad reads at chance | global-bag (directionless) + paired-control motion gain 4/4, mean +0.143 + `action_loss_gap` ≈0 across every cell | ✅ measured, ⚠️ no CI on the gain |
| **10** | **The objective governs action-specificity; the backbone governs adaptability** | EA V5 vs V5.1 at matched video backbone: loss reduction tied (−74.9% vs −73.6%), `effect_rel` +36% and `effect_vs_adapter` +26% for diffusion; both diffusion backbones above both flow backbones (n=2/objective) | ⚠️ interim — arms running, steps unmatched (9000 vs 8200), n=1 |
| **11** | **The Wan ceiling was base-specific, not intrinsic to output adapters** — the same adapter family cuts loss **−74.9%** and contributes **0.52** on EasyAnimate vs **−3.3%** / **0.047** on Wan, whose `adapter_base_cosine` 0.9989 means the adapted prediction was numerically almost the frozen base | EA V5/V5.1 vs Wan add-composition arms (`25192313`) | ⚠️ interim |

### ⚠ UPDATE 2026-08-06 — two claims above need scoping

**New evidence:**
[[../experiments/20260806-objective-governs-action-specificity-not-adapter-capacity]],
[[../experiments/20260805-turbo-action-tokens-binned-to-latent-grid]].

**(a) The Wan ceiling was NOT intrinsic to output adapters.** The same
adapter family cuts denoising loss **−74.9 %** and contributes **0.52** of
the prediction on EasyAnimate, against **−3.3 %** and **0.047** on the best
comparable Wan arm — whose `adapter_base_cosine` of 0.9989 means the adapted
prediction was *numerically almost the frozen base*. Claim 3 above ("the
backbone is not the limit") therefore needs rewording: the backbone **is** a
limit on how much the adapter can reshape at all; what it is *not* is a
limit on whether per-frame conditioning is addressable.

**(b) The objective governs action-specificity, not adaptation capacity.**
EA V5 (diffusion) vs V5.1 (flow) at matched video backbone: loss reduction
tied (−74.9 % vs −73.6 %), but `effect_rel` **+36 %** and
`effect_vs_adapter` **+26 %** for diffusion. Both diffusion backbones sit
above both flow backbones across independent families (n=2 per objective).
**This generalises the 0.45 % economics** — the objective allocates
gradient; capacity does not bind. ⚠ Interim, unmatched steps, differing text
backbones, and it is *"flow is weaker"* not *"flow fails"*.

**(c) A new dissociation, sharper than the old one.** On the distilled
Turbo base the action *reaches* the adapter better than anywhere else
(`effect_vs_adapter` 0.18–0.31) while `action_loss_gap` stays at ~0 and
`action_cos` never leaves 0.9998 — **effect without accuracy**. The
domain-corrector reading survives, now stated more precisely: the adapter
consumes the action without learning the action-conditioned dynamics.

➜ These revise, not overturn. The pathway principle stands; the *scope* of
"Wan fails" narrows to a statement about that base's adaptability, and the
objective joins the pathway as a governing variable.

### ⭐ SYNTHESIS 2026-08-07 — two capabilities, and why they do not fit in one adapter

Restated after the EA results and the action-free/action-conditioned split.
The thesis demonstrates **two separate things an adapter on a frozen video
prior can do**, and then shows they compete for the same gradient.

**(1) In-domain fine-tuning: strong, and worth stating positively.** The
adapter reshapes a frozen base substantially in-domain. On Wan × ACWM it
beats the frozen base on **6/6** quality metrics (FVD 1118 → 406, −64 %); on
EasyAnimate it cuts denoising loss **−74.9 %** and contributes **0.52** of
the prediction. This was first written up as the *negative* half of the
pathway contrast, because it carries no action information. It is also a
capability in its own right and should be named as one: **as a
domain adapter the method works well**, and the frozen base is a
configuration choice while doing it.

⚠ **Scope:** the Wan quality win is ACWM-only; on RT-1 the same adapter is a
net perceptual regression. And on Wan the adapter that produces those
metrics is *cosmetic* by contribution (`adapter_base_cosine` 0.9989),
whereas on EA it genuinely reshapes; the Wan ceiling was **base-specific,
not intrinsic to output adapters**.

**(2) Action conditioning: works, under conditions.** Per-frame modulation
of normalised activations carries the action (2.49× at matched contribution
and mask); cross-attention does not. What is carried is a **scalar**, how
much motion rather than which direction. And the **objective** decides how
much of the adaptation becomes action-conditioned at all: at matched video
backbone the two objectives adapt equally well (−74.9 % vs −73.6 %) while
`effect_rel` differs by **+36 %** in favour of diffusion, with both diffusion
backbones above both flow backbones across independent families.

**(3) The two do not fit in one adapter.** This is the pre-registered H-E
prediction, and the evidence now speaks to both branches:

| level | arrangement | outcome |
|---|---|---|
| **L1** shortcut adapter | **entangled** — one adapter learns actions *and* step size | few-step works **action-free**; **does not work** with actions |
| **L3** distilled base + action adapter | **separable** — acceleration lives in the base | action effect present and causally isolated |

⚠ **Pending characterisation** (2026-08-07): the exact form of the L1
action-conditioned failure, and the matched controls on both arms. H-E
predicts specifically that *action-following* degrades; a different failure
mode would be a different finding. See
[[open-experiments-for-thesis]] A0.4 and A7.

### Why diffusion appears better suited (2026-08-07)

Action following is now demonstrated on **four backbones across both
objectives**: DynamiCrafter (diffusion), EA V5 (diffusion), EA V5.1 (flow)
and Wan-distilled. Flow is not disqualified; diffusion is *better suited*.
Three candidate explanations, two already eliminated.

| # | Hypothesis | Status |
|---|---|---|
| **H1** | **Actions are worth more where the base is less committed.** A curved diffusion trajectory admits more consistent futures at a given state than a near-straight rectified-flow one, so the action has more marginal value in the loss and the gradient pays more for it | ⭐ **the live hypothesis** |
| H2 | Parameterisation: `prediction_type` changes what a fixed-scale adapter contribution is worth | 🛑 **eliminated** |
| H3 | Actions only carry signal at high noise, so the noise-level distribution decides | 🛑 **eliminated** |

**H2 is eliminated by the existing configs, at no cost.**
`prediction_type: velocity` holds across **all four** cells: DC
(diffusion+velocity), EA V5 (diffusion+velocity), EA V5.1 (flow+velocity),
Wan (flow+velocity). A variable that does not vary cannot explain a
difference. The parameterisation is therefore *controlled by construction*,
and the diffusion-versus-flow difference is attributable to `model_type`,
i.e. the trajectory geometry, rather than to the prediction target.

**H3 is eliminated** by the flat σ-sweep (H8 in [[ablation-axes]]).
**Report it anyway** in the thesis: it is the obvious first guess, and a
reader who does not see it tested will assume it was not.

**H1 fits the shape of the measurement.** Adaptation *capacity* is tied
(−74.9 % vs −73.6 % loss reduction) while *specificity* differs (+36 %
`effect_rel`, +26 % `effect_vs_adapter`). The difference is not how much the
adapter can do; it is how much of what it does the action gets credit for,
which is what a marginal-value argument predicts and a capacity argument
does not.

**⚠ The trade-off this implies, and it must be written rather than left for
a reader to find.** H1 cuts against the D3 result:

> **Curvature hurts the shortcut target and helps the adapter.** A curved
> arc makes the velocity-averaging target biased, which is why the shortcut
> objective is learnable on flow and not on diffusion. The same curvature
> leaves the base less committed at a given state, which is why action
> conditioning is more specific on diffusion.

If it holds, the thesis has a **trade-off rather than a preference**: the
geometry that makes acceleration easy makes conditioning harder. One
property then explains the whole D2/D3 split, which is a stronger §6.1 than
"diffusion is better suited".

**Test:** H1 predicts higher action-attributable variance in the objective
on diffusion at matched conditions. That is what the **IDM ceiling** (B5 in
[[open-experiments-for-thesis]]) measures, and it costs minutes of GPU.

**Why it is one claim rather than three.** Actions are worth ~0.45 % of a
teacher-forced denoising loss, so they already lose the gradient competition
to appearance. In-domain fine-tuning is what that objective *pays for*;
action conditioning is what it barely pays for; and a consistency objective
added to the same parameters is a second, larger claim on the same budget.
Separating acceleration from conditioning is therefore not a convenience,
it is what the loss economics requires.

### ⭐ SYNTHESIS 2026-08-06 — what the adapter actually extracts from the action

Three results that looked separate resolve into one statement:

| result | what it showed |
|---|---|
| **Global bag** (Wan, 07-31) | the signal is **directionless** — steering cos ≈ 0.00, temporal at chance, spatial at chance |
| **Motion tracking** (Turbo, 08-06) | the action **does** drive *how much* the arm moves — paired shuffled-action control positive **4/4 draws**, mean +0.143 |
| **Loss gap ≈ 0** (every cell, throughout) | feeding the *correct* action does not reduce error versus a wrong one |

> **The adapter reads the action as a scalar — "how much movement" — not as
> a directional command.**

This is sharper than "effect without accuracy", and it explains the
dissociation rather than restating it: a magnitude-only response is exactly
what squared error cannot reward, because MSE penalises slightly-misplaced
motion as hard as no motion at all. It also explains why the structure triad
reads at chance while the effect metrics are non-zero — a scalar gain has no
direction, no temporal placement and no spatial locus to detect.

**This is the campaign's first causally-isolated rung-3 (control) signal**,
and it arrived only because the control was correct: the frozen-base
comparison gave a spurious gap of 0.66 that vanished under a paired
shuffled-action control ([[../experiments/20260806-motion-tracking-is-action-driven-but-the-base-control-was-wrong]]).

⚠ **Scope tightly.** Modest effect; **no confidence interval on the gain**
yet (the logged CIs are on `corr(adapted, GT)`, not the gain, and that
correlation spans zero in 3 of 4 draws); one run, no seed replication; on
the distilled Turbo base only. Write it as *"a modest, consistently-signed
effect on motion magnitude"* — never as trajectory control.

### The central mechanistic finding, stated precisely

> Adapters **can** make a video foundation model action-conditioned — but only
> when the conditioning enters through **per-frame modulation of normalised
> activations**. Supplying the same per-frame action as **cross-attention
> tokens** does not work, however well-scaled (`action_token_norm`) and however
> well-aligned to the latent grid (`action_seq_len`): the signal survives the
> attention and **drowns at the residual add**.

Do **not** write this as "Wan adapters cannot incorporate actions". That is
contradicted by our own clean-room result and is the weaker, less interesting
claim. The finding is about *the pathway*, not about the backbone or the family.

Supporting chain: cross-attention output RMS ~0.01 against a stream of 1.8-3.0
([[../experiments/20260731-wan-action-trace-value-pathway-drowns]]) → scale fixes
buy 6-10x but never control → binning buys nothing → per-frame AdaLN concat, on
the *same* base and data, buys 2.49x with genuine frame localisation
(0.409 vs chance 0.200). AdaLN multiplies a *normalised* activation and is
scale-free by construction; cross-attention adds into an unnormalised residual
stream and is not.

### What the thesis explicitly does NOT claim

- Competitive few-step video generation. Rollouts are directionally better than
  the no-shortcut control and **not** competitive; compute for the full training
  runs was not available.
- A planning or RL methods contribution. Planning is included as the use case
  AVID names as future work, not as a control contribution
  ([[../../10_now/positioning]] anti-positioning).
- State-of-the-art video quality on any benchmark.
- That the objective-level limit is solved. Actions are worth ~0.45% of a
  teacher-forced denoising loss; nothing here changes that.

### Why this framing is defensible

Every ✅ row above has a **matched control** and a **null that was verified, not
assumed** (`base_null_violation` exactly 0 throughout; frozen-base invariance
checked before every run). The negative results are therefore *discriminating*
rather than *inconclusive* — which is the difference between an empirical
contribution and a list of things that did not work.

---



The chain the thesis argues, in order, and which chapter carries each link.
Settled in conversation 2026-07-26. This doc is the **narrative** layer;
[[../../10_now/positioning]] remains the **contribution/deliverable** layer and
[[ablation-axes]] the **experimental-design** layer. They must stay consistent
with each other — if this doc changes, check the other two.

> **⚠ UPDATED 2026-08-01 — the arc now has an ending.** Everything below
> through §7 stands. §8 ("the ending") was written when the ablation was still
> running and predicted a *phase boundary* in (base strength × dataset
> action-informativeness). The 07-30/08-01 campaign **replaced that guess with
> a measured three-part answer** — see the new §8 and §9. Ch1 should tell the
> arc through §7 and then land on §8/§9, not on the old boundary framing.

**Design constraint:** the arc is chronological (each step's limitation forces
the next), but the chapters stay **deliverable-separated** (D1–D4, CLAUDE.md
Part 12). Resolution: *the arc is the introduction's job*; the chapters keep
their evidence separated and merely **order** themselves along the arc.

---

## ⭐ The chain, restated 2026-08-04 — efficiency as the spine

**Adopted 2026-08-04 (Lukas); decision
[[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]].** Two nodes
move; everything else is unchanged.

```
DC + adapter works  →  AVID  →  TOO SLOW  →  flow models  →  rectified flow
      →  DISTILLATION TO FEW STEPS — three axes
           L1  shortcut adapter      speed in the ADAPTER      (entangled)
           L2  PDD / LoRA distil     speed in a 2nd ADAPTER    (separable)
           L3  distilled base        speed in the BASE         (free)
      →  ANALYSIS: what worked, what did not, and why
```

**What changed.** "Shortcut" was one node with one cell; it becomes three
**placements** of the same capability, distinguished by *where few-step
behaviour lives* and therefore by whether acceleration and conditioning are
**entangled or separable**. The ending changes from a diagnosis to a
**comparative analysis with a design recommendation**.

**What did not change.** The D2 mechanism campaign is not displaced — it is
the *conditioning* half of the final analysis node, which is where its 20+
runs already point. §§1–9 below stand as written.

**The prediction the economics bound makes** (and the reason this is a
designed experiment rather than a sweep): actions are worth ~0.45 % of a
teacher-forced denoising loss, so they are already outbid by appearance
correction; adding a consistency objective to the **same** adapter adds a
second, larger claim on the same gradient budget. **Predicted: L1 degrades
action-following; L2 and L3 preserve it.** If it holds, the recommendation
is *separate acceleration from conditioning*. If it fails, the framing
survives and the recommendation inverts.

⚠ L1 has never been trained (it is D4); L2 and L3 are ticketed but unrun.
Build the framing now; write no level as having worked until run ids land.

---

## The chain (as settled 2026-07-26, unchanged below)

```
ESTABLISH  (DynamiCrafter — the baseline cell, and where planning is contributed)

  DC + adapter works                [AVID-style action conditioning on a frozen
                                     DC base; our condition_center fix gives a
                                     ~6x faster escape from the blind basin]
    → PLANNING on it                [CONTRIBUTION: plan through the learned world
                                     model — AVID names this as its own future
                                     work and does not do it]
    → two limits, not one           [(a) SPEED: rollout cost = horizon x NFE x
                                     per-step; (b) HORIZON: ~16 frames / ~1 s is
                                     too short a future to plan over at all]

CONTRIBUTE  (Wan-5B flow — where the thesis's technical contributions live)

    → flow matching + a stronger base
      → HORIZON: ~241 frames / ~35-40 s in ONE rollout vs ~9-10 s        [sec 10]
      → CURVATURE: kappa=0 makes the shortcut target exact               [sec 4]
        → shortcut works on flow, NOT on diffusion  (0.302 vs 0.034;
          DC 0.084 vs 0.083)                                       [D3, measured]
      → the D2 mechanism study: what makes an adapter use actions   [sec 8-9, 12]
      → the framework itself: a new base family added to the official
        AVID repo in one session                                    [D1, sec 11]
```

**Both cells carry weight, and they carry different weight.** DC is the
*working baseline* and the host for the **planning contribution** — it is where
the thesis shows a learned world model is actually usable for selection. Wan is
where the *technical contributions* live: the horizon argument, the curvature
result, the D2 mechanism study, and the extensibility evidence.

**This replaces the earlier "DC is the spine, Wan is a generality test that
collapses" framing (superseded 2026-08-02).** That reading no longer matches the
evidence: shortcut is learnable on Wan and not on DC
([[../experiments/20260802-shortcut-works-on-flow-not-diffusion]]); Wan conditions
on actions fine given frame-addressable conditioning, beating the AVID/DC
reference at matched depth
([[../experiments/20260802-avid-wan-cleanroom-perframe-causal]]); and the horizon
advantage is Wan's alone. Writing Wan as the failure branch would now understate
every technical result in the thesis.

> ⚠️ **Positioning check.** [[../../10_now/positioning]] anti-positions the thesis
> as "not a control / RL paper", and this doc previously called planning "a
> demonstration, not a claimed contribution". Elevating planning to a
> contribution is consistent with that **only** if framed as *"we extend AVID to
> the planning use case AVID itself names as future work"* — not as a planning /
> RL methods contribution. Keep the claim scoped to that. ➜ check
> [[../../10_now/positioning]] stays consistent.

### 1. AVID works — the starting point, not a gap we fill

[[../related-work/avid]] (Rigter, Gupta, Hilmkil & Ma, arXiv:2410.12822v2)
adapts a frozen video diffusion model into an **action-conditioned world
model** using only the base's noise predictions, via a learned mask over the
base output (our `composition: mask_mix`).

**Action conditioning on a frozen video base is AVID's contribution, not
ours.** Framing this step as a gap we fill is false and would not survive a
viva — see the Correction section of [[../related-work/avid]].

Our own reproduction on real AVID code:
[[../experiments/20260715-avid-metaworld-native-gate-healthy]] (wandb
`pg3x72uc`, 11M adapter, frozen DynamiCrafter-512, MetaWorld) — ~9.5× train
loss drop, gate climbing 0.52 → 0.63 rather than saturating.

**Open gate on this step:** that run establishes *clean convergence and healthy
gate mechanics*. It does **not** measure action-sensitivity. Before the
planning step is built, run an action-sensitivity eval on this checkpoint —
see "Gates" below.

### 2. Planning on the AVID model — a citable extension

AVID evaluates video-prediction quality only. Its conclusion names planning as
future work: it aims *"to explore the use of synthetic data generated by AVID
adapters for planning tasks."*

So: train a small reward model on rollout states, plan through the world model.
**Updated 2026-08-02: this IS a claimed contribution** — AVID names planning as
future work and does not do it, so doing it is a contribution *relative to
AVID*. It is not a planning/RL methods contribution, which keeps it consistent
with the anti-positioning in [[../../10_now/positioning]] ("Not a control / RL
paper"). Scope the claim to "we extend AVID to the use case it names".

> **⚠ STATUS 2026-08-02 — EVIDENCE PENDING, NOT MISSING.** Lukas reports the
> planning demo **was run elsewhere** and will supply the artefacts. Treat the
> node as supported for drafting purposes, but it is **not yet in the vault**:
> no run id, no checkpoint, no numbers. **Do not put a figure or a number in the
> draft until the data lands.** ➜ ACTION: attach run id + outputs, then convert
> this flag into an experiment note and a ledger row.
>
> *(Prior status, kept for context:)* **NOT YET RUN — and it is now spine-critical.** With DC
> confirmed as the main story (and `condition_center` giving a DC cell that
> follows actions at 3.6x the AVID reference —
> [[../experiments/20260731-dc-condition-center-accelerates-escape]]), planning
> is the **second link of the main chain** and the only one with no evidence
> behind it. It is the highest-value remaining experiment for the thesis:
> higher than further Wan fixes, because it converts a working D2 cell into the
> motivation for D3/D4.
>
> Minimum viable version (time-boxed): a short-horizon planner (CEM or random
> shooting, ~10-20 candidate action sequences, horizon 4-8) over the DC arm-E
> checkpoint on ACWM Robot Arm, scored by a simple state/goal distance — enough
> to (a) show the world model is *usable* for selection, and (b) produce the
> wall-clock number that motivates few-step. Even a negative planning result is
> usable: it would bound what a 0.106-effect_rel world model supports.
>
> The cost arithmetic in step 3 stands on its own, so the *motivation* chain
> survives without it — but the *demonstration* is what makes the spine a
> system rather than a sequence of components. Ticket:
> [[../../20_Tickets/experiments/exp-eval-planning-through-dc-world-model]].

### 3. Too slow — the motivation for few-step

Rollout cost is NFE × per-step cost. Needs sourced wall-clock/NFE numbers
before it goes in the draft. The honest baseline is **not** only 50-step DDIM —
DPM-Solver and consistency-model sampling are the fair comparison (flagged as
an open framing question in [[../../10_now/positioning]]).

### 4. The curvature bias — the proven node

[[../theory/shortcut-v-averaging-bias]]. Frans et al. eq. (4) averages two
half-step velocities. That is exact for a **straight** interpolant and biased
on a **curved** VP diffusion arc: the arithmetic mean is the Euclidean
centroid, the manifold wants the Riemannian one, and the gap is the sagitta
(≈ κ·δ²/2). The true field is **not a fixed point** of the averaging rule, so
the bias is not trained away — it compounds up the doubling tower.

Measured on `ddim_micro_step_v` with zero model error: the averaged target
lands **5.1 % off at s=1/4, 16.1 % at s=1/2, 24.1 % at s=3/4**; the
endpoint/displacement target lands **0.000000** at every step.

**Phrase it precisely.** "Shortcut modelling does not work on diffusion" is
false and attackable (consistency distillation on VP diffusion demonstrably
works, and our own endpoint fix is exact on the curved manifold). The correct
claim is about **the published velocity-averaging target**, and it has two
escapes: fix the coordinates (endpoint inversion) or fix the geometry
(flow-matching base, κ=0). We characterise both.

This is the one link in the chain that is *proven* rather than observed. It
carries its own section; it is not a transition sentence.

### 5. Flow matching → Wan

κ=0 makes eq. (4) faithful as written. See
[[../../60_Updates/entries/2026-06-19-pivot-flow-matching-base]].

**⚠ This step changes two variables at once.** Wan is flow-matching (the D3
reason) *and* a much stronger base than DynamiCrafter — and base strength is
exactly what our own diagnosis blames for the failure that follows
([[ablation-axes]] Axis 5: copy-through pull scales with base strength;
DynamiCrafter's gate is healthy where Wan-5B saturates ≈0.99).

The narrative must name this confound rather than paper over it. **The
ablation is the confound resolution** — that is its job, and the reason it
belongs in the chain rather than in an appendix.

### 6. What fails on Wan is D2, not D3

The chain arrives at Wan through the shortcut argument, so a reader will
assume the shortcut objective failed there. **It did not — it has not been
cleanly tested there.**

- What is measured on Wan is **base-parity / action-blindness**: a **D2**
  failure, on action-conditioning runs with no consistency loss
  ([[../experiments/20260724-metaworld-cap-shift-triangle-base-parity]],
  wandb `hvxlbfjx`; the ACWM workhorse config in [[ablation-axes]] has no
  shortcut term).
- **D4 (combined action + shortcut) is gated** on D2 producing a working cell.

> **Correction 2026-07-26 — D3 is *not* gated on D2.** An earlier version of
> this section said it was. It is wrong, and the mistake is expensive: it would
> park the entire D3 chapter behind a D2 result that may never arrive.
>
> The **action-free shortcut run** ([[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]],
> [[../../20_Tickets/experiments/exp-shortcut-target-ab-actionfree]]) tests the
> pure D3 question — *can a step-size-conditioned adapter make the frozen base
> samplable in few steps?* — with actions stripped entirely. It needs action
> conditioning to work exactly not at all. Only **D4** requires both.
>
> Practically: D3 evidence can be gathered **in parallel with** the D2 ablation,
> not after it. Given that D2 is the part currently failing, that reordering is
> the single cheapest de-risking of the thesis schedule.

State the D2/D4 gating plainly. It shows the evidence boundary is understood, and
it is the deliverable separation CLAUDE.md Part 12 requires. The
`anchor_prob: 1.0` no-shortcut control is what proves the D2 failure is not the
consistency loss's fault — worth running early, because it protects the D3
chapter from being read as collateral damage.

### 7. The ablation — hypothesis-first, not axis-first

Same experiments, opposite readings:

- **Weak:** "Wan didn't work, so we varied model, dataset, adapter size…" →
  reads as flailing.
- **Strong:** "the failure has N candidate explanations; here is the design
  that discriminates between them." → reads as science.

Lead with the hypothesis column. Axes are from [[ablation-axes]]:

| Hypothesis for the D2 collapse | Axis that tests it | Status |
|---|---|---|
| Base too strong → copy-through pull | Axis 5 — backbone swap (DynamiCrafter, SkyReels-1.3B) | tickets drafted |
| Data does not reward actions | Axis 1 — MetaWorld vs ACWM | running |
| Optimisation trap: gate saturation | Axis 3 — `gate_cap`, AVID warmup | implemented |
| Optimisation trap: identity-on-base-input | Axis 4 — `condition_on_base_outputs: off` | implemented |
| Wrong injection / capacity | Axis 4 — AdaLN cell; adapter size | implemented |
| The shortcut objective itself | no-shortcut control (`anchor_prob: 1.0`) | config exists |

Every row ends in a **mechanism claim**, not a result row. That is what makes
a negative outcome a contribution rather than a postmortem.

### 8. The ending — three measured mechanisms, not a guessed boundary

The 2026-07-30 → 08-01 campaign answered the ablation. The ending is no longer
"a phase boundary in base-strength × data" — it is a **mechanism stack**, each
layer measured, each with a matched control.

**(i) The failure was ours, not the data's.** The unmodified AVID recipe
follows actions on *our* ACWM Robot Arm (effect_rel 0.0295, null 0) where our
three adapters were blind (0.0013–0.0056) — same frozen weights, same data,
same probe ([[../experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]).
This kills the "our data is too hard" reading and turns the thesis question
into a diffable engineering gap.

**(ii) Two opposite scale failures at the injection interface.** The
conditioning must arrive *informative* AND *stream-commensurate*:
- **DynamiCrafter:** a **learned pedestal** — the embedding grows 106× during
  training into a 99.7%-constant vector, 14× the time embedding, carrying
  0.5% action-driven variance vs the reference's 24%
  ([[../experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]]).
  Fix (`condition_center`): 0.003 → **0.106**, 3.6× the reference.
- **Wan:** the mirror image — faithful but **250× too quiet**; the action
  survives cross-attention (44–56% action-driven) and drowns at the residual
  add ([[../experiments/20260731-wan-action-trace-value-pathway-drowns]]).
  Fix (`action_token_norm`): 6–10×.
Both are *scale calibration*, in opposite directions, at the same interface.
Neither is visible in loss, gate, FID or sample quality.

**(iii) The adapter corrects the base rather than using the actions — and the
objective is why.** Decomposed ([[../experiments/20260731-why-wan-copies-the-base-decomposed]]):
~87% of the pred–base cosine is unavoidable shared-target convergence (present
with *no* base input); the removable part is **oracle-reading** — with
`condition_on_base_outputs` the adapter's function is ~100× more sensitive to
the base's prediction than to the actions. Underneath both sits the economics:
**actions explain 0.45% of the teacher-forced denoising loss**, so appearance
correction always pays more per unit of gradient. What action signal survives
is a **global bag** — arbitrary in direction (steering cos ≈ 0.00), unaligned
in time (px→latent correspondence at chance), uniform in space
([[../experiments/20260731-wan-action-signal-is-a-global-bag]]).

### 9. The result: it works on DC, fails on Wan, and the pathway is why

> **⚠ REWRITTEN 2026-08-03.** The previous version of this section stated a
> "two-factor law" over a table that **contained retracted numbers** — the
> SkyReels × RT-1 cell (0.0450 peak, the "35×") was voided when its
> `dataset_size` turned out to be 76 rather than 5000, *and* it was
> evaluated in-sample ([[methods-integrity]] I1, I2). It also stated the
> honest limit **globally**, which the DC evidence contradicts. Both are
> corrected below.

### DC is not action-blind — it is the working cell

Three independent readouts, all on ACWM Robot Arm
([[../experiments/20260731-dc-condition-center-accelerates-escape]],
[[../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]):

| | arm E (`condition_center`) | arm 0 (untreated) | reference / chance |
|---|---|---|---|
| `effect_rel` @3500 | **0.11479** | **0.04564** | AVID reference 0.0295 |
| steering cosine | **+0.117** | +0.106 | chance 0.000 |
| temporal alignment | **1.000** | 1.000 | chance 0.313 |
| spatial concentration | **0.470** | 0.489 | chance 0.100 |
| frozen-base null | 0.000 | 0.000 | — |
| adapted loss | **0.0357** | 0.0433 | — |

Read this carefully, because it is the thesis's positive result:

1. **All three structure axes clear chance on held-out `ind_test`.** This is
   rung 2 of the ladder, not just sensitivity — the action→output map is
   directional, temporally placed and spatially localised.
2. **The untreated control clears the AVID reference unaided** (1.55×), and
   arm F — AVID's exact 9,344-parameter encoder with no centering — reaches
   1.71×. The cell works *natively*; `condition_center` is an accelerator,
   not an enabler.
3. **Action-following wins on the denoising objective itself** — arm E's
   adapted loss is ~18% *below* the untreated control's. The
   action-following solution is not a trade against prediction quality; it
   is better on the very objective the blind basin optimises.
4. **Blindness on DC is a long transient, not a state.** Both controls sat
   at ≤0.005 through step 2000 and then rose sharply. This was *predicted
   in advance* by the 0.45% loss-share economics — escape pressure appears
   only once the easy variance is exhausted
   ([[../experiments/20260731-why-wan-copies-the-base-decomposed]] §3).

**Consequence for the thesis:** the D2 target claim from
[[../../10_now/positioning]] — *"adapters can make a frozen video model
action-following on at least one benchmark"* — **is satisfied**, and the
positive control the ablation needed exists.

### Wan fails on structure while succeeding on quality

Same adapter family, same data. Wan × ACWM beats its frozen base on **6/6**
quality metrics (FVD 1118 → 406, −64%) with all three structure probes **at
chance**. A 2.75× FVD improvement carrying no action information is a
**domain correction**.

> **The same adapter design is a domain adapter on Wan and an action
> conditioner on DC — never both on either.**

### What discriminates them: the pathway

Not the data (AVID follows actions on our data), not capacity (a clean 7.5M
arm sits below the DiT-clone arms), not the family (DC uses the same one),
not the objective term (no consistency loss present). **The injection
pathway.** Per-frame modulation of normalised activations carries the
action; cross-attention tokens survive the attention and drown at the
residual add — and the clean-room A/B shows this is *causal* at matched
adapter contribution and matched mask.

### The honest limits — per cell, not global

- **Rung 3 (control) — updated 2026-08-06.** A **causally-isolated but
  modest** effect on *motion magnitude* now exists on the distilled Turbo
  base (paired shuffled-action control, 4/4 draws, mean +0.143). It is
  **not** trajectory control: `action_loss_gap` ≈ 0 everywhere, so the
  correct action still does not reduce error versus a wrong one. On the
  non-distilled Wan arms the rollout-swap is null; on DC it is **still not
  measured**. Never write "control is demonstrated nowhere" — the DC
  structure probes pass (rung 2) and the Turbo paired control fires (rung 3,
  magnitude only). Ticket:
  [[../../20_Tickets/experiments/exp-eval-rollout-action-swap-dc-arme]].

  > ⚠ **The DC rollout videos (arm E `6oyu1inq`, arm F `86kb01su`) compare
  > *adapted vs frozen base*, not *true vs swapped actions* — so they
  > cannot support a control claim.** **Qualitative video gains — the
  > adapted model looking distinct from the base, with better motion
  > timing — are temporal-prior improvements, not action-following.** The
  > Wan cell is the standing counterexample: it beats its frozen base on
  > 6/6 quality metrics while sitting at chance on all three structure
  > probes. Re-generating the *same clips and seeds* with wrong-clip and
  > zero actions would convert this into control evidence, and it is a
  > generation pass, not a training run.

  **What the videos *do* support** — and it fills a gap flagged as
  blocking: **no DC run logs quality metrics at all** (all 18 checked), so
  these rollouts are currently the *only* evidence about the DC cell's
  output quality. As a qualitative quality demonstration on the cell that
  carries the positive D2 result, they belong in the draft — labelled
  qualitative, and making no claim about actions.
- **No DC run logs quality metrics** (all 18 checked), so whether the
  working cell improves or degrades perceptual quality is **unknown** —
  while the RT-1 cells are known to degrade it. This must be measured before
  the DC cell is written up as a success.
- **The runs were cancelled before convergence**, so no asymptotic level
  claim is supported, and `condition_center`'s level advantage (3.7× → 2.5×
  and still closing) **must not be quoted as a fixed number**. Quote the
  ~6× acceleration only.
- **`effect_rel` is monotone in gain**, so the sensitivity column is a
  screen; the structure triad is what carries the DC claim.
- **The economics bound stands** — actions are worth ~0.45% of a
  teacher-forced denoising loss. That is why escape is *late* on DC and why
  it never happens on Wan through a pathway that attenuates the signal.

**Why this is a result.** It is a positive claim with a matched negative
control at architecture level: adapter-based action conditioning on a frozen
video model **works**, and the condition on which it works is a property of
the injection pathway that no standard readout can see. The failure case is
not a failure to make it work — it is the measurement that isolates *what
makes it work*.

**Why this is a result and not a failure.** It is a *positive, mechanistic*
claim assembled from negative outcomes: plug-and-play action conditioning fails
twice — mechanically (scale mis-calibration at the interface; oracle-reading
when the base's answer is in the input) and economically (a teacher-forced
denoising objective prices actions at ~0.5% of the loss). The mechanical
failures are diagnosable with cheap probes and fixable with two normalisation
changes; the economic one is not fixable by architecture at all and bounds what
any adapter of this family can do. Naming that boundary — with the probes that
locate it — is the contribution.

**What would move it (future work, scoped honestly):** an objective that pays
for actions — action-CFG, rollout/multi-step losses, or action-conditional
consistency — plus the two structural repairs the bag-analysis implies
(enforced px→latent temporal binning; a gate that can localise). Open decision:
[[../../50_Decisions/open/wan-action-following-needs-objective-change]].

---

## 10. Why flow + a more capable base: the horizon argument (added 2026-08-02)

A practical claim that the chain above did not previously make, and which is
independent of the curvature argument:

**Flow matching plus a stronger base buys *horizon*, and horizon is what planning
actually needs.** DynamiCrafter's usable window is ~16 frames (~1 s at its fps);
the Wan cell trains on far longer windows — Lukas reports **241 frames on an
H100, decoding to ~35–40 s of video** versus ~9–10 s for the DC baseline.

Why that matters for the thesis's spine, in order:

1. **Longer planning horizons.** A planner can only evaluate futures the world
   model can generate. A ~1 s window bounds the horizon regardless of planner
   quality.
2. **One rollout instead of many.** Simulating several seconds in a *single*
   rollout removes the compounding-error and wall-clock cost of chaining short
   rollouts.
3. **Compounds with the few-step argument.** Fewer sampling steps (flow) x more
   simulated seconds per rollout (capacity) is a multiplicative reduction in
   planning cost — a sharper motivation for D3 than step count alone.

**State it honestly as two factors, not one.** This is *flow matching* **and**
*a more capable 5B base*; the storyline already flags that the pivot changes two
variables (§5). The horizon benefit is largely the second factor. The curvature
result ([[../experiments/20260802-shortcut-works-on-flow-not-diffusion]]) is what
isolates the first.

> ⚠️ **EVIDENCE PENDING.** The 241-frame figure is **not in the vault** — the
> largest `temporal_length` in any committed Wan config is 97. Needed before it
> is written: run id, frames, resolution, VRAM, decoded seconds, and the matched
> DC number. ➜ ACTION: locate and record.

## 11. The framework itself is a contribution (D1), and it is demonstrable

The repository is not scaffolding for the experiments — it *is* deliverable D1,
and 2026-08-02 produced direct evidence for the extensibility claim rather than
an assertion of it:

- The framework already composes the same adapter interface across
  **DynamiCrafter (diffusion/UNet)**, **Wan2.2 (flow/DiT)**, **SkyReels** and
  **OpenSora**.
- In one session, a **new backbone family was added to the *official AVID repo***
  as a third branch (`external_repos/avid/wan_diffusion/`) — AVID's recipe,
  unchanged, running on a rectified-flow DiT with a 4x-temporally-compressed
  latent space, reproducing their behaviour
  ([[../experiments/20260802-avid-wan-cleanroom-perframe-causal]]).

The second point is the strong form of the claim: *adding a new base model and
training an adapter on it is cheap, and the result is reproducible by anyone.*
It also doubles as the D1 evidence that the composition interface really is
base-agnostic, since it survived a diffusion→flow and UNet→DiT boundary.

---

## Chapter mapping

| Chain step | Chapter |
|---|---|
| Adapter families + why output/AVID | **Ch3 (new)** — `draft/25-adapters.md` (D1) |
| Composition rule, framework, conditioning | Ch4 Method (D1) |
| AVID reproduction | §5.1 (D2) |
| Planning demo | §5.1.y (D2, labelled sanity check) |
| Rollout cost → few-step motivation | Ch4 / Ch6 framing, **not** a result |
| Curvature bias | Ch4 method + own results subsection (D3) |
| Shortcut on DynamiCrafter | §5.2 (D3) — **no run backs this yet** |
| Flow + shortcut adapter, no action conditioning | §5.2 (D3) |
| Combined | §5.3 (D4) — likely descoped |
| Ablation + boundary | §5.4 + §6.1 |

---

## Gates — things that must be checked before the draft asserts them

1. **Action-sensitivity of the AVID/DynamiCrafter checkpoint.** Step 1 claims
   "this works"; `pg3x72uc` shows convergence and gate health, not action-use.
   Planning needs the model to *discriminate between action sequences* — if it
   is weakly action-sensitive, every candidate rollout scores alike and
   planning degenerates to random search. An eval pass on an existing
   checkpoint; no training. Compounding risk: MetaWorld is classified
   action-**redundant** in [[ablation-axes]] Axis 1, the worst case for a
   planning demo. If flat there, move the demo to Push Cube.
2. **A positive control for the ablation.** If every cell base-clones, "adapters
   cannot do this" is indistinguishable from "our pipeline has a bug" — and
   that is the first question at the viva. Need at least one cell where the
   adapter provably learns something the frozen base cannot produce.
   `pg3x72uc` is a positive control for *composition mechanics*; the missing
   one is for *action-use*.
3. **Sourced cost numbers** for step 3 (NFE × wall-clock), against an honest
   few-step baseline, not only 50-step DDIM.
4. **Step 5's "shortcut improves DynamiCrafter efficiency a bit"** is currently
   a premise without a run — the AVID shortcut run was volatile/cautionary and
   no clean few-step curve is in the vault
   ([[../experiments/avid-shortcut-anchor045-volatile-loss]]). Write the arc so
   the theory node carries this step if the run never lands.

## Metric to adopt

**Action Error Ratio** (AVID §4.2): train an action predictor on *real* videos;
report its error on generated videos ÷ its error on real videos. Stronger than
our shuffled/zeroed-action loss gap and AVID-comparable, so the D2 table can
carry an AVID-replica row against their published baselines (ControlNet,
ControlNet-Small, action-conditioned-from-scratch, Product-of-Experts,
action-CFG). Add to the metric list in [[ablation-axes]].

## Related

- [[storyline-experiment-requirements]] — **which experiments each node needs**, with status, tickets, and the minimum viable set
- [[ablation-axes]] — the experimental design this narrative rests on
- [[../related-work/avid]] — step 1 and step 2's source
- [[../theory/shortcut-v-averaging-bias]] — step 4, the proven node
- [[../../10_now/positioning]] — deliverables + contribution surface
- [[../../70_Thesis/outline]] — the chapter structure this maps onto
