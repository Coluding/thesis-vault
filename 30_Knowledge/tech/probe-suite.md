---
type: tech
status: living
last_updated: 2026-08-03
sources:
  - "src/generative_flow_adapters/evaluation/action_sensitivity.py @ 75721b7"
  - "src/generative_flow_adapters/evaluation/action_structure.py @ 75721b7"
  - "[[../writing/rubric/02-technical-skills]]"
  - "[[../writing/rubric/05-reflection]]"
  - "[[../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]"
  - "[[../experiments/20260731-wan-action-signal-is-a-global-bag]]"
---

# The probe suite — the instrument, its nulls, and its limits

> **Direct source for Ch4 §4.3.** This is the thesis's methodological
> contribution: the standard readouts cannot detect action-blindness, so we
> built instruments that can. Written as an *instrument specification* —
> what each probe measures, its null, its chance level, and the case where
> it misled us — because a validated instrument is what the rubric's
> "advanced and original analyses" means
> ([[../writing/rubric/02-technical-skills]]).
>
> Definitions below are read from the implementation at commit `75721b7`,
> not reconstructed: `src/generative_flow_adapters/evaluation/`.

## 1. Why the suite exists

Every standard readout is blind to whether a conditioned model uses its
conditioning:

| Readout | What it actually tracks | Why it is blind |
|---|---|---|
| Training loss | denoising quality | actions are worth ~0.45% of it |
| Gate / mask mean | how much the adapter contributes | says nothing about *what* it contributes |
| FID / FVD / LPIPS | perceptual quality of samples | a domain correction improves these with zero action information |
| PSNR / MSE / SSIM | pixel fidelity | improves under mean-regression |
| Sample inspection | plausibility | humans cannot see a missing counterfactual |

**The demonstration:** Wan × ACWM beats its frozen base on 6/6 quality
metrics (FVD 1118 → 406, −64%) while all three structure probes sit at
chance ([[../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]).
A model can be a *better predictor* and a *non-conditioned* one at the same
time, and nothing on the standard dashboard distinguishes the two.

## 2. The ladder

The probes form three rungs, and cells dissociate along them. **The
dissociation is the result** — so the ladder must be introduced before any
findings.

| Rung | Question | Instrument |
|---|---|---|
| 1 **Sensitivity** | does the action change the prediction *at all*? | `action_effect_rel` |
| 2 **Structure** | does it change it *in the right way* — directional, temporally placed, spatially localised? | the triad (A, B, C) |
| 3 **Control** | does that convert to rollout-level trajectory control? | rollout-action-swap |

A model can pass rung 1 and fail rung 2 (*sensitivity without control* —
the Wan case). Passing rung 2 does not entail rung 3, which is why the DC
cell's above-chance structure is **not** a control claim.

## 3. Rung 1 — action sensitivity

**Measure.** Relative movement of the prediction under an action
perturbation, over predicted frames only:

```
action_effect_rel = ‖pred_true − pred_perturbed‖ / ‖pred_true‖
```

Masked to `frame_mask == 1`: diffusion-forcing batches clamp observation
frames clean, and including them dilutes every norm toward zero — making an
action-blind model look "weakly sensitive".

**Four perturbation variants**, each isolating a different thing:

| Variant | Construction | What it isolates |
|---|---|---|
| `shuffle` | actions from a *different clip* | the cleanest test — actions stay on-distribution, only the correspondence breaks |
| `zero` | null actions | cheap, but a model trained with condition dropout has *seen* zeros and may treat them as a valid token, so a small zero-gap is weaker evidence |
| `roll` | own actions, rolled by half the sequence | separates "uses action magnitude" from "uses action *timing*" |
| `gauss` | noise matched to the batch's per-key mean/std | off-distribution control |

`shuffle` is the primary readout.

**Secondary readouts reported alongside**, because the primary one is not
interpretable alone:
- `adapter_rel_contribution` = ‖pred − base‖/‖base‖ — an adapter that
  barely moves the output has a trivially small action effect, and reading
  that as action-blindness is a misdiagnosis.
- `loss_gap` — whether the movement is in a *useful* direction. A model can
  have ~0 loss gap and still be action-sensitive (it moves, unhelpfully).

**Threshold.** 1% relative movement, applied to the bootstrap CI, not the
mean: `CI_hi < 0.01` ⇒ action-blind; `CI_lo > 0.01` ⇒ action-sensitive;
straddling ⇒ **inconclusive**, collect more draws. The bar is deliberately
generous — far below anything that could drive a planner — so failing it is
unambiguous while passing it proves little.

**Uncertainty.** Percentile bootstrap 95% CI (2000 iterations) over
(batch × draw) pairs.

## 4. Rung 2 — the structure triad

### Probe A — steering direction

Swap clip B's actions onto clip A's `x_t` and ask whether the prediction
moves *toward what clip B's data implies at that `x_t`*:

```
steer_cos = cos( pred(a_B) − pred(a_A),  target(x0_B | x_t) − target(x0_A | x_t) )
```

≈0 ⇒ the action→output map is arbitrary; >0 ⇒ real steering, possibly weak.
**Chance = 0.0**, reported with the per-clip sd and the sd of the mean.

⚠ **Parameterisation-sensitive.** The "target" differs by prediction type
and getting it wrong silently inverts the probe. Nothing is hardcoded: the
diffusion path substitutes through `DiffusionTrainingObjective.get_target`
(keyed off `model.prediction_type`, so `noise`/`velocity`/`x0` each get
their own — and for `x0`, oppositely signed — direction); the flow path
uses the rectified-flow velocity `noise − x0`.

### Probe B — temporal alignment

Zero the actions in one *pixel*-frame bin; measure where the resulting
prediction delta lands along the *latent* frame axis. A diagonal band means
the adapter learned which frames an action governs; a flat map means it
never did. Directly targets the px→latent correspondence, which matters
because action tokens are per pixel-frame while the DiT operates on latent
frames (Wan compresses 4× temporally).

**Chance** = `(2·tolerance + 1) / latent_frames` — hence 0.200 on one
geometry and 0.313 on another; always quote the cell's own chance level.

### Probe C — spatial concentration

Is the action-driven delta on the moving parts (arm/gripper) or spread
uniformly? **Chance** = the mask's area fraction.

## 5. The controls that make the suite trustworthy

This section is the reason the negative results are *discriminating* rather
than inconclusive, and it should be written out in full.

**(a) Paired draws, enforced.** All variants share one noise draw. If the
backbone re-draws noise per forward, the measured "action effect" is really
noise variance — a failure mode that produces a *plausible non-zero*
number. So it raises rather than warns: `_assert_paired` compares `x_t`
tensors and aborts.

**(b) A frozen-base null control.** The base cannot see the action, so its
loss must be identical across variants. Reported as
`base_null_violation`; anything above `1e-6` (a CUDA-nondeterminism
tolerance, not a fudge) means actions are leaking into the base and **every
number is void**. Measured at exactly 0 throughout the campaign.

**(c) Missing actions are an error, never a zero.** If a preprocessor emits
actions under an unexpected key, the perturbation changes nothing, every
variant equals the reference, and the report confidently declares the model
action-blind — *a measurement of nothing*. So a missing key raises, listing
the available keys. Validated against **every** batch, not just the first,
and always against the originally requested keys.

**(d) Condition dropout is forced off** during probing — it is a training
augmentation that would blank the action on a random subset of forwards and
blur every variant together.

**(e) Empirical chance next to analytic chance.** The frozen-base null is a
*harness* control: it proves the probe measures the adapter, but because
the base delta is identically zero it yields no spread to compare against.
So each probe also carries an empirical chance built by breaking the
correspondence while keeping the marginals — an isotropic direction and a
**disjoint** clip pair for A, shuffled rows for B, a random same-size
region for C. **If empirical and analytic chance disagree, the analytic
chance is wrong and the probe must not be read.**

**(f) A named trap, documented in the code.** The obvious mismatched
pairing for probe A — clip `i`'s delta against clip `i+1`'s steering
direction — is *not* at chance: `dtarget_i` involves clips `(i, i−1)` and
`dtarget_{i+1}` involves `(i+1, i)`, so they share a term and correlate at
about −0.5. The implementation skips to `i+2` to make the pairs disjoint,
which requires batch size ≥ 4.

**(g) Degenerate-input detection.** An adapter that does not respond at all
still yields a well-formed row-normalised matrix of zeros whose argmax is 0
for every row — an alignment score computed on that is an artefact, not a
measurement. Raw `|dpred|` mass is retained *before* normalisation to tell
the two apart.

## 6. Where the instrument misled us — and how it was caught

**This is the part that must be written, not omitted.**

`action_effect_rel` is **monotone in action-pathway gain**. Any intervention
that scales the action pathway — `action_token_norm`, `condition_center` —
raises it whether or not it improves the *information* the adapter
extracts. The vault's own gate control moved the metric 4.8× with the action
path untouched. For a period, several mechanism-fix claims rested on it.

**How it was discriminated:** a temporal *control* probe on the clean-room
A/B. The per-frame arm's diagonal concentration was 0.390 against the
pooled arm's 0.199 (chance 0.200) — and the pooled arm's per-frame response
rows were **bit-identical**, which a gain increase cannot produce. Gain
scales a response; it cannot manufacture per-frame differentiation where
none exists.

**The lesson, transferable:** a scalar sensitivity metric on a conditioned
model cannot separate *more signal* from *louder signal*. Only a probe with
internal structure — one whose null is a *shape*, not a magnitude — can.
This is what rung 2 exists for, and it is why the suite is a ladder rather
than a single number.

## 7. Known limits

- **Rung 3 is thin.** Rollout-action-swap has been run on Wan (null) and
  **not** on DC, which is the cell where rungs 1–2 pass. The strongest
  claim available is therefore per-cell.
- **`effect_rel` is never load-bearing alone** — see §6. It is a screen.
- **Single seed.** Bootstrap CIs are over (batch × draw), not over training
  runs; they do not capture run-to-run variance.
- **Probe B assumes a fixed px→latent ratio** per geometry; chance changes
  with it, so cross-geometry comparison of the raw score is invalid.
- **The suite measures the adapter, not the system.** It says nothing about
  whether a planner could use the model.
- **Not yet run on the binned RT-1 checkpoint** (`0fqjrqjl`) — the
  intervention most likely to change the Wan verdict.

## Related

- [[../writing/rubric/02-technical-skills]] — why this note is worth writing
- [[../writing/rubric/05-reflection]] — the blindness claim it supports
- [[../writing/ablation-axes]] — the metric ladder in the per-cell design
- [[../experiments/20260731-wan-action-signal-is-a-global-bag]] — the triad's first application
- [[../../70_Thesis/outline]] §4.3 — the section this feeds
