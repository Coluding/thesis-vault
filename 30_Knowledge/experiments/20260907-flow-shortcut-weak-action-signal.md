---
type: experiment
date: 2026-07-09
config: _needs verification_        # not stored alongside artifacts; flow + shortcut, action-conditioned MetaWorld
commit: _needs verification_        # impl repo at b72a746 or later
wandb_run_id: _needs verification_
ckpt_path: _needs verification_
status: completed                   # produced logged loss curves (W&B screenshots) + sample videos
deliverable: D4                     # action-conditioned (D2) shortcut (D3) flow world model
metrics:                            # eyeballed off exported W&B chart axes — NOT logged scalars
  base_loss: "flat noisy band ~0.10–0.20 over the captured window, NO visible descent"
  total_loss: "flat noisy band ~0.10–0.65, no trend"
  shortcut_direction_per_rung: "monotone in step size — N064 ~0.0005 → N001 ~0.1–0.5; only N001/N002 show any early descent"
notes: FIRST run with a genuinely pretrained WAN base (frozen) — prior runs loaded random WAN weights, so the adapter was learning everything from scratch with no prior. Now the base carries the scene; samples are coherent ("okayish") but the predicted arm does not track the action trajectory. base_loss sits flat ~0.10–0.20 (expected — strong frozen base near its floor). Real question is now the base-vs-adapted delta / action-following, not the base_loss trend. Artifacts in data/results/20260907/.
---

# exp: 2026-07-09 — first real-pretrained WAN base; coherent samples, weak action conditioning

Related: [[20260629-flow-vs-diffusion-shortcut-samples]] ·
[[avid-shortcut-anchor045-volatile-loss]] ·
[[../../10_now/product-state]] · [[../../10_now/architecture]] ·
[[../theory/shortcut-v-averaging-bias]]

> **User read (2026-07-09):** "loss is not really going down, the videos are
> okayish but the adapter is not really doing much — the action signal is not
> being picked up properly." This note documents the run and works through that
> hypothesis. The diagnosis section is an **analysed estimate** (shown reasoning),
> not a measured conclusion — the decisive test (§Decisive next tests) has not
> been run yet.

## What changed in this run (the key variable)

**This is the first run with a genuinely pretrained WAN model as the frozen
base** (user-stated, 2026-07-09). In all prior runs the "loaded" WAN model was
**emitting random weights** — so the composition `f = f_base + g(d)·Δ_φ` had a
*useless* `f_base`, and the adapter was effectively learning the entire
prediction from scratch with **no prior**. That is why earlier runs
(e.g. [[20260629-flow-vs-diffusion-shortcut-samples]]) showed a large `base_loss`
descent (~0.4 → ~0.13): the adapter+random-base was learning everything.

Now `f_base` is a real pretrained video prior. This **inverts how the loss curves
should be read** (see below) and is almost certainly why the samples are now
*coherent* (real arm / table / hoop) instead of the blur-and-colour-drift of the
random-base era.

## What ran

A **flow-matching + shortcut, action-conditioned** run with a **real pretrained
WAN frozen base** on MetaWorld, artifacts under `data/results/20260907/`:

| Artifact | Content |
|---|---|
| `Screenshot From 2026-07-09 15-20-23.png` | `train/shortcut_direction_loss/N064…N002` per-rung curves |
| `Screenshot From 2026-07-09 15-20-50.png` | `shortcut_direction_loss/N001`, pooled `shortcut_direction_loss`, `train/loss`, `train/base_loss` |
| `*.mp4` (×3) | Sample rollout grids, 2304×4608 portrait, 96 frames @ 3 fps, **6 rows × 3 cols** |

Loss semantics confirmed from the trainer (impl repo `training/trainer.py`, this
is the **flow** branch):
- **`base_loss`** (trainer.py:278) = the standard flow-matching denoising loss on
  the composed prediction `f = f_base + g(d)·Δ_φ`. **This is the objective the
  action-conditioned adapter is supposed to reduce.**
- **`train/loss`** = `base_loss` + weighted consistency terms.
- **`shortcut_direction_loss/N{n}`** (trainer.py:339) = D3 consistency term, `n =
  round(1/d)` steps; logged per step-size rung.

Config / wandb id / commit / ckpt not stored with the artifacts → `_needs
verification_`.

## What was observed

### Loss curves (eyeballed off W&B chart axes — not logged scalars)

- **`train/base_loss` — flat, noisy, ~0.10–0.20** (spikes to ~0.35), no downward
  trend. **With the real pretrained base this is now the *expected* shape, not the
  symptom** — see the reframing below. A strong frozen prior already sits near its
  own denoising floor, so there is little absolute loss left for the adapter to
  remove; a flat aggregate `base_loss` says almost nothing about whether the
  adapter works. The right signal is the **base-vs-adapted delta**, not the trend.
- **`train/loss` (total)** — same story, ~0.10–0.65, pure noise, no trend.
- **`shortcut_direction_loss` per rung** — the familiar **monotone-in-step-size**
  signature (cf. [[20260629-flow-vs-diffusion-shortcut-samples]]):

  | Rung | step `d` | loss band | shape |
  |---|---|---|---|
  | N064 | 1/64 | ~0.0002–0.0016 | flat noise |
  | N032 | 1/32 | ~0.0005–0.004 | flat noise |
  | N016 | 1/16 | ~0.001–0.009 | flat noise |
  | N008 | 1/8  | ~0.005–0.03  | flat noise |
  | N004 | 1/4  | ~0.02–0.11   | flat, maybe faint early dip |
  | N002 | 1/2  | ~0.03–0.28   | slight early descent then flat |
  | N001 | 1/1  | ~0.1–0.5     | **descends ~0.4→~0.15 over first ~600 steps**, then flat/noisy |

  So the **only loss component that visibly learns is the coarsest-rung shortcut
  consistency (N001)** — an objective the adapter can satisfy *without using
  actions*. The action-conditioned denoising term (`base_loss`) does not move.

> **⚠ Step-count discrepancy — needs verification.** The per-rung + N001 panels
> span ~2.7k steps; the `base_loss` / pooled / `train/loss` panels span only
> ~880. Either a different x-axis, a different logging cadence, or two runs
> overlaid. Resolve before over-reading the exact step where things flatten. The
> *shapes and magnitudes* above are robust to this; the *step counts* are not.

### Why the flat base_loss is expected now (not a regression)

Earlier I flagged the flat curve as a regression vs the 2026-06-29 clean descent.
**With the real-base context that comparison is apples-to-oranges and the flag is
withdrawn:**

- **2026-06-29:** random-weight base → adapter carries the whole prediction →
  `base_loss` starts high (~0.4) and descends as the adapter learns everything.
- **2026-07-09:** real pretrained base → base already predicts velocity well →
  `base_loss` starts *already low* (~0.15) with little headroom → flat is normal.

So a flat `base_loss` here is **not** evidence the adapter failed. It is exactly
what a correctly-loaded strong prior looks like. The real question moves to: **is
the adapted model better than the frozen base alone, and does it follow actions?**
Answer that with the base-vs-adapted delta (the `QualityMetricSuite` already
scores adapted-vs-frozen — see [[../../10_now/architecture]]) and the
action-shuffle test, **not** by staring at the `base_loss` trend.

### Sample videos (frames extracted & inspected)

The grid is **6 rows × 3 columns** (**user-confirmed layout 2026-07-09**):
**rows = denoising-step budget, few→many (1 step at top → 50 at bottom)**;
**columns = GT | base(frozen) | adapted(base+adapter)**. So each row is an
NFE-matched GT/base/adapted triple, and reading *down* a column shows quality vs
step count — this is the **shortcut few-step-rollout eval grid** (the D3 view).
Column 1 (GT) is identical down the rows because it's the same clip repeated per
step budget.

- **Column 1 (GT):** sharp Sawyer arm, gripper, orange puck, wooden table, hoop.
- **Columns 2–3 (base / adapted):** **blurred and ghosted**, and the blur is
  **worst in the top (few-step) rows and improves downward** as steps increase —
  the expected NFE→quality gradient. Top rows (1–few steps) are mush; only the
  many-step rows approach coherent structure.
- **Few-step gap (D3):** the whole point of the shortcut adapter is to make the
  *top* rows good; they are not yet. Consistent with the loss — N001 (1-step)
  `shortcut_direction_loss` is the largest, slowest-descending rung.
- **Action-relevant signal (NFE-matched):** even at the better (many-step) rows
  the **adapted arm pose diverges from the GT column** — GT reaches *down toward
  the puck*, the prediction swings *up/back* and smears. Averaging over motions,
  not committing to the action-specified one.

This visual divergence is **evidence consistent with weak action-following**, not
proof — a poorly-trained generator can also blur. The pose divergence (not just
blur) is the action-specific part.

### Second task — button-press, base-vs-adapted grid (`button/`, later run)

A second run on the **button-press** MetaWorld task (yellow button box), artifacts
under `data/results/20260907/button/` (2 sample videos + the same loss-panel
layout; loss curves are the **same story** — flat `base_loss` ~0.1–0.2, N001
shortcut descends ~0.4→~0.15 then flat, per-rung monotone). **User read: "a bit
better qualitatively — look at the adapter in the third column."**

Same grid convention: **rows = denoising steps (1→50 down), columns = GT | base |
adapted** (user-confirmed). So col2-vs-col3 *within a row* is an **NFE-matched
base-vs-adapted comparison**. What I see across both button videos:

- **Adapted (col3) is more task-directed than base (col2), at matched NFE.** In
  the many-step (lower) rows the col3 arm reaches down onto / engages the yellow
  button box; the col2 base wanders more — arm swung up/back or away from the box.
- **Few-step rows (top) are mush** for both columns — the shortcut few-step
  quality isn't there yet (the D3 gap).
- Both remain **blurry/ghosted** vs the crisp GT; col3 is the more action-directed
  of the two.

**Positive update: the adapter is not inert.** At matched step count it adds
task-directed structure the frozen base alone does not produce (col3 > col2). That
*weakens* the "inert adapter / actions fully ignored" reading and shifts the
problem to **(a) effect-magnitude + fidelity** and **(b) few-step transfer**.

> **User assessment (2026-07-11):** on the button eval, **the adapted rollout
> looks clearly better than the frozen base — despite `base_loss` not
> descending.** This is the reframe made concrete: with a strong frozen base the
> loss trend is near-uninformative (base already at its floor); the **base-vs-adapted
> eval is the right yardstick, and it's positive.** Calibration note: from
> extracted stills the adapted is at-least-as-good and often more box-engaged, but
> "clearly" is best judged on the temporal motion (full video) + the metric below,
> not single frames.
>
> **Quantify it — the number already exists.** The config scores
> `quality_metrics: [psnr, ssim, lpips, mse]` (+ fid/fvd) on **both** the adapted
> and the frozen-base rollout every eval cycle. See the delta below.

### Quantified base-vs-adapted delta (2026-07-11, eval-metric screenshots)

From `data/results/20260907/button/{adapted_eval,base_eval}.png` (W&B eval curves
over ~0.5k–3.8k steps; **numbers eyeballed off the chart axes, not logged
scalars**). The frozen base is constant-in-expectation (its curves are flat
noise); the adapted curves move with training. **The result splits cleanly by
metric type:**

| metric | base (frozen, flat) | adapted (trend) | adapter verdict |
|---|---|---|---|
| **PSNR** ↑ | ~15.6 | ~15.7 → **16.8**, rising post-2.5k | **wins, +~1.2 dB, widening** |
| **SSIM** ↑ | ~0.80 | ~0.815 → **0.833**, rising | **wins, growing** |
| **MSE** ↓ | ~0.0275 | ~0.0265 → **0.021**, falling | **wins, growing** |
| **LPIPS** ↓ | ~0.357 | 0.345 → **0.40**, rising | **loses** (degrades over training) |
| **FVD** ↓ | ~1250 | 1300 → **~1850**, rising | **loses** |
| **FID** ↓ | ~75 | 73 → **~100**, rising | **loses** |

**Reading — regression to the mean.** The adapter clearly beats the frozen base
on **reconstruction** (PSNR/SSIM/MSE) and the advantage **widens with training**
(base flat) → the adapter is measurably learning to predict the true future
better. **This overturns "the adapter isn't doing much."** But it simultaneously
**degrades perceptual/distribution** metrics (LPIPS/FID/FVD) as it trains — the
textbook signature of **reducing per-pixel error by blurring toward the
conditional mean** (lowers L2 / raises PSNR, kills realism). This *explains* the
visual ghosting/smear.

**Implications:**
1. **For a world model for planning, PSNR/MSE is the metric that counts** — and
   there the adapter clearly helps. The perceptual loss is a known, separable
   MSE-objective tradeoff, not an adapter failure.
2. **The blur may be partly the shortcut `distillation` target** (distillation-to-
   mean blurs) → ties the realism loss to D3; the action-free shortcut isolation
   ([[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]]) + a no-shortcut
   control ([[../../20_Tickets/experiments/exp-shortcut-zero-weight-control-run]]) would show
   how much blur the shortcut term causes.
3. A growing PSNR advantage is **suggestive of action-following** (PSNR = closeness
   to the real action-determined future) but the shuffle test
   ([[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]]) still proves
   it's action-driven vs a generically sharper prediction.

> **⚠ Unresolved — task-prior vs action-following.** col3 > col2 could be the
> adapter genuinely *reading the action vector*, OR a learned **scene/task prior**
> ("arm presses the button here") it would produce *for any action*. A GT-vs-pred
> grid cannot tell these apart. Only a **counterfactual** (feed a different / zeroed
> / swapped action, watch if col3 changes) can — which is exactly what the
> interactive debug UI ([[../../20_Tickets/feat-eval-interactive-action-debug-ui]])
> and the shuffle ablation ([[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]])
> are for.

## Reading (analysed estimate — shown reasoning, ranked by how I'd test)

The real signal is the **samples**, not the loss: coherent scene (from the base
prior). The button-press grid shows the adapter (col3) is **more task-directed
than the base (col2)** — so the adapter *is* contributing action-relevant
structure — but the effect is small and fidelity is poor. So the framing is
**"the adapter helps but not enough,"** not "the adapter is dead."

1. **Strong frozen base under-incentivises the adapter (leading hypothesis).**
   Now that `f_base` is a real prior, the base already explains most of the loss.
   The gradient pressure on `Δ_φ` is proportional to the *residual* the base
   can't predict — which is small — so the adapter learns only a weak correction:
   enough to nudge the arm toward the task (col3 > col2) but not enough to
   sharply follow the action trajectory. This fits the flat aggregate `base_loss`
   (little headroom) **and** the small-but-real col3-vs-col2 gap. **Test:**
   quantify the base-vs-adapted delta + action-shuffle (below). **Fix directions:**
   up-weight the adapter's effective learning (higher adapter LR / gate init), or
   *force* the base to need actions — condition-dropout on the base path, or a
   residual/base-subtracted target so the adapter trains on the action-dependent
   remainder the base can't predict.

2. **Adapter contribution small (not zero).** The button grid rules out a fully
   inert adapter (col3 ≠ col2), but the contribution may still be too small.
   **Test:** log `‖g·Δ‖ / ‖f_base‖` + gate value — expect small-but-nonzero, and
   watch whether it's still growing or has saturated low.

3. **Conditioning path broken / actions ignored.** The adapter runs but never
   sees usable actions — encoder collapse, condition-dropout too high, or the
   action embedding not reaching cross-attention. **Test:** shuffle / zero actions
   and compare samples + base-vs-adapted delta. No change ⇒ actions ignored.
   Distinguished from (1) by (2): inert-but-would-use-actions vs active-but-blind.

4. **Shortcut term captures the gradient budget.** The only visibly-learning
   component is coarse-rung shortcut consistency (N001) — satisfiable *without
   actions* (predict the base's own trajectory). The adapter may be spending
   capacity on consistency while the action-conditioned residual stays unlearned.
   **Test:** `shortcut_direction_weight = 0` control (pure action-conditioned
   flow) — does action-following improve?

5. **Action data content.** Lower priority now that the base-model swap is the
   known changed variable, but still cheap to rule out: action↔frame temporal
   alignment (off-by-one), normalisation stats, action diversity. **Test:** audit
   the dataloader / preprocessor.

6. **Flow-loss noise masking a small trend.** Flow-matching train loss with
   per-sample random `t` is high-variance; use **`eval_base_loss`** (held-out,
   averaged — the trainer computes it) rather than the raw train hairball for any
   trend read. Cheap sanity check.

## Decisive next tests (cheapest → most informative)

1. **Base-vs-adapted delta.** With a real frozen base this is *the* metric: score
   the frozen base alone vs base+adapter on the same eval (the `QualityMetricSuite`
   already does adapted-vs-frozen). If the adapter adds ~nothing over the base, the
   under-incentivisation / inert-adapter story is confirmed. **Do this first — it
   reframes everything the `base_loss` trend can't.**
2. **Action-shuffle / zero-action ablation.** Permute actions across the batch;
   compare samples + base-vs-adapted delta. Unchanged ⇒ adapter ignores actions.
3. **Adapter-magnitude logging.** `‖g·Δ‖ / ‖f_base‖` + gate value over training —
   separates inert (≈0) from active-but-blind.
4. **`shortcut_direction_weight = 0` control run.**
5. **Data-alignment audit** (lower priority — base swap is the known variable).

### For the separate D3 ("does the shortcut work?") question

The few-step gap in the NFE-row grid is confounded (undertraining vs base
strength vs shortcut). The clean isolation is an **action-free, shortcut-only
adapter** ([[../../20_Tickets/experiments/exp-shortcut-action-free-isolation]]): strip actions,
keep only step-size conditioning + shortcut losses, and ask whether the adapter
can distill the frozen base into a good few-step generator. Success = the top
(few-step) rows get good and beat the frozen base at matched low NFE. This is the
purest D3 test and decouples it from the action-conditioning question above.

## Open / needs verification

- ~~Fresh run or warm-start?~~ **Resolved 2026-07-09: fresh / from-scratch.**
- ~~Is flat base_loss a failure?~~ **Reframed 2026-07-09: expected** with a real
  frozen base; the base-vs-adapted delta is the real metric, not the trend.
- **Which WAN checkpoint** is the base, and confirmation the frozen weights are
  now genuinely loaded (not random) — the whole reframing rests on this
  user-stated fact.
- Config, commit, wandb id, ckpt path.
- Grid column semantics (GT vs which prediction / NFE per column).
- `shortcut_direction_weight`, `drop_condition_prob`, adapter LR for this run.
- Action data: episode count, normalisation stats, action↔frame alignment.
