---
type: experiment
date: 2026-06-17
config: data/results/20261706/config_run_anchor_prob=045.yaml  # diffusion_avid_shortcut_metaworld
commit: _needs verification_
wandb_run_id: _needs verification_   # wandb project: avid-shortcut-metaworld-0.45
ckpt_path: outputs/diffusion_avid_shortcut_metaworld  # config default — actual path _needs verification_
status: running
deliverable: D3
metrics:                # read off exported W&B charts (axis-eyeballed, not logged scalars)
  train_base_loss: "~0.06–0.07, smooth/stable"
  train_shortcut_direction_loss: "pooled scalar volatile, ~0.01–0.12 — RESOLVED as step-size mixing (see per-rung breakdown)"
  train_total_loss: "~0.07 with frequent spikes to ~0.2"
  shortcut_direction_loss_per_rung: "monotone with step size: N064≈0.002 → N002≈0.03 (~15× spread); each rung individually well-behaved"
notes: Larger MetaWorld dataset; shortcut got worse + loss unstable. Per-step-size shortcut-loss logging RESOLVED the volatility as a step-size-mixing artifact (data/results/20262206/loss_new.png).
---

# exp: AVID shortcut adapter on larger MetaWorld — volatile shortcut loss, degraded prediction

Related: [[../../20_Tickets/experiments/exp-shortcut-vs-image-only-anchor-baseline]] ·
[[../../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]] ·
[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]

## What ran

Real-backbone shortcut run — **DynamiCrafter512** base (frozen, velocity
prediction) + **output adapter** (`composition: avid_mask_mix`, `hidden_dim:
512`, output mask on) with **step-level conditioning**
(`use_step_level_conditioning: true`, `step_level_transform: log2`,
`step_level_hidden_dim: 64`). Action conditioning via structured encoder
(`action_dim: 4`, MLP). Trained on MetaWorld with **all envs and cameras**
(env/camera commented out in the config) — the "larger dataset" relative to the
earlier small-data run. `frame_stride: 4`.

Key shortcut settings:
- `shortcut_anchor_prob: 0.45` — ~45% of steps run anchor mode (finest step,
  standard loss only); the rest get coarse step levels + a consistency target.
- `shortcut_target_method: distillation`
- `shortcut_direction_weight: 1.0`
- `shortcut_step_schedule`: normalized, log2, min `1/128` (finest, grounded by
  the standard loss) → max `1` (one-step), base 2.
- `eval_metric: base_loss` (honest denoising loss drives `best.pt` selection,
  not the self-distilled total).

Run artifacts: `data/results/20261706/` — config + three W&B chart PNGs +
three sample-grid PNGs (`image (4|5|6).png`). wandb project
`avid-shortcut-metaworld-0.45`.

## What was observed

**Training curves** (from the exported W&B charts, ~1.9k steps; values
eyeballed off the axes, not logged scalars):

- `train/base_loss` — smooth, monotone-ish descent from ~0.4 to **~0.06–0.07**,
  stable. The honest denoising objective is learning fine.
- `train/shortcut_direction_loss` — **highly volatile**: spikes between ~0.01
  and ~0.12 across the entire run, no clean downward trend.
- `train/loss` (total) — settles ~0.07 but with frequent spikes to ~0.2; the
  spikes track the shortcut-direction term, not the base term.

**Samples** (`image (4|5|6).png`, multi-step rollout grids): some frames show a
coherent robot arm, but others are **collapsed / foggy** or show **red drift
artifacts** on the arm — degraded prediction quality. _The exact
column/NFE/ground-truth layout of the grids is `_needs verification_`._

## Reading

- The instability lives in the **shortcut/self-consistency term**, not the base
  denoising loss. Base loss converges cleanly and stably; the total loss's
  spikiness is inherited from `shortcut_direction_loss`.
- On the **larger** dataset the prediction got **worse** — consistent with the
  earlier qualitative "robust across NFE" read being a **small-data overfitting
  artifact** that does not survive more data. _(The small-data comparison run's
  pointers — wandb id / ckpt / dataset size — are `_needs verification_`.)_
- **CONFIRMED (2026-06-24): the pooled-scalar volatility was step-size
  mixing**, not genuine instability — see the per-rung breakdown below.

## Per-step-size breakdown — RESOLVED (2026-06-24)

Source: `data/results/20262206/loss_new.png` — six per-rung
`train/shortcut_direction_loss/N{steps}` curves over ~1.6k steps (legend
`diffusion_avid_shortcut_metaworld`; axis-eyeballed, not logged scalars). `N` =
number of steps = `1/d`, so **N064 is the finest step (d=1/64), N002 the
coarsest (d=1/2)**.

| Rung | step `d` | loss band | shape |
|---|---|---|---|
| N064 | 1/64 | ~0.001–0.003 | peak then gentle **downtrend** |
| N032 | 1/32 | ~0.002–0.003 | slow decline |
| N016 | 1/16 | ~0.002–0.005 | slight decline |
| N008 | 1/8  | ~0.004–0.008 | roughly flat |
| N004 | 1/4  | ~0.008–0.016 | flat |
| N002 | 1/2  | ~0.02–0.05  | flat, noisy |

**Reading:**

1. **Volatility = step-size mixing (the diagnosis question, answered).** Loss
   magnitude scales monotonically with step size — a **~15×** spread between
   the coarsest (N002 ~0.03) and finest (N064 ~0.002) rung. One `s_full` is
   drawn per batch, so the pooled scalar bounces between ~0.002 and ~0.03+
   purely from *which rung was sampled*, reproducing the observed 0.01–0.12
   pooled range. Each rung is individually well-behaved — no order-of-magnitude
   jumps within a rung. This is **Case A** from the diagnosis ticket.
2. **Secondary (not this ticket's question): coarse rungs plateau, fine rungs
   decline.** N064/N032 trend down; N002/N004/N008 are roughly flat. A
   plateau concentrated at the few-step/coarse rungs is the signature predicted
   by the **v-averaging target bias**
   ([[../../20_Tickets/bug-losses-shortcut-v-averaging-target]],
   [[../theory/shortcut-v-averaging-bias]]) — that bug remains the real fix for
   coarse-rung *convergence*, distinct from this *volatility* finding.

**Consequences:**

- The scaled rerun ([[../../20_Tickets/experiments/exp-shortcut-scale-episodes-longer-train]])
  is no longer gated on "is the objective stable?" — it is, the noise was a
  pooling artifact. It should still fold in (a) per-step-size loss
  reweighting/normalisation before pooling
  ([[../../20_Tickets/feat-shortcut-per-stepsize-loss-reweighting]]) and
  ideally (b) the endpoint-inversion target fix.
- Diagnosis ticket [[../../20_Tickets/experiments/exp-shortcut-per-stepsize-loss-diagnosis]]
  is answered (Case A).

## Implications

- Reframes [[../../20_Tickets/experiments/exp-shortcut-vs-image-only-anchor-baseline]]: its
  premise ("looks step-count robust qualitatively") no longer holds on real
  data. The relevant question shifts toward *whether the shortcut term helps at
  all* once not overfitting, and *whether the volatility is a logging/scaling
  artifact or real instability*.
- Feeds [[../../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]
  (anchor + warmup as the chosen mitigation — anchor_prob is the lever here).

## Open / needs verification

- wandb run id, git commit, actual ckpt path.
- Small-data comparison run pointers + both dataset sizes (episodes/frames).
- Sample-grid layout (which column is GT vs prediction vs which NFE).
- ~~Per-step-size loss breakdown result~~ — RESOLVED 2026-06-24 (Case A, above).
