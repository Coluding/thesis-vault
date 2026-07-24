---
type: experiment
date: 2026-06-29
config: _needs verification_   # three runs; configs not stored alongside the artifacts
commit: _needs verification_
wandb_run_id: _needs verification_
ckpt_path: _needs verification_
status: completed              # runs produced logged loss curves + sample videos
deliverable: D3
metrics:                       # eyeballed off exported W&B chart axes — NOT logged scalars
  flow_shortcut_base_loss: "~0.4 → ~0.13–0.15 over ~13–15k steps, smooth/stable"
  flow_shortcut_total_loss: "~0.2, spikes to ~0.6–0.7 (tracks shortcut term)"
  flow_shortcut_direction_per_rung: "monotone in step size: N064 ~0.0002–0.0007 → N001 ~0.2–0.42 (~3 orders spread)"
  flow_noshortcut_loss: "~0.42 → ~0.15–0.2 over ~15k steps, smooth/stable"
  diffusion_shortcut_base_loss: "~0.4 → ~0.07 over ~1.4k steps, smooth/stable"
  diffusion_shortcut_direction: "near-zero baseline with sparse spikes to ~1.2 (anchor-mode switching)"
notes: First sample-quality batch on the post-pivot flow-matching base. Loss okay-ish across all three; sample videos still poor (blur / fog / colour drift). Artifacts in data/results/20260629/.
---

# exp: 2026-06-29 — flow vs diffusion shortcut sample quality (first post-pivot batch)

Related: [[../../60_Updates/entries/2026-06-19-pivot-flow-matching-base]] ·
[[avid-shortcut-anchor045-volatile-loss]] ·
[[../theory/shortcut-v-averaging-bias]] ·
[[../../20_Tickets/feat-shortcut-per-stepsize-loss-reweighting]]

## What ran

Three runs, artifacts under `data/results/20260629/`:

| Condition | Path | Resolution | Frames | Loss curves |
|---|---|---|---|---|
| **Flow matching + shortcut** | `flow_matching/shortcut/` | 768×1280 | 16 | `loss1.png` (per-rung), `loss2.png` (pooled + base) |
| **Flow matching, no shortcut** (baseline) | `flow_matching/no_shortcut/` | 768×1280 | 16 | `image (9).png` (loss + base_loss) |
| **Diffusion + inversion shortcut** | `diffusion/shortcut/` | 1536×1600 | 16 | `image (10).png` (pooled), `image (11).png` (per-rung) |

The diffusion run uses an **inversion-based shortcut objective** (user-stated);
the flow runs use the standard flow-matching shortcut. This is the **first
sample-quality batch on the flow-matching base** chosen in the 2026-06-19 pivot
([[../../60_Updates/entries/2026-06-19-pivot-flow-matching-base]]) — flow base is
κ=0, so the v-averaging shortcut bias that motivated the pivot does not apply.

Each sample video is a multi-panel grid of MetaWorld rollouts (red Sawyer arm,
wooden table). The exact grid layout — which columns are ground-truth vs
prediction vs which NFE / step count — is **_needs verification_**.

## What was observed

### Loss curves (eyeballed off exported W&B chart axes — not logged scalars)

**Flow matching + shortcut** (`loss1.png`, `loss2.png`, ~13–15k steps):
- `train/base_loss` — smooth descent ~0.4 → **~0.13–0.15**, stable.
- `train/loss` (total) — settles ~0.2 with spikes to ~0.6–0.7; spikiness tracks
  the shortcut term, not the base term.
- `train/shortcut_direction_loss` per rung — **monotone in step size**, the same
  step-size-mixing signature seen in the AVID diffusion run
  ([[avid-shortcut-anchor045-volatile-loss]]):

  | Rung | step `d` | loss band | shape |
  |---|---|---|---|
  | N064 | 1/64 | ~0.0002–0.0007 | gentle downtrend |
  | N032 | 1/32 | ~0.0004–0.0012 | slow decline |
  | N016 | 1/16 | ~0.0015–0.003 | slight decline |
  | N008 | 1/8  | ~0.003–0.009 | roughly flat |
  | N004 | 1/4  | ~0.015–0.035 | flat |
  | N002 | 1/2  | ~0.05–0.14 | flat, noisy |
  | N001 | 1/1  | ~0.2–0.42 | flat, very noisy |

  ~3 orders of magnitude spread finest→coarsest; the pooled scalar (`loss2.png`)
  consequently bounces ~0.02–0.4 purely from which rung was sampled. Confirms the
  step-size-mixing diagnosis (Case A) carries over to the flow base.

**Flow matching, no shortcut** (`image (9).png`, ~15k steps):
- `train/loss` and `train/base_loss` both descend ~0.42 → **~0.15–0.2**, smooth
  and stable. (No shortcut terms logged — expected for the baseline.)

**Diffusion + inversion shortcut** (`image (10/11).png`, ~1.4k steps — much
earlier/shorter than the flow runs):
- `train/base_loss` — clean descent ~0.4 → **~0.07**, stable.
- `train/shortcut_direction_loss` (pooled and per-rung) — near-zero baseline with
  **sparse spikes to ~1.2**; per-rung curves (N064…N002) all sit near zero with
  intermittent isolated spikes. This is the anchor-mode switching signature (most
  steps anchor → ~0, occasional shortcut steps spike), qualitatively unlike the
  flow run's continuous per-rung bands — consistent with the different (inversion)
  objective.

### Training & sampling speed (user-observed, qualitative)

The flow-matching runs are **much faster than the diffusion run — both to train
and to sample** (user observation; exact steps/sec and wall-clock sampling time
**_needs verification_**). Consistent with the flow base's straight (κ=0) ODE
trajectory needing far fewer integration steps to sample than the diffusion
sampler. This is a concrete practical payoff of the 2026-06-19 pivot
([[../../60_Updates/entries/2026-06-19-pivot-flow-matching-base]]) and matters for
D3/D4 (fast few-step rollout for planning).

**Caveat — not a clean comparison.** This batch does not isolate flow vs
diffusion: the diffusion run was at higher resolution (1536×1600 vs 768×1280) and
a different backbone. A controlled speed measurement (matched model / resolution,
logged steps/sec + sampling NFE and wall-clock) is still **_needs verification_**.

### Sample videos (frame montages inspected, `data/results/20260629/**/*.mp4`)

Qualitatively **poor across all three conditions** — loss is okay-ish but
generation is not there yet:

- **Many panels recover the coarse structure** — red arm + wooden table + rail
  geometry are present and roughly correct in the early frames / better panels.
- **A large fraction are degraded**: blurred / foggy panels, collapsed frames,
  and **colour-drift artifacts** (orange, blue, green, magenta bleed) — strongest
  in the flow runs.
- **Degradation grows across the grid / over frames** — later panels are
  consistently worse, suggesting rollout drift and/or the harder
  prediction/few-step columns failing.
- Flow-shortcut and flow-no-shortcut look broadly similar in quality at this
  stage; the diffusion-inversion grid alternates coherent arm panels with blurry
  ones (likely a GT-vs-prediction column structure — _needs verification_).

## Reading

1. **The pivot didn't break training.** On the flow base, base_loss converges
   cleanly for all three runs and the shortcut term is well-behaved per-rung — the
   v-averaging bias that motivated the pivot is gone (κ=0). Loss health is not the
   blocker.
2. **Sample quality is the blocker, not the loss.** A converging base_loss with
   poor samples points at the adapter / generation path, not the objective:
   candidates are insufficient training (flow ~15k, diffusion only ~1.4k steps),
   resolution / VAE decode, action-conditioning strength, or rollout drift.
3. **Step-size mixing reproduces on flow.** The per-rung monotone spread (~3
   orders) confirms pooled-scalar volatility is a logging artifact, not
   instability — reinforces the need for per-step-size loss reweighting before
   pooling ([[../../20_Tickets/feat-shortcut-per-stepsize-loss-reweighting]]).
4. **Diffusion run is early.** At ~1.4k steps it is not comparable to the ~15k
   flow runs; its sample quality read is preliminary.

## Open / needs verification

- Configs, git commits, wandb run ids, ckpt paths for all three runs.
- Sample-grid layout (GT vs prediction columns, NFE / step-count per column).
- Dataset size (episodes/frames), backbone identity (WAN2.1?) and exact
  inversion-shortcut formulation for the diffusion run.
- Whether flow-shortcut vs flow-no-shortcut differ at matched few-step NFE (the
  D3 question) — not resolvable from these grids without the layout.
