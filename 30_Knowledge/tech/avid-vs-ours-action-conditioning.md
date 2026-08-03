---
type: tech
status: living
last_updated: 2026-07-30
sources: ["[[../../20_Tickets/experiments/exp-backbone-dc-robotarm-run]]", "[[ablation-axes]]"]
---

# AVID vs ours — DynamiCrafter action-conditioning path

Comparison of how the **original AVID** (latent/DynamiCrafter variant) injects
actions vs how **our DC output adapter** does it, prompted by a felt "something
is different" in action-following. Verified against source 2026-07-27 (file:line
below). Deliverable D2, DynamiCrafter base family.

## Headline: concat-of-halves (AVID) vs add-of-fulls (ours)

The one meaningful structural divergence is **how the action embedding combines
with the time embedding** inside the UNet ResBlocks:

- **AVID** (`add_act_time_emb=False`, its default):
  `emb = cat([time_emb₆₄, act_emb₆₄])` → time and action occupy **orthogonal
  64-dim subspaces**.
  `external_repos/avid/latent_diffusion/libs/dynamicrafter/lvdm/modules/networks/openaimodel3d.py:744`
- **Ours** (`add_act_time_emb` **forced True** at
  `src/generative_flow_adapters/adapters/output/dynamicrafter.py:74`):
  `emb = time_emb₁₂₈ + cond_emb₁₂₈` → time and action are **superimposed in the
  same 128-dim space**.
  `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py:794-797`

**Why we force add:** we feed a *full-width* (128) `adapter_embedding` from our
condition encoder rather than AVID's half-width (64) `act_emb`; concat of a
full-width cond with the time emb would double the width and break the ResBlock
addition (`dynamicrafter.py:68-74`). So the add is a deliberate, dimensionally
*correct* workaround — **not a bug** — but it changes the semantics: in the add
form a large time signal can **swamp** the action, whereas concat reserves the
action its own subspace. This is the most plausible reason actions are relatively
weaker in our DC adapter.

## The width question — resolved, no bug

`adapter_condition_output_dim` is sized off `null_action_emb.shape[1]`
(`openaimodel3d.py:441`) and the add-branch is forced, so the projected
conditioning is exactly `embed_dim` (128) and the ResBlock add is dimensionally
correct. The `:68-72` comment's concern is correctly handled; the only
consequence is the concat→add semantic change above.

## Verified differences

| # | Dimension | AVID (latent, MetaWorld ref) | Ours (DC ACWM) | Impact |
|---|---|---|---|---|
| 1 | **time⊕action** | **concat halves** (orthogonal 64+64) | **add fulls** (superimposed 128) | **MED — the divergence** |
| 2 | width | native halves | forced-add, consistent | none (no bug) |
| 3 | action encoder | 2-layer MLP →64 (`openaimodel3d.py:420-424`) | 4-layer structured →512→128 (`encoders.py:125-136`, `openaimodel3d.py:462-466`) | LOW |
| 4 | per-frame | `[b,T,A]` per-frame | `[b,T,A]`→512 per-frame | LOW (both per-frame) |
| 5 | discrete/continuous | **continuous** MLP | continuous MLP | none (both continuous) |
| 9 | action scaling | raw per-frame | **stride-summed if stride>1** (`acwm_phys.py:195`) | LOW–MED — keep `frame_stride:1` for parity |
| 10 | raw-`act` UNet branch | live (the path) | **dead** — UNet's own 7→128 `action_embed` unused; we go via `adapter_embedding` (`openaimodel3d.py:784,789`) | none functionally, wasted params |
| 8 | composition | `base·m + act·(1−m)`, `m=σ(gate+bias)`, bias 0 (`avid.py:136-137`) | `base·g + adapter·(1−g)`, `g=σ(gate+bias)`, bias 0 (`adapted_model.py:226-230`) | **identical form** |

Both bases are `action_conditioned=False` (action enters only through the
adapter); both use action dropout 0.0; neither trains a null-action CFG.

## Status 2026-07-30 — this hypothesis is now the prime suspect, and still untested

[[../experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]] showed
the **unmodified AVID recipe follows actions on ACWM Robot Arm** (effect_rel
0.029475, null 0) where our adapters are blind (DC `c3pcewxk` 0.0034) — same
frozen base weights, same episodes, same probe. That removes the data/OOD
explanation which displaced this note between 07-28 and 07-29, and promotes the
concat-vs-add divergence below to the leading candidate.

**The `concat` config has never been launched** (verified 2026-07-30: the
`dc-acwm-robotarm` wandb project holds only `kjgt3z0f`, `u9u7kxia`, `c3pcewxk` —
all `action_time_combine: add`, all crashed). Ticket:
[[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]]
(arm 1). ⚠ `diffusion_dc_acwm_robotarm_concat.yaml` as shipped also flips
`use_step_level_conditioning` and `shortcut_anchor_prob` — flip **only**
`action_time_combine` for a clean read.

## Implication → testable experiment (IMPLEMENTED 2026-07-27)

The concat-vs-add difference is now a **config toggle** —
`adapter.extra.action_time_combine: add | concat` (default `add`, preserves
current behaviour). `concat` sizes `time_embed_dim` and the adapter-condition
projection to `embed_dim//2` so `cat([time₆₄, cond₆₄]) = embed_dim`, matching
AVID exactly (verified: both modes yield emb width 128 = `ResBlock.emb_channels`).
Code: `adapters/output/dynamicrafter.py`, `backbones/.../openaimodel3d.py`
(action_conditioned=False branch), wired via `adapters/factory.py`.

Experiment vehicle: `configs/dynamicrafter/diffusion_dc_acwm_robotarm_concat.yaml`
(vs the `add` baseline `..._robotarm.yaml`). If action-following recovers under
`concat`, difference #1 was the culprit — the `action_inject_grad_norm` /
`eval_action_effect_rel` traces show it directly. See
[[../../20_Tickets/experiments/exp-backbone-dc-robotarm-run]].

## Reference run (ground truth)

The original AVID latent repo can run on our ACWM data as a baseline — latent
AVID is **continuous-action** (compatible, no discretization needed), the DC ckpt
`ckts/dynami512.ckpt` is present, and the MetaWorld harness already ran once
(2026-07-15). Needs a ~1-file `ACWMVideoDataModule` shim (mp4+metadata.pt → AVID
batch dict) + the AVID env (torch 2.1 / PL 1.9.3, separate from the main `.venv`).
