---
type: decision
status: open
created: 2026-07-21
decided_at:
updated: 2026-07-21
target_date:
scope: benchmark / thesis-scope
related:
  - "[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]"
  - "[[../../20_Tickets/done/exp-conditioning-action-shuffle-ablation]]"
  - "[[../../20_Tickets/chore-data-action-frame-alignment-audit]]"
  - "[[../../10_now/positioning]]"
---

# Decision: add an action-informative benchmark (ACWM-Phys) alongside/instead of MetaWorld for D2 claim (a)

## What's already decided (grilling session, 2026-07-21)

- **D2 evidence strategy**: target claim **(a)** "adapters make a frozen
  video model action-following on ≥1 benchmark"; fallback (b)+(c) — see
  [[../../10_now/positioning]] §D2.
- **User: the (a) result may come from other envs — MetaWorld is NOT
  required.** A second, action-informative benchmark is the primary (a)
  vehicle; MetaWorld keeps the diagnostic role (where action-redundancy was
  measured).

## Why (evidence)

The trained xattn adapter is measured fully action-blind on MetaWorld
(shuffled/zeroed actions move loss <1e-4 at every σ — 2026-07-21 probe), and
MetaWorld scripted demos plausibly make actions near-redundant given the
anchor frame (deterministic scripted trajectories). If the data doesn't
reward actions, no architecture/objective intervention can produce claim (a)
there. Selection criterion for the new benchmark: **given the anchor frame,
the future must be ambiguous until the actions are seen** (high
I(action; future | anchor)).

## Still open

1. **Which environment(s):** ACWM-Phys Push Cube (da=2, pusher target decides
   the future, published ACWM-DiT baselines: InD MSE 2.919e-3 / PSNR 25.35 at
   50 steps, their Table 1) is the leading candidate; Reacher (da=2) as the
   second. Alternatives considered: re-collect MetaWorld with
   noised/exploratory policies (pure ablation, no external baselines);
   Procgen/Coinrun (AVID setting; off the robotics story).
   **Release verified 2026-07-22** (HF `t1an/ACWM-Phys`): the paper's Push
   Cube = `rigid_dynamics/push_block` (actions [66, 2] in [−1,1];
   `pushcube_2` is the two-pusher da=4 ablation). 1500 ind_train + 100
   ind_test + 100 ood_test episodes, all fixed 66 frames @10 fps, videos
   released at **1024×1024** (not the paper's 240×240 training res) — so we
   DOWNSCALE into Wan's native max_area 589824 regime, no upscaling. ~120 MB
   total for the env. Caveat for any baseline comparison: their Table-1 eval
   is 240×240 full-episode; our protocol (41-frame windows, 768²) differs
   until matched.
2. **Port scope — DECIDED (2026-07-21, grilling): new `Translator` subclass**
   (`ACWMPhysTranslator`), not a conversion script. Verified feasible against
   the release format (HF `t1an/ACWM-Phys`: `episode_{i}.mp4` 240×240 +
   `metadata.pt` with `actions [T, action_dim]`) and the Translator contract
   (`list_episodes`/`load_clip` emitting `video`, `act`, and the latent-cache
   identity keys `env_name`/`episode_idx`/`start_idx`/`frame_stride`).
   `decord` (already a dependency) for mp4 window decoding. Satellites: a
   builder + `--dataset` switch in the train script,
   `conditioning.input_dim: 2` config, prompts entry or frame-only. Also the
   D1-friendly route (second Translator = framework extensibility demo).
3. **Sequencing — DECIDED (2026-07-21): parallel, starting now** — port runs
   locally (CPU-side) while the MetaWorld nobase overfit pair runs on the
   remote GPU. First milestone: Push Cube episodes listed + one smoke batch
   through the Wan2.2 preprocessor with latents cached.
4. **Advisor communication:** dataset addition is a scope change — needs a
   line in the next weekly update/deck.

## Consequences (when decided)

Derive: `chore-data-acwmphys-port` ticket (translator + conversion +
smoke), `exp-adapter-*-acwmphys-pushcube` run ticket, positioning D2 update,
related-work note for the ACWM-Phys paper (pending).
