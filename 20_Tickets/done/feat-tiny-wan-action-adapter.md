---
type: feat
scope: adapter
status: done
priority: high
created: 2026-06-23
updated: 2026-06-23
resolution: completed
closed_at: 2026-06-23
related:
  - "[[../30_Knowledge/tech/wan21-model-architecture]]"
  - "[[feat-wan21-backbone-integration]]"
  - "[[../30_Knowledge/related-work/avid]]"
---

# Tiny action-conditioned Wan adapter (AVID-style)

The Wan analogue of the AVID 11M/34M/145M action-conditioned U-Nets: a
structural copy of the Wan DiT at reduced width/depth, used as the trainable
delta on the frozen 1.3B base
(`prediction = base(x_t,t) + tinyWan_Δ(x_t,t,action,d)`).

## Design (decided 2026-06-23)
- Role: **adapter Δ** on the frozen base (not standalone).
- Injection: **AVID-style AdaLN** — `e = time_embed(t) + action_embed(a) +
  step_embed(log2 d)` drives every block's modulation (mirrors AVID adding the
  action MLP to the timestep embedding). Learned `null_action_emb` for CFG.
- Tiers: 11M / 34M / 150M.

## What was built
- NEW `backbones/wan/modules/action_model.py` `ActionWanModel`: reuses the
  vendored `WanAttentionBlock` / `Head` / RoPE; batched `[B,16,T,H,W]` forward;
  `condition_on_base_outputs` concats the base output on channels; zero-init
  head → delta≈0 at init. fp32-forced AdaLN modulation (the vendored block
  asserts fp32 — must disable autocast for that region under bf16).
- NEW `adapters/output/wan.py` `Wan21OutputAdapter` (OutputAdapterInterface):
  reads **raw** `action`/`step_level` from cond (embeds them itself), returns
  the delta. `from_config(wan_adapter_config_path, action_dim)`.
- EDIT `adapters/factory.py`: `backbone: wan` → `Wan21OutputAdapter`.
- NEW tier configs `configs/base/wan_adapter_{11m,34m,150m}.yaml`
  (dim 256/L10/h4 = 11.3M, 448/L10/h8 = 34.2M, 768/L16/h12 = 157M).
- NEW experiment `configs/diffusion_wan_avid_shortcut_metaworld.yaml`.

## Verified
- Tier param counts: 11.3M / 34.2M / 157M.
- Frozen base, identity composition at init (zero-init head), delta moves with
  both action and step_level.
- Real-base GPU training: `adapter=output/wan`, 11.5M trainable (0.81%), loss
  0.63→0.24 over 4 steps (shortcut + action conditioning active).
- Tests: `test_wan_avid_adapter_tiers_and_composition` (+ 32 wan/output tests).

## Update (2026-06-23): made conditioning-agnostic
`ActionWanModel` no longer takes raw `action` — it takes a **fused
`cond_embedding [B, cond_dim]`** from the external condition encoder
(`action_embed` → `cond_proj`, `null_action_emb` → `null_cond_emb`).
`Wan21OutputAdapter` now reads `cond["embedding"]` via
`resolve_condition_embedding` (like the transformer head + DynamiCrafter
adapter), and the factory passes `cond_dim = conditioning.output_dim`. So
adding modalities (proprio / goal / language) = swap the encoder to
`StructuredConditionEncoder`/`MultimodalConditionEncoder`, **no adapter change**.
Step level keeps its dedicated AdaLN path. Condition dropout / CFG is now
handled upstream by the encoder's `null_embedding` (dropped the adapter-side
`action_dropout_prob`). Verified: 12 CPU tests + real-base GPU training (loss
0.63→0.22, 11.66M trainable).

## Follow-ups
- Action/condition CFG at inference (encoder null path exists; sampler unused).
- Few-step shortcut inference sampler (D3) — uses `step_level` to roll out in
  1/2/4/8 steps; not built yet.
