---
type: experiment
date: 2026-08-03
config: configs/wan22/diffusion_wan22_action_concat_nobase_acwm_robotarm.yaml
commit: uncommitted working tree @ 2026-08-03
wandb_run_id: 6ruz55f6 (concat) vs vy9tcuco (cross-attention control)
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-CONCAT-nobase-run/checkpoints/
status: killed          # 7.36 h, step 1567
deliverable: D2
metrics:
  concat_effect_rel_400_800_1200: "0.00866 / 0.00895 / 0.00975"
  xattn_effect_rel_400_800_1200: "0.01234 / 0.01067 / 0.01056"
  concat_share_1200: 0.16755
  xattn_share_1200: 0.18114
  concat_eval_loss_1200: 0.13324
  xattn_eval_loss_1200: 0.13208
notes: "NEGATIVE RESULT. Channel-concat action injection is BELOW cross-attention at every matched step, on both effect_rel and action-driven share, with identical eval_loss. Does not reproduce GigaWorld-1's 2.2x concat-over-xattn result — plausibly because their concat carries a spatially-aligned control VIDEO while ours broadcasts a 7-DoF vector uniformly across H,W."
---

# Channel-concat action injection does not help on Wan (D2)

## Why it was run

Our own trace measured action tokens surviving cross-attention (44–56%
action-driven across all 10 blocks) and then **drowning at the residual add** —
xattn output RMS ~0.01 against a stream of 1.8–3.0
([[20260731-wan-action-trace-value-pathway-drowns]]). The 2026-08-02 literature
sweep found the consequence measured independently: **GigaWorld-1** on
Wan2.1-1.3B reports cross-attention at **0.1620** trajectory accuracy vs
**0.1576 for no control at all**, against **0.3528** for channel concatenation,
and states *"attention-side action tokens are easily overwhelmed by appearance
and semantic tokens."* Six works reach the same verdict for continuous actions
([[../../00_Inbox/2026-08-02-wan-based-world-models-action-injection]]).

## Result — negative, at matched steps

Single-key config diff vs the token-norm NOBASE arm (`vy9tcuco`): only
`action_injection` and `output_dir` differ (verified by a flattened-YAML key
diff).

| step | **concat** `effect_rel` | share | **xattn** `effect_rel` | share |
|---|---|---|---|---|
| 400 | 0.00866 | 0.136 | **0.01234** | 0.193 |
| 800 | 0.00895 | 0.156 | **0.01067** | 0.192 |
| 1200 | 0.00975 | 0.168 | **0.01056** | 0.181 |

`eval_loss` is identical (0.1332 vs 0.1321), so this is not a training-quality
difference. Concat is **rising** (0.0087 → 0.0098) while xattn is **falling**
(0.0123 → 0.0106) — they would cross around step ~1500–2000 — so "worse", not
"broken". But nothing resembling GigaWorld's 2.2×.

## Why it probably did not transfer

**Our concat and theirs are not the same operation.** GigaWorld and
Wan-Fun-Control concatenate a dense control **video** — Canny edges, depth, pose
maps — spatially aligned with the latent grid. We concatenate a **7-DoF vector
broadcast uniformly across H,W**, so it carries no spatial information: every
latent token in a frame sees the identical constant.

Their concat wins because it tells the model *where*. Ours only tells it *when*
— which the cross-attention token sequence already did. On that reading the two
landing in the same place is expected, and the sources' recommendation was real
but **conditional on spatially-structured conditioning**.

The version that would actually differ is **rendering actions into image
space** (EA-WM renders arm kinematics to the camera view; RynnWorld renders a
hand skeleton as depth video). That supplies spatial grounding and is a much
larger change than a config key. Not attempted — budget.

## What was built (kept, and correct)

- `action_injection: concat` in `backbones/wan/modules/action_model.py`:
  per-frame actions temporally binned (pixel-frames → latent-frames via
  `adaptive_avg_pool1d`, since Wan's VAE compresses time ~4×), broadcast across
  H,W, concatenated on the channel axis before the patch embedding.
- **Per-dim standardisation against running statistics** (frozen in eval).
  Measured on this dataset: actions RMS **0.054**, per-dim 0.0042–0.097 (a 23×
  spread) against ~unit-scale latents — so raw actions would enter **~18×
  quieter** than the channels beside them, the quietest dim ~200×. The
  standardised plane sits at RMS ~0.5.
- **No zero-init**, deliberately: zero-init protects a *pretrained* stream from
  a random branch; this adapter trains from scratch, and with actions worth
  0.45% of the loss a pathway starting at exactly zero risks never growing.
- 5 CPU tests (`tests/test_action_concat_injection.py`).

## Caveats

- n=1 per arm; killed at step 1567 (7.36 h) for budget, so the crossing point
  was never observed.
- The trend favours concat late; a longer run could change the sign. Unresolved.
- ⚠ The first test suite for this feature **passed vacuously** — with the
  default `predict_full=False` the output head is zero-initialised, so the model
  emits exactly 0.0 and every assertion compared zeros to zeros. The suite now
  asserts a non-zero output first. Worth remembering as a standing check.

## Related

- [[20260731-wan-action-trace-value-pathway-drowns]] — the drowning measurement
- [[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]
- [[../../00_Inbox/2026-08-02-wan-based-world-models-action-injection]]
