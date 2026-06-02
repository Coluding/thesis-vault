---
date: 2026-05-28
type: progress
scope: D1  # framework / adapter taxonomy
status: shipped
related:
  - "[[../../30_Knowledge/related-work/unicon]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../30_Knowledge/related-work/hyperalign]]"
  - "[[../../30_Knowledge/related-work/avid]]"
---

# UniCon ready for testing — step-level branch and configs landed

## Progress

The UniCon hidden-state adapter family is now caught up with everything we
learned while exercising HyperAlign and the AVID output adapter. All three
variants in `adapters/hidden_states/unicon.py` (`UniConHiddenStateAdapter`,
`ReplaceDecoderHiddenStateAdapter`, `FullSkipLayerControlAdapter`) gained:

- An adapter-side **step-level conditioning branch**
  (`use_step_level_conditioning`, `step_level_key`, `step_level_hidden_dim`,
  `step_level_transform`) routed through the shared
  `prepare_dynamicrafter_condition` helper — same plumbing the trainer uses
  for HyperAlign and AVID, so the shortcut/distillation loss now works
  uniformly across the three families.
- A baseline config `configs/diffusion_unicon_metaworld.yaml`
  (frozen DynamiCrafter + decoder copy + zero-init connectors, `mask_mix`
  composition, bf16 amp, video logging).
- A shortcut config `configs/diffusion_unicon_shortcut_metaworld.yaml`
  mirroring the HyperAlign shortcut setup
  (paper-faithful `distillation` target, dyadic step schedule, anchor
  probability 0.75).

## Tests

`tests/test_unicon_architecture.py` mirrors the HyperAlign architecture
tests. It builds a 2-block fake U-Net and covers:

- Attachment and module-replica shape (decoder copy for UniCon, full replica
  for `FullSkipControlNet`, decoder-only-no-connectors for
  `ReplaceDecoder`).
- Output shape under all three composition modes (`add`, `replace`,
  `mask_mix`) and backward through the gate path.
- Zero-init guarantees on `ZeroConvConnector` and `ZeroFTConnector` —
  the "identity at init" property the family relies on.
- The new step-level branch: `step_level_embed` shape, the
  positive-`cond_dim` precondition, and an end-to-end check that swapping
  `cond["step_level"]` between two batches produces materially different
  adapter outputs (the signal actually reaches `emb_fuse`, not just the
  embedding combine helper).
- Feature-store invariants (hooks populate / `clear_captured_base_features`
  wipes).

17/17 tests pass. The 21 existing HyperAlign tests are still green.

## What this unlocks

- Drop-in shortcut training for UniCon: same trainer, same loss, just point
  at `diffusion_unicon_shortcut_metaworld.yaml`.
- A first apples-to-apples comparison between the three adapter families on
  the same DynamiCrafter+MetaWorld setup (output-level AVID, hypernetwork
  HyperAlign, hidden-state UniCon).
- The two sibling variants (`ReplaceDecoder`, `FullSkipControlNet`) are
  ready as ablations — they share all infrastructure and only need a
  one-line `architecture:` change in the config.

## Open items

- Connector lineage (`zeroft` / `zeroconv`) — still pending verification
  against the UniCon paper PDF, see the open questions in
  [[../../30_Knowledge/related-work/unicon]].
- Parameter ladder across the three variants — measure once the FLOPs/params
  estimator ticket lands ([[../../20_Tickets/feat-adapter-flops-per-step-estimator]]).
