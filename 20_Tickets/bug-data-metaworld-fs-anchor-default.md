---
type: bug
scope: data
status: open
priority: medium
created: 2026-05-28
updated: 2026-05-28
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/per-sample-frame-stride-sampling]]"
  - "[[../30_Knowledge/tech/frame-stride-conditioning]]"
---

# MetaWorld translator feeds the wrong `fs`/`fps` to the frozen base

## Symptom

The MetaWorld translator currently writes
`fps=DEFAULT_METAWORLD_FPS=5` and `frame_stride=int(slice_stride)` (= 1
by default) into each batch
(`src/generative_flow_adapters/data/translators/metaworld.py:15,137-138`).
The batch preprocessor then resolves the base's fps-channel value via
`fs = batch.get("frame_stride", batch.get("fps"))`
(`batch_preprocessor.py:257`), which picks up the slice stride (`1`)
and feeds it to the frozen DynamiCrafter UNet's fps embedding.

The base UNet's pretrained anchor is `default_fs=10` (see
`external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:68`).
Feeding `1` instead of `10` parks the channel **9 units below its
trained anchor in the direction of maximum motion** — the opposite of
what MetaWorld's already-large per-frame motion (5 fps source) needs.

## What the fix is

Per [[../50_Decisions/decided/per-sample-frame-stride-sampling]] (Option
A, anchor at AVID convention):

1. In `src/generative_flow_adapters/data/translators/metaworld.py`:
   - Change `DEFAULT_METAWORLD_FPS = 5` to `10` (or rename to
     `DEFAULT_METAWORLD_FS = 10` if the name is misleading).
   - Write `"frame_stride": 10` (the AVID anchor) into the batch
     unconditionally — *not* `int(stride)`.
   - Pin the actual slice stride at `1` (contiguous reads). Read pixels
     and actions with `start : start + span` rather than
     `slice(start, start + span, stride)`.

2. In `src/generative_flow_adapters/data/dataset.py`: the
   `frame_stride` parameter on `TranslatedClipDataset` becomes a dead
   knob for MetaWorld. Either remove it from the user-facing API, or
   leave it but document that the translator overrides it. Prefer
   removing — fewer footguns.

3. In `scripts/train_hyperalign_shortcut_metaworld.py:65` and any
   sibling scripts: remove the `--frame-stride` CLI flag (it can no
   longer change anything).

4. The fps/frame_stride precedence in
   `batch_preprocessor.py:257`(`_extract_fs`) is now a non-issue (both
   keys carry `10`). **Do not** touch this code or open a separate
   ticket for it.

## Verification

- Print the resolved `cond["fs"]` for one MetaWorld batch and confirm it
  is `10`.
- Quick AVID-side sanity: load the AVID MetaWorld data module
  (`external_repos/avid/latent_diffusion/src/ldwma/lightning/data_modules/metaworld.py`)
  and confirm its `frame_stride=10, fps=10` matches what we now emit.
- Re-run any unit tests that touch the MetaWorld translator
  (`tests/test_metaworld_dataset.py`) and adjust expectations if the
  recorded numbers were checked.

## Why medium, not high

This bug has been silently mis-conditioning the base on every MetaWorld
run that already happened, but no D2 evidence has been collected yet
(per `10_now/product-state.md`, the experiments table is empty). So
nothing in the thesis is invalidated; only future runs need the fix.
Medium because it should land before the first cited D2 run, not
because there is anything to retroactively repair.

## Related

- Decision: [[../50_Decisions/decided/per-sample-frame-stride-sampling]]
- Audit: [[../30_Knowledge/tech/frame-stride-conditioning]]
- AVID convention reference:
  `external_repos/avid/latent_diffusion/src/ldwma/lightning/data_modules/metaworld.py:107-108`
- Base anchor:
  `external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:68`
