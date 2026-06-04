---
type: tech-note
status: living
last_updated: 2026-06-04
deliverable: D2, D3
---

# Frame-stride (`fs`) conditioning: where it flows and where it stops

> DynamiCrafter is conditioned on a frame-stride / fps scalar `fs`. In the
> current framework that signal reaches the **frozen base UNet** but is
> **not** an explicit conditioning input to the trainable adapter `Δ_φ`.
> The adapter sees frame stride only indirectly, through whatever the
> frozen base's activations happen to encode. This note records the
> observed wiring so the design question (should `Δ_φ` be conditioned on
> `fs`?) can be reasoned about against facts, not guesses.

All claims below are sourced to the implementation repo
`/home/lukas/projects/generative-flow-adapters/` as read on 2026-05-25
(uncommitted working tree).

## The path `fs` takes

1. **Dataset → batch.** The MetaWorld translator writes the slice stride
   into the batch as `frame_stride` (`data/translators/metaworld.py:138`)
   and a nominal `fps` (`DEFAULT_METAWORLD_FPS = 5`, line 15).
2. **Batch → `cond["fs"]`.** The preprocessor always extracts it:
   `_extract_fs` reads `batch.get("frame_stride", batch.get("fps"))`
   (`data/batch_preprocessor.py:256-266`) and assigns `cond["fs"]`
   (lines 149-151). This is **independent** of `condition_keys` (which
   defaults to `("act",)`, `batch_preprocessor.py:61`) — `fs` is handled
   on its own track, so it lands in `cond` even though it is not a
   declared structured condition.
3. **`cond["fs"]` → frozen base UNet.** The base UNet built for these runs
   (`external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml`)
   sets `fs_condition: true`, `default_fs: 10`, `fps_condition_type: "fps"`
   (lines 35, 68, 69). When `fs_condition` is on, the UNet timestep-embeds
   `fs` through its own `fps_embedding` MLP and **adds it to the time
   embedding** (`backbones/dynamicrafter/modules/networks/openaimodel3d.py:782-787`).

So the frozen base *is* conditioned on `fs`. The question is whether the
adapter is.

## Where it stops — per adapter family

### HyperAlign (hypernetwork) — `fs` does NOT reach `Δ_φ`

`_prepare_unet_runtime` builds `emb` **for the frozen UNet** and adds
`fs_embed` there, routed through the base model's `module.fps_embedding`
(`adapters/hypernetworks/hyperalign.py:676-682`). Its docstring is explicit
(lines 630-636):

> "The adapter's condition embedding is intentionally not consumed here —
> it reaches the hypernetwork through the condition_injection path instead."

The hypernetwork's explicit conditioning inputs are **action** (via
`condition_injection_mode: memory_tokens`) and **step_level** (via
`step_level_embed`). `fs` is **not** among them. So `Δ_φ` is conditioned on
`fs` only *indirectly* — through the frozen base activations HyperAlign
reads via its input-feature hooks (`_HyperAlignInputFeatureStore`,
hyperalign.py:687).

### UniCon (hidden-state) — slightly tighter, still indirect

`fs_embed` is added to `emb` (`adapters/hidden_states/unicon.py:483-489`),
and that `emb` is then fused with the adapter embedding via
`adapter.emb_fuse(torch.cat([emb, adapter_embedding], dim=1))` (line 493).
So `fs` leaks into the fused embedding the adapter conditions on — but the
`adapter_embedding` itself (built from action / step_level in
`_prepare_adapter_embedding`) carries no `fs`. `fs` enters only through the
shared base `emb`.

### Output adapter (AVID-style) — passes `fs` to its own network

The output adapter reads `cond.get("fs")` and forwards it to its own UNet
`self.module(..., fs=fs, ...)` (`adapters/output/dynamicrafter.py:109,143`).
Whether that has any effect depends on whether the *adapter's* UNet was
built with `fs_condition: true` — **not yet verified** for the output-adapter
configs.

## What the MetaWorld dataset adapter actually emits

The dataset adapter (`TranslatedClipDataset` + `MetaWorldTranslator`) *does*
take `fs` into account, but only as a faithful echo of a fixed stride — it
never samples or varies it:

- `TranslatedClipDataset` stores a single `frame_stride` (default `1`) at
  construction (`data/dataset.py:35,50`) and passes that same value to
  `load_clip(..., stride=self.frame_stride)` on **every** `__getitem__`
  (`dataset.py:101`). The only per-sample randomisation is the clip
  **start index** (`dataset.py:95`), not the stride.
- The translator slices with that stride and records both
  `"frame_stride": int(stride)` and `"fps": int(self.fps)` (=5 default)
  into each clip (`translators/metaworld.py:137-138`).

So `fs` is present in the batch, correct relative to the actual subsampling,
but **constant across the whole dataset**.

### Precedence bug: the slice stride wins over fps

The preprocessor resolves the conditioning value as
`fs = batch.get("frame_stride", batch.get("fps"))`
(`batch_preprocessor.py:257`). Since the translator *always* sets
`frame_stride`, it always wins and the emitted `fps` (5) is **never used**
for conditioning. The value that actually reaches the model is therefore the
**slice stride** (`1`), while the base declares `fps_condition_type: "fps"`
and `default_fs: 10` — i.e. it was trained expecting an *fps* number,
anchored at 10. Two semantic mismatches stack here:

- We feed a *stride* into a channel the base interprets as *fps*.
- The value (`1`) is far from the base's fps anchor (`10`), and the nominal
  MetaWorld fps we do know (`5`) is silently discarded.

## The compounding data issue (constant `fs=1`)

The training entrypoint defaults to `--frame-stride 1`
(`scripts/train_hyperalign_shortcut_metaworld.py:65`). Consequently every
MetaWorld training sample feeds `cond["fs"] = 1` into a base whose fps
anchor is `default_fs: 10`. Two observable facts follow:

- The fed value (`1`) is **not** the base's anchor (`10`).
- It is **constant across the entire dataset** — there is no `fs`
  *variation* for any path (direct or indirect) to learn from.

> **Analysed estimate (not measured):** feeding a constant, off-anchor `fs`
> likely pins the base's fps-embedding to a fixed off-distribution offset.
> Inputs: `default_fs: 10` and `fps_condition_type: "fps"` (config, observed)
> + the convention that DynamiCrafter's fps channel was trained over a range
> of stride values. The exact training range of the pretrained checkpoint is
> _needs verification_ (not pulled from the DynamiCrafter paper/weights yet),
> so treat "off-distribution" as a hypothesis, not a fact.

## The framework seam

The thesis composition rule as written is
`f(x_t, t, a_t, d) = f_base(x_t, t) + g(d) · Δ_φ(x_t, t, a_t, d)` — `fs`
appears in **neither** term. The existing structural-encoder note already
files `fs` under "existing DynamiCrafter conditioning, not part of the
thesis framework" ([[structural-encoder]], lines 110-112). The current code
is consistent with that framing: `fs` is a base-model input, not a
framework conditioning key.

## Resolved 2026-06-04 — load-time stride k + action SUM, constant `fs=1`

**Current resolution:** [[../../50_Decisions/decided/metaworld-frame-stride-load-time]].
The data is subsampled at a fixed **stride k** (default 4) so a 16-frame clip
covers k× more wall-clock — because 16 *contiguous* frames are only ~5% of a
300-frame episode and show no action effect. Dropped delta-actions are
**SUM-aggregated** per kept frame; the base is fed a **constant `fs=1`** via an
explicit `fs` key, decoupled from the real slice stride (recorded as
`frame_stride`). `Δ_φ` still does **not** see `fs`. Implemented in
`translators/metaworld.py` (`fs_value`, `_read_summed_actions`), `dataset.py`
(span uses `window_width*stride`), `batch_preprocessor.py` (`_extract_fs` prefers
`fs`), and the `--frame-stride` default (1→4) in the metaworld training scripts.

> **Superseded resolution (2026-05-28):** the earlier plan was Option A of
> [[../../50_Decisions/decided/per-sample-frame-stride-sampling]] — write
> `fps=frame_stride=10` and read contiguously to anchor the base at
> `default_fs=10`. That parked the fps channel at its pretrained anchor but left
> the 16-frame window too short. The successor keeps `fs` base-only (the part
> that survives) but reads strided and **keeps `fs=1`, not the anchor 10** — so
> the anchor-fix ticket [[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]
> is **superseded, not applied**. The `fps_condition_type` semantic ambiguity is
> sidestepped differently now: `fs` is a constant we don't rely on, so its exact
> interpretation no longer matters for the baseline.

## Related

- [[structural-encoder]] — the canonical adapter conditioning path `fs` is currently *outside* of
- [[shortcut-training-modes]] — `d` vs `fs` overlap question
- [[../related-work/hyperalign]] · [[../related-work/avid]] — the two adapters traced here
- Code: `data/batch_preprocessor.py:149,256` — `fs` extraction
- Code: `adapters/hypernetworks/hyperalign.py:676` + docstring 630-636 — `fs` → frozen base only
- Code: `adapters/hidden_states/unicon.py:483` — `fs` into fused emb
- Code: `adapters/output/dynamicrafter.py:109` — `fs` → adapter's own UNet
- Code: `scripts/train_hyperalign_shortcut_metaworld.py:65` — `--frame-stride` default `1`
- Config: `external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml:35,68,69`
