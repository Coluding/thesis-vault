---
type: decision
status: decided
created: 2026-06-04
decided_at: 2026-06-04
updated: 2026-06-04
scope: data
supersedes: "[[per-sample-frame-stride-sampling]]"
related:
  - "[[per-sample-frame-stride-sampling]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]"
---

# Decision: MetaWorld temporal coverage — load-time frame stride + action SUM, constant fs

## Status

**Decided & implemented 2026-06-04.** Supersedes the *contiguous-reads* choice
of [[per-sample-frame-stride-sampling]] (Option A). The part of that
decision that **survives**: `fs` is a base-only input, never a `Δ_φ`
conditioning key, and is not varied per sample. What **changes**: we no longer
read contiguously — we subsample at a fixed stride for a longer temporal window.

## Trigger

Observed during logging: a 16-frame clip is too short to show any meaningful
action effect. Grounded in the data — MetaWorld episodes are **300 frames**
(`ds/metaworld_corner2*.hdf5`), so a 16-frame **contiguous** clip (stride 1)
covers only **~5% of a trajectory**; the arm barely moves. This is exactly the
revisit trigger the prior decision named ("the adapter under-uses the base's
motion prior" / constant-`fs` masking a failure mode).

## Decision

Keep the 16 frames DynamiCrafter wants, but **spread them over k× wall-clock**
by reading every k-th frame at load time:

1. **Load-time frame stride k**, set in the **config** (`data.frame_stride`,
   default **4**) — not a CLI flag (the `--frame-stride` flag remains only as an
   optional override). A knob, so k=4/8 can be swept per config. Span budget
   allows up to k≈18 on 300-frame episodes (the action window — see below —
   binds at `length*k ≤ 300`).
2. **SUM the dropped delta-actions.** Stride k drops `a_{ik+1..ik+k-1}`; since
   MetaWorld actions are deltas `(Δx,Δy,Δz,Δgripper)`, the action attached to a
   kept frame is the **sum** of the k raw actions in its window (net control over
   the window). Reduces to the raw action when k=1. Only `act` is summed; state
   keys (proprio, ee_xyz, object positions…) are **sampled** at kept frames.
3. **Feed a constant `fs` to the base, decoupled from the slice stride.** We
   pass `fs=1` regardless of k (translator `fs_value`, default 1). We do **not**
   scale `fs` with k and do **not** condition `Δ_φ` on it. Rationale: MetaWorld's
   large per-frame disparity makes the base's fps prior a poor fit anyway, D2
   targets action-conditioned dynamics (not fps-robustness), and a constant `fs`
   carries no information — consistent with "`fs` is base-only." The adapter
   absorbs any base motion-prior mismatch.

## Why over the alternatives

- **Regenerate MetaWorld at a lower fps (fix upstream).** Rejected for now: (a)
  **no data generator exists in-repo** — AVID only *loads* metaworld; the
  `corner2` hdf5 came from an external pipeline we'd have to rebuild (envs +
  scripted policy + render 4900 episodes); (b) it **relocates** the action
  problem rather than removing it (render-every-k still needs the SUM, unless you
  lower the *control* rate, which changes the task/dynamics); (c) inflexible — it
  bakes one fps, whereas load-time stride is a sweepable knob. Reconsider later
  *only* if a chosen k proves right and the summed-`Δgripper` semantics (below)
  turn out to matter — then re-collect at that fps with a lowered control rate.
- **Mapped `fs(k)` to keep the base in-distribution (Option C).** Rejected for
  now: requires verifying DynamiCrafter's `fs`/`fps` convention (still
  `_needs verification_`), and the disparity argument above makes the payoff
  doubtful. The constant-`fs` choice is the cheap, no-verification path.

## Implemented (commit pending)

- `data/translators/metaworld.py` — `fs_value` param (default 1); `load_clip`
  reads pixels/state strided but actions via `_read_summed_actions` (sum each
  k-window); emits explicit `fs` (constant) + `frame_stride` (real stride,
  provenance); span budget uses `length*stride`.
- `data/dataset.py` — episode span/filtering uses `window_width * frame_stride`
  (the action window; no-op at stride 1).
- `data/batch_preprocessor.py` — `_extract_fs` prefers the explicit `fs` key,
  falling back to `frame_stride`/`fps` for legacy datasets.
- **`config.py` — new `DataConfig`** (`frame_stride`, `fs_value`, `window_width`,
  `hdf5`, `sampling`, `caption_mode`) on `ExperimentConfig.data`. **Frame stride
  is now set in the YAML `data:` block, not via CLI** (per user preference).
- **`data/builders.py` — `build_metaworld_clip_dataset(data, ...)`** resolves the
  translator+dataset from the `DataConfig` (CLI overrides win only when passed).
- `scripts/train_{avid_shortcut,hyperalign_shortcut,hyperalign}_metaworld.py` —
  build the dataset via `build_metaworld_clip_dataset(config.data, ...)`;
  `--frame-stride`/`--sampling` default to `None` (override-only).
- `configs/*metaworld*.yaml` — added a `data:` block with `frame_stride: 4`,
  `fs_value: 1` (all 7 metaworld configs).
- Tests in `tests/test_metaworld_dataset.py` (action-SUM, fs decoupling, action-
  window bound, `_extract_fs` precedence, DataConfig parse + builder override);
  25/25 pass, no stride-1 regressions.

## Caveats / watch in logging

- **Summed `Δgripper` is semantically iffy** (summing an open/close command).
  `Δx,Δy,Δz` sum cleanly to a net displacement; gripper is the weak point. If it
  bites, that's the argument for the upstream re-collection above.
- **Action magnitudes scale ~k×** after summing — action-input normalisation may
  want revisiting.
- **`temporal_length: 8` (action conditioning) vs 16 (video)** — pre-existing
  mismatch to confirm; the SUM keeps `act` shape `(B, 16, A)`, unchanged.
- **`fs=1` reverses** [[../../20_Tickets/bug-data-metaworld-fs-anchor-default]]
  (which wanted 1→10). That ticket is superseded — we deliberately keep `fs=1`.

## Update 2026-06-04 — new camera-split layout (collected at full resolution)

A **new HDF5 layout** landed:

    <env>/<camera>/episode_N/{pixels, depth}        # images, per camera angle
    <env>/sensors/episode_N/{action, proprio, ...}  # state, stored once
    env attrs: camera_names, frame_stride, frame_aggregation

The collector *can* apply its own `frame_stride` + `frame_aggregation` upstream
(`T = ceil(steps / frame_stride)`), but for the **current dataset it was
collected at `frame_stride = 1` (full resolution)** — no upstream subsampling or
aggregation. So our **load-time stride + action-SUM remains the live mechanism**,
and the configs keep `data.frame_stride: 4`. **No config change.**

Consequences:
- **Keep `data.frame_stride` at the desired load-time stride (4)** when the
  collection stride (env attr) is 1 — the data is full-res, so we do the
  subsampling. *Only if* a future dataset is collected with `frame_stride > 1`
  should `data.frame_stride` drop toward 1 to avoid subsampling twice (the SUM
  still composes either way, since summing deltas is associative).
- The **`Δgripper` summing caveat stays ours** (we aggregate at load), unless a
  future collection sets `frame_aggregation` upstream — then it becomes the
  collector's choice (recorded in the env attr).
- `fs_value` (constant fed to the base) is unchanged and still decoupled.

Translator + config support (implemented):
- `MetaWorldTranslator` auto-detects the layout (flat vs camera-split) and joins
  a camera view to its sensors by episode index. `depth` is read per-camera; the
  other state channels come from `sensors/`.
- **`DataConfig.env` / `DataConfig.camera`** select environment(s) / camera
  angle(s) — str, list, or omitted (= all). Multiple cameras multiply samples
  (same rollout, several views). Wired through `build_metaworld_clip_dataset`.
- Tests: camera-layout indexing, view↔sensors join, env/camera filtering,
  action-SUM under the camera layout (`tests/test_metaworld_dataset.py`).

## Future-revisit triggers

- Summed-`Δgripper` shown to hurt → re-collect at lower control rate (clean
  per-frame actions).
- A D2 result needing fps-robustness → revisit mapped `fs(k)` (Option C).
- Action-norm instability traced to the k× magnitude bump.

## Related

- [[per-sample-frame-stride-sampling]] — the superseded contiguous
  choice; its shelved SUM-action sub-resolution is now adopted.
- [[../../30_Knowledge/tech/frame-stride-conditioning]] — the audit; its
  "Resolved" section now points here.
- Code: `src/generative_flow_adapters/data/translators/metaworld.py` (both
  layouts + env/camera selection), `data/dataset.py`, `data/batch_preprocessor.py`,
  `data/builders.py` (`build_metaworld_clip_dataset`), `config.py` (`DataConfig`).
