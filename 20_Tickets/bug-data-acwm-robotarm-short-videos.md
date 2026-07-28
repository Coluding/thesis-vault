---
type: bug
scope: data
status: closed
priority: high
created: 2026-07-25
updated: 2026-07-25
resolution: fixed
resolution_note: episode length now probed from the mp4s (cached sidecar) instead of trusting metadata.pt's nominal `length`; 5 too-short episodes drop out of the 97-frame window pool
closed_at: 2026-07-25
related: ["[[bug-data-acwm-decord-dataloader-fork-deadlock]]"]
---

# bug: ACWM-Phys robot_arm precompute crashes — `IndexError: Out of bound indices`

## What

`scripts/precompute_latents.py` on `kinematics/robot_arm/ind_train` died in a
DataLoader worker:

```
IndexError: Out of bound indices: [79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96]
  .../translators/acwm_phys.py:125 in load_clip
```

## Root cause

`metadata.pt`'s per-episode `length` is **nominal**, not the decodable frame
count. For `robot_arm` every entry claims `length: 128` and ships `actions`
`[128, 7]`, but a few exported mp4s are genuinely shorter. Verified on the
local copy 2026-07-25:

| split | episodes | short videos |
|---|---|---|
| `kinematics/robot_arm/ind_train` | 2002 | **6** — ep1: 79, ep104: 80, ep106: 79, ep114: 96, ep115: 75, ep127: 121 |
| `kinematics/robot_arm/ind_test` | 105 | 0 |
| `kinematics/robot_arm/ood_test` | 105 | 0 |
| `rigid_dynamics/push_block/{ind_train,ind_test}` | 1500 / 50 | 0 |

Not a truncated download: the local files' sha256 match the upstream HF blob
hashes recorded in `ds/acwm-phys/.cache/huggingface/download/.../*.metadata`,
`ffmpeg -v error` decodes them cleanly, and `ffprobe -count_frames` agrees with
decord (`nb_read_frames=79`, `duration=7.9`). The release itself is like this —
push_block was unaffected, which is why this only surfaced now that robot_arm
became the D2 env.

`list_episodes()` reported `min(length, action_rows)` = 128, so the fixed-window
sampler happily emitted starts up to 31 for a 97-frame window, and decord
rejected frame indices ≥ the real frame count.

## Fix (generative-flow-adapters, `data/translators/acwm_phys.py`)

- New `_probe_frame_counts()`: opens every episode video once (8-thread pool,
  ~1 s for 2002 episodes locally) and caches `{video_path: n_frames}` to a
  `frame_counts.json` sidecar next to `metadata.pt`. Best effort — a read-only
  data dir just means re-probing.
- `list_episodes()` returns `min(nominal length, action rows, decoded frames)`
  and prints a one-line summary of the clipped episodes.
- Module docstring records that robot_arm's `length` is nominal.

## Effect on the robot_arm window pool

With `temporal_length: 97` and `num_windows: 32`: 1997 of 2002 episodes kept
(the 5 videos below 97 frames can't hold a window and are filtered by the
existing `ep.length >= span` rule), ep127 keeps 25 of its 32 starts. Total fixed
windows **63897**. No latents had been cached for robot_arm yet, so no cache
invalidation.

## Rule of thumb

For any video-backed translator, episode length must come from the decoder, not
from a metadata field — sidecar-cache the probe if it's expensive.
