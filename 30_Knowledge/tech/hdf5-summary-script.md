---
type: tech-note
status: living
last_updated: 2026-05-25
sources:
  - "code: scripts/hdf5_summary.py"
  - "code: src/generative_flow_adapters/data/translators/metaworld.py"
  - "data: ds/metaworld_corner2.hdf5"
  - "data: ds/metaworld_corner2_large.hdf5"
relevance: D2  # dataset inspection / debugging
---

# `scripts/hdf5_summary.py` — HDF5 inspection tool

Recursively walks an HDF5 file and prints every group/dataset key plus
per-dataset metadata (shape, dtype, chunks, compression, on-disk size) and
summary statistics (min/max/mean/std, NaN/inf counts, bool true/false
fractions). Group/dataset attrs are printed inline.

## Why
Quick sanity-check of the MetaWorld dumps in `ds/` without writing throwaway
h5py snippets. The files are nested as
`<env_name>/episode_<i>/<field>` (matches `MetaWorldTranslator`) and large
(`metaworld_corner2_large.hdf5` ≈ 11 GB), so naive `dset[()]` reads are a
non-starter.

## How
- Stats are **sampled** by default: datasets with > `--max-stat-elements`
  (default 5M) elements are strided along axis 0 so we never materialize the
  whole array. Sampled stats are tagged `[sampled]`. `--full-stats` forces a
  full read.
- `--max-children N` (default 10) limits children expanded per group — needed
  because there are ~50 envs × many episodes. `0` = no limit.
- `--max-depth`, `--no-stats` for cheaper/shallower passes.

## Findings worth noting (from `metaworld_corner2.hdf5`)
- Per-episode fields: `action (T,4)`, `pixels (T,128,128,3) uint8`,
  `depth (T,128,128)`, `tactile (T,2,64,64)`, `proprio (T,7)`,
  `force_torque (T,6)`, `ee_xyz/object_1_xyz/object_2_xyz (T,3)`,
  `gripper (T,)`, `bool_contact (T,) bool`. T≈300.
- `pixels`, `depth`, `tactile` are gzip-chunked; the rest are uncompressed.
- ⚠️ `object_2_xyz` can be **all-NaN** for some episodes (e.g. assembly-v3
  episode_0) — tasks with a single object leave the second object slot unset.
  Worth filtering/guarding if object positions are ever used as conditioning.

Run:
```bash
python scripts/hdf5_summary.py ds/metaworld_corner2.hdf5 --max-children 2
python scripts/hdf5_summary.py ds/metaworld_corner2_large.hdf5 --no-stats --max-depth 2
```
