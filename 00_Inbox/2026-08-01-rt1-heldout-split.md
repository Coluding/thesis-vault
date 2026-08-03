# RT-1 held-out eval split — provenance, build, re-eval job, blast radius

**Date:** 2026-08-01 (work finished early 2026-08-02)
**Trigger:** every RT-1 (and OpenVid) training job passed the same directory to
`--data-dir` and `--eval-data-dir`, so every RT-1 eval number in the campaign is
in-sample. Repo: `/home/lukas/projects/generative-flow-adapters/`, cluster
`snellius` (`lbierling`).
**Nothing was submitted.** Job files written, data split built, code patched
locally. The repo has **not** been rsynced to the cluster.

---

## 0. The bug, re-confirmed from the run logs (not just the source)

`scripts/train_wan22_i2v_metaworld_external.py` (at `HEAD` = `75721b7`):

| line | code |
|---|---|
| 419 | `if want_eval and args.eval_data_dir is not None:` — builds the eval dataset from that dir verbatim |
| 445 | `elif want_eval:` — the `val_fraction` `random_split` fallback, unreachable when 419 fires |

Confirmed in the actual training logs, which print the resolved eval split:

```
logs/rt1/wan-action-wan-rt1-tn-nobase-25107236.out:6
    eval dataset: rt1 split /home/lbierling/scratch-shared/rt1/train
logs/skyreels/acwm-robotarm-skyreels-skyreels-rt1-tn-nobase-25112302.out
    eval dataset: ACWM-Phys split /home/lbierling/scratch-shared/rt1/train
logs/skyreels/acwm-robotarm-skyreels-skyreels-rt1-oracle-25133625.out
    eval dataset: ACWM-Phys split /home/lbierling/scratch-shared/rt1/train
```

`grep -rn "eval-data-dir" jobs/` — ACWM is clean (every ACWM job passes
`$ROOT/ind_train` / `$ROOT/ind_test`). Affected: 4 RT-1 jobs + 1 OpenVid job
(list in §6).

### A second, independent defect found while verifying

`scripts/train_skyreels_acwm.py` has **no `val_fraction` fallback at all** —
`--eval-data-dir` is the only path to an eval set (the `if args.eval_data_dir is not None ...` block, pre-patch line ~176). Drop the
flag there and eval silently turns off entirely rather than falling back.

### A third, bigger one (see §2.4)

The SkyReels × RT-1 runs trained on **76 episodes**, not 5000 — the silent length
filter at `temporal_length: 97`. Printed in their own logs
(`dataset_size=76 steps=5000000 batch_size=6`) and never noticed.

---

## 1. Verified provenance of `train/` vs `full/`

### The scripts that made them

| dir | script | TFDS split | normalization |
|---|---|---|---|
| `rt1/train` (5000 eps, 16 G) | `jobs/experiments_cluster/rt1/convert_rt1.sh` → `jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py` | `train[:5000]` (`RT1_SPLIT` default, convert_rt1.sh:41) | per-dim std-norm computed **over those 5000 episodes** |
| `rt1/full` (87 212 eps, 110 G) | `jobs/experiments_cluster/rt1/submit_convert_rt1_shards.sh` (18-way array, `--no-normalize`) then `jobs/experiments_cluster/rt1/merge_rt1_shards.py` | `train[k*4846 : …]`, k = 0…17 | per-dim std-norm computed **once, globally over all 87 212** |

Both dirs are produced by the *same* converter (`convert_rt1_to_mp4meta.py`), so
the on-disk schema is byte-for-byte the same shape: `episode_{i}.mp4` +
`metadata.pt` = `list[dict]{video_path, actions[T,7], length, caption, clip_id}`.

### Episode-id convention (read from the real `metadata.pt`, not assumed)

* `train/`: `video_path = episode_{i}.mp4`, `clip_id = rt1_{i}`, i = 0…4999.
* `full/`: `video_path = shard_{k}/episode_{j}.mp4`, `clip_id = rt1_{g}` where
  `g` is a **global running index reassigned by the merge**
  (`merge_rt1_shards.py:… e["clip_id"] = f"rt1_{len(merged)}"`).
  Shard extents, printed from the merged manifest:
  `shard_0` → global `[0, 4845]`, `shard_1` → `[4846, 9691]`, …,
  `shard_16` → `[77536, 82381]`, `shard_17` → `[82382, 87211]` (4830, the remainder).

### Counts

```
TRAIN entries: 5000       length min/max/mean: 2 / 531 / 42.81
FULL  entries: 87212      length min/max:      2 / 650
full[4999] = shard_1/episode_153.mp4 ; full[5000] = shard_1/episode_154.mp4
```
(from `python $HOME/tmp_claude/verify_rt1.py` on the login node, reading both
`metadata.pt` files.)

So *by construction* `train[:5000]` = TFDS absolute examples 0…4999 =
all of `shard_0` (4846) + the first 154 of `shard_1`.

### Is `train ⊂ full`? — measured, and the naive index check FAILS

Comparing `train[i]` against `full[i]` element-wise:

```
(length, caption) match for i < 4330; first mismatch at i = 4330
exact raw-action match train[i] vs full[i]:  4330/5000
```

**`train/` and `full/`'s first 5000 entries are NOT in the same order.** The
cause is TFDS read ordering: `builder.as_dataset()` interleaves blocks across the
underlying tfrecord files, and `train[:5000]` and `train[0:4846]` touch a
different file set, so the *within-range enumeration order* differs even though
the *example set* is the same. Anyone assuming index alignment here gets a
silently wrong answer — this is why the whole build is content-based.

Matching by content instead (bucket `full` by `(length, caption)`, then compare
raw action tensors with `torch.allclose(atol=1e-4)`; raw recovered as
`a * action_std + action_mean` on both sides):

```
train episodes matched in full: 5000/5000   unmatched: 0
distinct full indices hit: 5139   (min 0, max 86237)
hits per shard: shard_0 4846, shard_1 160, shard_2 9, shard_3 9, shard_4 10,
                shard_5 11, shard_6 7, shard_7 11, shard_8 11, shard_9 7,
                shard_10 9, shard_11 6, shard_12 14, shard_13 10, shard_14 4,
                shard_15 4, shard_16 4, shard_17 7
hits with global idx >= 9692 (outside shard_0+shard_1): 133
```

**`train ⊂ full`: yes, 5000/5000.** The set lives in `shard_0` ∪ `shard_1`
exactly as predicted (4846 + 160 ≈ 5000 + 6 duplicates).

The **133 extra hits scattered across shards 2–17 are real** — re-checked at
`atol=1e-6`, **133/133 survive**. RT-1 (`fractal20220817_data`) genuinely
contains duplicate episodes: same instruction, same length, action sequences
identical to 1e-6 over hundreds of values. **A purely index-based held-out split
would therefore still leak.** The split below excludes them by content.

### The normalization trap (would have silently confounded the eval)

The two dirs carry **different action scales**:

```
TRAIN action_mean [ 0.0081,  0.0062, -0.0130,  0.0444, -0.0064,  0.0010,  0.0219]
TRAIN action_std  [ 0.0681,  0.0506,  0.0733,  0.1571,  0.1288,  0.1449,  0.3639]
FULL  action_mean [ 0.0070,  0.0063, -0.0126,  0.0433, -0.0058,  0.0009,  0.0219]
FULL  action_std  [ 0.0694,  0.0599,  0.0738,  0.1570,  0.1319,  0.1463,  0.3609]
max std ratio full/train = 1.1826
```

Up to **18% off on one dim.** Held-out clips cut straight from `full/` would have
been fed to the checkpoints on the wrong action scale — a confound that lands
directly on `eval_action_effect_rel`. The build undoes `full`'s normalization and
re-applies `train`'s.

---

## 2. What was built

Two directories, both written by
`/tmp/claude-…/scratchpad/build_val.py` (copy at
`snellius:~/tmp_claude/build_val.py`), seed `20260801`:

| dir | episodes | size | purpose |
|---|---|---|---|
| `$HOME/scratch-shared/rt1/val` | **400** | 0.37 GiB | Wan RT-1 (`temporal_length: 17`) |
| `$HOME/scratch-shared/rt1/val_long97` | **120** | 0.34 GiB | SkyReels RT-1 (`temporal_length: 97`) |

Nothing under `rt1/train` or `rt1/full` was read-modified or deleted.
Disk headroom checked first: `wstor_scratch1` quota 8 TiB, **8.96 % used**;
`df -h` shows 1.9 P available; home quota 200 GiB at 67.6 % (untouched — the
splits live on scratch). ~0.7 GiB total is a rounding error.

### 2.1 Selection rule (each clause load-bearing)

Candidate pool = every `full/` index that satisfies **all** of:
1. `global index >= 9692` — i.e. outside `shard_0 ∪ shard_1`, the TFDS range
   `train[:5000]` was drawn from (belt);
2. **not** content-matched to any `train/` episode by the §1 procedure (braces —
   this is what removes the 133 cross-shard duplicates);
3. `caption` appears in `train/`'s caption vocabulary — see §2.3.

```
POOL: 76370 episodes      usable>=17: 74373      usable>=97: 1476
```
`val` = 400 sampled uniformly from the `usable>=17` pool;
`val_long97` = 120 sampled uniformly from the `usable>=97` pool, disjoint from `val`.

Two dirs rather than one mixed dir on purpose: only ~1.9 % of RT-1 episodes reach
97 frames, so guaranteeing long clips inside a single 400-clip `val` would
over-represent long episodes ~10× and bias the width-17 eval.

### 2.2 Actions re-normalized to `train`'s stats

Per episode: `a_raw = a_full * std_full + mean_full`, then
`a_val = (a_raw - mean_train) / std_train`; `action_mean` / `action_std` in the
written metadata are **train's**. So the split is on exactly the action scale the
checkpoints were trained with.

### 2.3 Captions / T5 contexts — no 21 GB re-encode needed

`configs/prompts/rt1_captions.contexts.pt` is **20.99 GB** and is keyed by
`clip_id`; an unknown `clip_id` falls back to `__default__` ("A high quality
video."), which would silently change the frozen base's text conditioning between
training and held-out eval. Verified properties of the table (mmap-loaded on the
login node):

```
positive keys: 5001 (rt1_0 … rt1_4999 + __default__);  text_len 512
embedding shape (14, 4096) bf16
train clip_ids missing from table: 0
same-caption pairs (rt1_0/rt1_177, rt1_1/rt1_70, rt1_2/rt1_31, rt1_3/rt1_133,
rt1_4/rt1_48): embedding byte-identical -> True (all 5)
```

Because `T5(caption)` is deterministic, each held-out episode's `clip_id` is
**aliased to a `train/` clip_id carrying the same instruction text** — the
provider then returns exactly the embedding a fresh precompute would produce, at
zero cost. This is why clause (3) restricts the pool to train's 487-caption
vocabulary (98.7 % of the pool qualifies, so it barely narrows anything).
`val` uses 196 distinct aliases, `val_long97` 47. True provenance is preserved in
extra metadata keys the translator ignores: `source_full_index`,
`source_video_path`, `source_clip_id`.

*Latent-cache safety check:* the cache key is
`env_name | episode_idx | start_idx | frame_stride | TxHxW`
(`data/latent_cache.py:26-29`) — **not** `clip_id` — and
`ACWMPhysTranslator.__init__` derives `env_name` from the path
(`rt1-train` vs `rt1-val` vs `rt1-val_long97`). Confirmed by loading all three
splits. So the alias cannot collide in the latent cache.

### 2.4 Survival under the silent length filter — the number that matters

`data/dataset.py:70-72`:
```python
self._episodes = [ep for ep in translator.list_episodes() if ep.length >= span]
```
with `span = window_width * frame_stride`, and `ep.length` is
`min(metadata length, action rows, decodable mp4 frames)`
(`translators/acwm_phys.py:138-152`, `usable = min(...)` at :148). Measured by actually constructing
`build_rt1_clip_dataset` on each dir:

| split | episodes on disk | survive @ width 17 | survive @ width 97 |
|---|---|---|---|
| `rt1/train` (existing) | 5000 | **4868** | **76** |
| `rt1/val` (new) | 400 | **400** | 8 |
| `rt1/val_long97` (new) | 120 | **120** | **120** |

Two things fall out of that table:

* `val` is fully usable at the Wan width — no silent shrinkage.
* **The SkyReels × RT-1 runs trained on 76 episodes.** Their own logs say so
  (`dataset_size=76 steps=5000000 batch_size=6 eval=on gen_eval=on`, both
  `skyreels-rt1-tn-nobase-25112302.out` and `skyreels-rt1-oracle-25133625.out`),
  because `configs/skyreels/diffusion_skyreels_rt1_*.yaml:47` / `:62` still say
  `temporal_length: 97` with the comment "128-frame episodes" — copied from the
  ACWM Robot Arm config and never adjusted for RT-1's ~22–115-frame episodes.
  The Wan config *was* adjusted
  (`diffusion_wan22_action_rt1_tokennorm_nobase.yaml:57`, `temporal_length: 17`,
  → `dataset_size=4868`). This is independent of the in-sample bug and
  arguably worse: the SkyReels RT-1 headline numbers come from 76 clips,
  evaluated on those same 76 clips.

### 2.5 Post-write verification

Re-run straight off the written `metadata.pt` files:
```
val:        400 episodes | content-clashes with train: 0 | source_full_index [9692, 87157]
val_long97: 120 episodes | content-clashes with train: 0 | source_full_index [9748, 86559]
```
and a real sample drawn through `build_rt1_clip_dataset` →
`video (17, 256, 320, 3)  act (17, 7)  caption='place coke can upright'  task_name=rt1_164`.

---

## 3. The re-eval job

**Path:** `jobs/experiments_cluster/rt1/submit_eval_rt1_heldout.sh` (new, `chmod +x`).

```bash
# on the cluster, from ~/generative-flow-adapters, AFTER rsyncing the repo:
sbatch jobs/experiments_cluster/rt1/submit_eval_rt1_heldout.sh

# optional env overrides:
#   WAN_CKPT=/scratch-shared/lbierling/outputs/wan-rt1-tokennorm-nobase-run/checkpoints/step_00013800.pt
#   SKYREELS_CKPT=/scratch-shared/lbierling/outputs/skyreels-rt1-tokennorm-nobase-run/checkpoints/step_00004200.pt
#   SKIP_SKYREELS=1
```
Defaults to the newest `step_*.pt` in each run dir (`step_00013800.pt` /
`step_00004200.pt` as of now). `gpu_h100`, 1 GPU, `--time=4:00:00`.

**What it reports.** It re-runs the *exact* code path that produced the logged
numbers, so the output is directly comparable: `eval_action_effect_rel` (plus
`eval_action_cos`, `_loss_gap`, `_effect_vs_adapter`, `_base_null_violation`,
`eval_loss` / `eval_denoise_*`) from `Trainer._run_eval_cycle` →
`_action_sensitivity_eval`, and the quality set
`eval/adapted/{psnr,ssim,lpips,mse}` + `eval/adapted/{fid,fvd_i3d}` with the
`eval/base/*` counterparts from `_native_quality_eval`
(`training/trainer.py:1665`, `results.update(suite.compute(prefix=f"eval/{variant}"))`).
Batch sizes / `--num-windows` / `max_area` match the training jobs, and
`quality_eval_num_batches: 1` / `quality_dist_num_batches: 1` are left as the
configs have them, so the held-out values are computed on the same batch budget
as the in-sample ones (noisy, but *equally* noisy — that is the point).

**Expected runtime — measured, not guessed.** Reconstructed from the training
logs' cumulative `steps/s` (elapsed(N) = N / (steps/s at N)), the delta across a
step where a full eval cycle fires:

| run | step→step spanning a full eval cycle | wall |
|---|---|---|
| Wan `5w72bo01` | 400→401 / 800→801 / 1200→1201 | **1437 s / 1456 s / 1079 s** |
| SkyReels `gi44pv5k` | 400→401 / 800→801 | **1353 s / 2687 s** |
| (normal training step, for scale) | 399→400 | 4.8–20 s |

So ≈ 20–25 min of eval per arm, plus the 5B/1.3B model build and a cold VAE
encode for the held-out clips (the job passes `--no-latent-cache`, so eval clips
are encoded on the fly; ~64 clips for the loss eval). **Estimate: ~40 min for the
Wan arm, ~40–60 min for the SkyReels arm, ~1.5–2 h total.** The 4 h wall-clock is
deliberate headroom.

**Trainer support: it did NOT exist; I added it (minimal, additive).** Neither
training script could load a checkpoint at all — no `--resume`, no `--init-from`,
nothing (`grep -n "checkpoint" scripts/train_wan22_i2v_metaworld_external.py`
returns only `--ckpt-dir`, the frozen *base* weights). So "0-step eval-only" was
not reachable. Added to **both** `scripts/train_wan22_i2v_metaworld_external.py`
and `scripts/train_skyreels_acwm.py`:

* `--init-from PATH` — loads the checkpoint's trainable tensors with
  `strict=False` (frozen-base keys are legitimately "missing"; any *unexpected*
  key raises). Same load path `scripts/generate_wan22_i2v_compare.py:893`
  already uses. **Weights only** — `global_step` and optimizer state are *not*
  restored, which is what makes the next flag work.
* `--eval-only` — forces `baseline_eval_loss`/`baseline_eval_quality` on, sets
  `trainer.checkpoint_manager = None`, and calls `trainer.train(max_steps=0)`.
  The work is then done by the trainer's existing step-0 baseline block
  (`training/trainer.py:909-937`), which fires because `global_step` is still 0.
  `max_steps=0` skips the loop entirely, and the end-of-`fit` "final eval" is
  skipped because `_cadence_due(eval_every_n_steps)` is True at step 0 — so
  exactly **one** eval cycle runs.

  Detaching the checkpoint manager is **load-bearing, not hygiene**: without it
  `_run_eval_cycle` would write `best.pt` (`best_eval_metric` starts at `+inf`,
  so the first eval always "improves") and `fit()` would write `final.pt` —
  **both into the source run's `checkpoints/` dir**, overwriting the retained
  campaign checkpoints.

---

## 4. Config / job fixes (diff summary)

All in `/home/lukas/projects/generative-flow-adapters/`. `bash -n` + `py_compile`
clean. **Not rsynced to the cluster.**

| file | change |
|---|---|
| `jobs/experiments_cluster/rt1/submit_train_wan_rt1_tokennorm_nobase.sh` | new `RT1_EVAL_DIR="${RT1_EVAL:-$HOME/scratch-shared/rt1/val}"` + existence guard; `--eval-data-dir "$RT1_DIR"` → `"$RT1_EVAL_DIR"` |
| `jobs/experiments_cluster/rt1/submit_train_wan_rt1_action.sh` | same (→ `rt1/val`) |
| `jobs/experiments_cluster/rt1/submit_train_skyreels_rt1_tokennorm_nobase.sh` | same, → `rt1/val_long97` (this config runs width 97) |
| `jobs/experiments_cluster/rt1/submit_train_skyreels_rt1_oracle.sh` | same, → `rt1/val_long97` |
| `jobs/experiments_cluster/openvid/submit_train_wan_shortcut_openvid.sh` | `--eval-data-dir "$OPENVID_DIR"` **removed** (no second OpenVid dir exists), so `data.val_fraction: 0.01` carves a real random split; commented with the cost — latent *prefetch* turns off (the split shares the training dataset object), the latent *cache* still hits |
| `jobs/experiments_cluster/rt1/submit_eval_rt1_heldout.sh` | **new** — §3 |
| `scripts/train_wan22_i2v_metaworld_external.py` | `+--init-from`, `+--eval-only`, `+--allow-insample-eval`, + the same-dir guard (§5) |
| `scripts/train_skyreels_acwm.py` | same four |

ACWM jobs **untouched** — they were already correct.

---

## 5. The proposed same-dir guard (implemented; revert if you disagree)

**Where it does *not* belong: `_eval_is_separate`.** That variable
(`train_wan22_i2v_metaworld_external.py:362` at `HEAD`) reads

```python
_eval_is_separate = (not want_eval) or args.eval_data_dir is not None or eval_hdf5 is not None
_prefetch_ok = latent_cache_dir is not None and not args.precompute_latents and _eval_is_separate
```

and it gates **latent prefetching only** — i.e. a *performance* decision: "may I
wrap the training dataset in `LatentPrefetchDataset`, which yields latents with
no `video` key?". Its actual question is "does eval build its own dataset object,
so the wrapper can't starve the generation grid of pixels?" — and when
`eval_data_dir == data_dir` the answer to *that* question is still **yes**
(line 419 builds a fresh unwrapped dataset either way). Tightening
`_eval_is_separate` to require different paths would therefore be
**semantically wrong and a pure pessimisation**: it would disable prefetch on
runs that are perfectly safe to prefetch, and it would fix the data-validity bug
only as an accident of a performance flag. It is also the wrong *time*: it
happens after the 5B base is built, so a bad invocation still burns the model
load.

**Where it does belong: argument validation, immediately after
`parser.parse_args()`.** Implemented in both training scripts:

```python
if args.eval_data_dir is not None and args.data_dir is not None:
    if Path(args.eval_data_dir).resolve() == Path(args.data_dir).resolve():
        if not args.allow_insample_eval:
            raise SystemExit(
                f"--eval-data-dir == --data-dir ({args.data_dir}).\n"
                "That is NOT a held-out split: the --eval-data-dir branch builds the eval set from "
                "this dir verbatim, so every eval metric would be IN-SAMPLE.\n"
                "Fix: point --eval-data-dir at a real held-out split dir, or drop --eval-data-dir "
                "entirely to fall back to the data.val_fraction random split.\n"
                "If you really mean it (e.g. a deliberate memorization/overfit probe), pass "
                "--allow-insample-eval."
            )
        print("WARNING: --eval-data-dir == --data-dir; eval metrics are IN-SAMPLE ...")
```

Design notes: `.resolve()` so a symlink or trailing slash can't sneak past;
a hard `SystemExit` rather than a warning, because a warning in a 25-hour Slurm
log is exactly what did not get read for the last week; an explicit
`--allow-insample-eval` escape hatch because in-sample eval **is** the right
thing for `--overfit-index` memorization probes, so the guard must be
overridable rather than absolute; fail-fast placement so it costs seconds, not a
model load.

**Two adjacent footguns worth a follow-up (not fixed):**
1. `want_eval = bool(eval_every_n_steps) and (eval_hdf5 is not None or val_fraction > 0.0)`
   (line 174) — if a config sets `val_fraction: 0`, `--eval-data-dir` is
   **silently ignored** and eval never runs. `--eval-data-dir` should force
   `want_eval` True.
2. `train_skyreels_acwm.py` has no `val_fraction` path at all; dropping
   `--eval-data-dir` there disables eval silently rather than falling back.

---

## 6. Blast radius

### 6.1 wandb runs whose eval metrics are in-sample

Run ids pulled from the job stderr (`grep -o "/runs/[a-z0-9]*"`):

| wandb run | job | script / dir | status | why in-sample |
|---|---|---|---|---|
| **`5w72bo01`** `wan-rt1-TOKENNORM-NOBASE` | 25107236 | `submit_train_wan_rt1_tokennorm_nobase.sh` | killed @25.9 h, ckpts retained | `--eval-data-dir $RT1_DIR` |
| **`sgdftf6b`** `skyreels-rt1-TOKENNORM-NOBASE` | 25112302 | `submit_train_skyreels_rt1_tokennorm_nobase.sh` | killed @21.9 h, ckpts retained | same — **plus `dataset_size=76`** |
| **`gi44pv5k`** `skyreels-rt1-oracle` | 25133625 | `submit_train_skyreels_rt1_oracle.sh` | **RUNNING right now (7 h+ at time of writing)** | same — **plus `dataset_size=76`** |
| OpenVid Wan shortcut run | — | `submit_train_wan_shortcut_openvid.sh` | no output dir under `/scratch-shared/lbierling/outputs/*openvid*` and no `logs/openvid/` — **apparently never launched**; the job file was wrong, so any future launch would have been affected | `--eval-data-dir $OPENVID_DIR` |
| `submit_train_wan_rt1_action.sh` (`diffusion_wan22_action_rt1.yaml`) | — | no run dir found | same defect in the file | `--eval-data-dir $RT1_DIR` |

**`gi44pv5k` is live and producing in-sample numbers as this is written.** Decide
whether to let it finish (the loss curve is still valid; only the eval block is
tainted) or kill and relaunch with the patched job.

### 6.2 Vault claims that must be re-qualified — **not edited, listed only**

**`30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau.md`.** The
note already carries a 🛑 in-sample banner (added earlier today) and a ⚠ fit-noise
confound section, so most of this is *confirmation* rather than news. Still
outstanding:

1. **Frontmatter `metrics:` block** — `effect_rel_evals`, `effect_rel_mean_first5`
   (0.0271), `_last5` (0.0245), `_peak` (0.0332) are all in-sample and the
   frontmatter carries no marker. Anything reading the frontmatter
   programmatically (the ledger, a future deck) gets them unqualified.
2. **Title claim** — "in-distribution data lifts the plateau **2.3×**" and the
   comparison table `ACWM ~0.011` vs `RT-1 ~0.025` vs `AVID 0.0495`. The ACWM
   number comes from a genuinely held-out `ind_test`; the RT-1 number does not.
   **The 2.3× compares a held-out figure against an in-sample one** — that is a
   new defect, not covered by the existing banner, and it is the note's headline.
3. **"Reading" §1 — "the action economics is dataset-dependent"** rests entirely
   on that 2.3×. Same for §2 "a lightweight adapter reaches ~half the reference".
4. **SkyReels section** — "opens at **0.0450** — 91 % of the AVID full-UNet
   reference, and a **35× jump** over the same backbone's ACWM value (0.0013)".
   Same held-out-vs-in-sample asymmetry (ACWM 0.0013 is held-out), **and** the
   0.0450 is measured on a **76-episode** training set — add that.
5. **"shared RT-1 floor ≈ 0.02"** — both sides of that convergence are in-sample.
6. **The quality table** (FID/FVD/LPIPS/SSIM/PSNR/MSE, Wan + SkyReels vs base).
   The note already says "if anything *worse* than it reads". Add: the SkyReels
   column is a 76-clip in-sample score, and both columns will be replaced by the
   re-eval job's output.
7. **"CORRECTED at 18 evals: Wan × RT-1 SETTLES"** and the "data-dependent floor
   (~0.021 on RT-1 vs ~0.008–0.011 on ACWM)" refinement — same mixed comparison.
8. **Not in the note, should be:** the AVID reference `0.0495` is **also**
   in-sample, but for a different reason — `jobs/.../submit_probe_rt1_action.sh`
   runs `external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py`,
   which does `data.setup(); loader = data.train_dataloader()` (lines 175-176).
   AVID's own datamodule *does* define a real split
   (`src/ldwma/lightning/data_modules/rtx.py:22-23`,
   `train_split="train[:95%]"`, `val_split="train[95%:]"`), but the probe reads
   the **train** loader. So the note's caution "the AVID RT-1 reference was not
   necessarily measured this way … must not be quoted until checked" now
   resolves: **it was also in-sample**, which means the ratio comparisons are
   more like-for-like than feared, while every RT-1 number in the campaign
   (ours *and* the reference) is a training-set number.

**`30_Knowledge/experiments/_index.md`** — three rows:

* Row `2026-08-01 | skyreels-rt1-TOKENNORM-NOBASE (killed @21.9 h)` — "opens
  0.0450 (91 % of the AVID reference, 35× over its own ACWM 0.0013), settles
  ~0.019 — same floor as Wan" + the whole quality clause. Needs **in-sample** and
  **trained/evaluated on 76 episodes** qualifiers, and the wandb id `sgdftf6b`
  (the row currently gives only the run name).
* Row `2026-08-01 | 5w72bo01 (killed @25.9 h)` — "settles ~0.021 (peak 0.0332)
  vs ACWM's ~0.011 → dataset-dependent action economics" + quality clause. Needs
  the in-sample qualifier, and specifically that the "vs ACWM's ~0.011"
  comparison is in-sample vs held-out.
* Row `2026-07-29 | 93qrvr5v (AVID RT-1)` — "AVID follows actions on
  in-distribution RT-1 (effect_rel 0.0495, ~66 % action-driven, null 0)". Needs
  "measured on AVID's **train** dataloader" per §6.2.8.
* The ledger has **no `gi44pv5k` row yet** (the oracle run is still going) — when
  it lands it needs both qualifiers from the start.

**`30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control.md`** and
**`60_Updates/entries/2026-07-29-avid-rt1-follows-actions-blindness-is-data.md`**
— both carry the `0.0495` figure; add the train-dataloader provenance. (The
"blindness is data" *inference* is already marked superseded by `20260730`; this
is about the *measurement's* provenance.)

**Not affected:** every ACWM note and row (`20260728`, `20260730`, `20260731-*`),
because all ACWM jobs use `ind_train`/`ind_test`.

---

## 7. What I could not verify

* **The re-eval job has not been run.** What *is* confirmed: `py_compile` +
  `bash -n` clean; and both patched scripts were copied to
  `snellius:~/tmp_claude/guardtest/` (**not** into the cluster repo) and executed
  on the login node — `--help` lists `--init-from` / `--eval-only` /
  `--allow-insample-eval`, and the same-dir guard fires with the intended message
  on both. What is **not** confirmed: any GPU execution. The control flow through
  `trainer.py` is traced by reading only. Highest-risk step: whether
  `model.load_state_dict(payload["model"], strict=False)` reports zero
  *unexpected* keys for these particular checkpoints — the script raises if not,
  which is the intended failure mode but will need a re-check of the config.
* **The repo working tree was being modified concurrently.**
  `src/generative_flow_adapters/training/trainer.py` changed under me mid-session
  (mtime 2026-08-01 23:46; an unrelated uncommitted step-size-probe / dropout
  fix, `git diff` shows +42/-6). I did not touch that file, and I re-read the
  step-0 baseline block afterwards to confirm the `--eval-only` path is still
  valid — but line numbers cited here are from the working tree as of writing,
  not from a commit, and someone else may be editing this repo.
* **The repo has NOT been rsynced to the cluster.** The new job's guards
  (`--help | grep -q -- "--eval-only"`) will refuse to run until it is. I left it
  alone deliberately: `gi44pv5k` is mid-flight and several jobs are queued.
* **The 21 GB caption table was only mmap-inspected**, not fully materialised; I
  verified 5 same-caption pairs are byte-identical, not all 487 caption groups.
  The determinism argument (`T5(caption)` is a pure function, and
  `precompute_clip_captions.py:69` calls `encode(cap)` per clip) makes the
  remainder a formality, but it is an inference from 5 samples + code, not an
  exhaustive check.
* **Why the TFDS orderings diverge at exactly i = 4330** — I established *that*
  `train[:5000]` and `train[0:4846]`+`train[4846:9692]` enumerate the same
  examples in a different order, and worked around it entirely by content. The
  file-interleave explanation is my reading of TFDS behaviour, **not verified**
  against the reader config. It does not affect the split's validity.
* **Whether the 133 cross-shard duplicates are duplicate *videos* or only
  duplicate *action sequences*.** I matched on `(length, caption, actions@1e-6)`
  and did not compare pixels. They are excluded either way, so the split is safe
  regardless; but "RT-1 contains duplicate episodes" as a standalone claim is
  action-level only.
* **`submit_train_wan_rt1_action.sh` and the OpenVid job**: I found no run dirs
  or logs for them and concluded they were never launched. Absence of evidence —
  if either ran under a different `output_dir`, add it to §6.1.
* **SkyReels arm runtime for the re-eval** is extrapolated from two eval-cycle
  measurements (1353 s, 2687 s) that differ by 2×; the 4 h wall-clock covers the
  spread but the estimate itself is loose.
* **Whether the SkyReels RT-1 config *should* be re-run at `temporal_length: 17`.**
  §2.4 establishes the 76-episode fact; whether to relaunch that cell (and
  whether `val_long97` or a width-17 `val` is then the right eval dir) is a
  decision, not a finding — flagging it, not taking it.
