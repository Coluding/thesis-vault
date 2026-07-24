---
type: bug
scope: data
status: closed
priority: high
created: 2026-07-22
updated: 2026-07-22
resolution: fixed
resolution_note: spawn DataLoader workers + translator __getstate__ dropping lazy readers + close() after the parent-process geometry probe
closed_at: 2026-07-22
related: ["[[../30_Knowledge/experiments]]"]
---

# bug: ACWM-Phys DataLoader deadlocks on first worker batch (decord not fork-safe)

## What

`scripts/precompute_latents.py --dataset acwm_phys` froze as soon as the
DataLoader started with `num_workers >= 1`. `__getitem__` worked when called
directly in the parent process; the multi-worker loader produced zero batches
and hung forever.

## Root cause

Fork-after-decord. With `--max-area` set, both `precompute_latents.py` and
`train_wan22_i2v_metaworld_external.py` run a geometry probe `dataset[0]` in
the **parent** process. For ACWM-Phys that opens a decord `VideoReader`
(spawning FFmpeg decoder threads). PyTorch's DataLoader then forks its workers
at first iteration; the child inherits decord/FFmpeg mutex/thread state that
fork cannot replicate, and the worker's first `get_batch()` deadlocks. This is
a known decord limitation (decord is not fork-safe). MetaWorld/HDF5 had the
same latent hazard (open `h5py.File` handle across fork) but happened not to
trigger.

Reproduced deterministically: fork context + parent probe → hang (0 batches in
60s); spawn context → batches stream at ~0.2s/clip.

## Fix (generative-flow-adapters)

- `data/translators/acwm_phys.py`: `__getstate__` drops `_readers` so the
  translator pickles for spawn workers; each worker lazily re-opens its own
  readers (that path already existed).
- `data/translators/metaworld.py`: same for the lazy `_file` h5py handle.
- `scripts/precompute_latents.py`: probe moved before loader construction,
  `dataset.translator.close()` after the probe, and
  `multiprocessing_context="spawn"` whenever `num_workers > 0`.
- `scripts/train_wan22_i2v_metaworld_external.py`: `translator.close()` after
  the geometry probe and spawn context on all three DataLoaders (train, eval,
  `--precompute-latents`).

## Follow-up 2026-07-23: a *second*, cosmetic "freeze" on top of the real one

After the spawn fix, the first real cluster-script run (`--batch-size 4
--num-workers 8`) still *looked* hung and was Ctrl-C'd. It was not: 64 latents
had already been written to `latents.shared`. The progress line was gated on
`done % 25 == 0`, and `done` advances in steps of `batch_size` — with
`--batch-size 4` that only aligns every **100** windows (~8 min at ~5s/clip),
and for a batch size coprime with 25 it would never print at all until the end.

Fixed in `scripts/precompute_latents.py`:

- report on `done - last_report >= 25` (batch-size independent) and include an ETA;
- added `--progress {auto,bar,plain}` with a tqdm bar over *windows*. `auto`
  shows the bar only when stderr is a TTY — under sbatch/nohup the output is a
  file, where a bar's carriage returns would collapse the log into one line.

Measured on the 3090: **~4.5–5.7 s/clip** for 65-frame 768x768 windows, so
`ind_train` (1500 episodes x 2 valid starts = 3000 windows) is ~4.5–5 h;
`ind_test`/`ood_test` are 100 windows each (~10 min). The cache resumes
cleanly, so an interrupted run loses nothing.

Also noted: Ctrl-C takes ~1 min to tear down 8 spawn workers, and the main
process outlives the first `^C` — that is teardown latency, not a hang.

## Rule of thumb going forward

Any translator holding a lazy native handle (video decoder, HDF5, LMDB, ...)
must (a) open it per-process only, (b) drop it in `__getstate__`, and any
script that touches the dataset in the parent before iterating a multi-worker
loader must use spawn workers.
