---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-07-14
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[../bug-adapter-gate-saturation-mask-mix]]", "[[../../10_now/architecture]]", "[[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]]"]
---

# exp: run the native AVID repo (unmodified model/training code) on MetaWorld frames

## Why

Run the **real, unmodified AVID model + training code** (`AVIDAdapter`,
`scripts/train_avid.py`, the actual composition/loss/optimizer they published)
— only the **data** is swapped, from their RT1 robot data to our MetaWorld
frames. This isolates our diagnosis
([[../../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]) from any
possibility that something in *our own* trainer/preprocessor/composition wiring
is the confound, while still training on the task we actually care about.
(Originally scoped as a pure RT1-only sanity check — user redirected to
MetaWorld frames once the RT1 setup was scoped; that pivot is captured below.)

## Major finding while scoping this (before any run) — gate-bias mismatch

Read `AVIDAdapter.apply_model`
(`external_repos/avid/latent_diffusion/libs/dynamicrafter/lvdm/models/avid.py`)
and the three real published configs
(`configs/train/avid/{avid_11M,avid_34M,avid_145M}.yaml`) directly. Full
write-up: [[../bug-adapter-gate-saturation-mask-mix]] §"External validation".

**Headline:** AVID's own composition is our `mask_mix` formula exactly, but
their `init_mask_bias: 0.0` (σ(0)=0.5, balanced) vs. our `gate_bias: 4.0`
(σ(4)≈0.982, ~98% base at init) — a much more conservative starting point than
the reference recipe. This alone is independent, external evidence for the
gate-saturation bug, found by reading the exact code we're trying to compare
against — arguably the single most valuable outcome of this ticket so far,
before any GPU time is spent.

Also found: `AVIDAdapter.prepare_adapter` supports a `pretrain_steps` param
that forces `mask=0` (full adapter override) for N steps before switching to
the learned blend — the reference-implementation precedent for our
`composition: replace` "crazy experiment"
([[../feat-adapter-dynamicrafter-output-on-wan-base]]). Not used in any of the
three real AVID configs (`pretrain_steps` defaults to 0 there), so it's not
required for their published results, but it's a legitimate, paper-precedented
option.

## MetaWorld data wiring (2026-07-14)

Found that **MetaWorld data plumbing for this repo already partially existed**
(not built from scratch this session — a prior session or the user had started
it): `src/ldwma/lightning/data_modules/metaworld.py`
(`MetaworldClipDataset`/`MetaworldVideoDataModule`, reads
`ds/metaworld_corner2.hdf5`, present on disk) plus a working **eval** config
(`configs/eval/dynamicrafter_pretrained_metaworld.yaml`) that validated the
video-preprocessing params (batch_size=2, traj_len=16, 320×512 via bilinear
upsample from MetaWorld's native ~128×128 — a known, already-documented
off-distribution tradeoff, not a new concern). **No training config existed
yet** — only eval.

### Real bug found and fixed: action conditioning was silently dropped

`LatentVisualDiffusion.get_batch_input` (`lvdm/models/ddpm3d.py`) reads
`cond = {"act": batch["act"]} if "act" in batch else {}` — checking for the key
**`"act"`**. `MetaworldVideoDataModule` emitted the action tensor under
**`"action"`** instead. Since `"act" in batch` would be `False`, `cond` would
end up `{}` for the action-conditioned UNet — **training would have silently
run through this whole pipeline with zero action conditioning**, no error,
exactly the class of bug this whole investigation has been chasing on our own
WAN pipeline. **Fixed:** `metaworld.py` now emits `"act"`.

### Second bug found and fixed: action-dimension mismatch (would have crashed)

The action UNet's `action_dims` defaults to `7` (`openaimodel3d.py`) —
RT1's action dimension. **No config anywhere overrode it** for MetaWorld
(4-dim actions). Feeding a 4-dim action tensor into a `Linear(7, embed_dim)`
layer would hard-crash on the first real batch. **Fixed:** new config
`configs/train/act_cond_diffusion_11M_metaworld.yaml` (copy of
`act_cond_diffusion_11M.yaml` with `unet_config.params.action_dims: 4` added;
that file's own `data:` block is dead code when loaded as an
`action_config_file` — only `.model` is read — so only the UNet param needed
changing).

### New training config

`configs/train/avid/avid_11M_metaworld.yaml` — copy of `avid_11M.yaml`
(**real, unmodified `AVIDAdapter` composition and `train_avid.py` training
loop**) with exactly three changes:
- `action_config_file` → the new `_metaworld` UNet config above
- `data.target` → `ldwma.lightning.data_modules.metaworld.MetaworldVideoDataModule`
  (params matched to the validated eval config)
- `wandb.entity: causica` → `null` (the hardcoded value was the AVID *paper
  authors'* own wandb team — would have logged nowhere useful, or failed,
  under our account)

`base_config_file` (`dynamicrafter_512.yaml`), `adapter_params`
(`condition_adapter_on_base_outputs: True`, `learnt_mask: True`,
`init_mask_bias: 0.0` — see [[../bug-adapter-gate-saturation-mask-mix]]), and the
`lightning:` trainer block are **untouched** — this is the real reference
composition, just on our data.

## Setup status — what's left before a cluster launch

**Done (this session, 2026-07-14):**
- Confirmed the real DynamiCrafter checkpoint is on disk: `ckts/dynami512.ckpt`.
- Fixed the stale hardcoded checkpoint path
  (`/host_home/avid/dynamicrafter_512/model.ckpt`) in all `act_cond_diffusion_*`
  UNet configs to point at the local checkpoint.
- Fixed the `"act"`/`"action"` key bug and the `action_dims` mismatch (above).
- Built `avid_11M_metaworld.yaml` + `act_cond_diffusion_11M_metaworld.yaml`.

**Not done — needs a separate, heavier environment than our own `.venv`:**
- `poetry` and `python3.10` are **not installed** on this dev machine — did not
  attempt to install them locally (pinned `torch 2.1.0+cu118`,
  `pytorch-lightning 1.9.3`, `octo`, `open_clip_torch` — a genuinely separate,
  older toolchain from our Wan2.2 `.venv`) and per the standing instruction
  that real runs go to the cluster.
- All checkpoint paths above are **absolute local-machine paths**
  (`/home/lukas/...`) — will need re-pointing to the cluster filesystem
  (likely `/gpfs/home6/lbierling/generative-flow-adapters/...` based on other
  runs' wandb metadata) before launch.
- `logdir: /host_home/avid` / `wandb.save_dir: /host_home/wandb/` are
  container-mount-style paths — left as-is (unverified cluster mount
  convention); override via CLI dotlist args at launch (see recipe).
- **Not smoke-tested** — no local Poetry/TF env to validate against. The two
  bug fixes above are code-read-verified, not run-verified. Set
  `data.params.max_clips` to something small (e.g. 32) for the first launch
  to catch anything else quickly before committing to a long run.

## Recipe (cluster)

```bash
cd external_repos/avid/latent_diffusion
poetry install   # separate env from our repo's .venv — pinned old torch/PL/TF

# Re-point checkpoint paths first (see above), then:
./scripts/train.sh \
  --config configs/train/avid/avid_metaworld_11M.yaml \
  --script scripts/train_avid.py \
  lightning.trainer.logdir=<cluster-scratch-or-home-path> \
  lightning.trainer.logger.params.save_dir=<cluster-wandb-dir> \
  data.params.max_clips=32   # first launch only — drop once it's confirmed clean
```

`avid_11M_metaworld` (smoke/cheap) → scale to a `34M`/`145M` MetaWorld variant
(same pattern: copy `act_cond_diffusion_{34M,145M}.yaml` +
`action_dims: 4`, copy `avid_{34M,145M}.yaml` + the same 3 changes) once the
11M pipeline is confirmed working.

## Decision rule

- **Runs cleanly, loss descends at a healthy SNR** ⇒ the general
  adapter-on-frozen-video-diffusion approach works on genuinely untouched
  reference code, on our actual task — strengthens confidence that our own
  bugs (not the approach itself) are the cause of our weak signal. Compare
  the reference `mask_mean` trajectory (`avid.py`'s `info["mask_mean"]`,
  logged directly) against our `gate_value` logging once
  [[../../../20_Tickets/feat-training-adapter-contribution-magnitude-logging]]
  ships — same quantity, direct comparison, same task.
- **Also fails to learn / gate stays saturated** ⇒ would suggest a genuinely
  harder problem than composition/wiring bugs, worth a fresh look — though
  check the two bugs above didn't just get reintroduced by a config typo
  first.

## Guardrails

- Now a **like-for-like MetaWorld comparison**, not just a decoupled sanity
  check — but still don't report its numbers as D2/D4 evidence; the codebase
  (theirs) and our contribution are different things. Same spirit as
  [[../feat-adapter-dynamicrafter-output-on-wan-base]]'s "crazy experiment."
- Keep the Poetry env fully separate from our `.venv` — don't try to reconcile
  dependency versions between the two.
- The two fixes above (action key, action_dims) are code-read-verified, not
  run-verified — watch the first launch closely for anything else this
  scoping missed (e.g. `image_size`/`channels`/VAE compatibility were not
  independently re-derived, only carried over from the already-working eval
  config).

## Status (2026-07-15) — launched, running, healthy

Local run launched (`poetry run ./scripts/train.sh ...`, after fixing two more
local-run blockers found live: wrong-Python-env from shell-activated `.venv`
shadowing Poetry — fixed with `poetry run`; and `/host_home/avid` container-mount
paths in `avid_11M_metaworld.yaml`'s `logdir`/`wandb.save_dir` — fixed to local
paths under `external_repos/avid/latent_diffusion/outputs/`). Both fixes were
needed only for a *local* run — irrelevant on the cluster's actual container
setup, kept local-path versions since that's where this is running.

Run `pg3x72uc` — **clean, monotonic loss descent (~9.5× drop) and a gate
actively moving off its 0.5 init toward 0.63** over the first ~800 steps. Full
numbers + interpretation: [[../../30_Knowledge/experiments/20260715-avid-metaworld-native-gate-healthy]].
This is the strongest evidence yet for the gate-saturation hypothesis in
[[../bug-adapter-gate-saturation-mask-mix]] — the reference composition converges
cleanly on our own task when the gate isn't init-saturated.
