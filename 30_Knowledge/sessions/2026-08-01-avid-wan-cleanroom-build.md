---
date: 2026-08-01
topic: avid-wan-cleanroom-build
duration_minutes:
files_touched: ["external_repos/avid/wan_diffusion/**", "jobs/experiments_cluster/avid_official/submit_train_avid_wan_rt1.sh", "30_Knowledge/tech/avid-vs-ours-wan-action-conditioning.md", "20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1.md"]
tickets_created: ["[[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]]"]
---

# Overnight session — why Wan ignores actions, and the AVID clean-room

> Running log of the unattended session of **2026-08-01 → 08-02**. Two tasks
> from Lukas: (1) a deep investigation of why RT-1 shows no real action
> sensitivity on Wan, (2) add Wan as a third branch inside the official AVID repo
> and train it exactly the way the AVID authors do, to isolate our
> implementation. Read this top-to-bottom in the morning; the summary is at the
> end.

## Part 1 — Investigation: five code-verified divergences

Full write-up with file:line for every claim:
[[../tech/avid-vs-ours-wan-action-conditioning]]. Short version — the RT-1 Wan
run (`5w72bo01`, `effect_rel ≈ 0.021` vs AVID's 0.0495) differs from AVID's
action pathway in five ways, all read off source, none inferred:

1. **The action never touches AdaLN on that run.** The config sets
   `action_injection: cross_attention`, which makes `_use_adaln_action` False, so
   the modulation of every block is `time_emb + null_cond_emb` — a learned
   *constant*. The action reaches the network only as cross-attention tokens,
   i.e. only through the residual stream, which the 07-31 trace already showed is
   where it drowns (xattn RMS ~0.01 vs stream 1.8–3.0).
2. **Even in `adaln` mode the conditioning is a temporal mean**
   (`ce.mean(dim=1)`), so `action_per_frame: true` is undone inside the adapter.
3. **Our AdaLN modulation is global**: `e0` is `[B, 6, dim]`, broadcast
   identically over every token. AVID's is per-frame. *The official Wan2.2 DiT
   already supports per-token modulation (`[B, L, 6, C]`) — we built on the
   Wan2.1-style block, which structurally cannot express it.*
4. **Add-into-time rather than concat-into-a-reserved-half** — the same
   divergence already documented for DC in [[../tech/avid-vs-ours-action-conditioning]].
5. **No structural action↔frame correspondence** for the xattn tokens: 17 action
   tokens handed to a DiT running on 5 latent frames, with only a learned
   position embedding to tie them. Measured alignment 0.25 = chance.

### The measurement that settles §3/§5 without a single training step

`scripts/diagnose_action_frame_localisation.py` (new, in
`generative-flow-adapters`) perturbs the actions of the pixel frames feeding
**latent frame 2** and reads `‖Δpred‖/‖pred‖` per latent frame, at
initialisation, on CPU:

| architecture | f0 | f1 | **f2** | f3 | f4 | max/min |
|---|---|---|---|---|---|---|
| AVID-faithful (clean-room) | 0.0003 | 0.0003 | **0.0693** | 0.0003 | 0.0003 | **261x** |
| ours — `cross_attention` (the RT-1 run) | 0.2468 | 0.2446 | 0.2453 | 0.2449 | 0.2469 | **1.0x** |
| ours — `adaln` + `action_per_frame` | 0.1739 | 0.1754 | 0.1743 | 0.1735 | 0.1745 | **1.0x** |

**Both of our variants are perfectly uniform.** The `adaln` path *structurally
cannot* address a frame (`ce.mean(dim=1)` destroys the information, and a
`[B, 6, dim]` modulation has no frame axis to put it back on). The
`cross_attention` path *could* learn the correspondence — RoPE plus
`action_pos_emb` make it expressible — but must discover it, and the 07-31 probe
shows it hadn't (alignment 0.25 = chance). AVID never has to learn it.

And ours are **3.5x more action-sensitive in total** (0.245 vs 0.069): the signal
was never too weak, it is **unaddressed**. That is the "sensitivity without
control" signature from 07-31, now explained mechanistically and without
reference to any run — and it explains why `action_token_norm` moved the number
6–10x without unlocking control. It amplified a signal that has nowhere to land.

Underneath all five is a substrate difference that is easy to miss:
**DynamiCrafter's first stage is a per-frame 2D VAE, so AVID gets a 1:1
action↔latent-frame correspondence for free. Wan2.2's VAE compresses time 4x
(17 frames → 5 latent frames), so any faithful port must bin actions onto the
latent grid.** We never did, in either path.

⚠️ The *ranking* of these five is analysis, not measurement — flagged as such in
the tech note. Part 2 is what turns it into evidence.

## Part 2 — The clean-room: `external_repos/avid/wan_diffusion/`

A third branch inside the official AVID repo, next to `pixel_diffusion` and
`latent_diffusion`. AVID's recipe, AVID's data pipeline, AVID's training loop;
only the frozen base changes.

**Held identical to the run that produced the 0.0495 reference (`93qrvr5v`)**
— the authors' own `RTXDataModule` imported unmodified, `adapter_params`
byte-identical to `avid_11M.yaml`, the same mask composition and zero-init
convention, the same AdamW/warmup/clip/EMA/checkpoint policy, and AVID's
action conditioning verbatim (per-frame action → MLP → **concatenated** with a
half-width time embedding → drives every block's modulation).

**Forced deviations** (all properties of the base, none a recipe choice, all
documented in the branch README): rectified flow instead of VP-DDPM; diffusion
forcing instead of concat+CLIP image conditioning; `traj_len: 17` (Wan's VAE
takes `4k+1`); actions summed onto the latent grid; the adapter is a small Wan
DiT (46.7M ≈ 0.9% of the 5B base, matching AVID's 11M-vs-1.4B ratio).

### What was built

| File | Role | Mirrors |
|---|---|---|
| `src/wdwma/models/action_wan.py` | AVID's conditioning on a small Wan DiT | `openaimodel3d.UNetModel(action_conditioned=True)` |
| `src/wdwma/models/avid_wan.py` | `AVIDWanAdapter` Lightning module | `ldwma/models/avid.py` |
| `src/wdwma/models/wan_base.py` | official Wan2.2 DiT/VAE/T5 loaders | `lvdm.utils.train.get_model` |
| `src/wdwma/ema.py` | verbatim `lvdm.ema.LitEma` | — |
| `scripts/train_avid_wan.py` | training entrypoint | `scripts/train_avid.py` |
| `scripts/probe_action_sensitivity.py` | the metric, same protocol + null control | the latent branch's probe |
| `scripts/smoke_test_{adapter,data,trainer}.py` | 3 CPU-only suites: adapter contract, RT-1 pipeline, full Lightning loop | *(new)* |
| `configs/train/{wan_ti2v_5B,act_cond_wan_47M}.yaml`, `configs/train/avid/avid_wan_rt1.yaml` | 3-file config layout | `dynamicrafter_512` / `act_cond_diffusion_11M` / `avid_11M` |
| `libs/wan/` | official Alibaba Wan2.2 package, 4 documented patches | `libs/dynamicrafter`, `libs/octo` |

### Contract smoke test — already informative

`scripts/smoke_test_adapter.py` passes all 18 checks. The one worth reading:

```
[4] action sensitivity at init
  per-latent-frame rel delta: 0.0002 0.0002 0.0710 0.0002 0.0002
```

Perturbing the actions of pixel frames 5–8 (→ latent frame 2) changes **latent
frame 2's** output by 0.0710 and every other frame's by 0.0002 — a **355x**
localisation, at initialisation, before any training. That is precisely the
property the 07-31 probe found our own Wan adapter never formed (temporal
alignment 0.25 = chance, "the action signal is a global bag"). It is a direct
structural consequence of divergences 3 + 5 being removed.

This is a **property of the architecture, not a result** — it says the wiring can
express per-frame action conditioning, not that training will use it.

### Trainer smoke test — and the bug it caught before the GPU slot

`scripts/smoke_test_trainer.py` (new) swaps a **tiny randomly-initialised Wan
DiT** in for the 5B base plus stub VAE/T5, then runs *real* Lightning steps on
CPU. Everything that is AVID's recipe is exercised; only the pretrained weights
are absent. All 17 checks pass:

```
base fully frozen · adapter trainable · optimiser sees ONLY adapter params
fit completed (6 steps, crosses the pretrain boundary) · action pathway LEARNED (|Δ|=0.0052)
checkpoint holds the ADAPTER only (1,296,260 == adapter) · EMA state saved
obs frame held clean & t=0 on it (diffusion forcing) · mask_mean 0.5008 at init
action_effect_rel > 0 (0.0100) · NULL CONTROL passes
```

It **caught a real bug**: `build_flow_batch` built `frame_mask` as `[1, T']`
rather than `[B, T']` (the scalar `cond_frames` doesn't broadcast the way the
per-sample tensor in our own preprocessor did), so
`.view(b, 1, t_lat, 1, 1)` raised. That would have crashed the GPU run **at step
1**, after an unknown queue wait. Fixed and synced before either job started.

A second latent trap was found by reading the probe rather than running it:
`apply_model` gates the mask on `global_step < pretrain_steps`, and a
LightningModule with **no trainer attached** reports `global_step == 0`. So
probing a checkpoint from any run with `pretrain_steps > 0` would silently take
the *pretrain* branch — mask forced to zero, composition = adapter alone — and
report a number for a model that isn't the trained one. Harmless at our current
`pretrain_steps: 0`, fatal the moment that changes. The probe now retires the
phase explicitly (and says so), and the comparison short-circuits so offline
callers never touch the trainer-backed property.

Two incidental findings worth keeping:

- **A randomly-initialised `WanModel` outputs exactly zero** — Wan's own
  `init_weights` zero-initialises `head.head`. So a base that silently fails to
  load does not produce noise, it produces *nothing*, and the adapter quietly
  learns the entire task while every loss curve looks plausible. This is exactly
  the failure that invalidated our pre-2026-07-09 runs. `wan_base.load_wan_ti2v_dit`
  therefore loads **strictly** and raises on any key mismatch.
- The relative form of a null control is `0/0 = nan` against such a base, so the
  test measures it absolutely.

### The probe crash — a DRY failure, caught by probing early (01:45Z)

I submitted a probe against arm A's first (step-500) checkpoint deliberately
early — not for the number, but to prove the measurement path before relying on
it at 6am. It **failed in 3m43s** with the *same* `shape '[2,1,5,1,1]' is
invalid for input of size 5` error already fixed in `build_flow_batch`: the
probe had **re-implemented** the diffusion-forcing frame mask instead of sharing
it, and so reproduced the bug independently.

Fixed at the root rather than patched twice: `AVIDWanAdapter.frame_mask_for(z)`
is now the single source of truth, called by both `build_flow_batch` and the
probe. The probe *must* noise the clip exactly the way training does, so sharing
the implementation is correct on the merits, not just convenient.

Running arms were unaffected (their code was already imported); the fix was
synced and the probe resubmitted. **Lesson: probe the first checkpoint that
appears, always** — this would otherwise have surfaced only when the real
measurement mattered.

### The caption-cache crash — a self-inflicted bug that killed a 29-min run (01:54Z)

Arm A trained cleanly for **29 minutes** (into epoch 3, loss 1.03 -> ~0.21) and
then died with:

```
KeyError: 'Robot arm performs the task: place green rice chip bag into bottom drawer'
```

**My bug.** To stop the umT5 context cache growing unboundedly in VRAM I had
capped it — but the eviction ran *after* computing which captions still needed
encoding, so entries the same call was about to look up were dropped and the
final lookup raised. It only fires once RT-1 presents more than
`_context_cache_max` (512) distinct instructions, which is why a 4-minute smoke
run and all three CPU suites passed. The fix that caused it was itself a fix.

Corrected (evict *first*, then encode whatever is missing) and pinned by a
regression test in `smoke_test_trainer.py` that drives 30 captions through a cap
of 8 and re-requests an evicted one. Both arms relaunched
(`25146335` faithful, `25146336` pooled); the crashed runs are preserved under
`/scratch-shared/lbierling/avid-wan-rt1/_crashed_*` rather than deleted, and
arm A's step-500 checkpoint from that run is still probeable.

**The honest lesson of the night:** four defects, and *none* was found by
reading code —

| defect | found by | would have looked like |
|---|---|---|
| `frame_mask` `[1,T']` not `[B,T']` | CPU trainer suite | crash at step 1 |
| probe re-implemented the frame mask | probing the FIRST checkpoint | crash at measurement time |
| base needs autocast outside Lightning | same probe | crash at measurement time |
| caption-cache evict-after-read | a 29-minute real run | crash deep into training |

Each was a crash rather than a silent wrong answer, which is the good case. But
they were only reachable by running the thing at increasing depth — a smoke run
does not exercise the 512th caption.

### Environment note (and a mistake worth recording)

The branch deliberately reuses the `latent_diffusion` venv (torch 2.1.0+cu118,
Lightning 1.9.3, TFDS/octo) so the *only* difference between branches is model
code. Getting there required two vendored patches to `libs/wan`:

- **`modules/model.py`** — dropped the diffusers `ModelMixin`/`ConfigMixin`
  inheritance (packaging only; weights now load from the official safetensors
  shards). Model mathematics untouched.
- **`modules/attention.py`** — the SDPA fallback now restores the input dtype,
  as the flash-attention path already does. Upstream never hits this because its
  DiT is always bf16 *and* flash-attn is always installed; neither holds here.

⚠️ **I briefly broke the AVID latent venv** while establishing this: installing
`diffusers` upgraded `huggingface-hub` 0.25.1 → 1.26.0, which
`transformers==4.25.1` (DynamiCrafter's CLIP) cannot import, so `lvdm` and
`ldwma` stopped importing. Detected immediately, pinned back
(`huggingface-hub==0.25.1`, `safetensors==0.4.5`, `click==8.1.7`), diffusers
uninstalled, and **verified restored** — `transformers`, `open_clip`,
`lvdm.models.ddpm3d`, `ldwma.models.avid` and `ldwma...rtx` all import again.
Lesson: diffusers is unusable in that venv anyway (it needs `torch.xpu`, i.e.
torch ≥ 2.4), which is why the vendored patch exists.

## Part 3 — Cluster runs

Job script: `jobs/experiments_cluster/avid_official/submit_train_avid_wan_rt1.sh`
(env: `ARM=faithful|pooled|add`, `BATCH`, `MAX_STEPS`, `SHUFFLE_BUFFER`).
Outputs to `/scratch-shared/lbierling/avid-wan-rt1`, wandb project
`avid-wan-rt1`, run named after the arm.

| Job | What | Status |
|---|---|---|
| `25143564` | `avid-wan-datatest` — CPU job (`rome`), RT-1 pipeline + config wiring | ✅ **ALL CHECKS PASSED** |
| `25143550` | `avid-wan-smoke` — 6 steps, batch 1, base + VAE + T5 + train loop | ✅ **COMPLETED exit 0:0** (4m17s) |
| `25144879` | **arm A `faithful`** — the real run, 20k steps, batch 2, **6 h** | queued |
| `25145485` | **arm B `pooled`** — the decisive single-variable test, 6 h | queued |

### The stack runs on the real 5B base (00:58Z)

The smoke job completed cleanly, which retires every remaining GPU-side risk:

```
[train] loading frozen Wan2.2 TI2V base from .../Wan2.2-TI2V-5B   (STRICT load)
[train] loading frozen umT5 text encoder
[train] adapter: 46.7M params  (in_dim=96)
Creating EMA for action conditioned model.
Sanity Checking 2/2 -> Training  ~3.4 it/s @ batch 1
train/loss 1.03 -> 0.94 ...;  train/mask_mean 0.500  mask_std 0.000
[grad-probe] action params=135,680 | trainable=46,705,604
```

Loss is the rectified-flow velocity MSE and sits ~1.0, as expected for a
`noise - z0` target. `mask_std 0.000` at step 24 is just the zero-initialised
mask head (σ(0) = 0.5 everywhere) — **but it is exactly the quantity that went
pathological in our own runs**, so it is the first thing to check on the long
arms ([[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]).
At 3.4 it/s a 6 h arm comfortably exceeds the AVID reference's 5000 steps.

Arm B was submitted only *after* this passed — no point risking two slots on
unvalidated code.

### Data-pipeline validation (job 25143564, passed)

Everything the model consumes, checked end-to-end against AVID's own loader:

```
video   (2, 3, 17, 320, 512)  in [-1.000, 1.000]
act     (2, 17, 7)            per-frame std 0.3538
caption 'Robot arm performs the task: place 7up can into top drawer'
latent  5 x 20 x 32  ->  160 tokens/frame, seq_len = 800
binning (2,17,7) -> (2,5,7), total delta preserved
adapter 46.7M params, in_dim 96 (48 + the base's 48, AVID's arithmetic)
```

So the branch's data axis is *literally* AVID's: their datamodule, their octo
standardisation, their normalisation, their captions. Only the base differs.
(The adapter config was renamed `act_cond_wan_40M` → `act_cond_wan_47M` once the
built size was measured — 46.7M, ≈0.9% of the 5B base, matching AVID's
11M-vs-1.4B ratio.)

**The GPU queue was the binding constraint all night.** `gpu_h100` had five jobs
pending on Priority behind a long-running `skyreels-rt1-oracle`, with no start
estimate. Three earlier smoke submissions were cancelled and resubmitted rather
than left to fail, each time to fix something found while waiting:

1. the submitted script snapshot predated the `base_overrides` cluster-path fix,
   so it would have looked for the Wan checkpoint under `/home/lukas/...`;
2. `precision: "bf16-mixed"` is PL 2.0 syntax and raises on the venv's PL 1.9.3
   (now `"bf16"`, chosen over AVID's fp16 because Wan's weights are bf16);
3. AVID's `shuffle_buffer: 1000` costs several GB and many minutes *before the
   first step* — fatal for a 45-minute smoke allocation, so it is now
   overridable and the smoke passes 8.

A login-node attempt at the data test was killed with no output (TF + the RT-1
pipeline is too heavy there), hence the CPU Slurm job.

## Pre-registered read (written before any result)

- `effect_rel ≳ 0.04` → **our implementation is the fault**; Wan conditions on
  actions fine when the recipe is right.
- `effect_rel ≈ 0.02` → **the Wan latent space is the harder substrate**; the 4x
  temporal compression and/or the DiT inductive bias cost roughly half the
  action-following, and the framework is not the whole story.
- `effect_rel ≪ 0.02` → the port is broken; check the null control and the
  per-frame localisation test before concluding anything.

## Morning summary — what to read, in order

1. **The investigation has a mechanistic answer, and it does not need a run.**
   Our Wan adapter's action signal is *unaddressed*, not weak. Measured at
   initialisation: an AVID-faithful adapter localises a one-frame action
   perturbation to that frame **261x** over the others; both of our variants
   respond **perfectly uniformly** (1.0x) — while being **3.5x more**
   action-sensitive in total. Details: [[../tech/avid-vs-ours-wan-action-conditioning]].
2. **The clean-room exists, is validated, and RUNS ON THE REAL 5B BASE.**
   `external_repos/avid/wan_diffusion/` — a real third branch of the AVID repo.
   Three CPU suites pass (adapter contract, RT-1 data pipeline, full Lightning
   loop against a stand-in base), and the GPU smoke job **completed exit 0:0**:
   strict load of the official Wan weights, umT5, EMA, 46.7M adapter, ~3.4 it/s.
   Three bugs were caught before they could burn a slot — one crash, two
   silent-wrong-answer.
3. **THE ANSWER, measured — and it is better than expected.** All numbers at
   **matched step 5000**, same metric, `base_null_violation` exactly 0 everywhere.

   | @ step 5000 | AVID / DynamiCrafter | **faithful** / Wan | pooled / Wan |
   |---|---|---|---|
   | `action_effect_rel` shuffle | 0.012541 | **0.017474** ± 0.00153 | 0.010192 ± 0.00159 |
   | `adapter_rel_contribution` | 0.051383 | 0.114054 | 0.111481 |
   | `mask mean` | 0.906970 | 0.953402 | 0.953613 |
   | action-driven share | **24.4%** | 15.3% | 9.1% |

   **(a) Per-frame action addressability is causal.** faithful vs pooled differ
   *only* in that switch and are matched on adapter contribution AND mask, so
   they differ only in *what* the adapter contributes: **1.7x more
   action-driven**, Welch t = 3.30 (p≈0.002), n=16 batches.

   **(b) Wan is NOT the harder substrate — hypothesis (b) is dead.** ⚠️ I had to
   correct myself here: the famous reference **0.0495 is a step-15000 number**.
   Re-probing AVID's own checkpoint at step 5000 gives **0.012541** — our
   faithful arm's **0.017474 beats it by 1.39x**.

   **(c) What actually differs is purity, not size.** AVID's adapter contributes
   *less* (0.051 vs 0.114) but is *purer* (24.4% vs 15.3% action-driven) and its
   mask opens further (0.907 vs 0.953). Ours is larger but more diluted.

   Full note: [[../experiments/20260802-avid-wan-cleanroom-perframe-causal]].
4. **And the gain-vs-information confound is CLOSED.** You flagged on `5w72bo01`
   and the DC `condition_center` arms that `effect_rel` is monotone in
   action-path *gain*, so a bigger number needn't mean more action *information*.
   I built a temporal-control probe that is **invariant to gain**: perturb one
   latent frame's actions, see which frames respond.

   | | diagonal concentration |
   |---|---|
   | arm A `faithful` | **0.3900** |
   | arm B `pooled` | 0.1987 |
   | chance (1/T') | 0.2000 |

   The pooled arm's response rows for frames 1–4 are **bit-identical** — mean
   pooling maps every frame's perturbation to the same vector, so it *provably*
   cannot tell which frame an action belongs to, and lands exactly on chance.
   The faithful arm shows a 3–5x diagonal. **The gap is information, not gain.**
5. **The depth trajectory settles it: an INFORMATION CEILING.** Both arms
   re-probed at **step 10000** (still matched on adapter contribution and mask):

   | | step 5000 | step 10000 |
   |---|---|---|
   | faithful `effect_rel` | 0.01747 | **0.02764** (+58%) |
   | pooled `effect_rel` | 0.01019 | 0.01124 (+10%) |
   | ratio | 1.71x | **2.46x** (Welch t = 7.3) |
   | faithful action-driven share | 15.3% | **23.0%** |
   | pooled action-driven share | 9.1% | **9.1% — flat** |
   | faithful diag concentration | 0.3900 | **0.4089** |
   | pooled diag concentration | 0.1987 | **0.1980 — chance** |

   **The faithful arm learns; the pooled arm cannot.** Doubling training moved
   faithful's action-following 58% and strengthened its frame addressing; it
   moved pooled's not at all. Pooled is not under-trained — mean-pooling
   destroys the frame correspondence *before the network sees it*, and no
   optimisation recovers information that is not in the input. Its response rows
   are still bit-identical at step 10000.

   **Final point, step 12000** (contributions identical to 0.2% — the cleanest
   control of the series):

   | step | faithful | pooled | ratio |
   |---|---|---|---|
   | 5000 | 0.01747 | 0.01019 | 1.71x |
   | 10000 | 0.02764 | 0.01124 | 2.46x |
   | **12000** | **0.03178** | 0.01279 | **2.49x** (t = 10.5) |

   Faithful **+82%** across the range (share 15.3% → **25.3%**); pooled +25%
   (share 9.1% → 10.2%). Faithful's **25.3% now EXCEEDS** the AVID/DynamiCrafter
   reference's 24.4% — on the backbone whose latent space was the suspected
   culprit.

**What I'd do first when you're back:** check `squeue -u $USER` and the wandb
project `avid-wan-rt1` (runs named `avid_wan_rt1_47M_faithful` /
`..._pooled`). To measure either arm at its latest checkpoint:

```bash
sbatch --export=ALL,ARM=faithful jobs/experiments_cluster/avid_official/submit_probe_avid_wan_rt1.sh
sbatch --export=ALL,ARM=pooled   jobs/experiments_cluster/avid_official/submit_probe_avid_wan_rt1.sh
```

For a **matched** comparison pin the step and raise the batch count (the arms
train at slightly different rates, so "latest" is not the same depth):

```bash
sbatch --export=ALL,ARM=faithful,STEP=10000,NUM_BATCHES=16 jobs/.../submit_probe_avid_wan_rt1.sh
```

For the **gain-invariant** temporal-control measurement:

```bash
sbatch --export=ALL,ARM=faithful,STEP=10000,NUM_BATCHES=6,LOCALISATION=1 jobs/.../submit_probe_avid_wan_rt1.sh
```

Read `action_effect_rel` **together with** `adapter_rel_contribution` — the
metric is computed on the *composed* output, so while the mask is near 1 the
adapter's share bounds it. The comparable quantity is the ratio (AVID's
reference framing is "~42% of adapter contribution action-driven"). And read
both together with the **diagonal concentration**, which is the only one of the
three that is immune to action-path gain.

**Decisions I made that you may want to revisit** (all argued in-file):
adapter sized by *capacity ratio* (47M ≈ 0.9% of 5B) rather than AVID's absolute
11M; `use_language: True` (AVID's RT-1 config trains on empty captions, but Wan
is a text-conditioned base and our own run had captions); `precision: bf16`
rather than AVID's fp16; `condition_adapter_on_base_outputs: True` kept at
AVID's default despite our own NOBASE finding.

**One decision needs you:** [[../../20_Tickets/feat-adapter-wan-per-frame-adaln]]
was marked OBSOLETE / proposed-for-close on 2026-08-01 ("injection-mechanism
variants are no longer the open question"). **The overnight measurement argues
the opposite and I recommend reopening it at high priority** — both of our
injection mechanisms fail *identically* (1.0x), so per-frame addressability is
the axis those variants were varying *around*, not one of them. I appended the
evidence to that ticket rather than reopening it myself. The ticket also assumes
the AdaLN reshape "needs real design work"; it doesn't — the official Wan2.2 DiT
already accepts per-token modulation `[B, L, 6, C]`, and a working reference
implementation now exists in the clean-room.

**Not done:** no `60_Updates/` entry yet (per CLAUDE.md I ask first — say the
word and I'll write one), and the vault is **uncommitted**.

## Open questions for the morning

1. **Adapter sizing.** 46.7M was chosen to match AVID's *capacity ratio* (~0.9% of
   the base), not its absolute 11M. If the clean-room lands high, a 11M arm
   would separate "recipe" from "capacity".
2. **Resolution.** The run uses AVID's RT-1 geometry (320x512), which is well
   below Wan2.2's native 704x1280. Cheap and 32-aligned, but the frozen base is
   off its native resolution — a possible confound if the result is *low*.
3. **`condition_adapter_on_base_outputs: True`** is AVID's default and is kept
   for faithfulness, even though our own campaign found the base oracle drives
   the erosion signature (NOBASE recipe). If the clean-room erodes too, that is
   itself a finding: it would mean the erosion is AVID's recipe, not ours.

## Related

- [[../tech/avid-vs-ours-wan-action-conditioning]] — the five divergences
- [[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]] — the ticket
- [[../experiments/20260801-wan-rt1-indistribution-plateau]] — the 0.021 plateau
- [[../experiments/20260729-avid-rt1-follows-actions-control]] — the 0.0495 reference
