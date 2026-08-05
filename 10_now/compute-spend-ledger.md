---
type: ledger
scope: compute-accounting
status: living
last_updated: 2026-08-04
---

# Compute spend ledger

Per-experiment SBU accounting, for reporting to the supervisor.

## Accounts

| account | host | budget | note |
|---|---|---|---|
| `gisr108250` (`2410088_26/L2/47`) | `ssh snellius` (`lbierling`) | 280,000 initial · **3,548 remaining** | our own project allocation |
| `gusei17535` (`EINF-17535/L1`) | `ssh snellius1` (`lbierling1`) | 1,000,000 initial · 469,969 remaining | **supervisor's account. HARD CAP: we may spend at most 100,000 SBU.** |

**Billing note (matters, and is easy to get wrong):** on Snellius `#SBATCH --mem`
is a billing knob, not just a resource request. A gpu_h100 node is 720 GB across
4 GPUs, so **180 GB is one GPU's fair share and bills 192 SBU/h; asking 360 GB
bills 384 SBU/h for the same single GPU**. Several of our job templates still
carry `--mem=360G` from before this was understood. Always use 180G unless a run
demonstrably needs more.

Rate used throughout: **192 SBU/h** per H100 at 180 GB, **384** at 360 GB.

---

## Account `gusei17535` — supervisor's (cap 100,000)

| date | job | experiment | rate | hours | **SBU** | outcome |
|---|---|---|---|---|---|---|
| | _(environment setup — login-node only, no GPU billing)_ | | 0 | — | **0** | |
| 08-03 | 25183933 | env smoke (attempt 1) | 192 | 0.03 | **6** | ❌ wandb: no API key |
| 08-03 | 25184424 | env smoke (attempt 2, `WANDB_MODE=offline`) | 192 | 0.39 | **75** | ✅ **PASSED** — 30 steps, ckpt written |
| 08-03 | 25185841/25186717 | action-only (blind), gated | 192 | ~9.5 | ~1,825 | ⚠ gate froze at cap — confounded, see bug ticket |
| 08-04 | 25192286 | **action-only (blind) + `add`** | 192 | 17 (cap) | ~3,265 | 🔄 running |
| 08-04 | 25192313 | **token-norm NOBASE + `add`** | 192 | 17 (cap) | ~3,265 | 🔄 running — tests whether the gate throttled the whole campaign |
| 08-04 | 25192514/724/840, 25192986, 25193042, 25193194 | EasyAnimate integration smokes + NaN debug | 192 | ~0.2 | ~40 | ⚠ NaN cause not found in this batch |
| 08-04 | 25193772, 25193879, 25193964 | EA VAE round-trip + upstream reference trace | 192 | ~0.15 | ~29 | ✅ **falsified the VAE hypothesis**; trace revealed diffusers' pipeline is V5.1-only |
| 08-04 | 25194191, 25194201, 25194207 | EA `inpaint_latents` layout fix + 4-way ablation | 192 | ~0.1 | ~20 | ⚠ first two died on job-script bugs (empty glob; `set -u` + unset `PYTHONPATH`) |
| 08-04 | 25194220 | EA ablation — **falsified** the `inpaint_latents` theory | 192 | 0.04 | ~8 | ✅ all 5 variants NaN incl. upstream's own zero fallback ⇒ cause lay elsewhere |
| 08-04 | (login node, 0 GPU) | transformer key-set audit vs both checkpoints | 0 | — | **0** | ✅ **ROOT CAUSE**: diffusers 0.39.0 matches *neither* ckpt — 216 tensors dropped, 216 random |
| 08-04 | 25197000 | EA load via **EasyAnimate's own classes** + guard | 192 | 0.06 | ~12 | ✅ **GATE PASSED** — 0 unexpected / 0 missing, `denoise` finite |
| 08-04 | 25201096, 25201146, 25201198 | EA latent precompute probes (geometry check) | 192 | ~0.2 | ~40 | ✅ measured latents **16x25x28x36**; caught a read-only-ACL write failure |
| 08-04 | **25201376** | EA **V5** latent precompute, 3 splits | 192 | ~7 (cap 10) | ~1,345 | 🔄 1.25 s/clip, ETA ~5.5 h for ind_train |
| 08-04 | **25201377** | EA **V5.1** latent precompute, 3 splits | 192 | ~7 (cap 10) | ~1,345 | 🔄 separate VAE ⇒ separate cache; runs in parallel with V5 |
| | | | **RUNNING TOTAL (incl. committed)** | | **~11,269** | |

**Remaining against our 100,000 cap: ~88,731**

⚠ **`use_batch_timesteps_for_flow` was a CORRECTNESS bug, not a cosmetic one.** Without
it the preprocessor draws a per-frame sigma, so `x_t` carries mixed noise levels while
`t` is `[B, T]`; EasyAnimate's DiT takes one timestep per sample, so the wrapper collapses
it with `amax` and the model is told "max noise level" for a latent that is not uniformly
noised. That would have quietly degraded **V5.1 — the flow half of the comparison** —
while V5 (which discards the preprocessor's `t` entirely) was unaffected. An asymmetric
defect in a two-arm objective comparison is the worst possible kind: it would have looked
like a result.

**The lesson across five restarts:** each new arm's config was written fresh rather than
diffed against the working Wan reference. A single `comm -23` of the two `training.extra`
key sets surfaced all three problems in seconds, and should have been the FIRST step of
writing a new backbone config, not the fifth.

**Queued (dependency-gated, not yet billed):**

| job | arm | status | est. SBU |
|---|---|---|---|
| 25219819 / 25219820 | EA V5 / V5.1 at **batch 16, workers 12** | 💥 **OUT_OF_MEMORY at 1h10m, step 600** — HOST RAM, not GPU | ~450 spent |
| 25226078 / 25226079 | EA V5 / V5.1 at batch 8 / workers 8 | ⚠ **SIGINT-killed at 5h03, step 4,800** (exit 130, `KeyboardInterrupt` mid-VAE-decode, both within 12 s — external, not OOM, not our code). 306 videos, 0 eval failures | ~1,930 spent |
| 25232749 / 25232750 | EA V5 / V5.1, RESUMED from step 4,800 | 🛑 **KILLED at 1h23** — training against a broken text context (below). ~530 SBU | ~530 spent |

⚠ 25237956 / 25237957 **FAILED within minutes** — the adapter's action cross-attention got
a batch-1 context while CFG feeds a doubled cond‖uncond batch
(`view(2,-1,8,56)` on `1x97x448`). **This is the same error as slurm 25211177, which had
been "fixed" by forcing `guidance_scale=1.0` — i.e. by deleting CFG.** That suppression is
what caused the ghosting/dissolution later chased for hours as a separate mystery. Wan is
immune because its loop calls the model twice separately; EasyAnimate batches them.
Fixed in `_ComposedTransformer`: split the CFG batch, apply the adapter to each half,
recombine — which also keeps the delta unamplified by `w`, as intended.

⚠ **Why the pre-launch check missed it:** the standalone test used
`compose_fn = base_pred * 1.02`. That proved the seam fires on a doubled batch but never
touched the adapter's cross-attention — the only component that actually failed. Testing
the plumbing is not testing the appliance. The verification that worked was the REAL
training path with generation forced at step 1 (slurm 25239879).

✅ **RUNNING 2026-08-05, both arms frame-INSPECTED: V5 = 25240257, V5.1 = 25241732**

⚠ **The "one wrapper, both arms" premise is FALSE for text conditioning.** V5 and V5.1
share the video backbone but not the text backbone, and the flow arm failed FOUR separate
ways before running, each a different consequence of that:

| # | failure | cause |
|---|---|---|
| 1 | `BertTokenizer(vocab_file=None)` | hardcoded V5's BERT; V5.1 is Qwen2VL |
| 2 | `missing 2 required positional arguments` | EA's pipeline wants `text_encoder_2`/`tokenizer_2` even when absent |
| 3 | `Expected reduction dim 1 to have non-zero size` | `""` tokenises to **zero tokens** under Qwen2Tokenizer, but pads to 77 under BERT |
| 4 | `3584 vs 152064` | `Qwen2VLForConditionalGeneration` returns **logits** from `out[0]`, not hidden states |

Failure 3 is the one to remember: the empty-string null context **worked on diffusion and
crashed on flow** — an asymmetry between exactly the two things under comparison. It
crashed loudly, which was lucky; a quieter version of the same defect is how a convincing
but fake "diffusion beats flow" result gets manufactured. Fixed by using a real fixed
sentence (identical on both arms) instead of `""`.

Resolution: stop reimplementing per-stack encoding and delegate to EA's own
`encode_prompt(..., text_encoder_index=)`, which handles both. Encoder/tokenizer classes
are now resolved from each checkpoint's `model_index.json` rather than hardcoded.

**Write-up consequence:** "same backbone, only the objective differs" must be qualified.
The text encoders differ (BERT+T5 vs Qwen2VL) and CFG steers *through* text, so this is
not a peripheral confound. State it alongside the VAE and data-mix differences.

**Both arms verified by LOOKING at rendered frames** (V5 at step 400, V5.1 at the smoke
eval): clean robot-arm video in ground-truth, base and adapted columns. Every earlier
"generation works" claim rested on shapes/finiteness/file-existence and was wrong.
(batch 8 / workers 8, generation ON, CFG 6.0, fresh start — NOT resumed, because the
step-4,800 weights were fitted against a base that rendered noise).

**Working generation recipe** (established offline, frames INSPECTED at every stage):

| component | correct | what had been running |
|---|---|---|
| pipeline | EasyAnimate's OWN `EasyAnimateInpaintPipeline` | diffusers' — silently drops the T5 stream |
| text encoders | both (BERT 1024 + T5 2048), real prompt | none — literal zeros |
| `guidance_scale` | **6.0** | 1.0 (= guidance OFF) |
| VAE dtype | explicit `.to(bf16)` | fp32 biases vs bf16 activations |

Verified through our own wrapper (slurm 25237885): base and adapted both render coherent
video, `compose_fn` fires exactly 25 times (once per denoising step) on batch shape
`(2, 16, 25, 28, 36)` — batch 2 being the doubled cond‖uncond under CFG, so the adapter
delta applies to BOTH halves and passes through **unamplified**:
`(ε_u+δ) + w·((ε_c+δ)−(ε_u+δ)) = ε_u + w·(ε_c−ε_u) + δ`. That matches what `effect_rel`
measures during training (no CFG there). Amplifying δ by `w` instead would be "action
guidance" — a legitimate ablation, but a different experiment.

⚠ A near-miss worth recording: the configs still carried `inference_guide_scale: 1.0`,
which the trainer passes as `guide_scale` and which would have overridden the new default
straight back to guidance-off. Fixing the code default alone was NOT sufficient.

🛑 **(historical) All arms had been stopped pending this. Root cause found 2026-08-05.**

The user killed the 5h03 pair (which I had mis-recorded as an external SIGINT) because
**the generated videos were pure noise**. Inspecting the eval grid — layout is
`[ground_truth | base | adapted]` (`wandb_logger.py:234`) — the ground truth is clean and
the **FROZEN BASE panel is orange blocks**. The adapter was not at fault; it was visibly
pulling output back toward a robot shape.

Two independent defects in the text conditioning, both mine:

1. **Zeros instead of the empty-prompt embedding.** The wrapper fed a literal zero tensor
   to the text port — a VRAM optimisation stated in its own docstring. The real
   empty-prompt embedding measures **absmax 21.0** (e1) — nowhere near zero. Wan's wrapper
   has always used a cached `uncond_context.pt` (T5 of `""`), which was documented in this
   very codebase and not applied here. **Fixed**: `_real_null_embeds()` encodes `""` once
   on CPU, caches two small tensors, drops the 7.7 GB of encoder weights.
2. **The entire T5 stream missing at generation.** diffusers'
   `EasyAnimateInpaintPipeline` supports ONE text encoder (it passes
   `encoder_hidden_states` and never `encoder_hidden_states_t5`) because it is shaped for
   V5.1's single Qwen2VL encoder. **V5 has two.** Generation therefore ran with half its
   conditioning absent. Fixing this needs EasyAnimate's OWN inpaint pipeline, not a shim.

**Training was structurally sounder than generation:** `denoise` passes BOTH streams, so
it suffered only defect (1) — now fixed. Because the zero context was symmetric across
arms, the diffusion-vs-flow comparison stayed internally valid; both arms were simply
running against a handicapped base, so the absolute effect_rel values understate what a
correctly-conditioned base would give.

**The process failure worth keeping:** generation was declared working on the basis that
it ran, returned correct shapes and finite values, and wrote `.mp4` files. Nobody looked
at a frame. The user looked once and saw it immediately. Shape-and-liveness checks cannot
substitute for viewing the artifact — the same lesson as the VAE round-trip and the
loader key-set, now at the output end of the pipeline.

⚠ **Step numbering after the resume.** `--init-from` restores WEIGHTS ONLY — optimizer
state and `global_step` are not carried over — so the resumed runs' wandb curves restart
at step 0 while the weights already have **4,800 steps** of training behind them. True
total = reported step + 4,800. Both arms were resumed identically from their own
step-4,800 checkpoints, so the V5-vs-V5.1 comparison remains matched; only the absolute
step axis is offset. Expect a brief transient as Adam's moments rebuild.

Chosen over a fresh restart because a clean relaunch would have discarded 10 GPU-hours
(~1,930 SBU) and 5 h of wall-clock for the second time in one night, with a submission
deadline approaching. The cost is a documented offset rather than lost training.

⚠ **The batch-16 OOM was a host-RAM cgroup kill, and it was my error.** Evidence:
`MaxRSS 140.6 GB` vs `ReqMem 180G`, `DataLoader worker ... killed by signal: Killed`,
`oom_kill event`, and **zero** `CUDA out of memory` in either log. The GPU was never the
constraint.

Cause: batch 8→16 and workers 8→12 were raised **together**, so each of 12 workers
prefetched batches of 16 × 97-frame 480×640 uint8 clips (~1.4 GB per batch) on top of the
main process. Two changes at once, neither isolated.

Worse, the change bought **nothing**: measured throughput at batch 16 was ~2.9 samples/s
against ~2.96 at batch 8 — the GPU was already compute-bound, so the idle *memory* the
user spotted was not the limiting resource. Reverted to batch 8 / workers 8, the
configuration that had already run 3.2 h to step 4,000 without incident.

W&B project `coluding/EasyAnimate-objective-acwm-robotarm` — both arms log there, so
they overlay directly.

**Restart history — ~6.5 GPU-hours discarded (~1,250 SBU), most of it avoidable:**

| launched | killed after | why |
|---|---|---|
| 25202174/5 | never started | gated on a generation-eval memory check that failed — the gate working |
| 25209282/3 | **~3.2 h, at step 4,000** | ran with `--no-eval-gen`; user requires generations DURING the run |
| 25216954/5 | ~10 min | generation on, but the configs had **no `wandb:` block**, so no logger was constructed and every sample would have been discarded |
| 25217387/8 | ~12 min | W&B live, but batch 8 used only 37% of a 93 GB H100 — raised to 16 (→ 59% mem, ~80% GPU util) |
| 25218521/2 | ~25 min | **no `eval_step_schedule`** ⇒ `_native_eval_grid` no-ops ⇒ **zero sample videos all night**, silently, with `gen_eval=on` still printed. Config audit against the Wan arms also found `inference_frame_num` defaulting to 121 (vs the 97-frame training window) and a missing `use_batch_timesteps_for_flow` |

⚠ **The 25209282/3 kill was more expensive than it looked and was partly avoidable.**
Those arms had reached **step 4,000** — already past the Wan arms' 3,054 steps in 14.8 h —
and the weights were discarded. `--init-from <ckpt>` would have preserved them (weights
only; optimizer state and `global_step` are not restored). A fresh restart was chosen so
the two arms keep clean, directly comparable step counts, which is defensible — but the
cost should have been stated as 4,000 discarded steps, not "~2 h".

The `step_00004000.pt` checkpoints still exist under both run dirs if that training is
ever worth recovering.

The third is the instructive one: the startup banner printed `gen_eval=on`, so the run
*looked* correct. "Generation runs" and "generation is recorded" are separate claims and
only the first had been checked. Caught by the user asking "is it logging to wandb?" —
otherwise 17 h would have produced no samples. The Wan configs carry that block; the EA
configs were written fresh and never got one.

`afterok` means a failed precompute cancels the arm instead of training on a
half-filled cache — the countermeasure to the LoRA episode, where six failed launches
went unnoticed because nothing gated them. An earlier pair (25202134/5) was
additionally gated on a generation-eval memory check; that check **failed** and
correctly blocked them, which is the gate working rather than a setback.

**Generation is now ON in both arms** (superseding an earlier `--no-eval-gen` state).
Six distinct bugs had to be fixed in `EasyAnimateVideoModel.generate` first: Wan-dialect
kwargs swallowed by `**kwargs`; `inference_max_area` defaulting to 901,120 px (~70k
tokens at stride 8) and OOM-ing; a CPU generator used against CUDA tensors; a VAE dtype
split from `force_upcast`; HWC uint8 frames where CHW float was assumed; and — the one
that actually mattered — `_ComposedTransformer` returning a bare Tensor where the
pipeline expects a 1-tuple, so its `[0]` stripped the batch dim and a later
`chunk(2, dim=1)` split the TEMPORAL axis 25 → 13+12.

Both generation paths (`_native_eval_grid`, `_native_quality_eval`) are now wrapped so a
rollout fault logs a traceback and training CONTINUES. Diagnostics must never be able to
destroy the multi-hour run they are diagnosing.

Campaign total if both complete: **~17,800 SBU**, under a fifth of the cap.

**Cost of the EasyAnimate integration, stated plainly for the supervisor:** ~3 days of
wall-clock, **~130 SBU** (0.13% of the cap). The expensive resource was attention, not
allocation. The single highest-leverage fix was a ~30-line loader guard
(`_load_checked`) that turns a silent partial load into an immediate error; it would
have caught this in minutes. Recorded in
[[../20_Tickets/bug-diffusers-silently-drops-vae-weights]].

The whole EasyAnimate integration — three days of wall-clock, ~13 jobs — has cost
**~89 SBU**, under 0.1% of the cap. Worth stating plainly when reporting: the
integration was expensive in *time*, not in *allocation*, and the two should not be
conflated when justifying the spend.

⚠ `accinfo` on this account also lags (and it includes the supervisor's own jobs,
e.g. `dc-pdd-full`), so **this itemised table — not `accinfo` — is our figure against
the cap.**

### Environment provenance (for reproducibility)

- uv 0.12.1 · Python 3.11.13 · **torch 2.9.0+cu128** · **flash-attn 2.8.3+cu128torch2.9**
  (prebuilt wheel, `mjun0812/flash-attention-prebuild-wheels` release `v0.4.17`).
- ⚠ Install from **`pyproject.toml`**, NOT `requirements-lock.txt` — the lock is
  stale and pins `torch==2.11.0`, which silently yields cu130 and a flash-attn
  mismatch. `pyproject.toml` pins 2.9.0 through an explicit `pytorch-cu128` index.
- Checkpoint (32 GB) and ACWM data (132 GB) are **not duplicated**: both accounts
  are on the same cluster/filesystem, so `lbierling` granted `lbierling1` targeted
  POSIX ACL read access (`setfacl -m u:lbierling1:rX`) to exactly those paths.
  Revoke with `setfacl -R -x u:lbierling1 /scratch-shared/lbierling`.
  Note quota still bills the **owner** (`lbierling`) for that storage.
- `WANDB_MODE=offline` is the default in jobs on this account, so no personal API
  key lives in the supervisor's home dir; `wandb sync` from our account later.

---

## Account `gisr108250` — ours

Spend before 2026-08-01 is not itemised here (280,000 → ~48,000 over the
project's life). Itemised from 2026-08-01, when tracking began.

| date | job | experiment | rate | hours | **SBU** | outcome |
|---|---|---|---|---|---|---|
| 08-01→02 | 25107236 | Wan × RT-1 token-norm nobase | 192 | 25.9 | 4,973 | ⚠ in-sample eval |
| 08-01→02 | 25112302 | SkyReels × RT-1 token-norm nobase | 192 | 22.1 | 4,243 | 🛑 void (`dataset_size=76`) |
| 08-02 | 25133625 | SkyReels × RT-1 oracle | 192 | 7.7 | 1,478 | 🛑 void (same bug) |
| 08-02 | 25141979 | DC D3 shortcut arm | 384 | 3.8 | 1,459 | killed (scope cut to Wan) |
| 08-02 | 25141980 | DC D3 no-shortcut control | 384 | 3.5 | 1,344 | killed (scope cut) |
| 08-02 | 25141988 | Wan D3 shortcut | 384 | 16.7 | 6,413 | ✅ **D3 mechanism, 9× over control** |
| 08-02 | 25151959 | Wan D3 no-shortcut control | 192 | 13.2 | 2,534 | ✅ the matched control |
| 08-02 | 25144197 | DC structure triad probe | 192 | 0.8 | 154 | ✅ **DC has real action control** |
| 08-02 | 25145408 | Wan RT-1 information probe | 192 | 0.2 | 38 | ✅ |
| 08-02 | 25154218 | rollout wall-clock benchmark | 192 | 1.0 | 192 | ✅ **timing curves** |
| 08-02 | 25152246 | wall-clock (timed out) | 192 | 1.0 | 192 | ⚠ partial, superseded |
| 08-02 | 25158719/20 | AVID × Wan × RT-1 probes | 192 | 0.6 | 115 | ✅ **66%→26% backbone effect** |
| 08-02 | 25155284 + others | step-size blindness probes | 192/384 | ~0.6 | ~180 | ✅ |
| 08-03 | 25161324 | Wan concat injection arm | 192 | 7.4 | 1,421 | ✅ **negative result** |
| 08-03 | 25154241 | Wan × RT-1 binned | 384 | 5.8 | 2,227 | ✅ first clean held-out RT-1 |
| 08-03 | ×6 LoRA attempts | LoRA + action pathway | 192 | ~0.75 | ~145 | ❌ 6 failures (see below) |
| 08-03 | 25183721 | **LoRA + action pathway (7th)** | 192 | 9.0 | ~1,730 | 🔄 running clean past all prior failure points |
| | | | **TOTAL (itemised)** | | **~27,100** | |

### The six LoRA failures — ~145 SBU

Cheap in SBU, expensive in wall-clock. All six trace to one root cause worth
recording: **LoRA is the first adapter whose trainable weights live INSIDE the
frozen base**, so every framework assumption built for delta-adapters broke.

| # | job | died at | cause |
|---|---|---|---|
| 1 | 25164029 | 3:47 | recursion — LoRA re-entering `compose_fn` during `generate()` |
| 2 | 25165969 | 19:26 | no `grad_fn` — the base denoiser is `@torch.no_grad()` |
| 3 | 25166348 | 4:22 | OOM — a differentiable 5B pass stores all activations |
| 4 | 25172018 | 4:09 | OOM at batch 6 (batch was never the dominant term) |
| 5 | 25181349 | 4:00 | OOM — gradient checkpointing **silently inactive** (bad gate) |
| 6 | 25182655 | — | running |

---

## Conventions

- One row per submitted job. Log the SBU even for failures — they are part of
  the honest cost of the work.
- `rate × hours` from `sacct` (`billing=` in `AllocTRES` × `Elapsed`).
- `accinfo` **lags by up to a day**; the itemised sum here is the live figure.
- Mark outcome: ✅ used · ⚠ compromised · 🛑 void · ❌ failed · killed.
