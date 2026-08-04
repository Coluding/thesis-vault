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
| 08-04 | 25194191, 25194201, 25194207 | EA `inpaint_latents` layout fix + 4-way ablation | 192 | ~0.1 | ~20 | 🔄 first two died on job-script bugs (empty glob; `set -u` + unset `PYTHONPATH`) |
| | | | **RUNNING TOTAL (incl. committed)** | | **~8,519** | |

**Remaining against our 100,000 cap: ~91,481**

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
