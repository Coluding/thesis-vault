---
type: exp
scope: adapter
status: in-progress
priority: high
created: 2026-08-01
updated: 2026-08-02
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/tech/avid-vs-ours-wan-action-conditioning]]", "[[../../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]", "[[../../30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control]]", "[[exp-adapter-our-framework-avid-replication-robotarm]]"]
---

# AVID clean-room on Wan — RT-1 (is the Wan gap ours or the backbone's?)

## Question

Our Wan adapter settles at `action_effect_rel ≈ 0.021` on RT-1 (`5w72bo01`,
[[../../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]).
The official AVID recipe reaches **0.0495** on the same dataset
(`93qrvr5v`, [[../../30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control]]).
Two live explanations:

- **(a) our implementation** — the five code-verified divergences in
  [[../../30_Knowledge/tech/avid-vs-ours-wan-action-conditioning]];
- **(b) the Wan backbone** — in particular its 4x-temporally-compressed latent
  space, which destroys the action↔frame correspondence AVID gets for free from
  DynamiCrafter's per-frame 2D VAE.

Every experiment so far has varied **our** code, so (a) and (b) have never been
separated. This one does.

## Design

A **third branch inside the official AVID repo**:
`external_repos/avid/wan_diffusion/`, alongside `pixel_diffusion` and
`latent_diffusion`. AVID's recipe, AVID's data pipeline, AVID's training loop —
only the frozen base changes.

**Held identical to the 0.0495 reference**

| | |
|---|---|
| data | `ldwma.lightning.data_modules.rtx.RTXDataModule`, the authors' own RT-1 loader, imported unmodified |
| composition | `base·m + adapter·(1−m)`, `m = σ(logit + init_mask_bias)`, mask head zero-init, pred head not |
| `adapter_params` | `condition_adapter_on_base_outputs: True`, `learnt_mask: True`, `init_mask_bias: 0.0` — byte-identical to `avid_11M.yaml` |
| action conditioning | per-frame action → MLP → **concat** with half-width time emb → drives every block's modulation |
| optimisation | AdamW lr 1e-4, linear warmup 250, grad-clip 1.0, EMA on adapter, base frozen, adapter-only checkpoints |

**Forced deviations** (properties of the base, not recipe choices — full
rationale in the branch README):

1. rectified flow instead of VP-DDPM (Wan's frozen weights only mean anything
   under the flow parameterisation);
2. diffusion forcing instead of concat + CLIP image conditioning;
3. `traj_len: 17` not 16 (Wan2.2 VAE takes `4k+1` frames → 5 latent frames);
4. actions **summed onto the latent temporal grid** before conditioning
   (`bin_actions_to_latent_frames`) — AVID needs no binning;
5. the adapter is a small **Wan DiT** (46.7M ≈ 0.9% of the 5B base, matching
   AVID's 11M-vs-1.4B ratio) rather than a small UNet.

## Ablation arms (single-variable, inside the clean-room)

The adapter carries two switches whose **defaults are AVID's**. Each reproduces
one divergence from [[../../30_Knowledge/tech/avid-vs-ours-wan-action-conditioning]]
*inside* the otherwise-faithful clean-room, so a difference is attributable to
that one change rather than to "our whole implementation".

| Arm | Switch | Reproduces | Divergence |
|---|---|---|---|
| **A `faithful`** | — | AVID | — |
| **B `pooled`** | `action_temporal_pool=True` | `ce.mean(dim=1)` + global AdaLN broadcast | §2 + §3 |
| **C `add`** | `action_time_combine="add"` | `e = time_emb + cond_proj(ce)` | §4 |

Verified at init by `smoke_test_adapter.py` (perturb the actions of the pixel
frames feeding latent frame 2; read the per-latent-frame response):

```
faithful   0.0002 0.0002 0.0710 0.0002 0.0002   355x localised to frame 2
pooled     0.0141 0.0141 0.0141 0.0141 0.0141   perfectly uniform — the "global bag"
add        0.0004 0.0004 0.1277 0.0004 0.0004   still localised (independent axis)
```

The `pooled` row reproduces the exact signature measured on our own adapter
([[../../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]],
temporal alignment 0.25 = chance). Launch order: **A first**, then **B** (the
decisive test of the top-ranked hypothesis), then C if compute allows.

## Prediction (pre-registered 2026-08-01, before any result)

- `effect_rel ≳ 0.04` (near the reference) ⇒ **our implementation is the fault**;
  the Wan backbone conditions on actions fine when the recipe is right.
- `effect_rel ≈ 0.02` (near ours) ⇒ **the Wan latent space is the harder
  substrate**; the 4x temporal compression and/or the DiT inductive bias cost
  roughly half the action-following, and the framework is not the whole story.
- `effect_rel ≪ 0.02` ⇒ something in the port is broken; check the null control
  and the per-frame localisation smoke test before drawing any conclusion.

## Measurement

`wan_diffusion/scripts/probe_action_sensitivity.py` — the same metric, same
paired-noise protocol and same null control as the latent branch's probe
(`action_effect_rel = ‖pred_true − pred_variant‖ / ‖pred_true‖`, variants
shuffle/zero). Null control: the frozen base never receives `act`, so
`base_null_violation` must be ~0 or the numbers are void.

## Status — COMPLETE (2026-08-02 06:00Z)

- ✅ **branch built**: `external_repos/avid/wan_diffusion/` — adapter, Lightning
  module, official-Wan loaders, 3-file config layout, probe, job scripts,
  vendored official Wan2.2 package (4 documented patches, none touching model
  mathematics).
- ✅ **contract smoke test passes** (`scripts/smoke_test_adapter.py`, 18 checks) —
  including the property our own adapter lacks: perturbing latent frame 2's
  action changes frame 2's output **355x more** than any other frame's.
- ✅ **data pipeline validated on the cluster** (job `25143564`, CPU): video
  `(2,3,17,320,512)` in [-1,1], per-frame `act (2,17,7)`, real RT-1 captions,
  latent `5x20x32` → seq_len 800, binning exact, adapter 46.7M / in_dim 96.
- ✅ **head-to-head localisation measured** (see the tech note): clean-room 261x,
  both of our variants 1.0x (perfectly uniform).
- ✅ **END-TO-END ON THE REAL 5B BASE** — smoke job `25143550`, **COMPLETED
  exit 0:0** in 4m17s (2026-08-02 00:58Z). Confirms the whole stack on GPU:

  ```
  [train] loading frozen Wan2.2 TI2V base from .../Wan2.2-TI2V-5B   (strict load)
  [train] loading frozen umT5 text encoder
  [train] adapter: 46.7M params  (in_dim=96)
  Creating EMA for action conditioned model.
  Sanity Checking 2/2 -> Training  ~3.4 it/s @ batch 1
  train/loss 1.03 -> 0.94 ... ; train/mask_mean 0.500  mask_std 0.000 (zero-init logit)
  [grad-probe] action params=135,680 | trainable=46,705,604
  ```

  Loss is the rectified-flow velocity MSE and sits ~1.0 as expected. `mask_std
  0.000` at step 24 is the zero-initialised mask head, not the gate-saturation
  pathology — **worth re-checking on the long runs** (cf.
  [[../bug-adapter-gate-cap-equals-init-freezes-gate]]).
- ⚠️ First launch (`25144879` / `25145485`) **crashed at 29 min** on a caption-
  cache eviction bug of mine (evicted after computing what to encode → KeyError
  past 512 distinct RT-1 instructions). Fixed + regression-tested; outputs kept
  under `_crashed_*`.
- ✅ **Both arms COMPLETE**: arm A `faithful` **`25146335`** → step **14000**,
  arm B `pooled` **`25146336`** → step **13000**. Ended on the 6 h walltime
  (Slurm `TIMEOUT`, exit 0:0, no tracebacks — the normal ending, not a failure).
  28 / 26 checkpoints retained under
  `/scratch-shared/lbierling/avid-wan-rt1/avid_wan_rt1_47M_{faithful,pooled}/`.
- ✅ **Result at step 5000 measured** (see below) — the answer to the ticket.
- ⏳ arm C `add` not submitted — arms A and B answer the question; C is a
  refinement.

## First measurement — arm A, step 500 (NOT the verdict)

Probe `25146448` on `_crashed_faithful_25144879/checkpoints/epoch=1-step=500.ckpt`
(the pre-cache-fix run; training up to the crash was correct, only caption
*lookup* failed). 60 samples = 4 batches x 5 sigmas x 3 draws.

```
adapter_rel_contribution : 0.042128      mask mean : 0.974349
  variant  action_effect_rel   cos(true,var)   base_null_rel
  shuffle           0.003250        0.999994       0.000e+00
     zero           0.002499        0.999997       0.000e+00
base_null_violation (max): 0.000e+00      NULL CONTROL: PASS
```

**What this does establish:** the measurement path is sound end-to-end, and the
**null control is exactly zero** — the frozen Wan base is provably invariant to
the action perturbation, so no signal leaks through the harness.

**What it does NOT establish:** anything about the hypothesis. At step 500 the
mask sits at 0.974, i.e. the adapter contributes **2.6%** of the composed
output — it has barely entered the composition. `action_effect_rel` is measured
on the *composed* prediction, so it is bounded by that share.

Normalising it out for comparability with the vault's AVID framing
("~42% of adapter contribution action-driven"):

| | effect_rel | adapter contrib | action-driven share | step |
|---|---|---|---|---|
| **this run (arm A)** | 0.0033 | 0.042 | **~7.7%** | **500** |
| AVID x ACWM Arm (`rqp4s3gp`) | 0.0295 | — | ~42% | 5000 |
| AVID x RT-1 (`93qrvr5v`) | 0.0495 | — | ~66% | — |

⚠️ **10x less trained than the reference.** Do not read the gap as a result. The
comparison to make is arm A vs arm B at *equal* step count, which is what the
relaunched runs are for.

## Matched-depth A/B — step 1500 (in-flight, single measurement)

Probes `25146973` (faithful) / `25146974` (pooled), both on
`epoch=5-step=1500.ckpt`, same data, same seeds, 60 paired samples each.
Architecture verified in-log: `ARM: action_time_combine=concat
action_temporal_pool=True` for the pooled probe — the switch changes no
parameter shapes, so this check is load-bearing.

| @ step 1500 | arm A `faithful` | arm B `pooled` | ratio |
|---|---|---|---|
| `action_effect_rel` (shuffle) | **0.005744** | 0.004359 | 1.32x |
| `action_effect_rel` (zero) | **0.004914** | 0.004185 | 1.17x |
| `adapter_rel_contribution` | 0.068342 | 0.063111 | 1.08x |
| action-driven share | **8.4%** | 6.9% | 1.22x |
| `base_null_violation` | 0.000e+00 | 0.000e+00 | PASS |

Arm A trajectory: `effect_rel` 0.00325 (step 500) → 0.00574 (1500), with
`adapter_rel_contribution` 0.042 → 0.068. Both rising together; the
action-driven share is roughly flat (7.7% → 8.4%), i.e. the adapter is *not*
buying loss reduction by going action-blind.

### A second signal, independent of the probe: the mask admits the faithful adapter more

Training-dynamics snapshot at step ~2000 (from the run logs, not the probe):

| @ step 2000 | arm A `faithful` | arm B `pooled` |
|---|---|---|
| `train/loss` | **0.169** | 0.175 |
| `mask_mean` | **0.945** | 0.977 |
| → adapter's share of the composed output | **5.5%** | 2.3% |
| `mask_std` | **0.0238** | 0.0175 |
| mask range | 0.844–0.984 | 0.875–0.992 |

The learnt mask is effectively the model's own verdict on **how useful the
adapter is**, and it is admitting the faithful adapter at ~2.4x the share, with
more input-dependent variation. This is the mechanism the hypothesis predicts —
per-frame addressability makes the adapter worth trusting — and it is measured
by a completely different route than `action_effect_rel`.

**Confirmed as a trend, and WIDENING** (step 2000 → 3500):

| | step 2000 | step 3500 | adapter share |
|---|---|---|---|
| **faithful** `mask_mean` | 0.945 | **0.914** | 5.5% → **8.6%** |
| **pooled** `mask_mean` | 0.977 | 0.977 | 2.3% → **2.3%** (flat) |
| faithful `mask_std` | 0.0238 | **0.033** | |
| pooled `mask_std` | 0.0175 | 0.0199 | |

The faithful arm's mask **opens monotonically** — the composition keeps deciding
the adapter is worth more, now admitting it at **3.7x** the pooled arm's share
(up from 2.4x). The pooled arm's mask is **flat at 0.977**: after 3500 steps the
composition has learned nothing new about whether to trust that adapter.

Same actions, same data, same capacity, same objective — the only difference is
whether the conditioning can address an individual frame. This is the predicted
mechanism showing up in a quantity the probe never touches.

### 🛑 RETRACTED 2026-08-02 — this mask divergence did not survive

The trend above was read off **instantaneous progress-bar snapshots** (one batch
each). It does **not** hold:

- The step-5000 **probe** — 240 samples per arm, not one batch — measured the
  masks as **essentially equal**: 0.9534 (faithful) vs 0.9536 (pooled).
- At step ~10000 the instantaneous values have if anything **reversed**:
  faithful 0.934 vs pooled 0.895.

So "the composition admits the faithful adapter more" was **noise**, and the
mechanism story built on it ("the mask is a learned verdict on usefulness") is
**not supported**. Retained here rather than deleted because the retraction is
the useful record: single-batch `mask_mean` is far noisier than it looks, and
the probe's averaged value is the one to trust.

Fortunately the headline result does not depend on it — it is *strengthened* by
the masks being equal, since that removes a confound from the step-5000 A/B.

### Honest reading

The direction matches the prediction, **but the magnitude does not follow the
architecture**. At initialisation the faithful conditioning is **261x** more
frame-addressable than the pooled one
([[../../30_Knowledge/tech/avid-vs-ours-wan-action-conditioning]]); after 1500
steps that has produced only ~1.2–1.3x more action-following. Three reasons not
to conclude anything yet:

1. **No error bars.** One probe per arm, 60 paired samples. A 1.3x gap is not
   distinguishable from noise on this evidence.
2. **The adapter barely participates.** `mask ≈ 0.97` ⇒ ~3% of the composed
   output. `action_effect_rel` is measured on the composed prediction, so both
   arms are measuring a thin slice of a mostly-frozen-base output.
3. **Step 1500 vs the reference's 5000.**

If the gap stays ~1.3x at depth *with* error bars, that is itself a finding: it
would mean per-frame addressability is **necessary but far from sufficient**,
and that the dominant limiter is something else (most likely the mask keeping
the adapter out of the composition, i.e. the base-oracle/erosion dynamics
already documented on our own side).

## Matched-depth A/B at step 5000 — the reference's depth

Probes `25148010` (faithful) / `25148011` (pooled), both on
`epoch=19-step=5000.ckpt`, 60 paired samples each. Arm verified in-log
(`action_temporal_pool=False` vs `True`).

**The step-1500 confound is gone**: by step 5000 the two arms have converged to
essentially *identical* adapter contribution and mask, so they no longer differ
in how much the adapter supplies — only in **what** it supplies.

| @ step 5000 | arm A `faithful` | arm B `pooled` |
|---|---|---|
| `adapter_rel_contribution` | 0.1120 | 0.1150 |
| `mask mean` | 0.9540 | 0.9535 |
| `action_effect_rel` shuffle | **0.02272** ± 0.00566 | 0.01235 ± 0.00408 |
| `action_effect_rel` zero | **0.01878** ± 0.00405 | 0.00998 ± 0.00186 |
| **action-driven share** (shuffle) | **20.3%** | 10.7% |
| action-driven share (zero) | 16.8% | 8.7% |
| `base_null_violation` | 0.000e+00 | 0.000e+00 PASS |

Per-batch means (the independent unit), shuffle:

```
faithful : 0.020488  0.013284  0.039088  0.018021
pooled   : 0.006000  0.005261  0.022123  0.016023
```

### Statistical honesty

Point estimate: **~1.9x more action-driven at equal adapter contribution**, and
faithful is higher at every batch index. But with **n = 4 batches** the unpaired
t is only ~1.5 (shuffle) / ~2.0 (zero) — **suggestive, not significant**. Batch 3
is high for both arms, which is what you expect from shared data variance and is
why the batch is the right unit.

⇒ Reran pinned to the same checkpoint with **16 batches** — see below. **The
1.9x was noise-inflated; the answer is 1.68x, and it IS significant.**

## ✅ HIGH-POWER RESULT — step 5000 pinned, n=16 batches (240 samples/arm)

Probes `25148083` (faithful) / `25148084` (pooled). Full write-up:
[[../../30_Knowledge/experiments/20260802-avid-wan-cleanroom-perframe-causal]].

| | arm A `faithful` | arm B `pooled` |
|---|---|---|
| `adapter_rel_contribution` | 0.1141 | 0.1115 |
| `mask mean` | 0.9534 | 0.9536 |
| `action_effect_rel` shuffle | **0.017474** ± 0.001531 | 0.010192 ± 0.001587 |
| `action_effect_rel` zero | **0.013930** ± 0.000953 | 0.008706 ± 0.001010 |
| **action-driven share** | **15.3%** | 9.1% |
| null | 0.000e+00 | 0.000e+00 PASS |

Welch t on batch means: **3.30** (shuffle, p≈0.002), **3.76** (zero, p≈0.0007).

**Verdict on the pre-registered prediction.** Neither branch of the original
either/or is right as stated:

- Per-frame addressability **is** a real, causal cause (**1.7x**, significant,
  at matched adapter share and a clean null) — so part of the gap **is ours**.
- **Hypothesis (b) is dead.** ⚠️ The quoted reference 0.0495 is a **step-15000**
  number; comparing our step-5000 arms to it was a 3x depth mismatch. Re-probing
  AVID's own checkpoint **at step 5000** (job `25148170`) gives **0.012541** —
  our faithful arm's **0.017474 BEATS it by 1.39x**. The Wan latent space is not
  the harder substrate.

| @ step 5000 | AVID / DC | **faithful** / Wan | pooled / Wan |
|---|---|---|---|
| effect_rel shuffle | 0.012541 | **0.017474** | 0.010192 |
| `adapter_rel_contribution` | 0.051383 | 0.114054 | 0.111481 |
| `mask mean` | 0.906970 | 0.953402 | 0.953613 |
| action-driven share | **24.4%** | 15.3% | 9.1% |

- **What actually differs is purity, not size.** AVID's adapter contributes
  *less* (0.051 vs 0.114) but is *purer* (24.4% vs 15.3% action-driven), and its
  mask opens further (0.907 vs 0.953). Ours is larger but more diluted.
- **Neither is at its ceiling**: AVID goes 0.0125 (5k) → 0.0495 (15k), 4x from
  depth alone. Our arms run to ~12000.

### The confound is closed: it is information, not gain

Temporal-control probe (`--localisation`, jobs `25148466`/`25148467`, step 5000):
perturb one latent frame's actions, see which frames respond. **Invariant to
action-path gain**, unlike `effect_rel`.

| | diagonal concentration |
|---|---|
| arm A `faithful` | **0.3900** |
| arm B `pooled` | 0.1987 |
| chance (1/T', T'=5) | 0.2000 |

The pooled arm's rows for frames 1–4 are **bit-identical** (`0.0466 0.0110
0.0114 0.0120 0.0131` four times) — mean-pooling maps every frame's perturbation
to the same vector, so it provably cannot tell which frame an action belongs to.
The faithful arm shows a 3–5x diagonal among predicted frames.

⇒ The 1.7x `effect_rel` gap reflects **action information**, not a louder action
pathway. This answers the `effect_rel`-is-gain-monotone caveat raised on
`5w72bo01` and the DC `condition_center` arms.

### Depth trajectory 5000 → 10000: the pooled arm hits an INFORMATION CEILING

| | step 5000 | step 10000 |
|---|---|---|
| faithful `effect_rel` | 0.01747 | **0.02764** (+58%) |
| pooled `effect_rel` | 0.01019 | 0.01124 (+10%) |
| ratio | 1.71x | **2.46x** (Welch t = 7.3) |
| faithful share | 15.3% | **23.0%** |
| pooled share | 9.1% | **9.1% — flat** |
| faithful diag conc | 0.3900 | **0.4089** |
| pooled diag conc | 0.1987 | **0.1980 — chance** |

Adapter contribution and mask stay matched (0.1202/0.1235, 0.9468/0.9411).
Pooled's response rows are **still bit-identical** at 10000. It is not
under-trained: the information is destroyed before the network sees it.

**Final point @12000** (`25149903`/`25149904`; contributions identical to 0.2% —
0.125675 vs 0.125455):

| step | faithful | pooled | ratio | faithful share | pooled share |
|---|---|---|---|---|---|
| 5000 | 0.017474 | 0.010192 | 1.71x | 15.3% | 9.1% |
| 10000 | 0.027641 | 0.011239 | 2.46x | 23.0% | 9.1% |
| **12000** | **0.031779** | 0.012788 | **2.49x** (t=10.5) | **25.3%** | 10.2% |

Faithful **+82%** over the range; pooled +25%. **Faithful's 25.3% now exceeds
AVID's 24.4%.**

For scale, `effect_rel` 0.0227 (faithful, shuffle) is already **at our own
framework's RT-1 plateau (~0.021, `5w72bo01`)** at step 5000 and still rising —
but the honest comparator is arm B, not that number, because the clean-room
differs from our framework in many ways at once.

## Related

- [[../../30_Knowledge/tech/avid-vs-ours-wan-action-conditioning]] — the five divergences
- [[exp-adapter-our-framework-avid-replication-robotarm]] — the DC-side counterpart
  (varies our code towards AVID; this ticket varies AVID's code towards Wan)
