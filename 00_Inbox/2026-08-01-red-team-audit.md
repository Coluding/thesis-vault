# Red-team audit — overnight examiner pass, 2026-08-01

> Adversarial audit of the D2/D3 campaign. Everything here is an **attack**, not a
> verdict. Every number is traced to a wandb run (pulled live via the API for this
> audit), a `file:line`, or a vault note. Where a claim could not be verified it is
> marked `_needs verification_` rather than guessed. Nothing in the codebase,
> configs, or other vault notes was modified; no jobs were launched.
>
> **Concurrency note:** `30_Knowledge/experiments/_index.md` and
> `20260801-wan-rt1-indistribution-plateau.md` were edited *during* this audit
> (the `⚠ PROVISIONAL` / quality / effect_rel-confound sections). Where this
> document overlaps, it reached the same conclusion independently — and in §3 it
> shows the newly-written confound section contains its own factual error.
>
> **Two projects named in the audit brief do not exist:** `coluding/dc-avid-parity`
> (real name `coluding/dc-acwm-robotarm-avid-parity`) and
> `coluding/dc-shortcut-fewstep-d3` (real name
> `coluding/dc-shortcut-actionfree-robotarm`). `api.projects('coluding')` returns 84.

---

## 0. The one-paragraph version

Three things break the spine. **(1)** The headline metric `action_effect_rel` is
provably **gain-sensitive rather than information-sensitive**, and the vault
contains its own proof; both of the campaign's two "fixes" are explicit *scale*
interventions on the action pathway, and on the only cell where the resulting
sensitivity was structurally probed, all three structure axes came back **at
chance**. **(2)** Pulled from wandb, several load-bearing numbers **do not check
out**: the DC arms ran to step 3500, not 3000, and at step 3500 the **untreated
control reaches 0.0456 — 1.55× the AVID reference it is supposed to be blind
against**; the "adapted loss 18% below control" is arm E @2500 compared against
arm 0 @**500**, and at matched steps the real gap is 2.5–8.3% (and arm E is
*worse* at step 500). **(3)** The ending's "35× / data sets the level" compares
runs differing in recipe, horizon and dataset simultaneously, the Wan × RT-1 run
**evaluates on its own training directory**, and the SkyReels × RT-1 run trains on
a silently length-filtered subset. Separately, the "68× curvature signature" is
entirely an **N=1 step-size bucket artifact** — at every N ≥ 2 the two backbones
are within ~1.5×. And the parallel audit
`[[2026-08-01-quality-vs-sensitivity-inverse]]` supplies the decisive
cross-sectional fact: across 8 runs, **`effect_rel` is anti-correlated with fit
quality**, so "data sets the level" and "worse fit inflates the metric" are
currently the *same axis*.

---

## 1. The three claims most likely to be attacked in a defence

### CLAIM 1 (FATAL) — "DC + `condition_center` follows actions at 0.106 = 3.6× the AVID reference"

Sources: `[[../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]]`,
storyline §8(ii), `writing-plan-2026-08.md`. **This is the spine** — the only cell
called "working", the one that must host planning.

**(a) The numbers are stale by one eval, and the corrected numbers undercut the
claim.** All three arms logged a **step-3500** eval that the note does not contain
(`coluding/dc-acwm-robotarm-avid-parity`, all three `crashed`, ~22–23 h, seed 0):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | **3500** |
|---|---|---|---|---|---|---|---|
| arm E `6oyu1inq` | 0.02572 | 0.06310 | 0.08193 | 0.09198 | 0.10644 | 0.09168 | **0.11479** |
| arm 0 `tr0uovs5` | 0.00326 | 0.00352 | 0.00370 | 0.00461 | 0.01203 | 0.02883 | **0.04564** |
| arm F `86kb01su` | 0.00319 | 0.00358 | 0.00395 | 0.00580 | 0.02557 | 0.03915 | **0.05049** |

AVID reference: **0.029475**.

> **At step 3500 the UNTREATED CONTROL is 1.55× the AVID reference, and arm F is
> 1.71×.** All three DC arms — including the one with no treatment at all —
> exceed the number the thesis calls the reference.

That single fact reframes everything: "3.6× the AVID reference" is not a property
of `condition_center`. It is a property of *training DC past step ~3000*. The
treatment effect, measured correctly (arm E ÷ arm 0 at a matched step), is
**2.5×**, not 3.5×; and the "3.6× the reference" headline is 3.9× at step 3500,
against a reference that the untreated arm also clears.

Corrections to the note's frontmatter, all verified from wandb history:
`armE_peak_effect_rel: 0.10644` → **0.11479** (0.10644 is the step-2500 value);
`armE_latest_effect_rel: 0.09168` → **0.11479**; `arm0_latest: 0.02883` →
**0.04564**; `armF_latest: 0.03915` → **0.05049**. The note's own hedge ("the
0.106 → 0.092 dip is one eval; do not read as a peak") resolves *against* the peak
reading — the series went back **up**.

**(b) The "18% lower adapted loss" is a step-mismatched comparison.**
`eval_base_loss` (which, note, is the **adapted** model's denoising loss —
`trainer.py:425` names the composed loss `base_loss`; there is no adapted-vs-frozen
loss logged for these arms):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|
| arm E | 0.043482 | 0.035660 | 0.035275 | 0.032020 | **0.035757** | 0.030480 | 0.025940 |
| arm 0 | **0.043278** | 0.036591 | 0.036512 | 0.033955 | 0.037914 | 0.031705 | 0.028302 |

The vault's `0.0357 vs 0.0433` is **arm E @2500 vs arm 0 @500**. Matched-step, arm
E is **0.5% worse** at step 500, then 2.5% / 3.4% / 5.7% / 5.7% / 3.9% / 8.3%
better. The real effect is **2.5–8.3%**, not 18%.

This matters beyond arithmetic. The storyline uses the 18% to argue *"the
action-following solution is better on the very objective the blind basin
optimises"* — the load-bearing sentence for the economics story
(`20260731-why-wan-copies-the-base-decomposed` §3 repeats it verbatim). At 2.5–8.3%
that argument is much weaker, and at step 500 it runs backwards.

**(c) On the vault's own normalised metric, arm E is worse than the reference.**

| | arm E | AVID `rqp4s3gp` |
|---|---|---|
| `effect_rel` | 0.10644 (@2500) | 0.029475 |
| `effect ÷ adapter_rel_contribution` | **0.311** | **0.422** |
| ⇒ implied `adapter_rel_contribution` | **0.342** | 0.069838 |

Arm E's adapter moves the composed output **4.9× further from the frozen base**
and gets 3.6× the sensitivity. **Per unit of adapter work it is 0.74× the
reference.** The probe module's own docstring says exactly why that is the
discriminating quantity (`evaluation/action_sensitivity.py:34-39`).

**(d) `condition_center` is a gain knob and `effect_rel` is a gain metric.**
It is `nn.BatchNorm1d(dim, affine=False)` on the projected conditioning
(`backbones/dynamicrafter/modules/networks/openaimodel3d.py:492-495`, applied
`:872-884`), whose in-code justification is *"rescales to unit variance so the
surviving signal is not tiny."* On an embedding measured at `realised/RMS = 0.0050`
(`20260730-dc-parity-arms-null-action-embedding-pedestal`), dividing by the batch
σ amplifies the action-dependent component by ~two orders of magnitude.
`effect_rel = ‖pred(a)−pred(a')‖/‖pred(a)‖` on the **composed** output
(`action_sensitivity.py:368`) is monotone in that gain. **A rise after a gain
intervention is the mechanically expected outcome.**

**(e) The structure probes that would settle it have never been run on this cell.**
Vault-wide grep: `steer`, `rollout_swap`, `--action-analysis` appear **only** for
Wan × ACWM. Where they were run, after the analogous gain fix
(`20260731-wan-action-signal-is-a-global-bag`): steering cos **−0.002/−0.003**,
temporal alignment **0.25 vs 0.28 chance**, spatial concentration **11.4% vs 10%
chance** — a raised `effect_rel` that bought **zero structure**.

**(f) The spine cell has no quality evidence of any kind.** Verified twice: no
`diffusion_dc_acwm_robotarm*.yaml` sets `quality_metrics`, `quality_dist_metrics`
or any `baseline_eval_*` key; and enumerating every logged key in `6oyu1inq`,
`tr0uovs5`, `86kb01su` (38/38/37 keys) returns **no FID, FVD, LPIPS, PSNR, MSE or
SSIM**. Meanwhile every cell where quality *was* measured against the frozen base
is a perceptual regression (§Claim 2). **The examiner will ask whether the working
cell is better than the frozen base it started from, and the vault cannot answer.**

**(g) The reference number's provenance is weak.** `rqp4s3gp` logs **no
action-sensitivity metric at all** — 0.029475 comes from an offline probe script
whose output is not archived in wandb or the vault. The run is `finished` at
`trainer/global_step` **8079** against `max_steps: 15000`, while the note's
frontmatter says `status: running` "toward max_steps 15000". Its probe also ran at
`target_height: 384` while the train launcher forces **320** (verified: the
override `data.params.target_height=$HEIGHT` with `HEIGHT=320` is in
`submit_train_avid_acwm_robotarm.sh:123,136`; `submit_probe_acwm_robotarm_action.sh:84-90`
carries **no** height override) — so the reference was measured at a resolution
the training launcher's own comment calls *"off-distribution for the frozen base"*.

**Cheapest defence, in order:**
1. **(~1 GPU-h, no training)** Structure triad (`--action-analysis`) +
   `--rollout-action-swap` on the retained arm-E step-3500 checkpoint, with arm 0
   as the matched control. The probe suite already exists. **This decides whether
   the thesis has a positive result.**
2. **(~1 GPU-h)** Re-eval arm E and arm 0 with `quality_metrics`,
   `quality_dist_metrics`, `baseline_eval_quality: true`.
3. **(free)** Correct the four stale numbers and the 18% claim; report
   `eval_action_effect_vs_adapter` alongside `effect_rel` everywhere.
4. **(~1 GPU-h)** Re-probe the AVID reference at 320 with a genuine cross-episode
   shuffle (§5-N), and re-probe arm E on AVID's timestep grid at 120 samples.
   Until then, no "× the reference" multiplier is defensible.

---

### CLAIM 2 (FATAL to the ending) — "Same recipe, only the dataset changed: 35× on SkyReels; the two-factor law"

Sources: storyline §9, `_index.md` rows 27-28, `20260801-…-plateau`.

**(a) The recipe did not stay the same.** Verified from configs:

| | SkyReels × ACWM `8zjjn7wl` (the 0.0013) | SkyReels × RT-1 `sgdftf6b` (the 0.0450) |
|---|---|---|
| `action_token_norm` | **absent (off)** | **true** |
| `condition_on_base_outputs` | **true** (`…xattn_acwm_robotarm.yaml:52`) | **false** (`…rt1_tokennorm_nobase.yaml:64`) |
| dataset | ACWM Robot Arm | RT-1 |
| run end | killed @ ~897 steps | crashed @ 21.9 h / step 4399 |
| the note's own diagnosis of the low value | *"gate saturated to the cap → adapter starved"* (`20260728-…`) | — |

Three variables move together, and one of them is the intervention the campaign
separately credits with **6–10×** on Wan. Worse, the ACWM leg's own note blames
gate saturation — a starved adapter — not the data. (Horizon *is* matched for
SkyReels: `temporal_length: 97` both sides. Credit where due.)

**(b) The Wan leg has a horizon confound as well.**
`diffusion_wan22_action_rt1_tokennorm_nobase.yaml` sets `temporal_length: 17`
(5 latent frames); `diffusion_wan22_avid_xattn_tokennorm_nobase_acwm_robotarm.yaml`
sets **97** (25 latent frames). `effect_rel` is a ratio over the whole predicted
tensor; a 5.7× shorter horizon removes the far future — which no action explains —
from the denominator, raising the ratio mechanically.

**(c) The Wan × RT-1 run evaluates on its training data.** From `5w72bo01`'s
archived args, the **eval data directory equals the train data directory**
(`/scratch-shared/rt1/train`), whereas `ncztxyyo` uses a separate `ind_test`. So
every RT-1 Wan number — `effect_rel` *and* the FID/FVD/LPIPS/PSNR/MSE table — is
measured **in-sample**. This is not a nuance: "in-distribution data lifts the
plateau" is the note's title, and the run has no held-out set.

**(d) Quality: the RT-1 cells are a net perceptual regression.** Confirmed from
wandb (final evals):

| | Wan `5w72bo01` adapted | base | SkyReels `sgdftf6b` adapted | base |
|---|---|---|---|---|
| FID ↓ | **132.51** | 113.20 | **161.44** | 145.07 |
| FVD-i3d ↓ | 1575.6 | 1616.7 | **2514.8** | 2256.7 |
| LPIPS ↓ | **0.3823** | 0.3650 | **0.4977** | 0.4610 |
| SSIM ↑ | 0.6587 | 0.6485 | **0.4616** | 0.4830 |
| PSNR ↑ | 17.533 | 16.529 | 12.697 | 11.762 |
| MSE ↓ | 0.01765 | 0.02224 | 0.05375 | 0.06664 |

(bold = adapted worse). Two corrections to the vault's table: Wan **wins** FVD
(1575.6 vs 1616.7) and **wins** SSIM (0.6587 vs 0.6485 — the note says base SSIM
was "not logged"; it is, at 0.648497). So Wan's split is 2 losses (FID, LPIPS) vs
4 wins, not "every perceptual metric". SkyReels loses 4 of 6 as written.

**And the generalisation "the adapter degrades the base" is false — it reverses on
ACWM.** Per `[[2026-08-01-quality-vs-sensitivity-inverse]]`, on ACWM Robot Arm the
Wan token-norm arm `52o3uxz8` beats the frozen base on **all six** metrics
(FID 57.4 vs 90.1, FVD **406 vs 1118**, LPIPS 0.192 vs 0.239), as do `8zjjn7wl`,
`7bmzwv6u` and `vy9tcuco`. So the honest statement is *dataset-conditional*: the
adapter is a strong domain/appearance corrector on ACWM and a net perceptual
regression on RT-1. **Which is itself the more interesting D2 result** — a 2.75×
FVD win carrying action structure at chance is, by elimination, a domain
correction, and that is a positive, defensible finding the deliverable does not
currently claim.

**Cheapest defence:** one arm — **SkyReels × ACWM Robot Arm at the current
recipe** — converts a 3-variable comparison into 1. Then either match the Wan
horizons or drop the Wan leg. And re-run the RT-1 eval on a held-out split.

---

### CLAIM 3 — "`effect_rel` measures action information" (Threat A, assessed honestly)

**The concern is real; the evidence currently written for it is wrong; the
strongest evidence for it is elsewhere in the vault and unused.**

The frozen-base null does **not** address it — verified: `base_null_violation` is
computed from `denoise_base_only` **loss** deltas
(`action_sensitivity.py:372-374, 384-387`). It asserts the frozen base is
action-invariant, i.e. it is a harness-leak check. It says nothing about fit
quality. **The vault's new text is right about that.**

But the *supporting* evidence does not hold, and now fails twice over:

- **"Both RT-1 runs peak at the very first eval, when the fit is worst"** is
  **false for Wan, in two ways.** (i) The series in the note's own frontmatter
  rises for five evals and peaks at the **seventh**. (ii) The note lists 18 evals;
  wandb has **34**, and the true maximum is **0.047692 at step 13200** — the
  second-to-last eval, at the *best* fit, not the worst. Full history:
  `0.02336 0.02640 0.02744 0.02848 0.02986 0.02381 0.03315 0.02325 0.02487
  0.02011 0.02108 0.01840 0.01845 0.01676 0.01933 0.02225 0.02000 0.02545
  0.02272 0.02189 0.01978 0.04056 0.01909 0.02034 0.02088 0.01892 0.02362
  0.01735 0.02092 0.02492 0.02107 0.03195 **0.04769** 0.02494` — flat and noisy,
  **no trend**, min 0.01676, max 0.04769.
- DC arm E rises monotonically 0.0257 → 0.1148 over 3500 steps of falling loss.
  Again the opposite of the instability signature.

Only SkyReels `sgdftf6b` actually shows monotone decay from its first eval
(0.04501 → 0.03072 → 0.02840 → 0.01919 → … → 0.01732).

**⚠ But there is much stronger evidence for the instability reading than the
timing argument, and it arrived from a parallel audit tonight.**
`[[2026-08-01-quality-vs-sensitivity-inverse]]` tabulates 8 ACWM/RT-1 runs and
finds `effect_rel` **anti-correlated with fit quality across runs**: the four runs
that beat the frozen base on **all six** quality metrics (`8zjjn7wl` 0.0013,
`7bmzwv6u` 0.0022, `52o3uxz8` 0.0062, `vy9tcuco` 0.0077) sit at the bottom of the
`effect_rel` range, while the two that **lose four of six** (`sgdftf6b` 0.0173,
`gi44pv5k` 0.0211) sit at the top.

That is a **cross-sectional** version of the instability prediction — far harder to
dismiss than a within-run trajectory shape, because it does not depend on when the
peak occurs. It is also the single most damaging fact in this audit for the
two-factor law: **the "data sets the level" axis and the "worse fit ⇒ higher
`effect_rel`" axis are the same axis in this table**, since the RT-1 cells are
exactly the badly-fit ones. Take the concern as *upgraded*, not merely open.

Two honest caveats on it: quality and `effect_rel` also co-vary with dataset and
recipe here, so the correlation is not itself clean; and Wan × RT-1 wins FVD/SSIM
(§Claim 2d), so "quality" is metric-dependent. It still needs the gain-controlled
read below to settle.

**A third strand nobody has connected.**
`20260731-wan-action-trace-value-pathway-drowns` §"Early causal readout": raising
`gate_cap` 0.5 → 0.9 (sending the gate 0.5 → 0.899, i.e. adapter weight 50% → 10%)
dropped `effect_rel` from **0.0056** (`ncztxyyo`) to **0.00117** (GATEFIX) — a
**4.8× fall from a pure composition-weight change with the action pathway
untouched.** That is a direct in-vault demonstration that `effect_rel` tracks
adapter gain. The note mentions it in passing; the storyline does not.

**A second, free indicator nobody has read.** Both RT-1 runs log
`eval_action_effect_rel_zero` **consistently higher than the shuffle variant**
(Wan 0.0292–0.0609 vs 0.0168–0.0477; SkyReels 0.0795 → 0.0469 vs 0.0450 → 0.0173).
If the adapter used action *identity*, swapping to a different valid action should
move the prediction comparably to blanking it. `zero ≫ shuffle` says the model
responds to action **presence/magnitude**, not identity — the bag-of-actions
signature, visible in already-logged data on both RT-1 cells.

**How much depends on `effect_rel` being an information measure?** Essentially all
of D2. Every headline — 0.106, 3.6×, 6–10×, 35×, 2.3×, the two-factor law,
"blindness is our implementation" — is a comparison of `effect_rel` across
conditions that also differ in adapter gain.

**The minimal separating experiment is not a training run.** It is a
**gain-controlled read of existing checkpoints**: report `effect_rel` at matched
`adapter_rel_contribution` (or just report `eval_action_effect_vs_adapter`, already
computed at `trainer.py:640`), then run the structure triad. Sensitivity that
survives gain normalisation **and** is directional / temporally aligned /
spatially concentrated is information. Sensitivity that does neither is gain.
Cost: hours, no training. **Highest-value experiment in this audit.**

---

## 2. Confound register

**fatal** = cannot be written as stated · **weakens** = write with an explicit
caveat · **cosmetic** = fix the wording.

| # | Claim | Evidence cited | Confound | Sev | Fix |
|---|---|---|---|---|---|
| C1 | arm E = 3.6× the AVID reference | 0.10644 vs 0.029475 | At the step-3500 eval the **untreated control** reads 0.04564 = **1.55× the reference**, arm F 1.71×. The multiplier is a property of training length, not the treatment. Treatment effect at matched step = **2.5×** | **fatal** | Quote arm E ÷ arm 0 at a matched step; drop "× the reference" |
| C2 | arm E's adapted loss is 18% below control ⇒ "the action-following solution is better on the objective" | 0.0357 vs 0.0433 | **Step-mismatched**: E@2500 vs 0@**500**. Matched-step gap is 2.5–8.3%, and E is 0.5% *worse* at step 500 | **fatal** | Restate as 2.5–8.3%; the economics argument weakens accordingly |
| C3 | arm E's sensitivity is action *use* | `effect_rel` only | No steering / temporal / spatial / rollout-swap probe has **ever** run on any DC cell. The one cell probed (Wan) was **at chance on all three** | **fatal** | Run the existing suite on the arm-E ckpt |
| C4 | "DC + adapter **works**" (spine link 1) | arm E `effect_rel` + loss | **Zero quality metrics logged** (verified in configs *and* by enumerating all 38 wandb keys). Every measured cell in the campaign is a perceptual regression vs base | **fatal** | Re-eval both ckpts with quality metrics |
| C5 | 35× "data sets the level" (SkyReels) | 0.0450 vs 0.0013 | Three variables move together: dataset, `action_token_norm` off→on, `condition_on_base_outputs` true→false. The 0.0013 run's own note blames **gate saturation** | **fatal** | One SkyReels × ACWM run at the current recipe |
| C6 | 2.3× "data sets the level" (Wan) | RT-1 ~0.025 vs ACWM ~0.011 | `temporal_length` **17 vs 97**. Plus 8 further config deltas incl. **frozen vs live gate** and presence/absence of token-norm (§B7) | **fatal** | Match horizons and recipe on one axis |
| C7a | The two-factor law: "data sets the level" | RT-1 > ACWM `effect_rel` | Across 8 runs, `effect_rel` is **anti-correlated with fit quality** (`[[2026-08-01-quality-vs-sensitivity-inverse]]`), and the RT-1 cells are exactly the badly-fit ones. "Data sets the level" and "worse fit ⇒ higher `effect_rel`" are the **same axis** in the available data | **fatal** | Gain-controlled read + structure triad on an RT-1 ckpt |
| C7 | Wan × RT-1 is an "in-distribution" test | `5w72bo01` | **Eval directory = train directory.** Every RT-1 Wan number, sensitivity and quality alike, is in-sample | **fatal** | Re-eval on a held-out split |
| C8 | "91% of the AVID RT-1 reference" | 0.0450 vs 0.0495 | Cross-implementation (§5-N); AVID's RT-1 path applies octo-style per-dim action standardisation, ours passes raw (`data/translators/rt1.py:11-15`, conceded in `20260729-…` §Next); and it is a first-eval peak vs a step-15679 endpoint | **fatal** | Drop the % |
| C9 | AVID ref vs ours: 5.3×/8.7×/22.7× | `20260730-…` | (i) not training-matched — flagged; (ii) **unflagged:** the Wan 0.0056 is from `ncztxyyo`, gate frozen at 0.5 (50% adapter weight); the same config live-gated reads 0.00117 | weakens | State the gate condition |
| C10 | "`condition_center` = ~6× faster escape" | one trajectory per arm | Escape is a **late transition**; timing has far higher seed variance than a level. The only seed pair in the project (`n3dbgq4q` s0 vs `l2jcz9nx` s1) both stopped at ~1020 steps, giving a spread only in the *blind* regime (0.003399 vs 0.003785, ~11%). **No seed replicate exists past step ~1024 for any arm** | **fatal for "6×"** | Second seed of arm 0 and arm E |
| C11 | "arm F escapes 2× faster ⇒ architecture reopened" | 0.0392 vs 0.0288 | Arm F changes **two** keys vs arm 0: `use_native_action_embed: true` **and** `action_time_combine: concat` (verified in the run YAMLs) | weakens | Say "encoder + combine" |
| C12 | "DC consistency loss ~68× Wan's" = curvature | `t4bp8nki` / `pzmc2orq` | Already flagged: frozen Wan gate + different target rules. **Three more, unflagged:** (i) the 68× is **entirely an N=1 bucket artifact** — see §5-Q; (ii) it compares Wan@800 vs DC@**500** (each run has only 2 eval points); (iii) at DC's other eval point (step 0) its consistency loss is **0.0085, lower than Wan's** | **fatal** | Re-run both, one commit, live gates, one target, per-N reporting |
| C13 | Backbone differences attributed to flow-vs-diffusion / base strength | DC works, Wan/SkyReels collapse | Ticketed as objective × tokenizer. **It is ≥8 axes** — §5-H | **fatal for causal attribution** | The ticketed 4th cell closes exactly one axis; say so |
| C14 | "actions are worth 0.45% of the loss" as a property of the objective | σ-sweep on `ncztxyyo` | One checkpoint, one dataset, one training stage, **on the frozen-gate run**. Quoted in the storyline as a general law of teacher-forced denoising, while the campaign's own RT-1 result argues the share is dataset-dependent | weakens | Measure on RT-1 and DC |
| C15 | Wan "value pathway drowns" mechanism | probe of `ncztxyyo` @1000 | Frozen-gate checkpoint. Interior depths are gate-independent (defensible, flagged) — but the chain-closing "composed drel 0.0096 ≈ effect_rel" uses the gate-dependent composed value | cosmetic | Re-probe on GATEFIX |
| C16 | All Wan/SkyReels `effect_rel` points | configs | `action_sensitivity_draws: 2`, `batches: 2` ⇒ **4 samples per eval**, 2 donor pairings. DC uses 12. AVID's probe uses **120** | weakens | Raise draws/batches |
| C17 | "peak 0.0332" / "opens at 0.0450" | max over the series | Upward-biased max-statistics over noisy 4-sample means; and Wan's true max is 0.04769 @13200, not 0.0332 | weakens | Quote mean ± the bootstrap CI the probe already computes (`action_sensitivity.py:228`) |
| C18 | SkyReels × RT-1 is "RT-1" | `temporal_length: 97` | `data/dataset.py:71` **silently filters** `ep.length >= span`; RT-1 episodes are ~22–115 frames. A 97-frame window keeps only the long tail — the `423pjv8y` memorisation failure mode | **fatal until counted** | Print `len(ds.episodes)`. One line |
| C19 | Ours vs AVID `effect_rel` are the same measurement | `20260730-…` §probe header | Same algebra, **different perturbation, sampling, resolution and data pipeline** — §5-N | **fatal** | Re-probe both under one protocol |
| C20 | `effect ÷ adapter` as a normalised metric | `trainer.py:640` | Numerator is frame-masked (`action_sensitivity.py:83-96`); denominator is **not** (`trainer.py:433-437`) | cosmetic | Mask both |
| C21 | "our probe reproduces AVID's metric" | vault | **Three** `effect_rel` implementations exist in-repo — `evaluation/action_sensitivity.py`, `scripts/eval_action_sensitivity.py` (adds timestep stratification), `scripts/generate_wan22_i2v_compare.py:489-546` (σ-stratified, Wan-only) — plus AVID's. Four sampling schemes, one metric name | weakens | Record which path produced each number |
| C22 | Provenance of every in-house number | wandb | **`wandb.config` contains only `{"experiment": …}`** for every in-house run; all hyperparameters live in an uploaded YAML artifact + `metadata.args`. Config provenance is brittle for a thesis audit trail. And **8 of 10 audited runs did not complete** (7 crashed, 1 killed; the 2 `finished` AVID runs both stopped short of `max_steps`) | weakens | Log the resolved config to `wandb.config` going forward |
| C23 | AVID reference numbers (0.029475, 0.0495, 0.422, 0.0698) | `rqp4s3gp`, `93qrvr5v` | **Neither run logs any action-sensitivity metric.** The numbers come from an offline probe whose output is archived nowhere. `rqp4s3gp`'s frontmatter also says `status: running` toward 15000 while wandb shows `finished` at global_step **8079** | weakens | Commit the probe JSON to the vault |
| C24 | AVID reference measured in-distribution | `rqp4s3gp` | **Trained at height 320** (`submit_train_avid_acwm_robotarm.sh:123,136` forces it, with a comment that 384 "puts the frozen base off-distribution") but **probed at 384** (`submit_probe_acwm_robotarm_action.sh` has no override) | weakens | Re-probe at 320 |
| C25 | `ncztxyyo` is a `gate_cap: 0.9` datapoint | run name `acwm-robotarm-wan-cap09-shift5` | **The run name contradicts its own config**: the archived YAML says `gate_cap: 0.5`, and `eval_adapter_gate_std` is 3.7e-07, 0, 0. Anyone reading the run name gets it backwards | cosmetic (but a live trap) | Rename or annotate |

---

## 3. Numbers that do not check out

1. **`armE_peak_effect_rel: 0.10644` / `armE_latest: 0.09168` / `arm0_latest:
   0.02883` / `armF_latest: 0.03915`** — all four are step-≤3000 values; a
   step-3500 eval exists in all three runs. Correct values **0.11479 / 0.11479 /
   0.04564 / 0.05049**. Consequence: the untreated control clears the AVID
   reference, and "~3.5× higher level" becomes **2.5×**.

2. **"adapted loss ~18% below the control's (0.0357 vs 0.0433)."** Arm E @2500 vs
   arm 0 @**500**. Matched-step: 2.5–8.3%, and arm E is 0.5% *worse* at step 500.
   This is the most consequential arithmetic error found, because the storyline
   and `20260731-why-wan-copies-the-base-decomposed` §3 both build an argument on
   it.

3. **"Both RT-1 runs peak at the very first eval, when the fit is worst"**
   (`20260801-…:109-110`, written today). False for Wan twice over: the listed
   series peaks at eval 7, and the *full* 34-eval history peaks at **0.04769 @
   step 13200**, near the end.

4. **"peak 0.0332"** for Wan × RT-1. True max **0.047692**. The note lists 18 of
   34 evals.

5. **"3.6× the AVID reference."** Arithmetically right on `effect_rel`
   (0.10644/0.029475 = 3.61); **inverted** on the vault's own normalised metric
   (0.311/0.422 = 0.74); and at step 3500 the *untreated* arm is also above the
   reference.

6. **"35×, same recipe, only the dataset changed."** Recipe changed on two axes
   (C5). The settled-vs-settled figure is 0.0173/0.0013 = **13×**, still across
   different recipes.

7. **"2.3× the ACWM plateau"** at 5.7× shorter horizon, with 8 further config
   deltas including a frozen-vs-live gate (C6).

8. **"91% of the AVID reference."** First-eval peak vs a step-15679 endpoint,
   across implementations, across action normalisation (C8).

9. **"6–10× the control" for `action_token_norm`.** Against GATEFIX (0.00117) it
   is 9.7×; against the pre-fix run on the same config (0.0056) it is **2.0×**.
   The multiplier is a function of which control you pick. Flagged in
   `20260731-wan-action-trace-…:112-114`; not in the storyline, `_index.md`, or
   the writing plan.

10. **"the adapter's function is ~100× more sensitive to base_pred than to
    actions" (1.13 vs 0.0087).** The 1.13 is from **blanking** the oracle — an
    input the adapter has never seen — while 0.0087 is an in-distribution action
    swap. The fair comparator is the note's own `base_shuffle` row (0.84), giving
    ~96×. The headline survives; quote `base_shuffle`.

11. **"actions are worth 0.45% of the loss"** quoted as a property of "the
    teacher-forced denoising objective". Source: one σ-sweep, one frozen-gate Wan
    checkpoint, step 1000, ACWM only.

12. **`20260731-wan-tokennorm-nobase-training-results` frontmatter (`:23`)** still
    asserts *"NOBASE STOPS the erosion … pre-registered prediction confirmed"*
    while its own body (`:47-51`) corrects this to *"slows, not stops"*. The
    `_index.md` row (`:31`) carries the superseded version. If a deck or draft is
    generated from the index, the retracted claim ships.

13. **`_index.md` SIMPLE-arm row** — headline numbers (~0.0025, band 0.002–0.007,
    "worth 3–4×") with **no wandb id, no ckpt, no config**, and the note it points
    to does not contain them in its body. `_needs verification_`, hard rule 8.

14. **Storyline §9 table** puts "SkyReels ACWM 0.0013" and "Wan ACWM ~0.011" in one
    column as if same-recipe. 0.0013 is the 07-28 old recipe; 0.011 is the 07-31
    token-norm + nobase recipe. Columns are datasets; rows silently mix recipes.

15. **`20260729-shortcut-wan-vs-dc-curvature-signature`** is listed in `_index.md:36`
    with the "68×" headline and **no ⚠ in the ledger row**, though the note body
    carries a large confound warning and the writing plan lists it as unusable. The
    ledger row is what gets reused.

16. **`rqp4s3gp` frontmatter** says `status: running`, "run still training toward
    max_steps 15000". wandb: `finished`, `trainer/global_step` **8079**, 8.43 h.

17. **`20260730-…:37` "traj_len 16 × stride 4"** for AVID. The run's archived
    config exposes `sample_stride: 4`, `clip_stride: 16`, `action_aggregation:
    "sum"`; the **current on-disk** datamodule exposes `frame_stride: 1`,
    `stride: 8` and states *"Consecutive frames (frame_stride=1 parity): … NO
    stride-summing"* (`acwm.py:20,169,169-186`). **The datamodule was rewritten
    after the run.** The probe launcher loads the *current* config, so re-running
    the probe today would measure a different data pipeline than the one that
    produced 0.029475. `_needs verification_` — and see §5-N.

---

## 4. Unflagged blast radius of the `gate_cap` bug

I re-ran the audit programmatically with a YAML parser (so comments cannot poison
it) across all 20 configs that set `gate_cap`: **every one is now `gate_bias: 0.0`
/ `gate_cap: 0.9`.** The fix landed and the ticket's table is correct for the repo
as it stands.

**But that also means the checkout can no longer identify affected runs** — the
configs were edited in place in an uncommitted tree. Affected-run identification
must come from **wandb-archived run YAMLs**. I did that for the runs in scope.

**Already flagged:** `pzmc2orq`, `ncztxyyo`, the curvature note, the six configs.

**NOT flagged:**

| # | Artifact | Why affected | Missing flag |
|---|---|---|---|
| B1 | `20260728-acwm-robotarm-matrix-action-blind` §"Three distinct starvation signatures" — Wan's row reads *"gate stuck at init"*, `gate_mean 0.50 (std 0.00)` | That is **the bug**, not a starvation mechanism. Confirmed from wandb: `eval_adapter_gate_std` = 3.7e-07, 0, 0 | No ⚠ on the note at all; "three distinct starvation signatures" is the `_index.md:38` headline |
| B2 | The **Wan 0.0056** figure | Comparator in the 5.3×/8.7×/22.7× grid, in `20260729-…`, in storyline §8(i) and §9's table. Measured at 50% adapter weight with a frozen gate; live-gated the same config reads **0.00117** | Nowhere. The "blindness is our implementation" argument rests on a number that moves 4.8× with the bug. (Also: `ncztxyyo`'s `effect_rel` is **declining** across its three evals — 0.00591 → 0.00558 → 0.00490 — which the 07-28 note reports only as "flat") |
| B3 | The **"6–10×"** token-norm headline | Its denominator exists only because the bug was fixed | Flagged only in `20260731-wan-action-trace-…:112-114` |
| B4 | `configs/wan22/diffusion_wan22_action_rt1.yaml` — listed FROZEN | Verified clear for `5w72bo01` (**archived `gate_cap: 0.9`**, `gate_std` 1.7e-04–0.071, live). Any *earlier* run from that config would be affected; `5w72bo01` is currently **the only run in `Wan2.2-action-rt1`**, so the exposure is nil | Record that it was checked |
| B5 | The two **Push Cube** configs | Found only on the ticket's second audit pass. Which runs used them, and whether any vault claim cites them, is recorded nowhere. Projects `Wan2.2-avid-xattn-acwm-pushblock` and `dc-acwm-pushblock` exist | Enumerate their runs' archived `gate_cap` |
| B6 | `configs/wan22/diffusion_wan22_shortcut_openvid.yaml` | FROZEN per the ticket, and there is a **new untracked ticket** `exp-shortcut-flow-vs-diffusion-openvid.md` planning work on this axis | Put the warning in the ticket |
| B7 | **The gate_cap fix is itself an unlogged intervention in the 07-31 campaign** | Every 07-28 Wan number is at ~50% adapter weight; every 07-31 Wan number is at ~10% (gate pinned to the 0.9 cap: `5w72bo01` gate_mean 0.870–0.900, `sgdftf6b` 0.880–0.900). Any trend line drawn across 07-28 → 07-31 crosses a composition-weight discontinuity | Nowhere |
| B8 | `ncztxyyo`'s **run name says `cap09`** | Its archived config says `gate_cap: 0.5`. A reader taking the run name at face value concludes the opposite of the truth | Rename or annotate the run |

**Not affected (verified twice):** **no DynamiCrafter config sets `gate_cap` at
all** — confirmed by the YAML audit and by the archived run configs for
`6oyu1inq` / `tr0uovs5` / `86kb01su`. The DC parity arms, arm E/F, and `t4bp8nki`
are clean.

---

## 5. New threats nobody has named

### F (fatal) — The two headline fixes are gain interventions measured by a gain metric

Not a confound between conditions — a **circularity in the measurement**.

- `condition_center` = `BatchNorm1d(affine=False)`, justified in code as
  *"rescales to unit variance so the surviving signal is not tiny"*
  (`openaimodel3d.py:487-495`).
- `action_token_norm` = `nn.LayerNorm(dim)` on action tokens, verified pre-launch
  as raising *"token RMS 0.004 → 0.757"* (`20260731-wan-action-trace-…:91-94`);
  applied at `backbones/wan/modules/action_model.py:284-285`.

Both are, by design, **amplitude fixes on the action pathway**, and `effect_rel`
is monotone in that amplitude. **Reporting that an amplitude fix raised an
amplitude-sensitive metric is not a finding.** The vault contains the control
proving the metric is amplitude-sensitive (gate 0.5→0.899 moved `effect_rel` 4.8×,
§Claim 3) and does not use it.

**And the campaign already knows what happens when you check.** On Wan, after
token-norm produced its `effect_rel` gain, the structure triad came back **at
chance on all three axes**. The storyline nonetheless presents "scale calibration
at the injection interface" as a *positive* mechanistic contribution (§8(ii)). It
is currently better supported as a **negative** one: *scale calibration raises the
sensitivity metric and buys no structure.* That framing is defensible; the current
one is not.

### G (fatal until counted — a one-line check) — SkyReels × RT-1 may be training on a tiny length-filtered subset

`data/dataset.py:71`: `[ep for ep in translator.list_episodes() if ep.length >= span]`
— a **silent** filter. `diffusion_skyreels_rt1_tokennorm_nobase.yaml` sets
`temporal_length: 97`, `frame_stride: 1`; `_clip_meta` additionally requires
`action_span = length * stride` frames (`translators/acwm_phys.py:206-211`). So an
episode needs **≥97 frames**, and RT-1 episodes are documented in the translator's
own docstring as **~22–115 frames**.

If most RT-1 episodes are shorter, this run trained on a small, biased,
long-episode subset — the exact mechanism that made `423pjv8y` unusable
(`max_clips: 64`, memorisation). A memorising model would show precisely the
observed signature: a high opening `effect_rel` decaying monotonically —
which is what `sgdftf6b` does, and it is the *only* run in the campaign that does.

**Cheapest high-severity check in the audit.** Instantiate the dataset for that
config and print `len(ds.episodes)` / `len(ds)`. `_needs verification_` — the RT-1
`metadata.pt` is on the cluster.

(The Wan RT-1 config sidesteps this with `temporal_length: 17` — which is
presumably *why* the two RT-1 runs have different horizons, and is therefore the
root of C6 as well.)

### H (fatal for any causal attribution) — The DC↔Wan comparison is ≥8 confounds, not 2

The open ticket frames it as objective × tokenizer. Verified from the two archived
run configs, the working cell and the collapsing cell differ on **at least**:

| axis | DC arm 0/E (works) | Wan tokennorm-nobase (collapses) |
|---|---|---|
| objective | diffusion (velocity) | flow matching |
| VAE temporal ratio | 1 (per-frame 2D) | 4 (3D causal) |
| clip length | 16 frames | 97 frames |
| `frame_stride` | **4** (actions **summed** over the stride, `acwm_phys.py:214-215`) | **1** (raw per-frame deltas) |
| resolution | 384×512 | ~864×640 (`max_area 589824`) |
| batch | 24 | 12 |
| adapter | 11M UNet output adapter | 34M DiT-clone, cross-attention |
| composition | `avid_mask_mix` | `mask_mix` |
| conditioning shape | `act` [B,T,7] into ResBlock `emb` | per-frame `action_seq` tokens into cross-attn |
| gate | **no cap**, live | `gate_cap: 0.9`, pinned to the cap |
| `sigma_shift` | — | 5.0 |
| `pretrain_steps` | 0 | 200 |
| probe samples/eval | 12 | **4** |

The `frame_stride` line deserves its own note: DC conditions on **net 4-frame
displacements** while Wan conditions on **raw per-frame deltas** — a different
conditioning signal with a different SNR, on the same dataset, and a *coarser*
action representation than the "temporal smearing" hypothesis would predict is
harmful.

**Consequence:** no statement of the form "X fails because of the objective / base
strength / the tokenizer" is currently supported. The ticketed 4th cell closes
exactly **one** of these axes; say so rather than presenting it as the resolution.

### I (weakens) — `condition_center` has a train/eval semantics change and a rollout hazard the planning demo will hit

`BatchNorm1d` uses **batch statistics in train mode, running statistics in eval
mode**. The probe calls `model.eval()` (`action_sensitivity.py:314`), so every
reported arm-E number is under running stats while every gradient step was under
batch stats. During arm E's training the pedestal is itself *growing* (measured
×106 over 600 steps), so the running statistics are a lagging EMA of a
non-stationary quantity for much of training.

**More importantly for the spine's next link:** `_center_condition` bails out only
when `self.training and cond_emb.shape[0] < 2` (`openaimodel3d.py:882-883`). At
inference the normalisation always applies, using running statistics estimated on
the *training* action distribution. A planner scoring **counterfactual /
out-of-distribution action sequences** — the entire point of planning — will have
them whitened by training-distribution statistics, compressing exactly the
differences the planner needs. **The planning demo the storyline calls
spine-critical is being built on a conditioning pathway that whitens away action
variance at inference.** Worth knowing before spending the GPU hours.

### J (weakens) — RT-1's per-clip caption is an uncontrolled second conditioning channel

`diffusion_wan22_action_rt1_tokennorm_nobase.yaml` sets
`text_prompts_file: configs/prompts/rt1_captions.yaml`, and `translators/rt1.py`
surfaces each episode's `natural_language_instruction` as the **per-clip** caption.
ACWM uses one shared prompt.

So on RT-1 the model receives a task-identifying text stream *and* actions. The
shuffle probe permutes only the actions, producing a **(caption, action) mismatch**
that is strongly out-of-distribution and cannot occur on ACWM. OOD perturbations
move predictions more, independent of action understanding. This is a **third**
mechanism — alongside horizon (C6) and recipe (C5) — by which RT-1 inflates
`effect_rel` without a dataset-economics story.

Cheap control: probe with the caption shuffled *together with* the actions
(consistent counterfactual), and separately with the caption nulled.

*(Related config-hygiene flag: `sgdftf6b`'s `default_prompt` describes the ACWM
robot arm and its `conditioning.input_dim` comment says "ACWM-Phys robot_arm",
while the data is RT-1. The same stale comment is in the Wan RT-1 config. Verify
that the per-clip caption table actually overrides the default prompt in the
SkyReels path.)*

### K (weakens) — the `roll` variant would have answered the timing question for free

`action_sensitivity.py:127-129` implements a `roll` variant — *"this clip's own
actions, rolled by half the sequence. Identical marginal statistics, wrong
temporal alignment: separates 'uses action magnitude' from 'uses action timing'."*
**Every config in the campaign sets `action_sensitivity_variants: [shuffle, zero]`.**
The variant that discriminates information from magnitude has been implemented the
whole time and never enabled. A `roll`≈`shuffle` result with a large `shuffle`
effect is precisely the global-bag signature, at zero extra cost. Add `roll`
everywhere — and note that the already-logged `zero > shuffle` pattern (§Claim 3)
points the same way.

### L — n=1 audit: which conclusions would not survive a second seed

Verified from wandb: **only the DC arms set a seed at all.** `5w72bo01`,
`sgdftf6b`, `ncztxyyo`, `pzmc2orq`, `t4bp8nki` set no `--seed` and have no `seed`
key (only `inference_seed: 0`). The only replicate pair anywhere is
`n3dbgq4q` (s0) vs `l2jcz9nx` (s1), both stopped at ~1020 steps: `effect_rel`
0.003399 vs 0.003785 at step 1000, an **~11% run-to-run spread** — measured only in
the blind regime.

| conclusion | n | survives a 2nd seed? |
|---|---|---|
| arm E ≫ arm 0 at steps 500–2000 (0.026 vs 0.0033) | 1 | **yes** — 8× is far outside any plausible spread |
| "arm E peaks at 0.1064" | 1 arm, 1 eval, 12 samples | **no** — and it is now known to be wrong (0.11479 @3500) |
| "3.6× the reference" | 1 | **no** — the untreated control also clears the reference by step 3500 |
| **"~6× faster escape"** | 1 trajectory per arm | **no** — a late transition's *timing* is the highest-variance quantity in the campaign and was measured once per arm. This is the n=1 that matters most, because "acceleration, not binary unlock" is the claim's current honest form and rests entirely on one crossing time per arm |
| "arm F escapes 2× faster than arm 0" | 1 each, 2 variables | **no** |
| "adapted loss 18% below control" | 1 | **no** — it is a step-mismatch artifact regardless of seed |
| Wan RT-1 "settles to ~0.021" | 1 unseeded run, 4 samples/eval | **partly** — the *band* over 34 evals is credible; the peak is not |
| SkyReels "opens at 0.0450" | 1 eval, 4 samples | **no** |
| DC/Wan/SkyReels all blind on ACWM (07-28) | 3 runs, 3 backbones | **yes** — an order of magnitude below the reference, concordant across independent backbones |
| AVID follows actions on ACWM (0.0295, null 0) | 1 run, 120 samples | **yes** — largest sample count in the campaign, clean null |
| the pedestal mechanism (48× realised/RMS) | 1 arm + init-to-step-600 trajectory | **yes** — a 48× structural gap with an init control is not a seed artifact |

### N (fatal) — "× the AVID reference" compares two different measurements

The **algebra is identical** (`action_sensitivity.py:99-102` vs
`probe_action_sensitivity.py:117-119`: L2 ratio over the flattened **composed**
prediction — `base·mask + adapter·(1−mask)` on both sides, base in the
denominator). Everything around it differs:

| axis | ours (arm E, training-loop probe) | AVID (`rqp4s3gp` probe) |
|---|---|---|
| **shuffle donor** | a **different batch of 24 independent clips** (`action_sensitivity.py:317`) | `torch.roll(act, 1, dims=0)` **within the batch** (`:285`), over a dataloader with `shuffle=False` enumerating windows episode-by-episode at a fixed index stride — so adjacent batch items are **heavily overlapping windows of the same episode**. AVID's "shuffle" is closer to our `roll` variant than to a cross-clip swap |
| timestep | `t ~ U{0..999}` **per clip**, one pooled norm over 24 clips at 24 different `t` ⇒ a **norm-weighted** mean including the extreme bands | fixed grid `[100,300,500,700,900]`, per-`t` ratio, uniform mean, extremes excluded |
| samples per number | **12** pooled ratios (4 batches × 3 draws) | **120** (8 × 5 × 3) |
| dynamic rescale | applied — `scale_arr[t]` on `x_start` before `q_sample` (`trainer.py:341`; `use_dynamic_rescale: true`, `base_scale: 0.7`, `configs/base/dynamicrafter512.yaml:33-34`) | **not applied in the probe** (`:296`), although AVID's *training* applies it (`ddpm3d.py:762-766`) |
| resolution | 384, matching training | **384, model trained at 320** (verified from the two launchers) |
| autocast | bf16 | fp16 |
| data pipeline | current code | **rewritten since the run** — see §3 item 17. The probe launcher loads the *current* config, so the reference is **not reproducible from the checkout** |

The donor difference is first-order and biases the **reference** downward,
inflating every "× the reference" multiplier — the 3.6×, the 5.3×/8.7×/22.7× grid,
and the "91%".

The 07-30 note *did* check one of these (uniform vs grid timesteps: 0.0033 vs
0.0034) and concluded "the sampling difference accounts for nothing." That check
ran **in the blind regime**, where every ratio is ~0 and differences cannot show.
It does not license the comparison at 0.106.

**In the campaign's favour:** the *direction* of the 07-30 comparison (AVID follows
actions, ours does not) survives all of this, because every bias listed favours
**our** numbers and ours were still 5–23× lower. The comparison only breaks when
run in the other direction — i.e. for the "we beat the reference" claim.

### O (weakens) — the `--emb-scale` diagnostic measures the wrong tensor for `action_token_norm`

`scripts/eval_action_sensitivity.py:511-515` hooks `action_embedding` **before**
`action_pos_emb` is added; `action_token_norm` is applied **after** the positional
add (`action_model.py:284-285`). So the reported "action-driven fraction" for the
Wan token path is not the fraction LayerNorm actually normalises. `action_pos_emb`
is separately the **fastest-moving parameter in the adapter** (17.5%/400 steps),
so once it exceeds the action content, LN's normalisation is dominated by the
positional term and the action-varying fraction of the emitted tokens can shrink —
**invisible to that probe**. Re-hook after the positional add before the
"token-norm fixes transport" claim is written.

### P (fatal to the D3 claim) — the "68× curvature signature" is entirely an N=1 bucket artifact

Every headline number in `20260729-shortcut-wan-vs-dc-curvature-signature`
reproduces exactly from wandb (0.860571 / 0.0125554 = 68.5×; cosines 0.962861 /
0.710170; base losses 0.183948 / 0.0475988; contributions 0.138253 / 0.260276;
Wan FID 119.541 vs base 86.9998, FVD 1699.20 vs 1010.76). **The arithmetic is
clean. The interpretation is not.** Per-step-size `train/shortcut_direction_loss/N*`
final values:

| bucket | N001 | N002 | N004 | N008 | N016 | N032 | N064 |
|---|---|---|---|---|---|---|---|
| DC `t4bp8nki` | **2.89779** | 0.00334 | 0.00336 | 0.00310 | 0.00160 | 0.00197 | 0.00157 |
| Wan `pzmc2orq` | **0.04031** | 0.00552 | 0.00377 | 0.00272 | 0.00169 | 0.00145 | — |

**For every N ≥ 2 the two backbones are within ~1.5× of each other.** The entire
68× lives in the single-step bucket. That is a *much* more specific — and more
interesting — statement than "diffusion is harder to make self-consistent", and it
is a different claim: it is about the largest step size only, which is exactly
where the curvature argument predicts the *sagitta* to be largest, but also exactly
where a target-rule difference (`v_average` vs `endpoint_inversion`) bites hardest.
Two further problems: the 68× compares **Wan@800 vs DC@500** (each run has only
**two** eval points), and at DC's other eval point (step 0) its consistency loss is
**0.0085 — lower than Wan's**. `t4bp8nki` also logs **no quality metrics at all**,
so the FID/FVD comparison exists on the Wan side only.

**Report the per-N table, not the aggregate ratio.** It is both more defensible and
a better result.

### Q (cosmetic but will be asked) — D3 has no positive result and the vault says so in three places

`storyline-experiment-requirements` marks D3 *"❌ NO CLEAN POSITIVE RESULT"*;
`writing-plan-2026-08` lists the only D3 empirical comparison as confounded;
`thesis-storyline` §Gates item 4 concedes step 5 is "a premise without a run" —
while the chapter map allocates §5.2 to "Shortcut on DynamiCrafter" with "**no run
backs this yet**". Internally honest, but the top-level narrative
("→ flow matching → shortcut models: the D3 contribution") promises an empirical
D3 that does not exist. **Decide now** whether D3 is analytical (the curvature
derivation, which is genuinely proven) or empirical, and write the arc
accordingly. The derivation alone is defensible; the arc as written is not.

---

## 6. What would survive — the claims I tried and failed to break

1. **The curvature / v-averaging derivation (D3, node 4).** `5.1% at s=1/4, 16.1%
   at s=1/2, 24.1% at s=3/4` for the averaged target vs `0.000000` for endpoint
   inversion, on `ddim_micro_step_v` **with zero model error**. No model, no seed,
   no dataset, no metric-definition exposure. Backed by a passing regression test
   (`tests/test_shortcut_endpoint_inversion.py`, 4 tests) and a shipped
   implementation (`279cdb7`). **The single most defensible thing in the thesis**,
   currently framed as one node among many. Promote it.

2. **"AVID's recipe follows actions on ACWM Robot Arm; ours does not" (as of the
   07-28/07-30 snapshot).** 0.029475 vs 0.0013–0.0056, null exactly 0 on both
   sides, same frozen base weights, same data, 120 samples. **It survives §5-N**:
   every protocol asymmetry biases the comparison *toward* our runs and ours were
   still 5–23× lower. A conclusion that holds against its own measurement bias is a
   strong conclusion. ⚠ **But bound it in time** — by step 3500 our untreated DC
   control reaches 0.0456, above the reference. The correct statement is *"at
   matched early training our implementation is an order of magnitude less
   action-sensitive than the reference"*, not *"our adapters are action-blind"*.

3. **The learned-pedestal mechanism.** `realised/RMS` 0.0050 vs 0.238 (48×);
   `cond ÷ time` 14.45 vs 0.83; and crucially the **trajectory**: at init
   0.182× / 2.4%-varying, by step 600 magnitude ×106 and varying fraction ÷7. The
   init measurement is the control that makes it causal rather than
   correlational — the architecture is fine, training builds the pathology. Two
   independent routes (direct 0.34%, Jacobian 0.50%) agree to 1.5×. The note also
   retracts its own weaker gradient claim and corrects a bad printed metric.
   **Exemplary work; it will survive a viva.** State it as a claim about the
   *conditioning pathway*, not about action-following.

4. **"The action signal reaching Wan's output is a global bag."** Steering cos
   −0.002/−0.003 (±0.07), temporal alignment 0.25 vs 0.28 chance, spatial
   concentration 11.4% vs 10% chance — three independent axes, all at chance, with
   chance levels **computed** rather than assumed, on the best-recipe checkpoint;
   corroborated by `action_pos_emb` being the fastest-moving parameter. Hard to
   attack because it is a *null* with computed nulls: the usual attacks (gain,
   selection, n=1) all push toward finding *more* structure, not less.

5. **"Sensitivity without control" on Wan.** Rollout-action-swap (job 25104155):
   GT-tracking true 0.0818 / shuffle 0.0807 / zero **0.0791** — the true-action
   rollout is *worse* than the zero-action one, same seed/clip/solver. One
   clip/seed/donor is a real limit, but the result is adversarial to the campaign's
   own hopes, which is what makes it credible. Ditto the accompanying admission
   that the *base* tracks GT better than any adapted variant (0.061 vs ~0.080).
   **Independently corroborated by the already-logged `zero > shuffle` pattern on
   both RT-1 cells** (§Claim 3) — the model responds to action presence, not
   identity.

6. **The methodological contribution.** *"Loss, gate, FID and sample quality are
   all blind to action-blindness; only purpose-built probes separate them."*
   Supported by several independent instances: `20260721` (beats base on PSNR at
   every eval while shuffled actions move the loss <1e-5), `20260730` (AVID looks
   healthy on loss/mask/FID/FVD on both Push Cube and Robot Arm, probes blind on
   one), the RT-1 quality/sensitivity split, and now the DC parity arms where a
   clean loss curve hides a 35× spread in `effect_rel`. **Probably the campaign's
   second-strongest contribution after the derivation** — and, unlike the D2
   headline, it does not require `effect_rel` to be an information measure. It only
   requires that `effect_rel` and the quality metrics *disagree*, which is not in
   dispute.

7. **The arm E vs arm 0 experiment design.** Verified from the archived run
   configs: the only differences in the entire YAML are `name`, comment blocks,
   `output_dir`, and `adapter.extra.condition_center: true`; identical CLI args
   except `--output-dir`; **same `--seed 0`, same git commit
   `75721b787a4fdff99e765da787a803fbe066721e`, same data dir, same
   `--frame-stride 4 --target-height 384 --target-width 512 --batch-size 24`**;
   `eval_action_base_null_violation` exactly 0 at every eval in every arm. **This
   is a clean single-variable A/B and it deserves to be said so.** The attacks
   above are on what the metric *means* and on the *numbers quoted from it* — not
   on the experiment's construction, which is the best in the campaign.

8. **The negative-result discipline.** The vault pre-registers predictions (the
   NOBASE trajectory, the 0.02 target, the 0.012 / 0.2–1.0 gradient bands), records
   them before the outcome, and marks them *partially* confirmed rather than
   confirmed. It retracts (`20260729` superseded by `20260730`), flags its own
   confounds (Push Cube memorisation, training-mismatch, the gate bug), and
   corrects its own arithmetic (~300× → ~80×). **An examiner who reads this vault
   will trust the author** — which is worth more than any number here, and is why
   every fix below is cheap: the instruments already exist.

---

## Recommended order of work — all before Ch5 prose is written

| # | Action | Cost | Kills / saves |
|---|---|---|---|
| 1 | `len(ds.episodes)` for `diffusion_skyreels_rt1_tokennorm_nobase.yaml` | 1 line | Threat G |
| 2 | Correct the four stale DC numbers, the 18% loss claim, the Wan peak, and the `rqp4s3gp` status | free | §3 items 1–4, 16 |
| 3 | Structure triad + rollout-swap on the arm-E step-3500 ckpt vs arm 0 | ~1 GPU-h | **Decides whether the thesis has a positive D2 result** |
| 4 | Quality re-eval of arm E and arm 0 vs the frozen base | ~1 GPU-h | C4 |
| 5 | Report `effect ÷ adapter` everywhere; add `roll` to `action_sensitivity_variants` | free | C1c, K |
| 6 | Re-report the D3 result as the per-N table, not the 68× aggregate | free | §5-P — and it *improves* the result |
| 7 | One SkyReels × ACWM run at the current recipe | 1 run | C5, the ending |
| 8 | Second seed of arm E and arm 0 to ≥3500 steps | 2 runs | C10, the "6×" |
| 9 | Re-probe the AVID reference at 320 with a genuine cross-episode shuffle; re-probe arm E on AVID's grid at 120 samples | ~1 GPU-h | §5-N, every "× the reference" |
| 10 | Re-eval Wan × RT-1 on a held-out split | ~1 GPU-h | C7 |

## Related

- `[[2026-08-01-quality-vs-sensitivity-inverse]]` · `[[2026-08-01-input-blindness-audit]]` — parallel overnight audits; the quality/sensitivity anti-correlation there is the strongest single argument against `effect_rel` being an information measure, and it is folded into §Claim 3 and C7a above
- `[[../30_Knowledge/writing/thesis-storyline]]` · `[[../30_Knowledge/writing/storyline-experiment-requirements]]` · `[[../30_Knowledge/writing/writing-plan-2026-08]]`
- `[[../30_Knowledge/experiments/_index]]`
- `[[../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]`
- `[[../20_Tickets/experiments/exp-backbone-flow-without-temporal-vae-compression]]`
- `[[../50_Decisions/open/wan-action-following-needs-objective-change]]`
