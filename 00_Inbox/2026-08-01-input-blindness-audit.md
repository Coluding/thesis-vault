# Input-blindness audit — what is the adapter actually learning, and where is it blind?

> Overnight evidence audit, 2026-08-01. **Read-only**: no code, config or run was
> touched. Sources are (a) the `30_Knowledge/experiments/` notes 20260629–20260801,
> (b) the implementation repo at HEAD `75721b78`, (c) a fresh `wandb.Api()` pull
> made **2026-08-01** (summary = last logged value unless a step is given).
> Nothing here is written into `30_Knowledge/` — this is an inbox artifact for the
> user to triage.

---

## 1. Headline

The adapter is learning a **large, high-quality, action-independent correction of
the frozen base's *appearance/domain***, and almost nothing about dynamics. The
strongest single piece of evidence is new in this audit: on ACWM Robot Arm the
Wan token-norm adapter beats the frozen base on **all six** quality metrics —
FID 57.4 vs 90.1, FVD 406 vs 1118, LPIPS 0.192 vs 0.239 (`52o3uxz8`, step 3315,
wandb summary) — while its measured action-following sits at `effect_rel` ~0.011
with steering direction, temporal alignment and spatial concentration all at
chance ([[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]).
A 2.75× FVD improvement that carries no action information is, by elimination,
a **domain/appearance correction**; and a domain correction is exactly the kind
of function that can be *near-constant across clips*. That hypothesis has never
been tested. The campaign has measured action-dependence to exhaustion and has
measured dependence on **every other input either not at all or not cleanly**:
there is no probe anywhere in the repo that varies the conditioning frame /
start latent, no probe that isolates the timestep, no trustworthy step-size
number on any trained adapter, and on the Wan family the text path is
structurally dead. The "learned pedestal" that was found in DC's *action
embedding* is a mechanism that could equally be operating at the *output*, and
the suite as built cannot see it.

---

## 2. The input-dependence table

`clip` below = the conditioning frame / start latent, i.e. "which video am I
looking at". Line numbers are from repo HEAD `75721b78`.

| Input | Measured? | By what | What the evidence says | If unmeasured: the probe that settles it |
|---|---|---|---|---|
| **Start latent / conditioning frame** | **NO — total gap** | nothing | No probe anywhere perturbs `x_t`, `x0`, or DC's `cond["concat"]`. The closest is `base_shuffle` (`generate_wan22_i2v_compare.py:329`), which swaps a *different clip's base prediction* while leaving `x_t` intact — deliberately an inconsistent pair, not a clip swap. On Wan the observation frames live **inside** `x_t` (`wan22_batch_preprocessor.py:143-144`), so nothing isolates them. | **P1 — output variance decomposition.** See §5. Blind ⇒ the adapter's own output is ~the same tensor for every clip. |
| **Timestep t / σ** | **Partially — swept, never isolated** | `--sigma-sweep` (`generate_wan22_i2v_compare.py:461-607`); `--timesteps` (`eval_action_sensitivity.py:372-378, 697-719`) | `--sigma-sweep` rebuilds `x_t` **at each σ** (`L521-524`), so σ and `x_t` co-vary — it cannot answer "does the correction depend on t". What it does show: the adapter's relative deviation from base varies ~2× across σ (0.106 @ σ=0.05 → 0.053 @ σ=0.7, [[../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]] §2), i.e. **not flat**, but confounded. `--timesteps` pins t and re-runs the *action* probe; it reports `action_effect_rel(t)` (flat ±2% on DC arm 0, [[../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]] §2), never `‖pred(t₁)−pred(t₂)‖`. There is **no** `eval_timestep_effect_rel`. Architectural note: the output-head and Wan adapters collapse the per-frame timestep to one scalar via `t.flatten(1).amax(dim=1)` (`output_head.py:135-136`, `adapters/output/wan.py:101-102`) — **per-frame timestep structure is discarded by construction**. | **P2 — timestep probe.** See §5. |
| **Step size `d`** | **Probe exists; never produced a trustworthy number** | `eval_stepsize_effect_rel` (`trainer.py:703-775`); `--stepsize-perturb` (`generate_shortcut_fewstep.py:226-235`) | **This is the worst finding of the audit.** Across all `coluding` projects, exactly **one** run has ever logged `eval_stepsize_effect_rel` with a clean null: `hcrnc9gf` (`Wan2.2-shortcut-actionfree-acwm-robotarm`), an **8-step smoke test** — effect_rel 0.2568, cos 0.973, null **0**. The two real D3 runs (`pzmc2orq` Wan @1199, `t4bp8nki` DC @910) have **no step-size keys at all**. The nine DC ACWM parity runs *do* log it (effect_rel 0.049–0.061) **but their configs set `use_step_level_conditioning: false`** (`diffusion_dc_acwm_robotarm_armE_center.yaml:87`), so the adapter has no `d` input — and their null control is **violated**: `eval_stepsize_base_null_violation` = 0.028–0.039 (see §6). So the ~0.05 is at most ~1.4× a leak/noise floor and means nothing. **Net: the step-size conditioning of a trained shortcut adapter has never been measured.** D3 currently has zero evidence that its adapters are step-size conditioned. | **P3** — read `eval_stepsize_effect_rel` **and** `eval_stepsize_base_null_violation` off D3 arms A/B (jobs 25141979 / 25141980, project `dc-shortcut-fewstep-d3` — not yet created at pull time). **Gate: refuse to interpret effect_rel unless null < 1e-3.** |
| **Base output (the oracle)** | **YES — one probe** | `--base-attribution` (`generate_wan22_i2v_compare.py:271-343`) | drel: `base_zero` **1.13 / 1.01**, `base_shuffle` 0.84 / 0.78, `act_shuffle` **0.0087 / 0.0082** at σ ∈ {0.5, 0.83} — the ~100:1 oracle-vs-action ratio (job 25097452 on `ncztxyyo` step_00001000, [[../30_Knowledge/experiments/20260731-why-wan-copies-the-base-decomposed]] §2). **Corroborated and contextualised:** the ratio holds, but note (a) `_base_attribution` calls `model.adapter(...)` **directly** (`:334`), bypassing `AdaptedModel`'s condition encoder, so `cond["embedding"]` is absent and the AdaLN action route falls back to `null_cond_emb` (`action_model.py:219-220`) — the probe measures the **cross-attention path only**; (b) in *this* config that is harmless (`action_injection: cross_attention`, `..._gatelow_capshift_acwm_robotarm.yaml:67`) and the full-path `_action_trace` independently gives block-out drel **0.0085** ≈ the probe's 0.0087; (c) it **would** silently report `act_shuffle = 0.0` for any `action_injection: adaln` config. | Corroborated. Add a guard: raise if `cond` has no `"embedding"` while a condition encoder is configured. |
| **Action `a_t`** | **YES — exhaustively (7+ probes)** | `eval_action_effect_rel` (`trainer.py:630-701`), `--action-trace`, `--action-analysis`, `--rollout-action-swap`, `--emb-scale`, `--timesteps`, `probe_action_embedding_standalone.py` | Sensitivity without control. Wan ACWM plateau ~0.011; RT-1 ~0.021; DC after `condition_center` far higher (§3). Structure is at chance on Wan × ACWM: steering cos −0.002/−0.003, temporal alignment 0.25 vs 0.28 chance, spatial concentration 10.5–11.4% vs 10% chance ([[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]]). Rollout-swap: true actions track GT **no better** than shuffled or zero (0.0818 / 0.0807 / 0.0791, job 25104155). | — |
| **Text / prompt** | **NO — and structurally dead on Wan** | none | `Wan21OutputAdapter.__init__` (`adapters/output/wan.py:31-53`) never exposes `use_text_context`, and its `forward` (`:125-132`) never passes `context=` — so `ActionWanModel.text_embedding` is always `None`. **Text cannot reach the Wan/SkyReels adapter under any config.** `--use-prompt` / `--guide-scale` affect only the *base* rollout's CFG. On DC, `cond["context"]` **does** reach the adapter (`adapters/output/dynamicrafter.py:163-170, 214`) and is unmeasured. | Wan: no probe needed — it is blind by construction; state it in the method chapter. DC: add `text_zero` / `text_shuffle` variants to the P1 harness. |

**Latent trap, not currently triggered:** `_MLPBackbone.forward` does
`del x_t, base_output` (`output_head.py:185-186`) — that backbone is
*architecturally* frame-blind. No live config selects it (all use
`backbone: wan` / `transformer` / `unet`), so no result in the campaign is
affected — but it is a one-line config away from producing a
"working" adapter that provably cannot see the video.

---

## 3. Quality assessment — is the adapter a net improvement over the frozen base?

All numbers are wandb **summary** (last logged) values, pulled 2026-08-01.
`eval/adapted/*` vs `eval/base/*` **within the same run** is the valid
comparison; base values differ across runs because eval sets/steps differ.

| Backbone × dataset | Run | Step | FID ↓ | FVD-I3D ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ | MSE ↓ | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Wan × ACWM Arm (tokennorm) | `52o3uxz8` | 3315 | **57.4** / 90.1 | **406** / 1118 | **0.192** / 0.239 | **0.825** / 0.812 | **18.35** / 16.47 | **0.0146** / 0.0225 | **adapter wins 6/6** |
| Wan × ACWM Arm (tokennorm-nobase) | `vy9tcuco` | 3054 | **59.7** / 82.4 | **704** / 1282 | **0.199** / 0.237 | **0.825** / 0.811 | **18.29** / 16.67 | **0.0148** / 0.0215 | **adapter wins 6/6** |
| Wan × ACWM Arm (SIMPLE transformer) | `7bmzwv6u` | 2936 | **73.6** / 81.4 | **865** / 1059 | **0.230** / 0.240 | **0.8135** / 0.8127 | **17.12** / 16.71 | **0.0194** / 0.0213 | **adapter wins 6/6** |
| Wan × ACWM Arm (GATEFIX control) | `tny84p7k` | 3557 | **79.7** / 90.1 | **736** / 1118 | **0.2245** / 0.2389 | 0.8122 / 0.8125 | **16.95** / 16.47 | **0.0202** / 0.0225 | wins 5/6 (SSIM tie) |
| Wan × ACWM Arm (original blind arm) | `ncztxyyo` | 1200 | 91.0 / 88.3 | **780** / 902 | **0.2365** / 0.2394 | **0.8116** / 0.8102 | **16.84** / 16.47 | **0.0207** / 0.0226 | wins 5/6 (FID) |
| Wan × ACWM Arm (cap09-shift5) | `slsbey7x` | 1151 | **98.8** / 102.9 | 1082 / 1063 | 0.2255 / 0.2213 | 0.8143 / 0.8149 | **16.81** / 16.49 | **0.0208** / 0.0224 | mixed 3/6 |
| SkyReels × ACWM Arm | `8zjjn7wl` | 897 | **84.5** / 94.2 | **1992** / 2525 | **0.148** / 0.172 | **0.916** / 0.867 | **19.15** / 17.47 | **0.0122** / 0.0179 | **adapter wins 6/6** |
| **Wan × RT-1** (tokennorm-nobase) | `5w72bo01` | 13999 | 132.5 / **113.2** | **1576** / 1617 | 0.382 / **0.365** | **0.659** / 0.648 | **17.53** / 16.53 | **0.0176** / 0.0222 | **split** |
| **SkyReels × RT-1** (tokennorm-nobase) | `sgdftf6b` | 4399 | 161.4 / **145.1** | 2515 / **2257** | 0.498 / **0.461** | 0.462 / **0.483** | **12.70** / 11.76 | **0.0537** / 0.0666 | **split** |
| **SkyReels × RT-1 (ORACLE ON)** | `gi44pv5k` | 1360 (running) | 152.4 / **147.2** | 2498 / **2112** | 0.502 / **0.467** | 0.471 / **0.474** | **12.81** / 12.42 | **0.0524** / 0.0573 | **split** |
| Wan × ACWM Arm, **action-free shortcut** (D3) | `pzmc2orq` | 1199 | 119.5 / **87.0** | 1699 / **1011** | 0.289 / **0.234** | 0.807 / **0.811** | **16.62** / 16.45 | **0.0218** / 0.0226 | **split** ⚠frozen gate |
| Wan × ACWM **Push Block** | `sed5al2v` | 2799 | 120.4 / **111.9** | 328 / **296** | **0.1012** / 0.1035 | **0.960** / 0.948 | **19.33** / 19.04 | **0.0117** / 0.0125 | **split (mild)** ⚠frozen gate |
| Wan × MetaWorld (replace, nobase, 1-clip overfit) | `rxzwh4ak` | — | 408.7 / **111.4** | _nv_ | _nv_ | _nv_ | **23.39** / 16.51 | **0.0046** / 0.0223 | **split (extreme)** — [[../30_Knowledge/experiments/20260724-metaworld-cap-shift-triangle-base-parity]] |
| Wan × MetaWorld (AdaLN, button subset) | `_nv_` (proj `Wan2.2-avid-i2v-metaworld34-button`) | — | ~100 / ~75 | ~1850 / ~1250 | 0.40 / 0.357 | — | **16.8** / 15.6 | **0.021** / 0.0275 | **split** — [[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]] |
| **DC × ACWM Arm — every arm incl. the spine cell (arm E)** | `6oyu1inq`, `tr0uovs5`, `86kb01su`, `n3dbgq4q`, `l2jcz9nx`, `hbuu4lwx`, `1e0fe9ei`, `2us8hugq`, `t62nhyfu` | 1000–3856 | **absent** | **absent** | **absent** | **absent** | **absent** | **absent** | **cannot be assessed** |

### Reading

1. **The split is NOT universal, and it is NOT an ACWM phenomenon.** On ACWM
   Robot Arm the adapter is a *large* net improvement — including on every
   perceptual metric. Wan tokennorm cuts FVD 2.75× and FID 1.6×. The
   pixel-better/perceptual-worse split appears on **RT-1** (both backbones),
   **MetaWorld** (2026-07-09, 2026-07-12, 2026-07-16, 2026-07-24 runs), ACWM
   **Push Block**, and the **action-free D3 shortcut arm**.
2. **⚠ The split is co-linear with the frozen-gate bug on ACWM.** Every ACWM run
   that splits — `slsbey7x`, `ncztxyyo`, `sed5al2v`, `pzmc2orq` — logged
   `eval_adapter_gate_mean` **exactly 0.5** with `eval_adapter_gate_std`
   **exactly 0**: the `gate_cap == σ(gate_bias)` freeze
   ([[../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]). Every ACWM
   run that does **not** split — `tny84p7k`, `52o3uxz8`, `vy9tcuco`, `7bmzwv6u` —
   has a live gate (std 0.009–0.052). The two factors cannot be separated on
   ACWM. **RT-1 is the clean evidence:** all three RT-1 runs have live gates and
   split anyway. Clean two-factor read of the table:
   - live gate + ACWM → adapter wins everything (4/4 runs)
   - live gate + RT-1 → split (3/3 runs)
   - frozen gate + ACWM → split or mixed (4/4 runs)
3. **Two candidate mechanisms, and they may both be real.** ⚠ *analysed
   estimate — inputs: rows 1–2 above; reasoning:* (a) a gate frozen at 0.5 makes
   the composed output a literal 50/50 **average of two videos**, which is blurry
   by construction — that alone produces better L2 and worse FID/FVD/LPIPS
   without any statement about what was learned; (b) on RT-1, where the gate is
   live, the split must instead come from training-time hedging toward the
   conditional mean under an L2 objective. **Label: analysed estimate.** The
   practical consequence is the same either way — **do not cite the split as one
   phenomenon** until the frozen-gate arms are re-run with a live gate.
4. **The dividing variable for the *magnitude* of the win looks like base-domain
   gap.** ⚠ *analysed estimate — inputs: the table; reasoning:* the cells where
   the adapter wins everything are the ones where the frozen base is badly out of
   domain (synthetic ACWM renders; base FVD 1000–2500), leaving a large
   systematic appearance error that a correction removes and that both metric
   families reward. Where the base is already near-distribution (real RT-1; Wan
   base FID 113), there is nothing left to win except L2. **Analysed estimate,
   not a measured causal claim.**
5. **This corrects the note in
   [[../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]**,
   which attributes the RT-1 split to "once the oracle is removed". The
   **oracle-ON** SkyReels RT-1 arm (`gi44pv5k`, step 1360) shows the **same**
   split. Removing the oracle is not the cause. ⚠ caveat: `gi44pv5k` is at 1360
   steps vs `sgdftf6b`'s 4399 — not step-matched; the direction is what stands.
6. **The action-free D3 arm splits too** (`pzmc2orq`), so mean-regression does
   not *require* action conditioning — but that run also has a frozen gate, so
   this is suggestive, not established (see row 2).
7. **DC — the cell the writing plan names as the thesis spine — has no quality
   evidence at all.** Not one of the 18 runs in `dc-acwm-robotarm-avid-parity`
   logs any `eval/adapted/*` or `eval/base/*` key (verified via both `summary`
   and `scan_history`); the same is true of `dc-shortcut-actionfree-robotarm`,
   `dc-acwm-robotarm` and `dc-acwm-pushblock`. They log only losses, adapter
   diagnostics and `eval_step_grid` images. The open question in
   `30_Knowledge/writing/writing-plan-2026-08.md:56-58` ("check whether the same
   split holds for DC arm E before relying on the spine cell") **cannot be
   answered from existing data**, and the spine cell currently has *no measured
   generation quality of any kind*.
8. **The AVID reference cannot be compared on quality either.** `rqp4s3gp`,
   `93qrvr5v`, `5e4m9dxz`, `423pjv8y` log rich quality metrics (e.g. `rqp4s3gp`
   val FID 67.50, FVD 332.3, LPIPS 0.0857, PSNR 24.40, SSIM 0.9416, MSE 0.0193)
   but have **no frozen-base arm** — single-model training, so no adapted-vs-base
   delta exists and the split is not computable on the reference.
9. **Updated DC action numbers (final logged evals, later than the vault
   snapshot).** wandb summary: arm E `6oyu1inq` `eval_action_effect_rel`
   **0.11479** @ step 3856 (`effect_vs_adapter` 0.388); arm 0 `tr0uovs5`
   **0.04564** @ 3604; arm F `86kb01su` **0.05049** @ 3798; null 0 on all three.
   The 07-31 note's snapshot (E 0.0917, 0 0.0288, F 0.0392 @ ~3000) understates
   all three. The `condition_center` advantage at the final eval is
   **2.3–2.5×**, not the ~3.5× recorded — and **all three arms now exceed the
   AVID reference 0.0295**, strengthening "blindness is a long transient".

### Action sensitivity on the same runs (same pull, for context)

`eval_action_effect_rel` / `eval_action_effect_vs_adapter`, last logged:
RT-1 — `sgdftf6b` 0.0173, `gi44pv5k` 0.0211, `5w72bo01` 0.0249 (effect÷adapter
0.31–0.37); ACWM Arm — `tny84p7k` 0.0020, `7bmzwv6u` 0.0022, `52o3uxz8` 0.0062,
`vy9tcuco` 0.0077 (effect÷adapter 0.05–0.14). `eval_action_base_null_violation`
= 0 in every run that logs it. So the cell with **3–10× more action sensitivity
is the same cell where the adapter is a net quality regression** — which is
independent support for the fit-instability reading of `effect_rel`
(§4 item 6), not against it.

Git commits for the paired runs: `75721b78` (`5w72bo01`, `sgdftf6b`, `gi44pv5k`,
`52o3uxz8`, `vy9tcuco`, `7bmzwv6u`, `tny84p7k`, `86kb01su`), `1bfad67a`
(`8zjjn7wl`, `ncztxyyo`), `62df2b46` (`slsbey7x`), `4bf98af2` (`pzmc2orq`),
`ec52d384` (`sed5al2v`).

---

## 4. Ranked list of the most likely remaining problems

**1. The ACWM quality win may be a near-constant output correction — i.e. the
same "learned pedestal" failure, one level up.**
A 2.75× FVD improvement with action structure at chance is by elimination an
appearance/domain correction, and a global colour/contrast/sharpness offset is
input-independent by nature. If the correction is constant across clips, the D2
contribution reduces to "learned calibration of a frozen base", which is a much
weaker claim than the chapter currently plans to make. **Nothing in the repo can
currently distinguish this from a clip-dependent correction** (§2 row 1).
*Cheapest decisive experiment:* **P1** (§5) on the retained `vy9tcuco` /
`7bmzwv6u` checkpoints — no training, one probe job, ~30 lines added to an
existing probe mode.

**2. The whole ACWM quality story is confounded by the frozen-gate bug.**
Four of the seven ACWM cells ran with `gate_mean` pinned at exactly 0.5 and
`gate_std` exactly 0, and those four are exactly the four that show the
pixel-vs-perceptual split (§3 reading 2). A 50/50 blend of two videos is blurry
by construction, so on ACWM the "mean-regression signature" and "dead gate"
cannot be told apart. Any Ch5 sentence that reads the split as evidence about
*what the adapter learned* is currently unsupported on ACWM.
*Cheapest decisive experiment:* none needed for the four live-gate ACWM runs
(they already win everything); for the D3 arm, arm C of
[[../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]]
(job 25141988) is the re-run with `gate_cap: 0.9` and settles it for free.

**3. D3 has no trustworthy step-size measurement, at all.**
One clean number exists and it is from an 8-step smoke (`hcrnc9gf`). Everything
else is either missing or contaminated. The thesis's D3 claim ("the adapter is
step-size conditioned, so few-step rollout is real") is currently unsupported by
any measurement on a trained model.
*Cheapest decisive experiment:* read `eval_stepsize_effect_rel` +
`eval_stepsize_base_null_violation` off D3 arms A/B (25141979/25141980) at their
first eval; **do not interpret the effect unless the null is < 1e-3**. Zero
extra compute.

**4. The step-size probe's null control is failing on DC (0.028–0.039) and the
mechanism is unknown.**
`eval_stepsize_base_null_violation` should be exactly 0 — the frozen base does
not read `step_level`, and `prepare_dynamicrafter_condition` does not even inject
it when `use_step_level_conditioning: false` (`adapters/output/dynamicrafter.py:149-156`).
The value is identical to 16 digits across two different runs (`6oyu1inq` and
`tr0uovs5`: 0.035836007446050644), so it is **deterministic**, not sampling
noise; and condition dropout is ruled out (`_sample_condition_drop_mask` is
gated on `self.training`, `adapted_model.py:329`, and the probe runs under
`model.eval()`, `trainer.py:585`). Mechanism: _needs verification_.
*Cheapest decisive experiment:* call `model(x_t, t, cond, return_base=True)`
twice with the **identical** cond and assert bitwise equality of the base
output. Two minutes on any GPU. If it fails, the base forward is nondeterministic
and **every paired-forward probe on DC needs re-reading**; if it passes,
`step_level` is reaching the frozen base and that is a conditioning leak.

**5. The thesis spine cell (DC) has no generation-quality evidence.**
*Cheapest decisive experiment:* resume arm E from the retained step-3600
checkpoint for a single eval cycle with the quality eval enabled — one eval, no
training.

**6. `effect_rel` cannot separate action *information* from early-fit
*instability*.**
Already correctly flagged by the user in
[[../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]] §confound;
probe 25143284 is in flight. Adding to it: the RT-1 cells are *also* the cells
where the adapter is a net quality **regression**, which is independent support
for the instability reading — worse fit, more sensitivity to any perturbation.
*Cheapest decisive experiment:* the in-flight probe, read against the ACWM
chance baselines.

**7. Timestep dependence is unmeasured, and partly foreclosed by architecture.**
A correction that is constant across noise level is a distinct, publishable
failure mode and it has never been looked for. The scalar-`amax` collapse
(`output_head.py:135-136`, `adapters/output/wan.py:101-102`) additionally means
the adapter cannot see *which* frames are noised — the same
"pixel→latent correspondence never formed" pathology as the temporal action bag,
on a different channel.
*Cheapest decisive experiment:* **P2** (§5), a ~40-line mirror of
`_stepsize_sensitivity_eval`; runs inside the existing eval cycle.

**8. `_base_attribution` silently bypasses the condition encoder.**
Harmless for the numbers already published (cross-checked against `_action_trace`
to within 3%, §2), but it will fabricate `act_shuffle = 0.0` for any
`action_injection: adaln` config. A trap for anyone re-running the probe.
*Cheapest fix:* one assertion.

**9. Text is structurally dead on the Wan/SkyReels adapter.**
Not a bug for D2 (actions are the conditioning of interest), but it is a fact
about the architecture that the method chapter must state rather than imply
otherwise, and it removes one candidate explanation for the ACWM appearance
gains.

---

## 5. The two probes that do not exist and should

### P1 — clip-dependence of the adapter's own output (settles the headline)

**Where:** new variant block inside `_base_attribution`
(`scripts/generate_wan22_i2v_compare.py:326-339`), or a sibling
`--input-attribution` mode. The harness (batched clips, shared σ, shared noise
generator, direct `model.adapter(...)` call returning `adapter_output`) is
already exactly right.

**What to compute**, on a batch of N clips with a **shared** noise draw and a
shared `t`, using a **no-oracle checkpoint** (`vy9tcuco`, `7bmzwv6u`, `5w72bo01`)
so that rolling `x_t` does not create an inconsistent oracle pair:

```
A       = adapter_output                       # [N, C, T, H, W]
A_bar   = A.mean(dim=0, keepdim=True)          # the pedestal
clip_varying_frac(A) = ‖A − A_bar‖ / ‖A‖
```

**Reference (positive control), same formula on the base's own prediction:**
`clip_varying_frac(base_output)` — definitionally clip-dependent, and in the
same units, so it calibrates the scale exactly the way `--emb-scale` uses AVID's
0.238 to calibrate DC's 0.0050.

**Null control (the floor):** the same statistic with **one clip repeated N
times** under N *different* noise draws. That is the noise-driven variation
floor; anything at or below it is blind.

**Threshold.** Blind if `clip_varying_frac(adapter) < 0.1 ×
clip_varying_frac(base)`, or if it is within 2× of the repeated-clip floor.
Healthy if it is within ~2× of the base's value. (Threshold chosen to mirror the
48× ours-vs-AVID gap that was already accepted as decisive in
[[../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]] §3 —
**pre-register it before running.**)

**Second output, free:** the same decomposition per channel and per frame tells
you *what kind* of correction it is. A pedestal concentrated in the low-frequency
/ DC component across all clips = colour/contrast calibration; that is the
concrete form of "the D2 contribution is a learned calibration".

**Related existing machinery worth reusing:** the across-clip / per-frame
variance decomposition already exists at `eval_action_sensitivity.py:604-626` —
but it is applied only to `cond_emb`, **never to the adapter's output**. P1 is
that same code pointed one layer downstream.

### P2 — timestep-dependence of the correction

**Where:** `src/generative_flow_adapters/training/trainer.py`, a
`_timestep_sensitivity_eval` mirroring `_stepsize_sensitivity_eval`
(`:703-775`), wired at `:625` next to the existing probes.

**Mechanics:** hold `x_t` fixed, override only the `t` argument over
t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}·T, paired. Report
`eval_timestep_effect_rel = max ‖pred(t) − pred(t_ref)‖ / ‖pred(t_ref)‖` and
`eval_timestep_cos`.

**Control:** here the frozen base is a **positive** control, not a null — a
denoiser must move a lot with t. Emit `eval_timestep_base_effect_rel` in the
same units.

**Threshold:** timestep-blind if
`effect_rel(adapter) < 0.1 × effect_rel(base)`. Pre-register.

**Caveat to state:** holding `x_t` fixed while moving `t` is off-manifold. That
is intentional — it is a question about the *function*, not about the data
distribution — and it is the same convention the step-size probe already uses.

---

## 6. What I could not verify

- **Whether the adapter's correction is clip-dependent.** The central question of
  this audit. No probe exists, no checkpoint is reachable from this machine
  (`/scratch-shared/lbierling/outputs/` is empty here; local `outputs/` holds no
  post-07-20 probe artifacts), and I ran no jobs per the brief. **P1 is required.**
- **Whether the correction is timestep-dependent.** Same. The only adjacent
  evidence — the 07-21 σ-sweep `rel dev` column (0.106 → 0.053) — is confounded
  by `x_t` co-varying with σ.
- **Step-size dependence of any trained adapter.** Never measured (§2).
- **DC generation quality, including arm E.** Not logged on any of 18 runs.
- **The mechanism of the nonzero `eval_stepsize_base_null_violation` on DC.**
  Measured and deterministic; cause _needs verification_ (§4 item 4).
- **Whether the "domain-gap explains the split" reading is causal.** It is an
  analysed estimate (§3.2) built on within-run base-vs-adapted deltas across
  heterogeneous eval configs. A controlled test would be the same backbone ×
  adapter on a dataset where base FVD is deliberately varied.
- **Step-matching across the quality table.** Runs sit at different steps
  (897–13999) and different eval-set configurations; base FID/FVD differ between
  runs of the same cell, so **only within-run comparisons are used above** and
  cross-cell magnitudes are indicative only.
- **`rxzwh4ak` and the 2026-07-09 MetaWorld rows** are transcribed from vault
  notes, not re-pulled; their FVD/LPIPS fields are `_nv_` where the note does not
  give them.
- **"Our adapter vs AVID on quality"** — AVID's runs log quality but have no
  frozen-base arm, so no comparable adapted-vs-base delta exists (§3 reading 8).
- **Whether the ACWM split is the gate or the objective.** The two are perfectly
  co-linear in the existing data (§3 reading 2). Arm C of the D3 ticket
  (job 25141988) is the natural resolver.
- **FlowWM** (`8gfw9nva` running, `11vl8rjf`) logs only `eval/l1_vae_decoded`
  (0.09228 action vs 0.09858 base, oracle 0.07987, copy-last 0.10329) — **no
  perceptual metric at all**, so the split is unmeasurable there.
- **Convergence.** 9 of the 12 paired runs are `crashed`/`killed` mid-training,
  at steps spanning 897–13999. Nothing in §3 is a converged comparison.

---

## Related

- [[../30_Knowledge/experiments/20260731-why-wan-copies-the-base-decomposed]] — the oracle 100:1 and the 0.45% economics
- [[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]] — direction/time/space all at chance
- [[../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]] — the pedestal metric P1 transposes to the output
- [[../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]] — the quality split, and the effect_rel confound
- [[../30_Knowledge/writing/writing-plan-2026-08]] — the DC-quality question this audit reports as unanswerable
