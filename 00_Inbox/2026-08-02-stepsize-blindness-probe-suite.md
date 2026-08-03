# 2026-08-02 — step-size blindness probe suite (D3), built + verified, NOT submitted

The D3 analogue of the D2 action-blindness suite. Three probes, a 2×2 Slurm job
over {DynamiCrafter (diffusion), Wan2.2 (flow)} × {shortcut arm, matched control},
and a CPU unit-test suite with closed-form oracles for both model classes.

**Nothing has been run on a checkpoint. Every number below is either a code fact
(cited `file:line`), a synthetic-fixture result, or a unit-test result.** The job
is written and NOT submitted.

Motivation and design lessons taken from
[[../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]],
[[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]],
[[../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]] and
[[2026-08-01-effect-rel-is-a-gain-metric]].

---

## 1. What was implemented and where

All paths relative to `/home/lukas/projects/generative-flow-adapters/`.

| File | Lines | What |
|---|---|---|
| `src/generative_flow_adapters/evaluation/stepsize_structure.py` | new, 837 | Probes 2 + 3 core, backbone-agnostic |
| `scripts/probe_stepsize_embedding_standalone.py` | new, 394 | Probe 1, CPU-only, torch-only, no package import |
| `scripts/eval_stepsize_blindness.py` | new, 308 | GPU driver for probes 2 + 3 |
| `tests/test_stepsize_structure.py` | new, 21 tests | CPU oracles, both model classes |
| `jobs/experiments_cluster/acwm_phys/shortcut/submit_probe_stepsize_blindness.sh` | new | the 2×2 job |
| `src/generative_flow_adapters/evaluation/__init__.py:24-28,36-46` | edited | exports |

Key entry points:

- `stepsize_structure.py:268` `run_stepsize_probes(...)` — driver
- `stepsize_structure.py:204` `_consistency_target_2d(...)` — the diffusion/flow dispatch
- `stepsize_structure.py:616` `summarise(...)`, `:728` `format_stepsize_report(...)`
- `probe_stepsize_embedding_standalone.py:172` `_statistics(...)` — the pedestal decomposition

Nothing existing was modified except the `evaluation/__init__.py` export block. No
config, no training code, no vault note other than this file.

### Reuse rather than reimplementation

- The `2d` target is **never** derived locally. `stepsize_structure.py:225-247`
  calls `compute_self_consistency_target_v` / `compute_self_consistency_target_v_flow`
  (`training/shortcut_targets.py:51,130`) with the same arguments the training path
  passes at `training/trainer.py:1840-1843` (flow) and `:1868,1874-1878` (diffusion).
- The timestep-jump conversion reuses `ShortcutStepSchedule.to_timestep_jump`
  (`training/step_schedule.py:134`), matching `trainer.py:1868`.
- `_frame_mask_bool`, `_clip_vector`, `_cos`, `NULL_TOLERANCE` are imported from
  the D2 module `evaluation/action_structure.py:136,151,158,92` so the two
  campaigns' numbers are computed the same way.
- Step-level injection goes through `Trainer._inject_step_level` (`trainer.py:1921`);
  the test suite binds the **real** method rather than reimplementing it
  (`tests/test_stepsize_structure.py:86-87`).

---

## 2. The three probes

### Probe 1 — step-level embedding pedestal

`scripts/probe_stepsize_embedding_standalone.py`. CPU-only, torch-only, seconds.
Rebuilds the step-level MLP from checkpoint tensors alone and feeds it the
configured dyadic ladder.

**What the encoder is** (this is why a pedestal is even possible here):

- DC output adapter: `nn.Sequential(Linear(1, 64), SiLU, Linear(64, cond_dim))` at
  `adapters/output/dynamicrafter.py:113-121`, state-dict prefix `step_level_embed`,
  transform `log2` (`conditioning/utils/dynamicrafter_conditioning.py:38-51`).
- Wan `ActionWanModel`: `nn.Sequential(Linear(1, cond_hidden_dim), SiLU, Linear(cond_hidden_dim, dim))`
  at `backbones/wan/modules/action_model.py:125-126`, prefix `step_embed`,
  transform `log2` (`:221-224`). Added to the time embedding at `:225`.

Both are found automatically (`_PREFIX_CANDIDATES`, `probe_stepsize_embedding_standalone.py:52`).

**Definition.** With `emb = enc(transform(d))` over the ladder, all statistics in
float64 (`:194` — in float32 the mean of K identical rows is off by an ulp and
fabricates ~1e-7 of spurious "variation with d"):

- `input_independent_fraction` = share of per-element energy in the level-mean —
  the pedestal. `1.0` = ignores `d` entirely. This is the statistic the D2 note
  reported as "99.7 % input-independent".
- `realised_over_rms` = across-level RMS ÷ total RMS = `sqrt(1 − input_independent_fraction)`.
  Headline; **scale-free**, so a pathway-gain change cannot move it.
- `jacobian_realised_over_rms` — the linearised per-element form, kept because it
  is the shape the D2 headline (0.0050 vs 0.238) is quoted in.

**Null controls.**
1. *Constant-input null* (`:304`): feed the same level at every rung; across-level
   variation must be `0` **by construction**. Exit 4 if not. Verified 0.000e+00 on
   all fixtures after the float64 fix.
2. *Frozen-base null* (`:268`): the base must carry no step-level tensors. Checked
   **before** the encoder is rebuilt and exits with a structural error if violated.
   Reported together with the base-key count so a *vacuous* pass on an
   adapter-only checkpoint is visible rather than claimed as verified.

**Chance / reference level.** There is no permutation chance for a ratio, and the
D2 action numbers are a 7-D-input encoder — **they are not thresholds here** and
are printed as context only (`_D2_CONTEXT`, `:64`). The reference the probe
actually scores against is **the same architecture at random init**, computed by
the probe itself (`:328-346`). Trained ≪ random-init ⇒ the encoder *learned* a
pedestal; trained ≈ random-init ⇒ the step path has not moved. That is the D2
architectural-vs-learned distinction, made self-contained.

**Degenerate case.** An all-zero encoder gives 0/0 for every ratio. Reported as
`degenerate: true` with `None` values and printed as `DEGENERATE`, never as a
number.

### Probe 2 — consistency direction (the `steer_cos` analogue)

`stepsize_structure.py`, probe A.

```
consistency_cos = cos( pred(2d) − pred(d),  target_2d − pred(d) )
```

`target_2d` is the two-half-step composition the shortcut objective supervises,
obtained from the trainer's own target function (§3). ~0 ⇒ the model responds to
`d` **arbitrarily** — step-size sensitivity without step-size control, the exact
D3 transposition of the D2 failure. Computed per `(d, 2d)` rung over the ladder.

**Null controls.**
- *Frozen base* (`:484-488`): the base cannot see `step_level`, so
  `base_null_rel` must be ≤ `NULL_TOLERANCE` (1e-6). Driver exits 3 otherwise.
- *Forward-determinism* (`:413-422`): every number is a difference of two
  forwards, so the same forward is run twice and must reproduce. This is the
  control the 2026-08-01 train-mode-dropout bug
  ([[../20_Tickets/bug-eval-stepsize-probe-runs-in-train-mode]]) would have tripped
  directly. Driver exits 6 otherwise. The probe also sets `eval()` itself
  (`:343-349`), which is the fix that bug needed.
- *Isotropic-direction null* — validates the ±1/√D band on real tensor shapes.
- *Mismatched-clip null* — this clip's response vs another clip's consistency
  direction. Unlike the D2 action probe there is no shared-term problem (each
  `dtarget` is built from one clip alone), so a roll by 1 already gives disjoint
  pairs and batch size 2 suffices; noted at `:513-518`.
- *Wrong-rung null* (`:525-538`) — this rung's response vs the consistency
  direction for a **different** `d`. The sharpest null for D3: a model that moves
  with `d` along a `d`-*independent* direction scores the same on both.
  It ships with `target_rung_alignment` = cos(dtarget_d, dtarget_neighbour) so
  the null is *readable*: near +1 means neighbouring rungs ask for almost the
  same direction and the null cannot discriminate. (On the synthetic near-linear
  test oracle it is ≈0.99, i.e. uninformative — which is exactly why the
  alignment is reported rather than the null being trusted blindly.)

**Chance level.** 0.0, with per-clip sd 1/√D and sd-of-mean 1/√(D·n), printed
next to every number, plus a bootstrap CI95.

**Degenerate case.** A step-size-blind adapter returns bit-identical predictions
for `d` and `2d`, so `pred(2d) − pred(d)` is the zero vector and torch's cosine of
a zero vector is `0.0` — indistinguishable from a real "at chance". Such clips are
counted as `degenerate_clips` and **excluded**; if every clip degenerates the rung
reports `consistency_cos: null`, `degenerate: true` and the report prints
`STEP-SIZE BLIND — not a cosine of 0, there is no direction to score`
(`:495-500`, `:652-658`, `:760-768`). This is the D2-port bug transposed.

Additional bookkeeping: `clamped_clips` counts clips where `t < 2d`, so the
diffusion target's second sub-step clamped at `t = 0` and the relation is
truncated — a low-`t` batch cannot masquerade as a weak result (`:466-471`).

### Probe 3 — monotonicity / ordering

`||pred(d) − pred(d_ref)|| / ||pred(d_ref)||` across the whole ladder (`d_ref` =
smallest). Reports the full profile plus:

- `spearman` — rank correlation against ladder order. Chance 0, sd 1/√(K−1).
- `monotone_fraction` — adjacent non-decreasing pairs. Chance 0.5.
- `is_monotone`, and `chance_fully_monotone = 1/K!`.

Each analytic chance is printed next to an **empirical permutation estimate**
(`_monotonicity_permutation_chance`, `:591`); if they disagree the analytic chance
is wrong and the probe should not be read.

`eval_stepsize_effect_rel` is a **max** over levels (`trainer.py:807`) and so
cannot distinguish a clean ramp from a scrambled profile with the same maximum.
That claim is proved as a unit test, not asserted.

**Null control.** The base's own ladder profile, which must be identically zero;
reported as `base_null_degenerate` with its mass.

**Degenerate case.** An all-zero profile has all ranks tied ⇒ Spearman is
undefined (`_spearman_against_order` returns `nan`, `:566-580`) ⇒ reported as
`spearman: null`, `degenerate: true`, never as 0.

**Honesty caveat, stated in the report itself** (`:817-818`): monotone ordering is
what a *systematic* `d`-dependence looks like; it is **not** implied by the
shortcut objective. Probe 2 is the correctness probe; probe 3 separates
"systematic" from "erratic" among models that clear the `effect_rel ≠ 0` bar.

### Why gain cannot inflate any of the three

The red-team finding in [[2026-08-01-effect-rel-is-a-gain-metric]] is that
`effect_rel` is monotone in pathway gain. All three probes are invariant by
construction: probe 1's `realised/RMS` is a ratio; probe 2's `consistency_cos` is
a cosine; probe 3's Spearman is rank-based (invariant under any strictly
increasing rescaling). The gain companion `|dpred|/|dtarget|` is reported
*separately* per rung — it is the number a gain knob moves. Pinned by
`test_consistency_cos_is_invariant_to_step_pathway_gain`, which scales the
step-size pathway 100× and asserts the cosine does not move while `mag_ratio` does.

---

## 3. Diffusion vs flow — how it is handled

There is **no sign table anywhere**. `_consistency_target_2d`
(`stepsize_structure.py:204-247`) dispatches on `model.model_type` and validates
`model.prediction_type`:

| declared | target function called | `d` units | source of the target kind |
|---|---|---|---|
| `flow` / `flow_matching` | `compute_self_consistency_target_v_flow` (`shortcut_targets.py:130`) | sigma, with `trainer._flow_timestep_scale` | fixed (velocity average) |
| `diffusion` | `compute_self_consistency_target_v` (`shortcut_targets.py:51`) | timestep jump via `to_timestep_jump` | `trainer.config.shortcut_consistency_target` |

Which matches the arms: DC is `diffusion`/`velocity` with
`shortcut_consistency_target: endpoint_inversion`
(`configs/dynamicrafter/diffusion_dc_shortcut_d3arm_actionfree_robotarm.yaml:104`),
Wan is `flow`/`velocity` with `v_average`
(`configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml:90`).

`prediction_type` not in `(None, "velocity", "v")` **raises** (`:216-223`) rather
than reporting a number: both target functions are derived for a velocity head
(`invert_ddim_v` inverts the DDIM v-step; the flow target averages two
velocities). This is the D2 lesson — the verbatim Wan steering formula had a
signal coefficient of exactly zero under ε-prediction — turned into a refusal
instead of a silent zero.

A `model_type` that is neither also raises rather than falling through (`:233-238`).

One deliberate divergence from the D2 probe: `condition_drop_prob` is **not**
forced to 0 (`:328-338`). The action probe must switch the action on because the
action is what it perturbs; this probe perturbs `step_level`, and the D3 arms are
action-free by construction (`drop_condition_prob: 1.0` /
`action_dropout_prob: 1.0` on the Wan configs, `conditions: []` on the DC ones).
Overriding it would probe the adapter off-distribution. The forward-determinism
control covers the risk that a stochastic dropout breaks the pairing.

---

## 4. Verification evidence

Run with `.venv/bin/python` in `/home/lukas/projects/generative-flow-adapters/`.

**`py_compile`** — clean on all five touched files.
**`bash -n`** — clean on the job script.

**Argparse interrogated directly** (monkeypatched `parse_args`, inspected
`_actions`), never a `--help | grep`:

```
scripts/eval_stepsize_blindness.py: 26 options, missing=[]
scripts/probe_stepsize_embedding_standalone.py: 9 options, missing=[]
```

The job's own guard block was extracted and run **verbatim**: exit 0 with the real
flag list; exit 1 with a bogus flag injected (negative control). Both directions
confirmed, so the guard cannot false-positive the way the earlier grep-based one did.

**Unit tests** — `tests/test_stepsize_structure.py`, **21 passed** on CPU in 1.6 s.
`tests/test_action_structure.py` + `test_action_sensitivity.py` + `test_step_schedule.py`
+ `test_shortcut_endpoint_inversion.py` + `test_consistency_losses.py` still pass
(65 passed), so the `evaluation/__init__.py` change broke nothing.

Two independent oracle families:

1. **Analytic, flow, fully hand-derived** — `f(x, t, s) = x − s·b` at `x_t = b`.
   Everything collapses to a multiple of `b`: `dtarget = −(d/2)(1−d)·b`,
   `dpred = −d·b`, both negative multiples ⇒ cos exactly +1. It never calls
   `compute_self_consistency_target_v_flow`, so it would survive even if the
   trainer's own target function were wrong.
   - `test_flow_analytic_oracle_scores_plus_one` → +1.0 (abs 1e-4)
   - `test_flow_analytic_sign_flip_scores_minus_one` (`f = x + s·b`) → −1.0
   - `test_consistency_cos_is_invariant_to_step_pathway_gain[0.05|0.5|5.0]` →
     cos = +1 at all three; `mag_ratio` matches the closed form `2λ/(1−λd)`.

2. **Self-consistent oracle, both classes** — a model defined *as* the fixed point
   of the codebase's own stepping rule, built with `ddim_micro_step_v` /
   `invert_ddim_v` / `flow_micro_step_v`.
   - `test_exactly_consistent_model_scores_plus_one[diffusion-endpoint_inversion |
     diffusion-v_average | flow-v_average]` → +1.0 (abs 1e-3) on every
     (class, target-kind) combination the arms actually use.
   - `test_mirrored_model_scores_minus_one[diffusion|flow]` — the
     deliberately-wrong `pred(2d) = 2·pred(d) − target_2d` → −1.0 (abs 1e-3).
   - `test_target_kind_is_read_from_the_trainer_config_not_hardcoded` — an
     `endpoint_inversion` fixed point scores +1 under that config and strictly
     lower under `v_average`. Proves the target kind comes from
     `training.shortcut_consistency_target`, so DC and Wan are not silently
     measured against the same relation.
   - `test_wrong_model_class_does_not_score_plus_one` — the D2 trap transposed:
     the same DDIM-consistent oracle scores +1 declared `diffusion` and < 0.99
     declared `flow`. Only possible if the dispatch reads `model_type`.
   - `test_non_velocity_prediction_type_is_refused_not_reported` — raises.

Degenerate / harness tests:

- `test_step_size_blind_model_is_degenerate_not_at_chance` — the most important
  one. Asserts every rung reports `degenerate: true` / `consistency_cos: null` /
  `num_scored_clips == 0`, that probe 3 reports `spearman: null`, that the base
  null is exactly 0.0, and that the report says `STEP-SIZE BLIND` and
  `null control OK`.
- `test_leaking_base_is_reported_as_void_not_as_a_number` — `HARNESS ERROR` + `VOID`.
- `test_nondeterministic_forward_is_caught_by_the_repeat_control` — `NONDETERMINISTIC`.
- `test_model_without_return_base_is_refused` — raises.
- `test_effect_rel_cannot_separate_monotone_from_scrambled_but_probe_b_can` — two
  models with an identical **maximum** response and different orderings;
  `is_monotone` True vs False, Spearman strictly lower for the scrambled one.
- `test_monotonicity_chance_levels_match_their_permutation_estimates` — permuted
  chance lands on (0.000 ± 0.05, 0.500 ± 0.05), i.e. the analytic chance printed
  next to the score is the real one.
- `test_all_equal_profile_has_undefined_spearman_rather_than_zero`.

**Probe 1 on synthetic fixtures** (scratchpad, four hand-built checkpoints):

| fixture | `realised/RMS` | random-init ref | constant-input null | exit |
|---|---|---|---|---|
| random-init encoder | 0.525473 | 0.552788 | 0.000e+00 | 0 |
| deliberate pedestal (tiny input weights, bias = 1) | 3.61599e-07 | 0.552788 | 0.000e+00 | 0 |
| all-zero encoder | `DEGENERATE` | 0.552788 | 0.000e+00 | 0 |
| step-level tensor under `base_model.` | — refused before the rebuild | — | — | non-zero |

The pedestal fixture separates from random init by ~1.5 × 10⁶, and the degenerate
fixture reports `DEGENERATE` rather than a plausible number. These are synthetic
fixtures — **they say the probe works, they say nothing about any real run.**

**Two real defects were caught during this verification** (both the class of thing
the D2 port's oracle testing caught, so recording them):

1. `_statistics` in float32 made the constant-input null read 8.3e-08 instead of 0,
   because `mean(dim=0)` over K bitwise-identical rows is off by an ulp. The
   statistics now run in float64 and the null is exactly 0.000e+00.
2. `rebuild_encoder` globbed *every* key containing the prefix, so a
   `base_model.step_level_embed.*` tensor was spliced into the rebuilt Sequential
   and crashed on a shape mismatch instead of being reported as the structural
   null violation it is. The base-leak check now runs first and the rebuild
   excludes base-prefixed tensors.

**Not verified (cannot be, from this host):** no GPU, no cluster filesystem, no
checkpoints. Nothing was submitted and nothing that needs a GPU or a large
checkpoint download was run. Probes 2 and 3 have never executed against a real
backbone — only against CPU oracles.

---

## 5. The job

`jobs/experiments_cluster/acwm_phys/shortcut/submit_probe_stepsize_blindness.sh`
— 2×2 over {DC, Wan} × {shortcut, control}, modelled on
`jobs/experiments_cluster/acwm_phys/dc/submit_probe_dc_arm_structure.sh` (same
module loading, `UV_*`/`PATH` exports, venv activation, `#SBATCH --chdir`, and the
argparse fast-fail guard).

| arm | config | `training.output_dir` (verbatim) |
|---|---|---|
| DC shortcut | `configs/dynamicrafter/diffusion_dc_shortcut_d3arm_actionfree_robotarm.yaml:108` | `outputs/dc-shortcut-D3-arm-run` |
| DC control | `configs/dynamicrafter/diffusion_dc_noshortcut_control_actionfree_robotarm.yaml:115` | `outputs/dc-shortcut-NOSHORTCUT-control-run` |
| Wan shortcut | `configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml:85` | `outputs/wan-shortcut-actionfree-robotarm` |
| Wan control | `configs/wan22/diffusion_wan22_noshortcut_control_actionfree_robotarm.yaml:94` | `outputs/wan-noshortcut-control-actionfree-robotarm` |

Each arm runs probe 1 (CPU, first — a degenerate encoder already explains a blind
probe 2/3 without spending GPU time) then probes 2 + 3. Split: `ind_test`
(held out; the arms trained on `ind_train`). Ladder pinned to the one all four
configs train on — `1/128 … 1` — so the four cells sit in one table. DC gets
`--timesteps 300,500,700,900` (AVID's grid, minus 100 which clamps at the coarse
rungs); the flow path has no diffusion timestep sampler and rejects the flag.

Missing arms are **skipped with a message**, not fatal — the Wan control (Slurm
25150730, queued 2026-08-02) may not have checkpointed yet.

The final 2×2 comparison block reads only the JSON the runs just wrote and prints
nothing it did not read, so it cannot invent a number. Refused statistics (`null`)
and empty means (`NaN`) both print `n/a`, never `0` — verified against synthetic
summaries including a fully degenerate arm.

**Exact submit command** (login node, from the repo root):

```bash
mkdir -p logs/shortcut          # Slurm will NOT create --chdir's dir
sbatch jobs/experiments_cluster/acwm_phys/shortcut/submit_probe_stepsize_blindness.sh
```

**NOT SUBMITTED.** The D3 arms are mid-flight; a human reviews and submits.

**Expected runtime — analysed estimate, not measured.** Inputs: probe 1 is CPU
matrix work on a ≤64×512 MLP (seconds, measured on the fixtures). Probes 2+3 cost
per (batch × draw) roughly `K + 3·R` adapter forwards plus `2·R` inside the target
(K = 8 ladder levels, R = 7 rungs) ≈ 43 forwards; at 4 batches × 2 draws that is
~344 forwards per arm, ×5 for the DC timestep stratification. Against the training
throughput recorded in `30_Knowledge/tech/wan-shortcut-step-throughput.md`
(Wan micro-step 15.4 s including backward; a forward-only no-grad pass is a
fraction of that), plus Wan's 5B model load and latent-cache warm-up.
**Estimate: DC ~20–40 min/arm, Wan ~45–90 min/arm, ~3 h for all four.** The
`#SBATCH --time` is set to 4:00:00 with that margin. This is an estimate with
shown reasoning, not a measurement — treat the wall-clock as unknown until the
first run reports.

---

## 6. Needs human review before submission

1. **On-disk checkpoint layout — the one thing I could not verify.** There is no
   `/scratch-shared` on this host. `resolve_run_dir` tries
   `/scratch-shared/$USER/outputs/<run>`, `/scratch-shared/$USER/<run>` and
   `outputs/<run>` and skips the arm if none has a `checkpoints/*.pt`. Confirm at
   least one candidate matches, or the whole job will politely skip all four arms.
2. **Wan invocation details.** `--wan-ckpt-dir ckpts/Wan2.2-TI2V-5B` and
   `--latent-cache-dir $ROOT/latents.shared` follow
   `submit_train_wan_noshortcut_control_robotarm.sh:106-107`, but
   `submit_probe_wan_action_trace.sh:74` uses
   `/scratch-shared/$USER/ckpts/Wan2.2-TI2V-5B` instead. Pick whichever this
   cluster actually has. Also confirm `--max-area 589824` and `--num-windows 8`
   match the arms' training preprocessing, or the latent cache misses and a full
   VAE encode runs.
3. **`#SBATCH --chdir` is hardcoded to `/home/lbierling/...`**, copied from the D2
   job. Correct only if the cluster username is `lbierling`.
4. **Wan memory.** `--mem=360G` and one H100 copied from the Wan training job.
   Probes are forward-only under `no_grad`, so this should be generous — but the
   Wan arm holds the 5B plus its VAE, and probe 2's target takes two extra
   adapter forwards per rung. If it OOMs, drop `--num-batches` to 2 first.
5. **`--timesteps` grid for DC.** `t ≥ 300` was chosen so `t − 2d` does not clamp
   at the coarse rungs (`d = 1/2` ⇒ jump 500 ⇒ needs `t ≥ 1000`). The coarsest
   rungs *will* still clamp at every listed `t`; the probe counts and reports
   those clips (`clamped_clips`) rather than hiding them, but the coarse-rung
   numbers should be read with that in mind. Worth a decision on whether to
   restrict the reported ladder to rungs where `2d` fits inside `t`.
6. **Interpretation guard.** Probe 2's wrong-rung null is only discriminative when
   neighbouring rungs' consistency directions differ; `target_rung_alignment` is
   printed next to it for exactly that reason. On the synthetic oracle it was
   ≈0.99 (uninformative). If it comes back near 1 on the real arms, the wrong-rung
   column must not be read as a null.
7. **Timing.** The DC arm was at ~step 200 when this was written, with
   `eval_stepsize_effect_rel = 0`. Running the suite now would measure a very
   early checkpoint; a blind result at step 200 is not evidence of anything.
   Consider waiting for a later checkpoint, or run now as a **baseline** and
   repeat later — the job always takes the newest checkpoint (`ls -t | head -1`).
