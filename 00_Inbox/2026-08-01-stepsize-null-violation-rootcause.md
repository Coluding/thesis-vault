# Root cause: `eval_stepsize_base_null_violation` ≠ 0 on DC

> Overnight debug, 2026-08-01. **Read-only on the repo**: no code, config, vault
> note or job was modified; nothing was launched. Sources are (a) repo HEAD
> `75721b7` at `/home/lukas/projects/generative-flow-adapters/`, (b) a fresh
> `wandb.Api()` pull made 2026-08-01, (c) a CPU-only reproduction using the real
> vendored DC `ResBlock`. Follow-up to
> [[2026-08-01-input-blindness-audit]] §4 item 4, which left the mechanism as
> `_needs verification_` — **this note resolves it and corrects one inference in
> that audit.**

---

## 1. Verdict

**Nondeterminism. Specifically: the frozen base runs in `train()` mode during
the step-size probe, so its `nn.Dropout(p=0.1)` layers fire and draw a fresh
mask on every forward.**

It is **not** a conditioning leak — `step_level` is *structurally incapable* of
reaching the DC base (§3.1). It is **not** bf16/numerical noise — the identical
harness produces an *exact* 0 on Wan and an *exact* 0 for the action probe on
the very same DC runs (§2).

**Confidence: very high (~95%).** Four independent lines of evidence converge
(§3.5), and the one CPU experiment I could run reproduces the qualitative
signature exactly: **eval mode → `0.000000000`, train mode → `0.046–0.73`.**

What would take it to ~100%: the two-minute GPU check in §5 — load any DC arm,
call `model(x_t, t, cond, return_base=True)` twice with **identical** cond, and
compare `b` bitwise under `model.train()` vs `model.eval()`. Predicted result:
differs in train mode, bit-identical in eval mode. I could not run it here (no
GPU / no checkpoint on this machine, per the brief).

### Correction to the previous audit

[[2026-08-01-input-blindness-audit]] §4 item 4 ruled out train-mode effects
because *"the probe runs under `model.eval()`, `trainer.py:585`"*. **That
inference is wrong.** The `self.model.eval()` at `trainer.py:585` sits inside a
`try:` (`:591`) whose `finally:` (`:616`) restores `self.model.train(was_training)`
at **`:617`** — and the step-size probe is called at **`:626`**, i.e. *after* the
restore. The model is back in **train** mode by the time the probe runs.

```
trainer.py:584    was_training = self.model.training      # True during training
trainer.py:585    self.model.eval()
trainer.py:591    try:
trainer.py:616    finally:
trainer.py:617        self.model.train(was_training)      # <-- TRAIN MODE RESTORED HERE
trainer.py:622    result.update(... self._probe_eval() ...)          # sets eval() itself :793
trainer.py:624    result.update(self._action_sensitivity_eval(...))  # sets eval() itself :314
trainer.py:626    result.update(self._stepsize_sensitivity_eval(...))# <-- NO eval(), NO fork_rng
```

The audit's own "condition dropout is ruled out" conclusion happens to remain
true (`_sample_condition_drop_mask` is a *different* mechanism and affects only
the composed output, `adapted_model.py:328-334`) — but the premise it rested on
does not hold, and the real dropout is elsewhere: inside the base UNet.

---

## 2. Evidence

### 2.1 Reproduced from wandb (pull 2026-08-01)

`eval_stepsize_base_null_violation` — last logged value, project
`coluding/dc-acwm-robotarm-avid-parity` unless noted:

| run id | config | null violation | `effect_rel` | `stepsize_cos` | `action_base_null` |
|---|---|---|---|---|---|
| `6oyu1inq` | armE_center | 0.035836 | 0.05250 | 0.99862 | **0** |
| `tr0uovs5` | arm0_baseline | 0.035836 | 0.05051 | 0.99873 | **0** |
| `86kb01su` | armF_nativeavidencoder | 0.034054 | 0.04894 | 0.99880 | **0** |
| `hbuu4lwx` | armA_concat | 0.032661 | 0.06110 | 0.99813 | **0** |
| `t62nhyfu` | armC_concat_stride1 | 0.038753 | 0.05846 | 0.99829 | **0** |
| `l2jcz9nx` | arm0_baseline | 0.031039 | 0.05667 | 0.99839 | **0** |
| `n3dbgq4q` | arm0_baseline | 0.032277 | 0.06074 | 0.99816 | **0** |
| `1e0fe9ei` | armB_stride1 | 0.027751 | 0.05646 | 0.99841 | **0** |
| `2us8hugq` | armC_concat_stride1 | 0.028378 | 0.05583 | 0.99844 | **0** |
| `hcrnc9gf` | **Wan** `wan-shortcut-stepprobe2` (proj `Wan2.2-shortcut-actionfree-acwm-robotarm`, 8-step smoke) | **0 (exact)** | 0.25678 | 0.97342 | *(n/a)* |

Magnitude **0.028–0.039 confirmed**, matching the brief.

**This table is exhaustive.** A sweep of **all 46 projects** under the `coluding`
entity (`api.projects('coluding')`, then every run's summary keys) found exactly
**10 runs in the entire campaign** that have ever logged any `eval_stepsize_*`
key: the 9 DC parity runs above (**every one violated**) and `hcrnc9gf`
(**the only exact 0** — and it is Wan). The split is perfectly clean along the
backbone axis, with no counter-example anywhere in the history.

### 2.2 It is DC-only — Wan is exactly 0

`hcrnc9gf` logs `eval_stepsize_base_null_violation` = **0**, exactly, under the
same trainer, the same probe, the same bf16 autocast. **This single fact kills
the "bf16 accumulation noise" hypothesis outright**: the harness demonstrably
emits an exact zero when the base forward is deterministic. bf16 is bitwise
reproducible for identical inputs on identical kernels; it does not produce
1e-3, let alone 3e-2.

The structural difference: **Wan has no `Dropout` module anywhere.** A grep of
`src/generative_flow_adapters/backbones/wan22/` and `models/base/wan*.py` for
`Dropout`/`dropout` (excluding `action_dropout_prob`/`drop_condition_prob`)
returns **nothing**. DC has dropout on every residual branch (§3.2).

### 2.3 It drifts across evals — it is NOT a fixed constant

`scan_history` for `6oyu1inq` and `tr0uovs5` (`eval_stepsize_base_null_violation`
per eval step):

| step | `6oyu1inq` (armE_center) | `tr0uovs5` (arm0_baseline) |
|---|---|---|
| 0 | 0.027891 | 0.027891 |
| 500 | 0.033920 | 0.033920 |
| 1000 | 0.032277 | 0.032277 |
| 1500 | 0.044940 | 0.044940 |
| 2000 | 0.037119 | 0.037119 |
| 2500 | 0.038247 | 0.038247 |
| 3000 | 0.038733 | 0.038733 |
| 3500 | 0.035836 | 0.035836 |

Two readings, both important:

1. **The value moves across evals (0.0279 → 0.0449 → 0.0358) while the base is
   frozen.** A deterministic conditioning leak into a *frozen* base would give
   the *same* number at every eval. It does not. This is the signature of a
   freshly-drawn random mask.
2. **The value is bit-identical across two runs with *different adapters*** —
   `6oyu1inq` (armE, `condition_center` BatchNorm on) vs `tr0uovs5` (arm0,
   baseline) agree to all 6 printed digits at every step, while their
   `effect_rel` differs (0.06095 vs 0.06642 at step 500). So the null violation
   is a **pure property of the frozen base forward**, independent of the adapter
   — and reproducible run-to-run because both runs share a seed and an identical
   RNG-consumption pattern.

The audit read the cross-run identity as proof of determinism. It is not: it
proves *seeded* reproducibility. Which brings us to §2.4.

### 2.4 "Deterministic" was a red herring — dropout looks constant here

The audit inferred "deterministic, not sampling noise" from the 16-digit
agreement. But the statistic is a **relative L2 norm over a multi-million-element
tensor**, and such norms concentrate as ~1/√N. Measured in the CPU repro
(§4, n=12 independent draws, train mode):

```
mean = 0.114715   std = 2.60e-04   min = 0.114261   max = 0.115034
relative spread = 0.23%  -> stable to ~3 significant figures
```

**Pure randomness, yet constant to 3 s.f.** So "the number looks deterministic"
was never evidence against dropout. Combined with a fixed seed it becomes
*exactly* reproducible, which is what the wandb table shows.

### 2.5 The natural experiment: action null = 0 on the same runs, same instant

Every DC run above reports `eval_action_base_null_violation` = **0** while
reporting a step-size null of 0.028–0.039 — same model, same eval cycle, same
probe batches, same forward machinery. The difference is **entirely in the probe
code**:

| | action probe | step-size probe |
|---|---|---|
| location | `evaluation/action_sensitivity.py:242-388` | `training/trainer.py:703-775` |
| sets `model.eval()` | **yes** — `:314`, restored `:382` | **no** |
| `torch.random.fork_rng` + `manual_seed` per forward | **yes** — `:328`, `:354` | **no** |
| null violation on DC | **0** | **0.028–0.039** |

This is as close to a controlled A/B as the logged data can give, and it isolates
the cause to the two missing guards.

---

## 3. Mechanism

### 3.1 A `step_level` leak into the base is impossible by construction

`AdaptedModel.forward` does pass the probe's injected cond straight to the base:

- `adapted_model.py:181` — `base_output = self.base_model(x_t, t, cond=cond)`

and `_inject_step_level` (`trainer.py:1885-1902`) does put `step_level` into that
dict. **But the DC base filters it out with a strict whitelist**:

- `models/base/dynamicrafter_video.py:290-315` — `_to_lvdm_cond` builds the lvdm
  conditioning from **only** `c_concat`, `c_crossattn`, `concat`, `context`
  (`:307-315`) plus kwargs `fs`, `act`, `dropout_actions` (`:304-306`).

`step_level` is in none of those lists, so it is **dropped before the base UNet
is ever called**. The two base forwards in `_run(ref)` / `_run(lv)` therefore
receive **bit-identical arguments**.

Additionally, on the nine parity runs `use_step_level_conditioning: false`
(e.g. `configs/dynamicrafter/diffusion_dc_acwm_robotarm_armA_concat.yaml:77`), so
`prepare_dynamicrafter_condition` does not inject it into the *adapter* either.

⇒ **Identical inputs, different outputs ⇒ the base forward is nondeterministic.**
There is nothing else it can be.

### 3.2 The nondeterminism: `nn.Dropout(p=0.1)` in the frozen DC base

The DC base is built from `configs/base/dynamicrafter512.yaml` (referenced at
`configs/dynamicrafter/diffusion_dc_acwm_robotarm_armA_concat.yaml:57`), which
sets:

- `configs/base/dynamicrafter512.yaml:54` — `dropout: 0.1`
- `configs/base/dynamicrafter512.yaml:60` — `temporal_conv: True`

Those reach two dropout sites in the vendored UNet:

- `backbones/dynamicrafter/modules/networks/openaimodel3d.py:197` —
  `nn.Dropout(p=dropout)` in **every** `ResBlock.out_layers`.
- `backbones/dynamicrafter/modules/networks/openaimodel3d.py:210` — the
  `TemporalConvBlock` is constructed with a **hard-coded `dropout=0.1`**,
  independent of config, applied at `:280`, `:286`, `:292`.

The base is correctly `.eval()`-ed and frozen at construction
(`models/base/interfaces.py:18-22`, `models/base/dynamicrafter_video.py:154-156`)
— **but neither `BaseGenerativeModel` nor `BaseVideoModel` overrides
`train()`**. So `Trainer`'s `self.model.train()` (`trainer.py:500`) and the eval
loop's restore (`trainer.py:617`) recursively put the frozen base back into
train mode, re-arming both dropout sites.

### 3.3 The exact failing code path

```
trainer.py:617   self.model.train(was_training)          # base -> train mode
trainer.py:626   self._stepsize_sensitivity_eval(...)
trainer.py:748       p, b = self.model(x_t, t, c, return_base=True)   # called per level,
                                                                     # NO base_output= reuse
adapted_model.py:181     base_output = self.base_model(x_t, t, cond=cond)  # fresh forward
openaimodel3d.py:197         nn.Dropout(p=0.1)   -> FRESH MASK EVERY CALL
openaimodel3d.py:210/280/292 TemporalConvBlock dropout=0.1 -> FRESH MASK
trainer.py:754   null_viol = max(null_viol, ‖b(lv) − b(ref)‖ / ‖b(ref)‖)   # != 0
```

Note `trainer.py:748` is the **only** paired-forward site in the codebase that
does *not* reuse the base output. The training path already does it correctly:
`trainer.py:356` and `:414` pass `base_output=reusable_base`, so the shortcut
consistency **target is not dropout-mismatched** (important for §4.3).

### 3.4 Candidates eliminated

| Candidate | Eliminated how |
|---|---|
| `step_level` leaking into the base via shared nested cond (`dict(cond)` shallow copy) | `_to_lvdm_cond` whitelist drops `step_level` before the UNet (`dynamicrafter_video.py:303-315`). Also `_inject_step_level` never mutates in place — it builds `new_cond = dict(cond)` and assigns one new key (`trainer.py:1895-1902`). |
| `return_base=True` returning a post-composition / post-normalised tensor | `adapted_model.py:181-184` returns the raw `base_output` computed under `torch.no_grad()` **before** `_compose_with_adapter`. It is genuinely the frozen base. |
| bf16 / autocast accumulation noise | Wan `hcrnc9gf` reports **exactly 0** under the same autocast; action probe reports exactly 0 on DC. Exact zeros are achievable ⇒ not dtype noise. Also 0.03 is ~30× a plausible bf16 floor. |
| `BatchNorm1d` `adapter_condition_center` (the brief's prime suspect) | It lives in the **adapter's** UNet only — injected via `adapters/output/dynamicrafter.py:100` (`params["adapter_condition_center"]`) into the adapter's `UNetModel`. The frozen base is built from `dynamicrafter512.yaml`, which never sets it. It cannot touch `b`. Confirmed empirically: `tr0uovs5` (arm0, `condition_center` **off**) has the *same* null violation as `6oyu1inq` (armE, **on**). |
| `condition_drop_prob` / `_sample_condition_drop_mask` | Gated on `self.training` *and* affects only the composed output, never `base_output` (`adapted_model.py:194-196, 328-334`). Also `drop_condition_prob: 0.0` in the parity configs. |
| `_forward_and_loss` re-drawing noise/timestep between levels | It is called **once per batch**, outside `_run` (`trainer.py:737`). `x_t`/`t` are captured and reused verbatim for every level. Not the cause. |
| Stochastic VAE encode / random crop re-drawn per forward | Both happen in the preprocessor, upstream of `_forward_and_loss`; the probe reuses the already-materialised `x_t`. |
| Adapter-side dropout | Real (`act_cond_diffusion_11M.yaml:68` sets `dropout: 0.1`) and it *does* contaminate `effect_rel` — but it cannot affect `b`, which is the base-only tensor. Contributes to §4, not to the null. |

### 3.5 The four converging lines

1. `step_level` provably cannot reach the base (whitelist) ⇒ inputs identical.
2. The base has dropout 0.1 ×2 sites and is provably in train mode at `:626`.
3. Wan (no dropout modules) = exact 0; DC (dropout) = 0.028–0.039.
4. The action probe, which adds `eval()` + `fork_rng`, = exact 0 on the same runs.

---

## 4. Blast radius

### 4.1 `eval_stepsize_effect_rel` on the nine DC parity runs: **100% artifact**

Those configs set `use_step_level_conditioning: false`, and §3.1 shows the base
never sees `step_level` either. **Nothing in the model can respond to `d`, so the
ground-truth `effect_rel` is exactly 0.** The entire logged 0.049–0.061 is noise
— base dropout plus adapter dropout (`act_cond_diffusion_11M.yaml:68`) through
the `avid_mask_mix` composition.

This is stronger than "effect_rel ≈ null_violation, so the metric measures
nothing". Here the true value is *known* to be zero, which makes these nine runs
an accidental but excellent **calibration of the noise floor** on this exact
backbone × data × eval config:

> **DC ACWM Robot Arm noise floor: `effect_rel` ≈ 0.049–0.061 when the true
> effect is 0, with `null` ≈ 0.028–0.039 and `cos` ≈ 0.9981–0.9988.**

Any DC step-size number at or below ~0.06 is indistinguishable from blindness.
Keep this floor — after the fix it becomes the pre-registered threshold.

### 4.2 Which claims are affected

- **[[../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]]
  — not directly poisoned.** Its runs (`pzmc2orq`, `t4bp8nki`) **predate the
  probe and log no step-size keys at all** (verified: neither appears with any
  `eval_stepsize_*` key in the wandb pull). Its 68× consistency-loss claim is
  untouched by this bug. **But its "Next" item — *"Get `eval_stepsize_effect_rel`
  on these (resume with updated code…)"* — would, if executed today on the DC
  arm, produce a contaminated number.** That instruction needs the fix first.
- **[[2026-08-01-input-blindness-audit]] §2 (step-size row), §4 items 3–4, §6.**
  Its *conclusions* stand and are in fact strengthened — "D3 currently has zero
  evidence that its adapters are step-size conditioned" is correct. Only the
  §4-item-4 mechanism (`_needs verification_`) and its `model.eval()` premise
  need replacing with this note.
- **Nothing in `70_Thesis/` or `60_Updates/` cites a step-size effect number**
  (grep for `stepsize_effect_rel` across the vault returns tickets, the audit,
  and experiment notes only). **No thesis prose is currently poisoned.** This was
  caught before it propagated — the main win here.
- **`hcrnc9gf` (Wan, effect_rel 0.2568, null exactly 0) remains the one
  trustworthy step-size measurement**, and it is still only an 8-step smoke test.

### 4.3 Interaction with the `gate_cap` bug

**Independent for the runs that matter, but they compound where both are
present.**

- The D3 arms A/B config (`diffusion_dc_shortcut_d3arm_actionfree_robotarm.yaml:43`)
  states *"Clean-baseline intent: no gate_cap"* and sets `gate_bias: 0.0` with no
  `gate_cap` (`:64-65`). **Arms A/B are not affected by the gate freeze.** The two
  bugs do not interact there.
- Mechanically they *would* compound under `avid_mask_mix`, where
  `composed = base·gate + adapter·(1−gate)` (`adapted_model.py:282`): a gate
  frozen high pins a larger share of the composed output to the base, which is
  precisely the tensor carrying the dropout noise — inflating the base-noise
  share of `effect_rel`. So on any DC run with both a frozen gate and train-mode
  dropout, `effect_rel` is contaminated *more*, not less.
- On Wan the interaction is moot: no dropout ⇒ null 0 regardless of the gate.
  `pzmc2orq`'s frozen gate is a real confound for the *curvature* claim
  (already documented) but not for step-size, which it never logged.
- A `ValueError` guard against `gate_cap <= σ(gate_bias)` is already in place at
  `adapted_model.py:82-91`, so that bug cannot recur in new runs.

---

## 5. Proposed fix (described, **not applied**)

### 5.1 Primary — make the probe deterministic (mirror the action probe)

In `src/generative_flow_adapters/training/trainer.py`, `_stepsize_sensitivity_eval`
(`:703-775`). Two additions, both copied from the probe that already gets this
right (`evaluation/action_sensitivity.py:314, 328, 354, 382`):

```diff
     @torch.no_grad()
     def _stepsize_sensitivity_eval(self, batches):
         try:
             if not getattr(self.model, "supports_return_base", False):
                 return {}
             ...
             ref = min(levels); others = [lv for lv in levels if lv != ref]
             effect_rels, cosines, null_viol = [], [], 0.0
+            # The frozen base carries nn.Dropout(p=0.1) on every ResBlock
+            # (openaimodel3d.py:197) plus a hard-coded 0.1 TemporalConvBlock
+            # (:210). This probe is invoked from _evaluate AFTER the
+            # `finally: self.model.train(was_training)` at :617, so without this
+            # the base is in TRAIN mode and every forward draws a new mask.
+            was_training = self.model.training
+            self.model.eval()
+            try:
                 for batch_index, batch in enumerate(batches):
                     _l, _c, x_t, t, cond, _p, _b = self._forward_and_loss(batch)
                     n = int(x_t.shape[0])
+                    draw_seed = 1234 + batch_index
                     def _run(lv):
                         c = self._inject_step_level(...)
-                        with self._autocast():
-                            p, b = self.model(x_t, t, c, return_base=True)
+                        # Identical RNG state per level, so any residual
+                        # stochasticity is paired rather than differenced.
+                        with torch.random.fork_rng(devices=_rng_devices()), self._autocast():
+                            torch.manual_seed(draw_seed)
+                            p, b = self.model(x_t, t, c, return_base=True)
                         return p.float(), (b.float() if isinstance(b, Tensor) else None)
                     ...
+            finally:
+                self.model.train(was_training)
```

`_rng_devices` is already defined in `evaluation/action_sensitivity.py`; import
it, or inline the two-line equivalent.

**Do NOT "fix" this by caching the base once and passing `base_output=` to the
later levels.** That would make `null_viol` compare a tensor to itself — it would
report 0 *by construction* and silently destroy the only control that can detect
a genuine future conditioning leak. Keep the second real base forward; just make
it deterministic. (Reusing the base is right for *speed* in training, which is
why `:356`/`:414` do it — but it is wrong for a null control.)

### 5.2 Secondary — refuse to emit a number the control invalidates

Still in `_stepsize_sensitivity_eval`, at the output block (`:768-775`):

```diff
     out["eval_stepsize_base_null_violation"] = float(null_viol)
+    # Hard gate: a violated null means the paired forwards were not paired, so
+    # effect_rel is meaningless. Emitting it invites it onto a plot.
+    if null_viol > 1e-3:
+        out.pop("eval_stepsize_effect_rel", None)
+        out.pop("eval_stepsize_effect_rel_mean", None)
+        out.pop("eval_stepsize_cos", None)
+        if not self._stepsize_probe_warned:
+            self._stepsize_probe_warned = True
+            print(f"[stepsize-probe] null control violated ({null_viol:.3e} > 1e-3); "
+                  "effect_rel suppressed — the base forward is not deterministic.")
```

This implements the audit's own gate (*"refuse to interpret effect_rel unless
null < 1e-3"*) in code rather than in a reader's discipline.

### 5.3 Structural hardening (optional, recommended)

Override `train()` on the frozen base so it can never silently re-enter train
mode — in `models/base/interfaces.py` next to `freeze()` (`:18-22`), set a
`self._frozen = True` flag and have `train(mode)` no-op (forcing `mode=False`)
once frozen. This fixes the whole class of bug rather than this one probe, and
would have prevented it. It changes behaviour for every backbone, so it deserves
its own decision note before being applied.

### 5.4 The two-minute confirmation before applying anything

```python
# on any DC ACWM checkpoint, GPU
x_t, t, cond = <one preprocessed probe batch>
for mode in ("train", "eval"):
    model.train() if mode == "train" else model.eval()
    with torch.no_grad(), trainer._autocast():
        _, b1 = model(x_t, t, cond, return_base=True)
        _, b2 = model(x_t, t, cond, return_base=True)
    print(mode, "max|Δ| =", (b2 - b1).abs().max().item(),
          "rel =", ((b2-b1).norm()/b1.norm()).item())
```

Prediction: `train` → rel ≈ 0.03; `eval` → **exactly 0.0**.

---

## 6. Minimal reproduction (CPU-only, **it ran**)

Using the **real vendored `ResBlock`** from
`backbones/dynamicrafter/modules/networks/openaimodel3d.py` — 12 blocks, 32
channels, `dropout=0.1`, `use_temporal_conv=True`; the `zero_module` output convs
are re-initialised to non-zero to emulate a *pretrained* (rather than
identity-at-init) base. Two successive `no_grad` forwards on identical input:

```
 eval mode, 12 ResBlocks: base_null_violation = 0.000000
train mode, 12 ResBlocks: base_null_violation = 0.734353
```

Sweeping the residual-branch strength (the only free parameter, since I have no
real checkpoint):

| residual init std | null_viol (train) | null_viol (eval) |
|---|---|---|
| 0.002 | 0.046321 | **0.000000000** |
| 0.005 | 0.114922 | **0.000000000** |
| 0.01  | 0.223855 | **0.000000000** |
| 0.02  | 0.409767 | **0.000000000** |
| 0.05  | 0.733024 | **0.000000000** |

Two conclusions:

1. **Eval mode gives an exact zero at every setting** — matching Wan's `hcrnc9gf`
   and the action probe. The fix provably produces the required 0.
2. Even the *weakest* residual branch tested already exceeds the observed
   0.028–0.039, and the real DC512 UNet has far more than 12 blocks. The observed
   magnitude is comfortably *inside* the range train-mode dropout produces — it
   needs no additional mechanism. (This is a plausibility bound, not a
   quantitative match: without the real checkpoint the residual scale is a free
   parameter. The exact-zero result in eval mode is the load-bearing half.)

Plus the concentration check quoted in §2.4 (n=12 draws, spread 0.23%).

Scripts (scratch, outside the vault and the repo):
`/tmp/claude-1000/-home-lukas-projects-thesis-vault/0afaf6a4-196e-4e7e-8349-f1004799eb92/scratchpad/repro.py`,
`repro2.py`.

---

## 7. Urgency call — jobs 25141979 / 25141980

# **Do not kill them. Let them run.**

Reasoning:

1. **Training is not affected.** Dropout in the base during the *training*
   forward is ordinary regularisation, and it is applied identically to every
   arm. Critically, the shortcut consistency **target is not corrupted**: the
   training path computes the base **once** and reuses it across the paired
   step-size forwards (`trainer.py:356`, `:414`, via
   `AdaptedModel.forward(base_output=…)`, `adapted_model.py:154-181`). The
   probe at `:748` is the *only* paired site that recomputes. So the bug is
   confined to a diagnostic readout.
2. **The arms' primary measurement is untouched.** Per
   [[../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]],
   A/B are decided by **few-step quality per N** (FID/FVD/PSNR/LPIPS and the
   `eval_step_grid` videos over N ∈ {1,2,4,8,25,50}) — not by
   `eval_stepsize_effect_rel`. Every generation/quality path *does* set
   `self.model.eval()` (`trainer.py:1122`, `:1375`, `:1495`, `:1578`, `:1968`).
   **The headline A-vs-B comparison is valid as-is.**
3. **The affected number is recoverable offline for free.** `eval_stepsize_*` can
   be recomputed from a retained checkpoint after the fix, with no retraining —
   the probe needs only one batch and a forward. Both arms checkpoint regularly
   (`checkpoint_every_n_steps: 200`).
4. **Killing costs real GPU-days** and loses the quality-vs-N curve, which is the
   deliverable D3 actually lacks. Relaunching buys only a diagnostic scalar that
   a 5-minute offline probe recovers.
5. **Arms A/B carry no `gate_cap`** (config `:43`, `:64-65`), so the gate-freeze
   bug gives no independent reason to relaunch either.

**So:**

- **Let 25141979 / 25141980 run to completion.** Read them for few-step quality.
- **Treat any `eval_stepsize_effect_rel` / `_mean` / `_cos` they emit as VOID.**
  Both are DC, both inherit the ~0.05 noise floor of §4.1. Expect their null to
  land in 0.028–0.039; that is the tell, not a new finding.
- **Apply the §5 fix now**, so runs launched after tonight are clean.
- **Then re-derive the step-size number offline** from an A/B checkpoint with the
  fixed probe. Only accept it if `null < 1e-3`, and only call it evidence of
  step-size conditioning if `effect_rel` clears the §4.1 floor (~0.06) by a clear
  margin — `hcrnc9gf`'s 0.2568 shows that margin is achievable.
- **One risk to check:** `keep_last_checkpoints` retention on the d3arm config —
  confirm a checkpoint survives to run the offline probe against.
- **Arm C (25141988)** is Wan-side (`gate_cap: 0.9`,
  `diffusion_wan22_shortcut_actionfree_robotarm.yaml:62, :68`) and has **no
  dropout in its base** — its step-size probe is already trustworthy. Unaffected.

---

## Related

- [[2026-08-01-input-blindness-audit]] — §4 item 4 is the open question this closes; its `model.eval()` premise is corrected in §1
- [[../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]] — its "Next" step needs the fix before it is executed on DC
- [[../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]] — jobs 25141979 / 25141980 / 25141988
- [[../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]] — the other D3 confound; independent for arms A/B (§4.3)
