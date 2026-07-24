---
date: 2026-07-03
tags: [wan2.2, cfg, text-conditioning, adapter, generation, optimization]
---

# Wan2.2 text prompt injection + CFG-forward memoization

Two related changes to the plug-and-play Wan2.2 stack (`WanTI2VVideoModel` +
`AdaptedModel`). Motivation: injecting a task text prompt and generating with
`guide_scale=5` makes the base videos look **much better** — because in
frame-only mode the two CFG branches were identical, so `5·(cond−uncond)` was
`5·0` (guidance was inert). A real prompt makes `cond ≠ uncond`, so CFG actually
steers toward the task.

## 1. CFG double-forward memoization
Wan's loop calls the DiT twice per denoise step (positive then negative context)
and combines `uncond + g·(cond − uncond)`. In frame-only / prompt-free mode both
branches use the same cached unconditional embedding → identical forward → the
second is pure waste.

`_ComposedDiT` (the stand-in for `wan.model`) now holds a 1-entry cache keyed on
`(timestep value, id(x[0]), id(context[0]))`. Identical call within a step → hit
→ the frozen base **and** the adapter run once, not twice (~2× fewer 5B forwards
per step). Real CFG (positive ≠ negative context) → different `id(context[0])` →
miss → both run, unaffected. `generate()` now **always** installs the wrapper, so
base-only rollouts get the speedup too. Correctness relies on eval/no_grad
determinism. Regression test: `tests/test_wan_composed_dit_memoize.py` (4 cases,
CPU, no Wan/CUDA).

## 2. Text prompt injection (no T5 at train or inference)
Design: config points at a prompts file; `None` → frame-only (unchanged).

- **Offline** `external_repos/Wan2.2/precompute_prompt_contexts.py`: reads a
  prompts YAML (`{default, negative, tasks: {task_name: prompt}}`), T5-encodes
  each once → `<stem>.contexts.pt` table `{positive: {task_name: [L,C], __default__},
  negative, uncond, prompts, text_len}`. Negative defaults to the model's
  `sample_neg_prompt` (chosen so `g=5` reproduces the "better" base result).
- **Training** (`WanBatchPreprocessConfig.prompt_contexts_path`): the new
  `PromptContextProvider` maps each clip's `task_name` → cached embedding and sets
  `cond["context"]` (a list of per-sample `[Lᵢ,C]`). `denoise` already forwards
  `cond["context"]`; base is now conditioned on the task prompt, adapter learns
  its delta on top. Required a one-line `task_name` emit in `MetaWorldTranslator._base_clip`.
  Config field: `model.extra.text_prompts_file`; training script derives the
  `.contexts.pt` path and errors if missing.
- **Inference** (`generate(context=, context_null=, guide_scale=)`): a tiny
  `_CachedT5` stub returns cached embeddings by string so upstream's own
  positive/negative CFG loop runs at `g=5` **without loading T5**. Example
  `examples/wan22_generate_cond_frames.py` now generates unconditional vs
  conditional (task prompt + CFG) for base and adapted, and reports
  `mean|cond−uncond|`.

## 2b. Prompt augmentation — a *set* of prompts per task (2026-07-03)
Each `tasks:` entry (and `default`) may now be a **single string OR a list of
paraphrases**. `precompute_prompt_contexts.py` encodes each and stores
`positive[task_name]` as a single `[L,C]` tensor (str) or a **list of `[Lᵢ,C]`**
(list). `PromptContextProvider.contexts_for(..., sample=train)` picks a **random
paraphrase per clip during training** (augmentation) and the **first
deterministically at eval**. `train` is threaded `__call__ → _build_condition →
contexts_for` (previously `del`-ed). Backward compatible: existing single-string
`.contexts.pt` files load unchanged (values are plain tensors). Sampling uses
`random.randrange`. Tested CPU: single fixed, list eval=first, list train=varies,
unknown→`__default__`.

## 2c. Task-INDEPENDENT prompt pool (2026-07-03)
Simpler mode requested: one flat set of prompts sampled per clip, no task_name.
YAML now takes a top-level `prompts:` list; precompute builds `table["pool"] =
[default] + prompts` (encoded), `default` first = deterministic eval choice.
`PromptContextProvider.pool_mode` (pool present → takes precedence over `positive`)
makes `contexts_for(batch_size, sample=train)` sample the shared pool per clip —
`_build_condition` passes the **batch size**, not task_names, so **no task_name is
required** in pool mode. Task mode (2b) still works when only `tasks:` is given.
YAML `configs/prompts/metaworld_tasks.yaml` rewritten to the pool form. Tested CPU:
pool mode needs no task_name, train samples the pool, eval returns default (#0).

## Consistency note / knob
Because the AVID adapter has `condition_on_base_outputs: true`, at inference it's
injected per-CFG-branch, so composition is `CFG(base) + CFG(adapter_delta)` — the
adapter delta is also `g`-amplified (can't apply it once outside CFG without
reimplementing the combine step). Fine for a first cut; watch when tuning `g`.
Training uses a single (un-guided) base forward `base(pos)`; inference base is
`CFG(base)` — an inherent CFG-adapter train/inference gap, accepted for now
(future knob: train against the guided base).

Validated on CPU (provider mapping, fallback-to-`__default__`, frame-only
no-context fallback, stub, memoization); 46 existing tests still pass. GPU manual
test pending via the example script. Related: [[basevideomodel-external-repo-design]].

## 3. Native eval grid + metrics (2026-07-03)
The eval grid + quality metrics were built with the **retired** `FlowInferenceSampler`
(reimplemented rectified-flow ODE) — no `guide_scale`, wrong `shift` (3.0 vs Wan
5.0), never the native loop. So eval scored washed-out nonsense and couldn't show
the prompt+CFG improvement.

Fix: a **native eval path** in `trainer.py` that runs the frozen base's own i2v
`generate` (adapted via `AdaptedModel.generate`, base via `base_model.generate`),
producing **pixels**, gated on the base being a `BaseVideoModel` with `.generate`
(legacy sampler kept for Diffusion/DynamiCrafter). Key methods: `_native_eval_grid`
(shortcut step-size grid at `inference_num_steps`, high), `_native_quality_eval`
(metrics at `quality_eval_num_steps`, low), `_native_clip_rollout`,
`_native_batch_conditions`. Both run on **held-out eval batches** (they need the
raw obs frame — the preprocessed batch only keeps latents; a CPU test caught me
reading `video` from the wrong batch). `_maybe_generate_samples` + `_run_quality_eval`
gate to native when available. New `wandb_logger.log_step_size_grid_pixels`
(pixel twin of `log_step_size_grid`, no decode).

Config knobs (`training.extra`, in YAML): `inference_guide_scale` (5.0),
`inference_shift` (5.0), `inference_use_prompt`, `inference_frame_num`,
`inference_seed`. Training script injects `inference_max_area`,
`inference_temporal_length`, `inference_prompt_contexts_path` from what it already
computes. Step budgets: grid = `inference_num_steps` (looks great), metrics =
`quality_eval_num_steps` (cheap) — the user's ask. Full 46-test suite + a fake-model
CPU test of the grid/metrics/gating/geometry pass.
