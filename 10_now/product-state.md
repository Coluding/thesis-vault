---
last_updated: 2026-05-15
status: pre-results
---

# Experiment State

> What experiments have **actually run** and what came out. Not roadmap, not
> vision — the literal current state. AI overwrites this when runs finish
> or get killed.
>
> **Status: pre-results.** The framework is mostly built ([[architecture]])
> but the thesis lacks evidence-backed numbers across the four
> deliverables. This doc is deliberately short until D2 has its first
> ablation result. Per [[../CLAUDE]] hard rule 8: do not list a result as
> measured without citing a real run.

## What "exists" (codebase / configs)

The configs in `configs/` enumerate **planned experiments**, not finished
ones. They are the most concrete inventory of what the framework can run.
Grouped by deliverable:

### D1 — framework (configs that exist to demonstrate the taxonomy)

| Config | Adapter family | Backbone | Notes |
|---|---|---|---|
| `diffusion_lora_action.yaml` | LoRA | diffusers (dummy or HF) | Smallest "action-conditioning works" demo |
| `diffusion_hyper_lora_action.yaml` | Hypernetwork → LoRA | diffusers | Hypernetwork emits LoRA weights |
| `diffusion_multimodal_hyper.yaml` | Hypernetwork | — | Multimodal conditioning smoke test |
| `diffusion_hidden_unicon_decoder.yaml` | Hidden-state (UniCon) | diffusers | UniCon-style decoder hidden-state adapter |
| `diffusion_hidden_replace_decoder.yaml` | Hidden-state | diffusers | Replace-decoder variant |
| `diffusion_hidden_full_skip_controlnet.yaml` | Hidden-state | diffusers | Full-skip controlnet-style |
| `diffusion_output_dynamicrafter.yaml` | Output | DynamiCrafter | Video diffusion + output adapter |
| `diffusion_output_avid_training_test.yaml` | Output | DynamiCrafter (AVID-style) | AVID replication starting point |
| `diffusion_hyperalign_action.yaml` | Hypernetwork (HyperAlign) | diffusers | HyperAlign replication |
| `diffusion_hyperalign_fake_action.yaml` | Hypernetwork (HyperAlign) | diffusers | HyperAlign with synthetic action |
| `diffusion_hyperalign_metaworld.yaml` | Hypernetwork (HyperAlign) | diffusers / MetaWorld | Closest D2 config |
| `opensora_output_adapter.yaml` | Output | OpenSora | OpenSora wired _partial_ |
| `test_dynamicrafter_hyperalign_unet.yaml` | Hypernetwork | DynamiCrafter U-Net | Test-only |
| `test_dynamicrafter_metaworld_unet.yaml` | — | DynamiCrafter U-Net | Test-only |

### D3 — shortcut configs

| Config | Notes |
|---|---|
| `flow_output_shortcut.yaml` | Flow matching + output adapter + shortcut losses |
| `flow_output_shortcut_velocity.yaml` | Flow matching with velocity prediction + shortcut |
| `flow_hyper_shortcut_stepwise.yaml` | Hypernetwork + step-wise shortcut |
| `diffusion_output_shortcut_noise.yaml` | Diffusion (noise pred) + shortcut output adapter |
| `diffusion_output_shortcut_velocity.yaml` | Diffusion (velocity pred) + shortcut output adapter |
| `diffusion_output_dynamicrafter_shortcut_test.yaml` | DynamiCrafter + shortcut adapter (test) |

### D2/D4 — combined / action-conditioned

The clearest entrypoint is `scripts/train_hyperalign_metaworld.py`, paired
with `configs/diffusion_hyperalign_metaworld.yaml`. Working tree is
modified at HEAD — there's in-flight work on this exact path.

## What has actually run

_Needs verification — fill in from wandb / local logs._

| Experiment | Config | Status | wandb run id | ckpt | Result |
|---|---|---|---|---|---|
| _no entries yet_ | | | | | |

Per [[../CLAUDE]] hard rule 8, every row in this table must cite a real
run. Until that happens, this table stays empty rather than fabricated.

Concrete first-write candidates (rows the user can fill from their own
records — each becomes one experiment note under
`30_Knowledge/experiments/`):

- HyperAlign on MetaWorld — the in-flight one in
  `scripts/train_hyperalign_metaworld.py`.
- DynamiCrafter sanity smoke test that produced the now-deleted
  `tests/_outputs/dynamicrafter_sanity/*.png` artefacts.
- Any flow-shortcut config run end-to-end (`flow_output_shortcut.yaml`).
- The video-logging fix run that motivated commits `44b214b` and
  `88e4430`.

## Tests passing today

`pytest` (no CI runner wired). Files:

- `test_hyperalign_architecture.py`
- `test_dynamicrafter_checkpoint_sanity.py`
- `test_dynamicrafter_integration.py`
- `test_hyper_step_size_conditioning.py`
- `test_batch_preprocessor.py`
- `test_metaworld_dataset.py`
- `test_null_caption.py`
- `test_video_logging.py`

Tests cover the architecture / shape / wiring contract for each adapter
family, the data pipeline, and the video logging path. They do **not**
constitute experimental results — they ensure the code runs, not that the
adapters learn anything.

## What's planned

The four-deliverable plan in [[positioning]] is the high-level roadmap.
The concrete experiment backlog lives in `20_Tickets/` (not yet
populated). When experiments are queued, they should appear as
`20_Tickets/exp-{scope}-{slug}.md`.

## How this doc evolves

Trigger points for substantial rewrites:

1. **First real run finishes with logged outputs.** Replace the empty
   "What has actually run" table with one real row. Create a real
   `30_Knowledge/experiments/{slug}.md`. Update [[positioning]] D-status
   from "no evidence yet" to "first evidence in."
2. **D2 ablation across all four adapter families finishes.** Add a
   summary table and a Pareto figure reference. This is the headline D2
   chapter result.
3. **First shortcut few-step rollout curve.** Add a section "Shortcut
   results" with the curve and the comparison to non-shortcut baseline.
4. **D4 combined run.** Final summary section becomes the thesis result.

Per [[../CLAUDE]] hard rule 8: do not write any number into this doc
without citing the run that produced it.

## Related

- [[architecture]] — codebase and what it can do
- [[positioning]] — the four deliverables this evidence has to back
- [[../30_Knowledge/experiments/]] — per-run write-ups (folder to be
  populated)
- [[setup-status]] — vault coverage gaps
