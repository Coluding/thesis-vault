---
date: 2026-07-02
category: changed
deliverable: D2
meeting:
sources:
  - "[[2026-07-01-wan22-cond-frames-generation-script]]"
  - "[[wan22-i2v-diffusion-forcing]]"
---

# Root-caused washed-out Wan2.2 videos → BaseVideoModel plug-and-play redesign

## What went wrong (diagnosis)

The Wan2.2 cond-frame generations were garbage (sharp clamped observation
frames, predicted frames collapsing to washed-out/magenta mush). Ruled out:
- **Weights**: 100% key overlap (825/825), DiT fully loaded.
- **VAE**: ground-truth decodes perfectly.
- **Diffusion-forcing mechanism**: our clamp+timestep-0 matches upstream.

The cause was our **reimplemented sampling loop** (`FlowInferenceSampler` +
`Wan22DiffusionForcingPreprocessor`) diverging from Wan's native pipeline on
every axis at once: 256px (Wan native ~720p; config comment even says "looks
washed at 256px"), null zero-context instead of the precomputed
`uncond_context.pt`, `shift=3` vs native `5`, `guide_scale=1` (no CFG). Fixing
the recipe knobs only changed the *flavour* of garbage — confirming the problem
is that we reimplement the loop at all.

Verified separately: the **upstream `external_repos/wan22` `generate.py`
produces great one-frame-conditioned videos**. So the model is fine; our wiring
was the bug.

## The decision

Stop vendoring backbone code and reimplementing sampling. We only copied Wan /
DynamiCrafter in-tree to reach into layers for hypernetworks / ControlNet — that
line of work is dropped, so the copies are pure liability.

New design: a strict, backbone-agnostic **`BaseVideoModel`** interface
(`models/base/video_model.py`) — `encode`, `decode`, `denoise` (single-step
adapter/training seam), `generate` (native rollout). Backbones are **plug-and-
play**: a `WanTI2VVideoModel` wraps the upstream `wan.WanTI2V` and delegates
`generate` to *its own* loop, so base-only output matches upstream by
construction. DynamiCrafter will implement the same 4 methods later.

**Adapter injection seam**: Wan's loop calls `self.model(latent, t, context,
seq_len)` each step. We temporarily swap `wan.model` for a wrapper returning
`base_pred + adapter(x,t,cond)`. Under Wan's CFG this composes additively
(`uncond+δ + g·((cond+δ)−(uncond+δ)) == base+δ`). `adapter=None` ⇒ untouched
native rollout.

## Scope / rollout (agreed with Lukas)

- **Incremental**: new external-repo-backed wrapper added *alongside* the
  vendored `backbones/wan`; training path untouched until generation is
  validated, then migrate. Not deleting the vendored tree yet.
- **Wan2.2 first**; DynamiCrafter refit to the interface as follow-up.
- External repo made importable via a `sys.path` shim (not `pip install -e`,
  which would churn the venv's `flash_attn`/`numpy<2` pins); `import wan`
  already resolves.
- Frame-only conditioning uses `skip_text_encoder=True` + `uncond_context.pt`
  (already precomputed in the ckpt dir) — no prompt, no CLIP.

## New files

- `src/generative_flow_adapters/models/base/video_model.py` — interface.
- `src/generative_flow_adapters/models/base/wan_ti2v.py` — Wan2.2 impl.
- `examples/wan22_base_vs_adapted_generation.py` — base-only good-video check +
  zero-delta `ProbeAdapter` seam check (adapted must equal base; call count
  proves the seam fired).

## Open / next

- Validate base-only video quality (running) + adapted==base seam check.
- Reconcile the adapter's expected `(x_t, t, cond)` (per-token `t` `[1,seq_len]`
  from Wan's loop) with our action adapters before training through the seam.
- Migrate the trainer to drive `BaseVideoModel.generate` for eval panels and
  retire `FlowInferenceSampler` for Wan.
