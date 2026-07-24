---
type: feat
scope: adapter
status: open
priority: medium
created: 2026-07-15
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[../50_Decisions/open/action-conditioning-injection-mechanism]]", "[[feat-adapter-wan-action-cross-attention]]"]
---

# feat: per-frame FiLM/AdaLN broadcast in the Wan tiny-DiT adapter

## Why

The biggest single new lever found in the 2026-07-15 AVID-vs-ours structural
comparison ([[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] §"New,
real differences"), personally spot-checked and confirmed.

`adapters/output/wan.py:96-97`:

```python
if t.dim() > 1:
    t = t.flatten(1).amax(dim=1)
```

The Wan tiny-DiT adapter (`ActionWanModel`) collapses WAN's per-latent-frame
diffusion-forcing timestep `[B, T']` down to **one scalar per sample** before
its FiLM/AdaLN modulation, which is then broadcast identically to every
frame's tokens (`action_model.py`'s `unflatten(1,(6,dim))`). The **frozen
base** gets genuine per-token AdaLN (`model2_2.py:462-472`). This is an
existing, deliberately-commented simplification already in the code (not
introduced this session — the comment explains it as a known design choice),
not a bug per se, but a real architectural limitation.

**This is distinct from the already-fixed action-token-binning bug.** Even
with `action_seq_len` correctly pinned to the latent frame count (per-frame
action *tokens* properly aligned for the cross-attention path), the
adapter's AdaLN/FiLM conditioning pathway has **no mechanism at all** to
express "predict frame 5 differently from frame 9" — timestep, step-level,
and (in the default AdaLN-only injection mode) the action itself all collapse
to one global vector. AVID cannot have this asymmetry — its whole paradigm is
single-global-timestep diffusion, so base and adapter always see the
identical scalar `t` by construction; there's no equivalent "collapse" because
there's nothing to collapse.

Diffusion forcing's entire point is per-frame heterogeneity (clean
observation frame at t=0, noised future frames at varying σ) — an adapter
that can't condition on *which* frame it's predicting is fighting the base's
own training paradigm.

## What (rough — needs real design work, not a config flag)

Reshape `ActionWanModel`'s FiLM path so the per-frame `e0` modulation vector
broadcasts only to that frame's own tokens, mirroring how AVID's UNet achieves
per-frame conditioning "for free" via its `(b t)` batch-flattening trick
throughout the whole network (see `backbones/dynamicrafter/modules/networks/openaimodel3d.py`'s
`rearrange(x, "b c t h w -> (b t) c h w")` pattern, already used for the DC-UNet
adapter path this session). The Wan DiT is a sequence-transformer, not a
per-frame 2D-conv UNet, so the analogous reshape is less direct — this needs
real design work on how `ActionWanModel`'s attention blocks currently handle
the temporal axis before committing to an implementation.

**Cheaper diagnostic short of a full architecture change:** pass a per-frame
`frame_mask`/timestep signal into the adapter's conditioning dict explicitly
and see whether even a partial per-frame signal (e.g. via the existing
cross-attention path, once binning is fixed) changes anything, before
investing in the AdaLN reshape.

## Sequencing

Per the diagnosis note's do-now order: this is item 6, **sequenced last**
among the concrete fixes — it's the costliest to implement, and running it
before the cheaper items (gate_bias, grad accumulation, warmup) would
conflate multiple changes and make results hard to attribute.

## Related

- [[feat-adapter-wan-action-cross-attention]] — the cross-attention arm,
  which DOES get genuine per-frame tokens once `action_seq_len` is fixed, but
  is a separate injection pathway from the AdaLN one described here.
- [[../50_Decisions/open/action-conditioning-injection-mechanism]] — the
  decision this feeds into.
