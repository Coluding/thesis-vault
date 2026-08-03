---
type: feat
scope: adapter
status: open
priority: medium
created: 2026-07-15
updated: 2026-08-01
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

## Cleanup 2026-08-01 — **OBSOLETE**

Injection-mechanism variants are no longer the open question.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*

## ⚠️ 2026-08-02 — evidence AGAINST closing this. Please decide.

The overnight session ([[../30_Knowledge/sessions/2026-08-01-avid-wan-cleanroom-build]])
measured the cost of exactly the limitation this ticket describes, and it is
large. **Recommendation: do not close — reopen at high priority.**

**The measurement** (`scripts/diagnose_action_frame_localisation.py`, at
initialisation, CPU, no checkpoint): perturb the actions feeding latent frame 2,
read `‖Δpred‖/‖pred‖` per latent frame.

| architecture | f0 | f1 | **f2** | f3 | f4 | max/min |
|---|---|---|---|---|---|---|
| AVID-faithful per-frame conditioning | 0.0003 | 0.0003 | **0.0693** | 0.0003 | 0.0003 | **261x** |
| ours, `cross_attention` (the RT-1 run) | 0.2468 | 0.2446 | 0.2453 | 0.2449 | 0.2469 | **1.0x** |
| ours, `adaln` + `action_per_frame` | 0.1739 | 0.1754 | 0.1743 | 0.1735 | 0.1745 | **1.0x** |

Three reasons this reverses the "obsolete" call:

1. **It is not an injection-mechanism variant.** Both of our injection
   mechanisms fail *identically* (1.0x). That is why swapping between them
   never helped — the ticket's subject is the axis those variants were varying
   *around*, not one of the variants.
2. **It explains the residual puzzle.** Our adapters are **3.5x more**
   action-sensitive in total than the faithful one (0.245 vs 0.069) yet show no
   control. "Sensitivity without control"
   ([[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]])
   is precisely what an unaddressed signal looks like, and it explains why
   `action_token_norm` moved the number 6–10x without unlocking steering.
3. **It is cheaper to fix than this ticket assumed.** The ticket says the
   AdaLN reshape "needs real design work". It does not: the **official Wan2.2
   DiT already takes per-token modulation** — `e` of shape `[B, L, 6, C]`
   (`external_repos/Wan2.2/wan/modules/model.py:239`). Our `ActionWanModel` is
   built on the *Wan2.1*-style block (`[B, 6, C]`), which is why it cannot
   express this. A working reference implementation now exists in
   `external_repos/avid/wan_diffusion/src/wdwma/models/action_wan.py`.

Caveat on strength: for the **`adaln`** path the limitation is absolute
(`ce.mean(dim=1)` destroys the information). For **`cross_attention`** it is
learnable in principle (RoPE + `action_pos_emb`) but was empirically not learned
(alignment 0.25 = chance).

Confirmation still pending from the clean-room run
([[experiments/exp-adapter-avid-wan-cleanroom-rt1]]).
