---
type: tech
status: living
last_updated: 2026-08-01
sources: ["[[avid-vs-ours-action-conditioning]]", "[[../experiments/20260801-wan-rt1-indistribution-plateau]]", "[[../experiments/20260731-wan-action-signal-is-a-global-bag]]", "[[../experiments/20260731-wan-action-trace-value-pathway-drowns]]"]
---

# AVID vs ours — Wan action-conditioning path

Sibling of [[avid-vs-ours-action-conditioning]] (which covers the DynamiCrafter
branch). Prompted 2026-08-01 by the RT-1 result: our Wan adapter settles at
`action_effect_rel ≈ 0.021` ([[../experiments/20260801-wan-rt1-indistribution-plateau]],
`5w72bo01`) against the AVID reference **0.0495** on the same dataset
(`93qrvr5v`). Everything below is **read off source**, file:line given; nothing
inferred from run behaviour.

> ⚠️ **Interpretations not yet reviewed by Lukas** (written during the overnight
> session of 2026-08-01/02). The *facts* are code-sourced; the *ranking* of which
> divergence matters most is analysis, not measurement. The clean-room run
> [[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]] is what
> turns it into evidence.

## The five verified divergences

### 1. On the RT-1 run, the action never touches AdaLN at all

`configs/wan22/diffusion_wan22_action_rt1_tokennorm_nobase.yaml:` sets
`action_injection: cross_attention`. In the adapter,
`backbones/wan/modules/action_model.py:102`:

```python
self._use_adaln_action = self.action_injection in ("adaln", "both")   # -> False
```

and `:263-265`:

```python
adaln_cond = cond_embedding if self._use_adaln_action else None
e = self._conditioning_embedding(t, adaln_cond, step_level)
```

With `adaln_cond=None` and `cond_proj` present, `_conditioning_embedding`
(`:213-220`) takes the *else* branch and adds `self.null_cond_emb` — a **learned
constant**. So on this run the modulation of every block is
`time_emb + const`, and the action reaches the network **only** as
cross-attention context tokens.

AVID does the opposite: the action *is* the modulation
(`openaimodel3d.py:744-747`), and there is no action cross-attention at all.

### 2. Even in `adaln` mode, our conditioning is a temporal mean

`action_model.py:216-217`:

```python
if ce.dim() == 3:            # [B, T, cond_dim] -> pool over time
    ce = ce.mean(dim=1)
```

So the preprocessor's `action_per_frame: true` (which overrides `cond["action"]`
with the per-frame `[B, L, A]` sequence, `data/wan_batch_preprocessor.py:469-470`)
is **undone inside the adapter**: the per-frame structure is averaged away before
it reaches the modulation. `adaln` + `action_per_frame` ≈ `action_aggregation: mean`.

### 3. Our AdaLN modulation is global; AVID's is per-frame

`action_model.py:266`: `e0 = self.time_projection(e).unflatten(1, (6, self.dim))`
→ `[B, 6, dim]`. The block we build on
(`backbones/wan/modules/model.py`, the **Wan2.1**-style `WanAttentionBlock`) does
`e = (self.modulation + e).chunk(6, dim=1)`, giving `e[k]` of shape `[B, 1, dim]`
— **broadcast identically over every token**: every latent frame, every spatial
position.

AVID's `emb` is built per `(b, t)` and its UNet runs in `(b t) c h w` layout, so
each frame carries its own scale/shift (`openaimodel3d.py:733-747`).

**Worth noting:** the *official Wan2.2* DiT already supports per-token modulation
— `external_repos/Wan2.2/wan/modules/model.py:239` takes `e` of shape
`[B, L, 6, C]`. We built the adapter on the 2.1-style block, which structurally
cannot express per-frame conditioning. This is the cheapest thing to change.

### 4. Add-into-time, not concat-into-a-reserved-half

`action_model.py:218`: `e = e + self.cond_proj(ce)` — the same add-vs-concat
divergence documented for DC in [[avid-vs-ours-action-conditioning]] §Headline,
present again on the Wan side. AVID reserves `dim//2` for the action so the
timestep cannot occupy it (`openaimodel3d.py:419-434`).

### 5. No structural action↔frame correspondence for the xattn tokens

`action_seq` is a passthrough of the per-**pixel**-frame actions
(`data/wan_batch_preprocessor.py:514-517`; `action_seq_len` unset → `L = T`), so
the RT-1 config hands **17 action tokens** to a DiT operating on **5 latent
frames**. A learned `action_pos_emb` is added (`action_model.py:283`) but
cross-attention is otherwise permutation-invariant: nothing ties token `j` to
latent frame `j // 4`. The correspondence has to be *learned*, and the 07-31
probe measured temporal alignment **0.25 = chance**
([[../experiments/20260731-wan-action-signal-is-a-global-bag]]).

## The substrate difference underneath all of this

| | DynamiCrafter (AVID) | Wan2.2 TI2V-5B (ours) |
|---|---|---|
| first stage | 2D VAE, `perframe_ae: True` (`act_cond_diffusion_11M.yaml:49`) | 3D VAE, `temperal_downsample=[False, True, True]` (`vae2_2.py:896`) |
| temporal compression | **none** — 16 frames → 16 latent frames | **4x** — 17 frames → 5 latent frames |
| action ↔ latent frame | **1:1 by construction** | requires explicit binning |

AVID gets the alignment for free. Any faithful Wan port must **sum the actions
within each latent frame's window** before conditioning — otherwise there is no
correspondence to learn. Our pipeline never did this in the AdaLN path (§2
averages it away) and only implicitly in the xattn path (§5).

## Direct measurement: per-frame addressability at initialisation

Run 2026-08-01, `scripts/diagnose_action_frame_localisation.py` (in
`generative-flow-adapters`). No training, no checkpoint, no GPU — it perturbs the
actions of the pixel frames feeding **latent frame 2** and reads
`‖Δpred‖/‖pred‖` for every latent frame. Both models use their real
composition setting (AVID `learnt_mask` / ours `mask_mix`), so the prediction
head is live in both.

| architecture | f0 | f1 | **f2** | f3 | f4 | max/min |
|---|---|---|---|---|---|---|
| AVID-faithful (clean-room) | 0.0003 | 0.0003 | **0.0693** | 0.0003 | 0.0003 | **261x** |
| ours — `cross_attention` (the RT-1 run's setting) | 0.2468 | 0.2446 | 0.2453 | 0.2449 | 0.2469 | **1.0x** |
| ours — `adaln` + `action_per_frame` | 0.1739 | 0.1754 | 0.1743 | 0.1735 | 0.1745 | **1.0x** |

Two things follow, and the second is the more interesting one:

1. **Both of our variants are *perfectly* uniform** at init. The strength of the
   claim differs by path, and the distinction matters:
   - **`adaln` (and `both`): structurally impossible.** `ce.mean(dim=1)` destroys
     the per-frame information before it is ever embedded, and the resulting
     `[B, 6, dim]` modulation has no frame axis to put it back on. No amount of
     training recovers it.
   - **`cross_attention`: possible in principle, but must be *learned*.** Latent
     tokens know their frame through RoPE and the action tokens carry
     `action_pos_emb`, so attention *could* discover "token `j` belongs to latent
     frame `j // 4`". It is simply not given — and the 07-31 probe shows that
     after training it had **not** been learned (temporal alignment 0.25 =
     chance). AVID never has to learn it: the correspondence is structural.

   So the honest statement is: one of our two paths cannot express per-frame
   conditioning at all, and the other has to discover a correspondence that
   AVID gets for free — and empirically didn't.
2. **Ours are *more* action-sensitive in total** (0.245 vs 0.069) — roughly 3.5x.
   The problem was never that the action signal is too weak; it is that the
   signal is **unaddressed**. That is exactly the "sensitivity without control"
   signature recorded in
   [[../experiments/20260731-wan-action-signal-is-a-global-bag]] (steering cos
   0.00 with non-zero effect_rel), and it explains why scale fixes
   (`action_token_norm`, 6–10x) moved the number without unlocking control: they
   amplified a signal that has nowhere to land.

## Ranking (analysis, not measurement)

Ordered by expected effect on `action_effect_rel`, given the 07-31 evidence that
the signal *survives cross-attention and then drowns at the residual add*
([[../experiments/20260731-wan-action-trace-value-pathway-drowns]]):

1. **§1 + §3 together** — the action is confined to the residual stream (where it
   is 2 orders of magnitude below the stream RMS) instead of driving the
   normalised scale/shift, which is scale-free by construction. Highest expected
   effect; also explains why `action_token_norm` bought 6–10x but not an order of
   magnitude.
2. **§5** — a correspondence the model must discover rather than one it is given.
3. **§4** — the pedestal mechanism already measured on the DC side.
4. **§2** — only bites in `adaln`/`both` modes, which the headline RT-1 run did
   not use.

## What settles it

[[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]] — AVID's
recipe, AVID's data loader, AVID's training loop, Wan's base. All five
divergences above are removed at once; if `effect_rel` lands near 0.0495 the
fault is ours, if it lands near 0.021 the Wan latent space is the harder
substrate.

## Related

- [[avid-vs-ours-action-conditioning]] — the DC-branch sibling
- [[../experiments/20260801-wan-rt1-indistribution-plateau]] — the 0.021 plateau
- [[../experiments/20260729-avid-rt1-follows-actions-control]] — the 0.0495 reference
