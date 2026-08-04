---
type: bug
scope: infra
status: open
priority: high
created: 2026-08-04
updated: 2026-08-04
resolution:
resolution_note:
closed_at:
related: ["[[experiments/exp-backbone-flow-without-temporal-vae-compression]]", "[[../00_Inbox/2026-08-03-diffusion-i2v-backbone-candidates]]"]
---

# bug: `from_pretrained` silently discarded 20 VAE weight tensors

> **⚠⚠ SECOND CORRECTION 2026-08-04 — the NaN cause is now FOUND, and this ticket's
> scope was too narrow. The same silent-drop bug also hits the TRANSFORMER, where it
> is fatal.**
>
> diffusers 0.39.0's `EasyAnimateTransformer3DModel` expects **fused joint attention**
> (`attn1.add_{q,k,v}_proj`, `to_add_out`, `norm_added_{q,k}`). **Both** V5 and V5.1
> checkpoints ship **separate cross-attention** (`attn2.*`). Measured by comparing
> checkpoint key-sets to model key-sets, and confirmed independently through
> `from_pretrained`:
>
> ```
> V5   x diffusers 0.39.0 : 216 checkpoint tensors UNUSED, 216 weights RANDOM
> V5.1 x diffusers 0.39.0 : 216 checkpoint tensors UNUSED, 216 weights RANDOM
> ```
>
> `from_pretrained` warned and returned a 7B model that was **substantially randomly
> initialised**. That is the NaN. The correct loader is the checkpoint's own declared
> class from the EasyAnimate repo (`easyanimate/models/transformer3d.py:1347`), not
> diffusers'.
>
> **A statement in the "Consequence" section below was FALSE and load-bearing:** it
> claimed "the transformers both load cleanly ... so the diffusion-vs-flow comparison
> remains at the transformer level." They do not load cleanly. That false claim was the
> stated reason the experiment was judged to survive the VAE problem. Struck through
> below.
>
> **Note it is NOT a V5-vs-V5.1 split.** An earlier hypothesis that "diffusers supports
> V5.1, not V5" was falsified by the key-set audit — diffusers 0.39.0 matches *neither*.
> Symmetry is therefore restored for the experiment: both arms must use the EA repo
> class, so one loader still serves both.
>
> ---
>
> **⚠ CORRECTION 2026-08-04 — this [the VAE issue] is a real bug but it is NOT the cause of the
> transformer NaN.** Round-trip on a real ACWM frame (slurm 25193772):
> EasyAnimate's class **0.0118**, diffusers' class **0.0151**, mean-image
> baseline **0.1834**. The diffusers VAE reconstructs **12x better than
> baseline** even with the 20 tensors missing — degraded ~28%, not broken.
> The NaN cause is still open and lies in the transformer call (most likely the
> hand-built `inpaint_latents` or the null text context).
>
> **Process note worth keeping:** the diagnosis survived three tests because none
> of them could falsify it — noise input made every VAE fail equally, and shape
> checks passed regardless. The test that settled it compared against a
> *baseline* (mean-image error), which made the numbers interpretable instead of
> merely comparable.

## Symptom

The EasyAnimate **V5 (diffusion)** integration produced `NaN` from the transformer
the moment it was fed a real VAE latent. Random latents survived a few blocks;
real ones NaN'd at block 0.

## Root cause

`diffusers.AutoencoderKLMagvit` has **no `mid_block_use_attention` parameter** —
verified absent in **both 0.37.1 and 0.39.0**, so no upgrade fixes it. V5's
`vae/config.json` sets it `True`. diffusers prints *"not expected and will be
ignored"*, builds a mid-block **without attention**, and then drops the
checkpoint's 20 attention tensors:

```
V5   : missing=0  unexpected=20   <- encoder/decoder.mid_block.attentions.0.*
V5.1 : missing=0  unexpected=0
```

`from_pretrained` **warns and continues**. The result is a structurally wrong
autoencoder that returns correctly-shaped garbage.

V5's `model_index.json` even names a different class (`AutoencoderKL`) than its
own `vae/config.json` (`AutoencoderKLMagvit`) — a second signal that V5's VAE is
not diffusers-native. V5.1's agrees.

## Why it survived testing — the transferable lesson

The integration smoke test checked `encode`/`decode` **shapes**. Shapes were
perfect. **An autoencoder's test must be a round-trip:** `decode(encode(x)) ≈ x`.
Nothing else distinguishes a working VAE from a broken one that happens to
preserve tensor dimensions. Two guards now going in:

1. ✅ **DONE (2026-08-04)** — hard failure on non-empty `unexpected_keys`/`missing_keys`.
   Implemented as `_load_checked()` in `models/base/easyanimate_video.py`; both the
   transformer and the VAE now load through it. Verified to fire on V5 with the exact
   216/216 figures, on the real `from_pretrained` path. **Had this existed on day one it
   would have caught the transformer mismatch in ~2 minutes instead of ~3 days** — the
   guard was proposed for the VAE, and the bug it actually caught was a bigger one
   nobody was looking for.
2. **a round-trip assertion** in every backbone smoke test — still to do.

The same class of miss as the vacuous zero-init tests earlier in the campaign:
a check that passes because it measures structure rather than behaviour.

## Consequence for the diffusion-vs-flow experiment

Weakens — but does not kill — the reason this pairing was chosen. "One wrapper,
both arms, cannot diverge" is **false for the VAE**: V5.1 is diffusers-native,
V5 is not.

~~It does not touch the part under test. The **transformers** both load cleanly and
the adapter attaches to the transformer, so the diffusion-vs-flow comparison
remains at the transformer level.~~ **← FALSE, see the second correction at the top.
Both transformers load with 216 randomly-initialised tensors under diffusers 0.39.0.
This was written as fact on the basis of shapes and an absence of raised exceptions,
neither of which can detect a partial load — the same reasoning error the ticket
already documents one level down.** And the VAEs were *already* a recorded
confound (`mid_block_use_attention` True vs False, i.e. different latent
spaces) — this makes an existing difference visible in the loader rather than
introducing a new one.

## Fix in progress

Use EasyAnimate's own `easyanimate/models/autoencoder_magvit.py` (which does
support `mid_block_use_attention`) **only for pre-encoding**, per the user's
suggestion: the VAE is needed just to precompute latents and to decode samples,
never on the training hot path. So it stays out of `models/base/` entirely —
our framework sees latents plus the transformer. Repo cloned to
`/scratch-shared/lbierling1/EasyAnimate`.

**Env note:** that repo needs `pkg_resources`, removed in `setuptools>=81`.
Pinned `setuptools<81` on the supervisor's account (installed 80.10.2).

Verification pending (slurm 25193614): does EasyAnimate's class load V5 with
`missing=0, unexpected=0`, and does it round-trip?
