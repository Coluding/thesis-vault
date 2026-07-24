---
type: tech-note
status: living
last_updated: 2026-06-22
sources:
  - "code: src/generative_flow_adapters/backbones/wan/modules/model.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/attention.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/vae.py"
  - "code: src/generative_flow_adapters/models/base/wan.py"
  - "config: configs/base/wan2.1_t2v_1_3B.yaml"
relevance: D1 / D2  # exact structure of the frozen Wan base we adapt
---

# Wan2.1-T2V-1.3B — exact model structure

The frozen base is `WanModel`
(`backbones/wan/modules/model.py:375`) — a **DiT** (transformer over
patchified video latents), trained as **single-stage rectified flow /
velocity prediction**. All line refs are the vendored copy under
`backbones/wan/modules/`. See [[wan21-vs-pyramid-flow-backbone]] for the
flow-matching evidence and [[wan-vendoring-patches]] for our edits.

## Top-level dims (T2V-1.3B)

From `WanModel.__init__` (`model.py:385-456`), values from
`configs/base/wan2.1_t2v_1_3B.yaml` / repo `config.json`:

| param | value | meaning |
|---|---|---|
| `dim` | 1536 | transformer hidden width |
| `num_layers` | 30 | `WanAttentionBlock` count |
| `num_heads` | 12 | attention heads (head_dim 128) |
| `ffn_dim` | 8960 | FFN inner dim |
| `freq_dim` | 256 | sinusoidal time-embed width |
| `text_dim` | 4096 | umt5-xxl context width |
| `in_dim` / `out_dim` | 16 / 16 | Wan-VAE latent channels |
| `patch_size` | (1,2,2) | (t,h,w) patch — temporal unpatched |
| `text_len` | 512 | context tokens (zero-padded) |

~1.419B params (verified by strict load, `models/base/wan.py`).

## Forward contract

`WanModel.forward(x, t, context, seq_len, clip_fea=None, y=None)`
(`model.py:496-503`). Note it is **list-based**, which is why our wrapper
bridges batched `[B,C,T,H,W]` → list + `seq_len`
(`models/base/wan.py` `_seq_len` / `forward`):

- `x`: list of `[16, F, H, W]` latents (one per sample)
- `t`: `[B]` timesteps
- `context`: list of `[L, 4096]` text embeddings
- `seq_len`: packed token count (`ceil(F·H·W / (1·2·2))`)
- `clip_fea`, `y`: i2v only (unused for t2v)
- returns: list of `[16, F, H, W]` velocity predictions

## The three embedding streams (model.py:459-467, 536-562)

1. **Patchify** — `patch_embedding = nn.Conv3d(16, 1536, k=(1,2,2),
   s=(1,2,2))` (`model.py:459`); each latent → tokens `[L, 1536]`, flattened
   and zero-padded to `seq_len` (`model.py:537-546`).
2. **Time** — `sinusoidal_embedding_1d(256, t)` → `time_embedding`
   (`Linear(256→1536)→SiLU→Linear`) → `e` (`model.py:465, 550`); then
   `time_projection` (`SiLU→Linear(1536→6·1536)`) → `e0` reshaped to
   `[B,6,1536]` (`model.py:467, 552`) — the **6 modulation vectors** (AdaLN).
   Computed in fp32 under `amp.autocast(float32)` (`model.py:549-553`).
3. **Text** — `text_embedding` (`Linear(4096→1536)→GELU→Linear`)
   (`model.py:461, 557`), context zero-padded to `text_len=512`.

RoPE position is precomputed in `self.freqs` (`model.py:483-488`), applied to
q/k inside attention (`rope_apply`, `model.py:46`).

## The transformer block — `WanAttentionBlock` (model.py:241-320)

30 identical blocks (`self.blocks`, `model.py:471-475`). Each holds
(`__init__` `model.py:243-279`):
- `self_attn` = `WanSelfAttention` (q,k,v,o `Linear(1536,1536)`,
  optional RMSNorm on q,k; RoPE; SDPA/flash attention) (`model.py:108-159`)
- `cross_attn` = `WanT2VCrossAttention` (q from tokens, k/v from text
  context) — selected via `WAN_CROSSATTENTION_CLASSES['t2v_cross_attn']`
  (`model.py:235, 268`)
- `ffn` = `Linear(1536→8960)→GELU(tanh)→Linear(8960→1536)` (`model.py:274`)
- three `WanLayerNorm` (`norm1/norm2/norm3`) + `modulation =
  Parameter[1,6,1536]` (`model.py:262-279`)

Block forward (`model.py:281-320`), **AdaLN-Zero style** — the 6 time-derived
vectors `e[0..5]` (= `modulation + e0`) gate each sub-layer:
```
e = (modulation + e0).chunk(6)                       # model.py:301
x = x + self_attn(norm1(x)*(1+e[1]) + e[0]) * e[2]   # model.py:305-308  (shift,scale,gate)
x = x + cross_attn(norm3(x), context)                # model.py:313      (text cross-attn, ungated)
x = x + ffn(norm2(x)*(1+e[4]) + e[3]) * e[5]         # model.py:314-318  (shift,scale,gate)
```
Modulation arithmetic runs in fp32 (`assert e.dtype == torch.float32`,
`model.py:300`) — the reason our adapter/shortcut code must run the base under
autocast (see [[wan-vendoring-patches]]).

## Head + unpatchify (model.py:323-351, 581-585)

`Head` (`model.py:323`): `LayerNorm → Linear(1536 → 16·prod(patch)=64)`, with
its own 2-vector time modulation (`model.py:340-349`); `head.head.weight`
zero-init (`model.py:634`). `unpatchify` (`model.py:587-610`) folds tokens
back to `[16, F, H, W]`.

## Attention backend

`WanSelfAttention`/cross-attn call `flash_attention` (`model.py:149,179`),
which we alias to the `attention()` wrapper with a torch-SDPA fallback so it
runs without flash-attn (`attention.py`, see [[wan-vendoring-patches]]).

## Wan-VAE (separate, not the DiT)

`WanVAE` (`vae.py:619`): causal 3D VAE, `z_dim=16`, `vae_stride=(4,8,8)`,
per-channel mean/std normalization baked in (`vae.py:632-642`).
`encode(list[[3,T,H,W]])` / `decode(...)` operate on the normalized latent the
DiT lives in. Pixel↔latent for generation/training data.

## How we wrap it

`Wan21DiTWrapper(BaseGenerativeModel)` (`models/base/wan.py`): `model_type=
"flow"`, `prediction_type="velocity"`, batched↔list bridge, null text context
by default (frozen base sees no action — that lives in the adapter). See the
integration ticket [[../../20_Tickets/feat-wan21-backbone-integration]].
