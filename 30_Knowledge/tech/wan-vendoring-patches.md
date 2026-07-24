---
type: tech-note
status: living
last_updated: 2026-06-19
sources:
  - "code: src/generative_flow_adapters/backbones/wan/modules/model.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/attention.py"
  - "code: src/generative_flow_adapters/backbones/wan/__init__.py"
  - "code: src/generative_flow_adapters/backbones/wan/modules/__init__.py"
relevance: D1  # keeping the vendored Wan2.1 backbone runnable without flash-attn
---

# Wan2.1 vendoring patches — why our copy diverges from upstream

The Wan2.1 `wan/` package is vendored under
`src/generative_flow_adapters/backbones/wan/`. Four deliberate edits make it
import and run **without flash-attn and on CPU** (random-weight smoke tests).
Re-apply these if re-syncing from upstream.

1. **`wan/__init__.py` neutered.** Upstream eagerly imports the full pipelines
   (`text2video`/`image2video`/`vace`) → pulls in T5, CLIP, Wan-VAE, easydict.
   We only need the `WanModel` DiT, so the aggregator is now just a docstring.
2. **`wan/modules/__init__.py` trimmed.** Now exports only `WanModel` +
   `attention`/`flash_attention`; the T5/VAE/vace re-exports (heavy imports at
   load time) are dropped. Those files still exist — import them directly if a
   pixel decode path is ever needed.
3. **`model.py` attention alias.** `from .attention import flash_attention` →
   `from .attention import attention as flash_attention`. The model calls
   `flash_attention(...)` directly; the bare function asserts CUDA + flash-attn,
   while the `attention()` wrapper falls back to torch SDPA. All call sites use
   kwargs (`q,k,v,k_lens,window_size`) compatible with `attention()`.
4. **`attention.py` SDPA dtype restore (upstream bug).** The SDPA fallback
   casts q/k/v to bf16 but — unlike the flash path's `x.type(out_dtype)` —
   never restores the caller dtype, so a fp32 forward dies at the next Linear
   with "mat1 and mat2 must have the same dtype". Fixed by capturing
   `out_dtype = q.dtype` and returning `out.type(out_dtype)`.

On a real GPU with flash-attn installed, edits 3-4 are inert (the flash path
runs and already restores dtype); they only matter for the no-flash/CPU path.

See [[wan21-vs-pyramid-flow-backbone]] and the integration ticket
[[../../20_Tickets/feat-wan21-backbone-integration]].
