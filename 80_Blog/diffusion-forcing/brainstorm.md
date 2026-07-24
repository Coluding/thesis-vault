---
title: "To condition my video model on a frame, I set its noise level to zero"
slug: diffusion-forcing
type: blog-brainstorm
status: drafting
created: 2026-06-24
last_updated: 2026-06-24
deliverable: exploratory
sources:
  - "[[../../30_Knowledge/theory/diffusion-forcing]]"
  - "[[../../30_Knowledge/theory/diffusion-forcing.html]]"
  - "[[../../20_Tickets/feat-wan22-ti2v-diffusion-forcing-i2v]]"
  - "[[../../20_Tickets/fix-wan-flow-eval-video-grid-never-fired]]"
  - "paper: Chen et al., 'Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion', NeurIPS 2024"
---

# Brainstorm — Diffusion forcing

**Angle:** I had a frozen text-to-video model as a world-model base and needed it
to start from the *current observation frame*. Every instinct said "add an image
encoder or a concat channel" — both of which you can't bolt onto a *frozen* base.
The thing I'd never heard of, diffusion forcing, says: don't add anything. Give
that one frame a noise level of **0**. Conditioning isn't a module — it's a
number on the noise schedule. Once each frame's noise level is independent,
text-to-video, image-to-video, and autoregressive rollout are the *same weights*
with a different per-frame schedule.

**Audience:** ML practitioners who've used diffusion models and know what a
timestep / noise level is, but who (like me) only ever saw *one global* timestep
per sample. Assumed: forward process, denoising loop, attention. Not assumed:
diffusion forcing, teacher forcing as a continuous axis.

**Hook:** My eval videos looked like fog. The model was being asked to hallucinate
an entire scene from pure noise plus a 4-number action — and it showed. The fix
wasn't a better model; it was realizing I'd been holding one knob fixed that
didn't have to be.

**The "I hadn't heard of it" beat (honest):** I reached for the textbook answers
first — CLIP image cross-attention, masked-latent concat — and both were dead
ends on a frozen base (wrong input channels). The unlock was reframing
conditioning as a noise level, which I genuinely had not seen before.

**Storyline:**
1. Frozen T2V base → world model. Eval = fog (generate-everything-from-noise).
2. The obvious fixes (image encoder / concat) change the model's inputs → can't
   do it to a frozen base. The image-native variant was 14B (too big).
3. Diffusion forcing: per-frame independent noise level. A level of 0 = "given."
4. Conditioning is a number, not a module. Teacher forcing made continuous.
5. The same weights → T2V / I2V / autoregressive, by schedule alone.
6. Honest costs: it changes the training objective (masked loss on predicted
   frames), you re-clamp the observation each step, and in practice you use a
   simple *two-level* schedule, not all-different.
7. The general lesson: a capability you'd reach for a new module for can be
   latent in the objective — vary an axis you'd been holding fixed.

**Length/voice:** ~1,100–1,300 words, first-person, concrete, slightly wry.
Minimal code. One schedule diagram (ASCII) + link to the interactive explainer.

**Hard-rules check:** no benchmark numbers (none to cite yet — GPU run pending);
the "fog" eval is real (I saw it); method credited to the paper; the model-size
and architecture claims are from reading the vendored Wan source.
