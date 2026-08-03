---
type: paper
status: living
last_updated: 2026-08-01
title: "WEAVER: a world model for robotic manipulation (fidelity / consistency / efficiency)"
authors: []
venue: "no paper yet — README says 'Paper: coming soon' (checked 2026-08-01)"
year: 2026
url: https://github.com/arnavkj1995/WEAVER
local_pdf:
relevance: D3 / D4 — closest prior art to the combined deliverable
---

# WEAVER — action-conditioned flow world model with few-step distillation

> Found 2026-08-01 during the search for a flow model without temporal latent
> compression. Verified directly from the GitHub README and the HF model card;
> **no paper exists yet**, so everything here is sourced from the repo.

## Why this matters to us

It sits on **D3/D4's contribution surface**: an action-conditioned world model,
trained with **rectified flow**, on **per-frame SD3 latents** (no temporal
compression), with a released **few-step distilled checkpoint**. That is the
same destination the thesis is heading toward.

| property | WEAVER (sourced) | our thesis |
|---|---|---|
| task | action-conditioned robot world model | same (D2) |
| objective | rectified flow | flow matching + diffusion (D1 covers both) |
| latents | SD3 per-frame, no temporal compression | DC 2D per-frame; Wan/SkyReels 3D compressed |
| few-step | **ReFlow post-training**, "distill a multi-step teacher into a faster student rollout"; inference `model.val_steps=8` (lockstep) or `45` (staggered pyramid) | **step-size-conditioned shortcut adapters** with local + multi-step self-consistency (D3) |
| what is trained | the world model itself | **a small adapter on a FROZEN pretrained base** (D1) |
| data | DROID (+ `yilin-wu/droid_ood_data` OOD split) | ACWM-Phys, RT-1, OpenVid |
| license | MIT | — |

## Where the contribution still stands — and where it does not

**Preserved (as far as the repo shows):**

1. **Frozen base + plug-and-play adapter** is our contribution; WEAVER trains
   the world model. The whole D1 taxonomy question — which adapter family, at
   what cost, composed how — is not what WEAVER is about.
2. **ReFlow ≠ shortcut.** ReFlow is the rectified-flow re-coupling procedure;
   our D3 is a **step-size-conditioned** adapter trained with self-consistency
   `s(x,t,2d) ≈ ½s(x,t,d) + ½s(x_{t+d},t+d,d)`. One model that handles *any* d
   versus a distilled student at a fixed budget. (Cf. the standing warning not
   to conflate shortcut with general self-distillation.)

**Genuinely threatened:**

3. **"Fast action-conditioned world models for planning" as a novel goal** is
   no longer novel — it is released, MIT, with weights. Any framing that sells
   the *destination* rather than the *mechanism* now needs WEAVER cited in the
   same breath.
4. It is a **plausible baseline an examiner will ask about**, and a strong one:
   it reports SOTA policy evaluation and faster test-time planning.

## Answered 2026-08-02 — both discriminators fall our way

- **The dynamics transformer is trained from scratch**, and is *already*
  action-conditioned via **8-D joint deltas as a per-frame token**. WEAVER is
  therefore not an adapter-on-frozen-base result: **D1 and D2 are untouched.**
- **`WEAVER-ReFlow` is a plain 2-rectification ReFlow — NOT step-size
  conditioned.** A fixed-budget student, not one model serving any `d`.
  **The D3 shortcut contribution is not scooped.**
- Weights: 3 × 8.26 GB, MIT, released.

**✅ VERIFIED AT THE SOURCE 2026-08-02** — `scripts/reflow.sh` (raw file,
github.com/arnavkj1995/WEAVER/main):

```
model.val_steps=4
model.rectified_teacher_steps=50
model.rectified_student_rollout_steps=4      # FIXED student budget
model.rectified_rollout_loss_coeff=0.0       # off by default
```

**Fixed-budget rectification: a 50-step teacher distilled into a 4-step
student.** No step-size conditioning, no self-consistency, no shortcut
objective anywhere in the launcher. The student is trained for *one* budget.

(Correction to my earlier note: the README's `val_steps` 8/45 are the base/FT
models' inference modes; the **ReFlow student is 4 steps**, as the search agent
originally reported.)

**Assessed and rejected as a backbone for us:** it is flow-only on a per-frame
VAE, so it closes neither the objective nor the tokenizer confound, and its
already-action-conditioned from-scratch trunk leaves nothing for D1/D2 to do.
Its value is as a **baseline and a D3 target**.

## Still open

- Its action interface (8-D joint deltas) vs ours — mapping cost if used as a baseline.
- Sample quality / eval protocol, for baseline comparability.

## Related

- [[../../50_Decisions/open/d3-positioning-vs-weaver-reflow]] — the positioning
  decision this forces
- [[shortcut-models]] · [[consistency-models]] · [[self-distillation]]
- [[../../00_Inbox/2026-08-01-flow-model-no-temporal-vae-search]] — how it was
  found
