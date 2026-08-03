---
type: exp
scope: backbone
status: open
priority: high
created: 2026-08-01
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/thesis-storyline]]", "[[../../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]", "[[../../50_Decisions/open/wan-action-following-needs-objective-change]]"]
---

# exp: the missing cell — flow matching WITHOUT temporal latent compression

## The confound (user, 2026-08-01)

The D2 backbone matrix varies two things together:

| backbone | objective | latent tokenizer | action following |
|---|---|---|---|
| DynamiCrafter-512 | **diffusion** | **2D SD-VAE, per-frame, temporal ratio 1** | works (arm E, 0.106) |
| Wan2.2 TI2V-5B | **flow** | 3D causal VAE, temporally compressed | collapses |
| SkyReels | **flow** | 3D causal VAE, temporally compressed | collapses (worst) |

The one cell that works is the only one that is *both* diffusion *and*
per-frame-latent. Every conclusion the storyline draws about "flow vs
diffusion" is therefore **confounded with the tokenizer**.

**The mechanism this confound would imply is concrete and plausible.** With a
~4× causal temporal VAE, one latent frame mixes ~4 real frames. Per-frame
actions are therefore *smeared together before the adapter ever sees them* —
the adapter is asked to condition on `a_t` at a temporal resolution the latent
space does not represent. That is an equally good explanation for the "global
bag" finding (temporal alignment at chance,
[[../../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]])
as any objective-level story: you cannot align to a frame index the
representation has averaged away.

Note this also partly explains why the px→latent **temporal binning** fix
(`adaptive_avg_pool1d` in the output head) was needed at all — it is a
workaround for exactly this mismatch.

## The experiment

Add a **fourth cell: flow matching + a tokenizer with temporal compression
ratio 1** (per-frame 2D VAE). That single cell separates the two factors:

| result | conclusion |
|---|---|
| flow + per-frame latents **follows actions** | the tokenizer was the problem, not the objective. The D2 collapse story is about temporal latent compression — a **much more useful and general finding** than "flow is harder", and it directly predicts which future backbones will work. |
| flow + per-frame latents **still collapses** | the objective really is implicated; the flow-vs-diffusion framing survives, now with the confound closed. |

Either way the confound is closed, which is worth the run on its own — it is
the first question an examiner asks of the backbone table.

## Model selection

Requirements: flow matching / rectified flow; VAE temporal compression ratio
**1**; **I2V capable** (hard — the world model predicts the future from the
current frame); public weights; ≲14 B (base is frozen, only the adapter
trains); vendorable inference code.

The intersection is expected to be rare — temporal compression is what makes
video DiTs affordable, so most flow video models have it, and most per-frame
latent video models (SVD, Latte, AnimateDiff, Open-Sora v1.0) are diffusion.
If the intersection is genuinely empty among released checkpoints, fall back to
the **lowest available temporal ratio** (e.g. 2×), which still partially
separates the factors — and record the empty intersection as a finding.

### RESULT 2026-08-01 — better than a fourth cell: a controlled ablation

The overnight search (→ `00_Inbox/2026-08-01-flow-model-no-temporal-vae-search.md`)
landed on **OpenDWM CTSD** (`wzhgba/opendwm-models`, Apache-2.0 weights, MIT
code, ungated — license verified on the HF page 2026-08-01). One pipeline class
ships three checkpoint families that differ in **exactly our variable**:

| family | base | objective | tokenizer |
|---|---|---|---|
| `ctsd_21_*` | SD2.1 | diffusion | per-frame 2D VAE |
| `ctsd_35_*` | SD3.5-medium | **flow** | **per-frame 2D VAE ← the missing cell** |
| `ctsd_35_tvae_f17_*` | SD3.5-medium | flow | CogVideoX VAE, **tc=4** |

That is strictly better than adding a fourth backbone: the tokenizer swap is a
**one-line config change on the same base model**, so it removes architecture,
data and training recipe as confounds *simultaneously* — which comparing DC vs
Wan vs SkyReels across three codebases can never do. Reported source lines
(`ctsd.py` L952-954 VAE default, L963-964 `is_temporal_vae` true only for
CogVideoX, L977-983 SD3 → `FlowMatchEulerDiscreteScheduler`) are the agent's;
**re-verify at the source before downloading** — I have confirmed only the repo
and its licence directly.

**Bonus, directly on D3:** `ctsd_35_df16` reportedly uses a
`temporal_independent.FlowMatchEulerDiscreteScheduler` — per-frame-independent
flow timesteps over per-frame latents, i.e. diffusion forcing on rectified
flow. That design only makes sense *because* tc=1, which is itself evidence for
this ticket's mechanism.

### 🛑 NO-GO on adopting CTSD as a backbone (feasibility check, 2026-08-02)

The risk was real and it decided the question. Findings (agent's read of
OpenDWM @ `b0ecc3d4`; all five source anchors above verified correct, but four
corrections to the premise):

1. **`ctsd_35_*` is not one family** — `ctsd_35_tirda_*` (`frame_prediction_style: "ctsd"`)
   and `ctsd_35_df16_*` (diffusion forcing) are different training regimes.
2. **The families are not matched** on dataset mix or step count (`ctsd_21` bm
   variant is `nwa`; the `ctsd_35` one is `nwao`) — so the "one-line swap"
   framing above was too optimistic.
3. **The action vector is 2-D** — `cat([speed, steering])` through a hard-coded
   car bicycle model (`wheel_base=2.7`, `steering_ratio=14`), not a 7-DoF robot
   vector. Confirmed independently by arithmetic:
   `projection_class_embeddings_input_dim` = 2816 = 11×256 for the trio vs 3328
   = 13×256 for `df16` — exactly two extra scalars.
4. **`view_cam_emb` — the sole carrier of the action — is consumed at two sites,
   both gated on `enable_crossview`.** With cross-view off the entire numeric
   conditioning path is dead code; the action rides the multi-camera branch, and
   all 20+ configs are 6-view.

I2V is genuinely first-class and the layout conditioning *is* cleanly droppable
(0.8 dropout, dedicated action-CFG channel) — those two worries were unfounded.
What kills it is that using CTSD means acquiring nuScenes/Waymo/Argoverse and
rebuilding a 6-camera pipeline: **~2–3 weeks** (estimate benchmarked against
SkyReels ~1,090 lines and Wan ~2,319+1,050 test lines), driven by the 6-D
`[B,T,V,C,H,W]` vs our 5-D rank mismatch, three text encoders, and a parallel
config system. A driving prior also contributes nothing to tabletop
manipulation.

**Useful reframing the check produced:** the gating question was mis-posed. Our
design deliberately keeps the base a pure `f_base(x_t, t)` map with *all*
action in the adapter (`models/base/wan.py:40-43`), so CTSD's action interface
would never be used. What disqualifies it is that CTSD **isn't** a pure
`(x_t, t)` map.

**If the VAE axis must be closed:** `ctsd_35_tirda_bm_nwao_40k` vs
`ctsd_35_tvae_f17_tirda_bm_nwao_50k` (34.4 GB) is still a genuinely clean
one-variable pair — run it as a **bounded inference-only side-study**, not a
backbone adoption. Otherwise **bound the confound in the write-up**, which is
the cheap and honest option and is my recommendation.

⚠ Unverified: per-user scratch quota (`du` never returned; filesystem headroom
is fine at 1.9P of 2.5P), and whether the `.pth` files bundle VAE/text-encoder
weights (shifts the ~80–90 GB estimate).

**Methodological lesson worth keeping:** the automated screen over 612
HF-tagged video models was structurally blind to every hit, because world
models ship as research repos (raw `.pth` + JSON, no `model_index.json`, no
`vae/` subfolder, no pipeline tag). The surviving negative result is narrower
than "the intersection is empty": *among general-purpose T2V models packaged
for `diffusers`*, it is empty and the flow floor is tc=4. World-model repos are
the exception precisely because they inherit an image VAE from SD3/FLUX rather
than train a video tokenizer.

Also surfaced: **WEAVER** — action-conditioned, rectified flow, per-frame SD3
latents, with a released few-step ReFlow checkpoint. Not a backbone candidate
but **prior art on D3/D4** → [[../../30_Knowledge/related-work/weaver]],
[[../../50_Decisions/open/d3-positioning-vs-weaver-reflow]]. And **MiniWAM**
(frozen `sd-vae-ft-mse` + joint video/action flow matching, no released
weights) is close enough to our recipe to warrant a scoop check.

## Protocol once a model is chosen

Match the existing recipe exactly so the cell is comparable: `action_token_norm`
(or its equivalent for the new architecture), `condition_on_base_outputs:
false`, live gate (mind
[[../bug-adapter-gate-cap-equals-init-freezes-gate]]), same dataset
(ACWM robot_arm **and** RT-1 if budget allows), same probe suite.

**Report quality alongside sensitivity from the start** — the RT-1 cells are a
net perceptual regression vs their frozen base while improving pixel metrics,
and that split must not be discovered late again.
