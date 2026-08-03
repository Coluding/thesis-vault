---
type: exp
scope: eval
status: open
priority: high
created: 2026-08-01
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/thesis-storyline]]", "[[../../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]]", "[[../../30_Knowledge/writing/writing-plan-2026-08]]", "[[../../10_now/positioning]]"]
---

# exp: plan through the DC world model — the missing link of the main spine

## Why this is now the top experimental priority

The thesis spine is **DC → planning → too slow → flow → shortcut**
([[../../30_Knowledge/writing/thesis-storyline]], updated 2026-08-01). Every
link has evidence except the second:

| link | evidence |
|---|---|
| DC + adapter follows actions | **arm E: 0.106 = 3.6× the AVID reference** ([[../../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]]) |
| **planning on it** | **NONE — never run** |
| too slow | cost arithmetic (horizon × NFE × per-step); needs a measured wall-clock to be concrete |
| flow matching | pivot decision + runs |
| shortcut | D3 runs |

Planning is what turns the spine from *a sequence of components* into *a
system*. It is also AVID's own stated future work, so it is the citable
extension the positioning already claims. **Higher value than further Wan
mechanism work**, which is now well-characterised.

## Do we need a learned reward model? — No. Use action recovery.

ACWM's `metadata.pt` carries **only** `video_path`, `actions`, `length`
(verified 2026-08-01): **no rewards, no states**. A learned reward model would
therefore require inventing labels — extra work that adds a confound (is a
negative result the world model's fault or the reward model's?).

We have something better: **ground-truth action sequences**. That makes the
evaluation self-scoring.

### Design: inverse planning / action recovery

Give the planner the start frame **and the true future frame** as the goal.
Search action sequences through the world model. Then ask: **did it recover the
actions that actually produced that future?**

```
given  x_0 (start latent), x_H (true future latent, H steps ahead)
search a_{1..H} to minimise  || WorldModel(x_0, a_{1..H}) − x_H ||   (latent L2)
score  || a_recovered − a_true ||        <- ground truth, no labels needed
```

- **Planner:** CEM or random shooting — 10–20 candidates, horizon 4–8.
- **Scoring inside the planner:** latent-space L2 to the goal latent (a
  *distance*, not a learned reward — nothing to train).
- **Scoring of the experiment:** action-space error against the true sequence,
  plus goal-latent error achieved.

### Why this design is strong

1. **No reward model, no invented task, no label cost.**
2. **Unambiguous ground truth** — the true action sequence is known exactly.
3. **It has a built-in null:** planning through the *frozen action-free base*
   must be at chance **by construction** (the base ignores actions), so the
   control is guaranteed valid rather than assumed.
4. It measures precisely the thesis claim — whether action-conditioning is
   usable for *selecting* actions, not just for predicting frames.
5. It connects directly to the campaign's central finding: our models show
   *sensitivity without control*. Action recovery is the sharpest possible test
   of whether DC's 10×-higher effect_rel converts into control where Wan's did
   not.

### Baselines (all three required)

| baseline | expected if the world model is usable |
|---|---|
| random action sequences | recovery error ≈ prior spread |
| **frozen base (action-free)** | at chance — the built-in null |
| DC arm E (`condition_center`, 0.106) | **below both** |
| *(optional)* DC arm 0 (untreated) | between — isolates the fix's contribution |

### Plan in DC latent space, not pixels

The world model already predicts **DC 4-channel SD-VAE latents**, so the whole
search stays latent-side: encode the start and goal frames once, roll candidates
forward as latents, score with latent L2, and **never call the VAE decoder
inside the loop**. Decoding 20 candidates × horizon 8 would dominate the cost
and add nothing — the comparison is between predicted and true *latents*. (DC
encodes live, no precompute needed — verified 2026-07-30.) Decode only the
winning rollout, for the figure.

### Companion measurement (high value): inverse dynamics on latents = the ceiling

Train a small **inverse dynamics model (IDM)** on the *ground-truth* latents:

```
IDM:  (z_t, z_{t+1})  ->  a_t          trained on real transitions only
```

No new labels — latents come from the frozen VAE, actions from `metadata.pt`.
This measures **how much action information the latent transitions contain at
all**, independent of any world model, and it is the missing denominator for
the whole campaign:

- **IDM recovers actions well** ⇒ the information is there, and a world model
  that ignores it is genuinely failing. Our 0.45%-of-loss economics is then a
  statement about the *objective*, not about the data — and IDM accuracy
  becomes the ceiling that action recovery should be compared against.
- **IDM recovers actions poorly** ⇒ the actions are only weakly determined by
  visible frame transitions on this data. That would explain the 0.45% from the
  **data side** and would substantially reframe the boundary claim: not "the
  objective under-prices actions" but "these actions are barely observable in
  pixels". Either way it is a headline-grade result for §9.

Cheap (a few-layer conv/MLP on cached latents, minutes on one GPU), needs no
world model, and is worth running **before** the planner — it tells us whether
action recovery is even possible in principle, and calibrates every number in
the campaign. A learned IDM can additionally serve as a task-relevant scorer
for the planner (better than raw latent L2, which is dominated by background),
but that is optional.

### Also measure: wall-clock

Per planning step: candidates × horizon × NFE × per-step latency. **This is the
number that quantitatively motivates the entire shortcut half of the thesis**
and is worth the run even if recovery fails.

## Notes

- Keep it small: this is a demonstration, not a control/RL contribution
  ([[../../10_now/positioning]] anti-positioning).
- The wall-clock measurement is worth the run **even if planning fails** — it
  is the number that motivates D3.
- Do not gate the writing on this: Ch3/Ch5-D2/Ch6 can be drafted now
  ([[../../30_Knowledge/writing/writing-plan-2026-08]]).
- **Run order:** IDM ceiling first (cheap, calibrates everything) → action
  recovery with the three baselines → wall-clock. The IDM result may change how
  the planning result should be interpreted, so it comes first.
