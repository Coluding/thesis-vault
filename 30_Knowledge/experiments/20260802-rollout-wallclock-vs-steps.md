---
type: experiment
date: 2026-08-02
config: configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml · configs/dynamicrafter/diffusion_dc_shortcut_d3arm_actionfree_robotarm.yaml
commit: uncommitted working tree @ 2026-08-02
wandb_run_id: n/a (offline benchmark, slurm 25154218; partial run 25152246 timed out)
ckpt_path: /scratch-shared/lbierling/outputs/{wan-shortcut-actionfree-robotarm/checkpoints/step_00000600.pt, dc-shortcut-D3-arm-run/checkpoints/step_00000400.pt}
status: completed
deliverable: D3
metrics:
  wan_fixed_s: 17.6
  wan_per_step_s: 3.09
  wan_n50_s: 172.09
  dc_fixed_s: 2.5
  dc_per_step_s: 1.10
  dc_n50_s: 57.53
  dc_over_wan_per_step: 0.36      # DC is 2.8x FASTER per step
notes: "Rollout wall-clock vs solver steps N, both backbones, through the trainer's own eval rollout path (2 clips, 2 timed repeats + discarded warm-up, CUDA-synchronised). Both curves LINEAR. Per-step cost is set by model size (DC 1.4B vs Wan 5B), NOT by the objective — so a DC-vs-Wan wall-clock comparison measures size, not diffusion-vs-flow."
---

# Rollout wall-clock vs step count — the measured D3 motivation

## Result

| | fixed | **per solver step** | N=50 |
|---|---|---|---|
| **Wan** (flow, 5B) | 17.6 s | **3.09 s** | 172.09 s |
| **DC** (diffusion, 1.4B) | 2.5 s | **1.10 s** | 57.53 s |

Full curves (s/clip): Wan 19.65 / 22.77 / 29.11 / 41.54 / 94.95 / 172.09 and
DC 3.59 / 4.70 / 6.92 / 11.30 / 30.01 / 57.53 at N = 1/2/4/8/25/50.

Both are **linear to ~1%** (successive slope estimates: Wan 3.09/3.11/3.14/3.09,
DC 1.10/1.10/1.10). So `cost(N) ≈ fixed + per_step·N`, and the fixed term is
per-call overhead (VAE decode + loading) that planning amortises.

## Reading

1. **Per-step cost is a function of model size, not of the objective.** DC is
   **2.8× faster per step** than Wan — 1.4 B vs 5 B parameters. A DC-vs-Wan
   wall-clock comparison therefore measures *size*, not diffusion-vs-flow.
   **The thesis must not claim "flow is faster."** The defensible chain is:

   > per-step cost is set by architecture → flow's advantage is that it needs
   > *fewer steps* (straighter paths) → still not few enough for planning →
   > shortcut adapters close the remaining gap.

   Quoting DC's 50-step time against Wan's few-step time would be an unfair
   comparison in our own favour.

2. **Few-step speedup is linear in N with no floor — for latent-space
   planning.** An earlier reading of this data claimed the speedup was capped at
   ~10× by a decode floor; that is **wrong for planning**, where rollouts stay
   in latent space and the VAE is never called inside the loop
   ([[../../20_Tickets/experiments/exp-eval-planning-through-dc-world-model]]).
   The fixed term is paid once, not per candidate, so N=50 → N=1 is a full
   **50×**. The cap applies only to *video generation*, where every rollout is
   decoded.

3. **Planning cost (analysed estimate, from the measured slope).** Inputs: the
   per-step slope above, halved because the eval grid renders base *and*
   adapted while planning runs only the adapted model (⚠ the ½ is inferred, not
   directly measured — the phase markers sit outside the solver loop; base-gen
   7.51 s vs adapted-gen 7.62 s at N=1 is consistent with a clean split).
   CEM with 30 candidates × 5 iterations = 150 rollouts, sequential, DC:

   | N | per planning step |
   |---|---|
   | 50 | **≈ 1.15 h** |
   | 4 | ≈ 5.5 min |
   | 1 | **≈ 1.4 min** |

   This is the number that motivates D3.

## Method / caveats

- Timed one N at a time (single-entry `eval_step_schedule`) through
  `_native_eval_grid`, so the path is identical to training-time eval; CUDA
  synchronised around each call and a warm-up discarded.
- **Do not quote `ms_per_solver_step` from the tool's own division** — the
  script's self-check reported per-step spread of 475% (Wan) / 212% (DC),
  because the fixed term dominates at small N. Quote the *slope*.
- Batch-1 numbers. Parallel batching amortises the fixed term further, so the
  planning estimates above are conservative for a batched planner.
- DC checkpoint is step 400 (arm A was cancelled for budget) — irrelevant here:
  per-step latency is a property of the frozen base, not of training length.
- **Not measured: a DPM-Solver / fast-sampler baseline at matched NFE.** Until
  that exists, "diffusion rollout is slow" is attackable — high-order solvers
  reach ~10–20 steps without any of our machinery.

## Related

- [[../../20_Tickets/experiments/exp-shortcut-d3-fewstep-vs-noshortcut-control]]
- [[../writing/thesis-storyline]] — spine link 3 ("planning is too slow")
