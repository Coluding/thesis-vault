---
type: experiment
date: 2026-07-24
config: configs/wan22/diffusion_wan22_avid_xattn_{replace_nobase_overfit,gatelow_nobase_overfit,gatelow_nobase_gatecap_overfit,gatelow_cap_sigmashift}_metaworld.yaml
commit: 07ec01bd (triangle + cap/shift runs); cf133822 (uxrst2k5 reference)
wandb_run_id: rxzwh4ak, o79ki0ul, o9113j4h (failed), hvxlbfjx; uxrst2k5 (reference)
ckpt_path: remote (training box)
status: completed (rxzwh4ak still running at time of writing)
deliverable: D2
metrics:
  see_tables_below: true
notes: "Optimization traps are real + fixable (gate_cap validated); but fixing them does NOT produce action-following on MetaWorld -> data is action-redundant -> move to ACWM."
---

# MetaWorld overfit triangle + cap/σ-shift full-data run — base-parity persists after the traps are fixed

Dataset: MetaWorld `five_task_diverse.hdf5`, Wan2.2-TI2V-5B frozen base,
34M ActionWan xattn adapter, 97-frame windows. All four runs share commit
`07ec01bd`. `uxrst2k5` (commit `cf133822`, older) is the "gated + base-input,
uncapped" reference the no-base arms are compared against.

## Runs

| run (wandb) | config | composition | gate | base as input | data | state |
|---|---|---|---|---|---|---|
| `uxrst2k5` | xattn i2v (overfit) | mask_mix | uncapped | yes | 1 clip | crashed @342 (ref) |
| `o79ki0ul` | gatelow_nobase_overfit | mask_mix | uncapped | **no** | 1 clip | killed @600 |
| `o9113j4h` | gatelow_nobase_gatecap_overfit | mask_mix | **cap 0.9** | no | 1 clip | **failed — no logged data** |
| `rxzwh4ak` | replace_nobase_overfit | replace | none | no | 1 clip | running |
| `hvxlbfjx` | gatelow_cap_sigmashift | mask_mix | **cap 0.9** | yes | full (+σ-shift 5.0) | killed @900 |

## Training-seam trajectories (sourced: `wandb.Api()`, 2026-07-24)

| run | step | denoise Δ (base−adapted) | gate mean | adapter grad norm | pred-vs-base cosine |
|---|---|---|---|---|---|
| `o79ki0ul` (gate, no base-in, uncapped) | 1 → 600 | −0.860 → **+0.0012** | 0.50 → **0.992** | 3.43 → **0.005** | 0.008 → **0.031** |
| `hvxlbfjx` (cap 0.9 + σ-shift, full data) | 1 → 900 | −0.902 → **+0.0066** | 0.50 → **0.894** | 3.83 → **0.027** | −0.026 → **0.856** |
| `uxrst2k5` (gate, base-in, uncapped, ref) | 1 → 342 | −0.90 → +0.0008 | 0.50 → 0.99 | 4.4 → 0.003 | _(not logged this run)_ |

Reads:
- **The gate trap is real and now understood.** Uncapped runs (`o79ki0ul`,
  `uxrst2k5`) drive the gate to ~0.99 within ~70–150 steps and the adapter
  grad norm collapses to ~0.003–0.005 — learning dies. **The cap defuses it:**
  `hvxlbfjx` holds the gate at exactly 0.9 (the pull to saturate is still
  there) and the grad norm stays alive at 0.027 (5× the uncapped runs).
- **Base-input removal stops the pred from cloning, but not the gate.**
  `o79ki0ul` (no base input) has pred-vs-base cosine ~0.03 — the pred branch
  does **not** clone the base (no base velocity to copy) — yet the gate still
  saturates to 0.99 and mutes it. So base-input and gate are **two
  independent traps**. `hvxlbfjx` (base input on) clones: cosine 0.86.

## Generation-seam eval (post-fix pipeline; adapted vs frozen base)

| run | adapted FID | base FID | adapted PSNR | base PSNR | adapted MSE | base MSE | eval denoise Δ |
|---|---|---|---|---|---|---|---|
| `hvxlbfjx` (cap+σ-shift, full) | **64.57** | 64.70 | 16.55 | 16.12 | 0.0221 | 0.0244 | +0.0015 |
| `o79ki0ul` (gate, no base-in) | 122.45 | 111.40 | 16.54 | 16.51 | 0.0222 | 0.0223 | +0.0012 |
| `rxzwh4ak` (replace, no base-in) | 408.70 | 111.40 | **23.39** | 16.51 | **0.0046** | 0.0223 | **−0.3007** |

Reads:
- **`hvxlbfjx` (the intervention run): still base-parity.** With the gate trap
  defused *and* σ-shift concentrating supervision at high noise, the adapter
  beats the base by hair-thin margins on every pixel metric (FID 64.57 vs
  64.70, PSNR 16.55 vs 16.12, MSE 0.0221 vs 0.0244) — but the pred-vs-base
  cosine of **0.86** shows it got there by **cloning the base**, not by using
  actions. The margin is action-*independent* domain calibration, consistent
  with the 2026-07-21 σ-sweep (adapter beats base only at σ≈0.05). Countering
  the optimization did not create an action signal.
- **`rxzwh4ak` (replace, no gate, no base input): the 34M can't stand alone.**
  Denoise Δ −0.30 (adapter denoise *worse* than base). The overfit pixel
  metrics are the MSE-optimal-blur signature: it memorizes the single clip so
  MSE/PSNR/SSIM are far better than base (MSE 0.0046 vs 0.0223, PSNR 23.4 vs
  16.5) while FID/FVD/LPIPS are far worse (FID 409 vs 111) — low-error but
  blurry, distributionally off. Confirms the base-output input is a genuine
  crutch the small adapter needs.
- **`o79ki0ul`: composed output ≈ base.** Adapter contributes nothing (gate
  0.99), so FID/PSNR/MSE track the base to 3 digits.

## Why it hasn't worked (the interpretation)

Three independent failure factors, now separated by the triangle:

1. **Gate saturation** (optimization) — real, and **fixable** by the cap
   (`hvxlbfjx` proves the mechanism works: gate held, gradient alive).
2. **Identity-on-base-output** (optimization) — real; removing the input
   stops cloning (`o79ki0ul` cosine 0.03) but exposes trap #1.
3. **Adapter capacity** — 34M can't denoise standalone (`rxzwh4ak` Δ −0.30);
   fine, the thesis composition keeps the base.

**The decisive result:** even with #1 and #2 addressed and supervision
σ-shifted (`hvxlbfjx`), the full-data adapter **still converges to a
base-clone at base-parity** (cosine 0.86, Δ ≈ 0, FID within 0.1 of base).
Fixing the optimization does not help because **MetaWorld's actions carry
almost no loss-reducing signal given the observation** — the base already
predicts the (action-independent) future, so the adapter has no gradient
pressure to deviate. This is the pre-registered trigger for the dataset
decision ([[../../50_Decisions/open/second-dataset-action-informativeness]]):
move to an action-informative benchmark (ACWM-Phys Push Cube), where the
same countermeasures (cap 0.9 + σ-shift) — now validated as *mechanisms* —
have real action signal to bite on.

**Pending:** `o9113j4h` (gatelow-nobase-cap09 overfit) failed with no logged
data — re-run to complete the triangle's third arm (it would confirm
cap+no-base-input: pred should improve while the gate stays ≤0.9).
`rxzwh4ak` still running — final numbers to be refreshed on completion.
Definitive action-blindness proof = the action-shuffle probe on `hvxlbfjx`'s
checkpoint (loss gap under shuffled/zeroed actions), not yet run.

## Related
- [[20260721-replace-fix-validation-sigma-sweep-action-probe]] — the σ-sweep + action probe this extends
- [[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]] — the gate trap (cap now validated)
- [[../../20_Tickets/experiments/exp-adapter-gatelow-cap-sigmashift-metaworld-run]]
- [[../../50_Decisions/open/second-dataset-action-informativeness]] — the dataset move this triggers
