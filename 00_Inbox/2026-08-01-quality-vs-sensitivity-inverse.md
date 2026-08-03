2026-08-01 ~overnight — **the runs with the best quality have the worst action-following**

Pulled while verifying the input-blindness audit's "6/6 on ACWM" claim. Every
number below is the last logged eval of a real run (`r.summary._json_dict`,
projects `Wan2.2-avid-xattn-acwm-robotarm` / `SkyReels-avid-xattn-acwm-robotarm`).
W/L = adapted beats / loses to the **frozen base** on FID, FVD-I3D, LPIPS, SSIM,
PSNR, MSE in that order.

| run | what | step | quality | `effect_rel` |
|---|---|---|---|---|
| `8zjjn7wl` | SkyReels × ACWM | 897 | **WWWWWW** | **0.0013** |
| `7bmzwv6u` | Wan × ACWM SIMPLE | 2936 | **WWWWWW** | **0.0022** |
| `tny84p7k` | Wan × ACWM GATEFIX | 3557 | WWWLWW | 0.0020 |
| `ncztxyyo` | Wan × ACWM cap09 | 1200 | LWWWWW | 0.0049 |
| `52o3uxz8` | Wan × ACWM TOKENNORM | 3315 | **WWWWWW** | 0.0062 |
| `vy9tcuco` | Wan × ACWM TN-NOBASE | 3054 | **WWWWWW** | 0.0077 |
| `sgdftf6b` | SkyReels × RT-1 TN-NOBASE | 4399 | LLLLWW | **0.0173** |
| `gi44pv5k` | SkyReels × RT-1 TN-ORACLE | 1399 | LLLLWW | **0.0211** |

## Two things fall out

**1. The adapter is an excellent domain corrector and a non-existent action
conditioner.** `52o3uxz8` beats the frozen base on *all six* metrics —
FID 57.4 vs 90.1, **FVD 406 vs 1118 (2.75×)**, LPIPS 0.192 vs 0.239 — while its
action structure is at chance on all three axes
([[../30_Knowledge/experiments/20260731-wan-action-signal-is-a-global-bag]])
and `effect_rel` is 0.0062. A 2.75× FVD win carrying no action information is a
**domain correction**. That is a real, positive, honest D2 finding — and it is
*not* what the deliverable claims to be about.

**2. ⚠ The quality/sensitivity separation is perfect, and it cuts the wrong
way.** Every run that improves quality (6/6 or 5/6) has `effect_rel ≤ 0.0077`;
both runs that degrade it (2/6) have `effect_rel ≥ 0.0173`. **The two groups do
not overlap.** This is exactly what
[[../30_Knowledge/experiments/20260801-wan-rt1-indistribution-plateau]]'s
confound predicts: worse fit ⇒ more sensitivity to *any* perturbation ⇒ higher
`effect_rel`, with no action understanding involved.

**Do not over-read it yet — it is confounded.** The two low-quality runs are
*also* the only two RT-1 runs, so dataset and quality-outcome vary together
across these eight runs. The pattern cannot distinguish "RT-1 pays for actions"
from "RT-1 fits worse, inflating the metric". That is precisely the question
probe **25143284** was launched to settle, and this table raises its stakes
considerably.

## Corrections to existing vault claims

- The attribution of the RT-1 quality split to **oracle removal** is **wrong**:
  the oracle-ON arm `gi44pv5k` splits identically (LLLLWW). The split tracks the
  dataset, not the oracle.
- The pixel-vs-perceptual split is **not universal** — on ACWM with a live gate
  the adapter wins 6/6. It appears on RT-1 (and per the audit, MetaWorld / Push
  Block / the D3 action-free arm).

## To discuss in the morning before this goes into `30_Knowledge/`

Whether "the adapter is a domain corrector, not an action conditioner" becomes
a *named finding* of the thesis. My read: it is the most defensible positive
statement the campaign has produced, and it reframes D2 from a failure into a
characterisation. But it is an interpretation, so it needs sign-off.
