"""Figure: probability-flow trajectories under diffusion, rectified flow, and reflow.

Produces 70_Thesis/latex/figures/trajectory-curvature.pdf

The point of the figure is the SHAPE of the sampling trajectory, which is the
property Chapter 3's curvature argument turns on. Everything here is computed
analytically rather than sketched: with Gaussian-mixture data and Gaussian
noise, both the rectified-flow marginal velocity and the diffusion score have
closed forms, so the curves are exact for this toy problem.

    (a) diffusion probability-flow ODE
    (b) rectified flow with INDEPENDENT coupling
    (c) after one reflow -> straight by construction

    NOTE, and it is the point of the figure: (b) is not straighter than (a).
    With well-separated modes a noise sample near the origin has an ambiguous
    destination, so the marginal field bends late as the path commits to a
    mode. Straightness is what REFLOW buys, not what the linear interpolant
    buys on its own. Measured mean |trajectory - chord| is printed at the end
    and annotated on each panel; do not caption this figure as "diffusion
    curved, rectified flow straight".

Run:  <repo>/.venv/bin/python trajectory_curvature.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- problem
# data: two well-separated modes; noise: standard normal
MODES = np.array([-2.2, 2.2])
W = np.array([0.5, 0.5])
S = 0.25                      # data std within a mode
N_TRAJ = 17
T = np.linspace(0.0, 1.0, 400)   # t=0 noise, t=1 data (flow-matching convention)


def _posterior(x, mean, var, w):
    """Responsibilities of each mixture component, shape (K, ...)."""
    logp = -0.5 * ((x[None] - mean[:, None]) ** 2) / var[:, None] \
           - 0.5 * np.log(2 * np.pi * var[:, None]) + np.log(w)[:, None]
    logp -= logp.max(axis=0, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=0, keepdims=True)


def rf_velocity(x, t):
    """E[x1 - x0 | x_t = x] for x_t = (1-t)x0 + t x1, x0~N(0,1), x1~mixture.

    Per component the pair (x_t, x1-x0) is jointly Gaussian, so the conditional
    expectation is affine; the mixture case weights those by the posterior.
    """
    var_t = (1 - t) ** 2 + (t ** 2) * S ** 2          # Var(x_t | k)
    cov = t * S ** 2 - (1 - t)                        # Cov(x_t, x1-x0 | k)
    mean_t = t * MODES                                 # E[x_t | k]
    r = _posterior(x, mean_t, np.full_like(MODES, var_t), W)
    per_k = MODES[:, None] + (cov / var_t) * (x[None] - mean_t[:, None])
    return (r * per_k).sum(axis=0)


def diffusion_velocity(x, t):
    """Probability-flow ODE velocity for a VP schedule, in the same time
    convention (t=0 noise, t=1 data), so it is directly comparable to (b)."""
    s = 1.0 - t                                        # VP time: 0 data, 1 noise
    beta0, beta1 = 0.1, 12.0
    integral = beta0 * s + 0.5 * (beta1 - beta0) * s ** 2
    abar = np.exp(-integral)
    beta = beta0 + (beta1 - beta0) * s
    var = abar * S ** 2 + (1 - abar)
    mean = np.sqrt(abar) * MODES
    r = _posterior(x, mean, np.full_like(MODES, var), W)
    score = (r * (-(x[None] - mean[:, None]) / var)).sum(axis=0)
    # dx/ds for the PF ODE; negate because our t runs the other way
    return -(-0.5 * beta * x - 0.5 * beta * score)


def integrate(velocity, x0, ts):
    """Explicit RK4 along ts."""
    xs = [x0]
    x = x0.copy()
    for a, b in zip(ts[:-1], ts[1:]):
        h = b - a
        k1 = velocity(x, a)
        k2 = velocity(x + 0.5 * h * k1, a + 0.5 * h)
        k3 = velocity(x + 0.5 * h * k2, a + 0.5 * h)
        k4 = velocity(x + h * k3, b)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return np.stack(xs)


# ---------------------------------------------------------------- compute
x0 = np.linspace(-2.4, 2.4, N_TRAJ)

traj_diff = integrate(diffusion_velocity, x0, T)
traj_rf = integrate(rf_velocity, x0, T)
# reflow: re-couple each noise sample to the endpoint its own ODE reached,
# after which the conditional path is a straight segment by construction
x1_star = traj_rf[-1]
traj_reflow = (1 - T)[:, None] * x0[None] + T[:, None] * x1_star[None]

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 0.9, "figure.dpi": 200,
})
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.15), sharey=True)
INK, ACCENT = "#1f2933", "#b45309"

def bend(traj):
    chord = (1 - T)[:, None] * traj[0][None] + T[:, None] * traj[-1][None]
    return np.abs(traj - chord).mean()

panels = [
    (traj_diff,   "(a) diffusion probability flow"),
    (traj_rf,     "(b) rectified flow, independent coupling"),
    (traj_reflow, "(c) after one reflow"),
]
for ax, (traj, title) in zip(axes, panels):
    for j in range(N_TRAJ):
        ax.plot(T, traj[:, j], color=INK, alpha=0.55)
    # straight reference from each start to its own endpoint
    for j in range(N_TRAJ):
        ax.plot([0, 1], [traj[0, j], traj[-1, j]], color=ACCENT,
                lw=0.6, ls=(0, (2.5, 2.0)), alpha=0.85, zorder=0)
    ax.set_title(title, pad=4)
    ax.text(0.5, -3.05, rf"mean deviation from chord $= {bend(traj):.2f}$",
            ha="center", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("$t$")
    ax.set_xlim(0, 1); ax.set_ylim(-3.4, 3.4)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0\nnoise", "0.5", "1\ndata"])

axes[0].set_ylabel("$x$")
axes[0].plot([], [], color=INK, alpha=0.55, label="trajectory")
axes[0].plot([], [], color=ACCENT, lw=0.6, ls=(0, (2.5, 2.0)), label="straight reference")
axes[0].legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.8)

fig.tight_layout(pad=0.4)
out = "/home/lukas/projects/thesis-vault/70_Thesis/latex/figures/trajectory-curvature.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# quantify the departure from straightness, for the caption
for name, traj in [("diffusion", traj_diff), ("rectified flow", traj_rf),
                   ("reflow", traj_reflow)]:
    chord = (1 - T)[:, None] * traj[0][None] + T[:, None] * traj[-1][None]
    print(f"  {name:16s} mean |trajectory - chord| = {np.abs(traj - chord).mean():.4f}")
