"""L-angle 2 - local statistics of critical-line zeros for zeta, L(s, chi_-4), DH (same tools as e02).

Unfolding: x_n = P(N_smooth(gamma_n)) with P a cubic fitted to the index n, so the line zeros of every
function have mean spacing 1 (DH loses ~7% of its zeros to off-line pairs, which this absorbs).
A. repulsion exponent beta (MLE of s^beta exp(-b s^2)) + density network distilled by PySR.
B. pair correlation chi^2 vs Montgomery; form factor F(alpha) distilled by PySR (with min/max).
C. number variance vs exact GUE.
D. DH only: where do the off-line zeros sit relative to the line zeros?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from rh import RESULTS, lfun, nets, sr

rng = np.random.default_rng(12)
fam = lfun.family()
out = {"A": {}, "B": {}, "B2": {}, "C": {}}
grid = np.linspace(1e-3, 4, 800)
dg = grid[1] - grid[0]
fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))


def unfold(F):
    g = F["g"][F["g"] > 1000]
    y = F["ns"](g)
    P = np.polyfit(y, np.arange(len(g)), 3)
    return np.polyval(P, y), g


def gue_numvar(Lw):
    uu = np.linspace(1e-9, Lw, 4000)
    return Lw - 2 * np.trapezoid((Lw - uu) * np.sinc(uu) ** 2, uu)


Ls = np.exp(np.linspace(np.log(0.2), np.log(30), 30))
GUE = np.array([gue_numvar(Lw) for Lw in Ls])
alphas = np.linspace(0.05, 2.0, 140)
U = 3.0
edges = np.linspace(0, U, 121)
centers = (edges[1:] + edges[:-1]) / 2
mont = 1 - (np.sin(np.pi * centers) / (np.pi * centers)) ** 2

for name, F in fam.items():
    x, gl = unfold(F)
    s = np.diff(x)
    s = s[s < 4]
    # ---- A: repulsion
    model = nets.MLP(1, 1, width=48, depth=3).to(nets.DEV)
    S_, G_ = nets.tensor(s[:, None]), nets.tensor(grid[:, None])

    def nll():
        return -(model(S_)[:, 0].mean() - torch.logsumexp(model(G_)[:, 0], 0) - np.log(dg))

    nets.train(model, nll, epochs=1500, lr=3e-3)
    with torch.no_grad():
        f = model(G_)[:, 0]
        logp = (f - torch.logsumexp(f, 0) - np.log(dg)).cpu().numpy()
    m = (grid > 0.05) & (grid < 3.0)
    _, eq = sr.fit(np.c_[grid[m], np.log(grid[m])], logp[m], ["s", "logs"], f"l02_spacing_{name}",
                   unary="basic", binary=("+", "-", "*"), niterations=80, maxsize=13)

    def fam_nll(th):
        beta, b = th
        lp = beta * np.log(grid) - b * grid**2
        return -np.mean(beta * np.log(s) - b * s**2 - np.log(np.sum(np.exp(lp)) * dg))

    res = minimize(fam_nll, [1.5, 1.0], method="Nelder-Mead")
    e = np.array([1e-3, 0])
    se = 1 / np.sqrt((fam_nll(res.x + e) - 2 * res.fun + fam_nll(res.x - e)) / 1e-6 * len(s))
    small = np.mean(s < 0.25)
    print(f"A {name}: n={len(s)} beta={res.x[0]:.3f}+-{se:.3f}; P(s<0.25)={small:.4f} (GUE 0.0170);"
          f" PySR elbow {eq['elbow']}", flush=True)
    out["A"][name] = {"n": len(s), "beta": res.x[0], "beta_se": se, "b": res.x[1], "p_small": small, "pysr": eq,
                      "grid": grid[::8], "logp": logp[::8], "hist": np.histogram(s, 80, (0, 4), density=True)[0]}
    ax[0].plot(grid, np.exp(logp), label=F["label"])
    # ---- B: pair correlation
    cnt = np.zeros(len(centers))
    for k in range(1, 12):
        d = x[k:] - x[:-k]
        cnt += np.histogram(d[d < U], edges)[0]
    R2 = cnt / (len(x) * (edges[1] - edges[0]))
    err = np.sqrt(np.maximum(cnt, 1)) / (len(x) * (edges[1] - edges[0]))
    chi2 = np.mean(((R2 - mont) / err) ** 2)
    out["B"][name] = {"R2": R2, "u": centers, "chi2_per_bin_vs_montgomery": chi2}
    ax[1].plot(centers, R2, ".", ms=3, label=F["label"])
    # ---- B2: form factor
    seg, Ff, nseg = 1000, np.zeros(len(alphas)), 0
    for i0 in range(0, len(x) - seg, seg):
        xs_ = x[i0:i0 + seg] - x[i0]
        w = np.sin(np.pi * xs_ / xs_[-1]) ** 2
        Ff += np.abs(np.exp(2j * np.pi * alphas[:, None] * xs_[None, :]) @ w) ** 2 / np.sum(w**2)
        nseg += 1
    Ff /= nseg
    netF, _ = nets.fit_mlp(alphas[:, None], Ff, epochs=2000, width=32, depth=2)
    Fn = nets.predict(netF, alphas[:, None])[:, 0]
    _, eqF = sr.fit(alphas[:, None], Fn, ["a"], f"l02_formfactor_{name}", unary="basic",
                    binary=("+", "-", "*", "/", "min", "max"), niterations=80, maxsize=9)
    rmsF = np.sqrt(np.mean((Ff - np.minimum(alphas, 1)) ** 2))
    out["B2"][name] = {"alpha": alphas, "F": Ff, "pysr": eqF, "rms_vs_montgomery": rmsF}
    ax[2].plot(alphas, Ff, "-", lw=.9, label=F["label"])
    # ---- C: number variance
    var = []
    for Lw in Ls:
        a = rng.uniform(x[0], x[-1] - Lw, 20000)
        var.append((np.searchsorted(x, a + Lw) - np.searchsorted(x, a)).var())
    var = np.array(var)
    dev = np.sqrt(np.mean((var[Ls < 5] - GUE[Ls < 5]) ** 2))
    out["C"][name] = {"L": Ls, "var": var, "rms_dev_gue_L<5": dev}
    ax[3].semilogx(Ls, var, ".-", ms=3, label=F["label"])
    print(f"B {name}: R2 chi2/bin vs Montgomery {chi2:.2f}; form factor rms vs min(a,1) {rmsF:.3f},"
          f" PySR {eqF['elbow']}; C: number variance rms dev from GUE (L<5) {dev:.4f}", flush=True)

ax[0].plot(grid, 32 / np.pi**2 * grid**2 * np.exp(-4 * grid**2 / np.pi), "k--", lw=.8, label="GUE surmise")
ax[0].set_xlim(0, 3.5); ax[0].legend(fontsize=7); ax[0].set_title("spacing density (line zeros)")
ax[1].plot(centers, mont, "k-", lw=.8, label="Montgomery"); ax[1].legend(fontsize=7); ax[1].set_title("pair correlation")
ax[2].plot(alphas, np.minimum(alphas, 1), "k--", lw=.8, label="min(α,1)"); ax[2].legend(fontsize=7)
ax[2].set_title("form factor")
ax[3].semilogx(Ls, GUE, "k--", label="GUE"); ax[3].legend(fontsize=7); ax[3].set_title("number variance")

# ---------------- D: DH off-line zeros vs the line-zero gaps ------------------------------
F = fam["dh"]
x, gl = unfold(F)
og = F["off"][:, 1]
og = og[(og > gl[0]) & (og < gl[-1])]
idx = np.searchsorted(gl, og) - 1  # gap (gl[idx], gl[idx+1]) that contains the off-line pair
gap_with = np.diff(x)[idx]
gap_all = np.diff(x)
pos = (og - gl[idx]) / (gl[idx + 1] - gl[idx])
sig = F["off"][(F["off"][:, 1] > gl[0]) & (F["off"][:, 1] < gl[-1]), 0]
cc = np.corrcoef(gap_with, sig)[0, 1]
print(f"D DH: gaps holding an off-line pair: mean {gap_with.mean():.3f} (all gaps {gap_all.mean():.3f});"
      f" share of gaps > 2 that hold one: {np.mean(np.isin(np.nonzero(gap_all > 2)[0], idx)):.3f};"
      f" corr(gap, Re rho) {cc:.3f}", flush=True)
out["D"] = {"gap_with_mean": gap_with.mean(), "gap_with_median": np.median(gap_with),
            "share_big_gaps_with_pair": np.mean(np.isin(np.nonzero(gap_all > 2)[0], idx)),
            "corr_gap_sigma": cc, "rel_position_mean": pos.mean(),
            "hist_with": np.histogram(gap_with, 40, (0, 5), density=True)[0],
            "hist_all": np.histogram(gap_all, 40, (0, 5), density=True)[0]}
plt.tight_layout(); plt.savefig(RESULTS / "l02.png", dpi=110)
sr.save("l02", out)
