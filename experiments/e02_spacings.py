"""Angle 2 - local statistics of zeros (random-matrix view).

A. Nearest-neighbour spacings of unfolded zeros at 4 heights. A density network
   p(s) = exp(f(s))/Z is trained by maximum likelihood; PySR distils f(s).
   The exponent of s near 0 is the level-repulsion index beta (GUE: beta = 2).
B. Pair correlation R2(u): network smooths the histogram, PySR distils it
   (Montgomery's conjecture: 1 - (sin(pi u)/(pi u))^2).
C. Number variance Sigma^2(L): GUE predicts (log(2 pi L) + euler + 1)/pi^2 for
   large L; for zeta it saturates at L ~ log T (Berry: the primes take over).
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

from rh import RESULTS, data, nets, sr

rng = np.random.default_rng(1)
g = data.zeros()


def unfolded_sets():
    sets = {}
    sets["T~1e4"] = (data.unfold(g[1000:100000]), 3.5e4)
    sets["T~1e6"] = (data.unfold(g[-100000:]), 1.1e6)
    for w in ["1e12", "1e21"]:
        off, loc, _ = data.high_zeros(w)
        T = float(off) + loc.mean()
        # density is constant to 1e-15 across the window: x = gamma * log(T/2pi)/2pi
        sets[f"T~{w}"] = ((loc - loc[0]) * np.log(T / (2 * np.pi)) / (2 * np.pi), T)
    return sets


sets = unfolded_sets()
out = {"A": {}, "B": {}, "C": {}}
grid = np.linspace(1e-3, 4, 800)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))

# ---------------- A: spacing density network + distillation --------------------------
class LogDensity(torch.nn.Module):
    """f(s) = beta_free network; density = exp(f)/Z on [0, 4]."""

    def __init__(self):
        super().__init__()
        self.f = nets.MLP(1, 1, width=48, depth=3)

    def forward(self, s):
        return self.f(s)


def gue_surmise(s):
    return 32 / np.pi**2 * s**2 * np.exp(-4 * s**2 / np.pi)


for name, (x, T) in sets.items():
    s = np.diff(x)
    s = s / s.mean()
    s = s[s < 4]
    model = LogDensity().to(nets.DEV)
    S = nets.tensor(s[:, None])
    G = nets.tensor(grid[:, None])
    dg = grid[1] - grid[0]

    def nll():
        logZ = torch.logsumexp(model(G)[:, 0], 0) + np.log(dg)
        return -(model(S)[:, 0].mean() - logZ)

    nets.train(model, nll, epochs=1500, lr=3e-3)
    with torch.no_grad():
        f = model(G)[:, 0]
        logp = (f - torch.logsumexp(f, 0) - np.log(dg)).cpu().numpy()
    # distil the network's log-density (SymTorch-style: what the sub-module computes)
    m = (grid > 0.05) & (grid < 3.0)
    # variables s and log s, polynomial operators only: looks for c + beta*log s - b*s^2 + ...
    _, eq = sr.fit(np.c_[grid[m], np.log(grid[m])], logp[m], ["s", "logs"], f"e02_spacing_{name}",
                   unary="basic", binary=("+", "-", "*"), niterations=100, maxsize=13)
    # direct MLE for the repulsion family p ~ s^beta exp(-b s^2) (normalised numerically)
    def fam_nll(th):
        beta, b = th
        lp = beta * np.log(grid) - b * grid**2
        logZ = np.log(np.sum(np.exp(lp)) * dg)
        return -np.mean(beta * np.log(s) - b * s**2 - logZ)

    res = minimize(fam_nll, [1.5, 1.0], method="Nelder-Mead")
    H = np.zeros(2)
    for i in range(2):  # curvature -> standard error
        e = np.zeros(2); e[i] = 1e-3
        H[i] = (fam_nll(res.x + e) - 2 * res.fun + fam_nll(res.x - e)) / 1e-6
    se = 1 / np.sqrt(H * len(s))
    print(f"A {name}: n={len(s)} beta={res.x[0]:.3f}+-{se[0]:.3f} b={res.x[1]:.3f}  PySR elbow: {eq['elbow']}")
    out["A"][name] = {"n": len(s), "beta": res.x[0], "beta_se": se[0], "b": res.x[1], "b_se": se[1],
                      "pysr": eq, "logp_grid": logp[::8], "grid": grid[::8]}
    ax[0].plot(grid, np.exp(logp), label=f"{name} (net)")
ax[0].plot(grid, gue_surmise(grid), "k--", label="GUE surmise")
ax[0].plot(grid, np.exp(-grid), "k:", label="Poisson")
ax[0].set_xlim(0, 3.5); ax[0].legend(fontsize=7); ax[0].set_title("A: spacing density")

# ---------------- B: pair correlation -------------------------------------------------
U = 3.0
edges = np.linspace(0, U, 121)
centers = (edges[1:] + edges[:-1]) / 2
for name, (x, T) in sets.items():
    x = x[:100000]
    cnt = np.zeros(len(centers))
    for k in range(1, 12):
        d = x[k:] - x[:-k]
        cnt += np.histogram(d[d < U], edges)[0]
    R2 = cnt / (len(x) * (edges[1] - edges[0]))
    err = np.sqrt(np.maximum(cnt, 1)) / (len(x) * (edges[1] - edges[0]))
    net, _ = nets.fit_mlp(centers[:, None], R2, epochs=2500, width=32, depth=3)
    R2net = nets.predict(net, centers[:, None])[:, 0]
    _, eq = sr.fit(centers[:, None], R2net, ["u"], f"e02_pair_{name}", unary=["sin", "square"],
                   niterations=300, maxsize=14, populations=64,
                   nested={"sin": {"sin": 0}, "square": {"square": 0}})
    mont = 1 - (np.sin(np.pi * centers) / (np.pi * centers)) ** 2
    chi2 = np.mean(((R2 - mont) / err) ** 2)
    print(f"B {name}: chi2/bin vs Montgomery = {chi2:.2f}  PySR elbow: {eq['elbow']}")
    out["B"][name] = {"pysr": eq, "chi2_per_bin_vs_montgomery": chi2, "R2": R2, "u": centers}
    ax[1].plot(centers, R2, ".", ms=3, label=name)
ax[1].plot(centers, 1 - (np.sin(np.pi * centers) / (np.pi * centers)) ** 2, "k-", label="Montgomery")
ax[1].legend(fontsize=7); ax[1].set_title("B: pair correlation R2(u)")

# ---------------- B2: form factor (Fourier view of pair correlation) ------------------
# F(alpha) = <|sum_n e^{2 pi i alpha x_n}|^2>/N. Montgomery: F = |alpha| (|alpha|<1), 1 beyond.
alphas = np.linspace(0.05, 2.0, 140)
out["B2"] = {}
for name, (x, T) in sets.items():
    seg = 1000
    F = np.zeros(len(alphas))
    nseg = 0
    for i0 in range(0, min(len(x), 100000) - seg, seg):
        xs_ = x[i0:i0 + seg] - x[i0]
        w = np.sin(np.pi * xs_ / xs_[-1]) ** 2  # taper kills the alpha=0 spike
        E = np.exp(2j * np.pi * alphas[:, None] * xs_[None, :]) @ w
        F += np.abs(E) ** 2 / np.sum(w**2)
        nseg += 1
    F /= nseg
    netF, _ = nets.fit_mlp(alphas[:, None], F, epochs=2000, width=32, depth=2)
    Fn = nets.predict(netF, alphas[:, None])[:, 0]
    _, eq = sr.fit(alphas[:, None], Fn, ["a"], f"e02_formfactor_{name}", unary="basic",
                   binary=("+", "-", "*", "/", "min", "max"), niterations=100, maxsize=9)
    mont = np.minimum(alphas, 1)
    print(f"B2 {name}: rms(F - min(a,1)) = {np.sqrt(np.mean((F - mont) ** 2)):.3f}  PySR elbow: {eq['elbow']}")
    out["B2"][name] = {"alpha": alphas, "F": F, "pysr": eq, "rms_vs_montgomery": np.sqrt(np.mean((F - mont) ** 2))}
    ax[1].plot(alphas, F + 1.5, "-", lw=.8)
ax[1].plot(alphas, np.minimum(alphas, 1) + 1.5, "k--", lw=1, label="form factor min(α,1) (+1.5)")

# ---------------- C: number variance --------------------------------------------------
Ls = np.exp(np.linspace(np.log(0.2), np.log(200), 40))


def gue_numvar(Lw):
    """Exact GUE number variance: L - 2 int_0^L (L-u) (sin(pi u)/(pi u))^2 du."""
    uu = np.linspace(1e-9, Lw, 4000)
    return Lw - 2 * np.trapezoid((Lw - uu) * np.sinc(uu) ** 2, uu)


for name, (x, T) in sets.items():
    x = x[: min(len(x), 100000)]
    var = []
    for Lw in Ls:
        a = rng.uniform(x[0], x[-1] - Lw, 20000)
        n = np.searchsorted(x, a + Lw) - np.searchsorted(x, a)
        var.append(n.var())
    var = np.array(var)
    m = Ls < 2
    _, eq = sr.fit(Ls[m, None], var[m], ["L"], f"e02_numvar_{name}", unary="log", niterations=60, maxsize=10)
    sat = var[-8:].mean()
    print(f"C {name}: saturation level {sat:.3f}   (log log T/2pi)/pi^2 scale = {np.log(np.log(T/(2*np.pi)))/np.pi**2:.3f}"
          f"  PySR small-L: {eq['elbow']}")
    gue = np.array([gue_numvar(Lw) for Lw in Ls])
    dev_small = np.sqrt(np.mean((var[Ls < 5] - gue[Ls < 5]) ** 2))
    print(f"   C {name}: rms deviation from exact GUE Sigma^2 for L<5: {dev_small:.4f}")
    out["C"][name] = {"L": Ls, "var": var, "gue": gue, "rms_dev_gue_L<5": dev_small, "saturation": sat, "pysr_smallL": eq, "T": T}
    ax[2].semilogx(Ls, var, ".-", ms=3, label=name)
ax[2].semilogx(Ls, [gue_numvar(Lw) for Lw in Ls], "k--", label="GUE (exact)")
ax[2].legend(fontsize=7); ax[2].set_title("C: number variance Σ²(L)")
plt.tight_layout(); plt.savefig(RESULTS / "e02.png", dpi=110)
sr.save("e02", out)
