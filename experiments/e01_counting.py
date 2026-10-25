"""Angle 1 - counting zeros.

A. A network learns N(T) (number of zeros up to height T) from raw counts.
   PySR distils the network into a formula (expect Riemann-von Mangoldt
   N(T) ~ T/2pi log(T/2pi) - T/2pi + 7/8). A second network+PySR pass on the
   residual recovers the constant term. KAN is run on the same task.
B. Fluctuation scale: Var S(t) over windows at heights 1e3..1e21 (Selberg CLT
   predicts Var S ~ log log T / (2 pi^2)).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import torch

from rh import RESULTS, data, nets, sr

rng = np.random.default_rng(0)
g = data.zeros()
out = {}

# ---------------- A1: network learns N(T), PySR reads it ----------------------------
# Representation matters: N(T)/T as a function of L = log T is smooth and slowly varying,
# so the network can learn it to high relative accuracy.
T = np.exp(rng.uniform(np.log(50), np.log(g[-1]), 40000))
N = np.searchsorted(g, T).astype(float)
L = np.log(T)
zs = lambda l: ((l - 8) / 3)[:, None]
net, hist = nets.fit_mlp(zs(L), N / T * 10, epochs=4000, width=64, depth=3)
Lq = np.linspace(np.log(50), np.log(g[-1]), 500)
Tq = np.exp(Lq)
Ynet = nets.predict(net, zs(Lq))[:, 0] / 10  # network's N(T)/T
print("A1 net fit rel-rmse", np.sqrt(np.mean((Ynet * Tq - np.searchsorted(g, Tq)) ** 2 / np.searchsorted(g, Tq) ** 2)))
_, eqA = sr.fit(Lq[:, None], Ynet, ["L"], "e01_NT", unary="log", niterations=100, maxsize=14)
print("A1 PySR best (N/T vs L=log T):", eqA["best"])
print("   theory: L/(2pi) - (1+log 2pi)/(2pi) =", f"{1/(2*np.pi):.5f}*L - {(1+np.log(2*np.pi))/(2*np.pi):.5f}")
out["A1_network_distilled"] = eqA
# refine the elbow form a*L + b on raw counts (T > 1000), then identify the constants
m = T > 1000
A = np.c_[L[m], np.ones(m.sum())]
coef = np.linalg.lstsq(A * T[m, None], N[m], rcond=None)[0]  # weight: absolute error in N
ids = [sr.recognize(c, tol=1e-4 * abs(c)) for c in coef]
print("A1 refined on raw data: a=%.7f b=%.7f ->" % tuple(coef), ids)
out["A1_refined"] = {"a": coef[0], "b": coef[1], "identified": ids}
out["A1_theory"] = f"{1/(2*np.pi):.6f}*L - {(1+np.log(2*np.pi))/(2*np.pi):.6f}"

# ---------------- A2: residual after main term -> constant -------------------------
# The constant is invisible at the network's accuracy, so look at the exact residual,
# binned (S(T) averages out within a bin), and let PySR choose among simple forms.
main = lambda t: t / (2 * np.pi) * np.log(t / (2 * np.pi)) - t / (2 * np.pi)
edges = np.exp(np.linspace(np.log(100), np.log(g[-1]), 13))
bins = []
for a, b in zip(edges[:-1], edges[1:]):
    tr = rng.uniform(a, b, 40000)
    R = np.searchsorted(g, tr) - main(tr)  # = 7/8 + S(T) + O(1/T)
    bins.append((np.sqrt(a * b), R.mean(), R.std() / np.sqrt(len(R))))
bins = np.array(bins)
_, eqR = sr.fit(np.log(bins[:, :1]), bins[:, 1], ["L"], "e01_resid", unary="log", niterations=60,
                maxsize=8, weights=1 / bins[:, 2] ** 2)
wmean = np.sum(bins[:, 1] / bins[:, 2] ** 2) / np.sum(1 / bins[:, 2] ** 2)
werr = 1 / np.sqrt(np.sum(1 / bins[:, 2] ** 2))
chi2 = np.sum(((bins[:, 1] - wmean) / bins[:, 2]) ** 2)
out["A2_identified"] = sr.recognize(wmean, tol=3 * werr)
print("A2 identified:", out["A2_identified"])
print(f"A2 residual weighted mean {wmean:.5f} +- {werr:.5f} (chi2={chi2:.1f}, dof={len(bins)-1}); PySR {eqR['best']}")
out["A2_residual"] = {"bins": bins, "mean": wmean, "stderr": werr, "chi2": chi2, "pysr": eqR, "theory": 7 / 8}

# ---------------- A3: KAN on the same representation --------------------------------
try:
    from kan import KAN
    import sympy as sp

    x = torch.tensor(zs(Lq), dtype=torch.float32)
    y = torch.tensor(Ynet[:, None], dtype=torch.float32)
    ds = {"train_input": x, "train_label": y, "test_input": x, "test_label": y}
    kan = KAN(width=[1, 1], grid=5, k=3, seed=0, auto_save=False)
    kan.fit(ds, opt="LBFGS", steps=100, update_grid=True)
    kan.auto_symbolic(lib=["x", "x^2", "log", "exp", "sin"], verbose=0)
    kan.fit(ds, opt="LBFGS", steps=50)
    f = sp.expand(kan.symbolic_formula()[0][0].subs(sp.Symbol("x_1"), (sp.Symbol("L") - 8) / 3))
    print("A3 KAN formula (N/T vs L):", sp.N(f, 5))
    out["A3_kan"] = str(sp.N(f, 6))
except Exception as e:  # KAN is a bonus; never block the run
    print("KAN failed:", e)
    out["A3_kan"] = f"failed: {e}"

# ---------------- B: fluctuation scale of S(t) --------------------------------------
pts = []
W = 40000
for i0 in range(0, len(g) - W, W):
    ts = rng.uniform(g[i0], g[i0 + W], 20000)
    S = data.S_grid(ts, g)
    pts.append((np.exp(np.mean(np.log(ts))), S.var(), ((S - S.mean()) ** 4).mean() / S.var() ** 2))
# small heights: finer windows
for a, b in [(100, 1000), (1000, 5000), (5000, 20000)]:
    ts = rng.uniform(a, b, 20000)
    S = data.S_grid(ts, g)
    pts.append((np.sqrt(a * b), S.var(), ((S - S.mean()) ** 4).mean() / S.var() ** 2))
mp.mp.dps = 40
for w in ["1e12", "1e21"]:
    off, loc, first = data.high_zeros(w)
    ts = rng.uniform(loc[0], loc[-1], 4000)
    Nn = [first - 1 + int(k) for k in np.searchsorted(loc, ts)]
    S = np.array([float(mp.mpf(n) - mp.siegeltheta(mp.mpf(off) + mp.mpf(float(t))) / mp.pi - 1)
                  for n, t in zip(Nn, ts)])
    pts.append((float(off) + loc.mean(), S.var(), ((S - S.mean()) ** 4).mean() / S.var() ** 2))
pts = np.array(sorted(pts))
LL = np.log(np.log(pts[:, 0]))
print("B points (T, var, kurtosis):\n", np.c_[pts[:, 0], pts[:, 1], pts[:, 2]])
_, eqB = sr.fit(np.log(pts[:, :1]), pts[:, 1], ["logT"], "e01_varS", unary="log", niterations=80, maxsize=10)
lin = np.polyfit(LL, pts[:, 1], 1)
print("B PySR:", eqB["best"], " linear fit in loglogT:", lin, " theory slope 1/(2pi^2) =", 1 / (2 * np.pi**2))
out["B_varS"] = {"points": pts, "pysr": eqB, "slope_loglog": lin[0], "intercept": lin[1],
                 "theory_slope": 1 / (2 * np.pi**2)}

# ---------------- figure ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(Lq, np.searchsorted(g, Tq) / Tq, "k.", ms=2, label="true N(T)/T")
ax[0].plot(Lq, Ynet, "r-", lw=1, label="network")
ax[0].set_xlabel("log T"); ax[0].legend(); ax[0].set_title("A1: N(T)/T")
ax[1].errorbar(bins[:, 0], bins[:, 1], bins[:, 2], fmt="o", label="binned N - main term")
ax[1].axhline(7 / 8, c="k", ls="--", label="7/8")
ax[1].set_xscale("log"); ax[1].legend(); ax[1].set_title("A2: constant term")
ax[2].plot(LL, pts[:, 1], "o", label="Var S(t)")
xx = np.linspace(LL.min(), LL.max(), 50)
ax[2].plot(xx, np.polyval(lin, xx), "r-", label=f"fit slope {lin[0]:.4f}")
ax[2].plot(xx, xx / (2 * np.pi**2) + (lin[1] + (lin[0] - 1 / (2 * np.pi**2)) * LL.mean()), "k--",
           label="slope 1/2π²")
ax[2].set_xlabel("log log T"); ax[2].legend(); ax[2].set_title("B: Selberg fluctuation scale")
plt.tight_layout(); plt.savefig(RESULTS / "e01.png", dpi=110)
sr.save("e01", out)
