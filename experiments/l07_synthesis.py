"""L-angle 7 - synthesis: which of the distilled laws need the Euler product?

Loop 1 (l01 + l03): predict zeros from the Dirichlet coefficients alone,
    N_smooth(t) + S_X(t) = k - 1/2,  S_X(t) = -(1/pi) sum_{n<=X} b_n n^{-1/2} sin(t log n) w(n),
    b_n = coefficients of log F (zeta: Lambda(n)/log n; chi4: chi(n) Lambda(n)/log n; DH: no Euler product).
    Error = distance from each predicted point to the nearest true zero height, in mean spacings.
Gram locking (found while reading the l02 form factor of DH, which has a Bragg peak at alpha = 1/0.907):
    are line zeros pinned to theta(gamma)/pi = 1/2 mod 1 ("slots" of the full zero density)?
    Order parameter |<exp(2 pi i theta(gamma)/pi)>|; where do DH's off-line pairs sit in theta/pi mod 1?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from rh import RESULTS, data, lfun, sr

fam = lfun.family()
T0, T1 = 1000, 3000
XS = [1, 10, 100, 1000, 10000]
out = {"loop1": {}}


def bcoef(name, X):
    if name == "zeta":
        lam = data.von_mangoldt(X)
    else:
        lam = lfun.dirichlet_log_coeffs(lfun.coef(name, X))
    nn = np.arange(X + 1)
    b = np.zeros(X + 1)
    b[2:] = lam[2:] / np.log(nn[2:])
    return b


for name, F in fam.items():
    heights = np.r_[F["g"], F["off"][:, 1]]
    heights = np.sort(heights)
    ns_ = F["ns"]
    k0, k1 = int(np.ceil(ns_(T0) + 0.5)), int(np.floor(ns_(T1) + 0.5))
    res = {}
    for X in XS:
        b = bcoef(name, max(X, 2))
        idx = np.nonzero(b)[0]
        w = b[idx] / np.sqrt(idx) * (1 - np.log(idx) / np.log(max(X, 2))) ** 2  # Selberg-type taper
        S = lambda t: -np.sin(np.outer(np.atleast_1d(t), np.log(idx))) @ w / np.pi if X >= 2 else np.zeros(np.size(t))
        pred = []
        for k in range(k0, k1):
            t0 = brentq(lambda t: ns_(t) - (k - 0.5), T0 - 50, T1 + 50)
            if X < 2:
                pred.append(t0)
                continue
            tt = np.linspace(t0 - 2.5, t0 + 2.5, 101)
            Fv = ns_(tt) + S(tt) - (k - 0.5)
            roots = [brentq(lambda t: ns_(t) + S(t)[0] - (k - 0.5), a, bb)
                     for a, bb, fa, fb in zip(tt, tt[1:], Fv, Fv[1:]) if fa * fb < 0]
            pred.append(min(roots, key=lambda r: abs(r - t0)) if roots else t0)
        pred = np.array(pred)
        sp_ = lfun.mean_spacing("chi4", pred) if name != "zeta" else 2 * np.pi / np.log(pred / (2 * np.pi))
        if name == "dh":
            sp_ = lfun.mean_spacing("dh", pred)
        j = np.clip(np.searchsorted(heights, pred), 1, len(heights) - 1)
        err = np.minimum(np.abs(pred - heights[j]), np.abs(pred - heights[j - 1])) / sp_
        res[X] = {"median_err": float(np.median(err)), "mean_err": float(err.mean())}
        print(f"loop1 {name}: X={X:6d} median distance to nearest zero {np.median(err):.3f} spacings", flush=True)
    out["loop1"][name] = res

# ---------------- Gram locking ----------------------------------------------------------------
out["gram"] = {}
for name, F in fam.items():
    g = F["g"][F["g"] > 1000]
    ph = F["ns"](g) - (1 if name == "zeta" else 0)  # theta(gamma)/pi
    m = np.exp(2j * np.pi * ph).mean()
    fr = np.mod(ph, 1)
    res = {"order": abs(m), "phase": np.angle(m) / (2 * np.pi), "central_half": np.mean((fr > 0.25) & (fr < 0.75)),
           "hist_line": np.histogram(fr, 20, (0, 1), density=True)[0]}
    if name == "dh":
        fo = np.mod(lfun.n_smooth("dh", F["off"][F["off"][:, 1] > 1000, 1]), 1)
        res["hist_off"] = np.histogram(fo, 20, (0, 1), density=True)[0]
        res["off_near_boundary"] = np.mean((fo < 0.25) | (fo > 0.75))
    out["gram"][name] = res
    print(f"gram {name}: order |<e^(2 pi i theta/pi)>| = {abs(m):.3f}; share in central half of slot "
          f"{res['central_half']:.3f}" + (f"; off-line pairs near slot boundary {res['off_near_boundary']:.3f}"
                                          if name == "dh" else ""), flush=True)

sr.save("l07", out)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax = axs[0]
for name, r in out["loop1"].items():
    ax.semilogx(XS, [r[X]["median_err"] for X in XS], "o-", label=fam[name]["label"])
ax.set_xlabel("Dirichlet coefficients used, n ≤ X"); ax.set_ylabel("median error (mean spacings)")
ax.set_title("zeros predicted from the coefficients alone"); ax.legend()
xb = (np.arange(20) + 0.5) / 20
for name, r in out["gram"].items():
    axs[1].plot(xb, r["hist_line"], "o-", ms=3, label=f"{fam[name]['label']} line zeros (order {r['order']:.2f})")
axs[1].plot(xb, out["gram"]["dh"]["hist_off"], "s--", c="C3", ms=3, label="DH off-line pairs")
axs[1].set_xlabel("θ(γ)/π mod 1"); axs[1].set_ylabel("density"); axs[1].legend(fontsize=7)
axs[1].set_title("Gram locking: where zeros sit in their slot")
plt.tight_layout(); plt.savefig(RESULTS / "l07.png", dpi=110)
