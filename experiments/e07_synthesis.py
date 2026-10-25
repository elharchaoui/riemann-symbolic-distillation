"""Synthesis - fit the discovered equations together.

Loop 1 (e01 + e03): predict zeros from primes only.
    N(t) = t/2pi log(t/2pi e) + 7/8 + S(t),  S(t) ~ -1/pi sum_{p^k<=X} sin(k t log p)/(k p^{k/2})
    gamma_n := root of N(t) = n - 1/2 near the smooth guess. Accuracy vs X.
Loop 2 (e04): rebuild psi(x) from those predicted zeros with the distilled explicit formula
    (psi(x) - x)/sqrt(x) = -sum_gamma (cos(g u) + 2 g sin(g u))/(g^2 + 1/4) - log(2 pi)/sqrt(x).
    primes -> zeros -> primes: how much of the prime-counting error comes back?
Loop 3 (e02 + e05): GUE small gaps + collision time ~ -gap^2/2 -> collision times approach 0 from below
    as height grows (the heuristic behind Newman's conjecture Lambda >= 0, proved by Rodgers-Tao).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from rh import RESULTS, data, sr

g = data.zeros()
NZ = 2000
out = {}

# ---------------- Loop 1: zeros from primes -----------------------------------------------
Nsm = lambda t: t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e)) + 7 / 8


def S_primes(t, X, smooth):
    lam = data.von_mangoldt(X)
    ns = np.nonzero(lam)[0]
    w = lam[ns] / (np.sqrt(ns) * np.log(ns))  # = 1/(k p^{k/2}) for n = p^k (the e03 law)
    if smooth:
        w = w * (1 - np.log(ns) / np.log(X)) ** 2  # Selberg-type taper
    return -np.sum(w[None, :] * np.sin(np.outer(np.atleast_1d(t), np.log(ns))), 1) / np.pi


def predict_zeros(X, smooth=True, first=1):
    pred = np.empty(NZ)
    for k in range(first, first + NZ):
        t0 = brentq(lambda t: Nsm(t) - (k - 0.5), 5, 1e7)
        if X < 2:
            pred[k - first] = t0
            continue
        F = lambda t: Nsm(t) + S_primes(t, X, smooth)[0] - (k - 0.5)
        tt = np.linspace(t0 - 2.5, t0 + 2.5, 101) * 1.0
        Fv = Nsm(tt) + S_primes(tt, X, smooth) - (k - 0.5)
        roots = [brentq(F, a, b) for a, b, fa, fb in zip(tt, tt[1:], Fv, Fv[1:]) if fa * fb < 0]
        pred[k - first] = min(roots, key=lambda r: abs(r - t0)) if roots else t0
    return pred


true = g[:NZ]
FIRST_HI = 1_000_001
loop1, loop1_hi = {}, {}
for first, store in [(1, loop1), (FIRST_HI, loop1_hi)]:
    tr = g[first - 1:first - 1 + NZ]
    spacing = 2 * np.pi / np.log(tr / (2 * np.pi))
    for X in [1, 10, 100, 1000, 10000]:
        p = predict_zeros(X, first=first)
        err = np.abs(p - tr) / spacing
        nearest = np.abs(tr[None, :] - p[:, None]).argmin(1) == np.arange(NZ)
        store[X] = {"median_err_spacings": float(np.median(err)), "rms_err": float(np.sqrt(np.mean(err**2))),
                    "frac_correct_index": float(nearest.mean()), "pred": p}
        print(f"Loop1 zeros #{first}..: X={X:6d}: median |error| = {np.median(err):.3f} spacings, "
              f"zero identified correctly {100*nearest.mean():.1f}%")
out["loop1"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in loop1.items()}
out["loop1_hi"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in loop1_hi.items()}

# ---------------- Loop 2: back to primes ------------------------------------------------
XMAX = 10**5
psi = data.psi(XMAX)
u = np.linspace(np.log(20), np.log(XMAX - 1), 20000)
x = np.exp(u)
y = (psi[np.floor(x).astype(int)] - x) / np.sqrt(x)


def rebuild(gam):
    return (-(np.cos(np.outer(u, gam)) + 2 * gam * np.sin(np.outer(u, gam))) / (gam**2 + 0.25)).sum(1) \
        - np.log(2 * np.pi) / np.sqrt(x)


loop2 = {}
for label, gam in [("true zeros", true), ("zeros from primes X=10000", loop1[10000]["pred"]),
                   ("zeros from primes X=100", loop1[100]["pred"]), ("smooth guess (no primes)", loop1[1]["pred"])]:
    yy = rebuild(gam)
    r2 = 1 - np.var(y - yy) / np.var(y)
    loop2[label] = r2
    print(f"Loop2 psi(x) rebuilt from {label:28s}: R^2 = {r2:.3f}")
out["loop2_r2"] = loop2

# ---------------- Loop 3: GUE gaps x collision law -> Lambda >= 0 heuristic ----------------
# collision time of a pair with gamma-gap d (pair-only, e05 law dz/dt = 2/(zi-zj), z = 2 gamma): t_c = -d^2/2
# expected smallest normalised gap among M GUE spacings: P(s < e) ~ (pi^2/9) e^3  -> s_min ~ (9/(pi^2 M))^(1/3)
rows = []
for T in [1e4, 1e6, 1e12, 1e21, 1e30, 1e50, 1e100]:
    mean_gap = 2 * np.pi / np.log(T / (2 * np.pi))
    M = T / mean_gap * 0.01  # zeros in a window of relative width 1% around T
    s_min = (9 / (np.pi**2 * M)) ** (1 / 3)
    rows.append((T, mean_gap, s_min, -(s_min * mean_gap) ** 2 / 2))
rows = np.array(rows)
print("Loop3: height T, mean gap, expected min normalised gap, collision time t_c")
for r in rows:
    print("   %.0e  %.3e  %.3e  %.3e" % tuple(r))
out["loop3"] = rows

sr.save("e07", out)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
Xs = list(loop1)
ax[0].semilogx([max(X, 1) for X in Xs], [loop1[X]["median_err_spacings"] for X in Xs], "o-", label="median error (spacings)")
ax[0].semilogx([max(X, 1) for X in Xs], [loop1_hi[X]["median_err_spacings"] for X in Xs], "o--", label="median error, zeros #1e6..")
ax[0].semilogx([max(X, 1) for X in Xs], [1 - loop1_hi[X]["frac_correct_index"] for X in Xs], "s-", label="misidentified, zeros #1e6..")
ax[0].set_xlabel("primes used up to X"); ax[0].legend(); ax[0].set_title("Loop 1: zeros from primes")
m = (u > 8) & (u < 8.5)
ax[1].plot(u[m], y[m], "k.", ms=1, label="true (ψ-x)/√x")
ax[1].plot(u[m], rebuild(loop1[10000]["pred"])[m], "r-", lw=.7, label="via zeros predicted from primes")
ax[1].legend(fontsize=7); ax[1].set_title("Loop 2: primes → zeros → primes")
ax[2].loglog(rows[:, 0], -rows[:, 3], "o-")
ax[2].set_xlabel("height T"); ax[2].set_ylabel("|collision time|")
ax[2].set_title("Loop 3: closest pairs collide ever nearer t=0⁻")
plt.tight_layout(); plt.savefig(RESULTS / "e07.png", dpi=110)
