"""Angle 6 - Li's criterion: RH <=> lambda_n = sum_rho [1 - (1 - 1/rho)^n] >= 0 for all n.

A. lambda_n for n <= NMAX from 2M zeros (exact sum for low zeros, binomial series in 1/rho for the
   rest, analytic tail beyond the table). Check against known lambda_1..3.
B. Network learns lambda_n / n vs log n; PySR distils it; constants identified.
   (Keiper/Bombieri-Lagarias under RH: lambda_n ~ n/2 (log n - 1 + euler - log 2 pi).)
C. Residual after the asymptotic law: PySR on what is left.
D. Counterfactual: replace zero #1 by an off-line pair 1/2 +- delta + i gamma_1 and watch
   lambda_n: the growth rate of the anomaly is log|1 - 1/rho| > 0 iff Re rho < 1/2.
"""
import sys
from math import comb, lgamma
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, data, nets, sr

NMAX = 20000
NEXACT = 200000
KSER = 18
g = data.zeros()
n = np.arange(1, NMAX + 1)


def li_exact(rhos, n):
    """sum over rho in rhos and conjugates of 1 - (1-1/rho)^n  (rhos: complex array, Im > 0)."""
    w = np.log1p(-1 / rhos)  # principal log of (1 - 1/rho)
    out = np.zeros(len(n))
    for i0 in range(0, len(rhos), 4000):
        wc = w[i0:i0 + 4000]
        for j0 in range(0, len(n), 2000):
            nn = n[j0:j0 + 2000, None]
            out[j0:j0 + 2000] += 2 * np.sum(1 - np.exp(nn * wc[None, :]).real, 1)
    return out


def li_series(rhos, n, K=KSER):
    """Same sum via 1-(1-q)^n = -sum_{k>=1} C(n,k)(-q)^k, q = 1/rho, using moments M_k = sum Re q^k."""
    q = 1 / rhos
    M = np.array([2 * np.sum((q**k).real) for k in range(1, K + 1)])
    out = np.zeros(len(n))
    for k in range(1, K + 1):
        ck = np.exp([lgamma(m + 1) - lgamma(k + 1) - lgamma(m - k + 1) if m >= k else -np.inf for m in n])
        out += -ck * (-1) ** k * M[k - 1]
    return out


rho = 0.5 + 1j * g
lam = li_exact(rho[:NEXACT], n) + li_series(rho[NEXACT:], n)
T = g[-1]
lam += n.astype(float) ** 2 * (np.log(T / (2 * np.pi)) + 1) / (2 * np.pi * T)  # zeros beyond the table
known = [0.0230957089661210338, 0.0923457352280611, 0.2076389205543367]
print("lambda_1..3 computed:", lam[:3], "\n          known:   ", known)
print("min lambda_n (n<=%d): %.4f at n=%d; all positive: %s" % (NMAX, lam.min(), n[lam.argmin()], (lam > 0).all()))

# ---- B: network + PySR on lambda_n / n --------------------------------------------------
L = np.log(n)
zs = lambda l: ((l - 5) / 3)[:, None]
net, _ = nets.fit_mlp(zs(L), lam / n, epochs=4000, width=48, depth=3)
Lq = np.linspace(np.log(20), np.log(NMAX), 400)
Yn = nets.predict(net, zs(Lq))[:, 0]
_, eqB = sr.fit(Lq[:, None], Yn, ["L"], "e06_li", unary="log", niterations=80, maxsize=12)
m = n >= 100
A3 = np.c_[n[m] * L[m], n[m], np.ones(m.sum())]
coef = np.linalg.lstsq(A3 / n[m, None], lam[m] / n[m], rcond=None)[0]  # relative-error weighting
print("B PySR elbow:", eqB["elbow"])
print(f"B fit lambda_n = {coef[0]:.6f} n log n + {coef[1]:.6f} n + {coef[2]:.3f}")
ids = [sr.recognize(coef[0], tol=1e-4), sr.recognize(coef[1], tol=5e-4)]
c_theory = (np.euler_gamma - 1 - np.log(2 * np.pi)) / 2
print("B identified:", ids, f"  theory: 1/2 and (euler-1-log 2pi)/2 = {c_theory:.6f}")

# ---- C: residual ----------------------------------------------------------------------
resid = lam - n / 2 * (np.log(n) - 1 + np.euler_gamma - np.log(2 * np.pi))
_, eqC = sr.fit(n[::20, None].astype(float), resid[::20], ["n"], "e06_resid", unary="log",
                niterations=80, maxsize=12)
print("C residual: mean %.3f, std %.3f, range [%.2f, %.2f]; PySR elbow: %s"
      % (resid.mean(), resid.std(), resid.min(), resid.max(), eqC["elbow"]))
env = np.polyfit(np.log(n[100:]), np.log(np.abs(resid[100:] - resid[100:].mean()) + 1e-9), 1)

# ---- D: counterfactual off-line zero ---------------------------------------------------
cf = {}
for delta in (0.05, 0.1, 0.2):
    pair = np.array([0.5 + delta + 1j * g[0], 0.5 - delta + 1j * g[0]])
    lam_cf = lam - li_exact(rho[:1], n) + li_exact(pair, n)
    anomaly = lam_cf - lam
    rate = np.log(np.abs(1 - 1 / pair[1]))  # theory: growth from the Re<1/2 member
    k = n > NMAX // 4
    # envelope growth: fit log|anomaly| local maxima
    mx = [i for i in range(1, NMAX - 1) if abs(anomaly[i]) >= abs(anomaly[i - 1]) and abs(anomaly[i]) >= abs(anomaly[i + 1]) and k[i]]
    slope = np.polyfit(n[mx], np.log(np.abs(anomaly[mx])), 1)[0]
    neg = n[lam_cf < 0]
    first_neg = int(neg[0]) if len(neg) else None
    print(f"D delta={delta}: anomaly growth rate {slope:.3e} (theory log|1-1/rho| = {rate:.3e}); "
          f"first negative lambda_n: {first_neg}")
    cf[delta] = {"rate_fit": slope, "rate_theory": rate, "first_negative_n": first_neg,
                 "lam_cf": lam_cf[::10]}

sr.save("e06", {"lam_first": lam[:10], "known": known, "min": lam.min(), "all_positive": bool((lam > 0).all()),
                "pysr": eqB, "fit": coef, "identified": ids, "c_theory": c_theory, "resid_pysr": eqC,
                "resid_mean": resid.mean(), "resid_std": resid.std(), "resid_env_exponent": env[0],
                "counterfactual": cf, "n_sub": n[::10], "lam_sub": lam[::10], "resid_sub": resid[::10]})

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(L, lam / n, "k-", lw=1, label="λ_n / n (2M zeros)")
ax[0].plot(L, 0.5 * L + c_theory, "r--", label="½ log n + (γ-1-log2π)/2")
ax[0].set_xlabel("log n"); ax[0].legend(); ax[0].set_title("A/B: Li coefficients")
ax[1].plot(n, resid, "k-", lw=.5); ax[1].set_xscale("log"); ax[1].set_title("C: residual after asymptotic law")
for d_, v in cf.items():
    ax[2].plot(n[::10], v["lam_cf"], lw=.8, label=f"off-line δ={d_}")
ax[2].plot(n[::10], lam[::10], "k-", lw=1, label="true zeros")
ax[2].set_yscale("symlog", linthresh=10); ax[2].axhline(0, c="gray")
ax[2].legend(fontsize=7); ax[2].set_title("D: counterfactual — one zero off the line")
plt.tight_layout(); plt.savefig(RESULTS / "e06.png", dpi=110)
