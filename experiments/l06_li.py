"""L-angle 6 - Li coefficients lambda_n = sum_rho [1 - (1 - 1/rho)^n] for zeta, L(s, chi_-4) and DH.

Generalised Li criterion: all zeros of F on the line  <=>  lambda_n(F) >= 0 for every n.
A. lambda_n for n <= NLI from the zero lists (DH: line zeros, off-line rho and partner 1 - conj(rho)),
   tail beyond the list ~ n^2 (log(qT/2pi) + 1)/(2 pi T). Check lambda_1 = Lambda'/Lambda(1) from Arb.
B. Network + PySR on lambda_n / n; fit a n log n + b n + c. Expected b = (euler - 1 - log 2pi + log q)/2.
C. DH: the exact contribution of its off-line zeros, D_n = lambda_n - lambda_n(off-line zeros moved onto the
   line). Growth rate log|1 - 1/(1 - conj rho)| > 0. When does lambda_n(DH) first go negative?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

from rh import RESULTS, lfun, nets, sr

NLI = 2000
n = np.arange(1, NLI + 1)
fam = lfun.family()
out = {}


def li_sum(rhos, nn):
    """sum over rho in rhos (Im > 0) and conjugates of 1 - (1 - 1/rho)^n."""
    w = np.log1p(-1 / rhos)
    res = np.zeros(len(nn))
    for i0 in range(0, len(rhos), 2000):
        wc = w[i0:i0 + 2000]
        for j0 in range(0, len(nn), 1000):
            res[j0:j0 + 1000] += 2 * np.sum(1 - np.exp(nn[j0:j0 + 1000, None] * wc[None, :]).real, 1)
    return res


def lambda1_arb(name):
    """Lambda'/Lambda(1) = 1/2 log(q/pi) + 1/2 psi(1) + F'(1)/F(1) (odd type, a = 1)."""
    from flint import acb
    h = 1e-6
    v0, vp, vm = (lfun.value(name, complex(1 + d, 0)) for d in (0, h, -h))
    return 0.5 * np.log(lfun.SPEC[name]["q"] / np.pi) + 0.5 * digamma(1.0) + ((vp - vm) / (2 * h) / v0).real


fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for name, F in fam.items():
    rho = np.r_[0.5 + 1j * F["g"], F["off"][:, 0] + 1j * F["off"][:, 1], (1 - F["off"][:, 0]) + 1j * F["off"][:, 1]]
    lam = li_sum(rho, n) + n.astype(float) ** 2 * (np.log(F["q"] * F["T"] / (2 * np.pi)) + 1) / (2 * np.pi * F["T"])
    l1 = 0.0230957089661210338 if name == "zeta" else lambda1_arb(name)
    L = np.log(n)
    zs = lambda l: ((l - 4) / 2)[:, None]
    net, _ = nets.fit_mlp(zs(L), lam / n, epochs=3000, width=48, depth=3)
    Lq = np.linspace(np.log(20), np.log(NLI), 300)
    _, eq = sr.fit(Lq[:, None], nets.predict(net, zs(Lq))[:, 0], ["L"], f"l06_li_{name}", unary="log",
                   niterations=60, maxsize=12)
    m = n >= 50
    coef = np.linalg.lstsq(np.c_[n[m] * L[m], n[m], np.ones(m.sum())] / n[m, None], lam[m] / n[m], rcond=None)[0]
    b_th = (np.euler_gamma - 1 - np.log(2 * np.pi) + np.log(F["q"])) / 2
    q_from_b = np.exp(2 * coef[1] - (np.euler_gamma - 1 - np.log(2 * np.pi)))
    print(f"A {name}: lambda_1 zeros {lam[0]:.8f} vs exact {l1:.8f}; min lambda_n {lam.min():.4f} (n={n[lam.argmin()]});"
          f" all > 0: {(lam > 0).all()}", flush=True)
    print(f"B {name}: PySR {eq['elbow']}; fit {coef[0]:.5f} n log n + {coef[1]:.5f} n + {coef[2]:.2f};"
          f" theory b = {b_th:.5f}; q from b = {q_from_b:.3f}", flush=True)
    out[name] = {"lam1": lam[0], "lam1_exact": l1, "min": lam.min(), "all_positive": bool((lam > 0).all()),
                 "pysr": eq, "fit": coef, "b_theory": b_th, "q_from_b": q_from_b, "lam_sub": lam[::5]}
    ax[0].plot(L, lam / n - 0.5 * L, label=f"{F['label']}  (b={coef[1]:.3f}, theory {b_th:.3f})")
    if name == "dh":
        lam_dh = lam
ax[0].set_xlabel("log n"); ax[0].set_ylabel("λ_n/n − ½ log n"); ax[0].legend(fontsize=7)
ax[0].set_title("Li coefficients: same law, conductor shifts it")

# ---------------- C: DH off-line anomaly -----------------------------------------------------
off = fam["dh"]["off"]
rp = (1 - off[:, 0]) + 1j * off[:, 1]  # the member with Re < 1/2
rr = off[:, 0] + 1j * off[:, 1]
rate = np.log(np.abs(1 - 1 / rp))
top = np.argsort(rate)[::-1][:60]
print("C DH: largest growth rates log|1-1/rho| (Re rho < 1/2 member):",
      [(round(off[i, 0], 4), round(off[i, 1], 2), f"{rate[i]:.3e}") for i in top[:5]], flush=True)


def anomaly(nn, idx):
    on_line = 0.5 + 1j * off[idx, 1]
    return li_sum(rr[idx], nn) + li_sum(rp[idx], nn) - 2 * li_sum(on_line, nn)


D_small = anomaly(n, np.arange(len(off)))
print(f"C DH: off-line contribution for n <= {NLI}: max |D_n| = {np.abs(D_small).max():.4f}"
      f" (lambda_n there ~ {lam_dh[np.abs(D_small).argmax()]:.1f})", flush=True)
# extrapolate: lambda_n ~ fitted smooth law (line zeros behave like the law) + exact D_n of the top zeros
nb = np.unique(np.round(np.exp(np.linspace(np.log(NLI), np.log(3e6), 4000))).astype(np.int64)).astype(float)
Db = anomaly(nb, top)
a_, b_, c_ = out["dh"]["fit"]
smooth = a_ * nb * np.log(nb) + b_ * nb + c_
neg = nb[smooth + Db < 0]
first_neg = int(neg[0]) if len(neg) else None
n_dom = (1 / rate[top[0]]) * np.log(nb)  # scale where e^{rate n} ~ n
print(f"C DH: predicted first negative lambda_n (smooth law + exact off-line terms): n ≈ {first_neg}", flush=True)
out["dh_anomaly"] = {"max_abs_D_small": np.abs(D_small).max(), "rates_top": rate[top[:10]],
                     "zeros_top": off[top[:10]], "first_negative_pred": first_neg, "nb": nb[::20],
                     "Db": Db[::20], "smooth": smooth[::20]}
ax[1].plot(n, D_small, "k-", lw=.8)
ax[1].set_xlabel("n"); ax[1].set_title(f"DH: off-line part of λ_n, n ≤ {NLI}")
ax[2].plot(nb, smooth, "k--", label="smooth law (line zeros)")
ax[2].plot(nb, -Db, "C3-", lw=.8, label="−(off-line part), exact")
if first_neg:
    ax[2].axvline(first_neg, c="C3", ls=":", label=f"first λ_n < 0 ≈ {first_neg:.2e}")
ax[2].set_xscale("log"); ax[2].set_yscale("symlog", linthresh=10); ax[2].legend(fontsize=7)
ax[2].set_title("DH: when Li's criterion finally fails")
plt.tight_layout(); plt.savefig(RESULTS / "l06.png", dpi=110)
sr.save("l06", out)
