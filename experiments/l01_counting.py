"""L-angle 1 - counting zeros of zeta, L(s, chi_-4) and Davenport-Heilbronn (DH).

A. Same pipeline as e01: network learns N(T)/T vs L = log T, PySR distils, a*L + b refitted on raw counts.
   Theory (functional equation only): a = 1/2pi, b = (log(q/2pi) - 1)/2pi  ->  the data give the conductor
   q = 2pi exp(2pi b + 1). Constant term: 7/8 for zeta (pole at s=1), 1/8 for odd L-functions.
   For DH all zeros are counted (an off-line pair rho, 1 - conj(rho) counts 2).
B. DH only: what fraction of the zeros lies on the line, and how does the off-line count grow? (PySR)
C. Size of S(t) at matched heights (Selberg: Var S ~ log log T / 2pi^2 for Euler products).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, lfun, nets, sr

rng = np.random.default_rng(10)
fam = lfun.family()
out = {}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

for name, F in fam.items():
    z = F["all"]
    T = np.exp(rng.uniform(np.log(50), np.log(F["T"]), 40000))
    N = np.searchsorted(z, T).astype(float)
    L = np.log(T)
    zs = lambda l: ((l - 7) / 3)[:, None]
    net, _ = nets.fit_mlp(zs(L), N / T * 10, epochs=4000, width=64, depth=3)
    Lq = np.linspace(np.log(50), np.log(F["T"]), 400)
    Ynet = nets.predict(net, zs(Lq))[:, 0] / 10
    _, eq = sr.fit(Lq[:, None], Ynet, ["L"], f"l01_NT_{name}", unary="log", niterations=60, maxsize=12)
    m = T > 500
    coef = np.linalg.lstsq(np.c_[L[m], np.ones(m.sum())] * T[m, None], N[m], rcond=None)[0]
    q_est = 2 * np.pi * np.exp(2 * np.pi * coef[1] + 1)
    q_int = int(round(q_est))
    # constant term with the conductor the data chose
    main = lambda t: t / (2 * np.pi) * np.log(q_int * t / (2 * np.pi)) - t / (2 * np.pi)
    edges = np.exp(np.linspace(np.log(200), np.log(F["T"]), 11))
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        tr = rng.uniform(a, b, 40000)
        R = np.searchsorted(z, tr) - main(tr)
        bins.append((np.sqrt(a * b), R.mean(), R.std() / np.sqrt(len(R))))
    bins = np.array(bins)
    w = 1 / bins[:, 2] ** 2
    cmean, cerr = np.sum(bins[:, 1] * w) / w.sum(), 1 / np.sqrt(w.sum())
    chi2 = np.sum(((bins[:, 1] - cmean) / bins[:, 2]) ** 2)
    cid = sr.recognize(cmean, tol=3 * cerr, maxden=8)
    print(f"A {name}: PySR elbow {eq['elbow']}; raw a={coef[0]:.6f} (1/2pi={1/(2*np.pi):.6f}) b={coef[1]:.6f}"
          f" -> q = {q_est:.3f}; constant {cmean:.4f} +- {cerr:.4f} (chi2 {chi2:.1f}/{len(bins)-1}) -> {cid}", flush=True)
    out[name] = {"pysr": eq, "a": coef[0], "b": coef[1], "a_id": sr.recognize(coef[0], tol=1e-4 * coef[0]),
                 "q_est": q_est, "q_true": F["q"], "const": cmean, "const_err": cerr, "const_chi2": chi2,
                 "const_id": cid, "const_bins": bins}
    ax[0].errorbar(bins[:, 0], bins[:, 1], bins[:, 2], fmt="o-", ms=3, label=f"{F['label']}: q≈{q_est:.2f}")
for c in (7 / 8, 1 / 8):
    ax[0].axhline(c, c="k", ls="--", lw=.7)
ax[0].set_xscale("log"); ax[0].legend(fontsize=7); ax[0].set_title("N(T) − T/2π·log(qT/2πe): constant term")

# ---------------- B: DH on-line fraction ------------------------------------------------
F = fam["dh"]
Tg = np.linspace(200, F["T"], 200)
n_line = np.searchsorted(F["g"], Tg)
n_off = 2 * np.searchsorted(np.sort(F["off"][:, 1]), Tg)
frac = n_line / (n_line + n_off)
_, eqO = sr.fit(Tg[:, None], n_off / 2.0, ["T"], "l01_dh_offcount", unary="log", niterations=60, maxsize=10)
lin = np.polyfit(Tg, n_off / 2.0, 1)
sig = F["off"][:, 0]
print(f"B DH: {len(F['off'])} zeros with Re>1/2 below {F['T']:.0f}; fraction on line {frac[-1]:.4f};"
      f" off-line count PySR: {eqO['elbow']}; linear slope {lin[0]:.5f}/unit height", flush=True)
print(f"   Re rho of off-line zeros: max {sig.max():.4f}, median {np.median(sig):.4f}; any with Re>1: {(sig > 1).any()}")
out["dh_offline"] = {"n_off": len(F["off"]), "frac_line_final": frac[-1], "pysr": eqO, "slope": lin[0],
                     "intercept": lin[1], "sigma_max": sig.max(), "sigma_median": np.median(sig),
                     "T": Tg[::5], "frac": frac[::5], "sigma": sig, "gamma": F["off"][:, 1]}
ax[1].plot(Tg, frac, "k-")
ax[1].set_xlabel("T"); ax[1].set_ylabel("fraction of zeros on Re s = ½"); ax[1].set_title("DH: zeros on the line")
ins = ax[1].inset_axes([0.45, 0.12, 0.5, 0.45])
ins.hist(sig, 40, color="C3"); ins.set_xlabel("Re ρ (off-line)", fontsize=7); ins.tick_params(labelsize=6)

# ---------------- C: Var S at matched heights -------------------------------------------
windows = [(200, 1000), (1000, 4000), (4000, 12000), (12000, 25000)]
out["varS"] = {}
for name, F in fam.items():
    rows = []
    for a, b in windows:
        ts = rng.uniform(a, b, 30000)
        S = np.searchsorted(F["all"], ts) - F["ns"](ts)
        rows.append((np.sqrt(a * b), S.var(), ((S - S.mean()) ** 4).mean() / S.var() ** 2, S.mean()))
    rows = np.array(rows)
    out["varS"][name] = rows
    print(f"C {name}: Var S per window", np.round(rows[:, 1], 4), " kurtosis", np.round(rows[:, 2], 2), flush=True)
    ax[2].plot(np.log(np.log(rows[:, 0])), rows[:, 1], "o-", label=fam[name]["label"])
xx = np.log(np.log(np.array([np.sqrt(a * b) for a, b in windows])))
ax[2].plot(xx, xx / (2 * np.pi**2) + out["varS"]["chi4"][0, 1] - xx[0] / (2 * np.pi**2), "k--", label="slope 1/2π²")
ax[2].set_xlabel("log log T"); ax[2].set_ylabel("Var S(t)"); ax[2].legend(fontsize=7); ax[2].set_title("size of S(t)")
plt.tight_layout(); plt.savefig(RESULTS / "l01.png", dpi=110)
sr.save("l01", out)
