"""L-angle 8 - Gram locking across a family of L-functions (follow-up to the DH lattice found in l02).

Heuristic: Z(t) = 2 cos(theta) + 2 sum_{2<=n<=N} a_n n^{-1/2} cos(theta - t log n) + ...   (N = sqrt(qt/2pi))
The n = 1 term alone has zeros exactly at the slots theta = pi/2 mod pi. The other terms push zeros off their
slots, with mean-square size  P = sum_{2<=n<=N} |a_n|^2 / n.   Prediction: small P -> strong locking.
Jerby (arXiv:2310.14415) deforms this "core" into Z and shows DH's first off-line pair is a collision of two
core zeros; here the statistics are measured across many functions.

Measured per function, line zeros in t in [T0, T1]:
  R      = |< exp(2i theta(gamma)) >|                (lattice order; = |<exp(2 pi i S(gamma))>| exactly)
  f_line = (# line zeros) / ((theta(T1) - theta(T0))/pi)   (1 - vacancy rate)
Family: Dirichlet L-functions (Euler product) for every primitive character (one per conjugate pair) with
q <= 13, and the real functional-equation combination of each complex pair (DH type, no Euler product).
Plus zeta at heights 1e3 ... 1e21 from Odlyzko's tables (R from S at the zeros).
PySR then looks for R(P) and f_line(P).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, data, lfam, sr

T0, T1 = 2000.0, 3500.0
TM = (T0 + T1) / 2
QS = [1, 3, 4, 5, 7, 8, 11, 12, 13]


def specs():
    from flint import dirichlet_char
    out = []
    for q in QS:
        prim = [1] if q == 1 else [i for i in range(1, q + 1) if math.gcd(i, q) == 1 and dirichlet_char(q, i).is_primitive()]
        seen = set()
        for i in prim:
            j = lfam.conj_index(q, i) if q > 1 else 1
            if j in seen:
                continue
            seen.add(i)
            out.append(("dir", q, i))
            if j != i:
                out.append(("dh", q, i))
    return out


def P_of(a, q, T):
    N = int(np.sqrt(q * T / (2 * np.pi)))
    n = np.arange(2, N + 1)
    return float(np.sum(np.abs(a[2:N + 1]) ** 2 / n))


rows = []
for spec in specs():
    g = lfam.line_zeros(spec, T0, T1)
    th = lfam.theta(spec, g)
    R = abs(np.mean(np.exp(2j * th)))
    expected = (lfam.theta(spec, T1) - lfam.theta(spec, T0)) / np.pi
    a = lfam.coefficients(spec, 200)
    P = P_of(a, spec[1], TM)
    rows.append({"spec": spec, "R": R, "f_line": len(g) / expected, "P": P, "n": len(g),
                 "a2": abs(a[2]), "a3": abs(a[3]), "amax": float(np.max(np.abs(a[2:30])))})
    print(f"{spec}: n={len(g)} R={R:.3f} f_line={len(g)/expected:.4f} P={P:.3f}  |a_n| n<=6: "
          f"{np.round(np.abs(a[1:7]), 3).tolist()}", flush=True)

# ---- zeta along height (Odlyzko): R = |<exp(2 pi i S(gamma_n))>|
gz = data.zeros()
zrows = []
for lo, hi in [(1e3, 3e3), (1e4, 2e4), (1e5, 1.2e5), (6e5, 6.5e5), (1e6, 1.1e6)]:
    idx = np.nonzero((gz > lo) & (gz < hi))[0]
    S = data.S_at_zeros(gz[idx], idx[0] + 1)
    Tm = np.sqrt(lo * hi)
    zrows.append({"T": Tm, "R": abs(np.mean(np.exp(2j * np.pi * S))), "P": P_of(np.r_[0, np.ones(10**6)], 1, Tm),
                  "n": len(idx)})
for w in ["1e12", "1e21"]:
    off, loc, _ = data.high_zeros(w)
    S = data.S_at_high_zeros(w)
    Tm = float(off) + loc.mean()
    N = int(np.sqrt(Tm / (2 * np.pi)))
    P = float(np.log(N) + np.euler_gamma - 1 + 1 / (2 * N))  # sum_{n=2}^N 1/n
    zrows.append({"T": Tm, "R": abs(np.mean(np.exp(2j * np.pi * S))), "P": P, "n": len(S)})
for r in zrows:
    print(f"zeta at T~{r['T']:.2e}: R={r['R']:.3f} P={r['P']:.2f} (n={r['n']})", flush=True)

# ---- distil
dirs = [r for r in rows if r["spec"][0] == "dir"]
dhs = [r for r in rows if r["spec"][0] == "dh"]
allR = rows + [{"R": r["R"], "P": r["P"], "spec": ("zeta-high",)} for r in zrows]
X = np.array([[r["P"]] for r in allR])
y = np.array([r["R"] for r in allR])
_, eqR = sr.fit(X, y, ["P"], "l08_R_of_P", unary=["exp", "log", "sqrt"], niterations=80, maxsize=10)
Xd = np.array([[r["P"]] for r in dhs])
yd = np.array([r["f_line"] for r in dhs])
_, eqF = sr.fit(Xd, yd, ["P"], "l08_fline_of_P", unary=["exp", "log", "sqrt"], niterations=80, maxsize=10)
cR = np.corrcoef(np.log(X[:, 0]), np.log(y))[0, 1]
print(f"R(P) PySR elbow: {eqR['elbow']}  (corr log R vs log P {cR:.3f})", flush=True)
print(f"DH-type f_line(P) PySR elbow: {eqF['elbow']}", flush=True)
print(f"Dirichlet L: f_line min {min(r['f_line'] for r in dirs):.4f}, max {max(r['f_line'] for r in dirs):.4f}"
      f"; DH-type: f_line range {min(yd):.3f} .. {max(yd):.3f}", flush=True)
sr.save("l08", {"rows": rows, "zeta_height": zrows, "pysr_R": eqR, "pysr_fline": eqF, "corr_logR_logP": cR,
                "window": [T0, T1]})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for grp, mk, lab in [(dirs, "o", "Dirichlet L (Euler product)"), (dhs, "s", "DH type (no Euler product)")]:
    ax[0].plot([r["P"] for r in grp], [r["R"] for r in grp], mk, mfc="none", label=lab)
ax[0].plot([r["P"] for r in zrows], [r["R"] for r in zrows], "k^-", ms=5, label="ζ at T = 2·10³ … 10²¹")
for r in rows:
    if r["spec"] in [("dir", 1, 1), ("dh", 5, 2), ("dir", 4, 3)]:
        ax[0].annotate({("dir", 1, 1): "ζ", ("dh", 5, 2): "DH", ("dir", 4, 3): "χ₋₄"}[r["spec"]], (r["P"], r["R"]),
                       textcoords="offset points", xytext=(5, 4), fontsize=9)
ax[0].set_xscale("log"); ax[0].set_xlabel("P = Σ_{2≤n≤N} |a_n|²/n"); ax[0].set_ylabel("lattice order R")
ax[0].legend(fontsize=7); ax[0].set_title("Gram locking against the size of the non-leading terms")
ax[1].plot([r["P"] for r in dirs], [r["f_line"] for r in dirs], "o", mfc="none", label="Dirichlet L")
ax[1].plot([r["P"] for r in dhs], [r["f_line"] for r in dhs], "s", mfc="none", label="DH type")
ax[1].set_xscale("log"); ax[1].set_xlabel("P"); ax[1].set_ylabel("share of zeros on the line")
ax[1].legend(fontsize=7); ax[1].set_title("vacancies need the missing Euler product")
plt.tight_layout(); plt.savefig(RESULTS / "l08.png", dpi=110)
