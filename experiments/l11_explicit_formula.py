"""L-angle 11 - literature check of the locking law: the explicit formula already gives it for Euler products.

Arias de Reyna (arXiv:1403.7126, J. Number Theory 153 (2015)), after Schoissengeier:
    sum_{0 < gamma <= T} exp(2i theta(gamma)) = -psi(T/2pi) + O(T^{1/2} log^2 T),
so for zeta the lattice order in a window at height T is R = 1/log(T/2pi) to leading order.
The same contour argument for a primitive Dirichlet L-function, with the prime number theorem in progressions,
gives R ~ (q/phi(q)) / log(qT/2pi). More generally R ~ sqrt(q) |< c_n e(-n/q) >_n| / log(qT/2pi), n ~ qT/2pi,
c_n = coefficients of -F'/F. This script compares these with the measured R (l08, l10) and with the heuristic
random-phase law R(P) = 1 - P(1 - e^{-1/P}) (l09). For DH types c_n is driven by the off-line zeros themselves.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, data, lfam, lfun, sr

L8 = json.loads((RESULTS / "l08.json").read_text())
L10 = {tuple(x["spec"]): x for x in json.loads((RESULTS / "l10.json").read_text())["rows"]}
T0, T1 = L8["window"]
TM = (T0 + T1) / 2
model = lambda P: 1 - P * (1 - math.exp(-1 / P))
phi = lambda q: sum(1 for a in range(1, q + 1) if math.gcd(a, q) == 1)
rows = []
for r in L8["rows"]:
    spec = tuple(r["spec"])
    q = spec[1]
    n0, n1 = int(q * T0 / (2 * np.pi)), int(q * T1 / (2 * np.pi))
    a = lfam.coefficients(spec, n1 + 1)
    c = (a * data.von_mangoldt(n1 + 1)) if spec[0] == "dir" else lfun.dirichlet_log_coeffs(a)
    nn = np.arange(n0, n1 + 1)
    L = math.log(q * TM / (2 * np.pi))
    ef = math.sqrt(q) * abs(np.mean(c[n0:n1 + 1] * np.exp(-2j * np.pi * nn / q))) / L
    row = {"spec": spec, "P": r["P"], "R": r["R"] if spec[0] == "dir" else L10[spec]["R_all"],
           "R_line": r["R"], "pred_ef": ef, "pred_heur": model(r["P"])}
    if spec[0] == "dir":
        row["pred_q"] = q / phi(q) / L
    rows.append(row)
zrows = []
for z in L8["zeta_height"]:
    zrows.append({"T": z["T"], "R": z["R"], "n": z["n"], "pred_q": 1 / math.log(z["T"] / (2 * np.pi)),
                  "pred_heur": model(z["P"])})
eul = [x for x in rows if x["spec"][0] == "dir"] + zrows
rel = lambda k: float(np.sqrt(np.mean([((x[k] - x["R"]) / x["R"]) ** 2 for x in eul])))
print(f"Euler products ({len(eul)} points): rms relative error  (q/phi(q))/log(qT/2pi): {rel('pred_q'):.3f};"
      f"  heuristic R(P): {rel('pred_heur'):.3f}", flush=True)
for z in zrows:
    print(f"zeta T~{z['T']:.1e}: R={z['R']:.4f}  1/log(T/2pi)={z['pred_q']:.4f}  R(P)={z['pred_heur']:.4f}", flush=True)
dh = [x for x in rows if x["spec"][0] == "dh"]
for x in dh:
    print(f"{x['spec']}: R (all slots) {x['R']:.3f}; explicit-formula average {x['pred_ef']:.3f}; R(P) {x['pred_heur']:.3f}")
sr.save("l11", {"rows": rows, "zeta": zrows, "rms_q": rel("pred_q"), "rms_heur": rel("pred_heur")})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
dr = [x for x in rows if x["spec"][0] == "dir"]
ax[0].plot([x["pred_q"] for x in dr], [x["R"] for x in dr], "o", mfc="none", label="Dirichlet L, q ≤ 13")
ax[0].plot([z["pred_q"] for z in zrows], [z["R"] for z in zrows], "k^", ms=5, label="ζ, T = 2·10³ … 10²⁰")
ax[0].plot([0.01, 0.4], [0.01, 0.4], "k-", lw=.7)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("(q/φ(q)) / log(qT/2π)   [explicit formula, Arias de Reyna for q = 1]"); ax[0].set_ylabel("measured R")
ax[0].legend(fontsize=7); ax[0].set_title(f"Euler products: rms error {rel('pred_q'):.1%}, no parameter")
ax[1].plot([x["pred_heur"] for x in dh], [x["R"] for x in dh], "s", c="C3", label="heuristic R(P)")
ef_ok = [x for x in dh if x["pred_ef"] < 1]
ax[1].plot([x["pred_ef"] for x in ef_ok], [x["R"] for x in ef_ok], "D", c="C2", mfc="none",
           label="explicit-formula average of c_n")
ax[1].plot([0, .4], [0, .4], "k-", lw=.7)
ax[1].set_xlabel("prediction"); ax[1].set_ylabel("measured R (all slots)"); ax[1].legend(fontsize=7)
ax[1].set_title("DH types: neither prediction is exact")
plt.tight_layout(); plt.savefig(RESULTS / "l11.png", dpi=110)
