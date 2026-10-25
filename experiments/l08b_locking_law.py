"""L-angle 8b - the locking law, fitted on Euler-product functions only, and the DH selection effect.

1. Fit R(P) on Dirichlet L-functions + zeta at all heights (no vacancies there): PySR + two candidate forms.
2. DH types keep only their well-placed zeros on the line, which inflates R. Count every slot instead:
   R_all = |sum_line e^{2i theta} + 2 sum_offline e^{2i theta}| / (n_line + 2 n_off).
   DH (q=5) uses its actual off-line zeros; the other DH types put each missing pair at a slot boundary
   (e^{2i theta} = -1 relative to the line phase), which l07 found for 72% of DH's pairs.
   If the vacancy story is right, R_all falls back onto the Euler-product curve.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from rh import RESULTS, lfam, lfun, sr

L8 = json.loads((RESULTS / "l08.json").read_text())
T0, T1 = L8["window"]
rows = L8["rows"]
ep = [(r["P"], r["R"]) for r in rows if r["spec"][0] == "dir"] + [(r["P"], r["R"]) for r in L8["zeta_height"]]
P, R = np.array(ep).T
_, eq = sr.fit(P[:, None], R, ["P"], "l08b_R_euler", unary=["exp", "log", "sqrt"], niterations=100, maxsize=10)
forms = {"1/(1+cP)": lambda p, c: 1 / (1 + c * p), "c/P": lambda p, c: c / p,
         "exp(-cP)": lambda p, c: np.exp(-c * p)}
fits = {}
for k, f in forms.items():
    (c,), _ = curve_fit(f, P, R, p0=[1.0])
    rel = np.sqrt(np.mean(((f(P, c) - R) / R) ** 2))
    fits[k] = {"c": c, "rms_rel": rel}
    print(f"Euler-product fit {k}: c = {c:.4f}, rms relative error {rel:.3f}", flush=True)
print(f"PySR (Euler-product functions only): {eq['elbow']}", flush=True)
best = min(fits, key=lambda k: fits[k]["rms_rel"])
fb = lambda p: forms[best](p, fits[best]["c"])

# ---- DH types: count every slot
dh = []
d5 = lfun.load("dh")
for r in rows:
    if r["spec"][0] != "dh":
        continue
    spec = tuple(r["spec"])
    g = lfam.line_zeros(spec, T0, T1)
    zl = np.exp(2j * lfam.theta(spec, g))
    line_phase = zl.sum() / abs(zl.sum())
    n_off = (lfam.theta(spec, T1) - lfam.theta(spec, T0)) / np.pi - len(g)  # zeros missing from the line
    if spec == ("dh", 5, 2):
        og = d5["off"][(d5["off"][:, 1] > T0) & (d5["off"][:, 1] < T1), 1]
        tot = zl.sum() + 2 * np.exp(2j * lfam.theta(spec, og)).sum()
        n_all = len(g) + 2 * len(og)
        how = f"actual off-line zeros ({len(og)} pairs; count gap {n_off/2:.1f})"
    else:
        tot = zl.sum() - n_off * line_phase
        n_all = len(g) + n_off
        how = "pairs at slot boundaries"
    R_all = abs(tot) / n_all
    dh.append({"spec": spec, "P": r["P"], "R_line": r["R"], "R_all": R_all, "R_curve": float(fb(r["P"])),
               "f_line": r["f_line"]})
    print(f"{spec}: P={r['P']:.2f} R_line={r['R']:.3f} -> R_all={R_all:.3f} ({how}); Euler-product curve "
          f"{fb(r['P']):.3f}", flush=True)
dev_line = np.mean([abs(d["R_line"] - d["R_curve"]) / d["R_curve"] for d in dh])
dev_all = np.mean([abs(d["R_all"] - d["R_curve"]) / d["R_curve"] for d in dh])
print(f"DH types: mean relative distance from the curve: line zeros only {dev_line:.3f}, all slots {dev_all:.3f}",
      flush=True)
sr.save("l08b", {"pysr": eq, "fits": fits, "best": best, "dh": dh, "dev_line": dev_line, "dev_all": dev_all})

fig, ax = plt.subplots(figsize=(7, 4.6))
pp = np.exp(np.linspace(np.log(0.8), np.log(25), 200))
ax.plot(pp, fb(pp), "k-", lw=1, label=f"fit on Euler products: {best}, c={fits[best]['c']:.3f}")
ax.plot([r["P"] for r in rows if r["spec"][0] == "dir"], [r["R"] for r in rows if r["spec"][0] == "dir"],
        "o", mfc="none", label="Dirichlet L, q ≤ 13")
ax.plot([r["P"] for r in L8["zeta_height"]], [r["R"] for r in L8["zeta_height"]], "k^", ms=5,
        label="ζ at T = 2·10³ … 10²⁰")
ax.plot([d["P"] for d in dh], [d["R_line"] for d in dh], "s", c="C3", mfc="none", label="DH type: line zeros")
ax.plot([d["P"] for d in dh], [d["R_all"] for d in dh], "s", c="C3", label="DH type: all slots")
for d in dh:
    ax.plot([d["P"]] * 2, [d["R_line"], d["R_all"]], "-", c="C3", lw=.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("P = Σ_{2≤n≤√(qT/2π)} |a_n|²/n"); ax.set_ylabel("lattice order R")
ax.legend(fontsize=7); ax.set_title("one locking law for Euler products; DH's excess is mostly selection")
plt.tight_layout(); plt.savefig(RESULTS / "l08b.png", dpi=110)
