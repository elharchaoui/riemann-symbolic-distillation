"""L-angle 10 - exact off-line zeros of every DH-type function in the l08 window, and the exact slot count.

l08b placed each missing pair of the 11 other DH types at a slot boundary. Here the off-line zeros are found
(secant iteration seeded at minima of |F| on Re s = 0.55 ... 1.1, as for DH in l00), the count is checked
(line + 2 x off-line against theta/pi, and no drift of N_found - theta/pi across the window), and
    R_all = |sum_line e^{2i theta} + 2 sum_off e^{2i theta}| / (n_line + 2 n_off)
is compared with the parameter-free law R(P) = 1 - P(1 - e^{-1/P}) of l09.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, lfam, sr

L8 = json.loads((RESULTS / "l08.json").read_text())
T0, T1 = L8["window"]
model = lambda P: 1 - P * (1 - np.exp(-1 / P))
# a previous run is reused for functions whose count closed (|gap| <= 3); the others get a denser, wider scan
prev = {}
if (RESULTS / "l10.json").exists():
    prev = {tuple(x["spec"]): x for x in json.loads((RESULTS / "l10.json").read_text())["rows"]}
DENSE = (0.52, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.7, 1.9, 2.1)
rows = []
for r in L8["rows"]:
    if r["spec"][0] != "dh":
        continue
    spec = tuple(r["spec"])
    if spec in prev and abs(prev[spec]["count_gap"]) <= 3:
        rows.append(prev[spec])
        print(f"{spec}: reused (count gap {prev[spec]['count_gap']:+.1f}); R all slots {prev[spec]['R_all']:.3f}", flush=True)
        continue
    g = lfam.line_zeros(spec, T0, T1)
    expected = (lfam.theta(spec, T1) - lfam.theta(spec, T0)) / np.pi
    if spec in prev:  # failed before: dense scan, every minimum is a seed
        seeds = lfam.scan_seeds(spec, T0 - 5, T1 + 5, sigmas=DENSE, per_spacing=6, amax=np.inf)
    else:
        seeds = lfam.scan_seeds(spec, T0 - 5, T1 + 5, sigmas=(0.55, 0.65, 0.75, 0.85, 0.95, 1.1), amax=1.0)
    off = lfam.offline_zeros(spec, T0, T1, seeds)
    gap = len(g) + 2 * len(off) - expected
    tt = np.linspace(T0 + 1, T1 - 1, 3000)
    D = np.searchsorted(g, tt) + 2 * np.searchsorted(off[:, 1], tt) - (lfam.theta(spec, tt) - lfam.theta(spec, T0)) / np.pi
    drift = np.polyfit(tt, D, 1)[0] * (T1 - T0)  # zeros missing across the window (should be ~0)
    tot = np.exp(2j * lfam.theta(spec, g)).sum() + 2 * np.exp(2j * lfam.theta(spec, off[:, 1])).sum()
    R_all = abs(tot) / (len(g) + 2 * len(off))
    # boundary approximation of l08b, for comparison
    b = next(x for x in json.loads((RESULTS / "l08b.json").read_text())["dh"] if tuple(x["spec"]) == spec)
    r9 = next(x for x in json.loads((RESULTS / "l09.json").read_text())["BC"] if tuple(x["spec"]) == spec)
    rows.append({"spec": spec, "P": r["P"], "n_line": len(g), "n_off": len(off), "count_gap": gap, "drift": drift,
                 "R_line": r["R"], "R_all": R_all, "R_all_boundary": b["R_all"], "R_model": model(r["P"]), "R_t": r9["R_t"],
                 "sigma_max": float(off[:, 0].max()) if len(off) else None})
    print(f"{spec}: line {len(g)}, off-line {len(off)} (max Re {rows[-1]['sigma_max']:.3f}); count gap {gap:+.1f}, "
          f"drift {drift:+.1f} | R line {r['R']:.3f} -> all slots {R_all:.3f} (boundary approx. {b['R_all']:.3f});"
          f" law {model(r['P']):.3f}", flush=True)
for x in rows:  # rows reused from an older run may lack R_t
    x.setdefault("R_t", next(y for y in json.loads((RESULTS / "l09.json").read_text())["BC"]
                             if tuple(y["spec"]) == tuple(x["spec"]))["R_t"])
ok = [x for x in rows if abs(x["count_gap"]) <= 3]
dt = np.mean([abs(x["R_all"] - x["R_t"]) / x["R_t"] for x in ok])
print(f"functions whose count closes: {len(ok)}/{len(rows)}; R_all vs main-sum time average R_t: mean relative "
      f"difference {dt:.3f}", flush=True)
dl = np.mean([abs(x["R_line"] - x["R_model"]) / x["R_model"] for x in rows])
da = np.mean([abs(x["R_all"] - x["R_model"]) / x["R_model"] for x in rows])
ratio = np.array([x["R_all"] / x["R_model"] for x in rows])
print(f"distance from the law: line zeros {dl:.3f}; all slots (exact) {da:.3f}; R_all/law mean {ratio.mean():.3f}"
      f" +- {ratio.std():.3f}", flush=True)
sr.save("l10", {"rows": rows, "n_closed": len(ok), "dev_all_vs_Rt": dt, "dev_line": dl, "dev_all": da, "ratio_mean": ratio.mean(), "ratio_std": ratio.std()})

fig, ax = plt.subplots(figsize=(7, 4.4))
pp = np.exp(np.linspace(np.log(0.8), np.log(12), 200))
ax.plot(pp, model(pp), "k-", label="R = 1 − P(1 − e^(−1/P))")
ax.plot([x["P"] for x in rows], [x["R_line"] for x in rows], "s", c="C3", mfc="none", label="DH type: line zeros")
ax.plot([x["P"] for x in rows], [x["R_all"] for x in rows], "s", c="C3", label="DH type: all slots (exact off-line zeros)")
ax.plot([x["P"] for x in rows], [x["R_t"] for x in rows], "x", c="k", label="time average of the main-sum phase, R_t")
for x in rows:
    ax.plot([x["P"]] * 2, [x["R_line"], x["R_all"]], "-", c="C3", lw=.6)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("P"); ax.set_ylabel("R"); ax.legend(fontsize=7)
ax.set_title("DH types with every slot counted")
plt.tight_layout(); plt.savefig(RESULTS / "l10.png", dpi=110)
