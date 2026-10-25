"""L-angle 9 - a parameter-free locking law, and what the Dirichlet polynomial does and does not know.

Model. Z(t) ~ 2 Re(e^{i theta} F(t)) with F = sum_{n<=N} a_n n^{-1/2-it} = 1 + W, N = sqrt(qt/2pi) (the main sum of the
approximate functional equation; complex chi carry a constant phase eps^{-1/2}).
So Z = 2|F| cos(theta + arg F): zeros sit at theta = pi/2 - arg F (mod pi) and R = |< exp(-2i arg F) >|.
If W is a circular complex Gaussian with E|W|^2 = P (random phases n^{-it}), the Rician integral gives
        R(P) = 1 - P (1 - e^{-1/P})          (no fitted constant; R ~ 1 - P for small P, ~ 1/(2P) for large P).
Checks:
 A. formula vs the l08 data (Euler products: Dirichlet L, zeta at heights to 1e20), with sampling errors;
 B. ingredients, from the actual Dirichlet polynomial F on a fine grid: E|W|^2 vs P, |<exp(-2i arg F)>|_t,
    and R at the zeros of the core Z_core = 2 Re(e^{i theta} F);
 C. vacancies: does the core lose zeros (theta/pi count minus core zeros) where the true Z does / does not?
 D. which zeros does the core lose? zeta (Odlyzko) and DH (computed line + off-line zeros), grid 1/40 spacing:
    nearest-neighbour gap of the true zeros the core misses; share of DH off-line zeros inside a large core gap.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, data, lfam, lfun, sr

L8 = json.loads((RESULTS / "l08.json").read_text())
T0, T1 = L8["window"]
model = lambda P: 1 - P * (1 - np.exp(-1 / np.asarray(P, float)))
out = {}

# ---------------- A: formula vs data ----------------------------------------------------------
pts = [(r["P"], r["R"], r["n"], str(tuple(r["spec"]))) for r in L8["rows"] if r["spec"][0] == "dir"]
pts += [(r["P"], r["R"], r["n"], f"zeta T~{r['T']:.0e}") for r in L8["zeta_height"]]
P, R, n = (np.array([p[i] for p in pts], float) for i in range(3))
se = np.sqrt((1 + R**2) / (2 * n))  # rough standard error of |mean of unit phasors| near R
z = (R - model(P)) / se
rms = np.sqrt(np.mean(((R - model(P)) / R) ** 2))
fit = json.loads((RESULTS / "l08b.json").read_text())["fits"]["1/(1+cP)"]
print(f"A: R = 1 - P(1 - e^(-1/P)) on {len(P)} Euler-product points: rms relative error {rms:.3f}; "
      f"chi^2/dof = {np.mean(z**2):.2f}  (fitted 1/(1+cP), c={fit['c']:.3f}: {fit['rms_rel']:.3f})", flush=True)
out["A"] = {"rms_rel": rms, "chi2_dof": float(np.mean(z**2)), "fit_rms_rel": fit["rms_rel"], "points": pts,
            "z": z}

# ---------------- B, C: the actual Dirichlet polynomial -----------------------------------------
specs = [tuple(r["spec"]) for r in L8["rows"]]
rowsBC = []
for spec in specs:
    q = spec[1]
    a = lfam.coefficients(spec, 400).astype(complex)
    ts = [T0]
    while ts[-1] < T1:
        ts.append(ts[-1] + lfam.mean_spacing(spec, ts[-1]) / 12)
    t = np.array(ts)
    th = lfam.theta(spec, t)
    Nt = np.floor(np.sqrt(q * t / (2 * np.pi))).astype(int)
    nmax = Nt.max()
    nn = np.arange(1, nmax + 1)
    terms = a[1:nmax + 1][None, :] * nn ** -0.5 * np.exp(-1j * np.outer(t, np.log(nn)))
    terms[nn[None, :] > Nt[:, None]] = 0
    F = terms.sum(1)
    W = F - 1
    # complex chi: Z = 2 Re(eps^{-1/2} e^{i theta} F); the constant phase matters for counting when theta + arg F
    # is not monotone (real chi and DH types have eps = 1)
    ph = lfam.root_number(q, spec[2]) ** -0.5 if (spec[0] == "dir" and q > 1) else 1.0
    Pt = float(np.mean([np.sum(np.abs(a[2:k + 1]) ** 2 / np.arange(2, k + 1)) for k in Nt]))
    zc = np.real(ph * np.exp(1j * th) * F)
    sc = np.nonzero(np.sign(zc[:-1]) != np.sign(zc[1:]))[0]
    # linear interpolation of the core zeros, enough for theta at the zero
    tz = t[sc] - zc[sc] * (t[sc + 1] - t[sc]) / (zc[sc + 1] - zc[sc])
    thz = lfam.theta(spec, tz)
    R_core = abs(np.mean(np.exp(2j * thz)))
    R_t = abs(np.mean(np.exp(-2j * np.angle(F))))  # phase-independent
    expected = (lfam.theta(spec, T1) - lfam.theta(spec, T0)) / np.pi
    f_core = len(tz) / expected
    w = W / np.sqrt(np.mean(np.abs(W) ** 2))
    kurt = np.mean(np.abs(w) ** 4) / 2  # = 1 for a circular complex Gaussian
    r8 = next(r for r in L8["rows"] if tuple(r["spec"]) == spec)
    rowsBC.append({"spec": spec, "P": Pt, "E|W|^2": float(np.mean(np.abs(W) ** 2)), "kurt": kurt,
                   "R_true": r8["R"], "R_core": R_core, "R_t": R_t, "R_model": float(model(Pt)),
                   "f_true": r8["f_line"], "f_core": f_core})
    print(f"{spec}: P={Pt:.2f} E|W|^2={np.mean(np.abs(W)**2):.2f} kurt={kurt:.2f} | R true {r8['R']:.3f}, core "
          f"{R_core:.3f}, <e^-2i argF>_t {R_t:.3f}, model {model(Pt):.3f} | on-line share: true {r8['f_line']:.3f},"
          f" core {f_core:.3f}", flush=True)
out["BC"] = rowsBC
dirs = [r for r in rowsBC if r["spec"][0] == "dir"]
dhs = [r for r in rowsBC if r["spec"][0] == "dh"]
cc = np.corrcoef([r["f_core"] for r in dhs], [r["f_true"] for r in dhs])[0, 1]
print(f"C: core on-line share, Dirichlet L: {min(r['f_core'] for r in dirs):.3f}..{max(r['f_core'] for r in dirs):.3f}"
      f" (true {min(r['f_true'] for r in dirs):.3f}..{max(r['f_true'] for r in dirs):.3f}); DH types: core "
      f"{min(r['f_core'] for r in dhs):.3f}..{max(r['f_core'] for r in dhs):.3f}, true "
      f"{min(r['f_true'] for r in dhs):.3f}..{max(r['f_true'] for r in dhs):.3f}, corr {cc:.3f}", flush=True)
out["C"] = {"corr_dh_fcore_ftrue": cc}
# ---------------- D: which zeros does the core lose? ---------------------------------------------
def core_zeros(spec, step=40):
    q = spec[1]
    a = lfam.coefficients(spec, 400).astype(complex)
    ts = [T0]
    while ts[-1] < T1:
        ts.append(ts[-1] + lfam.mean_spacing(spec, ts[-1]) / step)
    t = np.array(ts)
    Nt = np.floor(np.sqrt(q * t / (2 * np.pi))).astype(int)
    nn = np.arange(1, Nt.max() + 1)
    F = np.zeros(len(t), complex)
    for i0 in range(0, len(t), 5000):
        blk = a[1:len(nn) + 1][None, :] * nn ** -0.5 * np.exp(-1j * np.outer(t[i0:i0 + 5000], np.log(nn)))
        blk[nn[None, :] > Nt[i0:i0 + 5000, None]] = 0
        F[i0:i0 + 5000] = blk.sum(1)
    zc = np.real(np.exp(1j * lfam.theta(spec, t)) * F)
    k = np.nonzero(np.sign(zc[:-1]) != np.sign(zc[1:]))[0]
    return t[k] - zc[k] * (t[k + 1] - t[k]) / (zc[k + 1] - zc[k])


out["D"] = {}
for spec in [("dir", 1, 1), ("dh", 5, 2)]:
    cz = core_zeros(spec)
    g = data.zeros() if spec[0] == "dir" else lfun.load("dh")["g"]
    g = g[(g > T0) & (g < T1)]
    sp = lfam.mean_spacing(spec, g)
    d = np.abs(cz[None, :] - g[:, None]).min(1)
    miss = d > 0.3 * sp
    nn_gap = np.minimum(np.r_[np.inf, np.diff(g)], np.r_[np.diff(g), np.inf]) / sp
    res = {"core_zeros": len(cz), "missed_line_zeros": int(miss.sum()), "missed_gap_median": float(np.median(nn_gap[miss])),
           "all_gap_median": float(np.median(nn_gap))}
    if spec[0] == "dh":
        off = lfun.load("dh")["off"]
        og = off[(off[:, 1] > T0) & (off[:, 1] < T1), 1]
        cg = np.diff(cz) / lfam.mean_spacing(spec, cz[:-1])
        big = np.nonzero(cg > 1.5)[0]
        res["offline_in_big_core_gap"] = float(np.mean([np.any((cz[big] < o) & (cz[big + 1] > o)) for o in og]))
        res["n_offline"] = len(og)
    out["D"][str(spec)] = res
    print(f"D {spec}: core misses {res['missed_line_zeros']} true line zeros, nearest-neighbour gap median "
          f"{res['missed_gap_median']:.3f} (all zeros {res['all_gap_median']:.3f})"
          + (f"; {res['offline_in_big_core_gap']:.3f} of {res['n_offline']} off-line zeros sit in a core gap > 1.5"
             if spec[0] == "dh" else ""), flush=True)

sr.save("l09", out)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
pp = np.exp(np.linspace(np.log(0.3), np.log(30), 300))
ax[0].plot(pp, model(pp), "k-", label="R = 1 − P(1 − e^(−1/P))  (no fit)")
ax[0].plot(pp, 1 / (1 + fit["c"] * pp), "k:", lw=.8, label=f"earlier fit 1/(1+{fit['c']:.2f}P)")
dP = [p for p in pts if not p[3].startswith("zeta")]
ax[0].errorbar([p[0] for p in dP], [p[1] for p in dP], [se[i] for i, p in enumerate(pts) if not p[3].startswith("zeta")],
               fmt="o", mfc="none", ms=5, label="Dirichlet L")
zP = [(i, p) for i, p in enumerate(pts) if p[3].startswith("zeta")]
ax[0].errorbar([p[0] for _, p in zP], [p[1] for _, p in zP], [se[i] for i, _ in zP], fmt="k^", ms=5,
               label="ζ, T = 2·10³ … 10²⁰")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("P"); ax[0].set_ylabel("R")
ax[0].legend(fontsize=7); ax[0].set_title("A: locking law from a random-phase model")
ax[1].plot([r["R_model"] for r in rowsBC], [r["R_core"] for r in rowsBC], "o", mfc="none", label="core zeros")
ax[1].plot([r["R_model"] for r in dirs], [r["R_true"] for r in dirs], "s", ms=4, label="true zeros, Dirichlet L")
ax[1].plot([r["R_model"] for r in dhs], [r["R_true"] for r in dhs], "s", c="C3", ms=4, label="true line zeros, DH type")
ax[1].plot([0, .45], [0, .45], "k-", lw=.7)
ax[1].set_xlabel("model R(P)"); ax[1].set_ylabel("measured R"); ax[1].legend(fontsize=7)
ax[1].set_title("B: model vs core vs true zeros")
ax[2].plot([r["f_core"] for r in dirs], [r["f_true"] for r in dirs], "o", mfc="none", label="Dirichlet L")
ax[2].plot([r["f_core"] for r in dhs], [r["f_true"] for r in dhs], "s", c="C3", mfc="none", label="DH type")
lo = min(min(r["f_core"] for r in rowsBC), min(r["f_true"] for r in rowsBC)) - .01
ax[2].plot([lo, 1.01], [lo, 1.01], "k-", lw=.7)
ax[2].set_xlabel("share of zeros kept by the core 2Re(e^{iθ}F)"); ax[2].set_ylabel("share on the line (true)")
ax[2].legend(fontsize=7); ax[2].set_title("C: does the main sum know about vacancies?")
plt.tight_layout(); plt.savefig(RESULTS / "l09.png", dpi=110)
