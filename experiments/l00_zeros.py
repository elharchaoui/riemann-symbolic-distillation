"""Zeros of L(s, chi_-4) and of the Davenport-Heilbronn function (see rh/lfun.py).

Critical-line zeros: sign changes of Z(t) on a grid of 1/6 mean spacing, Brent refinement, and a
closer look at every dip of |Z| without a sign change. Off-line zeros (DH only): secant iteration
in the complex plane, seeded at those dips and at minima of |F| along Re s = 0.6, 0.75, 0.9 (zeros far
from the line leave no dip). Blocks whose count does not close get a denser scan. Completeness: line count + 2 x off-line count must equal
theta(T)/pi + S(T) with S averaging to 0 (Turing-style check, per block of heights).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rh import DATA, lfun, sr

TMAX = {"chi4": 50000, "dh": 25000}
SEARCH_SIGMAS = (0.52, 0.56, 0.62, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3)


def block_check(name, g, off, T, rng):
    """Mean of N_found(t) - theta(t)/pi over random t in 25 blocks (= mean S, should be ~0)."""
    edges = np.linspace(0, T, 26)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        ts = rng.uniform(max(a, 20), b, 4000)
        nf = np.searchsorted(g, ts) + 2 * np.searchsorted(np.sort(off[:, 1]), ts)
        rows.append((a, b, float(np.mean(nf - lfun.n_smooth(name, ts)))))
    return rows


report = {}
for name, T in TMAX.items():
    t0 = time.time()
    cache = DATA / f"zeros_{name}_line.npz"
    if cache.exists():
        d = np.load(cache)
        g, dips, n_extra = d["g"], d["dips"], int(d["n_extra"])
    else:
        g, dips, n_extra = lfun.critical_zeros(name, T)
        np.savez(cache, g=g, dips=dips, n_extra=n_extra)
    print(f"{name}: {len(g)} zeros on the line below {T}; {len(dips)} dips, {n_extra} zeros recovered from dips "
          f"({time.time()-t0:.0f}s)", flush=True)
    off = np.zeros((0, 2))
    rng = np.random.default_rng(0)
    if name == "dh":
        # seeds: dips of |Z| on the line, and minima of |F| along vertical lines (zeros far from the line make no dip)
        seeds = list(dips) + lfun.scan_minima(name, 1.0, T, (0.6, 0.75, 0.9), per_spacing=3, amax=0.8)
        off = lfun.offline_zeros(name, T, sigmas=SEARCH_SIGMAS, seeds=seeds)
        print(f"   {len(off)} off-line zeros from {len(seeds)} seeds; max Re = {off[:, 0].max():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        blocks = block_check(name, g, off, T, rng)
        print("   mean S per block:", " ".join(f"{r[2]:+.2f}" for r in blocks), flush=True)
        for a, b, m in blocks:  # blocks that do not close: denser scan, all minima
            if abs(m) > 0.3:
                seeds = lfun.scan_minima(name, max(a, 1.0), b, (0.55, 0.65, 0.75, 0.85, 0.95, 1.05), per_spacing=4)
                more = lfun.offline_zeros(name, b, t_lo=max(a, 1.0), seeds=seeds)
                off = np.unique(np.r_[off, more], axis=0)
                print(f"   block {a:.0f}-{b:.0f} (mean S {m:+.2f}): dense scan -> {len(off)} off-line zeros", flush=True)
        off = off[np.argsort(off[:, 1])]
    blocks = block_check(name, g, off, T, rng)
    print("   mean S per block (final):", " ".join(f"{r[2]:+.2f}" for r in blocks), flush=True)
    np.savez(DATA / f"zeros_{name}.npz", g=g, off=off, dips=dips)
    report[name] = {"T": T, "n_line": len(g), "n_offline": len(off), "dips": len(dips), "blocks": blocks,
                    "seconds": time.time() - t0}
sr.save("l00", report)
