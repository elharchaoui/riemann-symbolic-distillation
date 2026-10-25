"""Two more L-functions, to see which zeta laws need the Euler product.

chi4  L(s, chi_-4) = 1 - 3^-s + 5^-s - ...    Euler product, functional equation   (q=4, odd)
dh    Davenport-Heilbronn f(s) = sum a_n n^-s, a = (1, k, -k, -1, 0) periodic mod 5,
      k = (sqrt(10 - 2 sqrt 5) - 2)/(sqrt 5 - 1).  Same type of functional equation (q=5, odd),
      NO Euler product, infinitely many zeros off the critical line.
      f = (1 - ik)/2 L(s, chi) + (1 + ik)/2 L(s, chi-bar),  chi mod 5 with chi(2) = i.

Both: Lambda(s) = (q/pi)^((s+1)/2) Gamma((s+1)/2) F(s) = Lambda(1 - s), real on the critical line.
Z(t) = e^{i theta(t)} F(1/2 + it) is real,  theta(t) = t/2 log(q/pi) + Im log Gamma((3/2 + it)/2),
N(T) = theta(T)/pi + S(T)  (no pole, so no "+1" as for zeta).
Values from Arb (python-flint).
"""
import math
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import loggamma

from . import DATA

KAPPA = (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)
SPEC = {
    "chi4": {"q": 4, "a": 1, "coef": [0, 1, 0, -1], "label": "L(s, χ₋₄)"},
    "dh": {"q": 5, "a": 1, "coef": [0, 1, KAPPA, -KAPPA, -1], "label": "Davenport–Heilbronn"},
}
_chars = {}


def _ch(q, i):
    from flint import ctx, dirichlet_char
    ctx.prec = 64
    if (q, i) not in _chars:
        _chars[q, i] = dirichlet_char(q, i)
    return _chars[q, i]


def value(name, s):
    """F(s) as a Python complex (s complex)."""
    from flint import acb
    s = acb(s.real, s.imag)
    if name == "chi4":
        v = _ch(4, 3).l(s)
    else:
        v = (1 - 1j * KAPPA) / 2 * _ch(5, 2).l(s) + (1 + 1j * KAPPA) / 2 * _ch(5, 3).l(s)
    return complex(float(v.real.mid()), float(v.imag.mid()))


def theta(name, t):
    q, a = SPEC[name]["q"], SPEC[name]["a"]
    t = np.asarray(t, float)
    return t / 2 * np.log(q / np.pi) + loggamma((0.5 + a + 1j * t) / 2).imag


def n_smooth(name, t):
    return theta(name, t) / np.pi


def Z(name, t):
    return (np.exp(1j * theta(name, t)) * value(name, complex(0.5, t))).real


def coef(name, N):
    """a_n for n = 0..N."""
    c = SPEC[name]["coef"]
    return np.array([c[n % len(c)] for n in range(N + 1)], float)


def mean_spacing(name, t):
    return 2 * np.pi / np.log(SPEC[name]["q"] * np.maximum(t, 10) / (2 * np.pi))


# ---- parallel workers ---------------------------------------------------------------
def _zvals(args):
    name, ts = args
    return np.array([Z(name, t) for t in ts])


def _refine(args):
    name, a, b = args
    return brentq(lambda t: Z(name, t), a, b, xtol=1e-11, rtol=1e-14)


def _dip(args):
    """|Z| has a local minimum without a sign change in [a, b]: look closer."""
    name, a, b, sgn = args
    r = minimize_scalar(lambda t: sgn * Z(name, t), bounds=(a, b), method="bounded",
                        options={"xatol": 1e-9})
    zm = Z(name, r.x)
    if sgn * zm >= 0:
        return []
    out = []  # two sign changes hidden in the cell
    for lo, hi in ((a, r.x), (r.x, b)):
        if np.sign(Z(name, lo)) != np.sign(Z(name, hi)):
            out.append(brentq(lambda t: Z(name, t), lo, hi, xtol=1e-11))
    return out


def _newton(args):
    """Secant iteration for a complex zero of F from s0."""
    name, s0 = args
    s1 = s0 + 1e-3
    f0, f1 = value(name, s0), value(name, s1)
    for _ in range(40):
        if f1 == f0:
            break
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        if not (0.0 < s2.real < 3 and abs(s2.imag - s0.imag) < 5):
            return None
        s0, f0, s1, f1 = s1, f1, s2, value(name, s2)
        if abs(s1 - s0) < 1e-12:
            break
    return s1 if abs(f1) < 1e-9 else None


def _absf(args):
    name, sigma, ts = args
    return np.array([abs(value(name, complex(sigma, t))) for t in ts])


def scan_minima(name, t_lo, t_hi, sigmas, per_spacing=4, amax=np.inf, procs=20):
    """Points sigma + it where |F| has a local minimum along vertical lines (seeds for off-line zeros)."""
    ts = [t_lo]
    while ts[-1] < t_hi:
        ts.append(ts[-1] + mean_spacing(name, ts[-1]) / per_spacing)
    ts = np.array(ts)
    seeds = []
    with Pool(procs) as pool:
        for sg in sigmas:
            a = np.concatenate(pool.map(_absf, [(name, sg, c) for c in np.array_split(ts, procs * 40)]))
            i = np.nonzero((a[1:-1] < a[:-2]) & (a[1:-1] < a[2:]) & (a[1:-1] < amax))[0] + 1
            seeds += [complex(sg, t) for t in ts[i]]
    return seeds


def critical_zeros(name, T, per_spacing=6, procs=20, t0=0.5):
    """Zeros on the critical line 0 < t < T (sign changes of Z, plus a dip check for close pairs)."""
    ts = [t0]
    while ts[-1] < T:
        ts.append(ts[-1] + mean_spacing(name, ts[-1]) / per_spacing)
    ts = np.array(ts)
    with Pool(procs) as pool:
        chunks = np.array_split(ts, procs * 40)
        z = np.concatenate(pool.map(_zvals, [(name, c) for c in chunks]))
        sc = np.nonzero(np.sign(z[:-1]) != np.sign(z[1:]))[0]
        roots = pool.map(_refine, [(name, ts[i], ts[i + 1]) for i in sc], chunksize=64)
        az = np.abs(z)
        dips = [i for i in range(1, len(z) - 1)
                if az[i] < az[i - 1] and az[i] < az[i + 1] and np.sign(z[i - 1]) == np.sign(z[i]) == np.sign(z[i + 1])]
        extra = pool.map(_dip, [(name, ts[i - 1], ts[i + 1], np.sign(z[i])) for i in dips], chunksize=16)
    extra = [r for pair in extra for r in pair]
    return np.unique(np.round(np.r_[roots, extra], 10)), ts[dips], len(extra)


def offline_zeros(name, T, sigmas=(0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.6), per_spacing=2, procs=20, t_lo=1.0,
                  seeds=None):
    """Zeros with Re s > 1/2, t_lo < Im s < T, by secant iteration from a grid of starting points
    (or from the given seed heights, e.g. dips of |Z| on the critical line)."""
    starts = []
    if seeds is not None:
        starts = [sd if isinstance(sd, complex) else complex(s, sd) for sd in seeds
                  for s in ([0] if isinstance(sd, complex) else sigmas)]
    else:
        t = t_lo
        while t < T:
            starts += [complex(s, t) for s in sigmas]
            t += mean_spacing(name, t) / per_spacing
    with Pool(procs) as pool:
        res = pool.map(_newton, [(name, s) for s in starts], chunksize=32)
    # a zero with Re < 1/2 found by the iteration stands for its partner 1 - conj(rho)
    found = sorted({(round(max(r.real, 1 - r.real), 7), round(r.imag, 7)) for r in res
                    if r is not None and abs(r.real - 0.5) > 1e-6 and t_lo - 1 < r.imag < T}, key=lambda p: p[1])
    return np.array(found).reshape(-1, 2)


def load(name):
    d = np.load(DATA / f"zeros_{name}.npz")
    return {k: d[k] for k in d.files}


ZETA_TMAX = 50000


def family():
    """The three functions compared: zeta (zeros below ZETA_TMAX), chi4, dh.
    Each entry: g = zeros on the critical line, off = (sigma, gamma) of zeros with sigma > 1/2,
    all = heights of all zeros (an off-line pair rho, 1 - conj(rho) counted twice), ns = smooth N(t), q."""
    from . import data
    z = data.zeros()
    fam = {"zeta": {"g": z[z < ZETA_TMAX], "off": np.zeros((0, 2)), "ns": data.n_smooth, "q": 1,
                    "label": "ζ(s)", "T": ZETA_TMAX, "coef": None}}
    for name in SPEC:
        d = load(name)
        fam[name] = {"g": d["g"], "off": d["off"], "ns": (lambda t, n=name: n_smooth(n, t)), "q": SPEC[name]["q"],
                     "label": SPEC[name]["label"], "T": float(np.ceil(d["g"][-1])), "coef": SPEC[name]["coef"]}
    for F in fam.values():
        F["all"] = np.sort(np.r_[F["g"], F["off"][:, 1], F["off"][:, 1]])
    return fam


def dirichlet_log_coeffs(a):
    """For F = sum a_n n^-s (a_1 = 1): c_n of -F'/F = sum c_n n^-s  (n = 0..N).
    Uses sum_{d | n} c_d a_{n/d} = a_n log n."""
    N = len(a) - 1
    r = a * np.log(np.maximum(np.arange(N + 1), 1))
    c = np.zeros(N + 1)
    for d in range(1, N + 1):
        c[d] = r[d]
        if c[d] != 0 and 2 * d <= N:
            m = N // d
            r[2 * d::d][: m - 1] -= c[d] * a[2: m + 1]
    return c
