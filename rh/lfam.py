"""A family of L-functions for the Gram-locking test (Arb via python-flint).

spec = ("dir", q, i)  Dirichlet L(s, chi) for the primitive character with Conrey index i mod q (q=1: zeta)
spec = ("dh",  q, i)  the real combination of chi and conj(chi) with a functional equation (Davenport-Heilbronn
                      type; ("dh", 5, 2) is DH itself): f = (a L(chi) + a eps L(chi-bar)) / (2 Re a), a = eps^(-1/2),
                      eps = root number of chi. Coefficients a_n = Re(a chi(n)) / Re(a), real, a_1 = 1.
All have Lambda(s) = (q/pi)^((s+p)/2) Gamma((s+p)/2) F(s), p = parity, and (for "dh" and real chi) Z(t) =
e^{i theta(t)} F(1/2 + it) real. For complex chi, Z(t) = e^{i theta} eps^(-1/2) L is real (flint hardy_z).
"""
import cmath
import math
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq
from scipy.special import loggamma

_cache = {}


def _char(q, i):
    from flint import ctx, dirichlet_char
    ctx.prec = 64
    if (q, i) not in _cache:
        _cache[q, i] = dirichlet_char(q, i)
    return _cache[q, i]


def parity(q, i):
    return int(_char(q, i).parity()) if q > 1 else 0


def root_number(q, i):
    """eps with Lambda(s, chi) = eps Lambda(1 - s, chi-bar), evaluated numerically."""
    import mpmath as mp
    key = ("eps", q, i)
    if key not in _cache:
        from flint import acb
        c = _char(q, i)
        cb = _char(q, conj_index(q, i))
        p = parity(q, i)
        s = complex(0.3, 1.7)

        def lam(ch, z):
            v = ch.l(acb(z.real, z.imag))
            return complex(mp.power(q / mp.pi, (z + p) / 2) * mp.gamma((z + p) / 2)) * complex(
                float(v.real.mid()), float(v.imag.mid()))
        _cache[key] = lam(c, s) / lam(cb, 1 - s)
    return _cache[key]


def conj_index(q, i):
    """Conrey index of the conjugate character."""
    c = _char(q, i)
    vals = [complex(c(n)) for n in range(1, q + 1)]
    for j in range(1, q + 1):
        if math.gcd(j, q) == 1:
            d = _char(q, j)
            if all(abs(complex(d(n)) - v.conjugate()) < 1e-9 for n, v in zip(range(1, q + 1), vals)):
                return j
    raise ValueError


def coefficients(spec, N):
    kind, q, i = spec
    c = _char(q, i) if q > 1 else None
    chi = np.array([complex(c(n)) if c else 1.0 for n in range(N + 1)]) if q > 1 else np.r_[0, np.ones(N)]
    if q > 1:
        chi[0] = 0
    if kind == "dir":
        return chi
    a = root_number(q, i) ** -0.5
    return (a * chi).real / a.real


def value(spec, s):
    from flint import acb
    kind, q, i = spec
    z = acb(s.real, s.imag)
    if kind == "dir":
        v = _char(q, i).l(z)
        return complex(float(v.real.mid()), float(v.imag.mid()))
    eps = root_number(q, i)
    a = eps ** -0.5
    v1 = _char(q, i).l(z)
    v2 = _char(q, conj_index(q, i)).l(z)
    v1 = complex(float(v1.real.mid()), float(v1.imag.mid()))
    v2 = complex(float(v2.real.mid()), float(v2.imag.mid()))
    return (a * v1 + a * eps * v2) / (2 * a.real)


def theta(spec, t):
    kind, q, i = spec
    p = parity(q, i)
    t = np.asarray(t, float)
    return t / 2 * np.log(q / np.pi) + loggamma((0.5 + p + 1j * t) / 2).imag


def Z(spec, t):
    kind, q, i = spec
    if kind == "dir":
        from flint import acb
        v = _char(q, i).hardy_z(acb(t))
        return float(v.real.mid())
    return (cmath.exp(1j * float(theta(spec, t))) * value(spec, complex(0.5, t))).real


def mean_spacing(spec, t):
    return 2 * np.pi / np.log(max(spec[1], 1) * t / (2 * np.pi))


def _zv(args):
    spec, ts = args
    return np.array([Z(spec, t) for t in ts])


def _ref(args):
    spec, a, b = args
    return brentq(lambda t: Z(spec, t), a, b, xtol=1e-10)


def _absf(args):
    spec, sigma, ts = args
    return np.array([abs(value(spec, complex(sigma, t))) for t in ts])


def _newton(args):
    """Secant iteration for a complex zero of F from s0."""
    spec, s0 = args
    s1 = s0 + 1e-3
    f0, f1 = value(spec, s0), value(spec, s1)
    for _ in range(40):
        if f1 == f0:
            break
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        if not (-2 < s2.real < 3 and abs(s2.imag - s0.imag) < 5):
            return None
        s0, f0, s1, f1 = s1, f1, s2, value(spec, s2)
        if abs(s1 - s0) < 1e-12:
            break
    return s1 if abs(f1) < 1e-9 else None


def offline_zeros(spec, T0, T1, seeds, procs=20):
    """Zeros with Re s > 1/2 in T0 < Im s < T1 from complex seeds (a Re < 1/2 hit stands for its partner)."""
    with Pool(procs) as pool:
        res = pool.map(_newton, [(spec, s) for s in seeds], chunksize=16)
    found = {(round(max(r.real, 1 - r.real), 7), round(r.imag, 7)) for r in res
             if r is not None and abs(r.real - 0.5) > 1e-6 and T0 < r.imag < T1}
    return np.array(sorted(found, key=lambda p: p[1])).reshape(-1, 2)


def scan_seeds(spec, T0, T1, sigmas=(0.6, 0.75, 0.9), per_spacing=4, amax=0.8, procs=20):
    """Minima of |F| along vertical lines."""
    ts = [T0]
    while ts[-1] < T1:
        ts.append(ts[-1] + mean_spacing(spec, ts[-1]) / per_spacing)
    ts = np.array(ts)
    seeds = []
    with Pool(procs) as pool:
        for sg in sigmas:
            a = np.concatenate(pool.map(_absf, [(spec, sg, c) for c in np.array_split(ts, procs * 4)]))
            k = np.nonzero((a[1:-1] < a[:-2]) & (a[1:-1] < a[2:]) & (a[1:-1] < amax))[0] + 1
            seeds += [complex(sg, t) for t in ts[k]]
    return seeds


def line_zeros(spec, T0, T1, per_spacing=8, procs=20):
    """Sign changes of Z on [T0, T1] (grid of 1/8 mean spacing). Close pairs inside one cell are missed;
    at 1/8 spacing this costs ~0.1% of zeros for GUE statistics."""
    ts = [T0]
    while ts[-1] < T1:
        ts.append(ts[-1] + mean_spacing(spec, ts[-1]) / per_spacing)
    ts = np.array(ts)
    with Pool(procs) as pool:
        z = np.concatenate(pool.map(_zv, [(spec, c) for c in np.array_split(ts, procs * 4)]))
        sc = np.nonzero(np.sign(z[:-1]) != np.sign(z[1:]))[0]
        return np.array(pool.map(_ref, [(spec, ts[k], ts[k + 1]) for k in sc], chunksize=32))
