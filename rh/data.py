"""Zeta zeros, primes and the smooth zero-counting function.

Zero sources: A. Odlyzko's tables (www-users.cse.umn.edu/~odlyzko/zeta_tables).
  zeros6  first 2,001,052 zeros (accuracy ~4e-9)
  zeros3  zeros #10^12+1 .. #10^12+10^4
  zeros4  zeros #10^21+1 .. #10^21+10^4
"""
import gzip
import re
from functools import lru_cache

import mpmath as mp
import numpy as np

from . import DATA

RAW = DATA / "raw"


@lru_cache
def zeros(n=None):
    """Imaginary parts gamma_1 < gamma_2 < ... of the first n nontrivial zeros."""
    cache = DATA / "zeros6.npy"
    if not cache.exists():
        with gzip.open(RAW / "zeros6.gz", "rt") as f:
            g = np.array([float(x) for x in f.read().split()])
        np.save(cache, g)
    g = np.load(cache)
    return g if n is None else g[:n]


@lru_cache
def high_zeros(which):
    """Zeros at large height. Returns (offset, local, first_index):
    gamma_k = offset + local[k - first_index]  (offset is an exact integer)."""
    fname, first = {"1e12": ("zeros3", 10**12 + 1), "1e21": ("zeros4", 10**21 + 1)}[which]
    text = (RAW / fname).read_text()
    offset = int(re.search(r"gamma\s*-\s*(\d+)", text).group(1))
    lines = text.splitlines()[7:]  # 7 header lines
    local = np.array([float(x) for x in " ".join(lines).split()])
    return offset, local, first


def theta(t):
    """Riemann-Siegel theta, asymptotic series (|error| < 1e-12 for t > 10)."""
    t = np.asarray(t, dtype=float)
    return t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8 + 1 / (48 * t) + 7 / (5760 * t**3)


def n_smooth(t):
    """Smooth part of N(t): theta(t)/pi + 1."""
    return theta(t) / np.pi + 1


def S_at_zeros(g, first_index=1):
    """S(gamma_n) with the midpoint convention N(gamma_n) = n - 1/2."""
    n = first_index + np.arange(len(g))
    return n - 0.5 - n_smooth(g)


def S_at_high_zeros(which):
    """Same as S_at_zeros for the high tables, computed in multiprecision."""
    offset, local, first = high_zeros(which)
    mp.mp.dps = 50
    out = np.empty(len(local))
    for k, x in enumerate(local):
        t = mp.mpf(offset) + mp.mpf(repr(float(x)))
        out[k] = float(first + k - mp.mpf("0.5") - mp.siegeltheta(t) / mp.pi - 1)
    return out


def S_grid(t, g):
    """S(t) = N(t) - theta(t)/pi - 1 on an arbitrary grid t (from the zero list g)."""
    return np.searchsorted(g, t) - n_smooth(t)


def unfold(g):
    """Map zeros to unit mean spacing: x_n = N_smooth(gamma_n)."""
    return n_smooth(g)


@lru_cache
def sieve(N):
    s = np.ones(N + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(N**0.5) + 1):
        if s[p]:
            s[p * p :: p] = False
    return np.nonzero(s)[0]


def von_mangoldt(N):
    """Lambda(n) for n = 0..N."""
    lam = np.zeros(N + 1)
    for p in sieve(N):
        pk = p
        while pk <= N:
            lam[pk] = np.log(p)
            pk *= p
    return lam


def mobius(N):
    mu = np.ones(N + 1, dtype=np.int64)
    mu[0] = 0
    for p in sieve(N):
        mu[p::p] *= -1
        mu[p * p :: p * p] = 0
    return mu


def psi(N):
    """Chebyshev psi(n) for integers n = 0..N."""
    return np.cumsum(von_mangoldt(N))
