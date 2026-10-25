"""Angle 3 - the zeros 'know' the primes.

Target: S(t) = N(t) - theta(t)/pi - 1 (fluctuation of the zero count), exact from the zero table.
Inputs: sin(t log n), cos(t log n) for EVERY integer n = 2..NMAX (no hint about primes).
Model: a linear layer (one-layer network) trained with an L1 penalty -> sparse support.

Questions the network must answer by itself:
  1. Which n matter? (theory: prime powers only)
  2. sin or cos?     (theory: sin only)
  3. Weight law?     PySR on the surviving weights (theory: -1/(pi k p^{k/2}) for n = p^k).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rh import RESULTS, data, nets, sr

rng = np.random.default_rng(3)
g = data.zeros()
NMAX = 150
M = 300000

t = rng.uniform(1e4, g[-1], M)
y = data.S_grid(t, g)
ns = np.arange(2, NMAX + 1)
logn = np.log(ns)

T_ = nets.tensor(t[:, None]).double()
LN = torch.as_tensor(logn[None, :], device=nets.DEV, dtype=torch.float64)
Y = nets.tensor(y).double()


def features(Tb):
    ph = Tb * LN  # float64: t log n up to 1.6e7 rad
    return torch.cat([torch.sin(ph), torch.cos(ph)], 1).float()


lin = torch.nn.Linear(2 * len(ns), 1, bias=True).to(nets.DEV)
torch.nn.init.zeros_(lin.weight)
X = features(T_)
lam = 2e-4


def loss():
    return torch.mean((lin(X)[:, 0] - Y.float()) ** 2) + lam * lin.weight.abs().sum()


nets.train(lin, loss, epochs=3000, lr=3e-3)
w = lin.weight.detach().cpu().numpy()[0]
a_sin, a_cos = w[: len(ns)], w[len(ns):]

# OLS for unbiased values on the support the network selected
Xn = X.cpu().numpy().astype(np.float64)
ols = np.linalg.lstsq(np.c_[Xn, np.ones(M)], y, rcond=None)[0]
o_sin, o_cos = ols[: len(ns)], ols[len(ns): 2 * len(ns)]

lam_n = data.von_mangoldt(NMAX)[2:]
is_pp = lam_n > 0
theory = -lam_n / (np.pi * np.sqrt(ns) * logn)

thr = 0.25 * np.abs(a_sin).max() * 0 + 3 * np.std(o_cos)  # noise floor from the cos weights
support = np.abs(o_sin) > thr
tp = np.sum(support & is_pp); fp = np.sum(support & ~is_pp); fn = np.sum(~support & is_pp)
print(f"support: {support.sum()} selected; prime powers hit {tp}/{is_pp.sum()}, false+ {fp}, missed {fn}")
print("largest |cos| weight:", np.abs(o_cos).max(), " largest |sin| weight:", np.abs(o_sin).max())
print("missed prime powers:", ns[~support & is_pp])
print("false positives:", ns[support & ~is_pp])

# PySR on the surviving weights. Inputs: n and k (the exponent if n = p^k, discovered support only)
k_of = np.array([round(np.log(n) / np.log(min(p for p in data.sieve(n) if n % p == 0))) if n > 1 else 0 for n in ns])
sel = support
_, eq = sr.fit(np.c_[ns[sel], k_of[sel]], o_sin[sel], ["n", "k"], "e03_weights", unary=["sqrt", "log"],
               niterations=120, maxsize=12)
print("PySR weight law:", eq["elbow"], "| best:", eq["best"])
# primes only: amplitude a_p * sqrt(p) should be constant -1/pi
primes = is_pp & (k_of == 1)
c = np.mean(o_sin[primes] * np.sqrt(ns[primes]))
print(f"mean a_p*sqrt(p) = {c:.5f} (theory -1/pi = {-1/np.pi:.5f}) -> recognized: {sr.recognize(c, tol=0.01)}")
corr = np.corrcoef(o_sin[is_pp], theory[is_pp])[0, 1]
print("corr(learned, theory) on prime powers:", corr)

# how much of S(t) do primes <= NMAX explain?
pred = np.c_[Xn, np.ones(M)] @ ols
r2 = 1 - np.var(y - pred) / np.var(y)
print(f"R^2 of S(t) explained by n<= {NMAX}: {r2:.3f}")

sr.save("e03", {"ns": ns, "w_sin_l1": a_sin, "w_cos_l1": a_cos, "w_sin_ols": o_sin, "w_cos_ols": o_cos,
                "theory": theory, "support": support, "is_prime_power": is_pp, "tp": tp, "fp": fp, "fn": fn,
                "pysr": eq, "c_prime": c, "c_identified": sr.recognize(c, tol=0.01), "corr": corr, "r2": r2})

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].stem(ns, o_sin, linefmt="C0-", markerfmt="C0.", basefmt=" ", label="learned sin weight")
ax[0].plot(ns[is_pp], theory[is_pp], "rx", label="-Λ(n)/(π√n log n)")
ax[0].plot(ns, o_cos, "k.", ms=2, label="learned cos weight")
ax[0].legend(); ax[0].set_xlabel("n"); ax[0].set_title("S(t) ≈ Σ a_n sin(t log n): network weights")
ax[1].loglog(ns[primes], -o_sin[primes], "o", label="primes  -a_p")
ax[1].loglog(ns[primes], 1 / (np.pi * np.sqrt(ns[primes])), "k-", label="1/(π√p)")
pp2 = is_pp & (k_of == 2)
ax[1].loglog(ns[pp2], -o_sin[pp2], "s", label="squares  -a_{p²}")
ax[1].loglog(ns[pp2], 1 / (2 * np.pi * np.sqrt(ns[pp2])), "k--", label="1/(2π p)")
ax[1].legend(); ax[1].set_title("weight law")
plt.tight_layout(); plt.savefig(RESULTS / "e03.png", dpi=110)
