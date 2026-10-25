"""L-angle 3 - which integers n does S(t) 'listen to'? (same L1 linear network as e03)

Target S(t) = N(t) - N_smooth(t) from each zero list (DH: all zeros, line and off-line), t in [1000, T].
Features sin(t log n), cos(t log n) for every n = 2..150.
Theory  zeta : a_n = -Lambda(n) / (pi sqrt(n) log n)            support = prime powers
        chi4 : a_n = -chi(n) Lambda(n) / (pi sqrt(n) log n)      support = odd prime powers, sign chi(n)
        DH   : log f = sum b_n n^-s has b_n != 0 on composites too (no Euler product), and the series
               does not converge on Re s = 1/2 (zeros up to Re 0.87). Formal weights -b_n / (pi sqrt n).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rh import RESULTS, data, lfun, nets, sr

rng = np.random.default_rng(13)
fam = lfun.family()
NMAX, M = 150, 200000
ns = np.arange(2, NMAX + 1)
logn = np.log(ns)
lam = data.von_mangoldt(NMAX)[2:]
is_pp = lam > 0
k_of = np.array([round(np.log(n) / np.log(min(p for p in data.sieve(n) if n % p == 0))) for n in ns])
out = {}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))

for col, (name, F) in enumerate(fam.items()):
    t = rng.uniform(1000, F["T"], M)
    y = np.searchsorted(F["all"], t) - F["ns"](t)
    if name == "zeta":
        b = lam / logn
    else:
        a = lfun.coef(name, NMAX)
        b = lfun.dirichlet_log_coeffs(a)[2:] / logn
    theory = -b / (np.pi * np.sqrt(ns))
    T_ = torch.as_tensor(t[:, None], device=nets.DEV, dtype=torch.float64)
    ph = T_ * torch.as_tensor(logn[None, :], device=nets.DEV, dtype=torch.float64)
    X = torch.cat([torch.sin(ph), torch.cos(ph)], 1).float()
    Y = nets.tensor(y)
    lin = torch.nn.Linear(2 * len(ns), 1).to(nets.DEV)
    torch.nn.init.zeros_(lin.weight)
    nets.train(lin, lambda: torch.mean((lin(X)[:, 0] - Y) ** 2) + 2e-4 * lin.weight.abs().sum(), epochs=3000, lr=3e-3)
    Xn = X.cpu().numpy().astype(np.float64)
    ols = np.linalg.lstsq(np.c_[Xn, np.ones(M)], y, rcond=None)[0]
    o_sin, o_cos = ols[: len(ns)], ols[len(ns): 2 * len(ns)]
    thr = 3 * np.std(o_cos)
    support = np.abs(o_sin) > thr
    expected = np.abs(b) > 1e-12
    tp, fp, fn = np.sum(support & expected), np.sum(support & ~expected), np.sum(~support & expected)
    comp = support & ~is_pp  # selected n that are not prime powers
    r2 = 1 - np.var(y - np.c_[Xn, np.ones(M)] @ ols) / np.var(y)
    corr = np.corrcoef(o_sin[expected], theory[expected])[0, 1]
    sign_ok = np.mean(np.sign(o_sin[support & expected]) == np.sign(theory[support & expected]))
    print(f"{name}: selected {support.sum()} n; expected support {expected.sum()} -> hit {tp}, missed {fn}, extra {fp};"
          f" selected non-prime-powers: {ns[comp].tolist()[:20]}{'...' if comp.sum() > 20 else ''} ({comp.sum()});"
          f" corr with theory {corr:.4f}; sign agreement {sign_ok:.3f}; R^2 {r2:.3f}", flush=True)
    res = {"ns": ns, "w_sin": o_sin, "w_cos": o_cos, "theory": theory, "support": support, "tp": tp, "fp": fp,
           "fn": fn, "n_expected": expected.sum(), "n_selected": support.sum(), "nonpp_selected": ns[comp],
           "n_nonpp_selected": comp.sum(), "corr": corr, "sign_ok": sign_ok, "r2": r2}
    if name == "chi4":  # weight law on the selected odd prime powers: |a_n| vs (n, k)
        sel = support & is_pp
        _, eq = sr.fit(np.c_[ns[sel], k_of[sel]], np.abs(o_sin[sel]), ["n", "k"], "l03_chi4_weights",
                       unary=["sqrt"], niterations=80, maxsize=10)
        pr = sel & (k_of == 1)
        c = np.mean(np.abs(o_sin[pr]) * np.sqrt(ns[pr]))
        print(f"   chi4 |a_n| PySR: {eq['elbow']}; mean |a_p| sqrt p = {c:.5f} -> {sr.recognize(c, tol=0.01)}"
              f"; sign(a_p) = -chi(p) for {np.mean(np.sign(o_sin[pr]) == -np.where(ns[pr] % 4 == 1, 1, -1)):.3f} of primes")
        res.update({"pysr": eq, "c_prime": c, "c_id": sr.recognize(c, tol=0.01)})
    if name == "dh":  # how many of the formally expected composites carry real weight?
        big_b = expected & ~is_pp & (np.abs(theory) > 2 * thr)
        res["composites_expected_big"] = ns[big_b]
        res["composites_expected_big_hit"] = np.mean(support[big_b]) if big_b.any() else None
        print(f"   DH composites with large formal weight: {ns[big_b].tolist()}; selected {res['composites_expected_big_hit']}")
    out[name] = res
    a_ = ax[col]
    a_.stem(ns, o_sin, linefmt="C0-", markerfmt="C0.", basefmt=" ", label="learned sin weight")
    a_.plot(ns[expected], theory[expected], "rx", ms=4, label="theory −b_n/(π√n)")
    a_.plot(ns[comp], o_sin[comp], "o", mfc="none", c="C3", ms=7, label="selected, not a prime power")
    a_.plot(ns, o_cos, "k.", ms=2)
    a_.set_title(f"{F['label']}: R²={r2:.2f}"); a_.set_xlabel("n"); a_.legend(fontsize=6)
plt.tight_layout(); plt.savefig(RESULTS / "l03.png", dpi=110)
sr.save("l03", out)
