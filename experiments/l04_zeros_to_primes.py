"""L-angle 4 - the 'primes' know where the zeros are, including how far off the line.

For F = sum a_n n^-s let -F'/F = sum c_n n^-s  (zeta: Lambda(n); chi4: chi(n) Lambda(n); DH: no Euler product,
c_n from the Dirichlet-series recursion). psi_F(x) = sum_{n<=x} c_n = -sum_rho x^rho / rho + (smooth terms) [+ x for zeta].
Target y(u) = (psi_F(x) - [x])/sqrt(x), u = log x, x in [e^2, 1e7].

A. One common exponent beta (profile scan) - as in e04.
B. One free exponent beta_j per zero height gamma_j (a 'linear layer with learnable exponents'):
   y ~ sum_j exp((beta_j - 1/2)(u - uc)) [A_j sin(gamma_j u) + B_j cos(gamma_j u)] + C0 e^{-u/2} + C1 u e^{-u/2}.
   The network only gets the heights. Does it find which zeros are off the line, and their real parts?
C. Explicit formula with the true zeros: R^2 with and without the off-line zeros.
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

rng = np.random.default_rng(14)
fam = lfun.family()
K, XMAX = 300, 10**7
u = np.sort(rng.uniform(2.0, np.log(XMAX - 1), 60000))
x = np.exp(u)
xi = np.floor(x).astype(np.int64)
uc = u.mean()
out = {}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))

for name, F in fam.items():
    if name == "zeta":
        c = data.von_mangoldt(XMAX)
        y = (np.cumsum(c)[xi] - x) / np.sqrt(x)
    else:
        c = lfun.dirichlet_log_coeffs(lfun.coef(name, XMAX))
        y = np.cumsum(c)[xi] / np.sqrt(x)
    # zero heights: line zeros and one entry per off-line pair; true real parts kept for scoring only
    hs = np.r_[F["g"], F["off"][:, 1]]
    re = np.r_[np.full(len(F["g"]), 0.5), F["off"][:, 0]]
    o = np.argsort(hs)[:K]
    gam, beta_true = hs[o], re[o]
    osc = np.c_[np.sin(np.outer(u, gam)), np.cos(np.outer(u, gam))]
    smooth = np.c_[np.exp(-u / 2), u * np.exp(-u / 2)]

    # ---- A: common beta
    betas = np.linspace(0.3, 0.9, 61)
    r2s = []
    for bb in betas:
        D = np.c_[np.exp((bb - 0.5) * (u - uc))[:, None] * osc, smooth]
        w = np.linalg.lstsq(D, y, rcond=None)[0]
        r2s.append(1 - np.var(y - D @ w) / np.var(y))
    r2s = np.array(r2s)
    ib = int(r2s.argmax())
    cA = np.polyfit(betas[max(ib - 1, 0):ib + 2], r2s[max(ib - 1, 0):ib + 2], 2)
    beta_common = -cA[1] / (2 * cA[0]) if cA[0] < 0 else betas[ib]

    # ---- B: per-zero exponents (Adam, all start at 1/2)
    OSC, SM, U_, Y = nets.tensor(osc), nets.tensor(smooth), nets.tensor(u - uc), nets.tensor(y)
    D0 = np.c_[osc, smooth]
    w0 = np.linalg.lstsq(D0, y, rcond=None)[0]
    bj = torch.full((K,), 0.5, device=nets.DEV, requires_grad=True)
    A = torch.tensor(w0[:K], device=nets.DEV, dtype=torch.float32, requires_grad=True)
    B = torch.tensor(w0[K:2 * K], device=nets.DEV, dtype=torch.float32, requires_grad=True)
    C = torch.tensor(w0[2 * K:], device=nets.DEV, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([{"params": [A, B, C], "lr": 3e-3}, {"params": [bj], "lr": 2e-3}])
    for ep in range(8000):
        opt.zero_grad()
        amp = torch.exp((bj[None, :] - 0.5) * U_[:, None])
        pred = (amp * OSC[:, :K]) @ A + (amp * OSC[:, K:]) @ B + SM @ C
        loss = torch.mean((pred - Y) ** 2)
        loss.backward()
        opt.step()
    r2B = 1 - loss.item() / np.var(y)
    bj = bj.detach().cpu().numpy()
    on = beta_true == 0.5
    spread = np.median(np.abs(bj[on] - 0.5)) * 1.4826 if on.any() else np.nan
    # amplitude-weighted view: low zeros carry the signal
    flagged = bj > 0.5 + max(5 * spread, 0.02)
    print(f"{name}: common beta = {beta_common:.4f} (R^2 {r2s.max():.4f}); per-zero fit R^2 {r2B:.4f};"
          f" line zeros: median |beta_j - 1/2| = {np.median(np.abs(bj[on] - 0.5)):.4f}; flagged {flagged.sum()}"
          f" (off-line among first K: {(~on).sum()})", flush=True)
    for j in np.nonzero(~on)[0]:
        print(f"    off-line zero gamma={gam[j]:.3f}: true Re = {beta_true[j]:.4f}, learned beta_j = {bj[j]:.4f}")
    # ---- C: explicit formula with the true zeros
    def ef(sel):
        rho = beta_true[sel] + 1j * gam[sel]
        terms = -2 * (np.exp(np.outer(u, rho)) / rho).real / np.sqrt(x)[:, None]
        if name == "dh":  # partner 1 - conj(rho) of each off-line zero
            off = sel & ~on
            rp = (1 - beta_true[off]) + 1j * gam[off]
            terms = np.c_[terms, -2 * (np.exp(np.outer(u, rp)) / rp).real / np.sqrt(x)[:, None]]
        Dm = np.c_[terms.sum(1), smooth]
        w = np.linalg.lstsq(Dm, y, rcond=None)[0]
        return 1 - np.var(y - Dm @ w) / np.var(y)
    r2_all = ef(np.ones(K, bool))
    r2_line = ef(on)
    print(f"    explicit formula, true zeros: R^2 = {r2_all:.4f}; line zeros only: {r2_line:.4f}", flush=True)
    out[name] = {"beta_common": beta_common, "profile_betas": betas, "profile_r2": r2s, "r2_perzero": r2B,
                 "beta_j": bj, "gamma": gam, "beta_true": beta_true, "line_median_dev": np.median(np.abs(bj[on] - 0.5)),
                 "flagged": int(flagged.sum()), "n_off_in_K": int((~on).sum()),
                 "flag_hits": int(np.sum(flagged & ~on)), "flag_false": int(np.sum(flagged & on)),
                 "r2_ef_all": r2_all, "r2_ef_line": r2_line, "c_max_abs": float(np.abs(c[:XMAX]).max())}
    ax[0].plot(betas, r2s, ".-", label=F["label"])
    ax[1].plot(gam[on], bj[on], ".", ms=3, label=f"{F['label']} (line zeros)")
    if (~on).any():
        ax[1].plot(gam[~on], bj[~on], "o", mfc="none", ms=8, c="C3", label="DH: learned β_j at off-line zeros")
        ax[1].plot(gam[~on], beta_true[~on], "k+", ms=10, label="DH: true Re ρ")
    if name != "zeta":
        m = (u > 14) & (u < 14.3)
        ax[2].plot(u[m], y[m], ".", ms=1, label=f"{F['label']}: ψ_F(x)/√x")
ax[0].axvline(0.5, c="k", ls="--", lw=.7); ax[0].set_xlabel("common β"); ax[0].set_ylabel("R²"); ax[0].legend(fontsize=7)
ax[0].set_title("A: one exponent for all zeros")
ax[1].axhline(0.5, c="k", ls="--", lw=.7); ax[1].set_xlabel("γ_j"); ax[1].set_ylabel("learned β_j"); ax[1].legend(fontsize=6)
ax[1].set_title("B: one exponent per zero")
ax[2].legend(fontsize=7); ax[2].set_xlabel("u = log x"); ax[2].set_title("the data the network sees")
plt.tight_layout(); plt.savefig(RESULTS / "l04.png", dpi=110)
sr.save("l04", out)
