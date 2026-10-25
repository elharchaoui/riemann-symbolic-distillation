"""Angle 4 - the primes 'know' the zeros.

Target: y(u) = (psi(x) - x)/sqrt(x), u = log x, x in [e^2, 1e7].
Model:  y(u) ~ exp((beta - 1/2)(u - uc)) * sum_{j<=K} [A_j sin(gamma_j u) + B_j cos(gamma_j u)] + C exp(-u/2)
  A_j, B_j : one free weight pair per zero (a linear layer, no formula given)
  beta     : ONE free scalar = real part of the zeros as seen by the primes
  C        : weight of a smooth exp(-u/2) term
Two ways to fit beta: profile scan (exact least squares for each beta) and joint Adam from beta0 = 0.85 / 0.15.
Then PySR reads A(gamma), B(gamma); C is matched to a closed form.
Theory: A = -2g/(g^2+1/4), B = -1/(g^2+1/4), beta = 1/2, C = -log(2 pi).
(An earlier version with a free-form MLP message got stuck: see e04a_zeros_to_primes_mlp.py.)
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

rng = np.random.default_rng(4)
K = 300
XMAX = 10**7
gam = data.zeros(K)
psi = data.psi(XMAX)
u = np.sort(rng.uniform(2.0, np.log(XMAX - 1), 60000))
x = np.exp(u)
y = (psi[np.floor(x).astype(np.int64)] - x) / np.sqrt(x)
uc = u.mean()
osc = np.c_[np.sin(np.outer(u, gam)), np.cos(np.outer(u, gam))]  # (M, 2K)
smooth = np.exp(-u / 2)[:, None]


def design(beta):
    return np.c_[np.exp((beta - 0.5) * (u - uc))[:, None] * osc, smooth]


# ---- profile scan --------------------------------------------------------------------
betas = np.linspace(0.30, 0.70, 41)
r2s = []
for b in betas:
    D = design(b)
    w = np.linalg.lstsq(D, y, rcond=None)[0]
    r2s.append(1 - np.var(y - D @ w) / np.var(y))
r2s = np.array(r2s)
i = int(r2s.argmax())
# refine with a parabola through the top 3
c2 = np.polyfit(betas[i - 1:i + 2], r2s[i - 1:i + 2], 2)
beta_prof = -c2[1] / (2 * c2[0])
print(f"profile scan: best beta = {beta_prof:.4f} (grid max at {betas[i]:.2f}, R^2 = {r2s[i]:.4f})")

# ---- joint gradient descent from two starts --------------------------------------------
OSC = nets.tensor(osc); SM = nets.tensor(smooth); U = nets.tensor(u - uc); Y = nets.tensor(y)
runs = {}
for b0 in (0.85, 0.15):
    beta = torch.tensor(b0, device=nets.DEV, requires_grad=True)
    W = torch.zeros(2 * K, device=nets.DEV, requires_grad=True)
    C = torch.zeros(1, device=nets.DEV, requires_grad=True)
    opt = torch.optim.Adam([{"params": [W, C], "lr": 1e-2}, {"params": [beta], "lr": 3e-3}])
    traj = []
    for ep in range(6000):
        opt.zero_grad()
        pred = torch.exp((beta - 0.5) * U) * (OSC @ W) + SM[:, 0] * C
        loss = torch.mean((pred - Y) ** 2)
        loss.backward(); opt.step()
        traj.append(beta.item())
    r2 = 1 - loss.item() / np.var(y)
    print(f"Adam from beta0={b0}: beta = {beta.item():.4f}, R^2 = {r2:.4f}")
    runs[b0] = {"beta": beta.item(), "r2": r2, "traj": traj[::20]}

# ---- read the per-zero weights (at beta = 1/2) -----------------------------------------
D = design(0.5)
w = np.linalg.lstsq(D, y, rcond=None)[0]
A, B, Cs = w[:K], w[K:2 * K], w[-1]
r2_half = 1 - np.var(y - D @ w) / np.var(y)
_, eqA = sr.fit(gam[:, None], A, ["g"], "e04_A", unary="basic", niterations=80, maxsize=11)
_, eqB = sr.fit(gam[:, None], B, ["g"], "e04_B", unary="basic", niterations=80, maxsize=11)
A_th, B_th = -2 * gam / (gam**2 + 0.25), -1 / (gam**2 + 0.25)
kA = np.sum(A * A_th) / np.sum(A_th**2)
print("A(g) PySR:", eqA["elbow"], f"| scale vs theory {kA:.4f}, corr {np.corrcoef(A, A_th)[0,1]:.4f}")
print("B(g) PySR:", eqB["elbow"], f"| corr {np.corrcoef(B, B_th)[0,1]:.4f}")
gA = np.mean(A * gam)
print(f"mean A*gamma = {gA:.4f} -> {sr.recognize(gA, tol=0.02)} (theory -2)")
print(f"smooth weight C = {Cs:.4f} -> {sr.recognize(-Cs, tol=0.02) and 'log(2*pi)' if abs(Cs + np.log(2*np.pi)) < 0.02 else sr.recognize(Cs, tol=0.02)} (theory -log 2pi = {-np.log(2*np.pi):.4f})")
r2_trunc = 1 - np.var(y - (osc @ np.r_[A_th, B_th] - np.log(2 * np.pi) * smooth[:, 0])) / np.var(y)
print(f"R^2: learned weights {r2_half:.4f}; explicit formula with true amplitudes {r2_trunc:.4f}")

sr.save("e04", {"beta_profile": beta_prof, "profile_betas": betas, "profile_r2": r2s,
                "beta_from_0.85": runs[0.85]["beta"], "beta_from_0.15": runs[0.15]["beta"],
                "r2": {str(k): v["r2"] for k, v in runs.items()}, "r2_truncated_theory": r2_trunc,
                "A_pysr": eqA, "B_pysr": eqB, "phi_pysr": {"elbow": f"A(γ) = {eqA['elbow']};  B(γ) = {eqB['elbow']}"},
                "phi_corr_theory": float(np.corrcoef(A, A_th)[0, 1]), "A_scale": kA, "mean_A_gamma": gA,
                "C": Cs, "b_pysr": {"elbow": f"C·e^(-u/2), C = {Cs:.4f}"},
                "beta_traj_085": runs[0.85]["traj"], "beta_traj_015": runs[0.15]["traj"]})

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(betas, r2s, "k.-"); ax[0].axvline(0.5, c="r", ls="--")
ax[0].set_xlabel("β (real part)"); ax[0].set_ylabel("best R²"); ax[0].set_title("profile: primes pick β")
for b0, rr in runs.items():
    ax[1].plot(np.arange(len(rr["traj"])) * 20, rr["traj"], label=f"start β={b0}")
ax[1].axhline(.5, c="k", ls="--"); ax[1].legend(); ax[1].set_xlabel("step"); ax[1].set_title("β during training")
ax[2].plot(gam, A, ".", ms=3, label="learned A_j (sin)")
ax[2].plot(gam, A_th, "k-", lw=.8, label="-2γ/(γ²+¼)")
ax[2].plot(gam, B * 10, ".", ms=3, label="learned B_j ×10 (cos)")
ax[2].plot(gam, B_th * 10, "k:", lw=.8, label="-10/(γ²+¼)")
ax[2].legend(fontsize=7); ax[2].set_xlabel("γ"); ax[2].set_title("per-zero weights")
plt.tight_layout(); plt.savefig(RESULTS / "e04.png", dpi=110)
