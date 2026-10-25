"""Angle 4 - the primes 'know' the zeros (Cranmer-style inductive bias: a sum over zeros).

Target: y(u) = (psi(x) - x)/sqrt(x), u = log x, x in [e^2, 1e7].
Model:  y(u) ~ sum_{j<=K} exp((beta - 1/2) u) * phi(gamma_j, sin(gamma_j u), cos(gamma_j u)) + b(u)
  phi  : shared MLP (the per-zero 'message'), distilled with PySR afterwards
  beta : ONE free scalar = real part of the zeros as seen by the primes (started away from 1/2)
  b(u) : small MLP for the smooth non-oscillating part, distilled with PySR
Theory: phi = -(cos(g u) + 2 g sin(g u)) / (g^2 + 1/4),  beta = 1/2,
        b(u) = -(log(2 pi) + log(1 - x^-2)/2)/sqrt(x) ~ -log(2 pi) e^{-u/2}.
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

u = np.sort(rng.uniform(2.0, np.log(XMAX - 1), 120000))
x = np.exp(u)
y = (psi[np.floor(x).astype(np.int64)] - x) / np.sqrt(x)

U = nets.tensor(u[:, None])
UC = float(u.mean())
Y = nets.tensor(y)
G = nets.tensor(gam[None, :])
ph = torch.as_tensor(u[:, None] * gam[None, :], device=nets.DEV, dtype=torch.float64)
SIN, COS = torch.sin(ph).float(), torch.cos(ph).float()


class ZeroSum(torch.nn.Module):
    def __init__(self, beta0):
        super().__init__()
        self.phi = nets.MLP(3, 1, width=48, depth=3)
        self.beta = torch.nn.Parameter(torch.tensor(float(beta0)))
        self.b = nets.MLP(1, 1, width=16, depth=2)

    def messages(self, idx=slice(None)):
        Gb = (G / 100).expand(len(U), -1)[idx]
        inp = torch.stack([Gb, SIN[idx], COS[idx]], -1)
        return self.phi(inp)[..., 0]

    def forward(self, idx=slice(None)):
        amp = torch.exp((self.beta - 0.5) * (U[idx] - UC))  # centred: stable for any beta
        return (amp * self.messages(idx)).sum(1) + self.b(U[idx] / 10)[:, 0]


EPOCHS = 8000
results = {}
for beta0 in (0.85, 0.15):
    torch.manual_seed(0)
    model = ZeroSum(beta0).to(nets.DEV)
    opt = torch.optim.Adam([{"params": [p for n, p in model.named_parameters() if n != "beta"]},
                            {"params": [model.beta], "lr": 5e-3}], lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    traj = []
    for ep in range(EPOCHS):
        idx = torch.randint(0, len(U), (8192,), device=nets.DEV)
        opt.zero_grad()
        loss = torch.mean((model(idx) - Y[idx]) ** 2)
        loss.backward()
        opt.step(); sched.step()
        traj.append(model.beta.item())
    with torch.no_grad():
        full = torch.cat([model(torch.arange(i, min(i + 20000, len(U)), device=nets.DEV))
                          for i in range(0, len(U), 20000)]).cpu().numpy()
    r2 = 1 - np.var(y - full) / np.var(y)
    print(f"beta0={beta0}: learned beta = {model.beta.item():.4f}   R^2 = {r2:.4f}")
    results[beta0] = (model, traj, r2, full)

best0 = max(results, key=lambda k: results[k][2])
model, _, r2, full = results[best0]
print("distilling the run started at beta0 =", best0)

# baseline: the truncated explicit formula itself with the same K zeros (true amplitudes)
th = (-(np.cos(u[:, None] * gam) + 2 * gam * np.sin(u[:, None] * gam)) / (gam**2 + 0.25)).sum(1) \
     - np.log(2 * np.pi) / np.sqrt(x)
r2_theory = 1 - np.var(y - th) / np.var(y)
print(f"truncated explicit formula with K={K}: R^2 = {r2_theory:.4f}")

# ---- distil phi (the per-zero message) -----------------------------------------------
gs = rng.choice(gam, 3000)
us = rng.uniform(2, np.log(XMAX), 3000)
inp = np.c_[gs / 100, np.sin(gs * us), np.cos(gs * us)]
with torch.no_grad():
    msg = model.phi(nets.tensor(inp))[:, 0].cpu().numpy()
_, eq_phi = sr.fit(np.c_[gs, inp[:, 1], inp[:, 2]], msg, ["g", "s", "c"], "e04_phi", unary="basic",
                   niterations=150, maxsize=15)
theory_phi = -(inp[:, 2] + 2 * gs * inp[:, 1]) / (gs**2 + 0.25)
print("phi PySR elbow:", eq_phi["elbow"], "\n    best:", eq_phi["best"])
print("corr(phi_net, phi_theory) =", np.corrcoef(msg, theory_phi)[0, 1],
      " slope =", np.polyfit(theory_phi, msg, 1)[0])

# the message is learned with amplitude normalised at u = UC: undo that before comparing
#   (exp((beta-1/2)(u-UC)) with beta = 1/2 is 1, so no rescaling is needed once beta ~ 1/2)
# per-zero amplitude on sin: compare A(g) = d phi/d s vs theory -2g/(g^2+1/4)
gg = np.linspace(gam[0], gam[-1], 200)
with torch.no_grad():
    def ph_(s, c):
        return model.phi(nets.tensor(np.c_[gg / 100, np.full_like(gg, s), np.full_like(gg, c)]))[:, 0].cpu().numpy()
    A = (ph_(1, 0) - ph_(-1, 0)) / 2
    B = (ph_(0, 1) - ph_(0, -1)) / 2
_, eq_A = sr.fit(gg[:, None], A, ["g"], "e04_ampA", unary="basic", niterations=80, maxsize=9)
print("sin-amplitude A(g) PySR:", eq_A["elbow"], "(theory -2g/(g^2+1/4) ~ -2/g)")

# ---- distil b(u) ----------------------------------------------------------------------
uu = np.linspace(2, np.log(XMAX), 300)
with torch.no_grad():
    bu = model.b(nets.tensor(uu[:, None] / 10))[:, 0].cpu().numpy()
_, eq_b = sr.fit(uu[:, None], bu, ["u"], "e04_b", unary="exp", niterations=80, maxsize=9)
print("b(u) PySR:", eq_b["elbow"], " (theory -log(2pi) exp(-u/2) = -1.8379*exp(-0.5u))")

sr.save("e04", {
    "beta_from_0.85": results[0.85][0].beta.item(), "beta_from_0.15": results[0.15][0].beta.item(),
    "r2": {str(k): v[2] for k, v in results.items()}, "r2_truncated_theory": r2_theory, "phi_pysr": eq_phi, "A_pysr": eq_A, "b_pysr": eq_b,
    "phi_corr_theory": np.corrcoef(msg, theory_phi)[0, 1], "A_grid": A, "B_grid": B, "g_grid": gg,
    "beta_traj_085": results[0.85][1][::20], "beta_traj_015": results[0.15][1][::20]})

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
m = (u > 9) & (u < 9.6)
ax[0].plot(u[m], y[m], "k.", ms=1, label="(ψ(x)-x)/√x")
ax[0].plot(u[m], full[m], "r-", lw=.7, label=f"network, {K} zeros")
ax[0].set_xlabel("log x"); ax[0].legend(); ax[0].set_title(f"zeros → primes (R²={r2:.3f})")
for b0, (_, tr, _, _) in results.items():
    ax[1].plot(tr, label=f"start β={b0}")
ax[1].axhline(.5, c="k", ls="--"); ax[1].legend(); ax[1].set_xlabel("step")
ax[1].set_title("real part β learned from prime data")
ax[2].plot(gg, A, label="learned sin amplitude")
ax[2].plot(gg, -2 * gg / (gg**2 + .25), "k--", label="-2γ/(γ²+1/4)")
ax[2].plot(gg, B, label="learned cos amplitude")
ax[2].plot(gg, -1 / (gg**2 + .25), "k:", label="-1/(γ²+1/4)")
ax[2].legend(); ax[2].set_xlabel("γ"); ax[2].set_title("per-zero message")
plt.tight_layout(); plt.savefig(RESULTS / "e04.png", dpi=110)
