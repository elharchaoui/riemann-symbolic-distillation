"""Angle 5 - de Bruijn-Newman heat flow: how do zeros interact?

Data: zeros z_j(t) of H_t for t in [-1, 1] (j = 1..KZ) and their exact velocities dz/dt = H''/H'.
Model (Cranmer GNN prior): v_i = sum_{j != i} phi(z_i, z_j) + psi(z_i)
  - neighbours j: all computed zeros and their mirror images -z_j (H_t is even)
  - phi antisymmetric: phi(a, b) = m(a, b) - m(b, a)   (action = reaction)
  - psi: smooth far-field drift from zeros outside the window
PySR distils phi. Theory: phi = 2/(z_i - z_j)  (repulsion as t increases, attraction as t decreases).
Consequence explored: backward-in-t collision times of close pairs (the mechanism behind Lambda >= 0).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rh import DATA, RESULTS, data

KZ = 40          # zeros tracked (z up to ~2*gamma_40 ~ 245)
KT = 20          # targets: inner zeros only (their neighbourhood is complete enough)
TS = np.linspace(-0.3, 1.0, 27)  # zeros 34/35 collide just below -0.3
CACHE = DATA / "heat_flow.npz"


_H = {}


def _init():
    from rh import heat
    for t in TS:
        _H[float(t)] = heat.Heat(float(t))


def track(j):
    """Follow zero j from t=0 outward in both directions (step 0.05, velocity predictor)."""
    import mpmath as mp

    i0 = int(np.argmin(np.abs(TS)))
    h0 = _H[float(TS[i0])]
    z0 = h0.zero(2 * mp.mpf(repr(float(data.zeros(KZ)[j]))))
    zs = {i0: (float(z0), float(h0.velocity(z0)))}
    for direction in (1, -1):
        z, i = z0, i0
        while 0 <= i + direction < len(TS):
            hc, hn = _H[float(TS[i])], _H[float(TS[i + direction])]
            dt = TS[i + direction] - TS[i]
            v1 = hc.velocity(z)
            guess = z + dt * v1
            zn = hn.zero(guess)
            assert abs(zn - guess) < 0.2, f"zero {j} jumped at t={TS[i + direction]}"
            z, i = zn, i + direction
            zs[i] = (float(z), float(hn.velocity(z)))
    return [zs[k] for k in range(len(TS))]


if not CACHE.exists():
    with Pool(14, initializer=_init) as pool:
        res = pool.map(track, range(KZ))
    Z = np.array([[r[k][0] for r in res] for k in range(len(TS))])
    V = np.array([[r[k][1] for r in res] for k in range(len(TS))])
    assert np.all(np.diff(Z, axis=1) > 0), "zero tracking lost order"
    np.savez(CACHE, t=TS, z=Z, v=V)
d = np.load(CACHE)
Z, V = d["z"], d["v"]
print("zeros tracked:", Z.shape, " velocity range:", V.min(), V.max())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rh import nets, sr

# graph: target i in 1..KT; neighbours = all other zeros and all mirrors
Zall = np.concatenate([Z, -Z], 1)  # (nt, 2KZ)
ZI = nets.tensor(Z[:, :KT])       # (nt, KT)
ZJ = nets.tensor(Zall)            # (nt, 2KZ)
mask = torch.ones(KT, 2 * KZ, device=nets.DEV)
for i in range(KT):
    mask[i, i] = 0
VT = nets.tensor(V[:, :KT])


class PairNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nets.MLP(3, 1, width=64, depth=3)
        self.psi = nets.MLP(1, 1, width=16, depth=2)

    def phi(self, a, b):
        f = lambda x, y: self.m(torch.stack([x / 100, y / 100, 10 / (x - y)], -1))[..., 0]
        return f(a, b) - f(b, a)

    def forward(self):
        a = ZI[:, :, None].expand(-1, -1, ZJ.shape[1])
        b = ZJ[:, None, :].expand(-1, KT, -1)
        b = torch.where(mask.bool(), b, a + 1000.0)  # dummy partner on the diagonal (masked anyway)
        pair = self.phi(a, b) * mask
        return pair.sum(-1) + self.psi(ZI[..., None] / 100)[..., 0]


torch.manual_seed(0)
net = PairNet().to(nets.DEV)
hist = nets.train(net, lambda: torch.mean((net() - VT) ** 2), epochs=6000, lr=2e-3)
with torch.no_grad():
    pred = net().cpu().numpy()
r2 = 1 - np.var(V[:, :KT] - pred) / np.var(V[:, :KT])
print(f"velocity fit R^2 = {r2:.5f}")

# distil phi on the pairs that actually occur
rng = np.random.default_rng(5)
ti = rng.integers(0, len(TS), 3000); ii = rng.integers(0, KT, 3000); jj = rng.integers(0, 2 * KZ, 3000)
keep = ii != jj
a = Z[ti, ii][keep]; b = Zall[ti, jj][keep]
with torch.no_grad():
    ph = net.phi(nets.tensor(a), nets.tensor(b)).cpu().numpy()
_, eq = sr.fit(np.c_[a, b], ph, ["zi", "zj"], "e05_phi", unary="basic", niterations=150, maxsize=12)
print("phi PySR elbow:", eq["elbow"], "| best:", eq["best"])
k = np.polyfit(1 / (a - b), ph, 1)
print(f"phi vs 1/(zi-zj): slope {k[0]:.4f} (theory 2) offset {k[1]:.2e}")

# check the law directly on exact data (no network): v_i - 2 sum_{j in window} 1/(z_i - z_j)
direct = np.array([[2 * np.sum(1 / (Z[t, i] - np.delete(Zall[t], i))) for i in range(KT)] for t in range(len(TS))])
resid = V[:, :KT] - direct
print("exact velocity minus 2*sum 1/(zi-zj) over window: mean %.4f std %.4f (far-field drift)" % (resid.mean(), resid.std()))

# ---- a real collision: zeros 34/35 (gamma 111.03, 111.87) meet as t decreases ---------------
def n_sign_changes(t, a, b, n=120):
    from rh import heat
    h = heat.Heat(t)
    zz = np.linspace(a, b, n)
    vals = [h.H(z) for z in zz]
    return sum(1 for p, q in zip(vals, vals[1:]) if p * q < 0)


j1 = 33
I0 = int(np.argmin(np.abs(TS)))
a_, b_ = Z[0, j1] - 1.5, Z[0, j1 + 1] + 1.5  # bracket around the pair at t = -0.3
lo, hi = -0.6, -0.3  # at hi: 2 real zeros in bracket; at lo: 0
assert n_sign_changes(hi, a_, b_) == 2 and n_sign_changes(lo, a_, b_) == 0
for _ in range(14):
    mid = (lo + hi) / 2
    if n_sign_changes(mid, a_, b_) == 2:
        hi = mid
    else:
        lo = mid
t_coll_exact = (lo + hi) / 2
# prediction from the distilled law: integrate dz/dt = 2 sum 1/(zi-zj) + far field (psi) from t=-0.3
with torch.no_grad():
    zc = Z[0].copy()
    tt = TS[0]
    while zc[j1 + 1] - zc[j1] > 1e-3:
        za = np.concatenate([zc, -zc])
        v = np.array([2 * np.sum(1 / (zc[i] - np.delete(za, i))) for i in range(KZ)])
        v += net.psi(nets.tensor(zc[:, None] / 100))[:, 0].cpu().numpy()
        dt = 0.01 * (zc[j1 + 1] - zc[j1]) ** 2
        zc -= dt * v
        tt -= dt
t_coll_law = tt
print(f"zeros 34/35 collision: exact (sign changes of H_t) t = {t_coll_exact:.4f}; "
      f"distilled law predicts t = {t_coll_law:.4f}; pair-only -(dz)^2/8 from t=0: {-(Z[I0, j1 + 1] - Z[I0, j1]) ** 2 / 8:.4f}")

# ---- consequence: backward collision of close pairs at large height (heuristic) --------
coll = {}
for w in ["1e12", "1e21"]:
    off, loc, _ = data.high_zeros(w)
    gaps = np.diff(loc)
    i = int(np.argmin(gaps))
    zloc = 2 * (loc - loc[i])  # z = 2 gamma, centred on the closest pair
    sel = np.arange(max(0, i - 200), min(len(loc), i + 202))
    z = zloc[sel].copy(); pi = np.where(sel == i)[0][0]
    t, dt = 0.0, 0.0
    # integrate dz/dt = 2 sum 1/(z_i - z_j) backwards with adaptive steps until the pair meets
    while z[pi + 1] - z[pi] > 1e-6 * (zloc[i + 1] - zloc[i]):
        D = z[:, None] - z[None, :]
        np.fill_diagonal(D, np.inf)
        v = 2 * np.sum(1 / D, 1)
        gap = z[pi + 1] - z[pi]
        dt = 0.02 * gap**2
        z -= dt * v
        t -= dt
    coll[w] = {"min_gap_gamma": gaps[i], "mean_gap_gamma": gaps.mean(), "collision_t": t,
               "pair_only_estimate": -(2 * gaps[i]) ** 2 / 8}
    print(f"{w}: closest pair gap {gaps[i]:.3e} (mean {gaps.mean():.3f}); backward collision at t = {t:.3e} "
          f"(pair-only estimate {-(2*gaps[i])**2/8:.3e})")

sr.save("e05", {"collision_34_35_exact": t_coll_exact, "collision_34_35_law": t_coll_law, "r2": r2, "pysr": eq, "slope_1_over_d": k[0], "far_field_mean": resid.mean(),
                "far_field_std": resid.std(), "collisions": coll, "t": TS, "z": Z[:, :12]})

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for j in range(12):
    ax[0].plot(TS, Z[:, j], "k-", lw=.8)
ax[0].set_xlabel("t"); ax[0].set_ylabel("zeros of H_t"); ax[0].set_title("zero trajectories")
order = np.argsort(a - b)
ax[1].plot((a - b), ph, ".", ms=2, label="learned φ(zi, zj)")
dd = np.linspace(-400, 400, 2000); dd = dd[np.abs(dd) > 2]
ax[1].plot(dd, 2 / dd, "k-", lw=.8, label="2/(zi-zj)")
ax[1].set_ylim(-1, 1); ax[1].set_xlabel("zi - zj"); ax[1].legend(); ax[1].set_title("distilled pair interaction")
ax[2].plot(V[:, :KT].ravel(), pred.ravel(), ".", ms=2)
ax[2].plot([V.min(), V.max()], [V.min(), V.max()], "k-")
ax[2].set_xlabel("exact dz/dt"); ax[2].set_ylabel("network"); ax[2].set_title(f"velocities R²={r2:.4f}")
plt.tight_layout(); plt.savefig(RESULTS / "e05.png", dpi=110)
