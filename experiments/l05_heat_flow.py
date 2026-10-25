"""L-angle 5 - de Bruijn-Newman heat flow for L(s, chi_-4) and for DH.

H_t(z) = int e^{t u^2} Phi_F(u) cos(z u) du with the kernel of F (rh/heat.phi_L); zeros of H_0 at z = gamma
(an off-line zero sigma + i gamma sits at z = gamma - i (sigma - 1/2)).
A. chi4: track 40 zeros for t in [0, 1], exact velocities H''/H', PairNet + PySR (as e05). Is the
   interaction law the same 2/(z_i - z_j)? (It should be: it comes from the heat equation, not from arithmetic.)
B. DH: follow each low off-line zero forward in t until it reaches the real axis (it meets its mirror image).
   Exact landing time t* vs the distilled law (pair only: t* = (sigma - 1/2)^2 / 2; with neighbours: ODE).
   Lambda_DH >= max t* > 0, while for zeta and chi4 nothing in the data is off the real axis at t = 0.
   Times: t (z = gamma) here; x4 gives the zeta convention (z = 2 gamma) in which Lambda_zeta is quoted.
C. Closest pairs: backward collision time from the distilled law (zeta vs chi4 at matched heights).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rh import DATA, RESULTS, lfun

KZ, KT = 40, 20
TS = np.linspace(0.0, 1.0, 21)  # zeros 30/31 (gap 0.73) meet near t = -0.07
CACHE = DATA / "heat_flow_chi4.npz"
_H = {}


def _kernel(name):
    import mpmath as mp
    from rh import heat
    if name == "chi4":
        return heat.phi_L([0, 1, 0, -1], 4, 30), 80, 420
    mp.mp.dps = 120
    k = (mp.sqrt(10 - 2 * mp.sqrt(5)) - 2) / (mp.sqrt(5) - 1)
    return heat.phi_L([0, 1, k, -k, -1], 5, 30), 120, 640


def heat_at(name, t):
    from rh import heat
    key = (name, round(float(t), 12))
    if key not in _H:
        phi, dps, nodes = _kernel(name)
        _H[key] = heat.Heat(float(t), phi=phi, umax=3.0, nodes=nodes, dps=dps)
    return _H[key]


def track_chi4(j):
    import mpmath as mp
    g0 = lfun.load("chi4")["g"][j]
    i0 = int(np.argmin(np.abs(TS)))
    h0 = heat_at("chi4", TS[i0])
    z0 = h0.zero(mp.mpf(repr(float(g0))))
    zs = {i0: (float(z0), float(h0.velocity(z0)))}
    for direction in (1, -1):
        z, i = z0, i0
        while 0 <= i + direction < len(TS):
            hc, hn = heat_at("chi4", TS[i]), heat_at("chi4", TS[i + direction])
            guess = z + (TS[i + direction] - TS[i]) * hc.velocity(z)
            zn = hn.zero(guess)
            assert abs(zn - guess) < 0.2, f"zero {j} jumped at t={TS[i + direction]}"
            z, i = zn, i + direction
            zs[i] = (float(z), float(hn.velocity(z)))
    return [zs[k] for k in range(len(TS))]


def land_dh(sz):
    """Follow the off-line zero z = gamma - i(sigma - 1/2) forward in t until Im z -> 0."""
    import mpmath as mp
    sigma, gamma = sz
    t = 0.0
    h = heat_at("dh", t)
    z = h.zero(complex(gamma, -(sigma - 0.5)))
    path = [(t, float(z.real), float(z.imag))]
    while abs(z.imag) > 2e-3:
        v = h.velocity(z)
        dt = min(0.004, 0.15 * float(z.imag) ** 2)
        tn = t + dt
        hn = heat_at("dh", tn)
        zn = hn.zero(z + dt * v)
        if abs(zn.imag) < 1e-12 or abs(zn - (z + dt * v)) > 0.05:  # stepped past the landing: halve
            _H.pop(("dh", round(float(tn), 12)), None)
            dt /= 2
            if dt < 1e-9:
                break
            hn = heat_at("dh", t + dt)
            zn = hn.zero(z + dt * hn.velocity(z))
            tn = t + dt
        t, z, h = tn, zn, hn
        path.append((t, float(z.real), float(z.imag)))
    # y^2 ~ 2 (t* - t) near landing
    return {"sigma": sigma, "gamma": gamma, "t_star": t + float(z.imag) ** 2 / 2, "path": path}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from rh import nets, sr

    out = {}
    # ------------------------- A: chi4 interaction law ---------------------------------------
    if not CACHE.exists():
        with Pool(14) as pool:
            res = pool.map(track_chi4, range(KZ))
        Z = np.array([[r[k][0] for r in res] for k in range(len(TS))])
        V = np.array([[r[k][1] for r in res] for k in range(len(TS))])
        assert np.all(np.diff(Z, axis=1) > 0), "zero tracking lost order"
        np.savez(CACHE, t=TS, z=Z, v=V)
    d = np.load(CACHE)
    Z, V = d["z"], d["v"]
    Zall = np.concatenate([Z, -Z], 1)
    ZI, ZJ, VT = nets.tensor(Z[:, :KT]), nets.tensor(Zall), nets.tensor(V[:, :KT])
    mask = torch.ones(KT, 2 * KZ, device=nets.DEV)
    for i in range(KT):
        mask[i, i] = 0

    class PairNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = nets.MLP(3, 1, width=64, depth=3)
            self.psi = nets.MLP(1, 1, width=16, depth=2)

        def phi(self, a, b):
            f = lambda p, q: self.m(torch.stack([p / 50, q / 50, 5 / (p - q)], -1))[..., 0]
            return f(a, b) - f(b, a)

        def forward(self):
            a = ZI[:, :, None].expand(-1, -1, ZJ.shape[1])
            b = ZJ[:, None, :].expand(-1, KT, -1)
            b = torch.where(mask.bool(), b, a + 1000.0)
            return (self.phi(a, b) * mask).sum(-1) + self.psi(ZI[..., None] / 50)[..., 0]

    torch.manual_seed(0)
    net = PairNet().to(nets.DEV)
    nets.train(net, lambda: torch.mean((net() - VT) ** 2), epochs=6000, lr=2e-3)
    with torch.no_grad():
        pred = net().cpu().numpy()
    r2 = 1 - np.var(V[:, :KT] - pred) / np.var(V[:, :KT])
    rng = np.random.default_rng(15)
    ti, ii, jj = rng.integers(0, len(TS), 3000), rng.integers(0, KT, 3000), rng.integers(0, 2 * KZ, 3000)
    keep = ii != jj
    a, b = Z[ti, ii][keep], Zall[ti, jj][keep]
    with torch.no_grad():
        ph = net.phi(nets.tensor(a), nets.tensor(b)).cpu().numpy()
    _, eq = sr.fit(np.c_[a, b], ph, ["zi", "zj"], "l05_phi_chi4", unary="basic", niterations=120, maxsize=12)
    k = np.polyfit(1 / (a - b), ph, 1)
    print(f"A chi4: velocity R^2 {r2:.5f}; PySR phi elbow: {eq['elbow']}; slope vs 1/(zi-zj) {k[0]:.4f}", flush=True)
    out["A"] = {"r2": r2, "pysr": eq, "slope": k[0]}

    # ------------------------- B: DH off-line zeros land on the real axis -----------------------
    dh = lfun.load("dh")
    low = [tuple(r) for r in dh["off"] if r[1] < 300]
    with Pool(len(low)) as pool:
        lands = pool.map(land_dh, low)
    g_all = np.r_[dh["g"], dh["off"][:, 1]]

    def law_landing(sigma, gamma, W=150):
        """Integrate dz/dt = 2 sum 1/(z_i - z_j) (complex zeros, mirrors -z) until the pair is real."""
        line = dh["g"][np.abs(dh["g"] - gamma) < W]
        offz = dh["off"][(np.abs(dh["off"][:, 1] - gamma) < W) & (np.abs(dh["off"][:, 1] - gamma) > 1e-6)]
        z = np.r_[line, offz[:, 1] - 1j * (offz[:, 0] - 0.5), offz[:, 1] + 1j * (offz[:, 0] - 0.5),
                  gamma - 1j * (sigma - 0.5), gamma + 1j * (sigma - 0.5)].astype(complex)
        z = np.r_[z, -z]
        n0 = len(z) // 2
        p = n0 - 2  # index of our zero (gamma - i y)
        t = 0.0
        while abs(z[p].imag) > 1e-4:
            D = z[:, None] - z[None, :]
            np.fill_diagonal(D, np.inf)
            v = 2 * np.sum(1 / D, 1)
            dt = 0.02 * abs(z[p].imag) ** 2
            z = z + dt * v
            t += dt
        return t + abs(z[p].imag) ** 2 / 2

    rows = []
    for L_ in lands:
        lawn = law_landing(L_["sigma"], L_["gamma"])
        pair = (L_["sigma"] - 0.5) ** 2 / 2
        rows.append((L_["gamma"], L_["sigma"], L_["t_star"], lawn, pair))
        print(f"B DH zero {L_['sigma']:.4f} + {L_['gamma']:.3f}i: exact landing t* = {L_['t_star']:.5f}; "
              f"law with neighbours {lawn:.5f}; pair only (sigma-1/2)^2/2 = {pair:.5f}"
              f"  [zeta convention x4: {4*L_['t_star']:.4f}]", flush=True)
    est_all = (dh["off"][:, 0] - 0.5) ** 2 / 2
    print(f"B DH: pair-law landing times for all {len(dh['off'])} off-line zeros: max {est_all.max():.4f}"
          f" (zero at {dh['off'][est_all.argmax()]}) -> heuristic Lambda_DH >= {est_all.max():.3f}"
          f" (zeta convention {4*est_all.max():.3f})", flush=True)
    out["B"] = {"rows": rows, "pair_law_max": est_all.max(), "pair_law_argmax": dh["off"][est_all.argmax()],
                "paths": {f"{L_['gamma']:.2f}": L_["path"] for L_ in lands}}

    # ------------------------- C: closest pairs, backward collision ------------------------------
    fam = lfun.family()
    out["C"] = {}
    for name in ("zeta", "chi4"):
        g = fam[name]["g"]
        scale = 2.0  # zeta convention z = 2 gamma for both, so the times are comparable
        gaps = np.diff(g)
        i = int(np.argmin(gaps[1000:])) + 1000
        sel = np.arange(i - 200, min(len(g), i + 202))
        z = scale * (g[sel] - g[i])
        p = int(np.where(sel == i)[0][0])
        t = 0.0
        while z[p + 1] - z[p] > 1e-6 * scale * gaps[i]:
            D = z[:, None] - z[None, :]
            np.fill_diagonal(D, np.inf)
            dt = 0.02 * (z[p + 1] - z[p]) ** 2
            z = z - dt * 2 * np.sum(1 / D, 1)
            t -= dt
        print(f"C {name}: closest pair (height {g[i]:.1f}) gap {gaps[i]:.4e}, mean gap there "
              f"{2*np.pi/np.log(fam[name]['q']*g[i]/(2*np.pi)):.3f}; backward collision t = {t:.3e}", flush=True)
        out["C"][name] = {"height": g[i], "gap": gaps[i], "t_collision": t}

    sr.save("l05", out)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for j in range(12):
        ax[0].plot(TS, Z[:, j], "k-", lw=.8)
    ax[0].set_xlabel("t"); ax[0].set_title(f"χ₋₄: zeros of H_t (velocity R²={r2:.4f})")
    ax[1].plot(a - b, ph, ".", ms=2, label="learned φ (χ₋₄)")
    dd = np.linspace(-200, 200, 2000); dd = dd[np.abs(dd) > 1]
    ax[1].plot(dd, 2 / dd, "k-", lw=.8, label="2/(zi−zj)"); ax[1].set_ylim(-1.5, 1.5); ax[1].legend()
    ax[1].set_title("pair law, χ₋₄")
    for L_ in lands:
        P = np.array(L_["path"])
        ax[2].plot(P[:, 0], -P[:, 2], label=f"σ={L_['sigma']:.3f}, γ={L_['gamma']:.1f}")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("distance from real axis"); ax[2].legend(fontsize=7)
    ax[2].set_title("DH: off-line zeros under the heat flow")
    plt.tight_layout(); plt.savefig(RESULTS / "l05.png", dpi=110)
