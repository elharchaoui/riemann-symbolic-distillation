"""De Bruijn-Newman family H_t(z) = int_0^inf e^{t u^2} Phi(u) cos(z u) du, its zeros and their velocities.

H_0(z) = xi(1/2 + i z/2)/8. Zeros of H_t are real for t >= Lambda; Lambda >= 0 (Rodgers-Tao),
RH <=> Lambda <= 0. dH/dt = -d^2H/dz^2, so a simple zero moves with velocity
dz/dt = -H_t(z)_t / H_t'(z) = H''/H'.
"""
import mpmath as mp

DPS = 70
UMAX = 1.3
NODES = 420


def _phi(u):
    s = mp.mpf(0)
    for n in range(1, 8):
        e = mp.pi * n * n * mp.exp(4 * u)
        s += (2 * mp.pi**2 * n**4 * mp.exp(9 * u) - 3 * mp.pi * n * n * mp.exp(5 * u)) * mp.exp(-e)
    return s


def phi_L(coef, q, nterms=24):
    """Kernel of an odd-type L-function F(s) = sum a_n n^-s with a_n periodic mod q:
    Lambda(1/2 + iz) = int Phi(u) e^{izu} du,  Phi(u) = 2 e^{3u/2} sum_n a_n n exp(-pi n^2 e^{2u}/q).
    Zeros of H_0 are at z = gamma (not 2 gamma as for zeta)."""
    def phi(u):
        x = mp.exp(2 * u)
        return 2 * mp.exp(1.5 * u) * mp.fsum(coef[n % q] * n * mp.exp(-mp.pi * n * n * x / q)
                                             for n in range(1, nterms) if coef[n % q])
    return phi


class Heat:
    def __init__(self, t, phi=_phi, umax=UMAX, nodes=NODES, dps=DPS):
        mp.mp.dps = dps
        self.t = mp.mpf(t)
        # Gauss-Legendre nodes on [0, umax]
        xs, ws = zip(*_gl(nodes, dps))
        self.u = [umax * (x + 1) / 2 for x in xs]
        base = [umax / 2 * w * mp.exp(self.t * u * u) * phi(u) for u, w in zip(self.u, ws)]
        self.w0 = base
        self.w1 = [b * u for b, u in zip(base, self.u)]
        self.w2 = [b * u * u for b, u in zip(base, self.u)]

    def H(self, z):
        return mp.fsum(w * mp.cos(z * u) for w, u in zip(self.w0, self.u))

    def dH(self, z):
        return -mp.fsum(w * mp.sin(z * u) for w, u in zip(self.w1, self.u))

    def d2H(self, z):
        return -mp.fsum(w * mp.cos(z * u) for w, u in zip(self.w2, self.u))

    def zero(self, z0):
        z0 = mp.mpc(z0) if isinstance(z0, complex) or isinstance(z0, mp.mpc) else mp.mpf(z0)
        return mp.findroot(self.H, z0, solver="newton", df=self.dH, tol=mp.mpf(10) ** (-30))

    def velocity(self, z):
        return self.d2H(z) / self.dH(z)


_cache = {}


def _gl(n, dps=DPS):
    if (n, dps) not in _cache:
        mp.mp.dps = dps
        import numpy as np
        x0, _ = np.polynomial.legendre.leggauss(n)
        nodes = []
        for x in x0:  # polish double-precision nodes with Newton in mp
            x = mp.mpf(x)
            for _ in range(6):
                p, dp = mp.legendre(n, x), mp.diff(lambda y: mp.legendre(n, y), x)
                x -= p / dp
            dp = mp.diff(lambda y: mp.legendre(n, y), x)
            nodes.append((x, 2 / ((1 - x * x) * dp * dp)))
        _cache[n, dps] = nodes
    return _cache[n, dps]
