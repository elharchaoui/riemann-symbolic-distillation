"""Blog figure: Li coefficients of the Davenport-Heilbronn function stay positive long after an off-line zero.

lambda_n for n <= 2000 is computed from all zeros (l06). Beyond that: the smooth law fitted in l06 (the line
zeros' contribution) plus the exact contribution of the 60 off-line zeros with the largest growth rate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh import RESULTS, lfun

l6 = json.loads((RESULTS / "l06.json").read_text())
a, b, c = l6["dh"]["fit"]
off = lfun.load("dh")["off"]
rr = off[:, 0] + 1j * off[:, 1]
rp = (1 - off[:, 0]) + 1j * off[:, 1]
top = np.argsort(np.log(np.abs(1 - 1 / rp)))[::-1][:60]


def li_pair_sum(rhos, n):
    w = np.log1p(-1 / rhos)
    return 2 * np.sum(1 - np.exp(n[:, None] * w[None, :]).real, 1)


n = np.unique(np.round(np.exp(np.linspace(np.log(2000), np.log(1.2e6), 60000)))).astype(float)
D = np.zeros(len(n))
for i0 in range(0, len(n), 5000):
    nn = n[i0:i0 + 5000]
    D[i0:i0 + 5000] = (li_pair_sum(rr[top], nn) + li_pair_sum(rp[top], nn)
                       - 2 * li_pair_sum(0.5 + 1j * off[top, 1], nn))
lam = a * n * np.log(n) + b * n + c + D
n_small = np.arange(1, 2001)[::5]
lam_small = np.array(l6["dh"]["lam_sub"])
first_neg = n[np.argmax(lam < 0)]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axvspan(1, 2000, color="#2c55c9", alpha=.08)
ax.plot(n_small, lam_small, color="#2c55c9", lw=1.6, label="computed from all zeros: every value positive")
pos = np.where(lam > 0, lam, np.nan)
neg = np.where(lam <= 0, lam, np.nan)
ax.plot(n, pos, color="#2c55c9", lw=.6, alpha=.8, label="predicted (smooth law + exact off-line zeros)")
ax.plot(n, neg, color="#c8412b", lw=.6, label="predicted negative values")
ax.axvline(first_neg, color="#c8412b", ls=":", lw=1)
ax.annotate(f"first negative value\nn ≈ {round(first_neg, -4):,.0f} (predicted)", (first_neg, 1e5), xytext=(1.2e4, 3e5), fontsize=9,
            color="#c8412b", arrowprops=dict(arrowstyle="->", color="#c8412b", lw=.8))
ax.text(40, -3e3, "range I checked\n(n ≤ 2,000)", fontsize=9, color="#2c55c9")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=100)
ax.set_xlim(1, 1.2e6)
ax.set_ylim(-1e8, 1e8)
ax.axhline(0, color="gray", lw=.6)
ax.set_xlabel("n (position in Li's list)")
ax.set_ylabel("Li number λₙ")
ax.set_title("Fake zeta (Davenport–Heilbronn): an off-line zero at height 86 stays hidden for 300,000 terms",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(RESULTS / "blog_li_dh.png", dpi=150)
print("first negative predicted at", first_neg)
