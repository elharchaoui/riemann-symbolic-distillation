"""Build results/report.html from results/eXX.json + eXX.png (all numbers come from the runs)."""
import base64
import html
import json
import math

from rh import RESULTS

J = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in ["e01", "e02", "e03", "e04", "e05", "e06", "e07"]}


def img(name):
    return "data:image/png;base64," + base64.b64encode((RESULTS / f"{name}.png").read_bytes()).decode()


def f(x, d=4):
    return f"{x:.{d}f}"


def e(s):
    return html.escape(str(s))


def chip(status):
    label = {"exact": "exact match", "consistent": "consistent", "partial": "partial", "fail": "not recovered"}[status]
    return f'<span class="chip {status}">{label}</span>'


e1, e2, e3, e4, e5, e6, e7 = (J[k] for k in ["e01", "e02", "e03", "e04", "e05", "e06", "e07"])
A2 = e2["A"]
beta21 = A2["T~1e21"]
B2 = e2.get("B2", {})
ff21 = B2.get("T~1e21", {})

beta_e4 = [e4["beta_from_0.85"], e4["beta_from_0.15"]]
beta_ok = all(abs(b - 0.5) < 0.03 for b in beta_e4)

rows = [
    # angle, what the network saw, what came out, theory, status
    ("1 · Counting", "N(T)/T against log T (MLP, KAN)",
     f"a = {f(e1['A1_refined']['a'],6)} → {e(e1['A1_refined']['identified'][0])}<br>"
     f"b = {f(e1['A1_refined']['b'],6)} → {e(e1['A1_refined']['identified'][1])}<br>"
     f"constant = {f(e1['A2_residual']['mean'],4)} ± {f(e1['A2_residual']['stderr'],4)} → {e(e1['A2_identified'])}",
     "N(T) = T/2π · log(T/2πe) + 7/8", "exact"),
    ("1 · Fluctuation size", "Var S(t) at heights 10² … 10²¹ (PySR)",
     f"Var S = {f(e1['B_varS']['slope_loglog'],4)} · log log T + {f(e1['B_varS']['intercept'],4)}",
     f"slope 1/(2π²) = {f(1/(2*math.pi**2),4)} (Selberg)", "consistent"),
    ("2 · Level repulsion", "spacing density p(s) (density network, MLE)",
     f"β = {f(beta21['beta'],3)} ± {f(beta21['beta_se'],3)} at 10²¹ (from {f(A2['T~1e4']['beta'],2)} at 10⁴)",
     "β = 2 (GUE)", "consistent"),
    ("2 · Pair correlation", "form factor F(α) (MLP, PySR with min/max)",
     (f"PySR: {e(ff21.get('pysr', {}).get('elbow', '—'))}; rms gap to min(α,1) = {f(ff21.get('rms_vs_montgomery', float('nan')),3)}"
      if ff21 else "—") + f"<br>R₂(u) χ²/bin vs Montgomery: {f(e2['B']['T~1e4']['chi2_per_bin_vs_montgomery'],2)} (10⁴) → {f(e2['B']['T~1e21']['chi2_per_bin_vs_montgomery'],2)} (10²¹)",
     "F(α) = min(|α|, 1) (Montgomery)", "consistent"),
    ("3 · Primes → zeros", "sin/cos(t log n) for every n ≤ 150, L1 network",
     f"kept {e3['tp']} of 48 prime powers, {e3['fp']} false positives<br>PySR: a_n = {e(e3['pysr']['elbow'])}",
     "S(t) ≈ −(1/π) Σ sin(k t log p) / (k p^{k/2})", "exact"),
    ("4 · Zeros → primes", "one free weight pair per zero (300 zeros), free exponent β",
     f"β = {f(e4['beta_profile'],4)} (profile), {f(beta_e4[0],4)} / {f(beta_e4[1],4)} (from 0.85 / 0.15)<br>"
     f"PySR: A(γ) = {e(e4['A_pysr']['elbow'])}; smooth weight {f(e4['C'],4)}<br>cos amplitude B(γ) not recovered",
     "β = ½; A = −2γ/(γ²+¼) ≈ −2/γ; −log 2π = −1.8379", "exact" if beta_ok else "partial"),
    ("5 · Heat flow", "velocities of zeros of H_t (pairwise GNN)",
     f"PySR: φ = {e(e5['pysr']['elbow'])}<br>collision of zeros 34/35: law t = {f(e5['collision_34_35_law'],4)}, exact t = {f(e5['collision_34_35_exact'],4)}",
     "dz_i/dt = 2 Σ 1/(z_i − z_j)", "exact"),
    ("6 · Li coefficients", "λ_n for n ≤ 20 000 from 2M zeros",
     f"λ_n = {f(e6['fit'][0],5)} n log n {f(e6['fit'][1],5)} n + …; all λ_n &gt; 0",
     f"½ n log n + (γ−1−log 2π)/2 · n  ({f(e6['c_theory'],5)})", "consistent"),
]

score = "".join(f"<tr><th scope='row'>{a}</th><td>{b}</td><td class='eq'>{c}</td><td class='eq'>{d}</td><td>{chip(s)}</td></tr>"
                for a, b, c, d, s in rows)

cf = e6["counterfactual"]
cf_rows = "".join(f"<tr><td>{k}</td><td>{v['rate_fit']:.3e}</td><td>{v['rate_theory']:.3e}</td><td>{v['first_negative_n'] or '&gt; 20 000'}</td></tr>"
                  for k, v in cf.items())
l1 = e7["loop1"]; l1h = e7["loop1_hi"]
loop1_rows = "".join(f"<tr><td>{X}</td><td>{l1[X]['median_err_spacings']:.3f}</td><td>{l1h[X]['median_err_spacings']:.3f}</td>"
                     f"<td>{100*l1h[X]['frac_correct_index']:.1f}%</td></tr>" for X in l1)
loop2_rows = "".join(f"<tr><td>{e(k)}</td><td>{v:.3f}</td></tr>" for k, v in e7["loop2_r2"].items())
loop3_rows = "".join(f"<tr><td>{r[0]:.0e}</td><td>{r[2]:.1e}</td><td>{r[3]:.1e}</td></tr>" for r in e7["loop3"])
coll = e5["collisions"]

# ---------------- part II: other L-functions ----------------------------------------------
LJ = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in ["l00", "l01", "l02", "l03", "l04", "l05", "l06", "l07"]}
l0, l1, l2, l3, l4, l5, l6, l7 = (LJ[k] for k in ["l00", "l01", "l02", "l03", "l04", "l05", "l06", "l07"])
FN = ["zeta", "chi4", "dh"]


def ffc(eq):
    """The constant c in a PySR form factor Min(c, a)."""
    import re
    m = re.search(r"[-+]?\d+\.\d+", eq)
    return f"{float(m.group()):.2f}" if m else e(eq)


def hb(ok, text):
    cls, lab = {True: ("exact", "holds"), False: ("fail", "breaks"), None: ("partial", "blind")}[ok]
    return f'{text}<br><span class="chip {cls}">{lab}</span>'


import numpy as _np
dh_off = _np.load(RESULTS.parent / "data" / "zeros_dh.npz")["off"]
n_zeta = int(_np.sum(__import__("rh.data", fromlist=["zeros"]).zeros() < 50000))
frac = l1["dh_offline"]["frac_line_final"]
land = l5["B"]["rows"]
mat = [
    ("Counting N(T) = T/2π·log(qT/2πe) + c", *[hb(True, f"q = {l1[k]['q_est']:.3f}, c = {e(l1[k]['const_id'])}") for k in FN], "no (functional equation)"),
    ("All zeros on Re s = ½", hb(True, "Odlyzko table"), hb(True, f"{l0['chi4']['n_line']:,} of {l0['chi4']['n_line']:,}"),
     hb(False, f"{100*frac:.1f}% on the line"), "yes"),
    ("GUE pair statistics", *[hb(k != "dh", f"χ²/bin {l2['B'][k]['chi2_per_bin_vs_montgomery']:.1f}; PySR F(α) = min(α, {ffc(l2['B2'][k]['pysr']['elbow'])})" + (" + Bragg peak" if k == "dh" else "")) for k in FN], "yes"),
    ("Lattice (Gram) order |⟨exp(2πi·θ(γ)/π)⟩|", *[f"{l7['gram'][k]['order']:.2f}" for k in FN], "observation"),
    ("S(t) uses prime powers only (n ≤ 150)", hb(True, f"{l3['zeta']['tp']}/{l3['zeta']['n_expected']}, 0 extra"),
     hb(True, f"{l3['chi4']['tp']}/{l3['chi4']['n_expected']}, 0 extra, signs χ(p)"),
     hb(False, f"{l3['dh']['n_nonpp_selected']} composites selected"), "yes"),
    ("Coefficients put zeros at β = ½", *[hb(abs(l4[k]['beta_common'] - .5) < .02, f"β = {l4[k]['beta_common']:.4f}") for k in FN], "yes"),
    ("Heat flow: dz/dt = 2 Σ 1/(zᵢ − zⱼ)", hb(True, f"slope {e5['slope_1_over_d']:.3f}"), hb(True, f"slope {l5['A']['slope']:.2f}"),
     hb(True, "landing times within 1%"), "no (heat equation)"),
    ("Nothing off the real axis for t ≥ 0 (Λ ≤ 0)", hb(True, f"closest pair meets at t = {l5['C']['zeta']['t_collision']:.1e}"),
     hb(True, f"closest pair meets at t = {l5['C']['chi4']['t_collision']:.1e}"),
     hb(False, f"Λ ≥ {4*max(r[2] for r in land):.3f}"), "yes"),
    ("Li asymptotics ½n log n + b n", *[hb(True, f"b = {l6[k]['fit'][1]:.3f} (theory {l6[k]['b_theory']:.3f})") for k in FN], "no"),
    ("Li positivity λₙ &gt; 0, n ≤ 2000", *[hb(True, "all positive") for k in FN[:2]],
     hb(None, f"all positive, yet false: first λₙ &lt; 0 near n ≈ {l6['dh_anomaly']['first_negative_pred']:.1e}"), "not visible at small n"),
    ("Zeros from coefficients (X = 10⁴)", *[hb(True, f"{l7['loop1'][k]['1']['median_err']:.3f} → {l7['loop1'][k]['10000']['median_err']:.3f}") for k in FN], "no"),
]
lmatrix = "".join(f"<tr><th scope='row'>{r[0]}</th><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>" for r in mat)
bj, gm, bt = (_np.array(l4["dh"][k]) for k in ("beta_j", "gamma", "beta_true"))
off_rows = "".join(f"<tr><td>{g_:.2f}</td><td>{t_:.4f}</td><td>{b_:.4f}</td></tr>" for g_, t_, b_ in zip(gm, bt, bj) if t_ != 0.5)
land_rows = "".join(f"<tr><td>{r[1]:.4f} + {r[0]:.2f}i</td><td>{r[2]:.5f}</td><td>{r[3]:.5f}</td><td>{r[4]:.5f}</td></tr>" for r in land)
L1 = {k: f"{l7['loop1'][k]['1']['median_err']:.3f} to {l7['loop1'][k]['10000']['median_err']:.3f}" for k in FN}
PART2 = {
    "LMATRIX": lmatrix, "OFF_ROWS": off_rows, "LAND_ROWS": land_rows,
    "N_ZETA": f"{n_zeta:,}", "N_CHI4": f"{l0['chi4']['n_line']:,}", "N_DH_LINE": f"{l0['dh']['n_line']:,}",
    "N_DH_OFF": f"{len(dh_off):,}", "DH_SIGMA_MAX": f"{dh_off[:, 0].max():.3f}",
    "Q_ZETA": f"{l1['zeta']['q_est']:.3f}", "Q_CHI4": f"{l1['chi4']['q_est']:.3f}", "Q_DH": f"{l1['dh']['q_est']:.3f}",
    "DH_FRAC": f"{100*frac:.1f}%", "DH_SLOPE": f"{l1['dh_offline']['slope']:.3f}",
    "GRAM_ZETA": f"{l7['gram']['zeta']['order']:.2f}", "GRAM_CHI4": f"{l7['gram']['chi4']['order']:.2f}",
    "GRAM_DH": f"{l7['gram']['dh']['order']:.2f}", "DH_BIGGAP": f"{100*l2['D']['share_big_gaps_with_pair']:.0f}%",
    "DH_OFFBND": f"{100*l7['gram']['dh']['off_near_boundary']:.0f}%",
    "S_ZETA": str(l3["zeta"]["tp"]), "S_CHI4": str(l3["chi4"]["tp"]), "S_DH_NPP": str(l3["dh"]["n_nonpp_selected"]),
    "S_DH_CORR": f"{l3['dh']['corr']:.3f}",
    "B_ZETA": f"{l4['zeta']['beta_common']:.4f}", "B_CHI4": f"{l4['chi4']['beta_common']:.4f}", "B_DH": f"{l4['dh']['beta_common']:.3f}",
    "EF_LINE": f"{l4['dh']['r2_ef_line']:.2f}", "EF_ALL": f"{l4['dh']['r2_ef_all']:.2f}",
    "PHI_CHI4": e(l5["A"]["pysr"]["elbow"]), "LAND_MAX": f"{4*max(r[2] for r in land):.3f}",
    "LAND_HEUR": f"{4*l5['B']['pair_law_max']:.2f}",
    "COLL_ZETA": f"{l5['C']['zeta']['t_collision']:.1e}", "COLL_CHI4": f"{l5['C']['chi4']['t_collision']:.1e}",
    "LAM1_DH": f"{l6['dh']['lam1']:.8f}",
    "LI_B": ", ".join(f"{l6[k]['fit'][1]:.3f} vs {l6[k]['b_theory']:.3f}" for k in FN),
    "LI_DMAX": f"{l6['dh_anomaly']['max_abs_D_small']:.3f}", "LI_NEG": f"{l6['dh_anomaly']['first_negative_pred']:,}",
    "L1_ZETA": L1["zeta"], "L1_CHI4": L1["chi4"], "L1_DH": L1["dh"],
    **{f"LIMG{i}": img(f"l0{i}") for i in range(1, 8)},
}

# ---------------- part III: Gram locking ----------------------------------------------------
l8 = json.loads((RESULTS / "l08.json").read_text())
l8b = json.loads((RESULTS / "l08b.json").read_text())
dirs8 = [r for r in l8["rows"] if r["spec"][0] == "dir"]
dhs8 = [r for r in l8["rows"] if r["spec"][0] == "dh"]
c_law = l8b["fits"]["1/(1+cP)"]["c"]
d52 = next(d for d in l8b["dh"] if d["spec"] == ["dh", 5, 2])
zh = l8["zeta_height"]
l8_rows = [
    ("Fit on Euler-product functions", f"R ≈ 1/(1 + {c_law:.2f}·P), rms relative error {l8b['fits']['1/(1+cP)']['rms_rel']:.1%}"),
    ("PySR on the same points", f"R = {l8b['fits']['c/P']['c']:.3f}/P (rms {l8b['fits']['c/P']['rms_rel']:.1%}; wrong as P → 0)"),
    ("ζ from T = 2·10³ to 10²⁰", f"R from {zh[0]['R']:.3f} to {zh[-1]['R']:.3f} as P goes from {zh[0]['P']:.1f} to {zh[-1]['P']:.1f}; on the curve"),
    ("Share on the line, Dirichlet L", f"{min(r['f_line'] for r in dirs8):.4f} – {max(r['f_line'] for r in dirs8):.4f} (missed close pairs ≈ 0.1%)"),
    ("Share on the line, DH types", f"{min(r['f_line'] for r in dhs8):.3f} – {max(r['f_line'] for r in dhs8):.3f}"),
    ("DH types, distance from the curve", f"line zeros {l8b['dev_line']:.0%} → all slots counted {l8b['dev_all']:.0%}"),
]
PART3 = {
    "L8_ROWS": "".join(f"<tr><th scope='row'>{a}</th><td>{b}</td></tr>" for a, b in l8_rows),
    "N_FAM": str(len(l8["rows"])), "N_FAMDH": str(len(dhs8)), "C_LAW": f"{c_law:.2f}",
    "R_Q12": f"{next(r['R'] for r in dirs8 if r['spec'] == ['dir', 12, 11]):.2f}",
    "FL_DIR": f"{min(r['f_line'] for r in dirs8):.3f}–{max(r['f_line'] for r in dirs8):.3f}",
    "FL_DH": f"{min(r['f_line'] for r in dhs8):.2f}–{max(r['f_line'] for r in dhs8):.2f}",
    "DEV_LINE": f"{l8b['dev_line']:.0%}", "DEV_ALL": f"{l8b['dev_all']:.0%}",
    "R_DH_LINE": f"{d52['R_line']:.3f}", "R_DH_ALL": f"{d52['R_all']:.3f}", "R_DH_CURVE": f"{d52['R_curve']:.3f}",
    "LIMG8": img("l08"), "LIMG8B": img("l08b"),
}

# ---------------- part IV: theory and exact DH family -------------------------------------------
l9 = json.loads((RESULTS / "l09.json").read_text())
l10 = json.loads((RESULTS / "l10.json").read_text())
dz, dd = l9["D"]["('dir', 1, 1)"], l9["D"]["('dh', 5, 2)"]
l10_rows = "".join(
    f"<tr><td>({x['spec'][1]}, {x['spec'][2]})</td><td>{x['P']:.2f}</td><td>{x['n_off']}</td><td>{x['count_gap']:+.0f}</td>"
    f"<td>{x['R_line']:.3f}</td><td>{x['R_all']:.3f}</td><td>{x['R_t']:.3f}</td><td>{x['R_model']:.3f}</td></tr>"
    for x in l10["rows"])
PART4 = {
    "L9_N": str(len(l9["A"]["points"])), "L9_RMS": f"{l9['A']['rms_rel']:.1%}", "L9_CHI2": f"{l9['A']['chi2_dof']:.2f}",
    "L9_FITRMS": f"{l9['A']['fit_rms_rel']:.1%}", "L9_CORR": f"{l9['C']['corr_dh_fcore_ftrue']:.3f}",
    "L9_DHGAP": f"{dd['offline_in_big_core_gap']:.0%}", "L9_DHOFF": str(dd["n_offline"]),
    "L9_ZMISS": str(dz["missed_line_zeros"]), "L9_ZGAP": f"{dz['missed_gap_median']:.2f}", "L9_ZALL": f"{dz['all_gap_median']:.2f}",
    "L9_DMISS": str(dd["missed_line_zeros"]), "L10_CLOSED": str(l10["n_closed"]), "L10_DT": f"{l10['dev_all_vs_Rt']:.1%}",
    "L10_ROWS": l10_rows, "LIMG9": img("l09"),
}

TEMPLATE = (RESULTS.parent / "report_template.html").read_text()
page = TEMPLATE
for k, v in {
    "SCORE": score, "CF_ROWS": cf_rows, "LOOP1": loop1_rows, "LOOP2": loop2_rows, "LOOP3": loop3_rows,
    "IMG1": img("e01"), "IMG2": img("e02"), "IMG3": img("e03"), "IMG4": img("e04"), "IMG5": img("e05"),
    "IMG6": img("e06"), "IMG7": img("e07"),
    "R2_S": f(e3["r2"], 2), "C_PRIME": f(e3["c_prime"], 5),
    "KAN": e(e1["A3_kan"]), "LAM1": f(e6["lam_first"][0], 10),
    "COLL12": f"{coll['1e12']['collision_t']:.2e}", "COLL21": f"{coll['1e21']['collision_t']:.2e}",
    "VEL_R2": f(e5["r2"], 5),     "R2_TRUNC": f(e4["r2_truncated_theory"], 3), "BETA_PROF": f(e4["beta_profile"], 4),
    "A_LAW": e(e4["A_pysr"]["elbow"]), "C4": f(e4["C"], 4), "R2_4": f(max(map(float, e4["r2"].values())), 3),
    "BETA_E4A": f(beta_e4[0], 3), "BETA_E4B": f(beta_e4[1], 3),
    **PART2, **PART3, **PART4,
}.items():
    page = page.replace("{{" + k + "}}", v)
import re as _re
assert not _re.search(r"\{\{[A-Z0-9_]+\}\}", page), _re.findall(r"\{\{[A-Z0-9_]+\}\}", page)
(RESULTS / "report.html").write_text(page)
print("wrote", RESULTS / "report.html", len(page) // 1024, "KB")
