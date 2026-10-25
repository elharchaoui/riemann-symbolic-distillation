"""Symbolic regression (PySR), SymTorch-style distillation, constant refinement."""
import json
import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "16")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

import numpy as np
import sympy as sp
import torch

from . import RESULTS

UNARY = {
    "basic": [],
    "log": ["log"],
    "trig": ["sin", "cos"],
    "exp": ["exp"],
    "full": ["log", "exp", "sin", "cos", "sqrt"],
}


def fit(X, y, names, tag, unary="log", binary=("+", "-", "*", "/"), niterations=150,
        maxsize=25, weights=None, extra_sympy=None, constraints=None, nested=None,
        populations=48, seed=0, **kw):
    """Run PySR; return (model, equations list). Equations are also saved to results/pysr/<tag>."""
    from pysr import PySRRegressor

    ops = UNARY[unary] if isinstance(unary, str) else list(unary)
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=list(binary),
        unary_operators=ops,
        maxsize=maxsize,
        populations=populations,
        constraints=constraints,
        nested_constraints=nested,
        extra_sympy_mappings=extra_sympy or {},
        output_directory=str(RESULTS / "pysr"),
        run_id=tag,
        model_selection="best",
        progress=False,
        verbosity=0,
        random_state=seed,
        deterministic=False,
        parallelism="multithreading",
        **kw,
    )
    model.fit(np.asarray(X, float), np.asarray(y, float), variable_names=list(names), weights=weights)
    return model, table(model)


def table(model, k=12):
    df = model.equations_
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "complexity": int(r["complexity"]),
            "loss": float(r["loss"]),
            "score": float(r["score"]),
            "equation": str(r["sympy_format"]),
        })
    best = str(model.sympy())
    elbow = max(rows, key=lambda r: r["score"])["equation"]
    return {"best": best, "elbow": elbow, "pareto": rows[-k:]}


def pick(eqs, max_complexity):
    """Lowest-loss equation on the Pareto front with complexity <= max_complexity."""
    cands = [r for r in eqs["pareto"] if r["complexity"] <= max_complexity]
    return min(cands, key=lambda r: r["loss"]) if cands else None


def simplify(expr, digits=5):
    e = sp.nsimplify(sp.sympify(expr), tolerance=1e-4, rational=False) if digits is None else sp.sympify(expr)
    return sp.N(e, digits)


def recognize(x, tol=None, maxden=16):
    """Simplest closed form (p/q) * pi^k  or  (p/q + (r/s) c) * pi^k, c in {log 2pi, euler, log 2},
    within tol of x. Returns a string or None."""
    from fractions import Fraction
    import math

    tol = tol or max(1e-9, 1e-5 * abs(x))
    consts = {"log(2*pi)": math.log(2 * math.pi), "euler": 0.5772156649015329, "log(2)": math.log(2)}
    best = None

    def consider(val, expr, cost):
        nonlocal best
        if abs(val - x) <= tol and (best is None or cost < best[0]):
            best = (cost, expr)

    def digits(*fr):
        return sum(len(str(abs(q.numerator))) + (len(str(q.denominator)) if q.denominator > 1 else 0) for q in fr)

    def rat(v):
        f = Fraction(v).limit_denominator(maxden)
        return f, abs(f.numerator) + f.denominator

    for k in (0, 1, -1, 2, -2):
        pk, ps = math.pi**k, ("" if k == 0 else f"*pi**{k}")
        f, c = rat(x / pk)
        consider(float(f) * pk, f"{f}{ps}", digits(f) + abs(k))
        for name, cv in consts.items():
            for r in [Fraction(n, d) for d in (1, 2, 4, 8) for n in range(-8, 9) if n]:
                f, c = rat(x / pk - float(r) * cv)
                e = f"({f} + {r}*{name}){ps}"
                consider((float(f) + float(r) * cv) * pk, e, digits(f, r) + 3 + abs(k))
        names = list(consts)
        small = [Fraction(n, d) for d in (1, 2) for n in (-2, -1, 1, 2)]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for r1 in small:
                    for r2 in small:
                        v = x / pk - float(r1) * consts[names[i]] - float(r2) * consts[names[j]]
                        f, _ = rat(v)
                        e = f"({f} + {r1}*{names[i]} + {r2}*{names[j]}){ps}"
                        consider((float(f) + float(r1) * consts[names[i]] + float(r2) * consts[names[j]]) * pk,
                                 e, digits(f, r1, r2) + 6 + abs(k))
    return best[1] if best else None


# --- SymTorch-style distillation: record what a sub-module sees and emits -------------

class Recorder:
    """Forward hook that stores inputs and outputs of a torch sub-module."""

    def __init__(self, module):
        self.inp, self.out = [], []
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        self.inp.append(args[0].detach().cpu().numpy().reshape(-1, args[0].shape[-1]))
        self.out.append(out.detach().cpu().numpy().reshape(-1, out.shape[-1]))

    def data(self):
        self.h.remove()
        return np.concatenate(self.inp), np.concatenate(self.out)


def distill(module, X, names, tag, max_points=4000, out_index=0, seed=0, **kw):
    """Run module on X, then fit a formula to its input->output map."""
    rec = Recorder(module)
    with torch.no_grad():
        module(torch.as_tensor(X, dtype=torch.float32, device=next(module.parameters()).device))
    xi, yo = rec.data()
    idx = np.random.default_rng(seed).choice(len(xi), min(max_points, len(xi)), replace=False)
    return fit(xi[idx], yo[idx, out_index], names, tag, **kw)


# --- refine constants of a found formula with gradient descent (symtorch) -------------

def refine(expr, X, y, names, steps=300):
    """Re-optimise the numeric constants of expr on (X, y). Returns sympy expression."""
    import symtorch

    model = symtorch.symtorchify(sp.sympify(expr), trainable=True)
    params = list(model.parameters())
    if not params:
        return sp.sympify(expr)
    inp = {n: torch.as_tensor(X[:, i], dtype=torch.float64) for i, n in enumerate(names)}
    tgt = torch.as_tensor(y, dtype=torch.float64)
    model = model.double()
    opt = torch.optim.LBFGS(params, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.mean((model(inp).reshape(-1) - tgt) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return sp.sympify(str(model))


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))
