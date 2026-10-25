# Riemann zeta zeros: symbolic distillation

Neural networks plus symbolic regression (PySR, KAN) applied to the zeros of the Riemann zeta function and two
other L-functions, to see which laws a pipeline can recover from data and which need the Euler product.
No proof of anything is claimed; most results are rediscoveries used as validation. The report says which is which.

- **Part I** (`experiments/e01`–`e07`): seven angles on ζ (counting, local statistics, primes ↔ zeros,
  de Bruijn–Newman heat flow, Li coefficients, synthesis).
- **Part II** (`l00`–`l07`): the same angles on L(s, χ₋₄) and the Davenport–Heilbronn function.
- **Part III–IV** (`l08`–`l11`): Gram locking across 34 L-functions. A random-phase law R = 1 − P(1 − e^(−1/P))
  fits, but the literature check (`l11`) shows the Euler-product case is a known consequence of the explicit
  formula (Arias de Reyna, arXiv:1403.7126): R ≈ (q/φ(q))/log(qT/2π). The Davenport–Heilbronn-type statistics
  (vacancies, lattice order) are the part without a known explanation.

The full write-up is `results/report.html` (built by `build_report.py`).

## Setup

```bash
uv sync
```

Zeta zero tables are A. Odlyzko's (https://www-users.cse.umn.edu/~odlyzko/zeta_tables/). Put `zeros6.gz`,
`zeros3` and `zeros4` in `data/raw/`. The zeros of L(s, χ₋₄) and of the Davenport–Heilbronn function are
included in `data/` (`l00_zeros.py` recomputes them, ~1.5 h on 20 cores).

## Run

```bash
uv run python experiments/e01_counting.py     # … each experiment writes results/<name>.json|png
uv run python build_report.py                 # rebuilds results/report.html
```

Layout: `rh/` holds the shared code (zero data, PySR wrapper and distillation, networks, heat-flow kernel,
L-function evaluation with Arb via python-flint).
