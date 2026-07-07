Companion code for Klein & Hoffart (ICML FMSD 2026). The pilot answers one sharp
empirical question:

> Can a model restricted to column-value statistics distinguish a legal database state from an illegally-corrupted one when the two states are matched on their low-order column marginals?

**Answer**: No.  Values-only models are at chance (50 ± 1 %).  A rule-executor with access to operational context (schema, constraints, derivation rules) is near-perfect (≥ 99.9 %).

---

## Setup

Requires Python ≥ 3.11.

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib`.

---

## Reproducing the experiment

### Stage 1 — Marginal Match Gate

Verifies that each corruption strategy produces (legal, illegal) pairs whose 1- and 2-way column marginals match within TV distance τ = 0.02.  This is the prerequisite for the identifiability claim.

```bash
python scripts/validate_construction.py
```

Expected output (≈ 30 s on a laptop):

```
[PASS] cardinality_break   mean=0.0041  max=0.0069  p95=0.0057  min_p=1.000
[PASS] derivation_break    mean=0.0025  max=0.0033  p95=0.0031  min_p=1.000
[PASS] fk_break            mean=0.0000  max=0.0000  p95=0.0000  min_p=1.000
[PASS] transition_break    mean=0.0024  max=0.0038  p95=0.0033  min_p=1.000

✓ GATE PASSED — all violation types within tau=0.02, proceed to Stage 2.
```

Exit code 0 on pass, 1 on failure (genuine finding, not a bug to suppress).

### Stage 2 — Operational Turing Test

Trains three baselines on 1 000 labeled (legal, illegal) pairs and evaluates on 500 test pairs across 5 random seeds.

```bash
python scripts/run_turing_test.py
```

Runtime: ≈ 20 minutes on a laptop (dataset construction dominates).  
Outputs:

| File | Contents |
|------|----------|
| `artifacts/turing_test_results.csv` | Per-seed accuracy and per-violation recall for each baseline |
| `artifacts/turing_test_figure.png` | Bar chart with 95 % bootstrap CIs |

**Results** (5 seeds, n\_train = 1 000 pairs, n\_test = 500 pairs):

| Baseline | Mean Acc | 95 % CI | Δ chance |
|----------|----------|---------|---------|
| Oracle (upper bound) | 1.0000 | [1.0000, 1.0000] | +0.5000 |
| XGBoost values-only | 0.5014 | [0.4968, 0.5068] | +0.0014 |
| XGBoost + schema | 0.9996 | [0.9992, 1.0000] | +0.4996 |

Per-violation illegal-state recall:

| Baseline | cardinality | derivation | fk | transition |
|----------|-------------|------------|----|------------|
| Oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost values-only | 0.502 | 0.487 | 0.536 | 0.537 |
| XGBoost + schema | 1.000 | 0.997 | 1.000 | 1.000 |

### Smoke tests

```bash
pytest tests/ -v
```

68 tests covering rule helpers, oracle, generator, and all four corruption strategies (rule firing, isolation, marginal preservation, determinism).  Runs in ≈ 3 s.

---

## Design

### Schema

Three tables, four operational rules.

```
customers(id, country, tier, signup_date)
orders(id, customer_id, status, prev_status, order_date, total)
order_items(id, order_id, product_id, quantity, unit_price, line_total)
```

**Rules** (`src/tf_pilot/rules.py`):

| Rule | Constraint |
|------|-----------|
| FK integrity | Every `orders.customer_id` ∈ `customers.id`; every `order_items.order_id` ∈ `orders.id` |
| Cardinality | 1–20 items per order; ≤ 3 open (pending/shipped) orders per customer |
| Derivation | `line_total = qty × unit_price × (1 − discount(tier, qty))`; `order.total = Σ line_totals × (1 + tax(country))` |
| Transition | `(prev_status, status)` ∈ {(pending,shipped), (shipped,delivered), (pending,cancelled)} |

### Corruption strategies

Each strategy corrupts exactly one rule while preserving 1- and 2-way column marginals within τ = 0.02:

- **fk_break**: replace ≈ 5 % of `orders.customer_id` with out-of-range IDs
- **cardinality_break**: move items within same-customer orders to exceed the 20-item limit; recomputes totals (same country ⇒ same tax rate)
- **derivation_break**: add paired zero-sum noise ≈ 0.1 % of mean total to `orders.total` only; leaves all line_totals untouched
- **transition_break**: swap `status` between one `delivered` and one initial `pending` order; 1-way status marginal is exactly preserved

### Feature sets

**Values-only** (`state_to_values_vector`): column-level aggregate statistics (mean, std, min, max, q25, q75 for numerics; entropy and mode-frequency for categorics) and two cross-column Pearson correlations.  Deliberately excludes any group-size statistic (e.g. items-per-order count) — those encode relational structure and would violate the identifiability claim for `cardinality_break`.

**Schema-aware** (`state_to_schema_vector`): values-only features concatenated with 7 operational proxy features that directly measure rule violations (orphan FK counts, cardinality overflows, line_total residuals, order-total residuals, illegal transition count).

### Why values-only must be at chance

The identifiability claim is: any classifier restricted to k-way marginals of column values has Bayes error ≥ ½ − O(τ) when the legal/illegal pair is τ-close in total variation on those marginals.  Stage 1 verifies τ ≤ 0.02 for all four corruption types.  The XGBoost values-only baseline is the empirical test of this claim.

---

## Module layout

```
src/tf_pilot/
  rules.py       — rule definitions, oracle; single source of truth
  generator.py   — generate_legal_state
  corruptor.py   — corrupt_to_illegal, one strategy per violation type
  verifier.py    — verify_marginal_match (TV distance + permutation test)
  features.py    — state_to_values_vector, state_to_schema_vector
  baselines.py   — OracleClassifier, XGBValuesOnly, XGBSchemaAware

scripts/
  validate_construction.py   — Stage 1 gate
  run_turing_test.py         — Stage 2 experiment

tests/
  test_smoke.py              — 68 smoke tests
```
