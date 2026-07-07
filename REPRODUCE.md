# Reproducing every paper claim

This document maps each empirical claim in the paper to the script that produces it. Random seeds, the marginal-match threshold `τ`, and the TOST margin `δ` are pre-registered in the source modules (see `src/tf_pilot/`) and were fixed before evaluation.

## Setup

The Python source is a proper package (`tf_pilot`) installable in editable mode:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tabpfn]"
```

Optional extras:
- `[dev]` — pytest + coverage for the smoke tests under `tests/`.
- `[tabpfn]` — pinned TabPFN v1 (Hollmann et al. 2022); auto-downloads weights on first use.

The `RDB-PFN` baseline depends on the third-party Wang et al. (2026) implementation. Install per the upstream project's instructions and place it at `external/RDBPFN/` (see `external/RDBPFN/README.md`).

All scripts run on a single CPU core. Full reproduction completes in ~45 minutes on a 2023 laptop. TabPFN v1 and TabICL inference are the dominant costs.

## Per-claim mapping

All scripts are invoked from the repo root via `python scripts/<name>.py` (or `python -m scripts.<name>` if the script supports module-mode invocation). Each writes its CSV into `artifacts/` (the path is configurable per-script).

### §3 Construction — marginal-match condition

**Claim:** every corruption preserves 1- and 2-way column-value marginals to within `τ < 0.02`, with permutation-null `p = 1.000`.

**Empirically:** max TV per strategy

| Strategy            | Stage 1 max TV | Spot-check max TV | permutation-null p |
|---------------------|----------------|-------------------|--------------------|
| `fk_break`          | 0.0000         | 0.0000            | 1.000              |
| `derivation_break`  | 0.0033         | 0.0054            | 1.000              |
| `cardinality_break` | 0.0069         | 0.0065            | 1.000              |
| `transition_break`  | 0.0038         | 0.0036            | 1.000              |

```bash
python scripts/validate_construction.py
```

### §4 Chance across architectures (Table 1, values-only and op.-grounded rows)

**Claim:** all three values-only architectures equivalent to chance under TOST (`p < 0.002`, margin `±2 pp`); operationally-grounded XGBoost reaches 0.9996; oracle reaches 1.0000.

**Scale:** 200 customers per state, 1,000 train / 500 test pairs, 5 seeds.

```bash
python scripts/run_turing_test.py
```

Result: [`results/turing_test_results.csv`](results/turing_test_results.csv).

Reproduces:
- XGBoost values-only: **0.5014** (TOST `p = 0.0018`)
- TabICL values-only: **0.5006** (`p < 10⁻⁴`)
- TabPFN v1 values-only: **0.5012** (`p < 10⁻⁴`)
- XGBoost + 7 rule features: **0.9996**
- Oracle: **1.0000**

### §4 TabICL signature, $\varphi$-correlation, TabPFN diagnostic

The same script writes [`results/tabpfn_results.csv`](results/tabpfn_results.csv) with per-instance prediction probabilities and the cross-architecture φ-correlation. For a deeper TabPFN-only diagnostic:

```bash
python scripts/run_tabpfn.py
```

### §4 More data does not help (scaling claim)

**Claim:** values-only accuracy is flat at `0.50` from 50 to 5,000 training pairs; the operationally-grounded model saturates at `~0.997` from 50 pairs.

**Scale:** N_TRAIN ∈ {50, 100, 250, 500, 1000, 2500, 5000}, 3 seeds.

```bash
python scripts/run_scaling_curve.py
```

Result: [`results/scaling_curve_results.csv`](results/scaling_curve_results.csv).

### §4 Row-level access does not help

**Claim:** TabICL on raw `orders` rows reaches `0.500` on all four violations.

**Scale:** 200 train / 100 test pairs per violation type, 5 seeds.

```bash
python scripts/run_row_level.py
```

Result: [`results/row_level_results.csv`](results/row_level_results.csv).

### §4 Relational baseline (Table 1, Relational rows)

**Claim:** HistGB with cross-table joins, FK coverage, per-group degrees, and the within-row `(prev_status, status)` joint reaches `0.889`. RDB-PFN reproduces the pattern.

```bash
python scripts/run_relation_only.py     # HistGB, 250 train / 150 test, 3 seeds
python scripts/run_rdbpfn.py            # RDB-PFN, 100 train / 50 test, 3 seeds
```

Results: [`results/relation_only_results.csv`](results/relation_only_results.csv), [`results/rdbpfn_results.csv`](results/rdbpfn_results.csv).

The RDB-PFN script imports from the upstream Wang et al. (2026) implementation; see `external/RDBPFN/README.md` for installation.

### §4 Mutual information probes

**Claim:** observed feature-label MI sits at the bottom of a label-shuffled null distribution (`z = −1.86`, pooled `p = 0.999`).

```bash
python scripts/run_mi_probe.py          # Kraskov k-NN MI on observed (x, y)
python scripts/run_mi_shuffle_null.py   # 1000 label permutations on the same x
```

Results: [`results/mi_probe_results.csv`](results/mi_probe_results.csv), [`results/mi_shuffle_null_results.csv`](results/mi_shuffle_null_results.csv).

### §4 Diagnostics suite

```bash
python scripts/run_diagnostics.py
```

Calibration plots, label-permutation invariance checks, feature-exclusion tests.

### §3/Appendix Construction export (used by the LLM pilots)

To export the test states as flat CSVs for the LLM follow-up experiments (Kimi, GPT-5.5):

```bash
python scripts/export_relational_dataset.py
```

Writes `artifacts/relational_export/{train,test}/state_*/` directories with `customers.csv`, `orders.csv`, `order_items.csv` per state. The LLM pilots in `experiments/` consume this directly.

## Smoke tests

```bash
pytest tests/
```

The smoke tests encode contractual claims of the construction:
1. `generate_legal_state` always produces a state the oracle accepts.
2. `corrupt_to_illegal` always fires exactly the targeted rule.
3. The 1-way `status` marginal is exactly preserved by `transition_break`.
4. The item value marginals are untouched by `cardinality_break`.
5. The `orders.total` column is untouched by `derivation_break` (so its value stays wrong).
6. Null corruption (deep copy) leaves the oracle unchanged.

A smoke-test failure is a design bug, not a data fluke.

## LLM source-artifact controls

```bash
cp .env.example .env       # fill in your API endpoint + keys
python experiments/run_kimi_baseline.py --max-states 10 --max-tokens 65536
python experiments/run_gpt55_baseline.py --max-states 10 --max-completion-tokens 65536
```

The paper-facing source-artifact controls use the 50-customer export, which
contains 50 legal and 50 illegal held-out states:

```bash
python scripts/export_relational_dataset.py \
    --n-train 0 --n-test 100 --n-customers 50 \
    --out-dir artifacts/relational_export_n50 --overwrite

# Kimi-K2.6, source artifacts in prompt
bash experiments/run_kimi_per_state.sh \
    --total 100 \
    --export-dir artifacts/relational_export_n50 \
    --prompt-mode find_violations \
    --output artifacts/kimi_n100_findv_results.csv \
    --traces-dir artifacts/kimi_n100_findv_traces

# GPT-5.5, high reasoning effort
python experiments/run_gpt55_baseline.py \
    --export-dir artifacts/relational_export_n50 \
    --max-states 100 \
    --prompt-mode find_violations \
    --reasoning-effort high \
    --max-completion-tokens 65536 \
    --output artifacts/gpt55_n100_high_findv_results.csv \
    --traces-dir artifacts/gpt55_n100_high_findv_traces

# GPT-5.5, SQL-executor variant
python experiments/run_gpt55_tools.py \
    --export-dir artifacts/relational_export_n50 \
    --max-states 100 \
    --max-completion-tokens 65536 \
    --max-tool-calls 20 \
    --output artifacts/gpt55_n100_tools_findv_results.csv \
    --traces-dir artifacts/gpt55_n100_tools_findv_traces
```

Precomputed CSVs are in
[`results/kimi_n100_findv_results.csv`](results/kimi_n100_findv_results.csv),
[`results/gpt55_n100_high_findv_results.csv`](results/gpt55_n100_high_findv_results.csv),
and
[`results/gpt55_n100_tools_findv_results.csv`](results/gpt55_n100_tools_findv_results.csv).
See `experiments/README.md` for setup details and smaller diagnostic controls.

## Compute

| Stage                               | Wall time (single CPU)  |
|-------------------------------------|-------------------------|
| Construction + marginal check       | ~2 min                  |
| Main OTT (5 seeds × 3 architectures)| ~25 min                 |
| Scaling curve                       | ~5 min                  |
| Row-level TabICL                    | ~3 min                  |
| Relational HistGB                   | ~2 min                  |
| RDB-PFN                             | ~5 min                  |
| MI probes (incl. 1000 shuffle)      | ~3 min                  |
| LLM source-artifact controls        | vendor/API dependent; GPT-5.5 high effort ~90 min, GPT-5.5 SQL ~16 min in our run |
| **Total reproduction**              | **~45 min** (non-LLM paper) + API-dependent LLM controls |
