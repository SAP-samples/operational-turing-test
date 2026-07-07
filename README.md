<div align="center">

[![Status](https://img.shields.io/badge/Status-ICML_FMSD_2026-1f5a9b?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Reproducible](https://img.shields.io/badge/Reproducible-45_min_single_CPU-2ea44f?style=flat-square)](REPRODUCE.md)

</div>

---

Tables in production systems are not just values. They are the visible trace of
application code, database constraints, triggers, and business rules. Strip that
operational layer away and the data reaching a model becomes a projection:
stored values remain, but much of the logic that makes a state legal is gone.

The **Operational Turing Test (OTT)** makes this gap falsifiable. It pairs legal
and rule-violating database states so ordinary 1- and 2-way column-value
statistics stay matched (`TV < 0.02`). Le Cam's lemma then bounds any
values-only classifier at chance. The result is not a capacity failure:
larger models and richer value representations cannot recover logic they cannot
evaluate.

<p align="center">
<img src="figures/ott_access_ladder_readme.png" width="920" alt="Operational Turing Test access ladder and empirical validation">
</p>

## Main result

The paper evaluates an access ladder rather than a leaderboard:

| Access level | What the model sees | Representative result |
| :-- | :-- | :-- |
| Leakage controls | column statistics, values-only baselines, raw rows | `0.50` accuracy |
| Relational structure | joins, key coverage, group counts, relational PFN baseline | `0.89` accuracy, but value-transformation recall remains `0.00-0.02` |
| Executable rule-derived audits | seven SQL audits derived from schema, trigger, and application rules | `1.00` classification accuracy |
| Oracle | direct rule evaluation | `1.00` classification accuracy |

The values-only controls use XGBoost, TabICL, and TabPFN and are statistically
equivalent to chance under a pre-registered TOST equivalence test (`p < 0.002`,
margin `±2 pp`). Relational structure recovers connectivity-visible rule
families such as foreign-key, cardinality, and state-transition violations. It
does not recover value-transformation logic: stored totals must be recomputed
from quantities, prices, discounts, and taxes.

The same access-ladder pattern appears on a second banking-ledger schema with
structurally distinct rule families: balance and cumulative-account rules are
not solved by value statistics alone.

### Frontier-LLM source-artifact control

The LLM controls receive the schema, trigger source, procedural rule tables,
and state CSVs. The task is still binary legality classification.

| Run                       | Accuracy | LEGAL on legal states | Illegal states labelled illegal |
| :------------------------ | :------: | :-------------------: | :-----------------------------: |
| GPT-5.5 default prompt    |  0.50    |         0/50          |              50/50              |
| GPT-5.5 high effort       |  0.50    |         0/50          |              50/50              |
| GPT-5.5 + SQL executor    |  0.50    |         0/50          |              50/50              |
| Kimi-K2.6                 |  0.46    |         2/50          |              44/50              |

The key diagnostic is not overall accuracy: several runs reject almost every
state as illegal. The striking failure is valid-state recall. GPT-5.5 accepts
no legal state even when given higher reasoning effort or a SQL execution
tool; Kimi-K2.6 accepts two legal states and hits the output cap on 14/100
states. The SQL-executor run can execute queries, but the audit logic is still
specified ad hoc by the model rather than compiled from the operational rules.

## Quick start

```bash
git clone https://github.com/SAP-samples/operational-turing-test.git
cd OTT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tabpfn]"
```

Run the headline experiment (`Table 1` in the paper):

```bash
python scripts/run_turing_test.py
```

Single CPU, no GPU. Full non-LLM reproduction takes about **45 minutes** on a
2023 laptop. The LLM controls are API-dependent and use credentials from a local
`.env` file; no keys are stored in the repository. See [`REPRODUCE.md`](REPRODUCE.md)
for the per-claim mapping.

## Repository structure

```
OTT/
├── README.md            project description (this file)
├── REPRODUCE.md         per-claim mapping: paper element → script → result file
├── LICENSE              Apache-2.0
├── pyproject.toml       Python package metadata + dependencies
├── requirements.txt     pip-style dependency list (mirrors pyproject)
├── .env.example         template for endpoint / key env vars (used by experiments/)
│
├── schema/
│   ├── schema.sql       three-table order-to-cash DDL (declarative rules)
│   └── trigger.sql      legal-transition trigger function (procedural rule)
│
├── src/tf_pilot/        the tf_pilot Python package
│   ├── generator.py     legal-state generator
│   ├── corruptor.py     four single-rule corruption strategies
│   ├── rules.py         operational-rule definitions (discount, tax, transitions)
│   ├── verifier.py      oracle classifier + per-rule audit predicates
│   ├── features.py      values-only summaries + executable audit outputs
│   └── baselines.py     XGBoost / TabICL / TabPFN / HistGB wrappers
│
├── scripts/             runnable experiments (one per paper claim)
│   ├── validate_construction.py      Stage-1 marginal-match gate
│   ├── run_turing_test.py            Table 1 main result
│   ├── run_scaling_curve.py          50 → 5,000 training pairs
│   ├── run_row_level.py              raw-row TabICL baseline
│   ├── run_relation_only.py          HistGB with relational features
│   ├── run_rdbpfn.py                 RDB-PFN baseline
│   ├── run_tabpfn.py                 TabPFN-only diagnostic
│   ├── run_mi_probe.py               feature-label mutual information
│   ├── run_mi_shuffle_null.py        label-shuffled MI null distribution
│   ├── run_diagnostics.py            calibration + permutation invariance checks
│   └── export_relational_dataset.py  flat-CSV state export for the LLM pilots
│
├── tests/               pytest smoke tests for the construction
├── results/             pre-computed CSVs for every paper table + LLM pilots
├── figures/             README hero + rendered diagnostic figures (PNG)
└── experiments/         frontier-LLM pilot scripts (Kimi-K2.6, GPT-5.5)
    ├── run_kimi_baseline.py
    ├── run_gpt55_baseline.py
    ├── run_kimi_per_state.sh
    └── README.md
```

## The benchmark

A test instance is a pair of database states `(S_legal, S_illegal)` over a small order-to-cash schema (`customers`, `orders`, `order_items`). The illegal state is generated by applying exactly one of four single-rule corruption strategies. Each is constructed so all 1- and 2-way column-value marginals between the two states stay within total variation distance `τ = 0.02`.

| Corruption          | Rule violated                       | Where the rule lives        | Empirical max TV |
| :------------------ | :---------------------------------- | :-------------------------- | :--------------: |
| `fk_break`          | referential integrity               | schema (FK declaration)     |     `0.0000`     |
| `cardinality_break` | per-customer / per-order limits     | mixed (CHECK + application) |     `0.0069`     |
| `derivation_break`  | line- and order-total value transformations | application code     |     `0.0033`     |
| `transition_break`  | legal value-state transitions       | trigger code                |     `0.0038`     |

The model is evaluated under one of four access tiers, each implemented in `src/features.py` and `src/relational_features.py`:

- **values-only** · column-level summaries used as leakage controls.
- **row-level** · raw `orders` rows fed per-row, then aggregated by majority vote per state.
- **relational** · values plus joins, key coverage, group degrees, and within-row value pairs such as `(prev_status, status)`.
- **operationally grounded** · values plus seven executable rule-derived audits, each a SQL check derived from the schema and rule code.

## Pre-registration

The marginal-match threshold `τ = 0.02`, the TOST equivalence margin `δ = 0.02`, the random seeds, and the per-baseline scales are fixed in [`src/config.py`](src/) and were registered before evaluation. The full pre-registration is reproducible from the constants in that file.

## Citation

If you use this benchmark or code, please cite:

```bibtex
@inproceedings{klein2026ott,
  title     = {Statistically Indistinguishable, Operationally Distinct: A Formal Barrier for Tabular Foundation Models},
  author    = {Klein, Tassilo and Hoffart, Johannes},
  booktitle = {Proceedings of the 2nd ICML Workshop on Foundation Models for Structured Data},
  address   = {Seoul, South Korea},
  year      = {2026}
}
```

## License

[Apache-2.0](LICENSE).
