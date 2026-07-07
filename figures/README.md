# Figures

Pre-rendered PNGs of every diagnostic figure produced by the scripts in
`scripts/`. These are the empirical originals; the figures included in the
write-up are TikZ/pgfplots rebuilds of the same data.

| File                             | Shows                                                        |
|----------------------------------|--------------------------------------------------------------|
| `ott_access_ladder_readme.png`   | README hero: conceptual ladder, empirical validation, LLM control |
| `turing_test_figure.png`         | Access ladder: values-only / row / relational / op.-grounded |
| `scaling_curve_figure.png`       | Values-only accuracy is flat over two orders of magnitude    |
| `row_level_figure.png`           | Per-violation recall under raw row-level access              |
| `relation_only_figure.png`       | Per-violation recall under relational features (HistGB)      |
| `rdbpfn_figure.png`              | Relational PFN baseline reproduces the relational pattern    |
| `mi_shuffle_null_figure.png`     | Observed MI vs. label-shuffled null distribution             |
| `diagnostics_calibration.png`    | Calibration diagnostics for the values-only models           |

To regenerate, run the corresponding `scripts/run_*.py` (each emits both the
result CSV and a PNG into `artifacts/`).
