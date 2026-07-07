# RDB-PFN (third-party dependency)

The relational-PFN baseline used by `scripts/run_rdbpfn.py` is the upstream
implementation of Wang et al. (2026), "Relational In-Context Learning via
Synthetic Pre-training with Structural Prior" (arXiv:2603.03805).

This directory is intentionally a placeholder. To run `scripts/run_rdbpfn.py`,
clone the upstream repository at this path, e.g.:

    git clone <upstream-url> external/RDBPFN

(See the cited paper for the canonical code link.) The script's import paths
expect the repository layout to be at `external/RDBPFN/` relative to the repo
root.

If you do not need the RDB-PFN baseline, you can skip this dependency. All
other scripts in `scripts/` run without it. The pre-computed RDB-PFN result
in `results/rdbpfn_results.csv` reproduces the row in Table 1 of the paper.
