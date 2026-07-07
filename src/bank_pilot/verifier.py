"""
Marginal-match verifier for the banking schema.

Reuses tf_pilot.verifier internals; only the per-table column list differs.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from tf_pilot import verifier as _v

# Columns included in marginal computation per table for the banking schema.
# IDs are excluded (surrogate, no distributional meaning); dates are excluded
# (skew cannot be balanced by permutation-based corruption).
BANK_TABLE_COLS: dict[str, list[str]] = {
    "accounts":          ["type", "balance_cents"],
    "transactions":      ["category"],
    "transaction_lines": ["amount_cents", "side"],
}


def verify_marginal_match(
    S: dict[str, pd.DataFrame],
    S_prime: dict[str, pd.DataFrame],
    order_k: int = 2,
    tau: float = _v.TAU,
    n_bins: int = _v.N_BINS,
    n_permutations: int = _v.N_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Drop-in banking-schema verifier."""
    saved = _v.TABLE_COLS
    _v.TABLE_COLS = BANK_TABLE_COLS
    try:
        return _v.verify_marginal_match(
            S, S_prime,
            order_k=order_k, tau=tau, n_bins=n_bins,
            n_permutations=n_permutations, seed=seed,
        )
    finally:
        _v.TABLE_COLS = saved
