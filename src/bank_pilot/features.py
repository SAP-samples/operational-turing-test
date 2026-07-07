"""
Featurizers for the banking schema.

  state_to_values_vector  — column-level aggregates only; no relational
                            structure, no rule awareness.
  state_to_schema_vector  — values vector + rule-derived audit features
                            (operationally grounded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tf_pilot.features import _num_stats, _cat_entropy

from .rules import (
    MIN_LINES_PER_TX, MAX_LINES_PER_TX,
    ACCOUNT_TYPES, TRANSACTION_CATEGORIES,
)


# ── values-only features ──────────────────────────────────────────────────────

def state_to_values_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Column-level statistics only. No schema, no relational, no rule.

    Excludes group-size statistics (lines per transaction, transactions per
    account) because those encode structural information and would violate
    the values-only identifiability claim for the cardinality and balance
    rules.
    """
    acc = S["accounts"]
    tx  = S["transactions"]
    ln  = S["transaction_lines"]

    feats: list[float] = []

    # accounts — column distributions
    feats += _cat_entropy(acc["type"])
    feats += _num_stats(acc["balance_cents"].astype(float))

    # transactions — column distributions
    feats += _cat_entropy(tx["category"])
    feats += [float(len(tx))]

    # transaction_lines — per-row column distributions (no grouping)
    feats += _num_stats(ln["amount_cents"].astype(float))
    feats += _cat_entropy(ln["side"])

    return np.array(feats, dtype=np.float64)


# ── seven audit features (operationally grounded) ─────────────────────────────

def _schema_features(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Seven rule-derived audit features:

      1. n_orphan_fk_lines_tx     — lines with transaction_id ∉ transactions.id
      2. n_orphan_fk_lines_acc    — lines with account_id ∉ accounts.id
      3. n_tx_under_min_lines     — transactions with fewer than MIN_LINES_PER_TX lines
      4. n_tx_over_line_limit     — transactions with more than MAX_LINES_PER_TX lines
      5. n_tx_unbalanced_sides    — transactions missing either a debit or a credit
      6. max_abs_tx_balance       — max |Σ debit − Σ credit| per transaction
      7. max_abs_account_balance  — max |account.balance − cumulative| per account
    """
    acc = S["accounts"]
    tx  = S["transactions"]
    ln  = S["transaction_lines"]

    valid_tx_ids  = set(tx["id"])
    valid_acc_ids = set(acc["id"])

    # 1, 2 — FK
    n_orphan_fk_lines_tx  = int((~ln["transaction_id"].isin(valid_tx_ids)).sum())
    n_orphan_fk_lines_acc = int((~ln["account_id"].isin(valid_acc_ids)).sum())

    # 3, 4, 5 — cardinality
    counts_per_tx = ln.groupby("transaction_id").size()
    n_tx_under_min_lines = int((counts_per_tx < MIN_LINES_PER_TX).sum())
    n_tx_over_line_limit = int((counts_per_tx > MAX_LINES_PER_TX).sum())
    sides_per_tx = ln.groupby("transaction_id")["side"].apply(set)
    has_both = sides_per_tx.apply(lambda s: "debit" in s and "credit" in s)
    n_tx_unbalanced_sides = int((~has_both).sum())

    # 6 — double-entry residual
    sums = (
        ln.groupby(["transaction_id", "side"])["amount_cents"]
        .sum()
        .unstack(fill_value=0)
    )
    debit  = sums.get("debit",  pd.Series(0, index=sums.index))
    credit = sums.get("credit", pd.Series(0, index=sums.index))
    if len(sums) > 0:
        max_abs_tx_balance = float((debit - credit).abs().max())
    else:
        max_abs_tx_balance = 0.0

    # 7 — account-balance residual
    if len(ln) > 0:
        signed = ln.assign(
            signed_amount=ln["amount_cents"]
            * ln["side"].map({"credit": 1, "debit": -1})
        )
        cumulative = signed.groupby("account_id")["signed_amount"].sum()
        cumulative = cumulative.reindex(acc["id"], fill_value=0)
    else:
        cumulative = pd.Series(0, index=acc["id"])
    expected = cumulative.values
    actual = acc.set_index("id")["balance_cents"].reindex(acc["id"]).values
    max_abs_account_balance = float(np.abs(actual - expected).max()) if len(acc) else 0.0

    return np.array([
        n_orphan_fk_lines_tx,
        n_orphan_fk_lines_acc,
        n_tx_under_min_lines,
        n_tx_over_line_limit,
        n_tx_unbalanced_sides,
        max_abs_tx_balance,
        max_abs_account_balance,
    ], dtype=np.float64)


def state_to_schema_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Values + seven audit features (operationally grounded)."""
    return np.concatenate([state_to_values_vector(S), _schema_features(S)])
