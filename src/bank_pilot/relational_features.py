"""
Relational featurizer for the banking schema.

Mirrors the structure of `tf_pilot.features._relation_features` for the
order-to-cash schema: cardinalities, FK validity / coverage, degree
distributions over the FK graph, per-category counts, and joined-context
summaries.

Deliberately excluded (these are the audit / oracle features):
  - per-transaction debit-credit balance residuals (rule 3 oracle)
  - per-account balance vs cumulative residual (rule 4 oracle)
  - thresholded line-count legality (the cardinality bounds themselves)
  - any explicit balanced-side check

This puts the relational tier strictly between values-only and the
rule-derived audit features, in line with the access-ladder framing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tf_pilot.features import _num_stats, _cat_entropy, _safe_num_stats, _frequency_vector

from .features import state_to_values_vector
from .rules import ACCOUNT_TYPES, TRANSACTION_CATEGORIES, SIDES


def _relation_features(S: dict[str, pd.DataFrame]) -> np.ndarray:
    acc = S["accounts"]
    tx  = S["transactions"]
    ln  = S["transaction_lines"]

    feats: list[float] = []

    n_acc = max(len(acc), 1)
    n_tx  = max(len(tx),  1)
    n_ln  = max(len(ln),  1)

    feats += [
        float(len(acc)),
        float(len(tx)),
        float(len(ln)),
        float(len(tx) / n_acc),
        float(len(ln) / n_tx),
        float(len(ln) / n_acc),
    ]

    # FK columns visible at the relation tier (without legality predicates).
    feats += _safe_num_stats(ln["transaction_id"])
    feats += _safe_num_stats(ln["account_id"])

    valid_tx_ids  = set(tx["id"])
    valid_acc_ids = set(acc["id"])
    feats += [
        float(ln["transaction_id"].isin(valid_tx_ids).mean()) if len(ln) else 0.0,
        float(ln["account_id"].isin(valid_acc_ids).mean()) if len(ln) else 0.0,
        float(ln["transaction_id"].nunique()),
        float(ln["account_id"].nunique()),
    ]

    # Degree distributions over the FK graph.
    lines_per_valid_tx = (
        ln[ln["transaction_id"].isin(valid_tx_ids)]
        .groupby("transaction_id")
        .size()
        .reindex(tx["id"], fill_value=0)
    )
    lines_per_valid_acc = (
        ln[ln["account_id"].isin(valid_acc_ids)]
        .groupby("account_id")
        .size()
        .reindex(acc["id"], fill_value=0)
    )
    feats += _safe_num_stats(lines_per_valid_tx)
    feats += _safe_num_stats(lines_per_valid_acc)

    # Side balance proxy: count of debit / credit lines per transaction
    # (NOT amount sums - the audit feature handles those).
    sides_per_tx = (
        ln.groupby(["transaction_id", "side"])
        .size()
        .unstack(fill_value=0)
    )
    n_debits_per_tx  = sides_per_tx.get("debit",  pd.Series(0, index=sides_per_tx.index))
    n_credits_per_tx = sides_per_tx.get("credit", pd.Series(0, index=sides_per_tx.index))
    feats += _safe_num_stats(n_debits_per_tx)
    feats += _safe_num_stats(n_credits_per_tx)

    # Per-category transaction counts.
    feats += _frequency_vector(tx["category"], TRANSACTION_CATEGORIES)
    # Per-side line counts.
    feats += _frequency_vector(ln["side"], SIDES)
    # Per-account-type counts.
    feats += _frequency_vector(acc["type"], ACCOUNT_TYPES)

    # Joined context summaries: tx category projected to lines.
    tx_cat_lookup = tx.set_index("id")["category"]
    line_tx_category = ln["transaction_id"].map(tx_cat_lookup)
    feats += _frequency_vector(line_tx_category, TRANSACTION_CATEGORIES)
    for cat in TRANSACTION_CATEGORIES:
        amts = ln.loc[line_tx_category == cat, "amount_cents"].astype(float)
        feats += _safe_num_stats(amts)

    # Joined context: account type projected to lines.
    acc_type_lookup = acc.set_index("id")["type"]
    line_acc_type = ln["account_id"].map(acc_type_lookup)
    feats += _frequency_vector(line_acc_type, ACCOUNT_TYPES)
    for t in ACCOUNT_TYPES:
        amts = ln.loc[line_acc_type == t, "amount_cents"].astype(float)
        feats += _safe_num_stats(amts)

    arr = np.array(feats, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def state_to_relation_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Values vector + relational structure features (no oracle predicates)."""
    return np.concatenate([state_to_values_vector(S), _relation_features(S)])


# ── rule-agnostic cumulative-aggregate features ──────────────────────────────
#
# The reviewer concern: "what if you gave the relational model generic
# cumulative-aggregate features (cumsum, group-balance) without telling it
# which aggregate matters?" Standard relational deep learning arguably
# includes these. We add cumsum-by-account, group totals/counts per
# transaction, and per-account net signed amount. These do NOT encode the
# audit predicates (no |actual - cumulative| residual, no per-tx debit-credit
# difference); they are unconditional aggregates a generic relational
# featurizer would compute.

def _cumagg_features(S: dict[str, pd.DataFrame]) -> np.ndarray:
    acc = S["accounts"]
    tx  = S["transactions"]
    ln  = S["transaction_lines"]
    feats: list[float] = []

    if len(ln) == 0:
        # 6 stats * 4 series + 4 account_types * 6 stats = 48 features
        return np.zeros(48, dtype=np.float64)

    # Per-account cumulative amount (unsigned and signed).
    unsigned_per_account = ln.groupby("account_id")["amount_cents"].sum()
    signed_amount = ln["amount_cents"] * ln["side"].map({"credit": 1, "debit": -1})
    signed_per_account = signed_amount.groupby(ln["account_id"]).sum()
    unsigned_per_account = unsigned_per_account.reindex(acc["id"], fill_value=0)
    signed_per_account = signed_per_account.reindex(acc["id"], fill_value=0)
    feats += _safe_num_stats(unsigned_per_account)
    feats += _safe_num_stats(signed_per_account)

    # Per-transaction total amount and signed total.
    unsigned_per_tx = ln.groupby("transaction_id")["amount_cents"].sum()
    signed_per_tx = signed_amount.groupby(ln["transaction_id"]).sum()
    unsigned_per_tx = unsigned_per_tx.reindex(tx["id"], fill_value=0)
    signed_per_tx = signed_per_tx.reindex(tx["id"], fill_value=0)
    feats += _safe_num_stats(unsigned_per_tx)
    feats += _safe_num_stats(signed_per_tx)

    # Per-account-type aggregate net signed amount (4 account types * 4 stats).
    acc_type_lookup = acc.set_index("id")["type"]
    line_acc_type = ln["account_id"].map(acc_type_lookup)
    for t in ACCOUNT_TYPES:
        sa = signed_amount.loc[line_acc_type == t]
        feats += _safe_num_stats(sa) if len(sa) else [0.0]*6

    arr = np.array(feats, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def state_to_relation_plus_cumagg_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Relational tier + rule-agnostic cumulative-aggregate features.

    The cumulative aggregates are cumsum-style summaries (per-account net
    signed amount, per-transaction signed total) that a generic relational
    featurizer would emit without knowing which rule matters. They do not
    encode the audit residuals (|account.balance - cumulative|, per-tx
    debit-credit difference); a model still has to learn to compare these
    aggregates against \texttt{account.balance} or zero.
    """
    return np.concatenate([
        state_to_values_vector(S),
        _relation_features(S),
        _cumagg_features(S),
    ])
