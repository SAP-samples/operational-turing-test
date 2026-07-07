"""
Operational rules for the second-schema instantiation: banking ledger.

Three tables (accounts, transactions, transaction_lines). Four rule families
that are *structurally distinct* from the order-to-cash schema:

  1. FK integrity              (declarative, schema)        — same shape as before
  2. Cardinality               (mixed)                       — 2-10 lines per tx, ≥1 debit & ≥1 credit
  3. Double-entry balance      (procedural, app code)        — NEW SHAPE: cross-row constraint
                                                                within sibling lines of the same parent
                                                                transaction; not decomposable to per-row
                                                                derivation.
  4. Account-balance consistency (procedural, app code)     — NEW SHAPE: cumulative cross-table aggregate
                                                                across ALL transactions for each account;
                                                                not a parent-child derivation.

The first schema's "derivation" is per-row (each line independently); the
banking schema's balance rules are aggregations *across* rows. Status
transitions are absent here. This is the structural difference the second
schema is meant to demonstrate.
"""
from __future__ import annotations

import pandas as pd

# ── Operational data ──────────────────────────────────────────────────────────

ACCOUNT_TYPES: list[str] = ["checking", "savings", "expense", "revenue"]
TRANSACTION_CATEGORIES: list[str] = [
    "purchase", "deposit", "transfer", "fee", "salary", "refund",
]
SIDES: list[str] = ["debit", "credit"]

MIN_LINES_PER_TX: int = 2
MAX_LINES_PER_TX: int = 10


# ── Rules ─────────────────────────────────────────────────────────────────────

def rule_fk_integrity(S: dict[str, pd.DataFrame]) -> bool:
    """Every transaction_lines.transaction_id resolves to a transaction;
    every transaction_lines.account_id resolves to an account."""
    valid_tx_ids = set(S["transactions"]["id"])
    valid_acc_ids = set(S["accounts"]["id"])
    if not set(S["transaction_lines"]["transaction_id"]).issubset(valid_tx_ids):
        return False
    if not set(S["transaction_lines"]["account_id"]).issubset(valid_acc_ids):
        return False
    return True


def rule_cardinality(S: dict[str, pd.DataFrame]) -> bool:
    """2 ≤ lines per transaction ≤ 10; every transaction has ≥1 debit AND ≥1 credit."""
    lines = S["transaction_lines"]
    if len(lines) == 0:
        return True
    counts = lines.groupby("transaction_id").size()
    if (counts < MIN_LINES_PER_TX).any():
        return False
    if (counts > MAX_LINES_PER_TX).any():
        return False
    # Each transaction has at least one debit and at least one credit.
    sides_per_tx = lines.groupby("transaction_id")["side"].apply(set)
    if not (sides_per_tx.apply(lambda s: "debit" in s and "credit" in s)).all():
        return False
    return True


def rule_double_entry(S: dict[str, pd.DataFrame], tol: int = 0) -> bool:
    """For each transaction, Σ amount(debit) == Σ amount(credit).

    This is a CROSS-ROW constraint within sibling lines of the same parent
    transaction, structurally distinct from the order-to-cash schema's per-row
    derivation formula.
    """
    lines = S["transaction_lines"]
    if len(lines) == 0:
        return True
    # Pivot: amount.sum() grouped by (transaction_id, side)
    sums = (
        lines.groupby(["transaction_id", "side"])["amount_cents"]
        .sum()
        .unstack(fill_value=0)
    )
    # If a transaction has only debits or only credits, the "missing" side
    # column is filled with 0; the difference is then the existing side total.
    debit  = sums.get("debit",  pd.Series(0, index=sums.index))
    credit = sums.get("credit", pd.Series(0, index=sums.index))
    return bool(((debit - credit).abs() <= tol).all())


def rule_account_balance(S: dict[str, pd.DataFrame], tol: int = 0) -> bool:
    """For each account a:
       a.balance_cents == Σ amount(side='credit', account=a) − Σ amount(side='debit', account=a)

    A cumulative cross-table aggregate across ALL transactions for the
    account; not a parent-child derivation.
    """
    lines = S["transaction_lines"]
    accounts = S["accounts"]
    if len(accounts) == 0:
        return True

    # Compute per-account cumulative balance from lines
    if len(lines) == 0:
        cumulative = pd.Series(0, index=accounts["id"])
    else:
        signed = lines.assign(
            signed_amount=lines["amount_cents"] * lines["side"].map({"credit": 1, "debit": -1})
        )
        cumulative = signed.groupby("account_id")["signed_amount"].sum()
        cumulative = cumulative.reindex(accounts["id"], fill_value=0)

    expected = cumulative.values
    actual = accounts.set_index("id")["balance_cents"].reindex(accounts["id"]).values
    return bool((abs(actual - expected) <= tol).all())


# ── Oracle ────────────────────────────────────────────────────────────────────

_RULES: dict[str, object] = {
    "fk_integrity":    rule_fk_integrity,
    "cardinality":     rule_cardinality,
    "double_entry":    rule_double_entry,
    "account_balance": rule_account_balance,
}


def oracle(S: dict[str, pd.DataFrame]) -> dict:
    results = {name: fn(S) for name, fn in _RULES.items()}
    return {
        "legal": all(results.values()),
        "violations": [name for name, passed in results.items() if not passed],
    }
