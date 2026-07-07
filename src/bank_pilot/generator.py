"""
Generate legal banking-ledger states.

Each transaction is a balanced multi-line entry: the lines for a transaction
have Σ debits = Σ credits (double-entry invariant). Account balances are
computed from the cumulative debit/credit history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import (
    ACCOUNT_TYPES, TRANSACTION_CATEGORIES,
    MIN_LINES_PER_TX, MAX_LINES_PER_TX,
    oracle,
)


def _balanced_split(amount_cents: int, n_debits: int, n_credits: int,
                    rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Split amount_cents across n_debits debit values and n_credits credit
    values such that Σ debits = Σ credits = amount_cents. Each individual
    value is at least 1 cent.
    """
    def _split(total: int, n: int) -> list[int]:
        if n == 1:
            return [total]
        # n-1 cut-points uniformly in [1, total-1], then take consecutive diffs
        cuts = sorted(rng.integers(1, total, size=n - 1).tolist())
        out = []
        prev = 0
        for c in cuts:
            out.append(int(c - prev))
            prev = c
        out.append(int(total - prev))
        # Replace any zero-or-negative with 1 cent and rebalance the largest entry
        for i, x in enumerate(out):
            if x < 1:
                out[i] = 1
        diff = total - sum(out)
        if diff != 0:
            j = int(np.argmax(out))
            out[j] += diff
        return out

    return _split(amount_cents, n_debits), _split(amount_cents, n_credits)


def generate_legal_state(
    n_accounts: int = 200,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return a legal banking state {accounts, transactions, transaction_lines}.
    Raises AssertionError if the oracle rejects the result.
    """
    rng = np.random.default_rng(seed)
    acc_rng, tx_rng, line_rng, amt_rng = rng.spawn(4)

    # ── Accounts ──────────────────────────────────────────────────────────────
    ids = np.arange(1, n_accounts + 1)
    accounts = pd.DataFrame({
        "id":            ids,
        "type":          acc_rng.choice(ACCOUNT_TYPES, n_accounts),
        "opened_date":   pd.Timestamp("2020-01-01") + pd.to_timedelta(
            acc_rng.integers(0, 1825, n_accounts), unit="D"
        ),
        "balance_cents": 0,  # filled in after lines are generated
    })

    # ── Transactions + lines ──────────────────────────────────────────────────
    # Choose number of transactions per account (1-5 each), with mean ~3.
    n_txs_per_account = tx_rng.integers(1, 6, n_accounts)
    n_total_txs = int(n_txs_per_account.sum())
    tx_rows: list[dict] = []
    line_rows: list[dict] = []
    tx_id = 1
    line_id = 1

    # Each transaction touches 1 primary account; some additionally pull from a
    # secondary "counterparty" account so the ledger is connected.
    for primary_acc_id in np.repeat(ids, n_txs_per_account):
        n_lines = int(line_rng.integers(MIN_LINES_PER_TX, MAX_LINES_PER_TX + 1))
        # At least 1 debit and 1 credit
        n_debits = int(line_rng.integers(1, n_lines))
        n_credits = n_lines - n_debits

        # Total transaction amount in cents (uniform in $5–$5000)
        total_cents = int(amt_rng.integers(500, 500_001))
        debit_amts, credit_amts = _balanced_split(
            total_cents, n_debits, n_credits, amt_rng
        )

        # Each line is assigned to either the primary account or a counterparty
        # account chosen at random.
        def _pick_account() -> int:
            return int(line_rng.choice(ids))

        posted_date = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(tx_rng.integers(0, 730))
        )
        category = str(tx_rng.choice(TRANSACTION_CATEGORIES))
        tx_rows.append({
            "id": tx_id,
            "posted_date": posted_date,
            "category": category,
        })
        # Distribute debits and credits across accounts; biased toward primary
        for amt in debit_amts:
            acc = primary_acc_id if line_rng.random() < 0.6 else _pick_account()
            line_rows.append({
                "id": line_id,
                "transaction_id": tx_id,
                "account_id": int(acc),
                "amount_cents": int(amt),
                "side": "debit",
            })
            line_id += 1
        for amt in credit_amts:
            acc = primary_acc_id if line_rng.random() < 0.6 else _pick_account()
            line_rows.append({
                "id": line_id,
                "transaction_id": tx_id,
                "account_id": int(acc),
                "amount_cents": int(amt),
                "side": "credit",
            })
            line_id += 1
        tx_id += 1

    transactions = pd.DataFrame(tx_rows)
    transaction_lines = pd.DataFrame(line_rows)

    # ── Compute account balances from the line history ────────────────────────
    signed = transaction_lines.assign(
        signed_amount=transaction_lines["amount_cents"]
        * transaction_lines["side"].map({"credit": 1, "debit": -1})
    )
    cumulative = signed.groupby("account_id")["signed_amount"].sum()
    cumulative = cumulative.reindex(accounts["id"], fill_value=0)
    accounts["balance_cents"] = cumulative.values.astype(int)

    S = {
        "accounts": accounts,
        "transactions": transactions,
        "transaction_lines": transaction_lines,
    }
    result = oracle(S)
    assert result["legal"], (
        f"Generator produced illegal state: {result['violations']}"
    )
    return S
