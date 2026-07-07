"""
Marginal-preserving corruptions for the banking schema. Four strategies, one
per rule family. Each violates exactly one rule and preserves all 1- and 2-way
column-value marginals to TV<0.02.

Mirrors the high-level design of src/tf_pilot/corruptor.py: corruptions either
exclude the affected column from the marginal feature vector by construction
(fk_break), permute within equivalence classes (cardinality), or apply zero-sum
noise to a numeric column (balance, account-balance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import MAX_LINES_PER_TX, MIN_LINES_PER_TX, oracle

VIOLATION_TYPES: frozenset[str] = frozenset({
    "fk_break",
    "cardinality_break",
    "balance_break",
    "account_balance_break",
})

RULE_FOR_VIOLATION: dict[str, str] = {
    "fk_break":              "fk_integrity",
    "cardinality_break":     "cardinality",
    "balance_break":         "double_entry",
    "account_balance_break": "account_balance",
}


def corrupt_to_illegal(
    S: dict[str, pd.DataFrame],
    violation_type: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if violation_type not in VIOLATION_TYPES:
        raise ValueError(
            f"Unknown violation type {violation_type!r}. Must be one of {VIOLATION_TYPES}."
        )
    rng = np.random.default_rng(seed)
    S_prime = {k: df.copy() for k, df in S.items()}

    dispatch = {
        "fk_break":              _corrupt_fk,
        "cardinality_break":     _corrupt_cardinality,
        "balance_break":         _corrupt_balance,
        "account_balance_break": _corrupt_account_balance,
    }
    S_prime = dispatch[violation_type](S_prime, rng)

    result = oracle(S_prime)
    target_rule = RULE_FOR_VIOLATION[violation_type]
    assert target_rule in result["violations"], (
        f"Corruption '{violation_type}' did not fire rule '{target_rule}'. "
        f"Oracle violations: {result['violations']}"
    )
    return S_prime


# ── Corruption strategies ─────────────────────────────────────────────────────

def _corrupt_fk(S: dict, rng: np.random.Generator) -> dict:
    """Replace ~5% of transaction_lines.transaction_id with out-of-range IDs.
    The transaction_id column is excluded from the marginal feature vector
    by construction, so TV=0.
    """
    lines = S["transaction_lines"].copy()
    max_valid = int(S["transactions"]["id"].max())

    n_break = max(1, int(len(lines) * 0.05))
    idx = rng.choice(len(lines), size=n_break, replace=False)
    invalid_ids = (max_valid + 1 + rng.integers(1, 1000, size=n_break)).astype(int)
    lines.iloc[idx, lines.columns.get_loc("transaction_id")] = invalid_ids

    S["transaction_lines"] = lines
    return S


def _corrupt_cardinality(S: dict, rng: np.random.Generator) -> dict:
    """Insert balanced (debit, credit) pairs into a single target transaction
    until it exceeds MAX_LINES_PER_TX. Each inserted pair has matching amounts
    (debit X cents and credit X cents to the SAME account), so:

      • Double-entry per transaction stays balanced (we add equal Σdebit and Σcredit).
      • Account-balance per account stays exact (the inserted debit and credit
        cancel for the chosen account).
      • The amount_cents column gains 2 entries per pair, drawn from the
        existing distribution (we sample from it). With $N{\\sim}1000$ existing
        lines and a few inserted pairs, the column histogram TV is $\\le$ 4/N.
      • The side column gains balanced (debit, credit) so its 1-way marginal is
        unchanged.

    Only the cardinality rule fires: target transaction has > MAX_LINES_PER_TX.
    """
    lines = S["transaction_lines"].copy()

    counts = lines.groupby("transaction_id").size()
    # Pick a target near the cardinality limit so we add few pairs.
    candidate_targets = counts[counts >= MAX_LINES_PER_TX - 1].index.tolist()
    if not candidate_targets:
        # Fall back to any transaction; we'll just add more pairs.
        candidate_targets = counts.index.tolist()

    target_tx = int(rng.choice(candidate_targets))
    target_count = int(counts[target_tx])

    # Number of pairs to insert: enough to push target above MAX_LINES_PER_TX.
    n_pairs = max(1, ((MAX_LINES_PER_TX - target_count) // 2) + 1)

    next_line_id = int(lines["id"].max()) + 1
    account_ids = S["accounts"]["id"].values
    existing_amounts = lines["amount_cents"].values

    new_rows = []
    for _ in range(n_pairs):
        # Sample an amount from the existing column distribution to keep the
        # 1-way amount_cents marginal close.
        amt = int(rng.choice(existing_amounts))
        # Pick an account that already has lines in this transaction if possible
        # so the joint (transaction_id × account_id) marginal also stays close.
        existing_accts_in_tx = lines.loc[
            lines["transaction_id"] == target_tx, "account_id"
        ].values
        if len(existing_accts_in_tx) > 0:
            acc = int(rng.choice(existing_accts_in_tx))
        else:
            acc = int(rng.choice(account_ids))

        new_rows.append({
            "id": next_line_id,
            "transaction_id": int(target_tx),
            "account_id": acc,
            "amount_cents": amt,
            "side": "debit",
        })
        next_line_id += 1
        new_rows.append({
            "id": next_line_id,
            "transaction_id": int(target_tx),
            "account_id": acc,
            "amount_cents": amt,
            "side": "credit",
        })
        next_line_id += 1

    lines = pd.concat([lines, pd.DataFrame(new_rows)], ignore_index=True)
    S["transaction_lines"] = lines
    return S


def _corrupt_balance(S: dict, rng: np.random.Generator) -> dict:
    """Break double-entry on one transaction while preserving the global
    amount_cents column histogram.

    Pick two distinct transactions T_a and T_b that each have ≥1 debit row.
    Add +δ to one debit's amount in T_a and -δ from one debit's amount in T_b.
    Net effect on the amount column: zero-sum (one row +δ, one row -δ → mean
    unchanged, histogram shifts by 2/N). T_a now has Σ(debit) ≠ Σ(credit), as
    does T_b. Both fire the double-entry rule, and the column-value 1-way
    marginal is approximately preserved.

    Crucially, we also adjust the corresponding account.balance_cents by ±δ
    to keep the account-balance rule from firing as a side-effect; otherwise
    a balance_break would also fire account_balance, conflating the two
    rule families.
    """
    lines = S["transaction_lines"].copy()
    accounts = S["accounts"].copy()

    debit_rows = lines[lines["side"] == "debit"]
    if len(debit_rows) < 2:
        raise RuntimeError("Need at least 2 debit rows for balance_break.")

    # Pick two distinct debits in distinct transactions
    tx_ids = debit_rows["transaction_id"].unique()
    if len(tx_ids) < 2:
        raise RuntimeError("Need debits in at least 2 distinct transactions.")

    chosen_tx_a, chosen_tx_b = rng.choice(tx_ids, size=2, replace=False)
    row_a = int(debit_rows[debit_rows["transaction_id"] == chosen_tx_a].index[0])
    row_b = int(debit_rows[debit_rows["transaction_id"] == chosen_tx_b].index[0])

    # Magnitude δ chosen well above 0 (so the rule fires) and well below the
    # amount-column histogram bin width.
    delta = max(1, int(lines["amount_cents"].mean() * 0.001))

    lines.at[row_a, "amount_cents"] = int(lines.at[row_a, "amount_cents"]) + delta
    lines.at[row_b, "amount_cents"] = int(lines.at[row_b, "amount_cents"]) - delta

    # Compensate the affected accounts so account-balance rule does NOT fire.
    # Row a's account had its debit increased by δ → account balance decreased by δ → bump balance up.
    # Row b's account had its debit decreased by δ → account balance increased by δ → bump balance down.
    acc_a = int(lines.at[row_a, "account_id"])
    acc_b = int(lines.at[row_b, "account_id"])
    accounts.loc[accounts["id"] == acc_a, "balance_cents"] = (
        accounts.loc[accounts["id"] == acc_a, "balance_cents"] - delta
    )
    accounts.loc[accounts["id"] == acc_b, "balance_cents"] = (
        accounts.loc[accounts["id"] == acc_b, "balance_cents"] + delta
    )

    S["transaction_lines"] = lines
    S["accounts"] = accounts
    return S


def _corrupt_account_balance(S: dict, rng: np.random.Generator) -> dict:
    """Break account-balance consistency by adding zero-sum noise to a
    handful of accounts.balance_cents entries. Keeps the column histogram
    essentially unchanged (TV ≈ 0.001-0.003) while breaking the balance rule.
    """
    accounts = S["accounts"].copy()
    n_acc = len(accounts)

    n_break = max(2, int(n_acc * 0.10))
    n_break += n_break % 2  # even, for pairing

    idx = rng.choice(n_acc, size=n_break, replace=False)
    mean_balance = float(np.abs(accounts["balance_cents"]).mean())
    scale = max(1.0, mean_balance * 0.001)

    half = n_break // 2
    magnitudes = rng.uniform(scale, scale * 2.0, size=half)
    noise = np.concatenate([magnitudes, -magnitudes])
    rng.shuffle(noise)

    balances = accounts["balance_cents"].values.copy().astype(np.int64)
    balances[idx] += noise.astype(np.int64)
    accounts["balance_cents"] = balances

    S["accounts"] = accounts
    return S
