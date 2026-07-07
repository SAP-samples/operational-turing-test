"""
Corrupt a legal state by violating exactly one rule.

Each corruption strategy uses permutation within equivalence classes where
possible, so that 1- and 2-way marginals over row values are preserved.
The specific TV-distance guarantee is verified in validate_construction.py.

Postcondition (enforced by assertion):
    oracle(corrupt_to_illegal(S, vt, seed))["violations"] contains the
    rule that corresponds to vt, and no others are required to fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import (
    LEGAL_TRANSITIONS, OPEN_STATUSES, MAX_OPEN_ORDERS, MAX_ITEMS_PER_ORDER,
    get_tax, oracle,
)

VIOLATION_TYPES: frozenset[str] = frozenset({
    "fk_break",
    "cardinality_break",
    "derivation_break",
    "transition_break",
})

# Maps each violation type to the rule name it must fire in the oracle.
RULE_FOR_VIOLATION: dict[str, str] = {
    "fk_break":          "fk_integrity",
    "cardinality_break": "cardinality",
    "derivation_break":  "derivation",
    "transition_break":  "transition",
}


def corrupt_to_illegal(
    S: dict[str, pd.DataFrame],
    violation_type: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """
    Return S' that violates exactly the rule targeted by violation_type.

    Args:
        S:              A legal state produced by generate_legal_state.
        violation_type: One of VIOLATION_TYPES.
        seed:           RNG seed for reproducibility.

    Returns:
        S' = deep copy of S with exactly one rule violated.

    Raises:
        ValueError if violation_type is unrecognised.
        AssertionError if the targeted rule is not violated after corruption
        (indicates a bug, not a design choice).
    """
    if violation_type not in VIOLATION_TYPES:
        raise ValueError(
            f"Unknown violation type {violation_type!r}. Must be one of {VIOLATION_TYPES}."
        )

    rng     = np.random.default_rng(seed)
    S_prime = {k: df.copy() for k, df in S.items()}

    dispatch = {
        "fk_break":          _corrupt_fk,
        "cardinality_break": _corrupt_cardinality,
        "derivation_break":  _corrupt_derivation,
        "transition_break":  _corrupt_transition,
    }
    S_prime = dispatch[violation_type](S_prime, rng)

    result      = oracle(S_prime)
    target_rule = RULE_FOR_VIOLATION[violation_type]
    assert target_rule in result["violations"], (
        f"Corruption '{violation_type}' did not fire rule '{target_rule}'. "
        f"Oracle violations: {result['violations']}"
    )
    return S_prime


# ── Corruption strategies ─────────────────────────────────────────────────────

def _corrupt_fk(S: dict, rng: np.random.Generator) -> dict:
    """
    Replace ~5% of orders.customer_id with out-of-range IDs.

    Marginal impact: the 1-way customer_id distribution shifts by the
    replacement fraction. The TV distance scales as n_break / n_orders,
    bounded well below tau=0.02 for typical states.
    """
    orders    = S["orders"].copy()
    max_valid = int(S["customers"]["id"].max())

    n_break = max(1, int(len(orders) * 0.05))
    idx     = rng.choice(len(orders), size=n_break, replace=False)

    invalid_ids = (max_valid + 1 + rng.integers(1, 1000, size=n_break)).astype(int)
    orders.iloc[idx, orders.columns.get_loc("customer_id")] = invalid_ids

    S["orders"] = orders
    return S


def _corrupt_cardinality(S: dict, rng: np.random.Generator) -> dict:
    """
    Move items between orders of the SAME customer so that one order exceeds
    MAX_ITEMS_PER_ORDER.

    Same-customer restriction: guarantees tier consistency, so every item's
    existing line_total remains derivation-valid after reassignment.  Only
    the order_id column in order_items changes; all value columns are untouched.

    Eligibility is checked precisely: a customer qualifies only if the total
    movable items (each donor can spare count-1) is enough to push the target
    order past MAX_ITEMS_PER_ORDER.  This avoids the edge case where customers
    have many 1-item orders (none can donate).
    """
    items  = S["order_items"].copy()
    orders = S["orders"].copy()

    cust_orders = orders.groupby("customer_id")["id"].apply(list).to_dict()
    counts      = items.groupby("order_id").size().to_dict()

    # Precise eligibility: movable items from donors must cover the shortfall.
    eligible_customers: list[int] = []
    for cid, oids in cust_orders.items():
        if len(oids) < 2:
            continue
        cnt      = {o: counts.get(o, 0) for o in oids}
        target   = min(oids, key=lambda o: cnt[o])
        needed   = MAX_ITEMS_PER_ORDER - cnt[target] + 1
        movable  = sum(max(0, cnt[o] - 1) for o in oids if o != target)
        if movable >= needed:
            eligible_customers.append(cid)

    if not eligible_customers:
        raise RuntimeError(
            "No customer can push an order past the item limit via same-customer permutation."
        )

    target_cust = int(rng.choice(eligible_customers))
    cust_oids   = cust_orders[target_cust]
    cnt         = {o: counts.get(o, 0) for o in cust_oids}
    target_oid  = min(cust_oids, key=lambda o: cnt[o])
    current     = cnt[target_oid]
    needed      = MAX_ITEMS_PER_ORDER - current + 1

    moved = 0
    for donor_oid in rng.permutation([o for o in cust_oids if o != target_oid]):
        donor_idx = items[items["order_id"] == donor_oid].index.tolist()
        to_move   = donor_idx[:-1]   # keep ≥1 item in every donor
        for row_idx in to_move:
            items.at[row_idx, "order_id"] = target_oid
            moved += 1
            if moved >= needed:
                break
        if moved >= needed:
            break

    assert moved >= needed, f"moved={moved} < needed={needed} (eligibility check bug)"

    # Recompute totals for this customer's orders (same country → same tax rate).
    country    = str(S["customers"].loc[S["customers"]["id"] == target_cust, "country"].iloc[0])
    tax_factor = 1.0 + get_tax(country)
    subtotals  = items.groupby("order_id")["line_total"].sum()

    for oid in cust_oids:
        orders.loc[orders["id"] == oid, "total"] = round(
            float(subtotals.get(oid, 0.0)) * tax_factor, 6
        )

    S["order_items"] = items
    S["orders"]      = orders
    return S


def _corrupt_derivation(S: dict, rng: np.random.Generator) -> dict:
    """
    Perturb orders.total by a tiny paired amount while leaving all line_totals
    and all order_items values untouched.

    This targets rule_derivation condition 2:
        orders.total ≠ sum(order_items.line_total) * (1 + tax(country))

    Why perturb orders.total rather than line_totals?
      Noise on line_total shifts the (product_id × line_total) 2-way histogram
      joint above tau=0.02.  Perturbing orders.total by O(0.1% of mean total)
      is negligible relative to the verifier's histogram bin width
      (bin_width ≈ range/20 >> perturbation), giving TV ≈ 0 on all marginals
      while still firing the rule (tol=1e-4 << perturbation).

    Paired noise: zero-sum across affected rows → 1-way mean of orders.total
    is exactly preserved.
    """
    orders   = S["orders"].copy()
    n_orders = len(orders)

    n_break  = max(2, int(n_orders * 0.10))
    n_break += n_break % 2  # ensure even for pairing

    idx          = rng.choice(n_orders, size=n_break, replace=False)
    mean_total   = float(orders["total"].mean())
    # Scale is 0.1% of mean total — far above rule tol (1e-4), far below bin width
    scale        = max(1.0, mean_total * 0.001)

    half       = n_break // 2
    magnitudes = rng.uniform(scale, scale * 2.0, size=half)
    noise      = np.concatenate([magnitudes, -magnitudes])
    rng.shuffle(noise)

    totals          = orders["total"].values.copy().astype(float)
    totals[idx]    += noise
    orders["total"] = totals

    # order_items are completely untouched → only condition 2 of rule_derivation fires
    S["orders"] = orders
    return S


def _corrupt_transition(S: dict, rng: np.random.Generator) -> dict:
    """
    Swap status values between one 'delivered' and one initial 'pending' order.

    After the swap:
      - former-delivered: prev_status='shipped', status='pending'
        → transition (shipped, pending) ∉ LEGAL_TRANSITIONS
      - former-pending:   prev_status=None, status='delivered'
        → (None, delivered) has no legal path

    The 1-way status marginal is exactly preserved (one-for-one swap).
    """
    orders = S["orders"].copy()

    delivered_idx = orders[orders["status"] == "delivered"].index.tolist()
    # Only initial pending orders (prev_status is None) for a clean swap
    initial_pending_idx = orders[
        (orders["status"] == "pending") & (orders["prev_status"].isna())
    ].index.tolist()

    if delivered_idx and initial_pending_idx:
        d_idx = int(rng.choice(delivered_idx))
        p_idx = int(rng.choice(initial_pending_idx))
        # Swap only the status column; prev_status stays, creating illegal pairs
        orders.at[d_idx, "status"], orders.at[p_idx, "status"] = (
            orders.at[p_idx, "status"],
            orders.at[d_idx, "status"],
        )
    else:
        # Fallback: directly manufacture an illegal (prev, curr) pair
        shipped_idx = orders[orders["status"] == "shipped"].index.tolist()
        if shipped_idx:
            chosen = int(rng.choice(shipped_idx))
            orders.at[chosen, "status"]      = "pending"
            orders.at[chosen, "prev_status"] = "shipped"  # (shipped, pending) ∉ LEGAL_TRANSITIONS
        else:
            # Last resort: set an unknown prev_status
            chosen = int(rng.choice(orders.index.tolist()))
            orders.at[chosen, "prev_status"] = "delivered"
            orders.at[chosen, "status"]      = "pending"

    S["orders"] = orders
    return S
