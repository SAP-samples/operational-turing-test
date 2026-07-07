"""
Operational rules for the synthetic relational database.

All rules are pure functions: (state: dict[str, DataFrame]) -> bool.
No rule logic lives anywhere else in the codebase.
"""
from __future__ import annotations

import pandas as pd

# ── Operational data ──────────────────────────────────────────────────────────

TAX_RATES: dict[str, float] = {
    "US": 0.08,
    "DE": 0.19,
    "GB": 0.20,
    "FR": 0.20,
    "JP": 0.10,
    "OTHER": 0.15,
}

# {tier: {min_quantity_threshold: discount_rate}}
DISCOUNT_TABLE: dict[str, dict[int, float]] = {
    "gold":   {1: 0.10, 5: 0.15, 10: 0.20},
    "silver": {1: 0.05, 5: 0.10, 10: 0.12},
    "bronze": {1: 0.00, 5: 0.05, 10: 0.07},
}

VALID_STATUSES: frozenset[str] = frozenset({"pending", "shipped", "delivered", "cancelled"})

# Only these prev→curr transitions are permitted.
# Initial orders have prev_status=None and are not checked here.
LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset({
    ("pending", "shipped"),
    ("shipped", "delivered"),
    ("pending", "cancelled"),
})

OPEN_STATUSES: frozenset[str] = frozenset({"pending", "shipped"})
MAX_OPEN_ORDERS: int = 3
MIN_ITEMS_PER_ORDER: int = 1
MAX_ITEMS_PER_ORDER: int = 20


# ── Derivation helpers ────────────────────────────────────────────────────────

def get_discount(tier: str, quantity: int) -> float:
    """Discount rate for a customer tier and item quantity (step function)."""
    brackets = DISCOUNT_TABLE.get(tier, {})
    rate = 0.0
    for min_qty in sorted(brackets, reverse=True):
        if quantity >= min_qty:
            rate = brackets[min_qty]
            break
    return rate


def get_tax(country: str) -> float:
    """Tax rate for a country."""
    return TAX_RATES.get(country, TAX_RATES["OTHER"])


def compute_line_total(unit_price: float, quantity: int, tier: str) -> float:
    """line_total = quantity * unit_price * (1 - discount(tier, quantity))"""
    return quantity * unit_price * (1.0 - get_discount(tier, quantity))


def compute_order_total(line_totals: list[float], country: str) -> float:
    """order.total = sum(line_totals) * (1 + tax(country))"""
    return sum(line_totals) * (1.0 + get_tax(country))


# ── Rules ─────────────────────────────────────────────────────────────────────

def rule_fk_integrity(S: dict[str, pd.DataFrame]) -> bool:
    """Rule 1: Every FK resolves to a row in the referenced table."""
    valid_customer_ids = set(S["customers"]["id"])
    valid_order_ids = set(S["orders"]["id"])
    if not set(S["orders"]["customer_id"]).issubset(valid_customer_ids):
        return False
    if not set(S["order_items"]["order_id"]).issubset(valid_order_ids):
        return False
    return True


def rule_cardinality(S: dict[str, pd.DataFrame]) -> bool:
    """Rule 2: 1–20 items per order; ≤3 open (pending/shipped) orders per customer."""
    items_per_order = S["order_items"].groupby("order_id").size()
    if len(items_per_order) > 0:
        if items_per_order.min() < MIN_ITEMS_PER_ORDER:
            return False
        if items_per_order.max() > MAX_ITEMS_PER_ORDER:
            return False
    open_orders = S["orders"][S["orders"]["status"].isin(OPEN_STATUSES)]
    open_per_customer = open_orders.groupby("customer_id").size()
    if (open_per_customer > MAX_OPEN_ORDERS).any():
        return False
    return True


def rule_derivation(S: dict[str, pd.DataFrame], tol: float = 1e-4) -> bool:
    """Rule 3: line_total and orders.total match their derivation formulas."""
    cust = S["customers"][["id", "country", "tier"]].rename(columns={"id": "customer_id"})
    ord_ = S["orders"][["id", "customer_id", "total"]].rename(columns={"id": "order_id"})
    ord_cust = ord_.merge(cust, on="customer_id", how="left")

    items = S["order_items"].merge(
        ord_cust[["order_id", "tier", "country"]], on="order_id", how="left"
    )

    # Check line_total
    expected_lt = items.apply(
        lambda r: compute_line_total(float(r["unit_price"]), int(r["quantity"]), str(r["tier"])),
        axis=1,
    )
    if (abs(items["line_total"] - expected_lt) > tol).any():
        return False

    # Check orders.total = sum(line_totals) * (1 + tax)
    order_subtotals = items.groupby("order_id")["line_total"].sum()
    for _, row in ord_cust.iterrows():
        subtotal = float(order_subtotals.get(row["order_id"], 0.0))
        expected = subtotal * (1.0 + get_tax(str(row["country"])))
        if abs(float(row["total"]) - expected) > tol:
            return False
    return True


def rule_transition(S: dict[str, pd.DataFrame]) -> bool:
    """Rule 4: status values are valid; every recorded prev→curr transition is legal."""
    if not set(S["orders"]["status"]).issubset(VALID_STATUSES):
        return False
    if "prev_status" not in S["orders"].columns:
        return True
    has_prev = S["orders"]["prev_status"].notna()
    transitions = list(zip(
        S["orders"].loc[has_prev, "prev_status"],
        S["orders"].loc[has_prev, "status"],
    ))
    return all(t in LEGAL_TRANSITIONS for t in transitions)


# ── Oracle ────────────────────────────────────────────────────────────────────

_RULES: dict[str, object] = {
    "fk_integrity": rule_fk_integrity,
    "cardinality":  rule_cardinality,
    "derivation":   rule_derivation,
    "transition":   rule_transition,
}


def oracle(S: dict[str, pd.DataFrame]) -> dict:
    """
    Run all rules. Returns {"legal": bool, "violations": list[str]}.

    This is the identifiability upper bound: it achieves perfect accuracy
    by executing the operational rules directly, not by reading row values.
    """
    results = {name: fn(S) for name, fn in _RULES.items()}
    return {
        "legal": all(results.values()),
        "violations": [name for name, passed in results.items() if not passed],
    }
