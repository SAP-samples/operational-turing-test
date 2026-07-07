"""
Generate legal relational database states.
All randomness is seeded; uses rules.py for derivation — no rule re-implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import (
    TAX_RATES, DISCOUNT_TABLE,
    OPEN_STATUSES, MAX_OPEN_ORDERS, MIN_ITEMS_PER_ORDER, MAX_ITEMS_PER_ORDER,
    compute_line_total, compute_order_total, oracle,
)

COUNTRIES: list[str] = list(TAX_RATES.keys())
TIERS: list[str] = list(DISCOUNT_TABLE.keys())
N_PRODUCTS: int = 50


def generate_legal_state(
    n_customers: int = 200,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Return a legal relational state S = {customers, orders, order_items}.
    Raises AssertionError if the oracle rejects the result (should never happen).
    """
    rng = np.random.default_rng(seed)
    cust_rng, order_rng, item_rng = rng.spawn(3)

    # ── Customers ─────────────────────────────────────────────────────────────
    ids = np.arange(1, n_customers + 1)
    customers = pd.DataFrame({
        "id":          ids,
        "country":     cust_rng.choice(COUNTRIES, n_customers),
        "tier":        cust_rng.choice(TIERS, n_customers),
        "signup_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(
            cust_rng.integers(0, 1460, n_customers), unit="D"
        ),
    })
    tier_map    = customers.set_index("id")["tier"].to_dict()
    country_map = customers.set_index("id")["country"].to_dict()

    # ── Orders ────────────────────────────────────────────────────────────────
    order_rows: list[dict] = []
    order_id = 1

    for cid in ids:
        n_total = int(order_rng.integers(1, 6))           # 1–5 per customer
        n_open  = int(order_rng.integers(0, min(MAX_OPEN_ORDERS, n_total) + 1))
        n_closed = n_total - n_open
        base_date = pd.Timestamp("2022-01-01") + pd.Timedelta(
            days=int(order_rng.integers(0, 365))
        )

        for _ in range(n_closed):
            if order_rng.random() < 0.7:
                status, prev = "delivered", "shipped"
            else:
                status, prev = "cancelled", "pending"
            order_rows.append({
                "id": order_id, "customer_id": cid,
                "status": status, "prev_status": prev,
                "order_date": base_date, "total": 0.0,
            })
            order_id += 1

        for _ in range(n_open):
            if order_rng.random() < 0.5:
                status, prev = "pending", None
            else:
                status, prev = "shipped", "pending"
            order_rows.append({
                "id": order_id, "customer_id": cid,
                "status": status, "prev_status": prev,
                "order_date": base_date, "total": 0.0,
            })
            order_id += 1

    orders = pd.DataFrame(order_rows)

    # ── Order items + order totals ────────────────────────────────────────────
    item_rows: list[dict] = []
    item_id = 1
    order_totals: dict[int, float] = {}

    for _, ord_row in orders.iterrows():
        oid     = int(ord_row["id"])
        cid     = int(ord_row["customer_id"])
        tier    = tier_map[cid]
        country = country_map[cid]
        n_items = int(item_rng.integers(MIN_ITEMS_PER_ORDER, MAX_ITEMS_PER_ORDER + 1))

        line_totals: list[float] = []
        for _ in range(n_items):
            qty   = int(item_rng.integers(1, 11))
            price = round(float(item_rng.uniform(5.0, 500.0)), 2)
            lt    = compute_line_total(price, qty, tier)
            item_rows.append({
                "id": item_id, "order_id": oid,
                "product_id": int(item_rng.integers(1, N_PRODUCTS + 1)),
                "quantity": qty, "unit_price": price,
                "line_total": round(lt, 6),
            })
            item_id += 1
            line_totals.append(lt)

        order_totals[oid] = round(compute_order_total(line_totals, country), 6)

    orders["total"] = orders["id"].map(order_totals)
    order_items = pd.DataFrame(item_rows)

    S = {"customers": customers, "orders": orders, "order_items": order_items}
    result = oracle(S)
    assert result["legal"], f"Generator produced illegal state: {result['violations']}"
    return S
