"""
State featurization for the XGBoost baselines.

state_to_values_vector  — aggregate statistics only; no schema, no rule awareness.
state_to_relation_vector — values + relational structure, joins, and FK topology;
                           no executable rule residuals or legality predicates.
state_to_schema_vector  — values vector + cheap operational features derived from rules.py.

Both produce fixed-length numpy arrays regardless of state size, so they can
serve as rows in a standard sklearn/XGBoost dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import (
    LEGAL_TRANSITIONS, OPEN_STATUSES, MAX_OPEN_ORDERS, MAX_ITEMS_PER_ORDER,
    DISCOUNT_TABLE, TAX_RATES,
    get_tax,
)

# ── vectorized discount lookup ────────────────────────────────────────────────
# Build a fast (tier, qty_bracket) → rate table so _schema_features avoids
# a pandas row-wise apply (which dominates runtime).
_TIER_ORDER    = sorted(DISCOUNT_TABLE.keys())           # consistent ordering
_TIER_IDX      = {t: i for i, t in enumerate(_TIER_ORDER)}
_QTY_BRACKETS  = sorted({min_qty for tbl in DISCOUNT_TABLE.values() for min_qty in tbl})

def _build_discount_matrix() -> np.ndarray:
    n_tiers    = len(_TIER_ORDER)
    n_brackets = len(_QTY_BRACKETS)
    mat        = np.zeros((n_tiers, n_brackets))
    for t, tier in enumerate(_TIER_ORDER):
        brackets = DISCOUNT_TABLE[tier]
        for b, min_qty in enumerate(_QTY_BRACKETS):
            # Rate for this bracket = highest rate where threshold ≤ min_qty
            rate = 0.0
            for threshold in sorted(brackets, reverse=True):
                if min_qty >= threshold:
                    rate = brackets[threshold]
                    break
            mat[t, b] = rate
    return mat

_DISCOUNT_MATRIX   = _build_discount_matrix()
_QTY_THRESHOLDS    = np.array(_QTY_BRACKETS)    # for np.searchsorted


def _vec_discount(tier_col: pd.Series, qty_col: pd.Series) -> np.ndarray:
    """Vectorized discount rates for arrays of tier strings and integer quantities."""
    tier_idx = np.array([_TIER_IDX.get(str(t), 0) for t in tier_col], dtype=int)
    qty_arr  = qty_col.values.astype(int)
    # bracket = index of the highest threshold ≤ qty
    bracket  = np.searchsorted(_QTY_THRESHOLDS, qty_arr, side="right") - 1
    bracket  = np.clip(bracket, 0, len(_QTY_THRESHOLDS) - 1)
    return _DISCOUNT_MATRIX[tier_idx, bracket]


# ── values-only features ──────────────────────────────────────────────────────

def _num_stats(series: pd.Series) -> list[float]:
    """6 aggregate statistics for a numeric series."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return [0.0] * 6
    return [
        float(s.mean()), float(s.std(ddof=0)),
        float(s.min()),  float(s.max()),
        float(s.quantile(0.25)), float(s.quantile(0.75)),
    ]


def _cat_entropy(series: pd.Series) -> list[float]:
    """2 features for a categorical series: entropy and mode frequency."""
    vc = series.value_counts(normalize=True)
    if len(vc) == 0:
        return [0.0, 0.0]
    probs   = vc.values
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return [entropy, float(probs.max())]


def _safe_num_stats(values) -> list[float]:
    """Numeric stats with NaN/inf cleaned for model-facing feature vectors."""
    stats = _num_stats(pd.Series(values))
    return [0.0 if not np.isfinite(x) else float(x) for x in stats]


def _frequency_vector(series: pd.Series, categories: list) -> list[float]:
    """Fixed-order normalized frequencies for categorical values."""
    if len(series) == 0:
        return [0.0] * len(categories)
    vc = series.value_counts(normalize=True, dropna=False)
    return [float(vc.get(cat, 0.0)) for cat in categories]


def state_to_values_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Featurize a state as a flat vector of aggregate statistics of column values.
    No schema information, no FK or rule awareness.

    Strictly column-level aggregates: the identifiability claim is that any
    classifier restricted to k-way marginals of these column values has Bayes
    error ≥ 0.5 - O(tau).  We therefore exclude any feature that aggregates
    over relational groups (e.g. items-per-order counts), as those encode
    structural/schema information and would violate the claim for
    cardinality_break.  Such structural features appear in state_to_schema_vector.

    Feature groups (in order):
      customers  : tier entropy/mode, country entropy/mode
      orders     : status entropy/mode, total stats (×6), row count
      order_items: quantity stats (×6), unit_price stats (×6),
                   line_total stats (×6), product_id entropy/mode
      cross-table: pearson(quantity, unit_price), pearson(unit_price, line_total)
    """
    cust  = S["customers"]
    ord_  = S["orders"]
    items = S["order_items"]

    feats: list[float] = []

    # customers — column value distributions
    feats += _cat_entropy(cust["tier"])
    feats += _cat_entropy(cust["country"])

    # orders — column value distributions
    feats += _cat_entropy(ord_["status"])
    feats += _num_stats(ord_["total"])
    feats += [float(len(ord_))]

    # order_items — column value distributions (individual rows, not grouped)
    feats += _num_stats(items["quantity"])
    feats += _num_stats(items["unit_price"])
    feats += _num_stats(items["line_total"])
    feats += _cat_entropy(items["product_id"].astype(str))

    # within-items correlations (column-pair statistics, not grouped)
    qty_p = items[["quantity", "unit_price"]].dropna()
    feats += [float(qty_p.corr().iloc[0, 1])] if len(qty_p) > 1 else [0.0]
    up_lt = items[["unit_price", "line_total"]].dropna()
    feats += [float(up_lt.corr().iloc[0, 1])] if len(up_lt) > 1 else [0.0]

    return np.array(feats, dtype=np.float64)


# ── relation-only features ───────────────────────────────────────────────────

def _relation_features(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Relational-structure features that a lightweight relation-aware model could
    exploit: joins, FK coverage, group sizes, and local neighborhood summaries.

    Deliberately excluded:
      - derivation residuals involving discount/tax formulas
      - thresholded legality checks such as count(items_per_order > 20)
      - explicit allowed-transition predicates

    This makes the feature tier stronger than values-only access but weaker than
    executable operational grounding.
    """
    cust  = S["customers"]
    ord_  = S["orders"]
    items = S["order_items"]

    feats: list[float] = []

    n_customers = max(len(cust), 1)
    n_orders    = max(len(ord_), 1)
    n_items     = max(len(items), 1)

    feats += [
        float(len(cust)),
        float(len(ord_)),
        float(len(items)),
        float(len(ord_) / n_customers),
        float(len(items) / n_orders),
        float(len(items) / n_customers),
    ]

    # FK columns become visible at the relation tier.
    feats += _safe_num_stats(ord_["customer_id"])
    feats += _safe_num_stats(items["order_id"])

    valid_customer_ids = set(cust["id"])
    valid_order_ids    = set(ord_["id"])
    feats += [
        float(ord_["customer_id"].isin(valid_customer_ids).mean()) if len(ord_) else 0.0,
        float(items["order_id"].isin(valid_order_ids).mean()) if len(items) else 0.0,
        float(ord_["customer_id"].nunique()),
        float(items["order_id"].nunique()),
    ]

    # Degree distributions over the FK graph. Include zero-degree valid nodes.
    orders_per_valid_customer = (
        ord_[ord_["customer_id"].isin(valid_customer_ids)]
        .groupby("customer_id")
        .size()
        .reindex(cust["id"], fill_value=0)
    )
    items_per_valid_order = (
        items[items["order_id"].isin(valid_order_ids)]
        .groupby("order_id")
        .size()
        .reindex(ord_["id"], fill_value=0)
    )
    feats += _safe_num_stats(orders_per_valid_customer)
    feats += _safe_num_stats(items_per_valid_order)

    # Items per customer via orders -> customers, without applying rule thresholds.
    order_to_customer = ord_.set_index("id")["customer_id"]
    item_customer_ids = items["order_id"].map(order_to_customer)
    valid_item_customers = item_customer_ids.dropna().astype(int)
    valid_item_customers = valid_item_customers[valid_item_customers.isin(valid_customer_ids)]
    items_per_customer = valid_item_customers.value_counts().reindex(cust["id"], fill_value=0)
    feats += _safe_num_stats(items_per_customer)

    # Per-status customer neighborhoods. This is structure conditioned on row
    # values, but not the "open order" rule or its threshold.
    status_categories = ["pending", "shipped", "delivered", "cancelled"]
    feats += _frequency_vector(ord_["status"], status_categories)
    for status in status_categories:
        per_customer = (
            ord_[(ord_["status"] == status) & (ord_["customer_id"].isin(valid_customer_ids))]
            .groupby("customer_id")
            .size()
            .reindex(cust["id"], fill_value=0)
        )
        feats += _safe_num_stats(per_customer)

    # Transition-pair distribution only. No lookup against LEGAL_TRANSITIONS.
    prev_categories = ["NONE", "pending", "shipped", "delivered", "cancelled"]
    prev_values = ord_["prev_status"].where(ord_["prev_status"].notna(), None)
    pair_values = [
        f"{'NONE' if p is None else p}->{s}"
        for p, s in zip(prev_values, ord_["status"])
    ]
    pair_series = pd.Series(pair_values)
    pair_categories = [f"{p}->{s}" for p in prev_categories for s in status_categories]
    feats += _frequency_vector(pair_series, pair_categories)

    # Joined context summaries: customer attributes projected to orders/items.
    cust_lookup = cust.set_index("id")[["country", "tier"]]
    order_country = ord_["customer_id"].map(cust_lookup["country"])
    order_tier    = ord_["customer_id"].map(cust_lookup["tier"])
    country_categories = sorted(TAX_RATES.keys())
    tier_categories    = sorted(DISCOUNT_TABLE.keys())
    feats += _frequency_vector(order_country, country_categories)
    feats += _frequency_vector(order_tier, tier_categories)

    for attr_values, categories in ((order_country, country_categories), (order_tier, tier_categories)):
        for cat in categories:
            totals = ord_.loc[attr_values == cat, "total"]
            feats += _safe_num_stats(totals)

    item_order_country = items["order_id"].map(ord_.set_index("id")["customer_id"]).map(cust_lookup["country"])
    item_order_tier    = items["order_id"].map(ord_.set_index("id")["customer_id"]).map(cust_lookup["tier"])
    feats += _frequency_vector(item_order_country, country_categories)
    feats += _frequency_vector(item_order_tier, tier_categories)

    for attr_values, categories in ((item_order_country, country_categories), (item_order_tier, tier_categories)):
        for cat in categories:
            qty = items.loc[attr_values == cat, "quantity"]
            feats += _safe_num_stats(qty)

    arr = np.array(feats, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def state_to_relation_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Values-only features plus relation-only structural features."""
    return np.concatenate([state_to_values_vector(S), _relation_features(S)])


# ── schema features (added on top of values-only) ────────────────────────────

def _schema_features(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Cheap operational features derived from rules.py semantics — but NOT
    calling oracle().  Fully vectorized; no pandas row-wise apply.

    Each feature is 0 in a legal state and >0 in an illegal one (or vice-versa).
    """
    cust  = S["customers"]
    ord_  = S["orders"]
    items = S["order_items"]

    valid_cust_ids  = set(cust["id"])
    valid_order_ids = set(ord_["id"])

    # ── Rule 1 proxies ────────────────────────────────────────────────────────
    n_orphan_fk_orders = int((~ord_["customer_id"].isin(valid_cust_ids)).sum())
    n_orphan_fk_items  = int((~items["order_id"].isin(valid_order_ids)).sum())

    # ── Rule 2 proxies ────────────────────────────────────────────────────────
    items_per_order          = items.groupby("order_id").size()
    n_orders_over_item_limit = int((items_per_order > MAX_ITEMS_PER_ORDER).sum())
    open_orders              = ord_[ord_["status"].isin(OPEN_STATUSES)]
    open_per_cust            = open_orders.groupby("customer_id").size()
    n_cust_over_open_limit   = int((open_per_cust > MAX_OPEN_ORDERS).sum())

    # ── Rule 3 proxies (vectorized) ───────────────────────────────────────────
    # Join: items → order → customer to get tier and country per item
    cust_lkp  = cust.set_index("id")[["tier", "country"]]
    ord_lkp   = ord_.set_index("id")["customer_id"]
    item_cids = items["order_id"].map(ord_lkp)            # item → customer_id
    item_tier = item_cids.map(cust_lkp["tier"])
    item_cty  = item_cids.map(cust_lkp["country"])

    # Vectorized expected line_total
    discounts   = _vec_discount(item_tier, items["quantity"])
    expected_lt = items["quantity"].values * items["unit_price"].values * (1.0 - discounts)
    residuals   = np.abs(items["line_total"].values.astype(float) - expected_lt)
    max_abs_lt_residual = float(residuals.max()) if len(residuals) > 0 else 0.0

    # Vectorized expected order total
    # tax rate per item (via country)
    country_arr = item_cty.values
    tax_rates   = np.array([get_tax(str(c)) for c in country_arr])
    # subtotal per order = sum(line_total) (use actual values, not expected)
    items_tmp   = items.assign(_oid=items["order_id"].values,
                               _lt=items["line_total"].values.astype(float),
                               _tax=tax_rates)
    # expected order total = sum(lt_i) * (1 + tax) — tax is same for all items of same order
    grouped_items = items_tmp.groupby("_oid")
    try:
        ord_expected = grouped_items.apply(
            lambda g: g["_lt"].sum() * (1.0 + g["_tax"].iloc[0]),
            include_groups=False,
        )
    except TypeError:
        # pandas<2.2 does not support include_groups on GroupBy.apply.
        ord_expected = grouped_items.apply(lambda g: g["_lt"].sum() * (1.0 + g["_tax"].iloc[0]))
    ord_actual   = ord_.set_index("id")["total"]
    common_ids   = ord_expected.index.intersection(ord_actual.index)
    order_resids = np.abs(
        ord_expected.loc[common_ids].values - ord_actual.loc[common_ids].values.astype(float)
    )
    max_abs_order_residual = float(order_resids.max()) if len(order_resids) > 0 else 0.0

    # ── Rule 4 proxies ────────────────────────────────────────────────────────
    has_prev = ord_["prev_status"].notna()
    if has_prev.any():
        prev_arr   = ord_.loc[has_prev, "prev_status"].values
        curr_arr   = ord_.loc[has_prev, "status"].values
        legal_set  = LEGAL_TRANSITIONS
        n_illegal_transitions = int(
            sum(1 for p, c in zip(prev_arr, curr_arr) if (p, c) not in legal_set)
        )
    else:
        n_illegal_transitions = 0

    return np.array([
        n_orphan_fk_orders,
        n_orphan_fk_items,
        n_orders_over_item_limit,
        n_cust_over_open_limit,
        max_abs_lt_residual,
        max_abs_order_residual,
        n_illegal_transitions,
    ], dtype=np.float64)


def state_to_schema_vector(S: dict[str, pd.DataFrame]) -> np.ndarray:
    """Values-only features concatenated with schema-derived operational features."""
    return np.concatenate([state_to_values_vector(S), _schema_features(S)])
