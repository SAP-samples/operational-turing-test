"""
Smoke tests for tf_pilot.

These tests encode the core contractual claims of the pilot:

  1. generate_legal_state always produces a state the oracle accepts.
  2. corrupt_to_illegal always fires exactly the targeted rule.
  3. The 1-way status marginal is exactly preserved by transition_break.
  4. The item value marginals are untouched by cardinality_break.
  5. The order table is untouched by derivation_break (orders.total stays wrong).
  6. Null corruption (deep copy) leaves the oracle unchanged.

A test failure here is a design bug, not a data fluke.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf_pilot.rules import (
    oracle,
    rule_fk_integrity, rule_cardinality, rule_derivation, rule_transition,
    compute_line_total, compute_order_total,
    LEGAL_TRANSITIONS, VALID_STATUSES,
    TAX_RATES, DISCOUNT_TABLE,
)
from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES, RULE_FOR_VIOLATION
from tf_pilot.features import state_to_values_vector, state_to_relation_vector, state_to_schema_vector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def S():
    return generate_legal_state(n_customers=80, seed=0)


@pytest.fixture(scope="module")
def corrupted(S):
    return {vt: corrupt_to_illegal(S, vt, seed=99) for vt in VIOLATION_TYPES}


# ── rules.py unit tests ───────────────────────────────────────────────────────

class TestRuleHelpers:
    def test_get_discount_gold_tier(self):
        from tf_pilot.rules import get_discount
        assert get_discount("gold", 1)  == 0.10
        assert get_discount("gold", 5)  == 0.15
        assert get_discount("gold", 10) == 0.20
        assert get_discount("gold", 4)  == 0.10   # below 5-threshold

    def test_get_discount_unknown_tier(self):
        from tf_pilot.rules import get_discount
        assert get_discount("platinum", 100) == 0.0

    def test_get_tax_known_country(self):
        from tf_pilot.rules import get_tax
        assert get_tax("DE") == 0.19
        assert get_tax("US") == 0.08

    def test_get_tax_unknown_country(self):
        from tf_pilot.rules import get_tax
        assert get_tax("ZZ") == TAX_RATES["OTHER"]

    def test_compute_line_total_bronze_below_threshold(self):
        # bronze, qty=3 → discount=0 (threshold is 5 for 5%)
        lt = compute_line_total(unit_price=100.0, quantity=3, tier="bronze")
        assert abs(lt - 300.0) < 1e-9

    def test_compute_line_total_gold_bulk(self):
        lt = compute_line_total(unit_price=100.0, quantity=10, tier="gold")
        assert abs(lt - 10 * 100.0 * 0.80) < 1e-9

    def test_compute_order_total(self):
        total = compute_order_total([100.0, 200.0], "DE")  # tax=0.19
        assert abs(total - 300.0 * 1.19) < 1e-9

    def test_legal_transitions_are_a_frozenset(self):
        assert ("pending", "shipped") in LEGAL_TRANSITIONS
        assert ("shipped", "pending") not in LEGAL_TRANSITIONS   # backwards
        assert ("delivered", "pending") not in LEGAL_TRANSITIONS # backwards


# ── Oracle tests on hand-crafted minimal states ───────────────────────────────

class TestOracleDirect:
    """Verify each rule fires on a hand-crafted violation, independent of generator."""

    def _minimal_legal(self):
        customers = pd.DataFrame([{
            "id": 1, "country": "US", "tier": "bronze",
            "signup_date": pd.Timestamp("2023-01-01"),
        }])
        orders = pd.DataFrame([{
            "id": 1, "customer_id": 1,
            "status": "delivered", "prev_status": "shipped",
            "order_date": pd.Timestamp("2023-06-01"),
            "total": round(10.0 * 1.08, 6),
        }])
        order_items = pd.DataFrame([{
            "id": 1, "order_id": 1, "product_id": 1,
            "quantity": 1, "unit_price": 10.0,
            "line_total": 10.0,   # bronze qty=1 → discount=0
        }])
        return {"customers": customers, "orders": orders, "order_items": order_items}

    def test_minimal_legal_passes(self):
        assert oracle(self._minimal_legal())["legal"]

    def test_fk_violation_detected(self):
        S = self._minimal_legal()
        S["orders"].loc[0, "customer_id"] = 999
        result = oracle(S)
        assert not result["legal"]
        assert "fk_integrity" in result["violations"]

    def test_cardinality_item_overflow_detected(self):
        S = self._minimal_legal()
        extra = [{"id": i + 2, "order_id": 1, "product_id": 1,
                  "quantity": 1, "unit_price": 10.0, "line_total": 10.0}
                 for i in range(20)]   # 21 items total
        S["order_items"] = pd.concat([S["order_items"], pd.DataFrame(extra)], ignore_index=True)
        result = oracle(S)
        assert not result["legal"]
        assert "cardinality" in result["violations"]

    def test_derivation_violation_detected(self):
        S = self._minimal_legal()
        S["order_items"].loc[0, "line_total"] = 999.0   # wrong value
        result = oracle(S)
        assert not result["legal"]
        assert "derivation" in result["violations"]

    def test_transition_violation_detected(self):
        S = self._minimal_legal()
        # (shipped, pending) is not in LEGAL_TRANSITIONS
        S["orders"].loc[0, "prev_status"] = "shipped"
        S["orders"].loc[0, "status"]      = "pending"
        result = oracle(S)
        assert not result["legal"]
        assert "transition" in result["violations"]

    def test_open_orders_limit_detected(self):
        S = self._minimal_legal()
        cust = S["customers"]
        extra_orders = pd.DataFrame([{
            "id": i + 2, "customer_id": 1,
            "status": "pending", "prev_status": None,
            "order_date": pd.Timestamp("2023-07-01"),
            "total": round(10.0 * 1.08, 6),
        } for i in range(4)])  # 4 more pending → 4 open total (> MAX_OPEN_ORDERS=3)
        S["orders"] = pd.concat([S["orders"], extra_orders], ignore_index=True)
        extra_items = pd.DataFrame([{
            "id": i + 2, "order_id": i + 2, "product_id": 1,
            "quantity": 1, "unit_price": 10.0, "line_total": 10.0,
        } for i in range(4)])
        S["order_items"] = pd.concat([S["order_items"], extra_items], ignore_index=True)
        result = oracle(S)
        assert not result["legal"]
        assert "cardinality" in result["violations"]


# ── Generator tests ───────────────────────────────────────────────────────────

class TestGenerator:
    def test_oracle_accepts_generated_state(self, S):
        result = oracle(S)
        assert result["legal"], f"Oracle violations: {result['violations']}"

    def test_required_tables_present(self, S):
        assert set(S.keys()) == {"customers", "orders", "order_items"}

    def test_customers_schema(self, S):
        for col in ("id", "country", "tier", "signup_date"):
            assert col in S["customers"].columns

    def test_orders_schema(self, S):
        for col in ("id", "customer_id", "status", "prev_status", "order_date", "total"):
            assert col in S["orders"].columns

    def test_order_items_schema(self, S):
        for col in ("id", "order_id", "product_id", "quantity", "unit_price", "line_total"):
            assert col in S["order_items"].columns

    def test_non_empty(self, S):
        assert len(S["customers"]) > 0
        assert len(S["orders"]) > 0
        assert len(S["order_items"]) > 0

    def test_deterministic(self):
        S1 = generate_legal_state(n_customers=40, seed=7)
        S2 = generate_legal_state(n_customers=40, seed=7)
        for table in ("customers", "orders", "order_items"):
            pd.testing.assert_frame_equal(S1[table], S2[table])

    def test_different_seeds_differ(self):
        S1 = generate_legal_state(n_customers=40, seed=1)
        S2 = generate_legal_state(n_customers=40, seed=2)
        # At least one table should differ
        any_diff = any(
            not S1[t].equals(S2[t]) for t in ("customers", "orders", "order_items")
        )
        assert any_diff

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
    def test_oracle_passes_across_seeds(self, seed):
        result = oracle(generate_legal_state(n_customers=50, seed=seed))
        assert result["legal"], f"seed={seed}: {result['violations']}"

    def test_statuses_are_valid(self, S):
        assert set(S["orders"]["status"]).issubset(VALID_STATUSES)

    def test_recorded_transitions_are_legal(self, S):
        has_prev = S["orders"]["prev_status"].notna()
        transitions = set(zip(
            S["orders"].loc[has_prev, "prev_status"],
            S["orders"].loc[has_prev, "status"],
        ))
        assert transitions.issubset(LEGAL_TRANSITIONS), f"Illegal transitions: {transitions - LEGAL_TRANSITIONS}"

    def test_items_per_order_in_range(self, S):
        counts = S["order_items"].groupby("order_id").size()
        assert counts.min() >= 1
        assert counts.max() <= 20

    def test_open_orders_per_customer_in_range(self, S):
        open_orders = S["orders"][S["orders"]["status"].isin({"pending", "shipped"})]
        open_per_cust = open_orders.groupby("customer_id").size()
        assert (open_per_cust <= 3).all()


# ── Feature-vector tests ─────────────────────────────────────────────────────

class TestFeatureVectors:
    def test_relation_vector_is_finite_and_extends_values(self, S):
        values = state_to_values_vector(S)
        relation = state_to_relation_vector(S)
        assert len(relation) > len(values)
        assert np.isfinite(relation).all()

    def test_relation_vector_length_is_seed_stable(self):
        S1 = generate_legal_state(n_customers=40, seed=1)
        S2 = generate_legal_state(n_customers=40, seed=2)
        assert len(state_to_relation_vector(S1)) == len(state_to_relation_vector(S2))

    def test_schema_vector_extends_values(self, S):
        values = state_to_values_vector(S)
        schema = state_to_schema_vector(S)
        assert len(schema) > len(values)


# ── Corruptor tests ───────────────────────────────────────────────────────────

class TestCorruptor:

    # ── Core postconditions ───────────────────────────────────────────────────

    @pytest.mark.parametrize("vt", sorted(VIOLATION_TYPES))
    def test_targeted_rule_violated(self, S, vt):
        S_prime = corrupt_to_illegal(S, vt, seed=1)
        result  = oracle(S_prime)
        assert RULE_FOR_VIOLATION[vt] in result["violations"], (
            f"{vt}: expected '{RULE_FOR_VIOLATION[vt]}' in violations, "
            f"got {result['violations']}"
        )

    @pytest.mark.parametrize("vt", sorted(VIOLATION_TYPES))
    def test_illegal_after_corruption(self, S, vt):
        S_prime = corrupt_to_illegal(S, vt, seed=1)
        assert not oracle(S_prime)["legal"]

    @pytest.mark.parametrize("vt", sorted(VIOLATION_TYPES))
    @pytest.mark.parametrize("seed", [0, 5, 42])
    def test_all_violation_types_multiple_seeds(self, S, vt, seed):
        S_prime = corrupt_to_illegal(S, vt, seed=seed)
        result  = oracle(S_prime)
        assert not result["legal"]
        assert RULE_FOR_VIOLATION[vt] in result["violations"]

    @pytest.mark.parametrize("vt", sorted(VIOLATION_TYPES))
    def test_deterministic(self, S, vt):
        S1 = corrupt_to_illegal(S, vt, seed=7)
        S2 = corrupt_to_illegal(S, vt, seed=7)
        for table in ("customers", "orders", "order_items"):
            pd.testing.assert_frame_equal(S1[table], S2[table])

    # ── Marginal-preservation spot-checks ────────────────────────────────────

    def test_transition_break_preserves_status_1way_marginal(self, S):
        """The 1-way status marginal must be exactly preserved after transition_break."""
        S_prime = corrupt_to_illegal(S, "transition_break", seed=1)
        orig    = S["orders"]["status"].value_counts().sort_index()
        new     = S_prime["orders"]["status"].value_counts().sort_index()
        pd.testing.assert_series_equal(orig, new, check_names=False)

    def test_cardinality_break_preserves_item_value_marginals(self, S):
        """
        cardinality_break permutes order_id within order_items but must not
        change any item value column.
        """
        S_prime = corrupt_to_illegal(S, "cardinality_break", seed=1)
        for col in ("quantity", "unit_price", "line_total", "product_id"):
            orig_sorted = S["order_items"][col].sort_values().reset_index(drop=True)
            new_sorted  = S_prime["order_items"][col].sort_values().reset_index(drop=True)
            pd.testing.assert_series_equal(orig_sorted, new_sorted, check_names=False)

    def test_derivation_break_does_not_touch_order_items(self, S):
        """derivation_break perturbs orders.total only; order_items must be untouched."""
        S_prime = corrupt_to_illegal(S, "derivation_break", seed=1)
        pd.testing.assert_frame_equal(S_prime["order_items"], S["order_items"])

    def test_derivation_break_does_not_touch_customers(self, S):
        """derivation_break must not touch the customers table."""
        S_prime = corrupt_to_illegal(S, "derivation_break", seed=1)
        pd.testing.assert_frame_equal(S_prime["customers"], S["customers"])

    def test_derivation_break_preserves_order_total_mean(self, S):
        """Paired noise on orders.total → mean is exactly preserved."""
        S_prime   = corrupt_to_illegal(S, "derivation_break", seed=1)
        orig_mean = float(S["orders"]["total"].mean())
        new_mean  = float(S_prime["orders"]["total"].mean())
        assert abs(orig_mean - new_mean) < 1e-6

    def test_fk_break_does_not_touch_customers_or_items(self, S):
        """fk_break should only modify orders.customer_id."""
        S_prime = corrupt_to_illegal(S, "fk_break", seed=1)
        pd.testing.assert_frame_equal(S_prime["customers"], S["customers"])
        pd.testing.assert_frame_equal(S_prime["order_items"], S["order_items"])

    # ── Isolation: only the targeted rule fires (where feasible) ─────────────

    def test_transition_break_does_not_break_fk(self, S):
        S_prime = corrupt_to_illegal(S, "transition_break", seed=1)
        assert rule_fk_integrity(S_prime), "transition_break accidentally broke FK integrity"

    def test_transition_break_does_not_break_derivation(self, S):
        S_prime = corrupt_to_illegal(S, "transition_break", seed=1)
        assert rule_derivation(S_prime), "transition_break accidentally broke derivation"

    def test_derivation_break_does_not_break_fk(self, S):
        S_prime = corrupt_to_illegal(S, "derivation_break", seed=1)
        assert rule_fk_integrity(S_prime), "derivation_break accidentally broke FK integrity"

    def test_cardinality_break_does_not_break_fk(self, S):
        S_prime = corrupt_to_illegal(S, "cardinality_break", seed=1)
        assert rule_fk_integrity(S_prime), "cardinality_break accidentally broke FK integrity"

    def test_cardinality_break_does_not_break_derivation(self, S):
        S_prime = corrupt_to_illegal(S, "cardinality_break", seed=1)
        assert rule_derivation(S_prime), "cardinality_break accidentally broke derivation"

    # ── Null corruption sanity check ─────────────────────────────────────────

    def test_null_corruption_preserves_legality(self, S):
        """A deep copy with no mutation must still pass the oracle."""
        S_copy = {k: df.copy() for k, df in S.items()}
        assert oracle(S_copy)["legal"]

    # ── Error handling ────────────────────────────────────────────────────────

    def test_unknown_violation_type_raises(self, S):
        with pytest.raises(ValueError, match="Unknown violation type"):
            corrupt_to_illegal(S, "nonexistent_break", seed=0)
