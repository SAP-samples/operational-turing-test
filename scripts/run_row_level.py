"""
Row-level access experiment for the journal version.

Tests whether a flat-table model (TabICL) can detect operational violations
when given raw row-level access to the orders table — no aggregation, no schema.

This experiment exists to answer: "is the values-only restriction a straw man?"

Design
------
For each violation type, we train TabICL on individual orders rows, labeled by
the state they came from (legal=1, illegal=0). At test time, we predict each
row in a state independently and aggregate via majority vote to a state-level
verdict. No join, no group-by, no schema context.

Expected results
----------------
All four violations should remain near chance at state level. The model sees
individual orders rows, but it does not know which schema-guided predicate to
evaluate and most rows in an illegal state are still locally ordinary. This
experiment tests raw row access, not relational access.

Outputs
-------
  artifacts/row_level_results.csv    — state-level accuracy per violation type
  artifacts/row_level_figure.png     — bar chart comparing row-level vs aggregate baseline

Usage
-----
    python scripts/run_row_level.py [--skip-gate-check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES

ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"
N_PAIRS     = 200      # per violation type
N_TEST      = 100      # test pairs per violation type
N_CUSTOMERS = 200
SEED_BASE   = 20_000_000

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)

# Aggregate-baseline accuracies from run_turing_test.py (for comparison figure)
AGGREGATE_BASELINE = {
    "cardinality_break": 0.502,
    "derivation_break":  0.487,
    "fk_break":          0.536,
    "transition_break":  0.537,
}


# ── row-level featurization ───────────────────────────────────────────────────

STATUS_MAP = {"pending": 0, "shipped": 1, "delivered": 2, "cancelled": 3}

def orders_to_rows(orders: pd.DataFrame) -> np.ndarray:
    """
    Convert an orders DataFrame to a numeric row matrix.

    Columns: customer_id, status (encoded), prev_status (encoded, -1 if null), total
    Deliberately uses no cross-table information and no group-by.
    """
    status     = orders["status"].map(STATUS_MAP).fillna(-1).values.astype(np.float32)
    prev_stat  = orders["prev_status"].map(STATUS_MAP).fillna(-1).values.astype(np.float32)
    cust_id    = orders["customer_id"].values.astype(np.float32)
    total      = orders["total"].values.astype(np.float32)
    return np.column_stack([cust_id, status, prev_stat, total])


# ── dataset builder ───────────────────────────────────────────────────────────

def build_row_dataset(
    vt: str,
    n_pairs: int,
    seed_offset: int,
    rows_per_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, int]]]:
    """
    Build a training set of individual rows + a test set of (state_rows, state_label) tuples.

    Returns
    -------
    X_train, y_train : individual rows for TabICL training
    test_states      : list of (row_matrix, true_state_label) for state-level evaluation
    """
    rng = np.random.default_rng(seed_offset)

    train_rows, train_labels = [], []
    test_states: list[tuple[np.ndarray, int]] = []

    n_train = n_pairs
    n_test  = N_TEST

    for i in range(n_train + n_test):
        seed   = seed_offset + i
        S      = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        Sp     = corrupt_to_illegal(S, vt, seed=seed + 500_000)

        for state, label in [(S, 1), (Sp, 0)]:
            rows = orders_to_rows(state["orders"])
            if i < n_train:
                # Training: optionally subsample rows per state
                if rows_per_state is not None and len(rows) > rows_per_state:
                    idx  = rng.choice(len(rows), rows_per_state, replace=False)
                    rows = rows[idx]
                train_rows.append(rows)
                train_labels.extend([label] * len(rows))
            else:
                test_states.append((rows, label))

    X_train = np.vstack(train_rows).astype(np.float32)
    y_train = np.array(train_labels, dtype=int)
    return X_train, y_train, test_states


# ── state-level aggregation ───────────────────────────────────────────────────

def predict_state(model, rows: np.ndarray, rule: str = "any") -> int:
    """
    Aggregate per-row predictions to a state-level verdict.

    rule="any"      → illegal if ANY row predicted 0 (illegal)
    rule="majority" → illegal if >50% rows predicted 0
    """
    preds = model.predict(rows)
    if rule == "any":
        return 0 if (preds == 0).any() else 1
    else:
        return 0 if (preds == 0).mean() > 0.5 else 1


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_gate_check: bool = False) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("(or use --skip-gate-check to bypass)")
        return 1

    try:
        import torch.nn.modules.transformer as _t, typing
        if not hasattr(_t, "Optional"):
            _t.Optional = typing.Optional
        from tabicl import TabICLClassifier
    except ImportError:
        print("ERROR: tabicl is not installed. Run: pip install tabicl")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    rows: list[dict] = []

    print(f"Row-level access experiment  (n_train_pairs={N_PAIRS}, n_test_pairs={N_TEST})")
    print(f"  Table: orders  |  Features: customer_id, status, prev_status, total")
    print(f"  Aggregation: majority vote over row predictions → state label")
    print()

    for vt in VIOLATION_TYPES_SORTED:
        seed_off = SEED_BASE + VIOLATION_TYPES_SORTED.index(vt) * 200_000
        print(f"  [{vt}] building row dataset…", end=" ", flush=True)

        # Subsample rows per state to keep TabICL training manageable
        # (orders table has ~600 rows per state for 200 customers)
        X_train, y_train, test_states = build_row_dataset(
            vt, N_PAIRS, seed_off, rows_per_state=5
        )
        print(f"{len(X_train)} training rows  ({y_train.mean():.2f} frac legal)")

        print(f"    fitting TabICL on row-level data…", end=" ", flush=True)
        model = TabICLClassifier(random_state=0, verbose=False, n_jobs=1)
        model.fit(X_train, y_train)
        print("done")

        # State-level evaluation — batch all test rows into one predict() call
        # then reconstruct per-state predictions from row offsets.
        all_test_rows   = np.vstack([r for r, _ in test_states])
        state_sizes     = [len(r) for r, _ in test_states]
        true_labels_arr = [l for _, l in test_states]

        all_preds = model.predict(all_test_rows)

        correct_maj, correct_any = 0, 0
        offset = 0
        for size, true_label in zip(state_sizes, true_labels_arr):
            row_preds = all_preds[offset: offset + size]
            offset   += size
            pred_maj  = 0 if (row_preds == 0).mean() > 0.5 else 1
            pred_any  = 0 if (row_preds == 0).any() else 1
            correct_maj += int(pred_maj == true_label)
            correct_any += int(pred_any == true_label)

        n_test_states = len(test_states)
        acc_maj = correct_maj / n_test_states
        acc_any = correct_any / n_test_states
        agg_baseline = AGGREGATE_BASELINE.get(vt, 0.5)

        print(f"    state-level acc  majority={acc_maj:.4f}  any-illegal={acc_any:.4f}  "
              f"(aggregate baseline={agg_baseline:.3f})")
        print()

        rows.append({
            "violation_type":      vt,
            "acc_majority_vote":   acc_maj,
            "acc_any_illegal":     acc_any,
            "acc_aggregate_baseline": agg_baseline,
        })

    # ── save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = ARTIFACTS / "row_level_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved → {csv_path}")

    # ── figure ─────────────────────────────────────────────────────────────────
    vt_labels = [v.replace("_break", "").replace("_", "\n") for v in VIOLATION_TYPES_SORTED]
    x = np.arange(len(VIOLATION_TYPES_SORTED))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df["acc_majority_vote"],     width, label="Row-level (majority vote)",  color="#3498db", alpha=0.85)
    ax.bar(x,          df["acc_any_illegal"],       width, label="Row-level (any-illegal)",     color="#e67e22", alpha=0.85)
    ax.bar(x + width,  df["acc_aggregate_baseline"],width, label="Aggregate baseline (XGBoost)",color="#e74c3c", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(vt_labels, fontsize=10)
    ax.set_ylabel("State-level accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Row-level vs Aggregate access — state-level accuracy by violation type\n"
                 f"(TabICL on raw orders rows; n_train={N_PAIRS} pairs, n_test={N_TEST} pairs)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = ARTIFACTS / "row_level_figure.png"
    plt.savefig(fig_path, dpi=150)
    print(f"  Figure saved → {fig_path}")

    # ── interpretation ─────────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print("INTERPRETATION")
    print("═" * 72)
    for _, r in df.iterrows():
        vt  = r["violation_type"]
        acc = r["acc_majority_vote"]
        agg = r["acc_aggregate_baseline"]
        delta = acc - agg
        if acc > 0.70:
            note = "detectable at row level (within-row or statistical signal)"
        elif acc > 0.55:
            note = "marginal row-level signal (may be statistical, not semantic)"
        else:
            note = "not detectable at row level → cross-table rule, requires schema"
        print(f"  {vt:<22}  row={acc:.3f}  agg={agg:.3f}  Δ={delta:+.3f}  → {note}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check))
