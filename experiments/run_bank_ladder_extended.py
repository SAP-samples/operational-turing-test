"""
Extended access-ladder evaluation on the bank schema.

Adds two tiers missing from the original `run_bank_ladder.py`:
  - rows-only XGBoost   (per-row predictions on transaction_lines, majority vote)
  - relational XGBoost  (state_to_relation_vector + XGBoost)

Final ladder reported:
  values   — XGBoost on values-only column aggregates       (expected ~0.50)
  rows     — XGBoost on raw transaction_lines rows + maj    (expected ~0.50)
  relation — XGBoost on values + relational structure       (expected: recovers FK/cardinality, misses balance)
  schema   — XGBoost on values + 7 audit features           (expected near 1.000)
  oracle   — direct rules.py call                            (1.000 by construction)

Outputs:
    artifacts/bank_ladder_extended_results.csv

Usage:
    python experiments/run_bank_ladder_extended.py \\
        --n-train 1000 --n-test 500 --n-accounts 200 \\
        --seeds 0 1 2 3 4 \\
        --output artifacts/bank_ladder_extended_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bank_pilot.generator import generate_legal_state
from bank_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from bank_pilot.rules import oracle as bank_oracle
from bank_pilot.features import state_to_values_vector, state_to_schema_vector
from bank_pilot.relational_features import state_to_relation_vector, state_to_relation_plus_cumagg_vector

VIOLATION_LIST = sorted(VIOLATION_TYPES)
SIDE_TO_INT = {"debit": 0, "credit": 1}


def make_pairs(n_pairs: int, n_accounts: int, seed: int):
    states, labels, vts = [], [], []
    rng = np.random.default_rng(seed)
    for i in range(n_pairs):
        S = generate_legal_state(n_accounts=n_accounts, seed=int(rng.integers(0, 2**31)))
        states.append(S); labels.append(1); vts.append("")
        vt = VIOLATION_LIST[i % len(VIOLATION_LIST)]
        try:
            Sp = corrupt_to_illegal(S, vt, seed=int(rng.integers(0, 2**31)))
        except RuntimeError:
            vt = VIOLATION_LIST[(i + 1) % len(VIOLATION_LIST)]
            Sp = corrupt_to_illegal(S, vt, seed=int(rng.integers(0, 2**31)))
        states.append(Sp); labels.append(0); vts.append(vt)
    return states, labels, vts


def featurize(states, fn):
    return np.vstack([fn(S) for S in states])


def fit_xgb(X, y, seed):
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed, verbosity=0,
    )
    clf.fit(X, y)
    return clf


# ── rows-only baseline ────────────────────────────────────────────────────────

def rows_to_features(state: dict[str, pd.DataFrame]) -> np.ndarray:
    """Per-row featurization of transaction_lines: one row -> one feature vector.

    Features are exactly the row's own columns (no group-by, no join):
      amount_cents (numeric), side (encoded), transaction_id (numeric id),
      account_id (numeric id).

    Without rule context, the model has the raw row values and the opaque
    grouping identifiers, but no instruction about which join or aggregate
    to evaluate.
    """
    ln = state["transaction_lines"]
    if len(ln) == 0:
        return np.empty((0, 4), dtype=np.float64)
    side_enc = ln["side"].map(SIDE_TO_INT).fillna(-1).to_numpy(dtype=np.float64)
    return np.column_stack([
        ln["amount_cents"].to_numpy(dtype=np.float64),
        side_enc,
        ln["transaction_id"].to_numpy(dtype=np.float64),
        ln["account_id"].to_numpy(dtype=np.float64),
    ])


def fit_rows_xgb(states, labels, seed):
    """Train XGBoost per row, with row labels = state label of source state."""
    X_blocks, y_blocks = [], []
    for S, y in zip(states, labels):
        rows = rows_to_features(S)
        if len(rows) == 0:
            continue
        X_blocks.append(rows)
        y_blocks.append(np.full(len(rows), y, dtype=int))
    X = np.vstack(X_blocks); y = np.concatenate(y_blocks)
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed, verbosity=0,
    )
    clf.fit(X, y)
    return clf


def predict_rows(clf, states):
    """Predict per row, then majority-vote per state. Ties default to 1 (legal)."""
    out = np.zeros(len(states), dtype=int)
    for i, S in enumerate(states):
        rows = rows_to_features(S)
        if len(rows) == 0:
            out[i] = 1
            continue
        row_preds = clf.predict(rows)
        c = Counter(row_preds.tolist())
        if c[1] >= c[0]:
            out[i] = 1
        else:
            out[i] = 0
    return out


# ── per-violation recalls ─────────────────────────────────────────────────────

def per_violation_recall(preds, labels, vts):
    """Recall = fraction of illegal states correctly flagged, per violation type."""
    out = {}
    for vt in VIOLATION_LIST:
        mask = np.array([t == vt for t in vts])
        if mask.sum() == 0:
            out[vt] = float("nan")
            continue
        # illegal label = 0; correct illegal flag = predicted 0
        correct = ((np.array(preds) == 0) & (np.array(labels) == 0) & mask).sum()
        total = mask.sum()
        out[vt] = float(correct) / float(total) if total else float("nan")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, n_train: int, n_test: int, n_accounts: int) -> dict:
    t0 = time.time()
    train_states, train_labels, _ = make_pairs(n_train, n_accounts, seed=seed)
    test_states,  test_labels,  test_vts = make_pairs(n_test,  n_accounts, seed=seed + 10_000)

    # values-only
    Xv_tr = featurize(train_states, state_to_values_vector)
    Xv_te = featurize(test_states,  state_to_values_vector)
    clf_v = fit_xgb(Xv_tr, np.array(train_labels), seed)
    pred_v = clf_v.predict(Xv_te)
    acc_v = float((pred_v == np.array(test_labels)).mean())
    rec_v = per_violation_recall(pred_v, test_labels, test_vts)

    # rows-only
    clf_r = fit_rows_xgb(train_states, train_labels, seed)
    pred_r = predict_rows(clf_r, test_states)
    acc_r = float((pred_r == np.array(test_labels)).mean())
    rec_r = per_violation_recall(pred_r, test_labels, test_vts)

    # relational
    Xrel_tr = featurize(train_states, state_to_relation_vector)
    Xrel_te = featurize(test_states,  state_to_relation_vector)
    clf_rel = fit_xgb(Xrel_tr, np.array(train_labels), seed)
    pred_rel = clf_rel.predict(Xrel_te)
    acc_rel = float((pred_rel == np.array(test_labels)).mean())
    rec_rel = per_violation_recall(pred_rel, test_labels, test_vts)

    # relational + rule-agnostic cumulative aggregates
    Xrelp_tr = featurize(train_states, state_to_relation_plus_cumagg_vector)
    Xrelp_te = featurize(test_states,  state_to_relation_plus_cumagg_vector)
    clf_relp = fit_xgb(Xrelp_tr, np.array(train_labels), seed)
    pred_relp = clf_relp.predict(Xrelp_te)
    acc_relp = float((pred_relp == np.array(test_labels)).mean())
    rec_relp = per_violation_recall(pred_relp, test_labels, test_vts)

    # schema (audit features)
    Xs_tr = featurize(train_states, state_to_schema_vector)
    Xs_te = featurize(test_states,  state_to_schema_vector)
    clf_s = fit_xgb(Xs_tr, np.array(train_labels), seed)
    pred_s = clf_s.predict(Xs_te)
    acc_s = float((pred_s == np.array(test_labels)).mean())
    rec_s = per_violation_recall(pred_s, test_labels, test_vts)

    # oracle
    pred_o = np.array([1 if bank_oracle(S)["legal"] else 0 for S in test_states])
    acc_o = float((pred_o == np.array(test_labels)).mean())
    rec_o = per_violation_recall(pred_o, test_labels, test_vts)

    elapsed = time.time() - t0
    return {
        "seed": seed,
        "n_train_pairs": n_train, "n_test_pairs": n_test, "n_accounts": n_accounts,
        "values_acc": acc_v, "rows_acc": acc_r, "relation_acc": acc_rel,
        "relation_plus_cumagg_acc": acc_relp,
        "schema_acc": acc_s, "oracle_acc": acc_o,
        "elapsed_s": round(elapsed, 1),
        **{f"values_recall_{vt}":   rec_v[vt]   for vt in VIOLATION_LIST},
        **{f"rows_recall_{vt}":     rec_r[vt]   for vt in VIOLATION_LIST},
        **{f"relation_recall_{vt}": rec_rel[vt] for vt in VIOLATION_LIST},
        **{f"relation_plus_cumagg_recall_{vt}": rec_relp[vt] for vt in VIOLATION_LIST},
        **{f"schema_recall_{vt}":   rec_s[vt]   for vt in VIOLATION_LIST},
        **{f"oracle_recall_{vt}":   rec_o[vt]   for vt in VIOLATION_LIST},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test",  type=int, default=500)
    p.add_argument("--n-accounts", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=str, default="artifacts/bank_ladder_extended_results.csv")
    args = p.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        print(f"[seed {s}] starting...", flush=True)
        row = run_one_seed(s, args.n_train, args.n_test, args.n_accounts)
        print(f"[seed {s}] values={row['values_acc']:.4f} rows={row['rows_acc']:.4f} "
              f"relation={row['relation_acc']:.4f} rel+cumagg={row['relation_plus_cumagg_acc']:.4f} "
              f"schema={row['schema_acc']:.4f} oracle={row['oracle_acc']:.4f}  ({row['elapsed_s']:.1f}s)",
              flush=True)
        results.append(row)

    cols = list(results[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nWrote {out_path}")
    df = pd.DataFrame(results)
    print("\n=== mean across seeds ===")
    for tier in ["values", "rows", "relation", "relation_plus_cumagg", "schema", "oracle"]:
        print(f"  {tier:<10s}  acc = {df[f'{tier}_acc'].mean():.4f}  "
              f"(min {df[f'{tier}_acc'].min():.4f}, max {df[f'{tier}_acc'].max():.4f})")


if __name__ == "__main__":
    main()
