"""
Access-ladder evaluation on the bank schema (second instantiation).

Demonstrates that the OTT construction generalises across operational
domains: under marginal-preserving corruptions the values-only XGBoost
remains at chance and rule-derived audit features close the gap.

Reports overall accuracy and per-violation recall for:
  - XGBoost on values-only features (expected: ~0.50)
  - XGBoost on values + 7 audit features (expected: near 1.000)
  - Oracle classifier (sanity, must be 1.000)

Usage:
    python experiments/run_bank_ladder.py \\
        --n-train 1000 --n-test 500 --n-accounts 200 \\
        --seeds 0 1 2 \\
        --output artifacts/bank_ladder_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bank_pilot.generator import generate_legal_state
from bank_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from bank_pilot.rules import oracle as bank_oracle
from bank_pilot.features import state_to_values_vector, state_to_schema_vector

VIOLATION_LIST = sorted(VIOLATION_TYPES)


def make_pairs(n_pairs: int, n_accounts: int, seed: int) -> tuple[list[dict], list[int], list[str]]:
    """Generate n_pairs (legal, illegal) pairs with one of each violation type per cycle."""
    states: list[dict] = []
    labels: list[int] = []
    violation_types: list[str] = []
    rng = np.random.default_rng(seed)

    for i in range(n_pairs):
        # legal anchor
        S = generate_legal_state(n_accounts=n_accounts, seed=int(rng.integers(0, 2**31)))
        states.append(S)
        labels.append(1)
        violation_types.append("")

        # corruption (cycle through types so each is well-represented)
        vt = VIOLATION_LIST[i % len(VIOLATION_LIST)]
        try:
            Sp = corrupt_to_illegal(S, vt, seed=int(rng.integers(0, 2**31)))
        except RuntimeError:
            # eligibility failure (rare): pick a different violation
            vt = VIOLATION_LIST[(i + 1) % len(VIOLATION_LIST)]
            Sp = corrupt_to_illegal(S, vt, seed=int(rng.integers(0, 2**31)))
        states.append(Sp)
        labels.append(0)
        violation_types.append(vt)

    return states, labels, violation_types


def featurize(states: list[dict], featurizer) -> np.ndarray:
    return np.vstack([featurizer(S) for S in states])


def fit_xgb(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def per_violation_recall(preds: np.ndarray, labels: list[int], vtypes: list[str]) -> dict[str, float]:
    """For each violation type, fraction of illegal states correctly flagged."""
    out: dict[str, float] = {}
    for vt in VIOLATION_LIST:
        idx = [i for i, (lbl, t) in enumerate(zip(labels, vtypes)) if lbl == 0 and t == vt]
        if not idx:
            out[vt] = float("nan")
        else:
            out[vt] = float(np.mean([preds[i] == 0 for i in idx]))
    return out


def run_one_seed(n_train: int, n_test: int, n_accounts: int, seed: int) -> dict:
    t0 = time.time()
    print(f"[seed={seed}] generating {n_train} train + {n_test} test pairs (n_accounts={n_accounts})...", flush=True)

    train_states, train_labels, _ = make_pairs(n_train, n_accounts, seed=seed)
    test_states, test_labels, test_vtypes = make_pairs(n_test, n_accounts, seed=seed + 10000)

    print(f"[seed={seed}] featurizing...", flush=True)
    Xv_train = featurize(train_states, state_to_values_vector)
    Xs_train = featurize(train_states, state_to_schema_vector)
    Xv_test = featurize(test_states, state_to_values_vector)
    Xs_test = featurize(test_states, state_to_schema_vector)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print(f"[seed={seed}] training XGBoost (values-only)...", flush=True)
    m_v = fit_xgb(Xv_train, y_train, seed=seed)
    pred_v = m_v.predict(Xv_test)

    print(f"[seed={seed}] training XGBoost (values + 7 audit features)...", flush=True)
    m_s = fit_xgb(Xs_train, y_train, seed=seed)
    pred_s = m_s.predict(Xs_test)

    print(f"[seed={seed}] running oracle...", flush=True)
    pred_o = np.array([1 if bank_oracle(S)["legal"] else 0 for S in test_states])

    out = {
        "seed": seed,
        "n_train_pairs": n_train,
        "n_test_pairs": n_test,
        "n_accounts": n_accounts,
        "values_acc": float((pred_v == y_test).mean()),
        "schema_acc": float((pred_s == y_test).mean()),
        "oracle_acc": float((pred_o == y_test).mean()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.update({f"values_recall_{vt}": v for vt, v in per_violation_recall(pred_v, list(test_labels), test_vtypes).items()})
    out.update({f"schema_recall_{vt}": v for vt, v in per_violation_recall(pred_s, list(test_labels), test_vtypes).items()})
    out.update({f"oracle_recall_{vt}": v for vt, v in per_violation_recall(pred_o, list(test_labels), test_vtypes).items()})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-accounts", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "artifacts" / "bank_ladder_results.csv")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in args.seeds:
        rows.append(run_one_seed(args.n_train, args.n_test, args.n_accounts, seed))

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"wrote {args.output}")
    print()
    print(f"{'seed':>5}  {'values':>7}  {'+7 audit':>9}  {'oracle':>7}  {'time':>6}")
    for r in rows:
        print(f"  {r['seed']:>3d}  {r['values_acc']:7.4f}  {r['schema_acc']:9.4f}  {r['oracle_acc']:7.4f}  {r['elapsed_s']:5.0f}s")

    print()
    print("MEAN across seeds:")
    for k in ["values_acc", "schema_acc", "oracle_acc"]:
        vals = [r[k] for r in rows]
        print(f"  {k:20s} {np.mean(vals):.4f}  ± {np.std(vals):.4f}")

    print()
    print("PER-VIOLATION recall (mean across seeds):")
    print(f"  {'access tier':<25s}  {'  '.join(f'{vt[:5]:>6s}' for vt in VIOLATION_LIST)}")
    for tier in ["values", "schema", "oracle"]:
        recalls = [np.mean([r[f"{tier}_recall_{vt}"] for r in rows]) for vt in VIOLATION_LIST]
        print(f"  {tier:<25s}  {'  '.join(f'{x:6.3f}' for x in recalls)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
