"""
Sensitivity sweep on the marginal-match tolerance (tau) and the TOST
equivalence margin (delta).

The original bank-ledger run uses corruption noise scale 0.001 (0.1% of
column mean), which produces 1- and 2-way TV well under tau=0.02. The
reviewer's concern is that the result may be brittle to this choice.
This script varies the corruption noise scale and reports:

  - measured 1-way TV between legal and illegal column-value marginals
  - values-only XGBoost accuracy (expected to stay near chance for any
    tau < 1 by Theorem 1; degradation should be graceful)
  - schema (audit-feature) XGBoost accuracy (should stay near 1.0)
  - TOST p-value at delta in {0.02, 0.05}

The corruption is performed with a monkey-patched noise scale so the
on-disk bank_pilot.corruptor module is not modified.

Usage:
    python experiments/run_bank_sensitivity_sweep.py \\
        --noise-scales 0.001 0.005 0.010 0.025 0.050 \\
        --n-train 500 --n-test 250 --seeds 0 1 2 \\
        --output artifacts/bank_sensitivity_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import bank_pilot.corruptor as corruptor_mod
from bank_pilot.generator import generate_legal_state
from bank_pilot.corruptor import VIOLATION_TYPES
from bank_pilot.rules import oracle as bank_oracle
from bank_pilot.features import state_to_values_vector, state_to_schema_vector

VIOLATION_LIST = sorted(VIOLATION_TYPES)
ORIGINAL_BALANCE = corruptor_mod._corrupt_balance
ORIGINAL_ACCOUNT_BALANCE = corruptor_mod._corrupt_account_balance


def _patch_noise(scale: float):
    """Return wrappers that override the hard-coded 0.001 noise scale."""

    def patched_balance(S, rng):
        accounts = S["accounts"].copy()
        lines = S["transaction_lines"].copy()

        debit_rows = lines[lines["side"] == "debit"]
        if len(debit_rows) < 2:
            raise RuntimeError("Need at least two debit rows.")

        tx_ids = debit_rows["transaction_id"].unique()
        if len(tx_ids) < 2:
            raise RuntimeError("Need debits across at least two transactions.")
        chosen = rng.choice(tx_ids, size=2, replace=False)
        row_a = int(debit_rows[debit_rows["transaction_id"] == chosen[0]].index[0])
        row_b = int(debit_rows[debit_rows["transaction_id"] == chosen[1]].index[0])

        delta = max(1, int(lines["amount_cents"].mean() * scale))
        lines.at[row_a, "amount_cents"] = int(lines.at[row_a, "amount_cents"]) + delta
        lines.at[row_b, "amount_cents"] = int(lines.at[row_b, "amount_cents"]) - delta

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

    def patched_account_balance(S, rng):
        accounts = S["accounts"].copy()
        n_acc = len(accounts)
        n_break = max(2, int(n_acc * 0.10))
        n_break += n_break % 2
        idx = rng.choice(n_acc, size=n_break, replace=False)
        mean_balance = float(np.abs(accounts["balance_cents"]).mean())
        sc = max(1.0, mean_balance * scale)
        half = n_break // 2
        magnitudes = rng.uniform(sc, sc * 2.0, size=half)
        noise = np.concatenate([magnitudes, -magnitudes])
        rng.shuffle(noise)
        balances = accounts["balance_cents"].values.copy().astype(np.int64)
        balances[idx] += noise.astype(np.int64)
        accounts["balance_cents"] = balances
        S["accounts"] = accounts
        return S

    return patched_balance, patched_account_balance


def make_pairs(n_pairs: int, n_accounts: int, seed: int):
    states, labels, vts = [], [], []
    rng = np.random.default_rng(seed)
    for i in range(n_pairs):
        S = generate_legal_state(n_accounts=n_accounts,
                                 seed=int(rng.integers(0, 2**31)))
        states.append(S); labels.append(1); vts.append("")
        vt = VIOLATION_LIST[i % len(VIOLATION_LIST)]
        try:
            Sp = corruptor_mod.corrupt_to_illegal(
                S, vt, seed=int(rng.integers(0, 2**31)))
        except RuntimeError:
            vt = VIOLATION_LIST[(i + 1) % len(VIOLATION_LIST)]
            Sp = corruptor_mod.corrupt_to_illegal(
                S, vt, seed=int(rng.integers(0, 2**31)))
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


def estimate_tv_one_way(legal_states, illegal_states, n_bins: int = 50) -> float:
    """Pooled 1-way TV over the values-only feature columns."""
    Xl = featurize(legal_states, state_to_values_vector)
    Xi = featurize(illegal_states, state_to_values_vector)
    tv_per_col = []
    for c in range(Xl.shape[1]):
        lo = float(min(Xl[:, c].min(), Xi[:, c].min()))
        hi = float(max(Xl[:, c].max(), Xi[:, c].max()))
        if hi - lo < 1e-12:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        hl, _ = np.histogram(Xl[:, c], bins=bins)
        hi_, _ = np.histogram(Xi[:, c], bins=bins)
        pl = hl / max(hl.sum(), 1)
        pi = hi_ / max(hi_.sum(), 1)
        tv_per_col.append(0.5 * np.abs(pl - pi).sum())
    return float(max(tv_per_col)) if tv_per_col else 0.0


def tost_p(p_hat: float, n: int, delta: float) -> float:
    """TOST equivalence p-value (Wald, balanced 2-class) at margin delta."""
    se_lo = (p_hat * (1 - p_hat) / n) ** 0.5
    z_lo = (p_hat - (0.5 - delta)) / max(se_lo, 1e-12)
    z_up = ((0.5 + delta) - p_hat) / max(se_lo, 1e-12)
    p_lo = 1 - stats.norm.cdf(z_lo)
    p_up = 1 - stats.norm.cdf(z_up)
    return max(p_lo, p_up)


def run_one(scale: float, n_train: int, n_test: int, n_accounts: int,
            seeds: list[int]) -> dict:
    pb, pab = _patch_noise(scale)
    corruptor_mod._corrupt_balance = pb
    corruptor_mod._corrupt_account_balance = pab
    try:
        accs_v, accs_s, tvs = [], [], []
        all_legal_test, all_illegal_test = [], []
        for seed in seeds:
            train_states, train_labels, _ = make_pairs(n_train, n_accounts, seed)
            test_states, test_labels, _   = make_pairs(n_test,  n_accounts, seed + 10_000)

            Xv_tr = featurize(train_states, state_to_values_vector)
            Xv_te = featurize(test_states,  state_to_values_vector)
            clf_v = fit_xgb(Xv_tr, np.array(train_labels), seed)
            pred_v = clf_v.predict(Xv_te)
            accs_v.append(float((pred_v == np.array(test_labels)).mean()))

            Xs_tr = featurize(train_states, state_to_schema_vector)
            Xs_te = featurize(test_states,  state_to_schema_vector)
            clf_s = fit_xgb(Xs_tr, np.array(train_labels), seed)
            pred_s = clf_s.predict(Xs_te)
            accs_s.append(float((pred_s == np.array(test_labels)).mean()))

            legal = [s for s, y in zip(test_states, test_labels) if y == 1]
            illegal = [s for s, y in zip(test_states, test_labels) if y == 0]
            tvs.append(estimate_tv_one_way(legal, illegal))
            all_legal_test += legal
            all_illegal_test += illegal
    finally:
        corruptor_mod._corrupt_balance = ORIGINAL_BALANCE
        corruptor_mod._corrupt_account_balance = ORIGINAL_ACCOUNT_BALANCE

    n_pooled = len(accs_v) * 2 * n_test
    p_hat = float(np.mean(accs_v))
    return {
        "noise_scale": scale,
        "values_acc_mean": p_hat,
        "values_acc_std": float(np.std(accs_v, ddof=0)),
        "schema_acc_mean": float(np.mean(accs_s)),
        "schema_acc_std": float(np.std(accs_s, ddof=0)),
        "max_1way_tv_mean": float(np.mean(tvs)),
        "max_1way_tv_max": float(np.max(tvs)),
        "tost_p_at_delta_002": tost_p(p_hat, n_pooled, 0.02),
        "tost_p_at_delta_005": tost_p(p_hat, n_pooled, 0.05),
        "n_pooled_test_states": n_pooled,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-scales", type=float, nargs="+",
                   default=[0.001, 0.005, 0.010, 0.025, 0.050])
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test",  type=int, default=250)
    p.add_argument("--n-accounts", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=str,
                   default="artifacts/bank_sensitivity_sweep.csv")
    args = p.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in args.noise_scales:
        t0 = time.time()
        print(f"[scale={scale:.4f}] starting...", flush=True)
        row = run_one(scale, args.n_train, args.n_test, args.n_accounts, args.seeds)
        row["elapsed_s"] = round(time.time() - t0, 1)
        print(
            f"[scale={scale:.4f}] values={row['values_acc_mean']:.4f}±{row['values_acc_std']:.4f}  "
            f"schema={row['schema_acc_mean']:.4f}  "
            f"max_1way_tv≈{row['max_1way_tv_mean']:.4f}  "
            f"TOST p(δ=.02)={row['tost_p_at_delta_002']:.4g}  "
            f"TOST p(δ=.05)={row['tost_p_at_delta_005']:.4g}  "
            f"({row['elapsed_s']}s)",
            flush=True,
        )
        rows.append(row)

    cols = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
