"""
Scaling curve: values-only accuracy vs. training set size.

Tests whether more labeled pairs help a values-only model eventually beat chance.
The identifiability claim predicts a flat curve at 0.50 regardless of N_TRAIN.
The schema-aware model should saturate quickly.

Produces:
  artifacts/scaling_curve_results.csv   — accuracy by (baseline, n_train, seed)
  artifacts/scaling_curve_figure.png    — line plot with error bands

Usage:
    python scripts/run_scaling_curve.py [--skip-gate-check]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.baselines import XGBValuesOnly, XGBSchemaAware

# ── configuration ─────────────────────────────────────────────────────────────
N_TRAIN_SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
N_TEST        = 500
N_SEEDS       = 3
N_CUSTOMERS   = 200
ARTIFACTS     = Path(__file__).resolve().parents[1] / "artifacts"

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)

BASELINES = {
    "XGBoost values-only": XGBValuesOnly,
    "XGBoost + schema":    XGBSchemaAware,
}


def build_dataset(
    n_pairs: int,
    seed_offset: int,
    n_customers: int,
) -> tuple[list[dict], list[int]]:
    rng    = np.random.default_rng(seed_offset)
    states : list[dict] = []
    labels : list[int]  = []

    for i in range(n_pairs):
        seed    = seed_offset + i
        S       = generate_legal_state(n_customers=n_customers, seed=seed)
        vt      = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 1_000_000)
        states += [S, S_prime]
        labels += [1, 0]

    return states, labels


def main(skip_gate_check: bool = False) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("Run: python scripts/validate_construction.py")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    # Pre-build the largest test set per seed (reused across all N_TRAIN sizes).
    # Seed offset uses a range far from training seeds to avoid overlap.
    TEST_SEED_OFFSET  = 80_000_000
    TRAIN_SEED_OFFSET =  2_000_000

    print(f"Scaling curve: N_TRAIN ∈ {N_TRAIN_SIZES}, N_TEST={N_TEST}, N_SEEDS={N_SEEDS}")
    print(f"Pre-building {N_SEEDS} test sets…")
    test_sets: list[tuple[list[dict], list[int]]] = []
    for seed in range(N_SEEDS):
        ts, tl = build_dataset(N_TEST, TEST_SEED_OFFSET + seed * 10_000, N_CUSTOMERS)
        test_sets.append((ts, tl))
    print(f"  done.\n")

    max_train = max(N_TRAIN_SIZES)

    for seed in range(N_SEEDS):
        t0 = time.time()
        print(f"Seed {seed}/{N_SEEDS - 1} — building max training set ({max_train} pairs)…")
        all_train_states, all_train_labels = build_dataset(
            max_train, TRAIN_SEED_OFFSET + seed * 100_000, N_CUSTOMERS
        )
        print(f"  built {len(all_train_states)} states ({time.time()-t0:.0f}s)")

        test_states, test_labels = test_sets[seed]

        for n_train in N_TRAIN_SIZES:
            # Use the first n_train pairs from the pre-built max set (consistent subsampling)
            n_rows = n_train * 2   # 2 states per pair
            train_states = all_train_states[:n_rows]
            train_labels = all_train_labels[:n_rows]

            for name, cls in BASELINES.items():
                model = cls(seed=seed)
                model.fit(train_states, train_labels)
                preds = model.predict(test_states)
                acc   = float(accuracy_score(test_labels, preds))

                all_rows.append({"seed": seed, "baseline": name, "n_train": n_train, "accuracy": acc})
                print(f"  n_train={n_train:5d}  {name:<28}  acc={acc:.4f}")

    # ── save CSV ───────────────────────────────────────────────────────────────
    results  = pd.DataFrame(all_rows)
    csv_path = ARTIFACTS / "scaling_curve_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = {"XGBoost values-only": "#e74c3c", "XGBoost + schema": "#3498db"}

    for name in BASELINES:
        sub   = results[results["baseline"] == name]
        means = sub.groupby("n_train")["accuracy"].mean()
        sds   = sub.groupby("n_train")["accuracy"].std()
        xs    = np.array(sorted(means.index))

        ax.plot(xs, means.loc[xs].values, marker="o", label=name,
                color=colors[name], linewidth=2, markersize=5)
        ax.fill_between(xs,
                        (means.loc[xs] - sds.loc[xs]).values,
                        (means.loc[xs] + sds.loc[xs]).values,
                        alpha=0.15, color=colors[name])

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)")
    ax.set_xscale("log")
    ax.set_xlabel("Training pairs (log scale)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Scaling Curve: Accuracy vs. Training Set Size\n"
                 f"(N_TEST={N_TEST} pairs, {N_SEEDS} seeds; band = ±1 SD)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(N_TRAIN_SIZES)
    ax.set_xticklabels([str(n) for n in N_TRAIN_SIZES], fontsize=9)

    plt.tight_layout()
    fig_path = ARTIFACTS / "scaling_curve_figure.png"
    plt.savefig(fig_path, dpi=150)
    print(f"  Figure saved → {fig_path}")

    # ── summary ────────────────────────────────────────────────────────────────
    print()
    print("SCALING CURVE SUMMARY")
    print(f"{'n_train':>8}  {'XGB values-only':>18}  {'XGB + schema':>14}")
    print("─" * 46)
    for n in N_TRAIN_SIZES:
        sub = results[results["n_train"] == n]
        v   = sub[sub["baseline"] == "XGBoost values-only"]["accuracy"].mean()
        s   = sub[sub["baseline"] == "XGBoost + schema"]["accuracy"].mean()
        print(f"{n:>8}  {v:>18.4f}  {s:>14.4f}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check))
