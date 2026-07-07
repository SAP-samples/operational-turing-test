"""
Relation-only baseline for the Operational Turing Test.

This script adds the missing middle rung:

  values-only      — column/value statistics, no relation structure
  relation-only    — joins, FK topology, group-size distributions, neighborhoods
  schema-aware     — executable rule predicates/residuals

The relation-only tier deliberately excludes derivation residuals, hard
threshold checks, and explicit illegal-transition predicates.

Outputs
-------
  artifacts/relation_only_results.csv   — per-seed accuracy and per-violation recall
  artifacts/relation_only_figure.png    — mean accuracy comparison

Usage
-----
    python scripts/run_relation_only.py [--skip-gate-check]
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
from tf_pilot.features import state_to_values_vector, state_to_relation_vector, state_to_schema_vector


ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"
N_TRAIN     = 1000
N_TEST      = 500
N_SEEDS     = 5
N_CUSTOMERS = 200

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)
def make_tree_model(seed: int):
    """
    Prefer XGBoost when installed; otherwise use sklearn's built-in histogram
    gradient boosting so the relation experiment remains laptop-runnable.
    """
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        ), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=64,
            random_state=seed,
        ), "HistGB"


FEATURE_SETS = {
    "values-only":   state_to_values_vector,
    "relation-only": state_to_relation_vector,
    "schema-aware":  state_to_schema_vector,
}


def build_dataset(
    n_pairs: int,
    seed_offset: int,
    n_customers: int,
) -> tuple[list[dict], list[int], list[str]]:
    rng = np.random.default_rng(seed_offset)
    states: list[dict] = []
    labels: list[int] = []
    vt_tags: list[str] = []

    for i in range(n_pairs):
        seed = seed_offset + i
        S = generate_legal_state(n_customers=n_customers, seed=seed)
        vt = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 1_000_000)
        states += [S, S_prime]
        labels += [1, 0]
        vt_tags += ["legal", vt]

    return states, labels, vt_tags


def per_violation_recall(
    labels: list[int],
    vt_tags: list[str],
    preds: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for vt in VIOLATION_TYPES_SORTED:
        idx = [i for i, tag in enumerate(vt_tags) if tag == vt]
        out[f"recall_{vt}"] = float(np.mean(preds[idx] == 0)) if idx else np.nan
    return out


def plot_results(results: pd.DataFrame, output_path: Path, n_train: int, n_test: int, n_seeds: int) -> None:
    names = results["baseline"].drop_duplicates().tolist()
    means = [results.loc[results["baseline"] == name, "accuracy"].mean() for name in names]
    stds = [results.loc[results["baseline"] == name, "accuracy"].std(ddof=1) for name in names]
    ses = [0.0 if not np.isfinite(std) else std / np.sqrt(n_seeds) for std in stds]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e74c3c", "#16a085", "#3498db"][:len(names)]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=ses, capsize=6, color=colors, alpha=0.85, zorder=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)", zorder=2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Perfect (1.00)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Relation-only baseline\n"
                 f"(n_train={n_train} pairs, n_test={n_test} pairs, {n_seeds} seeds)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.legend(fontsize=9)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.025,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Figure saved -> {output_path}")


def main(
    skip_gate_check: bool = False,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    n_seeds: int = N_SEEDS,
    n_customers: int = N_CUSTOMERS,
) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("Run: python scripts/validate_construction.py")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    rows: list[dict] = []

    print("Relation-only baseline")
    print(f"  n_train_pairs={n_train}, n_test_pairs={n_test}, seeds={n_seeds}")
    print("  relation-only features: joins, FK coverage, degree stats, transition-pair frequencies")
    print("  excluded: derivation residuals, threshold checks, explicit legality predicates")

    for global_seed in range(n_seeds):
        t0 = time.time()
        seed = global_seed * 10_000
        print(f"\nSeed {global_seed}/{n_seeds - 1} — building datasets...")
        train_states, train_labels, _ = build_dataset(n_train, seed, n_customers)
        test_states, test_labels, test_vts = build_dataset(n_test, seed + 5_000, n_customers)
        print(f"  built {len(train_states)} train / {len(test_states)} test states ({time.time() - t0:.0f}s)")

        for feature_name, featurizer in FEATURE_SETS.items():
            model, model_name = make_tree_model(global_seed)
            name = f"{model_name} {feature_name}"
            t1 = time.time()
            X_train = np.vstack([featurizer(S) for S in train_states])
            X_test = np.vstack([featurizer(S) for S in test_states])
            model.fit(X_train, np.array(train_labels))
            preds = model.predict(X_test)
            acc = float(accuracy_score(test_labels, preds))
            recalls = per_violation_recall(test_labels, test_vts, preds)
            print(f"  {name:<25} acc={acc:.4f} ({time.time() - t1:.0f}s)")
            rows.append({
                "seed": global_seed,
                "baseline": name,
                "accuracy": acc,
                **recalls,
            })

    df = pd.DataFrame(rows)
    csv_path = ARTIFACTS / "relation_only_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved -> {csv_path}")

    plot_results(df, ARTIFACTS / "relation_only_figure.png", n_train, n_test, n_seeds)

    print("\nMean accuracy:")
    summary = df.groupby("baseline")["accuracy"].agg(["mean", "std"])
    for baseline, row in summary.iterrows():
        print(f"  {baseline:<25} mean={row['mean']:.4f}  std={row['std']:.4f}")

    print("\nMean illegal-state recall by violation:")
    for baseline in df["baseline"].drop_duplicates():
        sub = df[df["baseline"] == baseline]
        vals = {
            vt: sub[f"recall_{vt}"].mean()
            for vt in VIOLATION_TYPES_SORTED
        }
        joined = "  ".join(f"{vt.replace('_break', '')}={v:.3f}" for vt, v in vals.items())
        print(f"  {baseline:<25} {joined}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="training legal/illegal pairs")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="test legal/illegal pairs")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-customers", type=int, default=N_CUSTOMERS)
    args = parser.parse_args()
    raise SystemExit(main(
        skip_gate_check=args.skip_gate_check,
        n_train=args.n_train,
        n_test=args.n_test,
        n_seeds=args.n_seeds,
        n_customers=args.n_customers,
    ))
