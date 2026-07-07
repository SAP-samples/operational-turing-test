"""
RDB-PFN inference baseline for the Operational Turing Test.

This runner uses the lightweight standalone inference package from:
  external/RDBPFN/inference

It evaluates RDB-PFN on the same fixed-size state feature tiers used by the
pilot:

  values-only   — column/value statistics only
  relation-only — values + joins/FK topology/group/neighborhood summaries
  schema-aware  — values + executable operational predicate features

Important: RDB-PFN's current standalone inference package accepts flat
numpy/pandas inputs. This is therefore a PFN-style baseline over our feature
tiers, not yet a native multi-table RDB-PFN preprocessing run.

Outputs
-------
  artifacts/rdbpfn_results.csv
  artifacts/rdbpfn_figure.png

Usage
-----
    ../external/RDBPFN/inference/.venv/bin/python scripts/run_rdbpfn.py
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

ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PILOT_ROOT / "artifacts"
DEFAULT_RDBPFN_INFERENCE = ROOT / "external" / "RDBPFN" / "inference"

sys.path.insert(0, str(PILOT_ROOT / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.features import (
    state_to_values_vector,
    state_to_relation_vector,
    state_to_schema_vector,
)


N_TRAIN = 100
N_TEST = 50
N_SEEDS = 3
N_CUSTOMERS = 200
VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)

FEATURE_SETS = {
    "RDB-PFN values-only": state_to_values_vector,
    "RDB-PFN relation-only": state_to_relation_vector,
    "RDB-PFN schema-aware": state_to_schema_vector,
}


def import_rdbpfn(inference_dir: Path):
    if not inference_dir.exists():
        raise FileNotFoundError(f"RDBPFN inference directory not found: {inference_dir}")
    sys.path.insert(0, str(inference_dir))
    from src.predictor import RDBPFNClassifier
    return RDBPFNClassifier


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


def accuracy(labels: list[int], preds: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels) == np.asarray(preds)))


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
    ax.set_title("RDB-PFN baseline over state feature tiers\n"
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
    inference_dir: Path,
    n_train: int,
    n_test: int,
    n_seeds: int,
    n_customers: int,
    feature_sets: list[str] | None,
) -> int:
    RDBPFNClassifier = import_rdbpfn(inference_dir)
    ARTIFACTS.mkdir(exist_ok=True)

    selected = FEATURE_SETS
    if feature_sets:
        wanted = set(feature_sets)
        selected = {name: fn for name, fn in FEATURE_SETS.items() if name.replace("RDB-PFN ", "") in wanted}
        missing = wanted - {name.replace("RDB-PFN ", "") for name in selected}
        if missing:
            raise ValueError(f"Unknown feature set(s): {sorted(missing)}")

    rows: list[dict] = []
    print("RDB-PFN baseline")
    print(f"  inference_dir={inference_dir}")
    print(f"  n_train_pairs={n_train}, n_test_pairs={n_test}, seeds={n_seeds}")
    print("  mode=flat standalone inference over pilot feature tiers")

    for global_seed in range(n_seeds):
        t0 = time.time()
        seed = global_seed * 10_000
        print(f"\nSeed {global_seed}/{n_seeds - 1} — building datasets...")
        train_states, train_labels, _ = build_dataset(n_train, seed, n_customers)
        test_states, test_labels, test_vts = build_dataset(n_test, seed + 5_000, n_customers)
        print(f"  built {len(train_states)} train / {len(test_states)} test states ({time.time() - t0:.0f}s)")

        for name, featurizer in selected.items():
            t1 = time.time()
            X_train = np.vstack([featurizer(S) for S in train_states]).astype(np.float32)
            X_test = np.vstack([featurizer(S) for S in test_states]).astype(np.float32)
            print(f"  {name:<24} features={X_train.shape[1]} fitting/predicting...", end=" ", flush=True)
            clf = RDBPFNClassifier.from_pretrained("RDBPFN")
            clf.fit(X_train, np.asarray(train_labels))
            preds = clf.predict(X_test)
            acc = accuracy(test_labels, preds)
            recalls = per_violation_recall(test_labels, test_vts, preds)
            print(f"acc={acc:.4f} ({time.time() - t1:.0f}s)")
            rows.append({
                "seed": global_seed,
                "baseline": name,
                "accuracy": acc,
                "n_train_pairs": n_train,
                "n_test_pairs": n_test,
                "n_features": X_train.shape[1],
                **recalls,
            })

    df = pd.DataFrame(rows)
    csv_path = ARTIFACTS / "rdbpfn_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved -> {csv_path}")

    plot_results(df, ARTIFACTS / "rdbpfn_figure.png", n_train, n_test, n_seeds)

    print("\nMean accuracy:")
    summary = df.groupby("baseline")["accuracy"].agg(["mean", "std"])
    for baseline, row in summary.iterrows():
        print(f"  {baseline:<24} mean={row['mean']:.4f}  std={row['std']:.4f}")

    print("\nMean illegal-state recall by violation:")
    for baseline in df["baseline"].drop_duplicates():
        sub = df[df["baseline"] == baseline]
        vals = {vt: sub[f"recall_{vt}"].mean() for vt in VIOLATION_TYPES_SORTED}
        joined = "  ".join(f"{vt.replace('_break', '')}={v:.3f}" for vt, v in vals.items())
        print(f"  {baseline:<24} {joined}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_RDBPFN_INFERENCE)
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="training legal/illegal pairs")
    parser.add_argument("--n-test", type=int, default=N_TEST, help="test legal/illegal pairs")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-customers", type=int, default=N_CUSTOMERS)
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=["values-only", "relation-only", "schema-aware"],
        help="Run only the selected feature set; may be repeated.",
    )
    args = parser.parse_args()
    raise SystemExit(main(
        inference_dir=args.inference_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        n_seeds=args.n_seeds,
        n_customers=args.n_customers,
        feature_sets=args.feature_set,
    ))
