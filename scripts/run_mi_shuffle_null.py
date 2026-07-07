"""
MI shuffle-null: converts "below estimator noise floor" into a quantitative
empirical statement.

Shuffles the legal/illegal labels 1000 times on the actual feature matrix
and recomputes total MI each time.  Reports where the observed MI sits in
the empirical null distribution (z-score and percentile).

This is the rigorous version of the null comparison in run_mi_probe.py.
Gaussian-null gives the right qualitative answer; shuffle-null uses the
actual feature distribution, making it exact and reviewer-resistant.

Outputs:
  artifacts/mi_shuffle_null_results.csv  — null distribution + per-violation stats
  artifacts/mi_shuffle_null_figure.png   — histogram of null with observed marked

Usage:
    python scripts/run_mi_shuffle_null.py [--skip-gate-check]
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
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.features import state_to_values_vector

ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"
N_PAIRS     = 500
N_CUSTOMERS = 200
N_SHUFFLES  = 1000
SEED_OFFSET = 9_000_000   # same as run_mi_probe.py

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)
H_Y = float(np.log(2))


def build_Xy(vt: str, seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for i in range(N_PAIRS):
        seed    = seed_offset + i
        S       = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 500_000)
        rows   += [state_to_values_vector(S), state_to_values_vector(S_prime)]
        labels += [1, 0]
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=int)


def total_mi(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    mi = mutual_info_classif(X, y, random_state=seed)
    return float(np.maximum(mi, 0.0).sum())


def main(skip_gate_check: bool = False) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    print(f"MI shuffle-null  (n_pairs={N_PAIRS}, n_shuffles={N_SHUFFLES})")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, vt in zip(axes_flat, VIOLATION_TYPES_SORTED):
        seed_off = SEED_OFFSET + VIOLATION_TYPES_SORTED.index(vt) * 100_000
        print(f"  [{vt}] building dataset…", end=" ", flush=True)
        X, y = build_Xy(vt, seed_off)
        print(f"{X.shape[0]} states", flush=True)

        obs_mi = total_mi(X, y, seed=42)
        print(f"    observed MI = {obs_mi:.6f} nats  (normalized {obs_mi/H_Y:.4f})")

        print(f"    running {N_SHUFFLES} label shuffles…", end=" ", flush=True)
        rng      = np.random.default_rng(seed_off)
        null_mis = []
        for s in range(N_SHUFFLES):
            y_perm = rng.permutation(y)
            null_mis.append(total_mi(X, y_perm, seed=s))
        null_mis = np.array(null_mis)
        print("done")

        null_mean = float(null_mis.mean())
        null_std  = float(null_mis.std())
        z_score   = (obs_mi - null_mean) / null_std if null_std > 0 else 0.0
        percentile = float((null_mis >= obs_mi).mean())   # fraction of null >= observed

        print(f"    null mean={null_mean:.6f}  std={null_std:.6f}")
        print(f"    z-score = {z_score:.3f}  (obs below null mean: {obs_mi < null_mean})")
        print(f"    percentile in null: {100*(1-percentile):.1f}th  "
              f"(p={percentile:.4f}, fraction of shuffles with MI ≥ observed)")
        print()

        all_rows.append({
            "violation_type": vt,
            "observed_mi":    obs_mi,
            "null_mean":      null_mean,
            "null_std":       null_std,
            "z_score":        z_score,
            "p_shuffle":      percentile,
        })

        # Plot
        ax.hist(null_mis, bins=40, color="#95a5a6", alpha=0.8, edgecolor="white",
                label=f"Shuffle null (n={N_SHUFFLES})")
        ax.axvline(obs_mi, color="#e74c3c", linewidth=2.0,
                   label=f"Observed ({obs_mi:.4f} nats)")
        ax.axvline(null_mean, color="#2c3e50", linewidth=1.0, linestyle="--",
                   label=f"Null mean ({null_mean:.4f})")
        ax.set_title(vt.replace("_", " "), fontsize=10)
        ax.set_xlabel("Total MI (nats)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7)

    # ── Pooled ─────────────────────────────────────────────────────────────────
    print("  [pooled] building combined dataset…", end=" ", flush=True)
    Xs, ys = [], []
    for vt in VIOLATION_TYPES_SORTED:
        X, y = build_Xy(vt, SEED_OFFSET + VIOLATION_TYPES_SORTED.index(vt) * 100_000)
        Xs.append(X); ys.append(y)
    X_all = np.vstack(Xs); y_all = np.concatenate(ys)
    print(f"{X_all.shape[0]} states")

    obs_pool = total_mi(X_all, y_all, seed=42)
    print(f"  pooled observed MI = {obs_pool:.6f} nats  (normalized {obs_pool/H_Y:.4f})")
    print(f"  running {N_SHUFFLES} shuffles on pooled data…", end=" ", flush=True)
    rng_pool   = np.random.default_rng(0)
    null_pool  = np.array([total_mi(X_all, rng_pool.permutation(y_all), seed=s)
                           for s in range(N_SHUFFLES)])
    print("done")

    null_pool_mean = float(null_pool.mean())
    null_pool_std  = float(null_pool.std())
    z_pool = (obs_pool - null_pool_mean) / null_pool_std if null_pool_std > 0 else 0.0
    p_pool = float((null_pool >= obs_pool).mean())

    print()
    print("═" * 72)
    print("POOLED SHUFFLE-NULL SUMMARY")
    print("═" * 72)
    print(f"  Observed MI      = {obs_pool:.6f} nats  (normalized {obs_pool/H_Y:.4f})")
    print(f"  Null mean        = {null_pool_mean:.6f} nats")
    print(f"  Null std         = {null_pool_std:.6f} nats")
    print(f"  z-score          = {z_pool:.3f}")
    print(f"  p (shuffle test) = {p_pool:.4f}")
    print()
    if obs_pool < null_pool_mean:
        print(f"  INTERPRETATION: Observed MI ({obs_pool:.4f}) < shuffle-null mean ({null_pool_mean:.4f}).")
        print(f"  The values-only features carry less MI than label-shuffled versions of")
        print(f"  the same data. There is no detectable signal above the noise floor.")
    else:
        sigma = (obs_pool - null_pool_mean) / null_pool_std
        print(f"  INTERPRETATION: Observed MI is {sigma:.1f}σ above null mean.")
        print(f"  Some signal may be present — inspect per-violation results.")

    # ── save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.loc[len(df)] = {
        "violation_type": "pooled",
        "observed_mi":    obs_pool,
        "null_mean":      null_pool_mean,
        "null_std":       null_pool_std,
        "z_score":        z_pool,
        "p_shuffle":      p_pool,
    }
    csv_path = ARTIFACTS / "mi_shuffle_null_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    plt.suptitle("MI Shuffle-Null Test — Total MI of Values-Only Features vs Label\n"
                 f"(n_pairs={N_PAIRS} per violation, {N_SHUFFLES} label permutations)",
                 fontsize=11)
    plt.tight_layout()
    fig_path = ARTIFACTS / "mi_shuffle_null_figure.png"
    plt.savefig(fig_path, dpi=150)
    print(f"  Figure saved → {fig_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check))
