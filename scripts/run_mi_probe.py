"""
Stage 1.5: Mutual Information Probe.

Measures mutual information between the values-only feature vector and the
legal/illegal label on a fresh test set.  This is the information-theoretic
primary result: if MI ≈ 0, no model restricted to column-value statistics can
do better than chance — classifier accuracy (Stage 2) is then a consequence,
not the primary claim.

Outputs:
  stdout  — per-violation and aggregate MI table
  artifacts/mi_probe_results.csv  — per-feature MI values for all violations

Usage:
    python scripts/run_mi_probe.py [--skip-gate-check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.features import state_to_values_vector

# ── configuration ─────────────────────────────────────────────────────────────
N_PAIRS     = 500       # pairs per violation type (1000 states each)
N_CUSTOMERS = 200
SEED_OFFSET = 9_000_000  # well away from Stage 1 and Stage 2 seeds
ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)

# H(Y) for balanced binary label = log(2) nats
H_Y = float(np.log(2))


def build_probe_dataset(
    vt: str,
    n_pairs: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a single violation type."""
    rng = np.random.default_rng(seed_offset)
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for i in range(n_pairs):
        seed   = seed_offset + i
        S      = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 500_000)

        rows  += [state_to_values_vector(S), state_to_values_vector(S_prime)]
        labels += [1, 0]

    return np.array(rows, dtype=np.float64), np.array(labels, dtype=int)


def main(skip_gate_check: bool = False) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("Run: python scripts/validate_construction.py")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    all_rows: list[dict] = []

    print(f"Stage 1.5: Mutual Information Probe")
    print(f"  n_pairs={N_PAIRS} per violation type, n_customers={N_CUSTOMERS}")
    print(f"  H(Y) = {H_Y:.4f} nats (balanced binary label)")
    print()

    aggregate_mi: dict[str, np.ndarray] = {}

    for vt in VIOLATION_TYPES_SORTED:
        print(f"  [{vt}] building dataset…", end=" ", flush=True)
        X, y = build_probe_dataset(vt, N_PAIRS, SEED_OFFSET + VIOLATION_TYPES_SORTED.index(vt) * 100_000)
        print(f"{X.shape[0]} states", flush=True)

        # sklearn's mutual_info_classif estimates MI using k-NN (Kraskov estimator)
        mi = mutual_info_classif(X, y, random_state=42)
        mi = np.maximum(mi, 0.0)  # estimator can return tiny negatives; clip to 0
        aggregate_mi[vt] = mi

        total_mi   = float(mi.sum())
        max_mi     = float(mi.max())
        max_feat   = int(mi.argmax())
        norm_total = total_mi / H_Y

        print(f"         total MI = {total_mi:.6f} nats  "
              f"max per-feature = {max_mi:.6f} nats (feat {max_feat})  "
              f"normalized total = {norm_total:.6f}")

        for feat_idx, mi_val in enumerate(mi):
            all_rows.append({
                "violation_type": vt,
                "feature_idx": feat_idx,
                "mi_nats": float(mi_val),
                "mi_normalized": float(mi_val / H_Y),
            })

    # ── aggregate over all violation types ────────────────────────────────────
    print()
    print("═" * 72)
    print("AGGREGATE (pooled across violation types)")
    print("═" * 72)

    # Pool all (X, y) pairs for an omnibus MI estimate
    print("  Building pooled dataset…", end=" ", flush=True)
    Xs, ys = [], []
    for vt in VIOLATION_TYPES_SORTED:
        seed_off = SEED_OFFSET + VIOLATION_TYPES_SORTED.index(vt) * 100_000
        X, y = build_probe_dataset(vt, N_PAIRS, seed_off)
        Xs.append(X); ys.append(y)
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"{X_all.shape[0]} states", flush=True)

    mi_all   = mutual_info_classif(X_all, y_all, random_state=42)
    mi_all   = np.maximum(mi_all, 0.0)
    total_mi = float(mi_all.sum())
    norm     = total_mi / H_Y

    # ── null baseline (estimator bias under true independence) ─────────────────
    # The KSG estimator is biased at finite n: even for truly independent X, Y it
    # returns positive values.  We quantify this bias by running the same estimator
    # on random Gaussian features (same shape) with the same label vector.
    n_null, d_null = X_all.shape
    rng_null    = np.random.default_rng(0)
    X_null      = rng_null.standard_normal((n_null, d_null))
    mi_null     = mutual_info_classif(X_null, y_all, random_state=42)
    mi_null     = np.maximum(mi_null, 0.0)
    null_total  = float(mi_null.sum())
    null_max    = float(mi_null.max())

    print(f"  Total MI (pooled)            = {total_mi:.6f} nats  (normalized {norm:.4f})")
    print(f"  Max per-feature MI (pooled)  = {float(mi_all.max()):.6f} nats")
    print(f"  Estimator null baseline      = {null_total:.6f} nats  "
          f"(KSG bias for {d_null} independent Gaussian features, n={n_null})")
    print(f"  Ratio obs / null             = {total_mi / null_total:.4f}x")
    print()
    if total_mi < null_total:
        print("  INTERPRETATION: Observed MI < estimator null baseline.")
        print("  The values-only features carry less apparent MI than random Gaussian")
        print("  noise of the same dimensionality and sample size.  The KSG estimator")
        print("  cannot detect any signal above its own finite-sample bias floor.")
        print("  Primary claim: no values-only classifier can systematically exceed chance.")
    else:
        print(f"  INTERPRETATION: Observed MI ({total_mi:.4f}) > null ({null_total:.4f}).")
        print("  Some signal above estimator noise — investigate top features.")

    # ── save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = ARTIFACTS / "mi_probe_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── per-feature top-10 summary ─────────────────────────────────────────────
    print()
    print("Top-10 features by mean MI across violation types:")
    mean_mi = np.array([aggregate_mi[vt] for vt in VIOLATION_TYPES_SORTED]).mean(axis=0)
    top10   = np.argsort(mean_mi)[::-1][:10]
    for rank, feat in enumerate(top10):
        print(f"  [{rank+1:2d}] feat {feat:3d}  mean MI = {mean_mi[feat]:.6f} nats  "
              f"normalized = {mean_mi[feat]/H_Y:.6f}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check))
