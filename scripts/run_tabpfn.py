"""
TabPFN v1 (Hollmann et al. 2022) values-only baseline.

No API key required. Model weights (~35 MB) download automatically from
HuggingFace on first run and are cached in ~/.cache/tabpfn/.

Uses identical seeds and dataset construction as run_turing_test.py so results
are directly comparable and can be merged into turing_test_results.csv.

Prerequisites:
    pip install -e .
    pip install "tabpfn<2"

Outputs:
    artifacts/tabpfn_results.csv       — per-seed accuracy + per-violation recall
    artifacts/turing_test_results.csv  — updated in-place with TabPFN rows merged in
    artifacts/turing_test_figure.png   — regenerated with TabPFN bar included

Usage:
    python scripts/run_tabpfn.py                        # CPU
    python scripts/run_tabpfn.py --device cuda          # GPU (A100 etc.)
    python scripts/run_tabpfn.py --device cuda --skip-gate-check
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
from tf_pilot.baselines import TabPFNValuesOnly

# ── identical configuration to run_turing_test.py ─────────────────────────────
N_TRAIN     = 1000
N_TEST      = 500
N_SEEDS     = 5
N_CUSTOMERS = 200
N_BOOTSTRAP = 2000
ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)
BASELINE_NAME = "TabPFN values-only"

# Complete ordered baseline list (for figure layout consistency)
ALL_BASELINES_ORDERED = [
    "Oracle (upper bound)",
    "XGBoost values-only",
    "XGBoost + schema",
    "TabICL values-only",
    "TabPFN values-only",
    "XGBoost + oracle feat",
]
ALL_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#e67e22", "#9b59b6"]


# ── dataset construction (mirrors run_turing_test.py exactly) ──────────────────

def build_dataset(
    n_pairs: int,
    seed_offset: int,
    n_customers: int,
) -> tuple[list[dict], list[int], list[str]]:
    rng     = np.random.default_rng(seed_offset)
    states  : list[dict] = []
    labels  : list[int]  = []
    vt_tags : list[str]  = []

    for i in range(n_pairs):
        seed    = seed_offset + i
        S       = generate_legal_state(n_customers=n_customers, seed=seed)
        vt      = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 1_000_000)
        states  += [S, S_prime]
        labels  += [1, 0]
        vt_tags += ["legal", vt]

    return states, labels, vt_tags


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(accs: list[float], seed: int = 0) -> tuple[float, float]:
    rng   = np.random.default_rng(seed)
    arr   = np.array(accs)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean())
             for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── figure (full 5-baseline chart) ────────────────────────────────────────────

def plot_merged(results: pd.DataFrame, output_path: Path, tost_delta: float = 0.02) -> None:
    ran_names  = [b for b in ALL_BASELINES_ORDERED if b in results["baseline"].unique()]
    colors     = [ALL_COLORS[ALL_BASELINES_ORDERED.index(b)] for b in ran_names]
    means      = [results[results["baseline"] == b]["accuracy"].mean() for b in ran_names]
    cis        = [bootstrap_ci(results[results["baseline"] == b]["accuracy"].tolist())
                  for b in ran_names]
    lows       = [m - ci[0] for m, ci in zip(means, cis)]
    highs      = [ci[1] - m for m, ci in zip(means, cis)]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs      = np.arange(len(ran_names))
    bars    = ax.bar(xs, means, color=colors, alpha=0.85, width=0.55, zorder=3)
    ax.errorbar(xs, means, yerr=[lows, highs], fmt="none",
                color="black", capsize=6, linewidth=1.5, zorder=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)", zorder=2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Perfect (1.00)", zorder=2)
    ax.axhspan(0.5 - tost_delta, 0.5 + tost_delta, alpha=0.07, color="gray",
               label=f"TOST equivalence band (±{tost_delta})", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(ran_names, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.12)
    n_seeds = results["seed"].nunique()
    ax.set_title("Operational Turing Test — Binary Classification Accuracy\n"
                 f"(n_train={N_TRAIN} pairs, n_test={N_TEST} pairs, {n_seeds} seeds; "
                 "bars = mean ± 95% bootstrap CI)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Figure saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_gate_check: bool = False, device: str = "cpu") -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("Run: python scripts/validate_construction.py")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    new_rows: list[dict] = []

    print(f"TabPFN v1 values-only baseline  (device={device})")
    print(f"  n_train={N_TRAIN}  n_test={N_TEST}  n_seeds={N_SEEDS}  "
          f"max_train={TabPFNValuesOnly.MAX_TRAIN} (TabPFN v1 limit)")
    print(f"  Weights will download automatically on first run (~35 MB, HuggingFace).")
    print()

    for global_seed in range(N_SEEDS):
        t0   = time.time()
        seed = global_seed * 10_000

        print(f"Seed {global_seed}/{N_SEEDS - 1} — building datasets…")
        train_states, train_labels, _ = build_dataset(N_TRAIN, seed, N_CUSTOMERS)
        test_states, test_labels, test_vts = build_dataset(N_TEST, seed + 5_000, N_CUSTOMERS)
        print(f"  built  {len(train_states)} train / {len(test_states)} test  ({time.time()-t0:.0f}s)")

        model = TabPFNValuesOnly(seed=global_seed, device=device)
        print(f"  fitting TabPFN…", end=" ", flush=True)
        t1 = time.time()
        model.fit(train_states, train_labels)
        print(f"done ({time.time()-t1:.0f}s)")

        preds = model.predict(test_states)
        acc   = float(accuracy_score(test_labels, preds))

        vt_recalls: dict[str, float] = {}
        for vt in VIOLATION_TYPES_SORTED:
            mask    = [vt == t for t in test_vts]
            if not any(mask):
                continue
            sub_idx    = [i for i, m in enumerate(mask) if m]
            sub_preds  = preds[sub_idx]
            vt_recalls[vt] = float(1.0 - sub_preds.mean())

        row = {"seed": global_seed, "baseline": BASELINE_NAME, "accuracy": acc, **vt_recalls}
        new_rows.append(row)
        print(f"  {BASELINE_NAME:<28}  acc={acc:.4f}  "
              + "  ".join(f"{vt.split('_')[0]}_rcl={v:.3f}" for vt, v in vt_recalls.items()))

    # ── save standalone CSV ────────────────────────────────────────────────────
    tabpfn_df = pd.DataFrame(new_rows)
    tabpfn_csv = ARTIFACTS / "tabpfn_results.csv"
    tabpfn_df.to_csv(tabpfn_csv, index=False)
    print(f"\n  TabPFN results saved → {tabpfn_csv}")

    # ── merge into main results CSV ────────────────────────────────────────────
    main_csv = ARTIFACTS / "turing_test_results.csv"
    if main_csv.exists():
        main_df  = pd.read_csv(main_csv)
        # Drop any stale TabPFN rows before merging
        main_df  = main_df[main_df["baseline"] != BASELINE_NAME]
        merged   = pd.concat([main_df, tabpfn_df], ignore_index=True)
        # Restore canonical column order
        vt_cols  = [c for c in merged.columns if c not in ("seed", "baseline", "accuracy")]
        merged   = merged[["seed", "baseline", "accuracy"] + vt_cols]
        merged.to_csv(main_csv, index=False)
        print(f"  Merged into {main_csv}")
        results_for_fig = merged
    else:
        print(f"  WARNING: {main_csv} not found — run run_turing_test.py first.")
        results_for_fig = tabpfn_df

    # ── summary ────────────────────────────────────────────────────────────────
    accs = tabpfn_df["accuracy"].tolist()
    mean = float(np.mean(accs))
    lo, hi = bootstrap_ci(accs)
    print()
    print(f"  {BASELINE_NAME}")
    print(f"  mean acc = {mean:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]  Δ chance = {mean-0.5:+.4f}")

    # ── regenerate figure ──────────────────────────────────────────────────────
    fig_path = ARTIFACTS / "turing_test_figure.png"
    plot_merged(results_for_fig, fig_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device for TabPFN (default: cpu; use 'cuda' on GPU)")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check, device=args.device))
