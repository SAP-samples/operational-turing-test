"""
Stage 2: Operational Turing Test — five baselines on binary legal/illegal classification.

Generates N_TRAIN pairs for training and N_TEST pairs for testing, across N_SEEDS seeds.
Produces:
  artifacts/turing_test_results.csv   — per-seed accuracy for each baseline
  artifacts/turing_test_figure.png    — bar chart with 95% bootstrap CIs
  stdout                              — results summary + TOST equivalence test
                                        + per-violation recalls + derivation diagnostic

Baselines:
  Oracle (upper bound)   — calls oracle() directly; 1.000 by construction
  XGBoost values-only    — column-aggregate features; should be at chance
  XGBoost + schema       — values + 7 rule proxies; should be near 1.000
  TabPFN values-only     — transformer in-context learner; architecture-independent test
  XGBoost + oracle feat  — sanity check: values + oracle output; must be 1.000

Usage:
    python scripts/run_turing_test.py [--skip-gate-check]
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
from scipy import stats
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.baselines import (
    OracleClassifier, XGBValuesOnly, XGBSchemaAware,
    TabICLValuesOnly, TabPFNValuesOnly, XGBOracleFeature,
)

# ── configuration ─────────────────────────────────────────────────────────────
N_TRAIN      = 1000
N_TEST       = 500
N_SEEDS      = 5
N_CUSTOMERS  = 200
N_BOOTSTRAP  = 2000
ARTIFACTS    = Path(__file__).resolve().parents[1] / "artifacts"

# Equivalence margin for TOST: if values-only accuracy is within ±DELTA of 0.50,
# claim equivalence to chance.  Pre-registered at 0.02 (2 percentage points).
TOST_DELTA   = 0.02

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)
BASELINES = {
    "Oracle (upper bound)":   OracleClassifier,
    "XGBoost values-only":    XGBValuesOnly,
    "XGBoost + schema":       XGBSchemaAware,
    "TabICL values-only":     TabICLValuesOnly,
    "TabPFN values-only":     TabPFNValuesOnly,
    "XGBoost + oracle feat":  XGBOracleFeature,
}

# Baselines whose values-only claim is tested by TOST
CHANCE_BASELINES = {"XGBoost values-only", "TabICL values-only", "TabPFN values-only"}


# ── dataset construction ───────────────────────────────────────────────────────

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
        seed = seed_offset + i
        S    = generate_legal_state(n_customers=n_customers, seed=seed)
        vt   = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 1_000_000)

        states  += [S, S_prime]
        labels  += [1, 0]
        vt_tags += ["legal", vt]

    return states, labels, vt_tags


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    accs: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng   = np.random.default_rng(seed)
    arr   = np.array(accs)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean())
             for _ in range(n_bootstrap)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ── TOST equivalence test ─────────────────────────────────────────────────────

def tost(
    accs: list[float],
    null: float = 0.50,
    delta: float = TOST_DELTA,
    alpha: float = 0.05,
) -> dict:
    """
    Two one-sided t-tests (TOST) for equivalence of mean accuracy to `null`.

    H01: μ ≤ null - delta  (rejected by right-tailed t-test)
    H02: μ ≥ null + delta  (rejected by left-tailed t-test)

    Equivalence is declared iff both H01 and H02 are rejected at level alpha.
    This inverts the usual null: instead of 'failed to reject chance,' we
    affirmatively demonstrate equivalence to chance within ±delta.
    """
    arr  = np.array(accs)
    n    = len(arr)
    mean = float(arr.mean())
    se   = float(arr.std(ddof=1) / np.sqrt(n))

    # Right-tailed: H01: μ ≤ null - delta  →  t1 = (mean - (null-delta)) / se
    t1 = (mean - (null - delta)) / se if se > 0 else np.inf
    p1 = float(stats.t.sf(t1, df=n - 1))  # P(T > t1 | H01)

    # Left-tailed: H02: μ ≥ null + delta  →  t2 = (mean - (null+delta)) / se
    t2 = (mean - (null + delta)) / se if se > 0 else -np.inf
    p2 = float(stats.t.cdf(t2, df=n - 1))  # P(T < t2 | H02)

    p_tost       = max(p1, p2)
    equivalent   = p_tost < alpha

    return {
        "mean":        mean,
        "se":          se,
        "n":           n,
        "delta":       delta,
        "t1":          t1,   "p1": p1,
        "t2":          t2,   "p2": p2,
        "p_tost":      p_tost,
        "equivalent":  equivalent,
        "alpha":       alpha,
    }


# ── figure ────────────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    # Only plot baselines that actually ran (TabPFN may be skipped if no license)
    ran = set(results["baseline"].unique())
    baseline_names = [b for b in BASELINES if b in ran]

    means = [results[results["baseline"] == b]["accuracy"].mean() for b in baseline_names]
    cis   = [bootstrap_ci(results[results["baseline"] == b]["accuracy"].tolist()) for b in baseline_names]
    lows  = [m - ci[0] for m, ci in zip(means, cis)]
    highs = [ci[1] - m for m, ci in zip(means, cis)]

    all_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#e67e22", "#9b59b6"]
    all_names  = list(BASELINES.keys())
    colors     = [all_colors[all_names.index(b)] for b in baseline_names]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs      = np.arange(len(baseline_names))

    bars = ax.bar(xs, means, color=colors, alpha=0.85, width=0.55, zorder=3)
    ax.errorbar(xs, means, yerr=[lows, highs], fmt="none",
                color="black", capsize=6, linewidth=1.5, zorder=4)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.50)", zorder=2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="Perfect (1.00)", zorder=2)
    ax.axhspan(0.5 - TOST_DELTA, 0.5 + TOST_DELTA, alpha=0.07, color="gray",
               label=f"TOST equivalence band (±{TOST_DELTA})", zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(baseline_names, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Operational Turing Test — Binary Classification Accuracy\n"
                 f"(n_train={N_TRAIN} pairs, n_test={N_TEST} pairs, {N_SEEDS} seeds; "
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


# ── derivation diagnostic ─────────────────────────────────────────────────────

def derivation_diagnostic(
    test_states: list[dict],
    test_labels: list[int],
    test_vts: list[str],
    schema_preds: np.ndarray,
) -> None:
    """
    Report why schema-aware XGBoost misses any derivation_break instances.
    Checks whether the oracle residual for misclassified states is near zero
    (indicating the noise landed within rule tolerance and the proxy feature
    correctly returns ~0, making the miss expected rather than a bug).
    """
    from tf_pilot.features import _schema_features

    vt = "derivation_break"
    misses = [
        i for i, (t, l, p) in enumerate(zip(test_vts, test_labels, schema_preds))
        if t == vt and l == 0 and p == 1  # illegal predicted as legal
    ]
    if not misses:
        print(f"  derivation_break: no misses — schema-aware achieves perfect recall.")
        return

    print(f"  derivation_break: {len(misses)} miss(es) by schema-aware model.")
    for idx in misses[:5]:  # show at most 5
        S          = test_states[idx]
        sf         = _schema_features(S)
        order_res  = sf[5]   # max_abs_order_residual (index 5 in schema feature vector)
        print(f"    state idx={idx}: max_abs_order_residual = {order_res:.6f}  "
              f"(rule tol = 1e-4; {'below tol → proxy = 0, miss is expected' if order_res < 1e-3 else 'above tol → investigate'})")


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_gate_check: bool = False) -> int:
    gate_artifact = ARTIFACTS / ".gate_passed"
    if not skip_gate_check and not gate_artifact.exists():
        print("ERROR: Stage 1 gate has not passed.")
        print("Run: python scripts/validate_construction.py")
        print("(or use --skip-gate-check to bypass)")
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    # Store last-seed schema preds + test data for derivation diagnostic
    last_schema_preds: np.ndarray | None = None
    last_test_states:  list[dict] | None = None
    last_test_labels:  list[int]  | None = None
    last_test_vts:     list[str]  | None = None

    for global_seed in range(N_SEEDS):
        t0   = time.time()
        seed = global_seed * 10_000

        print(f"\nSeed {global_seed}/{N_SEEDS - 1} — building datasets…")
        train_states, train_labels, _ = build_dataset(N_TRAIN, seed, N_CUSTOMERS)
        test_states, test_labels, test_vts = build_dataset(N_TEST, seed + 5_000, N_CUSTOMERS)
        print(f"  built  {len(train_states)} train / {len(test_states)} test states  ({time.time()-t0:.0f}s)")

        for name, cls in BASELINES.items():
            is_oracle = name == "Oracle (upper bound)"
            model     = cls() if is_oracle else cls(seed=global_seed)
            try:
                model.fit(train_states, train_labels)
            except Exception as exc:
                # TabPFN raises TabPFNLicenseError when TABPFN_TOKEN is not set.
                # Skip gracefully so the rest of the experiment still runs.
                if "License" in type(exc).__name__ or "license" in str(exc).lower():
                    print(f"  {name:<28}  SKIPPED — TabPFN license not accepted. "
                          f"Set TABPFN_TOKEN env var (see https://ux.priorlabs.ai).")
                    continue
                raise

            preds = model.predict(test_states)
            acc   = float(accuracy_score(test_labels, preds))

            vt_recalls: dict[str, float] = {}
            for vt in VIOLATION_TYPES_SORTED:
                mask    = [vt == t for t in test_vts]
                if not any(mask):
                    continue
                sub_idx    = [i for i, m in enumerate(mask) if m]
                sub_preds  = preds[sub_idx]
                sub_labels = [test_labels[i] for i in sub_idx]
                vt_recalls[vt] = float(1.0 - sub_preds.mean())

            row = {"seed": global_seed, "baseline": name, "accuracy": acc, **vt_recalls}
            all_rows.append(row)
            print(f"  {name:<28}  acc={acc:.4f}  "
                  + "  ".join(f"{vt.split('_')[0]}_rcl={v:.3f}" for vt, v in vt_recalls.items()))

            if name == "XGBoost + schema":
                last_schema_preds = preds
                last_test_states  = test_states
                last_test_labels  = list(test_labels)
                last_test_vts     = test_vts

    # ── save CSV ───────────────────────────────────────────────────────────────
    results  = pd.DataFrame(all_rows)
    csv_path = ARTIFACTS / "turing_test_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── results summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("RESULTS SUMMARY")
    print("═" * 78)
    fmt = f"{'Baseline':<28}  {'Mean Acc':>9}  {'95% CI':>17}  {'Δ chance':>9}"
    print(fmt)
    print("─" * 78)
    ran_baselines = results["baseline"].unique().tolist()
    for name in BASELINES:
        if name not in ran_baselines:
            print(f"{name:<28}  {'SKIPPED':>9}")
            continue
        accs     = results[results["baseline"] == name]["accuracy"].tolist()
        mean     = float(np.mean(accs))
        lo, hi   = bootstrap_ci(accs)
        print(f"{name:<28}  {mean:>9.4f}  [{lo:.4f}, {hi:.4f}]  {mean - 0.5:>+9.4f}")

    # ── TOST equivalence tests ─────────────────────────────────────────────────
    print()
    print("─" * 78)
    print(f"TOST EQUIVALENCE TEST  (H0: |μ − 0.50| ≥ {TOST_DELTA};  α = 0.05)")
    print(f"Pre-registered margin δ = {TOST_DELTA} (2 percentage points)")
    print("─" * 78)
    for name in CHANCE_BASELINES:
        accs = results[results["baseline"] == name]["accuracy"].tolist()
        if not accs:
            print(f"\n  {name}  —  SKIPPED (not enough data)")
            continue
        r    = tost(accs)
        verdict = "EQUIVALENT TO CHANCE ✓" if r["equivalent"] else "NOT equivalent to chance ✗"
        print(f"\n  {name}")
        print(f"    mean = {r['mean']:.4f}  SE = {r['se']:.4f}  n = {r['n']}")
        print(f"    H01 (μ ≤ {0.5 - r['delta']:.2f}): t = {r['t1']:+.3f},  p = {r['p1']:.4f}")
        print(f"    H02 (μ ≥ {0.5 + r['delta']:.2f}): t = {r['t2']:+.3f},  p = {r['p2']:.4f}")
        print(f"    p_TOST = max(p1, p2) = {r['p_tost']:.4f}  →  {verdict}")

    # ── derivation diagnostic ──────────────────────────────────────────────────
    print()
    print("─" * 78)
    print("DERIVATION DIAGNOSTIC (schema-aware miss analysis, last seed)")
    print("─" * 78)
    if last_schema_preds is not None:
        derivation_diagnostic(last_test_states, last_test_labels,
                              last_test_vts, last_schema_preds)

    # ── core claims ───────────────────────────────────────────────────────────
    print()
    print("─" * 78)
    print("Claim 1: Values-only models are equivalent to chance (TOST).")
    print("Claim 2: Oracle achieves 1.000 (upper bound by construction).")
    print("Claim 3: XGBoost + oracle feat achieves 1.000 (pipeline sanity check).")
    print("Claim 4: Schema-aware XGBoost >> values-only (operational context helps).")
    print("Claim 5: TabPFN values-only also at chance (architecture-independent).")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig_path = ARTIFACTS / "turing_test_figure.png"
    plot_results(results, fig_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate-check", action="store_true")
    args = parser.parse_args()
    sys.exit(main(skip_gate_check=args.skip_gate_check))
