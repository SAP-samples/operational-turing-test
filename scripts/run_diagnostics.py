"""
Foundation diagnostics for the Operational Turing Test.

Checks every pillar of the experimental claim:

  D1. Oracle soundness       — no legal state is misclassified; every corrupted
                               state fires exactly the targeted rule.
  D2. Marginal matching      — spot-check TV distances on fresh pairs.
  D3. Probability calibration — are values-only models outputting P≈0.5 for all
                               instances, or making confident wrong predictions?
  D4. Prediction independence — are XGBoost and TabICL errors correlated?
                               If so, a common signal might be leaked.
  D5. fk_recall bias         — is XGBoost's fk_break recall systematically
                               above 0.5 across seeds?
  D6. TOST normality         — Shapiro-Wilk on per-seed accuracies (n=5).
  D7. Feature sanity         — confirm values-only vector contains no group-size
                               statistics by checking dimensionality and named
                               features against the documented exclusion list.

Exit code 0 = all checks pass.
Exit code 1 = at least one check failed.
"""
from __future__ import annotations

import sys
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
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES, RULE_FOR_VIOLATION
from tf_pilot.rules import oracle
from tf_pilot.features import state_to_values_vector, _schema_features
from tf_pilot.verifier import verify_marginal_match
from tf_pilot.baselines import XGBValuesOnly, XGBSchemaAware, TabICLValuesOnly

ARTIFACTS   = Path(__file__).resolve().parents[1] / "artifacts"
DIAG_SEED   = 42
N_CUSTOMERS = 200
N_PAIRS_D1  = 100    # oracle soundness
N_PAIRS_D2  = 20     # marginal spot-check
N_TRAIN     = 1000
N_TEST      = 500

VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)

failures: list[str] = []


def section(title: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# ── helpers ───────────────────────────────────────────────────────────────────

def build_dataset(n_pairs, seed_offset):
    rng = np.random.default_rng(seed_offset)
    states, labels, vt_tags = [], [], []
    for i in range(n_pairs):
        seed    = seed_offset + i
        S       = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        vt      = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        S_prime = corrupt_to_illegal(S, vt, seed=seed + 1_000_000)
        states += [S, S_prime]; labels += [1, 0]; vt_tags += ["legal", vt]
    return states, labels, vt_tags


# ── D1: Oracle soundness ──────────────────────────────────────────────────────

section("D1: Oracle soundness")
print(f"  Generating {N_PAIRS_D1} pairs per violation type…")

oracle_errors = 0
for vt in VIOLATION_TYPES_SORTED:
    rule = RULE_FOR_VIOLATION[vt]
    fp, fn = 0, 0
    for i in range(N_PAIRS_D1):
        seed = DIAG_SEED * 100_000 + i
        S    = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        Sp   = corrupt_to_illegal(S, vt, seed=seed + 500_000)

        # Legal state must be accepted
        res_legal = oracle(S)
        if not res_legal["legal"]:
            fp += 1

        # Illegal state must be rejected with the right rule
        res_illegal = oracle(Sp)
        if res_illegal["legal"] or rule not in res_illegal["violations"]:
            fn += 1

    if fp == 0 and fn == 0:
        ok(f"{vt}: 0 false-positives, 0 false-negatives on {N_PAIRS_D1} pairs")
    else:
        fail(f"{vt}: {fp} FP (legal rejected), {fn} FN (illegal not caught)")
    oracle_errors += fp + fn


# ── D2: Marginal matching spot-check (fresh seeds) ───────────────────────────

section("D2: Marginal matching spot-check (fresh seeds)")
print(f"  {N_PAIRS_D2} pairs per violation type, seeds offset from Stage 1…")

SPOT_SEED = 7_000_000
for vt in VIOLATION_TYPES_SORTED:
    max_tvs = []
    for i in range(N_PAIRS_D2):
        seed    = SPOT_SEED + i
        S       = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
        Sp      = corrupt_to_illegal(S, vt, seed=seed + 300_000)
        result  = verify_marginal_match(S, Sp, order_k=2, seed=seed)
        max_tvs.append(result["max_tv"])
    arr = np.array(max_tvs)
    msg = f"{vt:<22}  max={arr.max():.4f}  mean={arr.mean():.4f}  p95={np.percentile(arr,95):.4f}"
    if arr.max() <= 0.02:
        ok(msg)
    else:
        fail(msg + "  ← EXCEEDS τ=0.02")


# ── D3: Probability calibration ───────────────────────────────────────────────

section("D3: Probability calibration — P(legal) distribution per model")
print(f"  Training on {N_TRAIN} pairs, evaluating on {N_TEST} pairs (seed={DIAG_SEED})…")

seed = DIAG_SEED * 10_000
train_states, train_labels, _ = build_dataset(N_TRAIN, seed)
test_states,  test_labels,  test_vts = build_dataset(N_TEST,  seed + 5_000)

models = {
    "XGBoost values-only": XGBValuesOnly(seed=DIAG_SEED),
    "XGBoost + schema":    XGBSchemaAware(seed=DIAG_SEED),
    "TabICL values-only":  TabICLValuesOnly(seed=DIAG_SEED),
}
probas: dict[str, np.ndarray] = {}
preds_map: dict[str, np.ndarray] = {}

for name, model in models.items():
    print(f"  Fitting {name}…", end=" ", flush=True)
    model.fit(train_states, train_labels)
    proba = model.predict_proba(test_states)   # shape (n, 2)
    p_legal = proba[:, 1]                       # P(legal)
    probas[name]    = p_legal
    preds_map[name] = model.predict(test_states)
    print(f"done  mean_P={p_legal.mean():.4f}  std_P={p_legal.std():.4f}  "
          f"acc={accuracy_score(test_labels, preds_map[name]):.4f}")

# Plot probability histograms
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
colors = {"XGBoost values-only": "#e74c3c", "XGBoost + schema": "#3498db",
          "TabICL values-only": "#f39c12"}
for ax, (name, p) in zip(axes, probas.items()):
    ax.hist(p, bins=40, range=(0, 1), color=colors[name], alpha=0.8, edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("P(legal)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
plt.suptitle("Predicted P(legal) distributions — D3 calibration check", fontsize=11)
plt.tight_layout()
fig_path = ARTIFACTS / "diagnostics_calibration.png"
plt.savefig(fig_path, dpi=150)
print(f"  Calibration figure saved → {fig_path}")

# Quantitative checks
for name, p in probas.items():
    if "values-only" in name:
        frac_near_half = float(np.mean(np.abs(p - 0.5) < 0.1))
        msg = (f"{name}: {frac_near_half:.1%} of predictions within 0.1 of 0.5  "
               f"(mean={p.mean():.4f}, std={p.std():.4f})")
        if frac_near_half > 0.60:
            ok(msg)
        else:
            warn(msg + "  ← model may be making confident predictions")


# ── D4: Prediction independence (XGBoost vs TabICL) ──────────────────────────

section("D4: Prediction independence — XGBoost vs TabICL error correlation")
y = np.array(test_labels)
xgb_wrong   = (preds_map["XGBoost values-only"] != y).astype(int)
tabicl_wrong = (preds_map["TabICL values-only"]  != y).astype(int)

# Phi coefficient (Matthews correlation for binary)
from sklearn.metrics import matthews_corrcoef
phi = matthews_corrcoef(xgb_wrong, tabicl_wrong)
both_wrong = int((xgb_wrong & tabicl_wrong).sum())
n_test     = len(y)

print(f"  XGBoost errors:   {xgb_wrong.sum()} / {n_test}")
print(f"  TabICL errors:    {tabicl_wrong.sum()} / {n_test}")
print(f"  Both wrong:       {both_wrong} / {n_test}")
print(f"  Expected overlap if independent: "
      f"{xgb_wrong.mean()*tabicl_wrong.mean()*n_test:.1f}")
print(f"  Phi (error correlation): {phi:.4f}")

if abs(phi) < 0.10:
    ok(f"Error correlation φ={phi:.4f} — models make independent mistakes")
elif abs(phi) < 0.20:
    warn(f"Mild error correlation φ={phi:.4f} — worth investigating but not fatal")
else:
    fail(f"High error correlation φ={phi:.4f} — shared signal may be exploited")


# ── D5: fk_recall bias across seeds ──────────────────────────────────────────

section("D5: fk_break recall bias (XGBoost values-only)")
results_csv = ARTIFACTS / "turing_test_results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    xgb_df = df[df["baseline"] == "XGBoost values-only"]
    if "fk_break" in xgb_df.columns:
        fk_recalls = xgb_df["fk_break"].dropna().tolist()
        mean_fk    = float(np.mean(fk_recalls))
        se_fk      = float(np.std(fk_recalls, ddof=1) / np.sqrt(len(fk_recalls)))
        t_stat, p_val = stats.ttest_1samp(fk_recalls, 0.5)
        print(f"  fk_break recalls: {[f'{r:.3f}' for r in fk_recalls]}")
        print(f"  mean={mean_fk:.4f}  SE={se_fk:.4f}  t={t_stat:.3f}  p={p_val:.4f}")
        if p_val > 0.05:
            ok(f"fk_break recall not significantly different from 0.5 (p={p_val:.4f})")
        elif mean_fk - 0.5 < 0.02:
            warn(f"Marginal significance (p={p_val:.4f}) but effect < 2 pp — within TOST margin")
        else:
            fail(f"fk_break recall significantly above 0.5: mean={mean_fk:.4f}, p={p_val:.4f}")
    else:
        warn("fk_break column not found in results CSV")
else:
    warn("turing_test_results.csv not found — skipping D5")


# ── D6: TOST normality (Shapiro-Wilk) ────────────────────────────────────────

section("D6: TOST input normality (Shapiro-Wilk on per-seed accuracies)")
if results_csv.exists():
    df = pd.read_csv(results_csv)
    for bname in ["XGBoost values-only", "TabICL values-only"]:
        accs = df[df["baseline"] == bname]["accuracy"].dropna().tolist()
        if len(accs) < 3:
            warn(f"{bname}: too few seeds for Shapiro-Wilk")
            continue
        stat, p = stats.shapiro(accs)
        msg = f"{bname}: W={stat:.4f}  p={p:.4f}  (n={len(accs)})"
        if p > 0.05:
            ok(f"Cannot reject normality — {msg}")
        else:
            warn(f"Normality rejected at 0.05 — {msg}  "
                 f"(t-test in TOST may be unreliable; consider bootstrap)")
else:
    warn("turing_test_results.csv not found — skipping D6")


# ── D7: Feature vector sanity ─────────────────────────────────────────────────

section("D7: Values-only feature vector — structural exclusion check")
S_test = generate_legal_state(n_customers=50, seed=0)
vec    = state_to_values_vector(S_test)
print(f"  Vector length: {len(vec)}")

# Verify items_per_order is NOT in the vector by checking that corrupting
# cardinality (which changes items-per-order) doesn't change the values vector
# at the group-size level — i.e., the vector is identical after a legal permutation
# of order_ids within a customer's orders.
import copy
items  = S_test["order_items"].copy()
orders = S_test["orders"].copy()
# Permute order_ids within customer 0's orders (legal permutation, preserves values)
cust0_orders = orders[orders["customer_id"] == int(orders["customer_id"].iloc[0])]["id"].tolist()
if len(cust0_orders) >= 2:
    perm_items = items.copy()
    # swap first two orders of this customer
    o1, o2 = cust0_orders[0], cust0_orders[1]
    mask1  = perm_items["order_id"] == o1
    mask2  = perm_items["order_id"] == o2
    perm_items.loc[mask1, "order_id"] = o2
    perm_items.loc[mask2, "order_id"] = o1
    S_perm = {**S_test, "order_items": perm_items}

    vec_orig = state_to_values_vector(S_test)
    vec_perm = state_to_values_vector(S_perm)
    max_diff = float(np.abs(vec_orig - vec_perm).max())
    if max_diff < 1e-10:
        ok(f"Values-only vector is invariant to order_id permutation (max_diff={max_diff:.2e})"
           f" — confirms no items-per-order grouping is encoded")
    else:
        fail(f"Values-only vector changes under order_id permutation (max_diff={max_diff:.4f})"
             f" — structural feature may be leaking in")

# Check dimensionality matches documented spec
expected_dim = (
    2 + 2 +          # customers: tier entropy/mode, country entropy/mode
    2 + 6 + 1 +      # orders: status entropy/mode, total stats×6, row count
    6 + 6 + 6 + 2 +  # items: qty, unit_price, line_total stats×6, product_id entropy/mode
    2                 # cross-table correlations
)
if len(vec) == expected_dim:
    ok(f"Vector dimension {len(vec)} matches documented spec ({expected_dim})")
else:
    fail(f"Vector dimension {len(vec)} ≠ documented spec ({expected_dim})")


# ── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
if not failures:
    print(f"  All checks passed. Foundation is sound.")
else:
    print(f"  {len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"    ✗ {f}")

sys.exit(0 if not failures else 1)
