"""
Stage 1 gate: verify that all four corruption strategies produce (S, S') pairs
whose 1- and 2-way marginals match within tau = 0.02 (TV distance).

Generates N_PAIRS state pairs per violation type, runs verify_marginal_match
on each, and prints a summary table.

Exit code 0  → gate passes; proceed to Stage 2.
Exit code 1  → gate fails; stop and report — this is a genuine finding.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── allow running as a script without `pip install -e .` ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES
from tf_pilot.verifier  import verify_marginal_match, TAU

N_PAIRS     = 50
N_CUSTOMERS = 200
SEED_OFFSET = 0          # legal states use seeds [0, N_PAIRS)
CORRUPT_SEED_OFFSET = 10_000   # corruption seeds offset to avoid overlap


def main() -> int:
    print(f"Stage 1: Marginal Match Verification")
    print(f"  tau={TAU}  n_pairs={N_PAIRS}  n_customers={N_CUSTOMERS}\n")

    summary: dict[str, dict] = {}
    gate_passed = True

    for vt in sorted(VIOLATION_TYPES):
        t0       = time.time()
        max_tvs  = []
        p_values = []
        worst_marginals: list[str] = []

        for i in range(N_PAIRS):
            seed   = SEED_OFFSET + i
            S      = generate_legal_state(n_customers=N_CUSTOMERS, seed=seed)
            S_prime = corrupt_to_illegal(S, vt, seed=CORRUPT_SEED_OFFSET + i)
            result  = verify_marginal_match(S, S_prime, order_k=2, seed=seed)
            max_tvs.append(result["max_tv"])
            p_values.append(result["p_value"])
            worst_marginals.append(result["worst_marginal"])

        arr     = np.array(max_tvs)
        passes  = bool(arr.max() <= TAU)
        elapsed = time.time() - t0

        summary[vt] = {
            "mean_max_tv":     float(arr.mean()),
            "max_max_tv":      float(arr.max()),
            "p95_max_tv":      float(np.percentile(arr, 95)),
            "min_p_value":     float(min(p_values)),
            "worst_marginal":  max(zip(max_tvs, worst_marginals))[1],
            "passes":          passes,
            "elapsed_s":       elapsed,
        }
        if not passes:
            gate_passed = False

        status = "PASS" if passes else "FAIL"
        print(f"  [{status}] {vt:<22}  "
              f"mean={arr.mean():.4f}  max={arr.max():.4f}  "
              f"p95={np.percentile(arr, 95):.4f}  "
              f"min_p={min(p_values):.3f}  ({elapsed:.0f}s)")
        if not passes:
            print(f"         worst marginal: {summary[vt]['worst_marginal']}")

    print()
    print("─" * 72)
    if gate_passed:
        print("✓ GATE PASSED — all violation types within tau=0.02, proceed to Stage 2.")
        return 0
    else:
        failing = [vt for vt, r in summary.items() if not r["passes"]]
        print(f"✗ GATE FAILED — {failing}")
        print()
        print("  This is a genuine finding, not a bug to hide.")
        print("  Examine worst_marginal above to understand which column pair")
        print("  the corruption leaves detectable signal in.  Consider:")
        print("    - reporting this as a disclosed limitation in the paper, or")
        print("    - redesigning the corruption strategy for the failing type.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
