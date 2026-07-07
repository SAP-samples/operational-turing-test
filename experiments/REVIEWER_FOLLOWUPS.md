# Reviewer-followup experiments — what to run

Three reviewer concerns require new runs. Code is in place; commands below.

## 1. Banking-ladder rows + relational tier (no API key)

**What is missing:** the original `run_bank_ladder.py` runs only values-only XGBoost, schema (audit features), and oracle. Reviewer flagged that rows-only and relational tiers are not reported on the banking schema, so Table 3 only confirms the easy part of the access ladder.

**New script:** `experiments/run_bank_ladder_extended.py` adds:
- rows-only XGBoost (per-row predictions on `transaction_lines`, majority vote)
- relational XGBoost (values + relational structure features defined in `src/bank_pilot/relational_features.py`)

**Run** (paper-config; ~15 min on a laptop CPU):
```bash
python experiments/run_bank_ladder_extended.py \
    --n-train 1000 --n-test 500 --n-accounts 200 \
    --seeds 0 1 2 3 4 \
    --output artifacts/bank_ladder_extended_results.csv
```

Expected pattern (from smoke test): values ≈ 0.50, rows ≈ 0.50, relation ≈ 0.75 (recovers FK/cardinality, misses balance/derivation), schema ≈ 1.00, oracle = 1.00.

## 2. Larger-n high-effort sweep (needs OPENAI_API_KEY)

**What is missing:** the high-effort sweep at n=10 is consistent with a true LEGAL-rate up to ~25% (Wilson 95% upper bound for 0/10 ≈ 0.25). Reviewer wants a tighter bound.

**Run** (n=50 at high effort; 5x cost of the n=10 sweep):
```bash
python experiments/run_gpt55_baseline.py \
    --reasoning-effort high \
    --max-states 50 \
    --export-dir artifacts/relational_export \
    --output artifacts/gpt55_high_effort_n50.csv \
    --traces-dir artifacts/gpt55_high_effort_n50_traces
```

If cost is a concern: `--max-states 30` (3x current cost) tightens the upper bound to ~10%.

## 3. Two new prompt framings (needs OPENAI_API_KEY and/or MOONSHOT_API_KEY)

**What is missing:** the original sweep used two prompts (`find_violations`, `verify_legality`). Reviewer wants chain-of-thought and "list rules then check each" variants to close the obvious "didn't try CoT" objection.

**Two new modes added** (`--prompt-mode cot_each_rule` and `--prompt-mode list_then_check`) to both `run_gpt55_baseline.py` and `run_kimi_baseline.py`. The response parser was hardened to extract the JSON verdict from the LAST line, tolerating reasoning prose before it.

**Run** (n=50 each, default reasoning effort; ~same cost as the original prompt sweep):
```bash
# GPT-5.5
python experiments/run_gpt55_baseline.py --prompt-mode cot_each_rule \
    --max-states 50 --export-dir artifacts/relational_export \
    --output artifacts/gpt55_cot_each_rule.csv \
    --traces-dir artifacts/gpt55_cot_each_rule_traces

python experiments/run_gpt55_baseline.py --prompt-mode list_then_check \
    --max-states 50 --export-dir artifacts/relational_export \
    --output artifacts/gpt55_list_then_check.csv \
    --traces-dir artifacts/gpt55_list_then_check_traces

# Kimi
python experiments/run_kimi_baseline.py --prompt-mode cot_each_rule \
    --max-states 50 --export-dir artifacts/relational_export \
    --output artifacts/kimi_cot_each_rule.csv \
    --traces-dir artifacts/kimi_cot_each_rule_traces

python experiments/run_kimi_baseline.py --prompt-mode list_then_check \
    --max-states 50 --export-dir artifacts/relational_export \
    --output artifacts/kimi_list_then_check.csv \
    --traces-dir artifacts/kimi_list_then_check_traces
```

## Notes

- All scripts support `--dry-run` (build prompts, do not call the API). Use this to validate the new prompts and see the exact text that will be sent.
- The `relational_export` directory is the n=50 test manifest used in the paper. If only `relational_export_n50` exists locally, point `--export-dir` at it.
- Token budget defaults to 65536 (`--max-completion-tokens`); the new CoT prompt may consume more reasoning tokens than the find_violations prompt — keep an eye on truncation in the traces.
