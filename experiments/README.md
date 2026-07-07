# Pilot experiments

Beyond-paper experiments not in the main reproduction harness. Each runs as
a standalone script.

## `run_kimi_baseline.py` — Kimi K2 with raw rule code

The reviewer's "deeper question": can a frontier LLM, given only the
**source artifacts** (schema DDL + trigger source + a natural-language
description of the procedural rules) and a database state, classify the
state as legal or illegal? This is the natural complement to the
operationally-grounded XGBoost baseline:

- **Operationally-grounded XGBoost** uses *seven rule-derived audit
  features*. It reaches 0.9996. Each feature is a one-line audit query
  against the schema and the rule code; the features encode the rules
  deterministically.
- **Kimi-with-raw-rules** receives the rules as text and has to do the
  computation autonomously. If it succeeds, operational grounding is
  *learnable from source*. If it fails, the seven-feature gap is the open
  architectural problem the paper isolates.

### Setup

```bash
pip install -e ".[dev]"     # the tf_pilot package + pytest
cp .env.example .env
# edit .env: set KIMI_ENDPOINT / KIMI_API_KEY  (Kimi)
#            and GPT55_ENDPOINT / GPT55_API_KEY  (GPT-5.5)
```

### Prerequisite: export the test states

The pilot scripts read state CSVs from `artifacts/relational_export/test/`.
Generate that directory first:

```bash
python scripts/export_relational_dataset.py
```

This writes 10 test states (5 legal, 5 illegal across all four violation
types) under `artifacts/relational_export/test/test_NNNNNN/`.

### Pilot run (10 states, ~5 min for GPT-5.5, ~30 min for Kimi)

```bash
python experiments/run_kimi_baseline.py  --max-states 10
python experiments/run_gpt55_baseline.py --max-states 10
```

Each call passes the schema (~1k tokens), trigger (~300 tokens), procedural
rules description (~400 tokens), and three CSVs (~30k tokens for 200
customers / 600 orders / 6000 items). Well within either model's context.

### Dry-run (no API call)

```bash
python experiments/run_kimi_baseline.py --dry-run --max-states 3
```

Builds the prompts and prints prompt-length statistics. Useful to verify
the wiring before paying for tokens.

### Output

`artifacts/kimi_results.csv` with columns:

```
state_id, true_label, true_violation, pred_label, pred_violation,
correct, violation_correct, elapsed_s, raw
```

`raw` is truncated to 500 chars for the CSV; full responses are not stored.

### Scaling beyond the pilot

If the pilot is encouraging, regenerate a larger test set and rerun:

```bash
# (uses your existing pipeline)
python -m src.export_relational_test --n-states 100 --seed 0
python experiments/run_kimi_baseline.py --max-states 100
```

At 100 states × ~30k tokens × $/Mtok, budget accordingly.

### Known caveats

- **Truncation risk**: orders.csv with 600 rows occasionally exceeds tidy
  Markdown rendering inside chat models. We embed raw CSV in fenced blocks.
  If Kimi truncates, switch to JSON-formatted state dumps.
- **Format compliance**: the system prompt requests a single JSON object.
  If Kimi wraps in prose, `parse_response()` strips code fences and grabs
  the first balanced `{...}`. Genuine parse failures are recorded as
  `pred_label = "PARSE_ERROR"`.
- **Determinism**: temperature is 0.0; results should be reproducible
  modulo vendor-side nondeterminism in batched serving.

---

## `run_gpt55_baseline.py` — GPT-5.5 with raw rule code

Sibling experiment to the Kimi script. Same prompt, same OTT states, same
per-state CSV/trace layout. Differences:

- Calls the OpenAI-v1-compatible endpoint hosting GPT-5.5 via `from openai import OpenAI`
  (a different host than the one used for Kimi-K2.6).
- Sends `max_completion_tokens` (the GPT-5+ reasoning-model parameter), not
  `max_tokens`. Sends no `temperature` (rejected by reasoning models).
- GPT-5.5 hides chain-of-thought behind the API; the script saves only the
  final `message.content`. Reasoning-token *count* is reported in
  `usage.completion_tokens_details.reasoning_tokens` and recorded in the CSV.

### Setup

```bash
# Same pip install as the Kimi script; GPT-5.5 needs the same `openai` package.
pip install openai python-dotenv
```

Add to `.env`:

```
GPT55_ENDPOINT="https://<your-gpt55-host>/openai/v1"
GPT55_API_KEY="<your-gpt55-api-key>"
GPT55_DEPLOYMENT="gpt-5.5"
```

### Pilot run (10 states, ~5 min)

```bash
python experiments/run_gpt55_baseline.py --max-states 10 \
    --max-completion-tokens 65536 \
    --output artifacts/gpt55_pilot_results.csv \
    --traces-dir artifacts/gpt55_traces
```

GPT-5.5 is dramatically faster than Kimi-K2.6 (~24 s/state vs ~210 s/state).

### Reasoning effort

The script exposes `--reasoning-effort {default,minimal,low,medium,high}`.
The paper-facing 100-state control uses `--reasoning-effort high`.

---

## Matched 100-state controls

The paper reports matched 100-state source-artifact controls on the same
50-customer export (50 legal + 50 illegal states). These are the CSVs copied
into `results/` as `*_n100_*`.

### 1. Export a 50-customer test set

Smaller states (50 customers, ~150 orders, ~1500 line-items) keep the
failure-mode-relevant complexity while making the 200-trial budget tractable.

```bash
python scripts/export_relational_dataset.py \
    --n-train 0 --n-test 100 --n-customers 50 \
    --out-dir artifacts/relational_export_n50 --overwrite
```

This writes a manifest with 50 legal + 50 illegal states.

### 2. Kimi-K2.6, n=100 (50 L + 50 I)

```bash
bash experiments/run_kimi_per_state.sh \
    --export-dir artifacts/relational_export_n50 \
    --total 100 \
    --prompt-mode find_violations \
    --output artifacts/kimi_n100_findv_results.csv \
    --traces-dir artifacts/kimi_n100_findv_traces
```

### 3. GPT-5.5 high effort, n=100

```bash
python experiments/run_gpt55_baseline.py \
    --export-dir artifacts/relational_export_n50 \
    --max-states 100 \
    --prompt-mode find_violations \
    --reasoning-effort high \
    --max-completion-tokens 65536 \
    --output artifacts/gpt55_n100_high_findv_results.csv \
    --traces-dir artifacts/gpt55_n100_high_findv_traces
```

### 4. GPT-5.5 with SQL executor, n=100

```bash
python experiments/run_gpt55_tools.py \
    --export-dir artifacts/relational_export_n50 \
    --max-states 100 \
    --max-completion-tokens 65536 \
    --max-tool-calls 20 \
    --output artifacts/gpt55_n100_tools_findv_results.csv \
    --traces-dir artifacts/gpt55_n100_tools_findv_traces
```

### Smaller diagnostic controls

The control replaces the system prompt's "find any violation" framing with
"default to LEGAL unless you can identify a specific violating row":

```bash
# Kimi
bash experiments/run_kimi_per_state.sh \
    --export-dir artifacts/relational_export_n50 \
    --total 50 \
    --prompt-mode verify_legality --label-filter legal \
    --output artifacts/kimi_n50_verify_results.csv \
    --traces-dir artifacts/kimi_n50_verify_traces

# GPT-5.5
python experiments/run_gpt55_baseline.py \
    --export-dir artifacts/relational_export_n50 \
    --max-states 50 \
    --prompt-mode verify_legality --label-filter legal \
    --max-completion-tokens 65536 \
    --output artifacts/gpt55_n50_verify_results.csv \
    --traces-dir artifacts/gpt55_n50_verify_traces
```

### Notes on the per-state subprocess wrapper

`run_kimi_per_state.sh` invokes `run_kimi_baseline.py` once per state in a
fresh Python process. This is for memory safety: long Kimi reasoning traces
can balloon the resident set, and a per-state subprocess lets the OS reclaim
memory between iterations. The wrapper also includes a reclaimable-RAM
threshold check (free + inactive + speculative pages) before each launch.
GPT-5.5 is fast enough to stay in a single process.

### Reproducing the headline numbers

After the matched runs above, the `results/` directory contains the same CSVs
shipped with the repo. Sanity checks:

| Run | Accuracy | LEGAL on legal states | Illegal states labelled illegal |
|-----|----------|-----------------------|----------------------------------|
| Kimi-K2.6, n=100 | 46/100 = 0.46 | 2/50 | 44/50 |
| GPT-5.5 high effort, n=100 | 50/100 = 0.50 | 0/50 | 50/50 |
| GPT-5.5 + SQL executor, n=100 | 50/100 = 0.50 | 0/50 | 50/50 |

The key failure mode is valid-state recall: models can often flag illegal
states while still rejecting legal database states.
