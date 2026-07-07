"""
Kimi K2 baseline for the Operational Turing Test.

Tests the reviewer's "deeper question": can a frontier LLM, given the raw
schema (DDL) + trigger source + a database state, classify the state as
legal or illegal without any pre-computed rule-derived features? This is
the natural complement to the operationally-grounded baseline (which
uses seven hand-written audit features): if Kimi can do it from source,
operational grounding is *learnable* from the source artifacts; if it
cannot, the seven-feature gap stands as the open architectural problem
this paper isolates.

Reads the test states from artifacts/relational_export/test/, formats
each as a prompt with the schema and trigger code in the system message,
and asks Kimi for a single LEGAL/ILLEGAL classification per state.

Usage:
    pip install python-dotenv pandas
    cp .env.example .env
    # edit .env with your API endpoint and key
    python experiments/run_kimi_baseline.py \\
        --model moonshotai-Kimi-K2-Instruct \\
        --max-states 10 \\
        --output artifacts/kimi_results.csv

Set --max-states to a small number first; each call uses ~30k tokens of
context (3 CSVs of ~600 / 6000 / 200 rows) so a full evaluation on
N_TEST = 500 pairs is non-trivial in cost. Start with the 10 pilot
states already in artifacts/relational_export/test/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import csv
import gc
import sys
import time
from pathlib import Path

# pandas was previously used here for the manifest read/write. Removed to avoid
# its ~50 MB import overhead — this script now sticks to the stdlib csv module
# so it can run on a memory-constrained host without pulling pandas in.

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # noqa: D401 — soft dependency for --dry-run
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False

# The OpenAI SDK is imported lazily inside main() so --dry-run works without it.


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_ROOT = REPO_ROOT / "artifacts" / "relational_export"
SCHEMA_SQL = REPO_ROOT / "schema" / "schema.sql"
TRIGGER_SQL = REPO_ROOT / "schema" / "trigger.sql"


def _openai_base_url(endpoint: str) -> str:
    """Return an OpenAI-v1-compatible base URL for Azure AI/OpenAI hosts."""
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/"
    return f"{endpoint}/openai/v1/"


def _make_openai_client(endpoint: str, api_key: str | None):
    from openai import OpenAI as _OpenAI

    auth_mode = os.environ.get("KIMI_AUTH", os.environ.get("GPT55_AUTH", "api_key")).lower()
    if auth_mode in {"azure_identity", "entra", "aad"}:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "KIMI_AUTH=azure_identity requires `pip install azure-identity`."
            ) from exc
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )
        return _OpenAI(base_url=_openai_base_url(endpoint), api_key=token_provider)

    if not api_key:
        raise RuntimeError("set KIMI_API_KEY in .env, or set KIMI_AUTH=azure_identity")
    return _OpenAI(base_url=_openai_base_url(endpoint), api_key=api_key)


def _message_extra(message, key: str):
    """Best-effort access to provider-specific fields preserved by the SDK."""
    if hasattr(message, key):
        return getattr(message, key)
    extra = getattr(message, "model_extra", None) or {}
    if key in extra:
        return extra[key]
    if hasattr(message, "model_dump"):
        data = message.model_dump()
        if key in data:
            return data[key]
    return None


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A PL/pgSQL trigger function that enforces the legal status-transition set.
3. A natural-language description of the procedural rules that live in
   application code (line- and order-total derivation formulas, per-customer
   open-order limit).
4. A database state, presented as the contents of three CSV files
   (customers, orders, order_items).

Your task: classify the state as LEGAL or ILLEGAL.

A state is LEGAL iff every one of the following holds. A single violation \
makes the state ILLEGAL:

- Referential integrity: every orders.customer_id occurs in customers.id;
  every order_items.order_id occurs in orders.id.
- Cardinality: each order has 1-20 line items; each customer has at most 3
  open orders (status in {pending, shipped}).
- Derivation:
    line_total  = quantity * unit_price * (1 - discount(tier, quantity))
    order.total = sum(line_total over its items) * (1 + tax(country))
  with the discount and tax functions as specified in the rules block.
- Status transition: the pair (prev_status, status) of every order lies in
  the allowed set
    {(pending, shipped), (shipped, delivered),
     (pending, cancelled), (pending, pending)}.

Reply with a single JSON object on one line, no prose:

  {"label": "LEGAL"}                      if the state passes every check
  {"label": "ILLEGAL", "violation": "X"}  if any check fails, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Do not include explanation, markdown, or anything outside the JSON object."""


VERIFY_LEGALITY_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A PL/pgSQL trigger function that enforces the legal status-transition set.
3. A natural-language description of the procedural rules that live in
   application code (line- and order-total derivation formulas, per-customer
   open-order limit).
4. A database state, presented as the contents of three CSV files
   (customers, orders, order_items).

Your task: verify whether the state is LEGAL.

A state is LEGAL iff every one of the following holds. A single violation \
makes the state ILLEGAL:

- Referential integrity: every orders.customer_id occurs in customers.id;
  every order_items.order_id occurs in orders.id.
- Cardinality: each order has 1-20 line items; each customer has at most 3
  open orders (status in {pending, shipped}).
- Derivation:
    line_total  = quantity * unit_price * (1 - discount(tier, quantity))
    order.total = sum(line_total over its items) * (1 + tax(country))
  with the discount and tax functions as specified in the rules block.
- Status transition: the pair (prev_status, status) of every order lies in
  the allowed set
    {(pending, shipped), (shipped, delivered),
     (pending, cancelled), (pending, pending)}.

Default to LEGAL unless you can identify a specific row, value, or pair \
that violates one of the four rules above. If every check passes on every \
row, the state is LEGAL.

Reply with a single JSON object on one line, no prose:

  {"label": "LEGAL"}                      if every check passes on every row
  {"label": "ILLEGAL", "violation": "X"}  only if you can identify a specific
                                          violating row/value, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Do not include explanation, markdown, or anything outside the JSON object."""


COT_EACH_RULE_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A PL/pgSQL trigger function that enforces the legal status-transition set.
3. A natural-language description of the procedural rules that live in
   application code (line- and order-total derivation formulas, per-customer
   open-order limit).
4. A database state, presented as the contents of three CSV files
   (customers, orders, order_items).

Your task: classify the state as LEGAL or ILLEGAL. Reason step by step. \
For each of the four rule families below, think through whether the state \
satisfies it before reaching a final verdict:

- Rule 1 (Referential integrity): for each row of orders and order_items, \
  decide whether the referenced parent row exists.
- Rule 2 (Cardinality): count the line items per order and the open orders \
  per customer; check the bounds.
- Rule 3 (Derivation): for at least a few representative orders, recompute \
  line_total and order.total from the discount/tax tables and compare.
- Rule 4 (Status transition): for each order, look up (prev_status, status) \
  in the allowed-transition set.

After this reasoning, reply with a single JSON object on the LAST line of \
your response, no surrounding markdown:

  {"label": "LEGAL"}                      if the state passes every check
  {"label": "ILLEGAL", "violation": "X"}  if any check fails, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Your reasoning may precede the JSON line, but the final line must be the JSON \
object only."""


LIST_THEN_CHECK_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A PL/pgSQL trigger function that enforces the legal status-transition set.
3. A natural-language description of the procedural rules that live in
   application code (line- and order-total derivation formulas, per-customer
   open-order limit).
4. A database state, presented as the contents of three CSV files
   (customers, orders, order_items).

Your task: classify the state as LEGAL or ILLEGAL by following this exact \
two-step procedure.

Step 1 (LIST): write out, in your own words, the complete list of rules the \
state must satisfy. Group them by rule family (referential integrity, \
cardinality, derivation, status transition). For each rule, note the exact \
predicate that has to hold over the rows.

Step 2 (CHECK): for each rule listed in Step 1, work through the state and \
explicitly verify the predicate. State which rows you checked and what value \
you computed. Do not skip rules.

After Step 2, reply with a single JSON object on the LAST line of your \
response, no surrounding markdown:

  {"label": "LEGAL"}                      if every rule in your Step 1 list
                                          passed in Step 2
  {"label": "ILLEGAL", "violation": "X"}  if any rule failed, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Your Step 1 list and Step 2 checks may precede the JSON line, but the final \
line must be the JSON object only."""


PROCEDURAL_RULES_DESCRIPTION = """\
The following rules are not in the DDL. They live in application code and
constrain legal database states:

1. Per-customer open-order limit: a customer may have at most 3 open orders,
   where 'open' means status in {pending, shipped}.

2. Line-total derivation:
       line_total = quantity * unit_price * (1 - discount(tier, quantity))
   where discount(tier, quantity) is the following table:

         tier      qty in [1, 4]   qty in [5, 9]   qty in [10, 19]
         ----      -------------   -------------   ---------------
         bronze        0.00            0.05             0.07
         silver        0.05            0.10             0.12
         gold          0.10            0.15             0.20

   (quantity is always in [1, 20]; the maximum bucket is closed at 19, and
   quantity = 20 is a per-order cardinality bound, not a discount band.)

3. Order-total derivation:
       order.total = sum(line_total for items in this order) * (1 + tax(country))
   where tax(country) is:
       country   DE    FR    GB    JP    US    (anything else)
       tax      0.19  0.20  0.20  0.10  0.08      0.15

4. Per-order item count: each order must have between 1 and 20 line items
   (inclusive).
"""


def build_prompt(schema_sql: str, trigger_sql: str, state_dir: Path) -> str:
    """Build the user message: schema + trigger + procedural rules + state CSVs."""
    customers = (state_dir / "customers.csv").read_text()
    orders = (state_dir / "orders.csv").read_text()
    items = (state_dir / "order_items.csv").read_text()

    return (
        "## Schema (DDL)\n\n"
        f"```sql\n{schema_sql}\n```\n\n"
        "## Trigger (PL/pgSQL)\n\n"
        f"```sql\n{trigger_sql}\n```\n\n"
        "## Procedural rules (application code)\n\n"
        f"{PROCEDURAL_RULES_DESCRIPTION}\n\n"
        "## Database state to classify\n\n"
        "### customers.csv\n"
        f"```csv\n{customers}```\n\n"
        "### orders.csv\n"
        f"```csv\n{orders}```\n\n"
        "### order_items.csv\n"
        f"```csv\n{items}```\n\n"
        "Classify this state. Reply with the single JSON object specified."
    )


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------

def parse_response(text: str | None) -> dict:
    """Extract the JSON object from the model's reply, tolerating leading/trailing prose."""
    if text is None:
        return {"label": "EMPTY_RESPONSE"}
    text = text.strip()
    if not text:
        return {"label": "EMPTY_RESPONSE"}
    # Strip code fences if present.
    if text.startswith("```"):
        # take everything inside the first fenced block
        lines = text.splitlines()
        body = []
        in_block = False
        for ln in lines:
            if ln.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                body.append(ln)
        text = "\n".join(body).strip()
    # Prefer the LAST line that parses as a JSON object (CoT prompts emit
    # reasoning before the verdict; the verdict is the last line).
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    # Fall back: first { ... last } substring (original behaviour for
    # one-line JSON responses).
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        return {"label": "PARSE_ERROR", "raw": text}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"label": "PARSE_ERROR", "raw": text}


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def load_test_manifest(
    export_root: Path,
    max_states: int | None,
    label_filter: str = "all",
) -> list[dict]:
    """Return the test rows of manifest.csv as a list of plain dicts.

    `label_filter`:
      - "all":     return all test rows
      - "legal":   only rows with label == 1
      - "illegal": only rows with label == 0
    """
    rows = []
    with open(export_root / "manifest.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("split") != "test":
                continue
            if label_filter == "legal" and int(r["label"]) != 1:
                continue
            if label_filter == "illegal" and int(r["label"]) != 0:
                continue
            rows.append(r)
            if max_states is not None and len(rows) >= max_states:
                break
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("KIMI_MODEL", "Kimi-K2.6"),
                   help="model deployment name (default: env KIMI_MODEL or Kimi-K2.6)")
    p.add_argument("--max-states", type=int, default=10,
                   help="Number of test states to classify (default: 10, the full pilot set)")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "artifacts" / "kimi_results.csv",
                   help="Where to write the results CSV")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Output cap. Kimi K2.5 is a thinking model; "
                        "reasoning tokens count toward this budget. Default 4096.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts and dump them to stdout, do not call the API")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between API calls (rate limiting)")
    p.add_argument("--traces-dir", type=Path,
                   default=REPO_ROOT / "artifacts" / "kimi_traces",
                   help="Directory for full per-state response traces (one .txt per state)")
    p.add_argument("--resume", action="store_true",
                   help="Skip API call for any state whose .content.txt already exists "
                        "in --traces-dir; reconstruct the row from the saved trace.")
    p.add_argument("--export-dir", type=Path,
                   default=REPO_ROOT / "artifacts" / "relational_export",
                   help="Directory containing manifest.csv + per-state subdirs (default: "
                        "artifacts/relational_export). Use this to point at a "
                        "differently-sized export, e.g. n50.")
    p.add_argument("--prompt-mode",
                   choices=["find_violations", "verify_legality",
                            "cot_each_rule", "list_then_check"],
                   default="find_violations",
                   help="Which system prompt to use: find_violations (default; the "
                        "audit framing) or verify_legality (default-LEGAL framing, "
                        "tests for instruction-following bias).")
    p.add_argument("--label-filter", choices=["all", "legal", "illegal"],
                   default="all",
                   help="Restrict the manifest to legal-only or illegal-only states. "
                        "Useful for the verify_legality condition (legal-only).")
    args = p.parse_args()

    # ---------- env / client ----------
    load_dotenv(REPO_ROOT / ".env", override=False)
    endpoint = os.environ.get("KIMI_ENDPOINT")
    api_key = os.environ.get("KIMI_API_KEY")

    if not args.dry_run and not endpoint:
        print("ERROR: set KIMI_ENDPOINT in .env (see .env.example)",
              file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        try:
            client = _make_openai_client(endpoint, api_key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    # ---------- inputs ----------
    schema_sql = SCHEMA_SQL.read_text()
    trigger_sql = TRIGGER_SQL.read_text()
    test = load_test_manifest(args.export_dir, args.max_states, args.label_filter)
    _PROMPT_DISPATCH = {
        "find_violations": SYSTEM_INSTRUCTIONS,
        "verify_legality": VERIFY_LEGALITY_INSTRUCTIONS,
        "cot_each_rule":   COT_EACH_RULE_INSTRUCTIONS,
        "list_then_check": LIST_THEN_CHECK_INSTRUCTIONS,
    }
    system_instructions = _PROMPT_DISPATCH[args.prompt_mode]
    print(f"  export_dir   = {args.export_dir}", flush=True)
    print(f"  prompt_mode  = {args.prompt_mode}", flush=True)
    print(f"  label_filter = {args.label_filter}", flush=True)
    print(f"  states       = {len(test)}", flush=True)

    # ---------- run ----------
    if not args.dry_run:
        args.traces_dir.mkdir(parents=True, exist_ok=True)

    # ---------- incremental CSV writer ----------
    # Open the CSV up front and append a row per state, with flush + fsync.
    # If the machine crashes, completed states are already on disk; just
    # rerun with --resume to pick up where we left off.
    csv_fields = [
        "state_id", "true_label", "true_violation",
        "pred_label", "pred_violation",
        "correct", "violation_correct",
        "elapsed_s", "finish_reason",
        "prompt_tokens", "completion_tokens", "reasoning_tokens",
        "content_trace", "reasoning_trace",
    ]
    csv_fh = None
    csv_writer = None
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = open(args.output, "w", newline="")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_fh.flush()
        os.fsync(csv_fh.fileno())

    n_correct = 0
    n_processed = 0
    for i, row in enumerate(test):
        state_dir = args.export_dir / row["path"]
        true_label = "LEGAL" if int(row["label"]) == 1 else "ILLEGAL"
        true_violation = "" if int(row["label"]) == 1 else row["violation_type"]

        if args.dry_run:
            prompt = build_prompt(schema_sql, trigger_sql, state_dir)
            print(f"=== {row['state_id']} (true={true_label}/{true_violation}) ===")
            print(f"  prompt length: {len(prompt):,} chars")
            print(prompt[:500] + "...")
            print()
            del prompt
            continue

        out_row = None  # populated by either the resume branch or the API branch

        # ---------- resume: skip API call if trace already exists ----------
        existing_content = args.traces_dir / f"{row['state_id']}.content.txt"
        existing_reasoning = args.traces_dir / f"{row['state_id']}.reasoning.txt"
        if args.resume and (existing_content.exists() or existing_reasoning.exists()):
            raw_resumed = existing_content.read_text() if existing_content.exists() else ""
            parsed = parse_response(raw_resumed) if raw_resumed else {"label": "EMPTY_RESPONSE"}
            pred_label = parsed.get("label", "?")
            pred_violation = parsed.get("violation", "")
            correct = int(pred_label == true_label)
            n_correct += correct
            out_row = {
                "state_id": row["state_id"],
                "true_label": true_label,
                "true_violation": true_violation,
                "pred_label": pred_label,
                "pred_violation": pred_violation,
                "correct": correct,
                "violation_correct": int(pred_violation == true_violation) if true_violation else "",
                "elapsed_s": "",
                "finish_reason": "resumed",
                "prompt_tokens": "",
                "completion_tokens": "",
                "reasoning_tokens": "",
                "content_trace": str(existing_content) if existing_content.exists() else "",
                "reasoning_trace": str(existing_reasoning) if existing_reasoning.exists() else "",
            }
            print(f"[{i+1:>3}/{len(test)}] {row['state_id']}  RESUMED  "
                  f"true={true_label:<7} pred={pred_label:<13} "
                  f"viol={true_violation:<20} pred_viol={pred_violation:<20}",
                  flush=True)
            del raw_resumed, parsed
        else:
            # ---------- API call branch ----------
            prompt = build_prompt(schema_sql, trigger_sql, state_dir)
            t0 = time.time()
            finish_reason = ""
            prompt_tokens = completion_tokens = reasoning_tokens = 0
            raw = ""
            reasoning = ""
            try:
                completion_kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                resp = client.chat.completions.create(**completion_kwargs)
                # The full prompt is now embedded in the request; release it.
                del prompt
                choice = resp.choices[0]
                msg = choice.message
                raw = msg.content or ""
                reasoning = _message_extra(msg, "reasoning_content") or ""
                finish_reason = choice.finish_reason or ""
                usage = resp.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                ctd = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = (
                    (getattr(ctd, "reasoning_tokens", 0) or 0)
                    if ctd else (getattr(usage, "reasoning_tokens", 0) or 0)
                )
                # Done with the response object — release before parse_response.
                del resp, choice, msg, usage, ctd
                parsed = parse_response(raw)
                elapsed = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                raw = f"<error: {type(exc).__name__}: {exc}>"
                parsed = {"label": "API_ERROR"}
                elapsed = time.time() - t0

            pred_label = parsed.get("label", "?")
            pred_violation = parsed.get("violation", "")
            correct = int(pred_label == true_label)
            n_correct += correct

            # Save traces to disk immediately, then drop from memory.
            content_trace = ""
            reasoning_trace = ""
            if raw:
                content_trace = str(args.traces_dir / f"{row['state_id']}.content.txt")
                (args.traces_dir / f"{row['state_id']}.content.txt").write_text(raw)
            if reasoning:
                reasoning_trace = str(args.traces_dir / f"{row['state_id']}.reasoning.txt")
                (args.traces_dir / f"{row['state_id']}.reasoning.txt").write_text(reasoning)

            out_row = {
                "state_id": row["state_id"],
                "true_label": true_label,
                "true_violation": true_violation,
                "pred_label": pred_label,
                "pred_violation": pred_violation,
                "correct": correct,
                "violation_correct": int(pred_violation == true_violation) if true_violation else "",
                "elapsed_s": round(elapsed, 2),
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "content_trace": content_trace,
                "reasoning_trace": reasoning_trace,
            }

            print(f"[{i+1:>3}/{len(test)}] {row['state_id']}  "
                  f"true={true_label:<7} pred={pred_label:<13} "
                  f"viol={true_violation:<20} pred_viol={pred_violation:<20} "
                  f"({elapsed:.0f}s, in={prompt_tokens}, out={completion_tokens}, "
                  f"think={reasoning_tokens}, finish={finish_reason})",
                  flush=True)

            # Drop the large strings before the next iteration.
            del raw, reasoning, parsed

        # ---------- write the row immediately, then fsync ----------
        if csv_writer is not None and out_row is not None:
            csv_writer.writerow(out_row)
            csv_fh.flush()
            os.fsync(csv_fh.fileno())
        n_processed += 1
        del out_row

        # Reclaim every reference Python is still holding from this iteration.
        gc.collect()

        if args.sleep > 0:
            time.sleep(args.sleep)

    if csv_fh is not None:
        csv_fh.close()

    if args.dry_run:
        return 0

    acc = n_correct / max(1, n_processed)
    print()
    print(f"wrote {args.output}")
    print(f"accuracy: {n_correct}/{n_processed} = {acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
