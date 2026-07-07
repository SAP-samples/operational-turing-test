"""
GPT-5.5 baseline for the Operational Turing Test.

Sibling experiment to run_kimi_baseline.py. Same prompt, same 10 OTT test
states, same per-state CSV / trace layout. Differences from the Kimi script:

  * Calls the OpenAI v1-compatible endpoint hosting GPT-5.5;
    Kimi runs against a different host (see run_kimi_baseline.py).
  * Calls via `from openai import OpenAI` with the `/openai/v1/` base URL.
  * Sends `max_completion_tokens` (the reasoning-model parameter), NOT
    `max_tokens`. Sends no `temperature` (GPT-5+ models reject it).
  * GPT-5.5 hides chain-of-thought behind the API; we save only the final
    content (`message.content`). The reasoning-token count is reported in
    `usage.completion_tokens_details.reasoning_tokens` and recorded in
    the CSV.

Usage:
    python experiments/run_gpt55_baseline.py \\
        --max-states 10 \\
        --max-completion-tokens 65536 \\
        --output artifacts/gpt55_pilot_results.csv \\
        --traces-dir artifacts/gpt55_traces
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
    """Return an OpenAI-v1-compatible base URL for Azure AI/OpenAI hosts.

    Older runs stored GPT55_ENDPOINT as the resource root and appended
    /openai/v1 here. Azure AI Foundry examples often provide the full
    https://.../openai/v1 endpoint. Accept both forms.
    """
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/"
    return f"{endpoint}/openai/v1/"


def _make_openai_client(endpoint: str, api_key: str | None):
    from openai import OpenAI as _OpenAI

    auth_mode = os.environ.get("GPT55_AUTH", "api_key").lower()
    if auth_mode in {"azure_identity", "entra", "aad"}:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GPT55_AUTH=azure_identity requires `pip install azure-identity`."
            ) from exc
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )
        return _OpenAI(base_url=_openai_base_url(endpoint), api_key=token_provider)

    if not api_key:
        raise RuntimeError("set GPT55_API_KEY in .env, or set GPT55_AUTH=azure_identity")
    return _OpenAI(base_url=_openai_base_url(endpoint), api_key=api_key)


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


AUDIT_IN_ENGLISH_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A plain-English specification of seven audit checks. Each check is the
   exact predicate the schema enforces (no trigger source, no application
   code: just the predicate, in English).
3. A database state, presented as the contents of three CSV files
   (customers, orders, order_items).

Your task: classify the state as LEGAL or ILLEGAL. The state is LEGAL iff
every one of the seven audit checks returns 0 (or the equivalent passing
condition, as specified per check). A single failing check makes the state
ILLEGAL.

Reply with a single JSON object on one line, no prose:

  {"label": "LEGAL"}                      if every check passes
  {"label": "ILLEGAL", "violation": "X"}  if any check fails, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Do not include explanation, markdown, or anything outside the JSON object."""


AUDIT_IN_ENGLISH_DESCRIPTION = """\
The state is LEGAL iff each of the following seven audit checks passes.
These are the rules in plain English; treat each as the authoritative
specification.

(1) Orphan-FK orders. Count the rows of the orders table whose customer_id
    does not appear as an id in the customers table. This count must be 0.

(2) Orphan-FK items. Count the rows of the order_items table whose order_id
    does not appear as an id in the orders table. This count must be 0.

(3) Items per order. For each order, count its rows in order_items. No
    order should have more than 20 items.

(4) Open orders per customer. For each customer, count their orders whose
    status is 'pending' or 'shipped' (these are the open statuses). No
    customer should have more than 3 open orders.

(5) Line-total derivation. For each row in order_items, the value of
    line_total must equal:
        quantity * unit_price * (1 - discount(tier, quantity))
    where tier is the customer's tier (looked up via order_id -> orders.customer_id
    -> customers.id) and discount is given by:
        tier      qty in [1,4]   qty in [5,9]   qty in [10,19]
        bronze        0.00            0.05             0.07
        silver        0.05            0.10             0.12
        gold          0.10            0.15             0.20
    The maximum violation residual across all order_items rows must be 0
    (within floating-point tolerance, e.g. 1e-6).

(6) Order-total derivation. For each order, the value of total must equal:
        (sum of line_total over its items) * (1 + tax(country))
    where country is the customer's country (looked up via orders.customer_id
    -> customers.id) and tax is given by:
        country   DE    FR    GB    JP    US    (anything else)
        tax      0.19  0.20  0.20  0.10  0.08      0.15
    The maximum violation residual across all orders must be 0 (within
    floating-point tolerance, e.g. 1e-6).

(7) Status transition. For each order with a non-null prev_status, the
    pair (prev_status, status) must lie in the allowed set:
        {(pending, shipped), (shipped, delivered),
         (pending, cancelled), (pending, pending)}.
    Orders whose prev_status is null are initial states and trivially
    satisfy this check (do not count them as violations).

If all seven checks return 0 (or the equivalent passing condition), the
state is LEGAL. Otherwise it is ILLEGAL with violation X being the family
that failed.
"""


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


# Same as PROCEDURAL_RULES_DESCRIPTION plus an explicit note about the
# null-prev_status initial-state convention. Used by the
# `find_violations_plus_convention` prompt mode to disentangle
# "source-as-text fails" from "the source is incomplete (the convention is
# not documented in the trigger code)."
PROCEDURAL_RULES_PLUS_CONVENTION = PROCEDURAL_RULES_DESCRIPTION + """
5. Initial-state convention: an order whose prev_status is null is an initial
   state and trivially satisfies the status-transition rule (do not count it
   as a violation).
"""


def build_prompt(schema_sql: str, trigger_sql: str, state_dir: Path,
                 prompt_mode: str = "find_violations") -> str:
    """Build the user message.

    For prompt_mode 'audit_in_english', omit the trigger source and
    procedural-rules block, and replace them with the plain-English audit
    specifications. This isolates rule execution from rule extraction.
    All other modes get the full schema + trigger + procedural-rules block.
    """
    customers = (state_dir / "customers.csv").read_text()
    orders = (state_dir / "orders.csv").read_text()
    items = (state_dir / "order_items.csv").read_text()

    if prompt_mode == "audit_in_english":
        rule_section = (
            "## Plain-English audit specifications\n\n"
            f"{AUDIT_IN_ENGLISH_DESCRIPTION}\n\n"
        )
    elif prompt_mode == "find_violations_plus_convention":
        # Same as find_violations but the procedural-rules block also
        # documents the null-prev_status initial-state convention.
        rule_section = (
            "## Trigger (PL/pgSQL)\n\n"
            f"```sql\n{trigger_sql}\n```\n\n"
            "## Procedural rules (application code)\n\n"
            f"{PROCEDURAL_RULES_PLUS_CONVENTION}\n\n"
        )
    else:
        rule_section = (
            "## Trigger (PL/pgSQL)\n\n"
            f"```sql\n{trigger_sql}\n```\n\n"
            "## Procedural rules (application code)\n\n"
            f"{PROCEDURAL_RULES_DESCRIPTION}\n\n"
        )

    return (
        "## Schema (DDL)\n\n"
        f"```sql\n{schema_sql}\n```\n\n"
        f"{rule_section}"
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
    p.add_argument("--model", default=os.environ.get("GPT55_DEPLOYMENT", "gpt-5.5"),
                   help="the GPT-5.5 hosting endpoint deployment name (default: env GPT55_DEPLOYMENT or gpt-5.5)")
    p.add_argument("--max-states", type=int, default=10,
                   help="Number of test states to classify (default: 10, the full pilot set)")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "artifacts" / "gpt55_results.csv",
                   help="Where to write the results CSV")
    p.add_argument("--max-completion-tokens", type=int, default=65536,
                   help="Output cap (reasoning-model parameter). Default 65536.")
    p.add_argument("--reasoning-effort", choices=["default", "minimal", "low", "medium", "high"],
                   default="default",
                   help="GPT-5+ reasoning_effort parameter. 'default' (the script default) "
                        "omits the parameter, so the API picks its own (empirically low). "
                        "'minimal'/'low'/'medium'/'high' set it explicitly.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts and dump them to stdout, do not call the API")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between API calls (rate limiting)")
    p.add_argument("--traces-dir", type=Path,
                   default=REPO_ROOT / "artifacts" / "gpt55_traces",
                   help="Directory for per-state response traces (final content only; "
                        "GPT-5.5 hides chain-of-thought behind the API)")
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
                            "cot_each_rule", "list_then_check",
                            "audit_in_english",
                            "find_violations_plus_convention"],
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
    endpoint = os.environ.get("GPT55_ENDPOINT", "").rstrip("/")
    api_key = os.environ.get("GPT55_API_KEY")

    if not args.dry_run and not endpoint:
        print("ERROR: set GPT55_ENDPOINT in .env",
              file=sys.stderr)
        return 2

    # GPT-5.5 lives behind the host's `/openai/v1/` shim, which is OpenAI
    # SDK-compatible. Use the OpenAI client; max_completion_tokens replaces
    # max_tokens, temperature is disallowed, and reasoning_content is not
    # surfaced (only the final content + a token count).
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
        "find_violations":                SYSTEM_INSTRUCTIONS,
        "verify_legality":                VERIFY_LEGALITY_INSTRUCTIONS,
        "cot_each_rule":                  COT_EACH_RULE_INSTRUCTIONS,
        "list_then_check":                LIST_THEN_CHECK_INSTRUCTIONS,
        "audit_in_english":               AUDIT_IN_ENGLISH_INSTRUCTIONS,
        "find_violations_plus_convention": SYSTEM_INSTRUCTIONS,
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
            prompt = build_prompt(schema_sql, trigger_sql, state_dir, prompt_mode=args.prompt_mode)
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
            # ---------- API call branch (OpenAI v1 client) ----------
            prompt = build_prompt(schema_sql, trigger_sql, state_dir, prompt_mode=args.prompt_mode)
            t0 = time.time()
            finish_reason = ""
            prompt_tokens = completion_tokens = reasoning_tokens = 0
            raw = ""
            reasoning = ""  # GPT-5.5 hides chain-of-thought; always blank
            try:
                # Build kwargs so we omit reasoning_effort when 'default'
                # (matches the original pilot's API call exactly) and
                # include it when explicitly set.
                completion_kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=args.max_completion_tokens,
                )
                if args.reasoning_effort != "default":
                    completion_kwargs["reasoning_effort"] = args.reasoning_effort
                resp = client.chat.completions.create(**completion_kwargs)
                # Done with prompt; release ~500 KB before next state.
                del prompt
                choice = resp.choices[0]
                raw = choice.message.content or ""
                finish_reason = choice.finish_reason or ""
                usage = resp.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                ctd = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0 if ctd else 0
                del resp, choice, usage, ctd
                parsed = parse_response(raw)
                elapsed = time.time() - t0
            except Exception as exc_inner:  # noqa: BLE001
                # OpenAI SDK normalises HTTP errors into APIError subclasses,
                # so a single except is sufficient.
                raw = f"<error: {type(exc_inner).__name__}: {exc_inner}>"
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
