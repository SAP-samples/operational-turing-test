"""
GPT-5.5 + SQL-tool baseline for the Operational Turing Test.

Sibling experiment to run_gpt55_baseline.py. Same OTT states, same model
endpoint, same per-state CSV/trace layout. Differences:

  * The user prompt does NOT contain the state CSVs. Instead, the model
    receives a `run_sql` function-calling tool that executes SELECT queries
    against an in-memory SQLite database loaded from the state's three CSVs.
  * The conversation runs multi-turn: while the model issues tool calls, we
    execute them and feed back results, until it returns a final JSON
    verdict or hits the per-state tool-call cap.
  * Per-state we record number of tool calls, total prompt/completion tokens,
    reasoning tokens, and the full conversation transcript (as JSON).

This closes the apples-to-apples gap with the operationally-grounded
baseline, which has SQL audit features (i.e. it has a SQL executor implicitly).
With this script the LLM has a SQL executor too.

Usage:
    python experiments/run_gpt55_tools.py \\
        --export-dir artifacts/relational_export_n50 \\
        --max-states 20 \\
        --reasoning-effort high \\
        --max-tool-calls 20 \\
        --output artifacts/gpt55_n20_tools_findv_results.csv \\
        --traces-dir artifacts/gpt55_n20_tools_findv_traces
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False

REPO_ROOT = Path(__file__).resolve().parent.parent
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
# Prompts (no CSVs in the user message; the model queries SQLite via run_sql)
# -----------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are an auditor of relational database states. \
You will be given:

1. The DDL (CREATE TABLE statements) of a three-table order-to-cash schema.
2. A PL/pgSQL trigger function that enforces the legal status-transition set.
3. A natural-language description of the procedural rules that live in
   application code (line- and order-total derivation formulas, per-customer
   open-order limit).
4. A `run_sql` tool that executes SELECT queries against an in-memory SQLite
   database loaded from the state's three CSVs (customers, orders,
   order_items). Use this tool to inspect the state. The tool runs SELECT
   queries only; no INSERT/UPDATE/DELETE. Each call returns up to 200 result
   rows (the rest are truncated; query with appropriate aggregations or
   LIMITs to stay within budget). You may call the tool as many times as
   needed.

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

Issue queries iteratively to gather evidence. When you have enough evidence,
return your verdict as a single JSON object on one line, no prose:

  {"label": "LEGAL"}                      if the state passes every check
  {"label": "ILLEGAL", "violation": "X"}  if any check fails, where X is one
                                          of fk_break, cardinality_break,
                                          derivation_break, transition_break

Do not include explanation, markdown, or anything outside the JSON object \
in your final answer."""


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


def build_user_prompt(schema_sql: str, trigger_sql: str) -> str:
    """User prompt: schema + trigger + procedural rules. No CSVs."""
    return (
        "## Schema (DDL)\n\n"
        f"```sql\n{schema_sql}\n```\n\n"
        "## Trigger (PL/pgSQL)\n\n"
        f"```sql\n{trigger_sql}\n```\n\n"
        "## Procedural rules (application code)\n\n"
        f"{PROCEDURAL_RULES_DESCRIPTION}\n\n"
        "## Database state\n\n"
        "The current state is loaded into an in-memory SQLite database with "
        "the three tables `customers`, `orders`, `order_items` and the same "
        "columns as the DDL above. Use the `run_sql` tool to inspect it. "
        "When you are confident, reply with the JSON verdict specified."
    )


SQL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Execute a SELECT query against the in-memory SQLite database "
            "loaded from the current state's three CSVs (customers, orders, "
            "order_items). SELECT only; INSERT/UPDATE/DELETE are rejected. "
            "Up to 200 rows are returned per call; the rest are truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single SELECT statement.",
                }
            },
            "required": ["query"],
        },
    },
}


# -----------------------------------------------------------------------------
# SQLite executor
# -----------------------------------------------------------------------------

MAX_RESULT_ROWS = 200


def load_state_into_sqlite(state_dir: Path) -> sqlite3.Connection:
    """Load customers/orders/order_items CSVs into an in-memory SQLite DB.

    The schema mirrors the production DDL but uses SQLite-native types so the
    same SELECTs work; foreign keys are NOT enforced (we want the database to
    accept whatever rows are in the CSVs, including illegal ones, so the
    auditor can find them).
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            country TEXT,
            tier TEXT,
            signup_date TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            status TEXT,
            prev_status TEXT,
            order_date TEXT,
            total REAL
        );
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price REAL,
            line_total REAL
        );
    """)

    def _load(table: str, csv_path: Path, columns: list[str]) -> None:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            placeholders = ",".join("?" for _ in columns)
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
            rows = [tuple(row[c] for c in columns) for row in reader]
        cur.executemany(sql, rows)

    _load("customers", state_dir / "customers.csv",
          ["id", "country", "tier", "signup_date"])
    _load("orders", state_dir / "orders.csv",
          ["id", "customer_id", "status", "prev_status", "order_date", "total"])
    _load("order_items", state_dir / "order_items.csv",
          ["id", "order_id", "product_id", "quantity", "unit_price", "line_total"])

    conn.commit()
    return conn


def run_sql_tool(conn: sqlite3.Connection, query: str) -> dict:
    """Execute a SELECT query; return {columns, rows, truncated, row_count} or {error}."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty query"}
    stripped = query.strip().rstrip(";").lstrip()
    # Reject anything other than a SELECT or WITH (CTE).
    head = stripped.split(None, 1)[0].lower() if stripped else ""
    if head not in ("select", "with"):
        return {"error": f"only SELECT/WITH queries are allowed (got: {head!r})"}
    try:
        cur = conn.cursor()
        cur.execute(stripped)
        cols = [d[0] for d in (cur.description or [])]
        all_rows = cur.fetchall()
        truncated = len(all_rows) > MAX_RESULT_ROWS
        rows = all_rows[:MAX_RESULT_ROWS]
        return {
            "columns": cols,
            "rows": [list(r) for r in rows],
            "row_count": len(all_rows),
            "truncated": truncated,
        }
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# -----------------------------------------------------------------------------
# Response parsing (final JSON verdict)
# -----------------------------------------------------------------------------

def parse_response(text: str | None) -> dict:
    if text is None:
        return {"label": "EMPTY_RESPONSE"}
    text = text.strip()
    if not text:
        return {"label": "EMPTY_RESPONSE"}
    if text.startswith("```"):
        lines = text.splitlines()
        body, in_block = [], False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                body.append(line)
        text = "\n".join(body).strip()
    # Find the first balanced JSON object.
    start = text.find("{")
    if start < 0:
        return {"label": "PARSE_ERROR", "raw": text[:200]}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {"label": "PARSE_ERROR", "raw": text[:200]}
    return {"label": "PARSE_ERROR", "raw": text[:200]}


# -----------------------------------------------------------------------------
# Manifest loader (same as baseline; duplicated to keep this script standalone)
# -----------------------------------------------------------------------------

def load_test_manifest(export_root: Path, max_states: int, label_filter: str) -> list[dict]:
    out = []
    with open(export_root / "manifest.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "test":
                continue
            if label_filter == "legal" and int(row["label"]) != 1:
                continue
            if label_filter == "illegal" and int(row["label"]) != 0:
                continue
            out.append(row)
            if len(out) >= max_states:
                break
    return out


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("GPT55_DEPLOYMENT", "gpt-5.5"))
    p.add_argument("--max-states", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "artifacts" / "gpt55_tools_results.csv")
    p.add_argument("--max-completion-tokens", type=int, default=65536)
    p.add_argument("--max-tool-calls", type=int, default=20,
                   help="Per-state cap on run_sql tool calls. Default 20.")
    p.add_argument("--reasoning-effort", choices=["default", "minimal", "low", "medium", "high"],
                   default="default",
                   help="NOTE: the host's /v1/chat/completions endpoint rejects "
                        "reasoning_effort when tools are also passed (HTTP 400, 'Function "
                        "tools with reasoning_effort are not supported for gpt-5.5'). To "
                        "combine the two you must use the /v1/responses endpoint, which "
                        "this script does not target. Leave at 'default' for the tool-loop "
                        "runs.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--traces-dir", type=Path,
                   default=REPO_ROOT / "artifacts" / "gpt55_tools_traces")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-dir", type=Path,
                   default=REPO_ROOT / "artifacts" / "relational_export")
    p.add_argument("--label-filter", choices=["all", "legal", "illegal"], default="all")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=False)
    endpoint = os.environ.get("GPT55_ENDPOINT", "").rstrip("/")
    api_key = os.environ.get("GPT55_API_KEY")

    if not args.dry_run and not endpoint:
        print("ERROR: set GPT55_ENDPOINT in .env", file=sys.stderr)
        return 2

    client = None
    if not args.dry_run:
        try:
            client = _make_openai_client(endpoint, api_key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    schema_sql = SCHEMA_SQL.read_text()
    trigger_sql = TRIGGER_SQL.read_text()
    test = load_test_manifest(args.export_dir, args.max_states, args.label_filter)

    print(f"  export_dir       = {args.export_dir}", flush=True)
    print(f"  reasoning_effort = {args.reasoning_effort}", flush=True)
    print(f"  max_tool_calls   = {args.max_tool_calls}", flush=True)
    print(f"  states           = {len(test)}", flush=True)

    if not args.dry_run:
        args.traces_dir.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "state_id", "true_label", "true_violation",
        "pred_label", "pred_violation",
        "correct", "violation_correct",
        "elapsed_s", "finish_reason", "n_tool_calls",
        "prompt_tokens", "completion_tokens", "reasoning_tokens",
        "transcript_trace",
    ]
    csv_fh = csv_writer = None
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = open(args.output, "w", newline="")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_fh.flush()
        os.fsync(csv_fh.fileno())

    n_correct = n_processed = 0
    user_prompt = build_user_prompt(schema_sql, trigger_sql)

    for i, row in enumerate(test):
        state_dir = args.export_dir / row["path"]
        true_label = "LEGAL" if int(row["label"]) == 1 else "ILLEGAL"
        true_violation = "" if int(row["label"]) == 1 else row["violation_type"]

        if args.dry_run:
            print(f"=== {row['state_id']} (true={true_label}/{true_violation}) ===")
            print(f"  user prompt length: {len(user_prompt):,} chars")
            print(user_prompt[:400] + "...")
            continue

        # ---------- resume ----------
        existing_transcript = args.traces_dir / f"{row['state_id']}.transcript.json"
        out_row = None
        if args.resume and existing_transcript.exists():
            try:
                transcript = json.loads(existing_transcript.read_text())
                final_text = transcript.get("final_content", "") or ""
                parsed = parse_response(final_text)
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
                    "elapsed_s": transcript.get("elapsed_s", ""),
                    "finish_reason": transcript.get("finish_reason", "resumed"),
                    "n_tool_calls": transcript.get("n_tool_calls", ""),
                    "prompt_tokens": transcript.get("prompt_tokens", ""),
                    "completion_tokens": transcript.get("completion_tokens", ""),
                    "reasoning_tokens": transcript.get("reasoning_tokens", ""),
                    "transcript_trace": str(existing_transcript),
                }
                print(f"[{i+1:>3}/{len(test)}] {row['state_id']}  RESUMED  "
                      f"true={true_label:<7} pred={pred_label:<13} "
                      f"viol={true_violation:<20} pred_viol={pred_violation:<20} "
                      f"tool_calls={out_row['n_tool_calls']}", flush=True)
            except Exception:
                out_row = None  # fall through to fresh API call

        if out_row is None:
            # ---------- fresh API call with tool loop ----------
            t0 = time.time()
            n_tool_calls = 0
            final_content = ""
            finish_reason = ""
            cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
            transcript_messages: list[dict] = []
            tool_call_log: list[dict] = []  # human-readable log of (query, result_summary)

            try:
                conn = load_state_into_sqlite(state_dir)
            except Exception as exc:
                print(f"[{i+1:>3}/{len(test)}] {row['state_id']}  SQLITE_LOAD_ERROR: {exc}", flush=True)
                conn = None

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ]
            transcript_messages = [dict(m) for m in messages]

            api_error = None
            try:
                while conn is not None and n_tool_calls <= args.max_tool_calls:
                    completion_kwargs = dict(
                        model=args.model,
                        messages=messages,
                        max_completion_tokens=args.max_completion_tokens,
                        tools=[SQL_TOOL_DEF],
                    )
                    if args.reasoning_effort != "default":
                        completion_kwargs["reasoning_effort"] = args.reasoning_effort

                    resp = client.chat.completions.create(**completion_kwargs)
                    choice = resp.choices[0]
                    msg = choice.message
                    finish_reason = choice.finish_reason or ""

                    usage = resp.usage
                    cumulative["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                    cumulative["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                    ctd = getattr(usage, "completion_tokens_details", None)
                    cumulative["reasoning_tokens"] += (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0

                    # Append the assistant message (must include tool_calls if any)
                    assistant_payload: dict = {"role": "assistant", "content": msg.content or ""}
                    if msg.tool_calls:
                        assistant_payload["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ]
                    messages.append(assistant_payload)
                    transcript_messages.append(dict(assistant_payload))

                    if not msg.tool_calls:
                        final_content = msg.content or ""
                        break

                    # Run each tool call, append the tool message
                    for tc in msg.tool_calls:
                        if tc.function.name != "run_sql":
                            tool_result = {"error": f"unknown tool: {tc.function.name}"}
                        else:
                            try:
                                args_obj = json.loads(tc.function.arguments or "{}")
                            except json.JSONDecodeError as exc:
                                args_obj = {"_parse_error": str(exc)}
                            query = args_obj.get("query", "") if "_parse_error" not in args_obj else ""
                            tool_result = run_sql_tool(conn, query) if query else {"error": "missing query"}
                            tool_call_log.append({
                                "query": query,
                                "result_summary": (
                                    f"error: {tool_result['error']}" if "error" in tool_result
                                    else f"row_count={tool_result['row_count']}"
                                            f"{' (truncated)' if tool_result.get('truncated') else ''}"
                                ),
                            })
                        n_tool_calls += 1

                        tool_payload = {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result, default=str)[:8000],  # cap response size
                        }
                        messages.append(tool_payload)
                        transcript_messages.append(dict(tool_payload))

                    # Soft cap: stop if we've issued too many tool calls
                    if n_tool_calls > args.max_tool_calls:
                        finish_reason = "tool_call_cap"
                        # Force one final call without tools to coerce a verdict
                        coerce_msg = {
                            "role": "user",
                            "content": (
                                f"You have issued {n_tool_calls} tool calls (the cap). "
                                "Reply now with the final JSON verdict only."
                            ),
                        }
                        messages.append(coerce_msg)
                        transcript_messages.append(dict(coerce_msg))
                        final_kwargs = dict(
                            model=args.model,
                            messages=messages,
                            max_completion_tokens=args.max_completion_tokens,
                        )
                        if args.reasoning_effort != "default":
                            final_kwargs["reasoning_effort"] = args.reasoning_effort
                        resp = client.chat.completions.create(**final_kwargs)
                        choice = resp.choices[0]
                        final_content = choice.message.content or ""
                        usage = resp.usage
                        cumulative["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                        cumulative["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                        ctd = getattr(usage, "completion_tokens_details", None)
                        cumulative["reasoning_tokens"] += (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
                        transcript_messages.append({"role": "assistant", "content": final_content})
                        break
            except Exception as exc:
                api_error = f"{type(exc).__name__}: {exc}"

            elapsed = time.time() - t0
            if conn is not None:
                conn.close()

            if api_error is not None:
                parsed = {"label": "API_ERROR", "raw": api_error}
            else:
                parsed = parse_response(final_content)

            pred_label = parsed.get("label", "?")
            pred_violation = parsed.get("violation", "")
            correct = int(pred_label == true_label)
            n_correct += correct

            # Save transcript
            transcript = {
                "state_id": row["state_id"],
                "true_label": true_label,
                "true_violation": true_violation,
                "messages": transcript_messages,
                "tool_call_log": tool_call_log,
                "final_content": final_content,
                "finish_reason": finish_reason,
                "n_tool_calls": n_tool_calls,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": cumulative["prompt_tokens"],
                "completion_tokens": cumulative["completion_tokens"],
                "reasoning_tokens": cumulative["reasoning_tokens"],
                "api_error": api_error,
            }
            transcript_path = args.traces_dir / f"{row['state_id']}.transcript.json"
            transcript_path.write_text(json.dumps(transcript, indent=2, default=str))

            out_row = {
                "state_id": row["state_id"],
                "true_label": true_label,
                "true_violation": true_violation,
                "pred_label": pred_label,
                "pred_violation": pred_violation,
                "correct": correct,
                "violation_correct": int(pred_violation == true_violation) if true_violation else "",
                "elapsed_s": round(elapsed, 2),
                "finish_reason": finish_reason or ("api_error" if api_error else ""),
                "n_tool_calls": n_tool_calls,
                "prompt_tokens": cumulative["prompt_tokens"],
                "completion_tokens": cumulative["completion_tokens"],
                "reasoning_tokens": cumulative["reasoning_tokens"],
                "transcript_trace": str(transcript_path),
            }
            print(f"[{i+1:>3}/{len(test)}] {row['state_id']}  "
                  f"true={true_label:<7} pred={pred_label:<13} "
                  f"viol={true_violation:<20} pred_viol={pred_violation:<20} "
                  f"({elapsed:.0f}s, tool_calls={n_tool_calls}, "
                  f"in={cumulative['prompt_tokens']}, "
                  f"out={cumulative['completion_tokens']}, "
                  f"think={cumulative['reasoning_tokens']}, "
                  f"finish={finish_reason})", flush=True)

        if csv_writer is not None and out_row is not None:
            csv_writer.writerow(out_row)
            csv_fh.flush()
            os.fsync(csv_fh.fileno())
        n_processed += 1
        del out_row
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
