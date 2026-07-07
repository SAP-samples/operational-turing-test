"""
Export generated states as a small relational dataset scaffold.

This is not a model runner. It creates a clean file layout that a future
Relational Transformer / relational GNN adapter can consume without depending
on the pilot's Python objects.

Layout
------
  artifacts/relational_export/
    schema.json
    manifest.csv
    train/state_000000/{customers.csv,orders.csv,order_items.csv}
    test/state_000000/{customers.csv,orders.csv,order_items.csv}

Usage
-----
    python scripts/export_relational_dataset.py --n-train 100 --n-test 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tf_pilot.generator import generate_legal_state
from tf_pilot.corruptor import corrupt_to_illegal, VIOLATION_TYPES


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
VIOLATION_TYPES_SORTED = sorted(VIOLATION_TYPES)


SCHEMA = {
    "tables": {
        "customers": {
            "primary_key": "id",
            "columns": {
                "id": "integer",
                "country": "categorical",
                "tier": "categorical",
                "signup_date": "datetime",
            },
        },
        "orders": {
            "primary_key": "id",
            "columns": {
                "id": "integer",
                "customer_id": "integer",
                "status": "categorical",
                "prev_status": "categorical",
                "order_date": "datetime",
                "total": "float",
            },
        },
        "order_items": {
            "primary_key": "id",
            "columns": {
                "id": "integer",
                "order_id": "integer",
                "product_id": "integer",
                "quantity": "integer",
                "unit_price": "float",
                "line_total": "float",
            },
        },
    },
    "foreign_keys": [
        {
            "from_table": "orders",
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "id",
        },
        {
            "from_table": "order_items",
            "from_column": "order_id",
            "to_table": "orders",
            "to_column": "id",
        },
    ],
    "task": {
        "type": "state_classification",
        "label_column": "label",
        "positive_label": 1,
        "positive_label_name": "legal",
        "negative_label": 0,
        "negative_label_name": "illegal",
    },
    "notes": [
        "This export intentionally includes relational structure and labels only.",
        "It does not include executable rule predicates or oracle features.",
        "Use violation_type for diagnostics, not as a model input.",
    ],
}


def write_state(state_dir: Path, state: dict[str, pd.DataFrame]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in state.items():
        frame.to_csv(state_dir / f"{table_name}.csv", index=False)


def export_split(
    split: str,
    n_pairs: int,
    seed_offset: int,
    n_customers: int,
    out_dir: Path,
) -> list[dict]:
    rng = np.random.default_rng(seed_offset)
    rows: list[dict] = []
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    state_idx = 0
    for i in range(n_pairs):
        seed = seed_offset + i
        legal = generate_legal_state(n_customers=n_customers, seed=seed)
        vt = VIOLATION_TYPES_SORTED[int(rng.integers(0, len(VIOLATION_TYPES_SORTED)))]
        illegal = corrupt_to_illegal(legal, vt, seed=seed + 1_000_000)

        for state, label, tag in ((legal, 1, "legal"), (illegal, 0, vt)):
            state_id = f"{split}_{state_idx:06d}"
            rel_path = f"{split}/{state_id}"
            write_state(out_dir / rel_path, state)
            rows.append({
                "state_id": state_id,
                "split": split,
                "label": label,
                "violation_type": tag,
                "path": rel_path,
            })
            state_idx += 1

    return rows


def main(
    n_train: int,
    n_test: int,
    n_customers: int,
    out_dir: Path,
    overwrite: bool,
) -> int:
    if out_dir.exists():
        if not overwrite:
            print(f"ERROR: output directory exists: {out_dir}")
            print("Use --overwrite to replace it.")
            return 1
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting relational scaffold -> {out_dir}")
    print(f"  train_pairs={n_train}, test_pairs={n_test}, n_customers={n_customers}")

    manifest_rows = []
    manifest_rows += export_split("train", n_train, 30_000_000, n_customers, out_dir)
    manifest_rows += export_split("test", n_test, 40_000_000, n_customers, out_dir)

    with (out_dir / "schema.json").open("w", encoding="utf-8") as f:
        json.dump(SCHEMA, f, indent=2)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    print(f"  states exported: {len(manifest)}")
    print(f"  schema: {out_dir / 'schema.json'}")
    print(f"  manifest: {out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100, help="training legal/illegal pairs")
    parser.add_argument("--n-test", type=int, default=50, help="test legal/illegal pairs")
    parser.add_argument("--n-customers", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "relational_export")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(
        n_train=args.n_train,
        n_test=args.n_test,
        n_customers=args.n_customers,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    ))
