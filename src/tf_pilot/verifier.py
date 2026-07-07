"""
Marginal-match verifier for the Operational Turing Test pilot.

For each (S, S') pair, computes all k-way marginal TV distances (k ∈ {1, 2})
over the columns listed in TABLE_COLS.  Reports the maximum TV distance and a
permutation-test p-value using the max-TV statistic (which naturally accounts
for the multiple-marginal structure without double-correcting).

Gate condition: max_tv <= TAU (default 0.02).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

TAU: float = 0.02
N_BINS: int = 20
N_PERMUTATIONS: int = 200

# Columns included in marginal computation per table.
# IDs and date columns are excluded: IDs are surrogates (no distributional
# meaning), and date skew cannot be balanced by permutation-based corruption.
TABLE_COLS: dict[str, list[str]] = {
    "customers":   ["country", "tier"],
    "orders":      ["status", "prev_status", "total"],
    "order_items": ["product_id", "quantity", "unit_price", "line_total"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _bin_edges(s1: pd.Series, s2: pd.Series, n_bins: int) -> np.ndarray:
    """Fixed histogram edges from combined support of two series."""
    combined = pd.concat([s1, s2]).dropna().astype(float)
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5])
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[-1] += 1e-9   # include the max value in the last bin
    return edges


def _tv_1way(col_s: pd.Series, col_sp: pd.Series, edges: np.ndarray | None) -> float:
    """TV distance for one column (numeric or categorical)."""
    if edges is not None:
        p, _ = np.histogram(col_s.dropna().astype(float),  bins=edges)
        q, _ = np.histogram(col_sp.dropna().astype(float), bins=edges)
        p = p / p.sum() if p.sum() > 0 else np.zeros_like(p, dtype=float)
        q = q / q.sum() if q.sum() > 0 else np.zeros_like(q, dtype=float)
        return float(0.5 * np.abs(p - q).sum())
    # Categorical
    all_vals = set(col_s.dropna()) | set(col_sp.dropna())
    vc_s  = col_s.value_counts(normalize=True)
    vc_sp = col_sp.value_counts(normalize=True)
    p = np.array([vc_s.get(v, 0.0)  for v in all_vals])
    q = np.array([vc_sp.get(v, 0.0) for v in all_vals])
    return float(0.5 * np.abs(p - q).sum())


def _tv_2way(
    df_s: pd.DataFrame, df_sp: pd.DataFrame,
    c1: str, c2: str,
    edges: dict[str, np.ndarray | None],
) -> float:
    """TV distance for the joint distribution of two columns."""
    def discretize(col: pd.Series, e: np.ndarray | None) -> pd.Series:
        if e is not None:
            return pd.Series(
                np.digitize(col.fillna(col.median()).astype(float), e),
                index=col.index,
            )
        return col.fillna("__NA__").astype(str)

    s_c1, s_c2   = discretize(df_s[c1],  edges[c1]),  discretize(df_s[c2],  edges[c2])
    sp_c1, sp_c2 = discretize(df_sp[c1], edges[c1]),  discretize(df_sp[c2], edges[c2])

    keys_s  = [*zip(s_c1.tolist(),  s_c2.tolist())]
    keys_sp = [*zip(sp_c1.tolist(), sp_c2.tolist())]
    vc_s    = pd.Series(keys_s).value_counts(normalize=True)
    vc_sp   = pd.Series(keys_sp).value_counts(normalize=True)
    all_keys = set(vc_s.index) | set(vc_sp.index)
    p = np.array([vc_s.get(k, 0.0)  for k in all_keys])
    q = np.array([vc_sp.get(k, 0.0) for k in all_keys])
    return float(0.5 * np.abs(p - q).sum())


def _precompute_edges(
    S: dict[str, pd.DataFrame],
    S_prime: dict[str, pd.DataFrame],
    n_bins: int,
) -> dict[str, dict[str, np.ndarray | None]]:
    edges: dict[str, dict[str, np.ndarray | None]] = {}
    for table, cols in TABLE_COLS.items():
        edges[table] = {}
        df_s, df_sp = S[table], S_prime[table]
        for col in cols:
            if col not in df_s.columns:
                continue
            edges[table][col] = (
                _bin_edges(df_s[col], df_sp[col], n_bins)
                if _is_numeric(df_s[col]) else None
            )
    return edges


# ── public API ────────────────────────────────────────────────────────────────

def verify_marginal_match(
    S: dict[str, pd.DataFrame],
    S_prime: dict[str, pd.DataFrame],
    order_k: int = 2,
    tau: float = TAU,
    n_bins: int = N_BINS,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Compute k-way marginal TV distances (k ∈ {1, …, order_k}) between S and S'.

    Permutation test uses the max-TV statistic over all 1-way marginals across
    all tables; this accounts for the multi-marginal structure without the
    double-correction that per-marginal Bonferroni would introduce.

    Returns
    -------
    dict with keys:
        marginals          list[dict]  per-marginal records
        max_tv             float       largest TV distance observed
        worst_marginal     str         name of marginal with largest TV
        p_value            float       permutation p-value (max-TV statistic)
        n_marginals        int         total marginals tested
        passes_tau         bool        max_tv <= tau
    """
    rng   = np.random.default_rng(seed)
    edges = _precompute_edges(S, S_prime, n_bins)
    records: list[dict] = []

    for table, cols in TABLE_COLS.items():
        df_s, df_sp = S[table], S_prime[table]
        present = [c for c in cols if c in df_s.columns]

        # 1-way
        for col in present:
            tv = _tv_1way(df_s[col], df_sp[col], edges[table][col])
            records.append({"name": f"{table}.{col}", "table": table, "k": 1, "tv_distance": tv})

        # 2-way
        if order_k >= 2:
            for c1, c2 in combinations(present, 2):
                tv = _tv_2way(df_s, df_sp, c1, c2, edges[table])
                records.append({"name": f"{table}.{c1}×{c2}", "table": table, "k": 2, "tv_distance": tv})

    max_tv       = max((r["tv_distance"] for r in records), default=0.0)
    worst        = max(records, key=lambda r: r["tv_distance"])["name"] if records else ""

    # Permutation test: pool rows per table, random split, recompute max 1-way TV
    perm_maxes: list[float] = []
    for _ in range(n_permutations):
        pm = 0.0
        for table, cols in TABLE_COLS.items():
            df_s, df_sp = S[table], S_prime[table]
            present     = [c for c in cols if c in df_s.columns]
            n_s         = len(df_s)
            pooled      = pd.concat([df_s[present], df_sp[present]], ignore_index=True)
            perm_idx    = rng.permutation(len(pooled))
            grp_a       = pooled.iloc[perm_idx[:n_s]]
            grp_b       = pooled.iloc[perm_idx[n_s:]]
            for col in present:
                pm = max(pm, _tv_1way(grp_a[col], grp_b[col], edges[table][col]))
        perm_maxes.append(pm)

    perm_arr = np.array(perm_maxes)
    p_value  = float((perm_arr >= max_tv).mean())

    return {
        "marginals":      records,
        "max_tv":         max_tv,
        "worst_marginal": worst,
        "p_value":        p_value,
        "n_marginals":    len(records),
        "passes_tau":     max_tv <= tau,
    }
